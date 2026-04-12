#!/usr/bin/env python
"""GAP 15: Higher-order combinatorial ablation (3-way).

Ablate feature triplets in all 7 combinations (A, B, C, AB, AC, BC, ABC)
and measure reconstruction cosine drop. Compute:
- Pairwise redundancy ratio R_AB = max(d_A, d_B) / d_AB
- Three-way redundancy ratio R_ABC = median(d_A, d_B, d_C) / d_ABC
- Interaction term I_ABC = d_ABC - (d_A + d_B + d_C - d_AB - d_AC - d_BC)

If R < 1: features have synergy (joint effect > sum of parts)
If R > 1: features are redundant (joint effect < sum of parts)

Select 10 triplets of co-active features from the circuit graph
(features that share causal edges).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from itertools import combinations

import anndata as ad
import novae
import numpy as np
import pandas as pd
import torch

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "22_combinatorial_ablation.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SLIDE = ("brain", "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs")
N_TRIPLETS = 10
N_CELLS_SAMPLE = 500


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_ablation(model, adata, sae, features_to_zero, layer_idx):
    """Run forward pass with specified features zeroed at layer_idx.
    Returns aggregator output (novae_latent)."""
    gnn = model.encoder.gnn

    def ablation_hook(module, inp, out):
        with torch.no_grad():
            out_cpu = out.detach().cpu().float()
            z, _ = sae.encode(out_cpu)
            for fid in features_to_zero:
                z[:, fid] = 0.0
            out_mod = sae.decode(z).to(out.device).to(out.dtype)
        return out_mod

    handle = gnn.convs[layer_idx].register_forward_hook(ablation_hook)
    model.compute_representations(adata, zero_shot=True)
    handle.remove()
    return np.asarray(adata.obsm["novae_latent"], dtype=np.float32).copy()


def main():
    log("=" * 60)
    log("GAP 15: Higher-order combinatorial ablation")

    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    # Load aggregator SAE for measuring effects
    agg_sae = TopKSAE(d_in=64, n_features=2048, k=16)
    agg_sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    agg_sae.eval()

    # Load source layer SAE for ablation
    src_layer = 0  # ablate at conv_0
    src_sae = TopKSAE(d_in=128, n_features=4096, k=32)
    src_sae.load_state_dict(torch.load(SAE_DIR / "conv_0.pt", map_location="cpu"))
    src_sae.eval()

    # Load slide
    tissue, slide_name = SLIDE
    adata = ad.read_h5ad(DATA_DIR / tissue / f"{slide_name}.h5ad")
    novae.spatial_neighbors(adata)
    log(f"slide: {adata.n_obs} cells")

    # Select triplets: top features from circuit edges at conv_0
    edges = pd.read_parquet(OUT_DIR / "causal_circuit_edges.parquet")
    src_feats = edges[edges["source_layer"] == 0]["source_feature"].value_counts()
    top_features = src_feats.head(30).index.tolist()

    # Form triplets from combinations of top features
    all_triplets = list(combinations(top_features[:15], 3))[:N_TRIPLETS]
    log(f"testing {len(all_triplets)} triplets from conv_0")

    # Baseline: no ablation
    log("running baseline")
    model.compute_representations(adata, zero_shot=True)
    baseline = np.asarray(adata.obsm["novae_latent"], dtype=np.float32).copy()

    # Subsample cells for efficiency
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(baseline.shape[0], size=min(N_CELLS_SAMPLE, baseline.shape[0]), replace=False)
    baseline_sample = baseline[sample_idx]

    def effect_size(ablated_latent):
        """Cosine distance from baseline (mean across sampled cells)."""
        abl_sample = ablated_latent[sample_idx]
        cos = np.sum(baseline_sample * abl_sample, axis=1) / (
            np.linalg.norm(baseline_sample, axis=1) * np.linalg.norm(abl_sample, axis=1) + 1e-9
        )
        return float(1.0 - cos.mean())  # cosine distance

    results = []
    for ti, (a, b, c) in enumerate(all_triplets):
        log(f"\ntriplet {ti+1}/{len(all_triplets)}: F{a}, F{b}, F{c}")
        t0 = time.time()

        # 7 ablation conditions
        conditions = {
            "A": [a],
            "B": [b],
            "C": [c],
            "AB": [a, b],
            "AC": [a, c],
            "BC": [b, c],
            "ABC": [a, b, c],
        }

        effects = {}
        for cond_name, feats in conditions.items():
            lat = run_ablation(model, adata, src_sae, feats, src_layer)
            effects[cond_name] = effect_size(lat)
            del lat
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

        # Compute redundancy metrics
        d_A, d_B, d_C = effects["A"], effects["B"], effects["C"]
        d_AB, d_AC, d_BC = effects["AB"], effects["AC"], effects["BC"]
        d_ABC = effects["ABC"]

        # Pairwise redundancy: R_AB = max(d_A, d_B) / d_AB
        R_AB = max(d_A, d_B) / max(d_AB, 1e-9)
        R_AC = max(d_A, d_C) / max(d_AC, 1e-9)
        R_BC = max(d_B, d_C) / max(d_BC, 1e-9)

        # Three-way redundancy
        R_ABC = np.median([d_A, d_B, d_C]) / max(d_ABC, 1e-9)

        # Interaction term (inclusion-exclusion)
        I_ABC = d_ABC - (d_A + d_B + d_C - d_AB - d_AC - d_BC)

        results.append({
            "triplet_idx": ti,
            "feat_A": a, "feat_B": b, "feat_C": c,
            "d_A": d_A, "d_B": d_B, "d_C": d_C,
            "d_AB": d_AB, "d_AC": d_AC, "d_BC": d_BC,
            "d_ABC": d_ABC,
            "R_AB": R_AB, "R_AC": R_AC, "R_BC": R_BC,
            "R_ABC": R_ABC,
            "I_ABC": I_ABC,
        })

        log(f"  d_A={d_A:.4f} d_B={d_B:.4f} d_C={d_C:.4f}")
        log(f"  d_AB={d_AB:.4f} d_AC={d_AC:.4f} d_BC={d_BC:.4f}")
        log(f"  d_ABC={d_ABC:.4f}")
        log(f"  R_ABC={R_ABC:.3f} I_ABC={I_ABC:.6f} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "combinatorial_ablation.parquet", index=False)
    log(f"\nwrote combinatorial_ablation.parquet ({len(df)} triplets)")

    # Summary
    summary = {
        "n_triplets": len(df),
        "source_layer": src_layer,
        "mean_R_ABC": round(float(df["R_ABC"].mean()), 4),
        "median_R_ABC": round(float(df["R_ABC"].median()), 4),
        "mean_R_AB": round(float(df["R_AB"].mean()), 4),
        "mean_I_ABC": round(float(df["I_ABC"].mean()), 6),
        "n_redundant_ABC": int((df["R_ABC"] > 1).sum()),
        "n_synergistic_ABC": int((df["R_ABC"] < 1).sum()),
        "interpretation": (
            "redundant" if df["R_ABC"].median() > 1 else "synergistic"
        ),
    }
    json.dump(summary, open(OUT_DIR / "combinatorial_ablation.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))

    log("\nGAP 15 verdict:")
    if summary["median_R_ABC"] > 1:
        log(f"  Features are REDUNDANT (R_ABC = {summary['median_R_ABC']:.3f} > 1)")
        log("  Joint ablation effect < sum of individual effects")
    else:
        log(f"  Features show SYNERGY (R_ABC = {summary['median_R_ABC']:.3f} < 1)")
        log("  Joint ablation effect > sum of individual effects")
    log("DONE")


if __name__ == "__main__":
    main()
