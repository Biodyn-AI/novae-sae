#!/usr/bin/env python
"""GAP 9: PMI co-activation vs causal edge comparison.

For each pair of features connected by a causal circuit edge, compute
the pointwise mutual information (PMI) of their co-activation across
cells. Then compare: do causally connected features also co-activate
more than random pairs?

PMI(A,B) = log2(P(A,B) / (P(A) * P(B)))
where P(A) = fraction of cells where feature A fires (TopK active),
P(A,B) = fraction where both fire simultaneously.

Output: atlas/novae-human-0/causal/pmi_vs_causal.parquet
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "24_pmi_vs_causal.log"

SAE_SPECS = {
    **{f"conv_{i}": (128, 4096, 32) for i in range(9)},
    "conv_9": (64, 2048, 16),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def encode_binary(act_path, sae, k):
    """Encode activations and return binary (active/inactive) per feature."""
    act = np.load(act_path).astype(np.float32, copy=False)
    n = act.shape[0]
    # Binary: feature is active if it's in the TopK for that cell
    active = np.zeros((n, sae.n_features), dtype=np.bool_)
    chunk = 32768
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            _, idx = sae.encode(torch.tensor(act[s:e]))
            idx_np = idx.numpy()
            for b in range(e - s):
                active[s + b, idx_np[b]] = True
    del act
    return active


def main():
    log("=" * 60)
    log("GAP 9: PMI co-activation vs causal edge comparison")

    edges = pd.read_parquet(OUT_DIR / "causal_circuit_edges.parquet")
    log(f"loaded {len(edges)} causal edges")

    # Get unique (layer, feature) pairs from edges
    layers_needed = set()
    for _, e in edges.iterrows():
        layers_needed.add(int(e["source_layer"]))
        layers_needed.add(int(e["target_layer"]))
    log(f"layers needed: {sorted(layers_needed)}")

    # Use per-cell activations from brain slide (same as circuit tracing)
    slide_dir = ACT_DIR / "per_slide" / "brain__Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs"

    # Load binary activation matrices per layer
    layer_active = {}
    for layer_idx in sorted(layers_needed):
        name = f"conv_{layer_idx}"
        act_path = slide_dir / f"{name}_percell.npy"
        if not act_path.exists():
            log(f"  SKIP layer {layer_idx}: no per-cell activations")
            continue
        d_in, n_feat, k = SAE_SPECS[name]
        sae = TopKSAE(d_in=d_in, n_features=n_feat, k=k)
        sae.load_state_dict(torch.load(SAE_DIR / f"{name}.pt", map_location="cpu"))
        sae.eval()
        active = encode_binary(act_path, sae, k)
        layer_active[layer_idx] = active
        log(f"  layer {layer_idx}: {active.shape}, active rate = {active.mean():.4f}")
        del sae
        gc.collect()

    # Compute PMI for each causal edge
    results = []
    n_cells = None
    for _, e in edges.iterrows():
        src_l = int(e["source_layer"])
        tgt_l = int(e["target_layer"])
        src_f = int(e["source_feature"])
        tgt_f = int(e["target_feature"])

        if src_l not in layer_active or tgt_l not in layer_active:
            continue

        src_active = layer_active[src_l]
        tgt_active = layer_active[tgt_l]
        if n_cells is None:
            n_cells = src_active.shape[0]

        p_a = src_active[:, src_f].mean()
        p_b = tgt_active[:, tgt_f].mean()
        p_ab = (src_active[:, src_f] & tgt_active[:, tgt_f]).mean()

        if p_a > 0 and p_b > 0 and p_ab > 0:
            pmi = float(np.log2(p_ab / (p_a * p_b)))
        elif p_ab == 0:
            pmi = -10.0  # floor for zero co-occurrence
        else:
            pmi = 0.0

        results.append({
            "source_layer": src_l,
            "source_feature": src_f,
            "target_layer": tgt_l,
            "target_feature": tgt_f,
            "cohen_d": float(e["cohen_d"]),
            "p_source": float(p_a),
            "p_target": float(p_b),
            "p_joint": float(p_ab),
            "pmi": pmi,
        })

    df = pd.DataFrame(results)
    log(f"computed PMI for {len(df)} edges")

    # Random baseline: PMI for random (non-causal) feature pairs
    rng = np.random.default_rng(42)
    random_pmis = []
    for _ in range(min(1000, len(df) * 3)):
        src_l = rng.choice(list(layer_active.keys()))
        tgt_l = rng.choice([l for l in layer_active.keys() if l > src_l] or [src_l])
        src_f = rng.integers(0, layer_active[src_l].shape[1])
        tgt_f = rng.integers(0, layer_active[tgt_l].shape[1])

        p_a = layer_active[src_l][:, src_f].mean()
        p_b = layer_active[tgt_l][:, tgt_f].mean()
        p_ab = (layer_active[src_l][:, src_f] & layer_active[tgt_l][:, tgt_f]).mean()

        if p_a > 0 and p_b > 0 and p_ab > 0:
            random_pmis.append(float(np.log2(p_ab / (p_a * p_b))))
        elif p_ab == 0:
            random_pmis.append(-10.0)

    # Compare
    causal_pmi = df["pmi"].values
    random_pmi = np.array(random_pmis)

    from scipy import stats
    u_stat, u_p = stats.mannwhitneyu(causal_pmi, random_pmi, alternative="greater")

    # Overlap: fraction of causal edges with PMI > median random PMI
    median_random = float(np.median(random_pmi))
    overlap_frac = float((causal_pmi > median_random).mean())

    df.to_parquet(OUT_DIR / "pmi_vs_causal.parquet", index=False)
    log(f"wrote pmi_vs_causal.parquet")

    summary = {
        "n_causal_edges_with_pmi": len(df),
        "n_random_pairs": len(random_pmis),
        "causal_pmi_mean": round(float(causal_pmi.mean()), 4),
        "causal_pmi_median": round(float(np.median(causal_pmi)), 4),
        "random_pmi_mean": round(float(random_pmi.mean()), 4),
        "random_pmi_median": round(float(np.median(random_pmi)), 4),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(u_p),
        "overlap_fraction": round(overlap_frac, 4),
        "correlation_pmi_vs_d": round(float(np.corrcoef(causal_pmi, df["cohen_d"].values)[0, 1]), 4) if len(df) > 2 else None,
    }
    json.dump(summary, open(OUT_DIR / "pmi_vs_causal.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))

    log(f"\nGAP 9 verdict:")
    log(f"  causal PMI: mean={summary['causal_pmi_mean']:.3f}, median={summary['causal_pmi_median']:.3f}")
    log(f"  random PMI: mean={summary['random_pmi_mean']:.3f}, median={summary['random_pmi_median']:.3f}")
    log(f"  Mann-Whitney p={summary['mannwhitney_p']:.2e}")
    log(f"  {overlap_frac:.1%} of causal edges have PMI above random median")
    log("DONE")


if __name__ == "__main__":
    main()
