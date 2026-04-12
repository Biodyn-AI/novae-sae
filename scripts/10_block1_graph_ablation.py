#!/usr/bin/env python
"""Block 1.4 — graph ablation suite (H8). The spatial CRISPRi analogue.

For each slide:
  1. Run Novae's compute_representations normally (full spatial graph).
  2. Replace adata.obsp spatial graphs with self-loop-only matrices (k=0).
  3. Run compute_representations again.
  4. SAE-encode both aggregator outputs.
  5. Per feature: contextual_dependency = (mean|full| - mean|isolated|) / max(mean|full|, eps)

Aggregate across slides → per-feature contextual_dependency_score.

H8 prediction: >20% of SAE features show >50% activation drop when
neighborhoods are zeroed out, confirming Novae uses spatial context vs
being a glorified cell-type classifier.

Output: atlas/causal/graph_ablation.parquet
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import anndata as ad
import novae
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "10_block1_graph_ablation.log"

# To keep compute reasonable, subsample slides — pick a stratified set
SLIDE_SUBSET_NAMES = [
    "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",  # brain (small)
    "Xenium_V1_hKidney_nondiseased_section_outs",          # kidney
    "Xenium_V1_hLiver_nondiseased_section_FFPE_outs",      # liver
    "Xenium_V1_hPancreas_nondiseased_section_outs",        # pancreas
    "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs",     # skin
    "Xenium_V1_hLymphNode_nondiseased_section_outs",       # lymph_node
    "Xenium_V1_hColon_Non_diseased_Base_FFPE_outs",        # colon
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def encode_features(sae: TopKSAE, X: np.ndarray) -> np.ndarray:
    """Encode aggregator output → feature activations."""
    chunk = 32768
    out = []
    sae.to(DEVICE)
    with torch.no_grad():
        for start in range(0, X.shape[0], chunk):
            xb = torch.tensor(X[start:start + chunk], dtype=torch.float32, device=DEVICE)
            z, _ = sae.encode(xb)
            out.append(z.cpu().numpy())
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return np.vstack(out)


def collapse_graph_to_self_loops(adata) -> None:
    """Replace adata's spatial obsp with self-loop-only matrices.

    Novae stores the spatial graph in adata.obsp under several keys; we set
    them all to identity so each cell has only itself as a neighbor.

    IMPORTANT: do NOT touch adata.obs — `novae_sid` is set by
    `novae.spatial_neighbors` and is REQUIRED by `compute_representations`.
    Clearing it triggers an AssertionError. Only clear the cached latent in
    obsm so the next compute_representations call writes a fresh one.
    """
    n = adata.n_obs
    eye = sp.eye(n, format="csr", dtype=np.float32)
    for key in list(adata.obsp.keys()):
        if "spatial" in key.lower() or "connectivit" in key.lower() or "distance" in key.lower():
            adata.obsp[key] = eye.copy()
    if "novae_latent" in adata.obsm:
        del adata.obsm["novae_latent"]


def main() -> None:
    log("=" * 72)
    log("Block 1.4: graph ablation suite (H8)")

    log("loading model + SAE")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    slides_to_process = [s for s in manifest["slides"] if s["name"] in SLIDE_SUBSET_NAMES]
    log(f"  processing {len(slides_to_process)} slides")

    # Per-feature accumulator
    n_features = 2048
    full_sum = np.zeros(n_features, dtype=np.float64)
    iso_sum = np.zeros(n_features, dtype=np.float64)
    n_cells_processed = 0

    for slide in slides_to_process:
        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"\n[{slide['slide_idx']+1}/{len(manifest['slides'])}] {slide['tissue']}/{slide['name']}")
        try:
            a = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"  ERROR load: {e}")
            continue

        # === Pass 1: full graph ===
        try:
            novae.spatial_neighbors(a)
        except Exception as e:
            log(f"  ERROR spatial_neighbors: {e}")
            del a
            continue

        log(f"  {a.n_obs:,} cells; running compute_representations (full graph)")
        t0 = time.time()
        try:
            model.compute_representations(a, zero_shot=True)
        except Exception as e:
            log(f"  ERROR full compute: {type(e).__name__}: {e}")
            del a
            continue
        full_latent = a.obsm["novae_latent"].copy()
        log(f"    full pass {time.time()-t0:.0f}s")

        # === Pass 2: collapsed graph (self-loops only) ===
        log(f"  collapsing graph to self-loops")
        collapse_graph_to_self_loops(a)
        log(f"  running compute_representations (k=0 isolated)")
        t0 = time.time()
        try:
            model.compute_representations(a, zero_shot=True)
            iso_latent = a.obsm["novae_latent"].copy()
            log(f"    isolated pass {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  ERROR isolated compute: {type(e).__name__}: {e}")
            del a
            continue

        # === Encode both via SAE ===
        log(f"  encoding via SAE")
        full_feats = encode_features(sae, full_latent)
        iso_feats = encode_features(sae, iso_latent)

        # Accumulate per-feature mean abs activation
        full_sum += np.abs(full_feats).sum(axis=0)
        iso_sum += np.abs(iso_feats).sum(axis=0)
        n_cells_processed += full_feats.shape[0]

        del a, full_latent, iso_latent, full_feats, iso_feats
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    log(f"\nprocessed {n_cells_processed:,} cells across {len(slides_to_process)} slides")

    full_mean = full_sum / max(n_cells_processed, 1)
    iso_mean = iso_sum / max(n_cells_processed, 1)

    rows = []
    for fid in range(n_features):
        f_full = float(full_mean[fid])
        f_iso = float(iso_mean[fid])
        if f_full > 1e-9:
            dep = (f_full - f_iso) / f_full
        else:
            dep = float("nan")
        rows.append({
            "feature_idx": fid,
            "mean_abs_full": f_full,
            "mean_abs_isolated": f_iso,
            "contextual_dependency": dep,
        })
    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "graph_ablation.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {out_path} ({len(df)} rows)")

    has_signal = df["mean_abs_full"] > 1e-9
    deps = df.loc[has_signal, "contextual_dependency"].dropna()
    summary = {
        "n_features": int(len(df)),
        "n_with_full_signal": int(has_signal.sum()),
        "n_dep_gt_0.5": int((deps > 0.5).sum()),
        "n_dep_gt_0.2": int((deps > 0.2).sum()),
        "median_dep": float(deps.median()) if len(deps) else None,
        "mean_dep": float(deps.mean()) if len(deps) else None,
        "slides_processed": len(slides_to_process),
        "cells_processed": int(n_cells_processed),
    }
    json.dump(summary, open(OUT_DIR / "graph_ablation.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))

    log("\nH8 evaluation: >20% of features should show >50% activation drop when k=0")
    if summary["n_with_full_signal"] > 0:
        frac = summary["n_dep_gt_0.5"] / summary["n_with_full_signal"]
        log(f"  result: {frac*100:.1f}% of active features have dep > 0.5")
        if frac > 0.20:
            log(f"  → H8 CONFIRMED: Novae uses spatial context")
        else:
            log(f"  → H8 REFUTED")
    log("DONE")


if __name__ == "__main__":
    main()
