#!/usr/bin/env python
"""Reviewer control: feature-permutation graph ablation.

Reviewer concern: the self-loop graph ablation (script 10) is a large
distributional shift that replaces the aggregation operation entirely.
The 67% context-dependency number is expected a priori under any such
shift. A stronger control: keep the Delaunay graph intact but permute
node features across cells, so each cell receives messages from its
spatial neighbors but the content of those messages is scrambled.

Protocol:
  1. Load h5ad. Run novae.spatial_neighbors -> full graph.
  2. Pass 1 (baseline): compute_representations, SAE-encode, record
     per-feature mean |activation|.
  3. Pass 2 (feature permutation): permute adata.X ROWS (cells) while
     KEEPING adata.obsp spatial graphs intact. Each cell's content is
     now a random cell's content, but its graph position is unchanged.
     compute_representations again, SAE-encode, record means.
  4. Per-feature: feature_perm_dependency = (mean|full| - mean|perm|) / mean|full|.

Aggregate over slides. Compare to the self-loop ablation result.
Output: atlas/novae-human-0/causal/reviewer_controls/H_feature_perm_ablation.parquet
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
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal" / "reviewer_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "29_feature_perm_ablation.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# Small, fast slides first. Add more if compute allows.
SLIDE_SUBSET_NAMES = [
    "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",   # brain (24k)
    "Xenium_V1_hPancreas_nondiseased_section_outs",         # pancreas
    "Xenium_V1_hKidney_nondiseased_section_outs",           # kidney
]
RNG_SEED = 2026


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def encode_features(sae: TopKSAE, X: np.ndarray) -> np.ndarray:
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


def main() -> None:
    log("=" * 72)
    log("Feature-permutation graph ablation (fixed graph, shuffled features)")

    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    slides_to_process = [s for s in manifest["slides"] if s["name"] in SLIDE_SUBSET_NAMES]
    log(f"{len(slides_to_process)} slides")

    n_features = 2048
    full_sum = np.zeros(n_features, dtype=np.float64)
    perm_sum = np.zeros(n_features, dtype=np.float64)
    n_cells_processed = 0
    rng = np.random.default_rng(RNG_SEED)

    for slide in slides_to_process:
        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"\n[{slide['tissue']}] {slide['name']}")
        try:
            a = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"  ERROR load: {e}")
            continue
        novae.spatial_neighbors(a)

        log(f"  {a.n_obs:,} cells; full pass")
        t0 = time.time()
        try:
            model.compute_representations(a, zero_shot=True)
        except Exception as e:
            log(f"  ERROR full: {e}")
            continue
        full_latent = a.obsm["novae_latent"].copy()
        log(f"    full {time.time()-t0:.0f}s")

        # Feature permutation: shuffle cells' gene-expression rows.
        log("  permuting node features (X rows)…")
        perm_idx = rng.permutation(a.n_obs)
        X = a.X
        if hasattr(X, "toarray"):
            # sparse: permute row index array
            import scipy.sparse as sp
            a.X = X[perm_idx].copy() if sp.issparse(X) else X[perm_idx]
        else:
            a.X = np.asarray(X)[perm_idx]
        if "counts" in a.layers:
            a.layers["counts"] = a.layers["counts"][perm_idx]
        # Remove cached latent to force recompute
        if "novae_latent" in a.obsm:
            del a.obsm["novae_latent"]

        log("  permuted pass")
        t0 = time.time()
        try:
            model.compute_representations(a, zero_shot=True)
        except Exception as e:
            log(f"  ERROR perm: {e}")
            continue
        perm_latent = a.obsm["novae_latent"].copy()
        log(f"    perm {time.time()-t0:.0f}s")

        full_feats = encode_features(sae, full_latent)
        perm_feats = encode_features(sae, perm_latent)
        full_sum += np.abs(full_feats).sum(axis=0)
        perm_sum += np.abs(perm_feats).sum(axis=0)
        n_cells_processed += full_feats.shape[0]

        del a, full_latent, perm_latent, full_feats, perm_feats
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    log(f"\nprocessed {n_cells_processed:,} cells across {len(slides_to_process)} slides")
    full_mean = full_sum / max(n_cells_processed, 1)
    perm_mean = perm_sum / max(n_cells_processed, 1)

    rows = []
    for fid in range(n_features):
        f_full = float(full_mean[fid])
        f_perm = float(perm_mean[fid])
        dep = (f_full - f_perm) / f_full if f_full > 1e-9 else float("nan")
        rows.append({
            "feature_idx": fid,
            "mean_abs_full": f_full,
            "mean_abs_feature_permuted": f_perm,
            "feature_perm_dependency": dep,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "H_feature_perm_ablation.parquet", index=False)

    # Join with self-loop result for comparison
    try:
        selfloop = pd.read_parquet(ROOT / "atlas" / "novae-human-0" / "causal" / "graph_ablation.parquet")
        merged = df.merge(
            selfloop[["feature_idx", "contextual_dependency"]].rename(
                columns={"contextual_dependency": "selfloop_dependency"}),
            on="feature_idx", how="left",
        )
        merged.to_parquet(OUT_DIR / "H_feature_perm_ablation.parquet", index=False)

    except Exception as e:
        log(f"  merge note: {e}")
    has_full = df["mean_abs_full"] > 1e-9
    deps = df.loc[has_full, "feature_perm_dependency"].dropna()
    summary = {
        "n_features": int(len(df)),
        "n_active": int(has_full.sum()),
        "n_dep_gt_0.5": int((deps > 0.5).sum()),
        "n_dep_gt_0.2": int((deps > 0.2).sum()),
        "median_dep": float(deps.median()) if len(deps) else None,
        "mean_dep": float(deps.mean()) if len(deps) else None,
        "slides_processed": len(slides_to_process),
        "cells_processed": int(n_cells_processed),
        "interpretation": (
            "If feature_perm_dependency ≈ self-loop dependency, the self-loop "
            "intervention was actually measuring loss-of-neighbor-content, not "
            "loss-of-graph-topology — which is the correct, biologically meaningful "
            "effect. If feature_perm dep << self-loop dep, the self-loop number "
            "was mostly driven by the architectural shift; then the 'context-"
            "dependent' claim must be narrowed."
        ),
    }
    (OUT_DIR / "H_feature_perm_ablation.summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
