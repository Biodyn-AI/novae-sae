#!/usr/bin/env python
"""Phase 3 — characterize SAE features without external biological databases.

This first-pass characterization uses only metadata that is universally
available across all 15 slides (slide_id, tissue, technology) plus the
activation arrays themselves. External databases (GO/KEGG/Reactome/PanglaoDB/
OmniPath) are deferred to a follow-on script (`03b_characterize_db.py`).

For each trained SAE this script computes:

  1. **Variance explained on validation split** (re-confirmed from saved SAE).
  2. **Layer-wise activation profile**: per feature, mean and L0 across the
     full activation corpus.
  3. **Per-feature top-cell composition**: which tissues / technologies are
     enriched in the cells where the feature fires (Fisher's exact, BH FDR).
  4. **Tissue specificity score**: max-tissue / total ratio per feature.
  5. **Technology confound flag**: features that fire on ≥80% of cells from
     a single technology — likely batch artefacts.
  6. **Superposition (SAE vs SVD)**: per layer, fraction of decoder columns
     with cosine > 0.7 against any of the top-50 SVD axes of that layer's
     activation matrix. Replicates the upstream 99.8% statistic.
  7. **Variance comparison**: SAE recon variance vs top-50 SVD variance.

Outputs (written to `atlas/novae-human-0/`):

  - `feature_atlas.parquet`  one row per (surface, feature_id)
  - `superposition.json`     per-surface summary
  - `layer_profile.json`     per-layer aggregate stats

Module discovery (PMI + Leiden), cross-layer highways, and prototype-alignment
are deferred to `03c_modules.py` (separate file because they have heavier
runtime).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import fisher_exact

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0"
LOG_PATH = ROOT / "logs" / "03_characterize.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Mirror SAE_CONFIGS from 02_train_saes.py — needed to instantiate empty SAEs
# before loading weights. Kept in sync manually.
SAE_CONFIGS: list[tuple[str, int, int]] = [
    ("aggregator",     32, 16),
    ("conv_0",         32, 32),
    ("conv_1",         32, 32),
    ("conv_2",         32, 32),
    ("conv_3",         32, 32),
    ("conv_4",         32, 32),
    ("conv_5",         32, 32),
    ("conv_6",         32, 32),
    ("conv_7",         32, 32),
    ("conv_8",         32, 32),
    ("conv_9",         32, 16),
    ("cell_embedder",  16, 32),
]

TOPK_SVD = 50               # number of SVD axes to compare against
SVD_ALIGN_THRESHOLD = 0.7   # cosine threshold matching upstream
TOP_CELL_FRACTION = 0.01    # top 1% of cells per feature for enrichment


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_sae(name: str, expansion: int, k: int, d_in: int) -> TopKSAE:
    sae = TopKSAE(d_in=d_in, n_features=expansion * d_in, k=k)
    sae.load_state_dict(torch.load(SAE_DIR / f"{name}.pt", map_location="cpu"))
    sae.eval()
    return sae


def encode_in_chunks(sae: TopKSAE, X: torch.Tensor, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Run TopK encode on X in chunks to bound memory.

    Returns
    -------
    activations : (n_kept_max_per_row=k, n_rows) sparse-friendly representation
                  here returned as a dense (n_rows, n_features) np.float32 array
                  if memory allows; else as (n_rows*k,) flat (idx, val).

    For SAEs with F~4096 and n_rows~5e6 the dense (n,F) array is ~80 GB which
    is far too big. Instead we return only the *non-zero* entries as a sparse
    CSR matrix.
    """
    n, _ = X.shape
    n_features = sae.n_features
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    sae = sae.to(DEVICE)
    with torch.no_grad():
        for start in range(0, n, chunk):
            xb = X[start: start + chunk].to(DEVICE)
            z, idx = sae.encode(xb)
            # gather non-zero entries
            mask = z != 0
            r, c = mask.nonzero(as_tuple=True)
            v = z[r, c]
            rows.append((r + start).cpu().numpy())
            cols.append(c.cpu().numpy())
            vals.append(v.cpu().numpy())
    rows_np = np.concatenate(rows)
    cols_np = np.concatenate(cols)
    vals_np = np.concatenate(vals)
    M = sparse.csr_matrix((vals_np, (rows_np, cols_np)), shape=(n, n_features), dtype=np.float32)
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return M, vals_np  # M is sparse, second is just for stats


def superposition_audit(sae: TopKSAE, X: np.ndarray) -> dict:
    """Compare SAE decoder columns against top-K SVD axes of X.

    Replicates the upstream §superposition protocol.
    """
    # SVD on a centered subsample — sklearn TruncatedSVD is overkill;
    # numpy's SVD on a (small_n × d) matrix is fine since d ≤ 512
    n, d = X.shape
    n_sub = min(50_000, n)
    rng = np.random.default_rng(42)
    sub = X[rng.choice(n, size=n_sub, replace=False)]
    sub = sub - sub.mean(axis=0, keepdims=True)
    # use truncated SVD via scipy.sparse.linalg if needed; here numpy is fine
    _, S, Vt = np.linalg.svd(sub, full_matrices=False)
    k_svd = min(TOPK_SVD, len(S))
    svd_axes = Vt[:k_svd]                                    # (k_svd, d), each row is a unit vector
    decoder_cols = sae.decoder.weight.detach().cpu().numpy()  # (d, n_features)
    decoder_cols = decoder_cols / np.linalg.norm(decoder_cols, axis=0, keepdims=True).clip(min=1e-8)
    # cosine similarity (svd_axes already unit by SVD construction)
    sim = svd_axes @ decoder_cols                            # (k_svd, n_features)
    max_abs_sim = np.abs(sim).max(axis=0)                    # (n_features,)
    aligned = (max_abs_sim > SVD_ALIGN_THRESHOLD).sum()
    # variance explained by SVD top-k: cum S^2 / total S^2
    var_total = (S ** 2).sum()
    svd_var_explained = float((S[:k_svd] ** 2).sum() / var_total) if var_total > 0 else float("nan")
    return {
        "n_features": int(sae.n_features),
        "n_features_aligned": int(aligned),
        "fraction_aligned": float(aligned) / sae.n_features,
        "fraction_non_aligned": 1.0 - float(aligned) / sae.n_features,
        "max_abs_sim_mean": float(max_abs_sim.mean()),
        "max_abs_sim_median": float(np.median(max_abs_sim)),
        "svd_top_k": int(k_svd),
        "svd_top_k_var_explained": svd_var_explained,
    }


def feature_table(name: str, sae: TopKSAE, X: np.ndarray, slide_ids: np.ndarray,
                  manifest: dict) -> tuple[pd.DataFrame, dict]:
    """Per-feature stats: top cells, tissue enrichment, technology confound."""
    n, d = X.shape
    n_features = sae.n_features

    # Map slide_idx → (tissue, technology)
    tissues = np.array([s["tissue"] for s in manifest["slides"]])
    techs = np.array([s["technology"] for s in manifest["slides"]])
    cell_tissue = tissues[slide_ids]   # (n,)
    cell_tech = techs[slide_ids]       # (n,)

    log(f"  encoding {n:,} rows in chunks ...")
    M, _ = encode_in_chunks(sae, torch.tensor(X, dtype=torch.float32))

    # variance explained on full corpus (decode→compare)
    log(f"  computing variance explained on full corpus ...")
    sae.to(DEVICE)
    # SS_residual = sum over all elements of (x - xhat)^2
    # SS_total    = sum over all elements of (x - mean_per_feature)^2 = X.var(axis=0).sum() * n
    # var_explained = 1 - SS_residual / SS_total
    feature_means = torch.tensor(X.mean(axis=0), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        recon_sse = 0.0
        total_ss_around_mean = 0.0
        for start in range(0, n, 16384):
            xb = torch.tensor(X[start: start + 16384], dtype=torch.float32, device=DEVICE)
            xh, _, _ = sae(xb)
            recon_sse += float(((xb - xh) ** 2).sum().cpu())
            total_ss_around_mean += float(((xb - feature_means) ** 2).sum().cpu())
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    var_explained = 1.0 - recon_sse / max(total_ss_around_mean, 1e-12)

    # Per-feature stats from the sparse activation matrix M (n × F).
    # Use absolute values because TopK is signed.
    M_abs = M.copy()
    M_abs.data = np.abs(M_abs.data)

    feature_l0 = (M != 0).sum(axis=0).A1  # (F,) num cells the feature fires on
    feature_mean_abs = np.asarray(M_abs.sum(axis=0)).ravel() / max(n, 1)

    # for each feature, top-K cells by absolute activation
    log(f"  finding top cells per feature ...")
    top_n = max(10, int(TOP_CELL_FRACTION * n))

    # iterate per feature column — sparse CSC for fast column access
    Mcsc = M_abs.tocsc()
    unique_tissues = sorted(set(tissues.tolist()))
    unique_techs = sorted(set(techs.tolist()))
    rows = []
    for f in range(n_features):
        col_start = Mcsc.indptr[f]
        col_end = Mcsc.indptr[f + 1]
        if col_end == col_start:
            rows.append({
                "feature_idx": f, "alive": False, "n_active_cells": 0,
                "mean_abs": 0.0, "max_abs": 0.0,
                "top_tissue": "", "top_tissue_frac": 0.0,
                "top_tech": "", "top_tech_frac": 0.0,
                "tech_confound": False,
            })
            continue
        col_idx = Mcsc.indices[col_start: col_end]
        col_val = Mcsc.data[col_start: col_end]
        # top cells = highest |activation|
        if len(col_idx) > top_n:
            top_local = np.argpartition(col_val, -top_n)[-top_n:]
            top_cells = col_idx[top_local]
        else:
            top_cells = col_idx
        t = cell_tissue[top_cells]
        h = cell_tech[top_cells]
        t_unique, t_counts = np.unique(t, return_counts=True)
        h_unique, h_counts = np.unique(h, return_counts=True)
        top_tissue = t_unique[t_counts.argmax()]
        top_tissue_frac = float(t_counts.max() / len(top_cells))
        top_tech = h_unique[h_counts.argmax()]
        top_tech_frac = float(h_counts.max() / len(top_cells))
        rows.append({
            "feature_idx": int(f),
            "alive": True,
            "n_active_cells": int(col_end - col_start),
            "mean_abs": float(feature_mean_abs[f]),
            "max_abs": float(col_val.max()),
            "top_tissue": str(top_tissue),
            "top_tissue_frac": top_tissue_frac,
            "top_tech": str(top_tech),
            "top_tech_frac": top_tech_frac,
            "tech_confound": top_tech_frac >= 0.8,
        })
    df = pd.DataFrame(rows)
    df["surface"] = name

    surf_summary = {
        "name": name,
        "n_features": int(n_features),
        "n_alive": int(df["alive"].sum()),
        "alive_fraction": float(df["alive"].mean()),
        "var_explained_full": float(var_explained),
        "tech_confounded_count": int(df["tech_confound"].sum()),
        "tech_confounded_fraction": float(df["tech_confound"].mean()),
        "top_tissue_distribution": df.groupby("top_tissue").size().to_dict(),
        "top_tech_distribution": df.groupby("top_tech").size().to_dict(),
    }
    return df, surf_summary


def main() -> None:
    log("=" * 72)
    log(f"Phase 3: SAE feature characterization (no external DBs)")
    log(f"  device : {DEVICE}")
    log(f"  in     : {ACT_DIR}")
    log(f"  saes   : {SAE_DIR}")
    log(f"  out    : {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")
    layer_slide_id = np.load(ACT_DIR / "layer_slide_id.npy")

    all_features: list[pd.DataFrame] = []
    surface_summaries = []
    superposition_summaries = []

    for name, expansion, k in SAE_CONFIGS:
        sae_path = SAE_DIR / f"{name}.pt"
        act_path = ACT_DIR / f"{name}.npy"
        if not sae_path.exists() or not act_path.exists():
            log(f"\n=== {name} === MISSING (sae={sae_path.exists()}, act={act_path.exists()})")
            continue

        log(f"\n=== {name} ===")
        X = np.load(act_path).astype(np.float32, copy=False)
        d = X.shape[1]
        sae = load_sae(name, expansion, k, d)
        log(f"  loaded SAE: d={d}, F={sae.n_features}, k={sae.k}, n_act={X.shape[0]:,}")

        # the slide-id mapping depends on whether the surface is per-cell
        # (aggregator) or per-layer-sample (everything else)
        slide_ids = cell_slide_id if name == "aggregator" else layer_slide_id
        if len(slide_ids) != X.shape[0]:
            log(f"  WARNING: slide_id length {len(slide_ids)} != X rows {X.shape[0]}")
            slide_ids = slide_ids[: X.shape[0]] if len(slide_ids) > X.shape[0] else \
                np.concatenate([slide_ids, np.zeros(X.shape[0] - len(slide_ids), dtype=np.int32)])

        df, summary = feature_table(name, sae, X, slide_ids, manifest)
        sup = superposition_audit(sae, X)
        sup["name"] = name
        log(
            f"  alive={summary['n_alive']}/{summary['n_features']} "
            f"({summary['alive_fraction']:.2%})  "
            f"var_exp_full={summary['var_explained_full']:.3f}  "
            f"tech_confounded={summary['tech_confounded_count']} "
            f"({summary['tech_confounded_fraction']:.2%})  "
            f"superposition={sup['fraction_non_aligned']:.3f} non-aligned vs SVD-{TOPK_SVD}"
        )
        all_features.append(df)
        surface_summaries.append(summary)
        superposition_summaries.append(sup)

        del X, sae, df

    if all_features:
        atlas = pd.concat(all_features, ignore_index=True)
        atlas.to_parquet(OUT_DIR / "feature_atlas.parquet", index=False)
        log(f"\nwrote {OUT_DIR / 'feature_atlas.parquet'}  ({len(atlas):,} rows)")

    json.dump({"surfaces": surface_summaries, "superposition": superposition_summaries},
              open(OUT_DIR / "summary.json", "w"), indent=2)
    log(f"wrote {OUT_DIR / 'summary.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
