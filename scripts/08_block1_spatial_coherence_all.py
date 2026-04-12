#!/usr/bin/env python
"""Block 1.2 — spatial coherence (Moran's I) on ALL 2048 aggregator features (H7).

Currently we have Moran's I on only 250 features (the causal subset). This
script computes it on every aggregator SAE feature, per slide, then aggregates.

For each (feature, slide):
  1. Find cells in this slide where the feature fires (top 1% by activation).
  2. If <50 active cells, skip.
  3. Build k=10 spatial weights matrix from cell coordinates.
  4. Compute Moran's I on the activation vector.

Aggregate per feature: mean Moran's I across slides where it has signal.

H7 prediction: mean Moran's I significantly above zero, increasing with depth.

Output: atlas/causal/spatial_coherence_all.parquet
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "08_block1_spatial_coherence_all.log"

K_NEIGHBORS = 10
MIN_ACTIVE = 50
TOP_FRAC = 0.01  # top 1% of cells


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def morans_i(values: np.ndarray, W: sparse.csr_matrix) -> float:
    """Standard Moran's I given a sparse weight matrix W and value vector."""
    n = len(values)
    if n < 2:
        return float("nan")
    v = values - values.mean()
    var = (v ** 2).sum()
    if var <= 0:
        return float("nan")
    # I = (n / sum(W)) * (v.T @ W @ v) / var
    Wv = W @ v
    num = float(v @ Wv)
    s = float(W.sum())
    if s <= 0:
        return float("nan")
    return (n / s) * (num / var)


def main() -> None:
    log("=" * 72)
    log("Block 1.2: spatial coherence on all 2048 aggregator features")

    log("loading aggregator activations + SAE")
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    log(f"  agg: {agg.shape}")
    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    # Encode all activations to feature-space as sparse CSR.
    # Dense would be (4.5M, 2048) float32 ≈ 36 GB — OOM on 32 GB laptop.
    # Sparse k=16 is ~72M nonzeros ≈ 600 MB.
    log("encoding aggregator → sparse feature activations")
    n_cells = agg.shape[0]
    n_features = 2048
    k_sae = 16
    chunk = 32768
    rows_buf = np.empty(n_cells * k_sae, dtype=np.int64)
    cols_buf = np.empty(n_cells * k_sae, dtype=np.int32)
    data_buf = np.empty(n_cells * k_sae, dtype=np.float32)
    write = 0
    t_enc = time.time()
    with torch.no_grad():
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            xb = torch.tensor(agg[start:end], dtype=torch.float32)
            z, idx = sae.encode(xb)  # z: (B, 2048); idx: (B, k)
            B = end - start
            vals = z.gather(-1, idx).cpu().numpy()  # (B, k) signed
            idx_np = idx.cpu().numpy().astype(np.int32)  # (B, k)
            n = B * k_sae
            rows_buf[write:write + n] = np.repeat(np.arange(start, end, dtype=np.int64), k_sae)
            cols_buf[write:write + n] = idx_np.ravel()
            data_buf[write:write + n] = vals.ravel()
            write += n
    rows_buf = rows_buf[:write]
    cols_buf = cols_buf[:write]
    data_buf = data_buf[:write]
    feats = sparse.csr_matrix(
        (data_buf, (rows_buf, cols_buf)),
        shape=(n_cells, n_features),
    )
    del rows_buf, cols_buf, data_buf, agg
    gc.collect()
    log(f"  feats (CSR): shape={feats.shape}, nnz={feats.nnz:,}, {time.time()-t_enc:.1f}s")

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")

    # Per-feature accumulator: list of (slide_idx, n_active, morans_i)
    feature_results = {f: [] for f in range(n_features)}

    for slide in manifest["slides"]:
        sid = slide["slide_idx"]
        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"\n[{sid+1}/{len(manifest['slides'])}] {slide['tissue']}/{slide['name']}")

        # Get cell mask for this slide
        cell_mask = cell_slide_id == sid
        slide_cells_global = np.where(cell_mask)[0]
        n_cells_slide = len(slide_cells_global)
        log(f"  {n_cells_slide:,} cells in this slide")

        # Load coordinates
        try:
            a = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"  ERROR load: {e}")
            continue
        if "spatial" not in a.obsm:
            log(f"  no spatial obsm, skip")
            del a
            continue
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        if len(coords) != n_cells_slide:
            log(f"  WARN coord count {len(coords)} != cell count {n_cells_slide}")
            del a
            continue
        del a

        # Build k-NN spatial weights once for this slide
        t_knn = time.time()
        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, n_jobs=-1).fit(coords)
        _, idx = nbrs.kneighbors(coords)
        # Drop self (first neighbor)
        idx = idx[:, 1:]
        rows = np.repeat(np.arange(n_cells_slide), K_NEIGHBORS)
        cols = idx.ravel()
        data = np.ones_like(rows, dtype=np.float32)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells_slide, n_cells_slide))
        # Symmetric
        W = (W + W.T) / 2
        log(f"  k-NN graph built in {time.time()-t_knn:.1f}s")

        # Per-feature: extract activation on slide cells, compute Moran's I
        # Row-slice from global CSR, convert to CSC for fast column access.
        slide_feats_csr = feats[slide_cells_global]  # (n_cells_slide, n_features) CSR
        slide_feats = slide_feats_csr.tocsc()
        del slide_feats_csr

        t_mi = time.time()
        n_computed = 0
        # CSC column pointers — directly jump to each column's nonzeros.
        indptr = slide_feats.indptr
        indices = slide_feats.indices
        data = slide_feats.data
        for fid in range(n_features):
            start_ix = indptr[fid]
            end_ix = indptr[fid + 1]
            n_nonzero = end_ix - start_ix
            if n_nonzero < MIN_ACTIVE:
                continue
            col = np.zeros(n_cells_slide, dtype=np.float32)
            col[indices[start_ix:end_ix]] = data[start_ix:end_ix]
            mi = morans_i(np.abs(col), W)
            if np.isnan(mi):
                continue
            feature_results[fid].append((sid, int(n_nonzero), float(mi)))
            n_computed += 1
        log(f"  computed Moran's I for {n_computed} features in {time.time()-t_mi:.1f}s")

        del slide_feats, W
        gc.collect()

    # Aggregate per feature
    rows = []
    for fid, entries in feature_results.items():
        if not entries:
            rows.append({
                "feature_idx": fid,
                "morans_i_mean": None,
                "morans_i_median": None,
                "n_slides": 0,
                "n_active_total": 0,
            })
            continue
        ns = [n for _, n, _ in entries]
        mis = [mi for _, _, mi in entries]
        rows.append({
            "feature_idx": fid,
            "morans_i_mean": float(np.mean(mis)),
            "morans_i_median": float(np.median(mis)),
            "n_slides": len(entries),
            "n_active_total": int(sum(ns)),
        })
    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "spatial_coherence_all.parquet"
    df.to_parquet(out_path, index=False)
    log(f"\nwrote {out_path} ({len(df)} rows)")

    has_signal = df["morans_i_mean"].notna()
    summary = {
        "n_features": int(len(df)),
        "n_with_signal": int(has_signal.sum()),
        "n_high_coherence_gt_0.5": int((df["morans_i_mean"] > 0.5).sum()),
        "n_above_0.1": int((df["morans_i_mean"] > 0.1).sum()),
        "median_morans_i": float(df.loc[has_signal, "morans_i_mean"].median()) if has_signal.any() else None,
        "mean_morans_i": float(df.loc[has_signal, "morans_i_mean"].mean()) if has_signal.any() else None,
    }
    json.dump(summary, open(OUT_DIR / "spatial_coherence_all.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
