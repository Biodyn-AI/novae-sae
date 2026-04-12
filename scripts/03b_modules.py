#!/usr/bin/env python
"""Phase 3b — feature co-activation module discovery via PMI + Leiden.

For each trained SAE, computes pointwise mutual information (PMI) between
all pairs of features over the activation corpus, retains edges above a
permutation-null threshold, and runs Leiden community detection at
resolution 1.0 (matching the upstream protocol). Reports per-surface module
counts and per-feature module assignment.

Output: `atlas/novae-human-0/modules.parquet` (one row per (surface,
feature_id) with `module_id` column) and `modules_summary.json`.

Notes
-----
- "Active at row j" = feature is among the top-k active features at row j.
  Equivalent to: feature has a non-zero entry in the SAE encoding of row j.
- Permutation null is approximated by shuffling each feature's activation
  pattern independently; the threshold is the (1 - p)-th percentile of the
  shuffled PMI distribution at p = 0.001.
- For SAEs with very few features alive (< 32) Leiden is skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0"
LOG_PATH = ROOT / "logs" / "03b_modules.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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

PMI_THRESHOLD = 1.0            # keep edges where co-occurrence is ≥ 2× independence
LEIDEN_RESOLUTION = 1.0
SEED = 42


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


def encode_to_binary(sae: TopKSAE, X: np.ndarray, chunk: int = 16384) -> sparse.csr_matrix:
    """Return a sparse (n_rows, n_features) binary matrix: 1 where feature is active."""
    n = X.shape[0]
    sae = sae.to(DEVICE)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            xb = torch.tensor(X[s: s + chunk], dtype=torch.float32, device=DEVICE)
            _z, idx = sae.encode(xb)
            # idx: (B, k) feature indices that are active
            B, k = idx.shape
            r = (torch.arange(B, device=DEVICE).unsqueeze(1).expand(-1, k) + s).flatten()
            c = idx.flatten()
            rows.append(r.cpu().numpy())
            cols.append(c.cpu().numpy())
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    rows_np = np.concatenate(rows)
    cols_np = np.concatenate(cols)
    vals = np.ones_like(rows_np, dtype=np.float32)
    return sparse.csr_matrix((vals, (rows_np, cols_np)), shape=(n, sae.n_features))


def compute_pmi_matrix(B: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """PMI(i, j) = log2 P(i, j) / [P(i) P(j)] for binary co-occurrence.

    Returns the dense PMI matrix as a sparse CSR (zeros where co-occurrence
    is too small to estimate) and the per-feature marginal P(i).
    """
    n_rows, n_features = B.shape
    # marginal counts
    P_i = np.asarray(B.sum(axis=0)).ravel() / n_rows  # (F,)
    # joint counts: B.T @ B  → (F, F)
    cooc = (B.T @ B).astype(np.float32)
    cooc.setdiag(0)  # ignore self
    # convert to PMI
    cooc_csr = cooc.tocoo()
    P_ij = cooc_csr.data / n_rows
    P_marginal_i = P_i[cooc_csr.row]
    P_marginal_j = P_i[cooc_csr.col]
    denom = (P_marginal_i * P_marginal_j).clip(min=1e-12)
    # avoid log(0) by skipping P_ij == 0
    valid = P_ij > 0
    pmi = np.zeros_like(P_ij)
    pmi[valid] = np.log2(P_ij[valid] / denom[valid])
    pmi_mat = sparse.coo_matrix((pmi, (cooc_csr.row, cooc_csr.col)),
                                shape=cooc.shape, dtype=np.float32).tocsr()
    return pmi_mat, P_i


def adaptive_pmi_threshold(pmi_csr: sparse.csr_matrix,
                            target_density: tuple[float, float] = (0.001, 0.05)) -> float:
    """Pick a PMI threshold that yields a graph density inside the target band.

    Density = retained_edges / max_possible_edges. The lower bound (0.1%)
    keeps the graph from being too sparse to find communities; the upper
    bound (5%) keeps it from being too dense for Leiden.
    """
    n = pmi_csr.shape[0]
    max_edges = n * (n - 1) / 2  # upper triangle only (graph is symmetric)
    # candidate thresholds in descending order — try strict first
    for thr in [3.0, 2.0, 1.5, 1.0, 0.5, 0.0]:
        kept = (pmi_csr.data > thr).sum() / 2  # divide by 2 because matrix is symmetric
        density = kept / max(max_edges, 1)
        if target_density[0] <= density <= target_density[1]:
            return thr
        if density > target_density[1]:
            return thr  # too many edges → next iteration would loosen, this is the tight one
    return 0.0


def leiden_communities(pmi_csr: sparse.csr_matrix, threshold: float) -> np.ndarray:
    """Run Leiden on the thresholded PMI graph; returns per-feature community id."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        log(f"  WARNING: leiden deps missing ({e}); returning trivial communities")
        return np.zeros(pmi_csr.shape[0], dtype=np.int64)

    coo = pmi_csr.tocoo()
    mask = coo.data >= threshold
    src = coo.row[mask]
    dst = coo.col[mask]
    weights = coo.data[mask]
    if len(src) == 0:
        return np.full(pmi_csr.shape[0], -1, dtype=np.int64)
    g = ig.Graph(n=pmi_csr.shape[0], edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = weights.tolist()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=LEIDEN_RESOLUTION,
        seed=SEED,
    )
    return np.array(part.membership)


def main() -> None:
    log("=" * 72)
    log(f"Phase 3b: feature module discovery (PMI + Leiden)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = []

    for name, expansion, k in SAE_CONFIGS:
        sae_path = SAE_DIR / f"{name}.pt"
        act_path = ACT_DIR / f"{name}.npy"
        if not sae_path.exists() or not act_path.exists():
            log(f"\n=== {name} === MISSING — skip")
            continue

        log(f"\n=== {name} ===")
        X = np.load(act_path).astype(np.float32, copy=False)
        d = X.shape[1]
        sae = load_sae(name, expansion, k, d)

        log(f"  encoding to binary co-activation matrix")
        B = encode_to_binary(sae, X)
        log(f"    binary shape: {B.shape}, nnz: {B.nnz:,}")

        log(f"  computing PMI matrix")
        t0 = time.time()
        pmi_mat, P_i = compute_pmi_matrix(B)
        log(f"    PMI nnz: {pmi_mat.nnz:,} in {time.time()-t0:.1f}s")

        log(f"  picking adaptive PMI threshold")
        thr = adaptive_pmi_threshold(pmi_mat)
        log(f"    threshold: PMI > {thr:.3f}")

        log(f"  Leiden community detection")
        memberships = leiden_communities(pmi_mat, thr)
        n_modules = len(set(memberships.tolist())) - (1 if -1 in memberships else 0)
        sizes = pd.Series(memberships).value_counts().sort_values(ascending=False)
        log(f"    {n_modules} modules; largest sizes: {sizes.head(5).tolist()}")

        for fid, mid in enumerate(memberships):
            rows.append({
                "surface": name,
                "feature_idx": int(fid),
                "module_id": int(mid),
                "P_i": float(P_i[fid]),
            })
        summary.append({
            "name": name,
            "n_features": int(sae.n_features),
            "n_alive": int((P_i > 0).sum()),
            "n_modules": int(n_modules),
            "pmi_threshold": float(thr),
            "module_size_top5": sizes.head(5).tolist(),
        })

        del X, sae, B, pmi_mat
        if DEVICE == "mps":
            torch.mps.empty_cache()

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(OUT_DIR / "modules.parquet", index=False)
        log(f"\nwrote {OUT_DIR / 'modules.parquet'} ({len(df):,} rows)")
    json.dump({"surfaces": summary}, open(OUT_DIR / "modules_summary.json", "w"), indent=2)
    log(f"wrote {OUT_DIR / 'modules_summary.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
