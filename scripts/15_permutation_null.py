#!/usr/bin/env python
"""§4.7/5 — permutation null for top-gene enrichment tests.

For the aggregator SAE, re-run the top-gene identification + enrichment
pipeline under K=30 random cell-label shuffles to establish a null
distribution of the best-library FDR achieved by each feature under the
null hypothesis of no label-feature association. Then compare each
feature's real FDR to the shuffle null's 5th percentile → flag features
whose enrichments are no better than a random label assignment.

Simplified version of the full per-library permutation null:
  - For each feature, the "null enrichment score" is the fraction of top-N
    cells that fall into the most-common Enrichr library category AFTER
    shuffling cell labels.
  - Instead of re-running Enrichr (expensive web API), we use a local proxy:
    each feature's "enrichment strength" = its `top_domain_l7_frac` (the
    fraction of top cells in the single most-enriched l7 domain). Under
    label shuffle, this should approach the corpus baseline.

Output: atlas/novae-human-0/causal/permutation_null.parquet
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

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "15_permutation_null.log"

TOP_FRAC = 0.001
MIN_TOP = 50
N_SHUFFLES = 30
RNG_SEED = 2026


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_global_domains(level: int) -> np.ndarray:
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    chunks = []
    for s in manifest["slides"]:
        path = ACT_DIR / "per_slide" / f"{s['tissue']}__{s['name']}" / f"domains_level{level}.npy"
        chunks.append(np.load(path, allow_pickle=True).astype(object))
    return np.concatenate(chunks)


def encode_sparse(agg, sae, k_sae, chunk=32768):
    n_cells = agg.shape[0]
    rows = np.empty(n_cells * k_sae, dtype=np.int64)
    cols = np.empty(n_cells * k_sae, dtype=np.int32)
    data = np.empty(n_cells * k_sae, dtype=np.float32)
    write = 0
    with torch.no_grad():
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            xb = torch.tensor(agg[start:end], dtype=torch.float32)
            z, idx = sae.encode(xb)
            B = end - start
            vals = z.gather(-1, idx).cpu().numpy()
            idx_np = idx.cpu().numpy().astype(np.int32)
            n = B * k_sae
            rows[write:write + n] = np.repeat(np.arange(start, end, dtype=np.int64), k_sae)
            cols[write:write + n] = idx_np.ravel()
            data[write:write + n] = vals.ravel()
            write += n
    return sparse.csr_matrix(
        (data[:write], (rows[:write], cols[:write])),
        shape=(n_cells, sae.n_features),
    ).tocsc()


def main() -> None:
    log("=" * 72)
    log("§4.7/5 permutation null for enrichment tests")

    log("loading data")
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")
    slide_meta = {s["slide_idx"]: s for s in manifest["slides"]}
    cell_tissue = np.array([slide_meta[sid]["tissue"] for sid in cell_slide_id])
    n_cells = agg.shape[0]
    log(f"  agg: {agg.shape}")

    l7 = load_global_domains(7)
    l7_classes, l7_inv = np.unique(l7, return_inverse=True)
    del l7
    log(f"  l7: {len(l7_classes)} classes")

    tissues, tissue_inv = np.unique(cell_tissue, return_inverse=True)
    log(f"  tissues: {len(tissues)}")

    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    log("encoding → sparse CSC")
    t0 = time.time()
    feats = encode_sparse(agg, sae, 16)
    del agg, sae
    gc.collect()
    log(f"  {feats.shape}, nnz={feats.nnz:,}, {time.time()-t0:.1f}s")

    n_features = feats.shape[1]
    n_top = max(MIN_TOP, int(TOP_FRAC * n_cells))
    indptr = feats.indptr
    indices = feats.indices
    data = feats.data

    rng = np.random.default_rng(RNG_SEED)

    log(f"computing baseline + {N_SHUFFLES} shuffled l7 and tissue top-class fracs per feature")
    t_start = time.time()

    # Baseline: per-feature top_l7_frac and top_tissue_frac
    baseline_l7 = np.zeros(n_features, dtype=np.float32)
    baseline_tissue = np.zeros(n_features, dtype=np.float32)
    for fid in range(n_features):
        si, ei = indptr[fid], indptr[fid + 1]
        nnz = ei - si
        if nnz < MIN_TOP:
            continue
        col_rows = indices[si:ei]
        col_vals = np.abs(data[si:ei])
        k = min(n_top, nnz)
        top_local = np.argpartition(col_vals, -k)[-k:]
        top_ix = col_rows[top_local]
        # l7 top-class frac
        top_l7 = l7_inv[top_ix]
        baseline_l7[fid] = float(np.bincount(top_l7, minlength=len(l7_classes)).max() / k)
        # tissue top-class frac
        top_tissue = tissue_inv[top_ix]
        baseline_tissue[fid] = float(np.bincount(top_tissue, minlength=len(tissues)).max() / k)

    log(f"  baseline done in {time.time()-t_start:.1f}s")

    # Shuffled: for each shuffle, permute l7_inv and tissue_inv, recompute
    shuffle_l7 = np.zeros((N_SHUFFLES, n_features), dtype=np.float32)
    shuffle_tissue = np.zeros((N_SHUFFLES, n_features), dtype=np.float32)

    for s in range(N_SHUFFLES):
        l7_perm = rng.permutation(l7_inv)
        tissue_perm = rng.permutation(tissue_inv)
        for fid in range(n_features):
            si, ei = indptr[fid], indptr[fid + 1]
            nnz = ei - si
            if nnz < MIN_TOP:
                continue
            col_rows = indices[si:ei]
            col_vals = np.abs(data[si:ei])
            k = min(n_top, nnz)
            top_local = np.argpartition(col_vals, -k)[-k:]
            top_ix = col_rows[top_local]
            top_l7_s = l7_perm[top_ix]
            shuffle_l7[s, fid] = float(np.bincount(top_l7_s, minlength=len(l7_classes)).max() / k)
            top_tissue_s = tissue_perm[top_ix]
            shuffle_tissue[s, fid] = float(np.bincount(top_tissue_s, minlength=len(tissues)).max() / k)
        if (s + 1) % 10 == 0 or s == N_SHUFFLES - 1:
            log(f"  shuffle {s+1}/{N_SHUFFLES} done ({time.time()-t_start:.0f}s)")

    # Per-feature: z-score of baseline vs shuffle null
    shuffle_l7_mean = shuffle_l7.mean(axis=0)
    shuffle_l7_std = shuffle_l7.std(axis=0)
    shuffle_tissue_mean = shuffle_tissue.mean(axis=0)
    shuffle_tissue_std = shuffle_tissue.std(axis=0)

    z_l7 = np.where(shuffle_l7_std > 0, (baseline_l7 - shuffle_l7_mean) / shuffle_l7_std, 0.0)
    z_tissue = np.where(shuffle_tissue_std > 0, (baseline_tissue - shuffle_tissue_mean) / shuffle_tissue_std, 0.0)

    # p-value: fraction of shuffles with frac >= baseline (one-sided)
    p_l7 = (shuffle_l7 >= baseline_l7[None, :]).mean(axis=0)
    p_tissue = (shuffle_tissue >= baseline_tissue[None, :]).mean(axis=0)

    rows = []
    for fid in range(n_features):
        rows.append({
            "feature_idx": fid,
            "baseline_l7_frac": float(baseline_l7[fid]),
            "shuffle_l7_mean": float(shuffle_l7_mean[fid]),
            "shuffle_l7_std": float(shuffle_l7_std[fid]),
            "z_l7": float(z_l7[fid]),
            "p_l7": float(p_l7[fid]),
            "baseline_tissue_frac": float(baseline_tissue[fid]),
            "shuffle_tissue_mean": float(shuffle_tissue_mean[fid]),
            "shuffle_tissue_std": float(shuffle_tissue_std[fid]),
            "z_tissue": float(z_tissue[fid]),
            "p_tissue": float(p_tissue[fid]),
            "survives_l7_perm": bool(p_l7[fid] < 0.05),
            "survives_tissue_perm": bool(p_tissue[fid] < 0.05),
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "permutation_null.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {out_path} ({len(df)} rows, total {time.time()-t_start:.0f}s)")

    active = baseline_l7 > 0
    n_active = int(active.sum())
    summary = {
        "n_features": int(len(df)),
        "n_active": n_active,
        "n_shuffles": N_SHUFFLES,
        "n_survives_l7_perm": int(df["survives_l7_perm"].sum()),
        "n_survives_tissue_perm": int(df["survives_tissue_perm"].sum()),
        "fraction_survives_l7_perm": float(df["survives_l7_perm"].sum() / n_active) if n_active else None,
        "fraction_survives_tissue_perm": float(df["survives_tissue_perm"].sum() / n_active) if n_active else None,
        "median_z_l7": float(np.median(z_l7[active])),
        "median_z_tissue": float(np.median(z_tissue[active])),
    }
    json.dump(summary, open(OUT_DIR / "permutation_null.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
