#!/usr/bin/env python
"""Block 1.3 — confound suite (slide/tissue/tech chi-square).

For each aggregator SAE feature, test whether its top-cell distribution
across {slide, tissue, technology} is significantly different from the
corpus baseline distribution. Chi-square test, BH-corrected.

Survival rate = features with FDR<0.05 against the corpus baseline.
A feature whose top cells are uniformly distributed (not concentrated)
fails the test → it's noise. A feature whose top cells are concentrated
in one slide → may be slide-confounded.

Output: atlas/causal/confounds.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gc

import numpy as np
import pandas as pd
import torch
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "09_block1_confound_suite.log"

TOP_FRAC = 0.001
MIN_TOP = 50


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def chi2_against_baseline(observed_counts: np.ndarray, expected_fractions: np.ndarray) -> tuple[float, float]:
    """Chi-square goodness of fit. Returns (chi2 statistic, p-value)."""
    n = observed_counts.sum()
    expected = expected_fractions * n
    # Filter zero-expected categories
    mask = expected > 0
    if not mask.any():
        return float("nan"), 1.0
    obs = observed_counts[mask]
    exp = expected[mask]
    try:
        chi2, p = stats.chisquare(obs, exp)
        return float(chi2), float(p)
    except Exception:
        return float("nan"), 1.0


def main() -> None:
    log("=" * 72)
    log("Block 1.3: confound suite")

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    log("loading activations")
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")
    log(f"  agg: {agg.shape}")

    # Slide → metadata
    slide_meta = {s["slide_idx"]: s for s in manifest["slides"]}
    cell_tissue = np.array([slide_meta[sid]["tissue"] for sid in cell_slide_id])
    cell_tech = np.array([slide_meta[sid]["technology"] for sid in cell_slide_id])

    # Corpus baseline distributions
    tissues, tissue_counts = np.unique(cell_tissue, return_counts=True)
    techs, tech_counts = np.unique(cell_tech, return_counts=True)
    slides_uniq, slide_counts = np.unique(cell_slide_id, return_counts=True)
    log(f"  tissues: {len(tissues)}, technologies: {len(techs)}, slides: {len(slides_uniq)}")

    tissue_baseline = tissue_counts / tissue_counts.sum()
    tech_baseline = tech_counts / tech_counts.sum()
    slide_baseline = slide_counts / slide_counts.sum()

    tissue_to_idx = {t: i for i, t in enumerate(tissues)}
    tech_to_idx = {t: i for i, t in enumerate(techs)}
    slide_to_idx = {s: i for i, s in enumerate(slides_uniq)}

    log("encoding aggregator → sparse feature activations")
    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    # Dense (4.5M, 2048) float32 ≈ 36 GB — OOM on 32 GB laptop.
    # Use sparse: TopK k=16 gives ~72M nonzeros ≈ 600 MB.
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
            z, idx = sae.encode(xb)
            B = end - start
            vals = z.gather(-1, idx).cpu().numpy()
            idx_np = idx.cpu().numpy().astype(np.int32)
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
    ).tocsc()
    del rows_buf, cols_buf, data_buf, agg
    gc.collect()
    log(f"  feats (CSC): shape={feats.shape}, nnz={feats.nnz:,}, {time.time()-t_enc:.1f}s")

    n_top = max(MIN_TOP, int(TOP_FRAC * n_cells))
    log(f"  top-N per feature: {n_top}")

    log("running chi-square per feature")
    indptr = feats.indptr
    indices = feats.indices
    data = feats.data
    rows = []
    for fid in range(n_features):
        start_ix = indptr[fid]
        end_ix = indptr[fid + 1]
        n_nonzero = end_ix - start_ix
        if n_nonzero < MIN_TOP:
            rows.append({
                "feature_idx": fid,
                "tissue_chi2": None, "tissue_p": None,
                "tech_chi2": None, "tech_p": None,
                "slide_chi2": None, "slide_p": None,
                "n_top_used": 0,
            })
            continue
        col_rows = indices[start_ix:end_ix]
        col_vals = np.abs(data[start_ix:end_ix])
        # Top cells by |activation|
        k = min(n_top, n_nonzero)
        top_local = np.argpartition(col_vals, -k)[-k:]
        top_ix = col_rows[top_local]

        # Build observation counts
        ttypes = cell_tissue[top_ix]
        techs_top = cell_tech[top_ix]
        slides_top = cell_slide_id[top_ix]

        tissue_obs = np.zeros(len(tissues), dtype=np.int64)
        for t in ttypes:
            tissue_obs[tissue_to_idx[t]] += 1
        tech_obs = np.zeros(len(techs), dtype=np.int64)
        for t in techs_top:
            tech_obs[tech_to_idx[t]] += 1
        slide_obs = np.zeros(len(slides_uniq), dtype=np.int64)
        for s in slides_top:
            slide_obs[slide_to_idx[s]] += 1

        ti_chi2, ti_p = chi2_against_baseline(tissue_obs, tissue_baseline)
        te_chi2, te_p = chi2_against_baseline(tech_obs, tech_baseline)
        sl_chi2, sl_p = chi2_against_baseline(slide_obs, slide_baseline)

        rows.append({
            "feature_idx": fid,
            "tissue_chi2": ti_chi2, "tissue_p": ti_p,
            "tech_chi2": te_chi2, "tech_p": te_p,
            "slide_chi2": sl_chi2, "slide_p": sl_p,
            "n_top_used": int(k),
        })

    df = pd.DataFrame(rows)

    # BH-correct each set of p-values
    for col in ["tissue_p", "tech_p", "slide_p"]:
        ps = df[col].fillna(1.0).values
        _, fdr, _, _ = multipletests(ps, method="fdr_bh")
        df[col.replace("_p", "_fdr")] = fdr

    # Survival flags: significant deviation from baseline (a real, not random feature)
    df["tissue_concentrated"] = df["tissue_fdr"] < 0.05
    df["tech_concentrated"] = df["tech_fdr"] < 0.05
    df["slide_concentrated"] = df["slide_fdr"] < 0.05

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "confounds.parquet"
    df.to_parquet(out_path, index=False)
    log(f"\nwrote {out_path} ({len(df)} rows)")

    summary = {
        "n_features": int(len(df)),
        "n_tissue_concentrated_fdr_lt_0.05": int(df["tissue_concentrated"].sum()),
        "n_tech_concentrated_fdr_lt_0.05": int(df["tech_concentrated"].sum()),
        "n_slide_concentrated_fdr_lt_0.05": int(df["slide_concentrated"].sum()),
        "n_features_passing_all_three": int(
            (df["tissue_concentrated"] & df["tech_concentrated"] & df["slide_concentrated"]).sum()
        ),
    }
    json.dump(summary, open(OUT_DIR / "confounds.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
