#!/usr/bin/env python
"""Reviewer control: cross-technology coherence baseline on raw expression.

If ρ(mean gene expression on Xenium vs. MERSCOPE) is already ≈0.56,
the SAE features add nothing beyond the expression data. If it's <0.3,
the SAE has earned the claim.

For each of the 7 cross-technology pairs from cross_tech_coherence.parquet,
load the two h5ad files, intersect the gene panels, compute mean
log-expression per gene, and correlate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ATLAS = ROOT / "atlas" / "novae-human-0"
OUT_DIR = ATLAS / "causal" / "reviewer_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "28_crosstech_rawexpr_baseline.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def mean_logexpr(path: Path) -> pd.Series:
    """Per-gene mean log-expression. The h5ad files in this corpus store
    log1p-normalized values in .X already; we just average across cells."""
    a = ad.read_h5ad(path, backed="r")
    X = a.X
    genes = a.var_names.astype(str)
    n_cells, n_genes = X.shape
    out = np.zeros(n_genes, dtype=np.float64)
    chunk = 50000
    for s in range(0, n_cells, chunk):
        sl = X[s:s + chunk]
        if hasattr(sl, "toarray"):
            sl = sl.toarray()
        sl = np.asarray(sl, dtype=np.float32)
        out += sl.sum(axis=0)
    out /= max(n_cells, 1)
    return pd.Series(out, index=genes)


def main() -> None:
    log("=" * 72)
    log("Cross-technology raw-expression baseline")

    ref = pd.read_parquet(ATLAS / "causal" / "cross_tech_coherence.parquet")
    log(f"{len(ref)} tissue pairs")

    rows = []
    for _, row in ref.iterrows():
        tissue = row["tissue"]
        xen = row["xenium_slide"]
        alt = row["alt_slide"]
        alt_tech = row["alt_tech"]
        xen_path = DATA_DIR / tissue / f"{xen}.h5ad"
        alt_path = DATA_DIR / tissue / f"{alt}.h5ad"
        if not xen_path.exists():
            log(f"  MISSING: {xen_path}")
            continue
        if not alt_path.exists():
            log(f"  MISSING: {alt_path}")
            continue
        log(f"[{tissue}] xenium={xen}  alt({alt_tech})={alt}")
        try:
            t0 = time.time()
            x_prof = mean_logexpr(xen_path)
            a_prof = mean_logexpr(alt_path)
            log(f"  profile compute: {time.time() - t0:.1f}s  "
                f"(xen_n={len(x_prof)}, alt_n={len(a_prof)})")
        except Exception as e:
            log(f"  ERROR: {type(e).__name__}: {e}")
            continue
        # Lowercase genes to merge across case differences
        x_prof.index = x_prof.index.str.lower()
        a_prof.index = a_prof.index.str.lower()
        common = x_prof.index.intersection(a_prof.index)
        log(f"  common genes: {len(common)}")
        xv = x_prof.loc[common].values
        av = a_prof.loc[common].values
        rho, rho_p = spearmanr(xv, av)
        r, r_p = pearsonr(xv, av)
        rows.append({
            "tissue": tissue,
            "xenium_slide": xen,
            "alt_slide": alt,
            "alt_tech": alt_tech,
            "n_common_genes": int(len(common)),
            "raw_expr_spearman": float(rho),
            "raw_expr_spearman_p": float(rho_p),
            "raw_expr_pearson": float(r),
            "raw_expr_pearson_p": float(r_p),
            "sae_feature_spearman": float(row["spearman_rho"]),
        })
        log(f"  raw-expr ρ={rho:.3f}  (SAE-feature ρ={row['spearman_rho']:.3f})")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "G_crosstech_rawexpr_baseline.parquet", index=False)
    summary = {
        "n_pairs": int(len(df)),
        "raw_expr_spearman_mean": float(df["raw_expr_spearman"].mean()) if len(df) else None,
        "sae_feature_spearman_mean": float(df["sae_feature_spearman"].mean()) if len(df) else None,
        "delta_sae_over_raw": float(
            df["sae_feature_spearman"].mean() - df["raw_expr_spearman"].mean()
        ) if len(df) else None,
        "interpretation": (
            "If SAE ρ > raw-expr ρ, the SAE captures tech-invariant structure "
            "beyond the expression data itself. If SAE ρ ≈ raw-expr ρ, the "
            "cross-tech coherence claim is driven by shared biology, not by "
            "the model."
        ),
    }
    (OUT_DIR / "G_crosstech_rawexpr_baseline.summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
