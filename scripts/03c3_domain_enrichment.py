#!/usr/bin/env python
"""Phase 3c v3 — domain-aware characterization for the aggregator SAE.

After Phase 3d populates per-slide `domains_level{7,12,20}.npy`, every cell in
the global aggregator activation array has a Novae-assigned domain label at
each of the three hierarchy levels. The labels share a global namespace
(`D1001`, `D1008`, ...), so a feature's top cells can be enriched against the
corpus-wide domain distribution rather than just slide/tissue/tech.

For each aggregator SAE feature × level, this script computes:
  - top_domain        — the most frequent domain label in the feature's top cells
  - top_domain_frac   — its fraction
  - log2_enrichment   — log2((top_frac + eps) / (corpus_frac + eps))
  - fisher_p          — Fisher's exact test, top-domain count vs the rest
  - fisher_fdr        — Benjamini-Hochberg adjusted across all features

Outputs (`atlas/novae-human-0/bio/v3/`):
  - aggregator_domain_enrichment.parquet
  - domain_summary.json

Run only after Phase 3d finishes for all 15 slides.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "bio" / "v3"
LOG_PATH = ROOT / "logs" / "03c3_domain_enrichment.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TOP_CELL_FRAC = 0.001
MIN_TOP_CELLS = 50
DOMAIN_LEVELS = [7, 12, 20]


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


def encode_to_top_cells(sae: TopKSAE, X: np.ndarray, n_top: int) -> dict[int, np.ndarray]:
    """Chunked encode → top-N cells per feature (mirrors 03c2)."""
    sae.to(DEVICE)
    n, _ = X.shape
    n_features = sae.n_features
    rows, cols, vals = [], [], []
    chunk = 32768
    with torch.no_grad():
        for start in range(0, n, chunk):
            xb = torch.tensor(X[start:start + chunk], dtype=torch.float32, device=DEVICE)
            z, _ = sae.encode(xb)
            mask = z != 0
            r, c = mask.nonzero(as_tuple=True)
            v = z[r, c].abs()
            rows.append((r + start).cpu().numpy())
            cols.append(c.cpu().numpy())
            vals.append(v.cpu().numpy())
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    rows_np = np.concatenate(rows)
    cols_np = np.concatenate(cols)
    vals_np = np.concatenate(vals)
    M = sparse.csr_matrix(
        (vals_np, (rows_np, cols_np)), shape=(n, n_features), dtype=np.float32
    ).tocsc()
    top: dict[int, np.ndarray] = {}
    for f in range(n_features):
        s, e = M.indptr[f], M.indptr[f + 1]
        if s == e:
            top[f] = np.array([], dtype=np.int64)
            continue
        idx = M.indices[s:e]
        v = M.data[s:e]
        k = min(n_top, len(v))
        top[f] = idx[np.argpartition(v, -k)[-k:]]
    return top


def build_global_domain_array(manifest: dict, level: int) -> np.ndarray:
    """Concatenate per-slide domain arrays in manifest order. Length = total cells."""
    parts = []
    for slide in manifest["slides"]:
        sdir = ACT_DIR / "per_slide" / f"{slide['tissue']}__{slide['name']}"
        path = sdir / f"domains_level{level}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing domain file for slide {slide['name']}: {path}\n"
                f"Phase 3d has not finished yet. Run scripts/03d_assign_domains.py."
            )
        arr = np.load(path, allow_pickle=True).astype(str)
        if len(arr) != slide["n_cells"]:
            raise ValueError(
                f"Slide {slide['name']}: domain array len {len(arr)} != "
                f"manifest n_cells {slide['n_cells']}"
            )
        parts.append(arr)
    return np.concatenate(parts)


def enrichment_for_level(
    top_cells_per_feature: dict[int, np.ndarray],
    domains_global: np.ndarray,
    level: int,
) -> pd.DataFrame:
    """Per-feature top-domain + Fisher-exact enrichment vs corpus baseline."""
    unique, corpus_counts = np.unique(domains_global, return_counts=True)
    label_to_idx = {lab: i for i, lab in enumerate(unique)}
    n_labels = len(unique)
    n_total = int(corpus_counts.sum())
    log(f"  level {level}: {n_labels} unique labels, {n_total:,} cells")

    rows = []
    raw_p = []
    for f, cells in top_cells_per_feature.items():
        if len(cells) == 0:
            rows.append(
                {
                    "feature_idx": int(f),
                    "level": level,
                    "n_top_cells": 0,
                    "top_domain": "",
                    "top_domain_frac": float("nan"),
                    "corpus_frac": float("nan"),
                    "log2_enrichment": float("nan"),
                    "fisher_p": float("nan"),
                }
            )
            raw_p.append(1.0)
            continue
        labels = domains_global[cells]
        u, c = np.unique(labels, return_counts=True)
        order = np.argsort(-c)
        top_lab = u[order[0]]
        top_count = int(c[order[0]])
        n_top = int(len(cells))
        top_idx = label_to_idx[top_lab]
        corpus_top_count = int(corpus_counts[top_idx])
        # Fisher's exact test:
        #            in top set | not in top set
        # top_lab        a              b
        # other          c              d
        a = top_count
        b = corpus_top_count - top_count
        c_ = n_top - top_count
        d_ = n_total - corpus_top_count - c_
        if min(a, b, c_, d_) < 0:
            p = 1.0
        else:
            try:
                _, p = stats.fisher_exact([[a, b], [c_, d_]], alternative="greater")
            except Exception:
                p = 1.0
        eps = 1e-9
        top_frac = top_count / n_top
        corpus_frac = corpus_top_count / n_total
        log2enr = float(np.log2((top_frac + eps) / (corpus_frac + eps)))
        rows.append(
            {
                "feature_idx": int(f),
                "level": level,
                "n_top_cells": n_top,
                "top_domain": str(top_lab),
                "top_domain_frac": float(top_frac),
                "corpus_frac": float(corpus_frac),
                "log2_enrichment": log2enr,
                "fisher_p": float(p),
            }
        )
        raw_p.append(float(p))

    raw_p = np.array(raw_p, dtype=np.float64)
    raw_p = np.nan_to_num(raw_p, nan=1.0)
    _, fdr, _, _ = multipletests(raw_p, method="fdr_bh")
    for i, r in enumerate(rows):
        r["fisher_fdr"] = float(fdr[i])
    return pd.DataFrame(rows)


def main() -> None:
    log("=" * 72)
    log("Phase 3c v3: domain enrichment for aggregator SAE")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading manifest + activations")
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    log(f"  agg: {agg.shape}")

    log("loading aggregator SAE + finding top cells")
    sae = load_sae("aggregator", 32, 16, 64)
    n_top = max(MIN_TOP_CELLS, int(TOP_CELL_FRAC * agg.shape[0]))
    log(f"  top-cells per feature: {n_top}")
    top_cells = encode_to_top_cells(sae, agg, n_top)
    del sae

    all_rows = []
    summary = {"levels": {}}
    for level in DOMAIN_LEVELS:
        log(f"\n--- level {level} ---")
        domains = build_global_domain_array(manifest, level)
        df_lvl = enrichment_for_level(top_cells, domains, level)
        all_rows.append(df_lvl)
        n_sig = int((df_lvl["fisher_fdr"] < 0.05).sum())
        n_strong = int((df_lvl["log2_enrichment"] > 2).sum())
        summary["levels"][f"level_{level}"] = {
            "n_unique_domains": int(df_lvl["top_domain"].nunique()),
            "n_features_with_fdr_lt_0.05": n_sig,
            "n_features_with_log2enr_gt_2": n_strong,
            "median_log2_enrichment": float(df_lvl["log2_enrichment"].median()),
        }
        log(
            f"  {n_sig}/{len(df_lvl)} features FDR<0.05, "
            f"{n_strong}/{len(df_lvl)} log2-enr>2, "
            f"median log2-enr={summary['levels'][f'level_{level}']['median_log2_enrichment']:.2f}"
        )

    df = pd.concat(all_rows, ignore_index=True)
    out_pq = OUT_DIR / "aggregator_domain_enrichment.parquet"
    df.to_parquet(out_pq, index=False)
    log(f"\nwrote {out_pq} ({len(df)} rows)")

    summary["n_features"] = int(df["feature_idx"].nunique())
    summary["total_rows"] = int(len(df))
    json.dump(summary, open(OUT_DIR / "domain_summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
