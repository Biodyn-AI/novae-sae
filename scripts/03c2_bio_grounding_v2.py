#!/usr/bin/env python
"""Phase 3c v2 — bio-grounded characterization with fold-change ranking.

Fixes the hub-gene artefact in 03c_bio_grounding.py: instead of ranking top
genes per feature by absolute mean expression in the top cells (which is
dominated by hub genes like EPCAM, PTPRC, PKM, CXCR4 in 60–80 % of features),
this v2 ranks by **fold change vs corpus baseline**:

  fold_change(g, f) = mean_expr(g, top cells of f) / mean_expr(g, corpus baseline)

The corpus baseline is a single random sample of cells drawn proportionally
across all slides. Both per-feature and baseline mean expression are computed
in the same single pass through the slide files.

Outputs (`atlas/novae-human-0/bio/v2/`):
  - aggregator_top_genes_v2.parquet      (gene, mean_expr, baseline_mean, fold_change)
  - aggregator_enrichment_v2.parquet     (same schema as v1)
  - bio_summary_v2.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import gseapy as gp
import novae
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "bio" / "v2"
LOG_PATH = ROOT / "logs" / "03c2_bio_grounding_v2.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TOP_CELL_FRAC = 0.001
MIN_TOP_CELLS = 50
TOP_GENES_PER_FEATURE = 20
BASELINE_FRAC = 0.05            # 5% random subsample of each slide for baseline
ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "PanglaoDB_Augmented_2021",
    "CellMarker_Augmented_2021",
]
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


def encode_to_top_cells(sae: TopKSAE, X: np.ndarray, n_top: int) -> dict[int, np.ndarray]:
    """Same as in 03c — chunked encode to find top-N cells per feature."""
    sae.to(DEVICE)
    n, _ = X.shape
    n_features = sae.n_features
    rows = []
    cols = []
    vals = []
    chunk = 32768
    with torch.no_grad():
        for start in range(0, n, chunk):
            xb = torch.tensor(X[start: start + chunk], dtype=torch.float32, device=DEVICE)
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
    M = sparse.csr_matrix((vals_np, (rows_np, cols_np)), shape=(n, n_features), dtype=np.float32).tocsc()
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


def compute_top_genes_with_fold_change(
    top_cells_per_feature: dict[int, np.ndarray],
    cell_slide_id: np.ndarray,
    manifest: dict,
) -> pd.DataFrame:
    """Single pass through the slides; for each one, compute:
    - per-feature gene sums (top cells in this slide)
    - per-slide gene sums on a 5 % random baseline subsample (corpus background)
    Then aggregate across slides and rank by fold-change.
    """
    n_features = max(top_cells_per_feature.keys()) + 1 if top_cells_per_feature else 0
    slide_starts = np.array([s["cell_offset_start"] for s in manifest["slides"]], dtype=np.int64)

    # group: slide_idx -> feature -> [local_cell_idx]
    slide_buckets: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for f, cells in top_cells_per_feature.items():
        if len(cells) == 0:
            continue
        sids = np.searchsorted(slide_starts, cells, side="right") - 1
        for sid in np.unique(sids):
            mask = sids == sid
            local = (cells[mask] - slide_starts[sid]).tolist()
            slide_buckets[int(sid)][int(f)].extend(local)

    # accumulators
    feat_gene_sum: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    feat_cell_count: dict[int, int] = defaultdict(int)
    baseline_gene_sum: dict[str, float] = defaultdict(float)
    baseline_cell_count = 0

    rng = np.random.default_rng(SEED)

    for sid in sorted(slide_buckets):
        slide = manifest["slides"][sid]
        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"  [{sid+1}/{len(manifest['slides'])}] {slide['tissue']}/{slide['name']}")
        try:
            adata = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"    ERROR load: {type(e).__name__}: {e}")
            continue
        try:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        except Exception as e:
            log(f"    WARN normalize: {type(e).__name__}: {e}")

        var_names = [str(g).lower() for g in adata.var_names.tolist()]
        X = adata.X
        is_sparse = sparse.issparse(X)
        n_cells_slide = adata.n_obs

        # baseline subsample
        n_bg = max(100, int(BASELINE_FRAC * n_cells_slide))
        bg_idx = rng.choice(n_cells_slide, size=min(n_bg, n_cells_slide), replace=False)
        sub_bg = X[bg_idx]
        bg_sums = (np.asarray(sub_bg.sum(axis=0)).ravel() if is_sparse else sub_bg.sum(axis=0))
        baseline_cell_count += len(bg_idx)
        for gi, g in enumerate(var_names):
            if bg_sums[gi] > 0:
                baseline_gene_sum[g] += float(bg_sums[gi])

        # per-feature accumulation
        for f, local_cells in slide_buckets[sid].items():
            local_arr = np.asarray(local_cells, dtype=np.int64)
            sub = X[local_arr]
            gene_sums = (np.asarray(sub.sum(axis=0)).ravel() if is_sparse else sub.sum(axis=0))
            feat_cell_count[f] += len(local_cells)
            for gi, g in enumerate(var_names):
                if gene_sums[gi] > 0:
                    feat_gene_sum[f][g] += float(gene_sums[gi])

        del adata, X
        gc.collect()

    # baseline means
    baseline_means = {g: s / max(baseline_cell_count, 1) for g, s in baseline_gene_sum.items()}

    # build dataframe with fold-change ranking
    rows = []
    for f in range(n_features):
        if not feat_gene_sum[f]:
            continue
        n_cells = feat_cell_count[f]
        if n_cells == 0:
            continue
        # compute fold change for each gene with non-zero baseline
        candidates = []
        for g, total in feat_gene_sum[f].items():
            mean_top = total / n_cells
            mean_bg = baseline_means.get(g, 0.0)
            if mean_bg <= 0:
                continue
            fc = mean_top / mean_bg
            # also require minimum absolute expression to avoid ranking up
            # extremely-low-expressed noisy genes
            if mean_top < 0.05:
                continue
            candidates.append((g, mean_top, mean_bg, fc))
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[3])
        for rank, (g, mt, mb, fc) in enumerate(candidates[:TOP_GENES_PER_FEATURE]):
            rows.append({
                "surface": "aggregator",
                "feature_idx": int(f),
                "rank": int(rank),
                "gene": g,
                "mean_expr_in_top_cells": float(mt),
                "baseline_mean_expr": float(mb),
                "fold_change": float(fc),
                "n_top_cells": int(n_cells),
            })
    return pd.DataFrame(rows)


def load_libraries() -> dict[str, dict[str, list[str]]]:
    libs: dict[str, dict[str, list[str]]] = {}
    for lib in ENRICHR_LIBRARIES:
        log(f"  loading library: {lib}")
        try:
            d = gp.get_library(lib, organism="Human")
            libs[lib] = {term: [g.upper() for g in genes] for term, genes in d.items()}
            log(f"    {len(libs[lib])} terms")
        except Exception as e:
            log(f"    FAIL {type(e).__name__}: {e}")
    return libs


def run_enrichment(top_genes_df: pd.DataFrame, libraries: dict, background: list[str]) -> pd.DataFrame:
    rows = []
    grouped = top_genes_df.groupby("feature_idx")
    n_features = len(grouped)
    log(f"  enrichment for {n_features} features × {len(libraries)} libraries")
    t0 = time.time()
    for i, (f, sub) in enumerate(grouped):
        gene_list_up = [g.upper() for g in sub.sort_values("rank")["gene"].astype(str).tolist()]
        for lib_name, gene_sets_dict in libraries.items():
            try:
                result = gp.enrich(
                    gene_list=gene_list_up,
                    gene_sets=gene_sets_dict,
                    background=background,
                    outdir=None,
                    cutoff=1.0,
                    no_plot=True,
                    verbose=False,
                )
                df = result.results if hasattr(result, "results") else result.res2d
                if df is None or len(df) == 0:
                    continue
                top = df.sort_values("Adjusted P-value").iloc[0]
                rows.append({
                    "surface": "aggregator",
                    "feature_idx": int(f),
                    "library": lib_name,
                    "top_term": str(top.get("Term", "")),
                    "p": float(top.get("P-value", float("nan"))),
                    "fdr": float(top.get("Adjusted P-value", float("nan"))),
                    "overlap": str(top.get("Overlap", "")),
                })
            except Exception:
                continue
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            log(f"    {i+1}/{n_features}  ({elapsed/60:.1f} min, ETA {(n_features-i-1)/((i+1)/elapsed)/60:.1f} min)")
    return pd.DataFrame(rows)


def main() -> None:
    log("=" * 72)
    log("Phase 3c v2: bio-grounded characterization with fold-change ranking")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading model + manifest + activations")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")
    log(f"  agg: {agg.shape}")

    log("loading aggregator SAE + finding top cells")
    sae = load_sae("aggregator", 32, 16, 64)
    n_top = max(MIN_TOP_CELLS, int(TOP_CELL_FRAC * agg.shape[0]))
    log(f"  top-cells per feature: {n_top}")
    top_cells = encode_to_top_cells(sae, agg, n_top)
    del sae

    log("\n--- top genes (fold-change ranked) ---")
    df_top = compute_top_genes_with_fold_change(top_cells, cell_slide_id, manifest)
    df_top.to_parquet(OUT_DIR / "aggregator_top_genes_v2.parquet", index=False)
    log(f"  wrote {OUT_DIR / 'aggregator_top_genes_v2.parquet'} ({len(df_top)} rows)")

    log("\n--- enrichment v2 ---")
    background = [g.upper() for g in model.cell_embedder.gene_names]
    libraries = load_libraries()
    df_enr = run_enrichment(df_top, libraries, background)
    df_enr.to_parquet(OUT_DIR / "aggregator_enrichment_v2.parquet", index=False)
    log(f"  wrote {OUT_DIR / 'aggregator_enrichment_v2.parquet'} ({len(df_enr)} rows)")

    summary = {
        "version": "v2_fold_change",
        "n_features": int(df_top["feature_idx"].nunique()),
        "top_genes_rows": len(df_top),
        "enrichment_rows": len(df_enr),
        "n_features_with_significant_enrichment": int(
            (df_enr.groupby("feature_idx")["fdr"].min() < 0.05).sum()
        ) if len(df_enr) else 0,
    }
    json.dump(summary, open(OUT_DIR / "bio_summary_v2.json", "w"), indent=2)
    log(f"\n{json.dumps(summary, indent=2)}")
    log("DONE")


if __name__ == "__main__":
    main()
