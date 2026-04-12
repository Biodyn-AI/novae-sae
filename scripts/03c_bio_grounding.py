#!/usr/bin/env python
"""Phase 3c — biology-grounded SAE characterization.

Adds top genes + GSE enrichment to the feature atlas using gseapy + Enrichr
gene-set libraries (downloaded automatically on first use). Operates on the
two SAE surfaces where gene-level interpretation makes sense:

  1. **aggregator** (the headline cell-in-niche SAE):
     For each feature, find the top 0.1% cells by activation magnitude across
     the corpus, group them by source slide, load each slide once, average the
     normalized gene expression of those cells per feature. Top genes per
     feature = top by mean expression.

  2. **cell_embedder** (the pre-graph scGPT-space SAE):
     Direct decoder projection — the SAE decoder columns live in the same
     512-d space as the per-cell embedded features, which are themselves a
     linear combination of (linear+normalized) full-vocab gene embeddings.
     Top genes per feature = argmax of `(processed_full_gene_emb @ decoder_col)`.

Gene-set enrichment via gseapy.enrich() against:
  - GO_Biological_Process_2023
  - KEGG_2021_Human
  - Reactome_2022
  - PanglaoDB_Augmented_2021
  - CellMarker_Augmented_2021

Outputs (`atlas/novae-human-0/bio/`):
  - aggregator_top_genes.parquet     (feature, rank, gene, mean_expr)
  - cell_embedder_top_genes.parquet  (same)
  - aggregator_enrichment.parquet    (feature, library, term, p, fdr, overlap)
  - cell_embedder_enrichment.parquet (same)
  - bio_summary.json                 aggregate stats

Wall-time on M2 Pro: dominated by enrichr queries (gseapy uses local libraries
once downloaded). Aggregator: 2048 features × 5 libraries ≈ 10 min.
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
import torch.nn.functional as Fnn
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "bio"
LOG_PATH = ROOT / "logs" / "03c_bio_grounding.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TOP_CELL_FRAC_AGG = 0.001    # top 0.1% of cells per aggregator feature
MIN_TOP_CELLS = 50           # at least 50 cells per feature even in small splits
TOP_GENES_PER_FEATURE = 20   # match upstream §feature characterization
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


# ---------------------------------------------------------------- aggregator path

def encode_aggregator_to_sparse_topcells(sae: TopKSAE, X: np.ndarray, n_top: int) -> dict[int, np.ndarray]:
    """For each feature, return the indices of its top-`n_top` cells across X.

    Done in chunks to avoid materializing the full (n, F) dense activation matrix.
    """
    sae.to(DEVICE)
    n, _ = X.shape
    n_features = sae.n_features
    chunk = 32768

    rows = []
    cols = []
    vals = []
    with torch.no_grad():
        for start in range(0, n, chunk):
            xb = torch.tensor(X[start: start + chunk], dtype=torch.float32, device=DEVICE)
            z, _idx = sae.encode(xb)
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
    M = sparse.csr_matrix((vals_np, (rows_np, cols_np)), shape=(n, n_features), dtype=np.float32)

    # per-feature top cells: convert to CSC and use argpartition per column
    Mcsc = M.tocsc()
    top: dict[int, np.ndarray] = {}
    for f in range(n_features):
        s, e = Mcsc.indptr[f], Mcsc.indptr[f + 1]
        if s == e:
            top[f] = np.array([], dtype=np.int64)
            continue
        idx = Mcsc.indices[s:e]
        v = Mcsc.data[s:e]
        k = min(n_top, len(v))
        top_local = np.argpartition(v, -k)[-k:]
        top[f] = idx[top_local]
    return top


def aggregator_top_genes(
    top_cells_per_feature: dict[int, np.ndarray],
    cell_slide_id: np.ndarray,
    manifest: dict,
) -> pd.DataFrame:
    """For each feature, average normalized gene expression across its top cells.

    Loads each slide once, normalizes (sc.pp.normalize_total + log1p), and
    accumulates per-feature gene-sum + per-slide cell counts. Top genes per
    feature = top mean expression across all slides where the gene was in
    the panel.
    """
    n_features = max(top_cells_per_feature.keys()) + 1 if top_cells_per_feature else 0

    # group: slide_idx -> feature -> [local_cell_idx]
    slide_buckets: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    slide_starts = np.array([s["cell_offset_start"] for s in manifest["slides"]], dtype=np.int64)

    log(f"  grouping {sum(len(v) for v in top_cells_per_feature.values())} (feature, cell) pairs by slide ...")
    for f, cells in top_cells_per_feature.items():
        if len(cells) == 0:
            continue
        # vectorized slide lookup via searchsorted on the offsets
        sids = np.searchsorted(slide_starts, cells, side="right") - 1
        for sid in np.unique(sids):
            mask = sids == sid
            local_idxs = (cells[mask] - slide_starts[sid]).tolist()
            slide_buckets[int(sid)][int(f)].extend(local_idxs)

    # accumulate gene sums + counts
    # use a (n_features, max_n_genes_in_panel) but with variable gene names per slide → use dict
    feat_gene_sum: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    feat_cell_count: dict[int, int] = defaultdict(int)

    for sid in sorted(slide_buckets):
        slide = manifest["slides"][sid]
        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"  [{sid+1}/{len(manifest['slides'])}] loading {slide['tissue']}/{slide['name']}")
        try:
            adata = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"    ERROR loading: {type(e).__name__}: {e}")
            continue

        # normalize as Novae does
        try:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        except Exception as e:
            log(f"    WARN normalize: {type(e).__name__}: {e}")

        var_names = [str(g).lower() for g in adata.var_names.tolist()]
        X = adata.X
        is_sparse = sparse.issparse(X)

        # accumulate per feature in this slide
        for f, local_cells in slide_buckets[sid].items():
            local_arr = np.asarray(local_cells, dtype=np.int64)
            sub = X[local_arr]
            if is_sparse:
                gene_sums = np.asarray(sub.sum(axis=0)).ravel()
            else:
                gene_sums = sub.sum(axis=0)
            feat_cell_count[f] += len(local_cells)
            for gi, g in enumerate(var_names):
                if gene_sums[gi] > 0:
                    feat_gene_sum[f][g] += float(gene_sums[gi])

        del adata, X
        gc.collect()

    # build top-N per feature
    rows = []
    for f in range(n_features):
        if not feat_gene_sum[f]:
            continue
        n_cells = feat_cell_count[f]
        if n_cells == 0:
            continue
        # mean expression
        gene_means = {g: s / n_cells for g, s in feat_gene_sum[f].items()}
        ordered = sorted(gene_means.items(), key=lambda x: -x[1])[:TOP_GENES_PER_FEATURE]
        for rank, (g, m) in enumerate(ordered):
            rows.append({
                "surface": "aggregator",
                "feature_idx": int(f),
                "rank": int(rank),
                "gene": g,
                "mean_expr_in_top_cells": float(m),
                "n_top_cells": int(n_cells),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------- cell_embedder direct path

def cell_embedder_top_genes(sae: TopKSAE, model: novae.Novae) -> pd.DataFrame:
    """Direct decoder projection: the cell_embedder SAE decoder columns live in
    the same 512-d space as the (linear+normalized) gene embeddings."""
    sae.to(DEVICE)
    with torch.no_grad():
        # apply the same linear + L2 norm path as cell_embedder.forward
        full_gene_emb = model.cell_embedder.embedding.weight.to(DEVICE)        # (60697, 512)
        full_gene_emb = model.cell_embedder.linear.to(DEVICE)(full_gene_emb)  # (60697, 512)
        full_gene_emb = Fnn.normalize(full_gene_emb, dim=0, p=2)
        decoder = sae.decoder.weight.to(DEVICE)                                # (512, F)
        scores = (full_gene_emb @ decoder).cpu().numpy()                       # (60697, F)
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()

    gene_names = model.cell_embedder.gene_names                                # list of 60697 strs
    n_features = sae.n_features
    rows = []
    for f in range(n_features):
        col = scores[:, f]
        top_idx = np.argpartition(col, -TOP_GENES_PER_FEATURE)[-TOP_GENES_PER_FEATURE:]
        # sort within the top-K by descending score
        top_idx = top_idx[np.argsort(-col[top_idx])]
        for rank, gi in enumerate(top_idx):
            rows.append({
                "surface": "cell_embedder",
                "feature_idx": int(f),
                "rank": int(rank),
                "gene": gene_names[gi],
                "score": float(col[gi]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- enrichment

def load_libraries() -> dict[str, dict[str, list[str]]]:
    """Pre-download and cache all enrichr libraries as {lib_name: {term: [genes]}}.

    gseapy caches downloads in ~/.cache/gseapy. Subsequent runs are instant.
    """
    libs: dict[str, dict[str, list[str]]] = {}
    for lib in ENRICHR_LIBRARIES:
        log(f"  loading library: {lib}")
        t0 = time.time()
        try:
            d = gp.get_library(lib, organism="Human")
            libs[lib] = {term: [g.upper() for g in genes] for term, genes in d.items()}
            log(f"    {len(libs[lib])} terms, {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"    FAIL {type(e).__name__}: {e}")
    return libs


def enrichment_for_feature(
    top_genes: list[str], libraries: dict[str, dict[str, list[str]]], background: list[str]
) -> list[dict]:
    """Run gp.enrich() against each pre-loaded library, return top hit per library."""
    rows = []
    if not top_genes:
        return rows
    for lib_name, gene_sets_dict in libraries.items():
        try:
            result = gp.enrich(
                gene_list=top_genes,
                gene_sets=gene_sets_dict,
                background=background,
                outdir=None,
                cutoff=1.0,
                no_plot=True,
                verbose=False,
            )
        except Exception as e:
            rows.append({"library": lib_name, "top_term": f"ERROR: {type(e).__name__}",
                         "p": float("nan"), "fdr": float("nan"), "overlap": ""})
            continue
        df = result.results if hasattr(result, "results") else result.res2d
        if df is None or len(df) == 0:
            continue
        df_sorted = df.sort_values("Adjusted P-value")
        top = df_sorted.iloc[0]
        rows.append({
            "library": lib_name,
            "top_term": str(top.get("Term", "")),
            "p": float(top.get("P-value", float("nan"))),
            "fdr": float(top.get("Adjusted P-value", float("nan"))),
            "overlap": str(top.get("Overlap", "")),
        })
    return rows


def run_enrichment(
    top_genes_df: pd.DataFrame, surface: str,
    libraries: dict[str, dict[str, list[str]]], background: list[str],
) -> pd.DataFrame:
    """Run enrichment for every feature in top_genes_df."""
    rows = []
    grouped = top_genes_df.groupby("feature_idx")
    n_features = len(grouped)
    log(f"  enrichment for {n_features} features × {len(libraries)} libraries ...")
    t0 = time.time()
    for i, (f, sub) in enumerate(grouped):
        gene_list = sub.sort_values("rank")["gene"].astype(str).tolist()
        gene_list_up = [g.upper() for g in gene_list]
        feature_rows = enrichment_for_feature(gene_list_up, libraries, background)
        for r in feature_rows:
            r.update({"surface": surface, "feature_idx": int(f)})
            rows.append(r)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (n_features - i - 1) / rate / 60
            log(f"    {i+1}/{n_features}  ({elapsed/60:.1f} min done, {rate:.1f} feat/s, ETA {eta_min:.1f} min)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- assign_domains

def per_slide_assign_domains(model: novae.Novae, manifest: dict, agg: np.ndarray) -> None:
    """For each slide, restore cached novae_latent and run assign_domains.

    Saves per-slide (n_cells,) domain label arrays under
    `activations/novae-human-0/per_slide/<slide_dir>/domains_level7.npy`.
    Skips slides where the domain file already exists.
    """
    log("  per-slide assign_domains ...")
    for sid, slide in enumerate(manifest["slides"]):
        sdir = ACT_DIR / "per_slide" / f"{slide['tissue']}__{slide['name']}"
        out_path = sdir / "domains_level7.npy"
        if out_path.exists():
            log(f"    [{sid+1}/{len(manifest['slides'])}] {slide['name']} SKIP (already done)")
            continue
        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        try:
            adata = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"    [{sid+1}] {slide['name']} ERROR load: {type(e).__name__}: {e}")
            continue

        try:
            novae.spatial_neighbors(adata)
        except Exception as e:
            log(f"    [{sid+1}] {slide['name']} ERROR neighbors: {type(e).__name__}: {e}")
            del adata
            continue

        # restore cached latent
        start, end = slide["cell_offset_start"], slide["cell_offset_end"]
        cached = agg[start:end]
        if cached.shape[0] != adata.n_obs:
            log(f"    [{sid+1}] cached shape {cached.shape} != adata.n_obs {adata.n_obs} — fallback to re-encode")
            try:
                model.compute_representations(adata, zero_shot=True)
            except Exception as e:
                log(f"    re-encode failed: {type(e).__name__}: {e}")
                del adata
                continue
        else:
            adata.obsm["novae_latent"] = cached.copy()

        try:
            model.assign_domains(adata, level=7)
            domains = adata.obs["novae_domains_7"].astype(str).values
            np.save(out_path, domains)
            n_unique = len(set(domains))
            log(f"    [{sid+1}/{len(manifest['slides'])}] {slide['name']} → {n_unique} unique domains")
        except Exception as e:
            log(f"    [{sid+1}] assign_domains ERROR: {type(e).__name__}: {e}")
        finally:
            del adata
            gc.collect()


# ---------------------------------------------------------------------- main

def main() -> None:
    log("=" * 72)
    log("Phase 3c: bio-grounded characterization")
    log(f"  device : {DEVICE}")
    log(f"  out    : {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load shared resources
    log("loading model + manifest + activations")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")
    log(f"  agg: {agg.shape}, slides: {len(manifest['slides'])}")

    # ---- 1) per-slide assign_domains
    per_slide_assign_domains(model, manifest, agg)

    # ---- 2) aggregator top genes
    log("\n--- aggregator SAE: top-cell expression averaging ---")
    sae_agg = load_sae("aggregator", 32, 16, 64)
    n_top_cells = max(MIN_TOP_CELLS, int(TOP_CELL_FRAC_AGG * agg.shape[0]))
    log(f"  top-cells per feature: {n_top_cells}")
    top_cells = encode_aggregator_to_sparse_topcells(sae_agg, agg, n_top_cells)
    n_with_top = sum(1 for v in top_cells.values() if len(v) > 0)
    log(f"  features with ≥1 top cell: {n_with_top}/{sae_agg.n_features}")

    df_agg = aggregator_top_genes(top_cells, cell_slide_id, manifest)
    df_agg.to_parquet(OUT_DIR / "aggregator_top_genes.parquet", index=False)
    log(f"  wrote {OUT_DIR / 'aggregator_top_genes.parquet'} ({len(df_agg)} rows)")
    del sae_agg, top_cells

    # ---- 3) cell_embedder top genes (direct projection)
    log("\n--- cell_embedder SAE: direct decoder projection ---")
    sae_emb = load_sae("cell_embedder", 16, 32, 512)
    df_emb = cell_embedder_top_genes(sae_emb, model)
    df_emb.to_parquet(OUT_DIR / "cell_embedder_top_genes.parquet", index=False)
    log(f"  wrote {OUT_DIR / 'cell_embedder_top_genes.parquet'} ({len(df_emb)} rows)")
    del sae_emb

    # ---- 4) enrichment
    background = [g.upper() for g in model.cell_embedder.gene_names]
    log(f"\n--- enrichment (background size: {len(background):,}) ---")

    log("loading enrichr gene-set libraries ...")
    libraries = load_libraries()

    log("aggregator enrichment ...")
    enr_agg = run_enrichment(df_agg, "aggregator", libraries, background)
    enr_agg.to_parquet(OUT_DIR / "aggregator_enrichment.parquet", index=False)
    log(f"  wrote {OUT_DIR / 'aggregator_enrichment.parquet'} ({len(enr_agg)} rows)")

    log("cell_embedder enrichment ...")
    enr_emb = run_enrichment(df_emb, "cell_embedder", libraries, background)
    enr_emb.to_parquet(OUT_DIR / "cell_embedder_enrichment.parquet", index=False)
    log(f"  wrote {OUT_DIR / 'cell_embedder_enrichment.parquet'} ({len(enr_emb)} rows)")

    # ---- 5) summary
    summary = {
        "agg_n_features_top_cells": n_with_top,
        "agg_top_genes_rows": len(df_agg),
        "embedder_top_genes_rows": len(df_emb),
        "agg_enrichment_rows": len(enr_agg),
        "embedder_enrichment_rows": len(enr_emb),
        "agg_n_features_with_significant_enrichment": int(
            (enr_agg.groupby("feature_idx")["fdr"].min() < 0.05).sum()
        ) if len(enr_agg) else 0,
        "embedder_n_features_with_significant_enrichment": int(
            (enr_emb.groupby("feature_idx")["fdr"].min() < 0.05).sum()
        ) if len(enr_emb) else 0,
        "libraries": ENRICHR_LIBRARIES,
    }
    json.dump(summary, open(OUT_DIR / "bio_summary.json", "w"), indent=2)
    log(f"\n{json.dumps(summary, indent=2)}")
    log("DONE")


if __name__ == "__main__":
    main()
