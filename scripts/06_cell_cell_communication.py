#!/usr/bin/env python
"""Phase 5 — niche-level cell-cell communication.

For each Novae niche at level 20, compute ligand-receptor co-expression scores
using the LIANA consensus L-R resource (~4600 pairs). For each (slide, niche,
L-R pair) we compute the mean of min(ligand_expr, receptor_expr) across cells
in that niche, then aggregate across slides and compare to a corpus baseline.

Output: atlas/web/public/data/niche_communication.json
  {
    "level_20": {
      "<niche_code>": {
        "lb": "<auto-derived label>",
        "n_cells_total": <int>,
        "lr": [{"L": "...", "R": "...", "score": <float>, "enrichment": <float>, "n_slides": <int>}, ...]
      }, ...
    }
  }

Plus a niche-summary table for the home/about pages.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import liana as li
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent

DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ACT_DIR = ROOT / "activations" / "novae-human-0"
ATLAS_DIR = ROOT / "atlas" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "web" / "public" / "data"
LOG_PATH = ROOT / "logs" / "06_cell_cell_communication.log"

LEVELS = [7, 12, 20]
TOP_LR_PER_NICHE = 30


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("Phase 5: niche-level cell-cell communication")

    # 1. Load L-R database
    log("loading consensus L-R resource")
    lr_db = li.resource.select_resource("consensus")
    lr_db["ligand"] = lr_db["ligand"].astype(str).str.upper()
    lr_db["receptor"] = lr_db["receptor"].astype(str).str.upper()
    # Some receptors are complexes joined with '_'; we'll skip those for the simple
    # min-of-pair scoring (proper complex handling needs the per-subunit min)
    lr_db = lr_db[~lr_db["receptor"].str.contains("_")].drop_duplicates().reset_index(drop=True)
    log(f"  {len(lr_db)} L-R pairs (after dropping multi-subunit receptors)")
    log(f"  {lr_db['ligand'].nunique()} ligands, {lr_db['receptor'].nunique()} receptors")

    # 2. Load manifest, niche labels (already derived in 05)
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    niche_index = json.load(open(OUT_DIR / "niche_index.json"))

    payload: dict[str, dict] = {}

    for LEVEL in LEVELS:
        log(f"\n{'=' * 60}")
        log(f"LEVEL {LEVEL}")
        log(f"{'=' * 60}")

        level_key = f"level_{LEVEL}"
        niche_labels = {k: v["lb"] for k, v in niche_index[level_key].items()}
        log(f"  {len(niche_labels)} known niches at level {LEVEL}")

        # 3. Aggregate per (niche, L-R pair) across slides
        # accumulator: niche -> {(L, R) -> [(slide_score, n_cells_in_niche_in_slide), ...]}
        accumulator: dict[str, dict[tuple[str, str], list[tuple[float, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Also accumulate corpus baseline per slide × pair: mean min(L,R) over all cells in the slide
        corpus_per_slide: dict[int, dict[tuple[str, str], float]] = {}
        n_cells_per_niche: dict[str, int] = defaultdict(int)

        for slide in manifest["slides"]:
            sid = slide["slide_idx"]
            h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
            log(f"\n[{sid+1}/{len(manifest['slides'])}] {slide['tissue']}/{slide['name']}")
            try:
                a = ad.read_h5ad(h5ad)
            except Exception as e:
                log(f"  ERROR load: {type(e).__name__}: {e}")
                continue

            # Normalize
            try:
                sc.pp.normalize_total(a)
                sc.pp.log1p(a)
            except Exception as e:
                log(f"  WARN normalize: {type(e).__name__}: {e}")

            var_names_upper = [str(g).upper() for g in a.var_names.tolist()]
            gene_to_idx = {g: i for i, g in enumerate(var_names_upper)}
            measurable = lr_db[
                lr_db["ligand"].isin(gene_to_idx) & lr_db["receptor"].isin(gene_to_idx)
            ].reset_index(drop=True)
            if len(measurable) == 0:
                log(f"  no L-R pairs measurable on this slide (panel mismatch)")
                del a
                continue
            log(f"  {len(measurable)} L-R pairs measurable on this slide ({slide['technology']})")

            # Load per-cell niche labels
            sdir = ACT_DIR / "per_slide" / f"{slide['tissue']}__{slide['name']}"
            try:
                niches = np.load(sdir / f"domains_level{LEVEL}.npy", allow_pickle=True).astype(str)
            except Exception as e:
                log(f"  WARN no domain labels: {e}")
                del a
                continue

            if len(niches) != a.n_obs:
                log(f"  WARN niche count {len(niches)} != n_obs {a.n_obs}")
                del a
                continue

            # Get expression for L-R genes only (memory efficient)
            all_lr_genes = sorted(set(measurable["ligand"].tolist() + measurable["receptor"].tolist()))
            gene_idx = [gene_to_idx[g] for g in all_lr_genes]
            X = a.X[:, gene_idx]
            if sparse.issparse(X):
                X = np.asarray(X.todense())
            else:
                X = np.asarray(X)
            gene_pos = {g: i for i, g in enumerate(all_lr_genes)}

            # Corpus baseline (mean min(L,R) over all cells in slide)
            corpus_scores = {}
            for r in measurable.itertuples():
                l_pos = gene_pos[r.ligand]
                r_pos = gene_pos[r.receptor]
                mn = np.minimum(X[:, l_pos], X[:, r_pos])
                cs = float(np.nanmean(mn))
                if np.isnan(cs) or np.isinf(cs):
                    continue  # skip pairs with invalid expression values
                corpus_scores[(r.ligand, r.receptor)] = cs
            corpus_per_slide[sid] = corpus_scores

            # Per-niche scores
            unique_niches, inverse = np.unique(niches, return_inverse=True)
            for ni, niche in enumerate(unique_niches):
                mask = inverse == ni
                n_in_niche = int(mask.sum())
                if n_in_niche < 20:
                    continue  # too few cells for stable estimate
                n_cells_per_niche[str(niche)] += n_in_niche
                X_niche = X[mask]
                for r in measurable.itertuples():
                    l_pos = gene_pos[r.ligand]
                    r_pos = gene_pos[r.receptor]
                    mn = np.minimum(X_niche[:, l_pos], X_niche[:, r_pos])
                    niche_score = float(np.nanmean(mn))
                    if np.isnan(niche_score) or np.isinf(niche_score):
                        continue  # skip pairs with invalid expression values (e.g. log1p of negatives)
                    accumulator[str(niche)][(r.ligand, r.receptor)].append((niche_score, n_in_niche))

            del X, a
            gc.collect()

        log(f"\naggregating across slides for level {LEVEL}")

        # 4. Build niche → top L-R pairs
        niche_lr_results: dict[str, list[dict]] = {}
        n_lr_total = 0
        for niche, lr_scores in accumulator.items():
            rows = []
            for (L, R), entries in lr_scores.items():
                tot_w = sum(n for _, n in entries)
                if tot_w == 0:
                    continue
                weighted_score = sum(s * n for s, n in entries) / tot_w
                if np.isnan(weighted_score) or np.isinf(weighted_score):
                    continue
                slide_ids_for_pair = []
                for sid, csc in corpus_per_slide.items():
                    if (L, R) in csc:
                        slide_ids_for_pair.append(sid)
                if not slide_ids_for_pair:
                    continue
                baseline_vals = [corpus_per_slide[sid][(L, R)] for sid in slide_ids_for_pair]
                corpus_baseline = float(np.nanmean(baseline_vals))
                if np.isnan(corpus_baseline) or np.isinf(corpus_baseline):
                    continue
                enrichment = (weighted_score + 1e-9) / (corpus_baseline + 1e-9)
                if np.isnan(enrichment) or np.isinf(enrichment):
                    continue
                rows.append({
                    "L": L,
                    "R": R,
                    "score": round(weighted_score, 4),
                    "enrichment": round(enrichment, 3),
                    "n_slides": len(entries),
                })
            rows.sort(key=lambda r: (-r["enrichment"], -r["score"]))
            niche_lr_results[niche] = rows[:TOP_LR_PER_NICHE]
            n_lr_total += len(rows[:TOP_LR_PER_NICHE])

        log(f"  level {LEVEL}: {len(niche_lr_results)} niches with at least one L-R signal, {n_lr_total} (niche, L-R) entries")

        payload[level_key] = {}
        for niche, lr_rows in niche_lr_results.items():
            payload[level_key][niche] = {
                "lb": niche_labels.get(niche, "Unannotated"),
                "n_cells_total": int(n_cells_per_niche.get(niche, 0)),
                "n_lr_pairs": len(lr_rows),
                "lr": lr_rows,
            }

    # 5. Bake JSON — allow_nan=False catches any NaN/inf that slipped past
    # the filters above (NaN literals are not valid JSON and break JS parsing)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "niche_communication.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    log(f"\n{'=' * 60}")
    log(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    for lvl_key in payload:
        log(f"  {lvl_key}: {len(payload[lvl_key])} niches")

    # 6. Bake the gene → CCC participation index, used by the Gene Search page
    # to show "L-R interactions in Novae niches" for any clicked ligand or receptor.
    log(f"\nbuilding gene_ccc_index.json")
    gene_ccc: dict[str, list[dict]] = defaultdict(list)
    for level_key, niches in payload.items():
        level_num = int(level_key.replace("level_", ""))
        for niche_code, niche_data in niches.items():
            niche_lb = niche_data.get("lb", "")
            for lr in niche_data.get("lr", []):
                L = lr["L"]
                R = lr["R"]
                # Each entry records the gene's role, its partner, the niche, and the score.
                gene_ccc[L].append({
                    "role": "ligand",
                    "partner": R,
                    "niche": niche_code,
                    "niche_lb": niche_lb,
                    "level": level_num,
                    "score": lr["score"],
                    "enrichment": lr["enrichment"],
                })
                gene_ccc[R].append({
                    "role": "receptor",
                    "partner": L,
                    "niche": niche_code,
                    "niche_lb": niche_lb,
                    "level": level_num,
                    "score": lr["score"],
                    "enrichment": lr["enrichment"],
                })

    # Sort each gene's interactions by enrichment desc
    for g in gene_ccc:
        gene_ccc[g].sort(key=lambda r: -r["enrichment"])
        # Cap at 50 to keep payload small
        gene_ccc[g] = gene_ccc[g][:50]

    out_ccc = OUT_DIR / "gene_ccc_index.json"
    with open(out_ccc, "w") as f:
        json.dump(dict(gene_ccc), f, separators=(",", ":"), allow_nan=False)
    log(f"  wrote {out_ccc} ({out_ccc.stat().st_size / 1024:.1f} KB)")
    log(f"  {len(gene_ccc)} unique L/R genes indexed")

    log("\nDONE")


if __name__ == "__main__":
    main()
