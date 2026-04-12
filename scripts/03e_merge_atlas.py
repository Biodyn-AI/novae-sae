#!/usr/bin/env python
"""Merge v1 + v2 + v3 enrichment results into a single atlas-ready table.

Inputs:
  atlas/novae-human-0/
    feature_atlas.parquet                       (49152 × 11) — base, all 12 surfaces
    bio/aggregator_enrichment.parquet           (10240 × 7)  — v1 Enrichr (5 libs)
    bio/cell_embedder_enrichment.parquet        (40783 × 7)  — v1 Enrichr (5 libs)
    bio/v2/aggregator_top_genes_v2.parquet      (40950 × 8)  — fold-change ranked
    bio/v2/aggregator_enrichment_v2.parquet     (10240 × 7)  — v2 Enrichr (5 libs)
    bio/cell_embedder_top_genes.parquet         (163840 × 5) — direct projection
    bio/v3/aggregator_domain_enrichment.parquet (6144 × 9)   — domain l7/l12/l20

Output:
  atlas/novae-human-0/feature_atlas_full.parquet

The merged table has one row per (surface, feature_idx), preserving the v1 base
columns and adding wide bio columns. NaN/empty for surfaces where a particular
column is undefined (e.g. domain_l7 only exists for aggregator).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
ATLAS = ROOT / "atlas" / "novae-human-0"
BIO = ATLAS / "bio"
LOG_PATH = ROOT / "logs" / "03e_merge_atlas.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Map Enrichr library long names to short column suffixes
LIB_SHORT = {
    "GO_Biological_Process_2023": "GO_BP",
    "KEGG_2021_Human": "KEGG",
    "Reactome_2022": "Reactome",
    "PanglaoDB_Augmented_2021": "PanglaoDB",
    "CellMarker_Augmented_2021": "CellMarker",
}


def pivot_enrichment(df_enr: pd.DataFrame, surface: str, source_tag: str) -> pd.DataFrame:
    """Wide-format: one row per feature_idx, columns top_<LIB>_<source>+_fdr."""
    sub = df_enr[df_enr["surface"] == surface].copy() if "surface" in df_enr.columns else df_enr.copy()
    sub["lib_short"] = sub["library"].map(LIB_SHORT)
    sub = sub.dropna(subset=["lib_short"])
    rows = {}
    for fid, grp in sub.groupby("feature_idx"):
        row: dict = {"feature_idx": int(fid)}
        for _, r in grp.iterrows():
            short = r["lib_short"]
            row[f"top_{short}_{source_tag}"] = str(r["top_term"])
            row[f"top_{short}_{source_tag}_fdr"] = float(r["fdr"])
        rows[int(fid)] = row
    return pd.DataFrame(list(rows.values()))


def top_genes_string(df_top: pd.DataFrame, key_col: str, n: int = 5) -> pd.DataFrame:
    """Per feature_idx, sort by `key_col` desc and join the first n gene names."""
    out = []
    for fid, grp in df_top.groupby("feature_idx"):
        sg = grp.sort_values(key_col, ascending=False).head(n)
        out.append(
            {
                "feature_idx": int(fid),
                "top_genes": ",".join(sg["gene"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(out)


def pivot_domain(df_dom: pd.DataFrame) -> pd.DataFrame:
    """Wide-format domain enrichment: one row per feature_idx, columns per level."""
    rows = {}
    for fid, grp in df_dom.groupby("feature_idx"):
        row: dict = {"feature_idx": int(fid)}
        for _, r in grp.iterrows():
            lvl = int(r["level"])
            row[f"top_domain_l{lvl}"] = str(r["top_domain"])
            row[f"top_domain_l{lvl}_frac"] = float(r["top_domain_frac"])
            row[f"top_domain_l{lvl}_log2enr"] = float(r["log2_enrichment"])
            row[f"top_domain_l{lvl}_fdr"] = float(r["fisher_fdr"])
        rows[int(fid)] = row
    return pd.DataFrame(list(rows.values()))


def main() -> None:
    log("=" * 72)
    log("Merge atlas: v1 base + v1 enrichment + v2 fold-change + v3 domains")

    base = pd.read_parquet(ATLAS / "feature_atlas.parquet")
    log(f"  base feature_atlas: {base.shape}")

    # --- v2 fold-change-ranked top genes for aggregator ---
    agg_top_v2 = pd.read_parquet(BIO / "v2" / "aggregator_top_genes_v2.parquet")
    agg_top_str = top_genes_string(agg_top_v2, key_col="fold_change", n=5)
    agg_top_str["surface"] = "aggregator"
    log(f"  v2 aggregator top-genes: {len(agg_top_str)} features")

    # --- cell_embedder direct-projection top genes ---
    ce_top = pd.read_parquet(BIO / "cell_embedder_top_genes.parquet")
    ce_top_str = top_genes_string(ce_top, key_col="score", n=5)
    ce_top_str["surface"] = "cell_embedder"
    log(f"  cell_embedder top-genes: {len(ce_top_str)} features")

    top_genes_df = pd.concat([agg_top_str, ce_top_str], ignore_index=True)

    # --- v2 Enrichr enrichment for aggregator (fold-change-ranked input) ---
    agg_enr_v2 = pd.read_parquet(BIO / "v2" / "aggregator_enrichment_v2.parquet")
    agg_enr_v2_wide = pivot_enrichment(agg_enr_v2, surface="aggregator", source_tag="v2")
    agg_enr_v2_wide["surface"] = "aggregator"
    log(f"  v2 aggregator enrichment: {agg_enr_v2_wide.shape}")

    # --- v1 Enrichr enrichment for cell_embedder ---
    ce_enr = pd.read_parquet(BIO / "cell_embedder_enrichment.parquet")
    ce_enr_wide = pivot_enrichment(ce_enr, surface="cell_embedder", source_tag="v2")
    ce_enr_wide["surface"] = "cell_embedder"
    log(f"  cell_embedder enrichment: {ce_enr_wide.shape}")

    enrich_wide = pd.concat([agg_enr_v2_wide, ce_enr_wide], ignore_index=True)

    # --- v3 domain enrichment for aggregator ---
    dom = pd.read_parquet(BIO / "v3" / "aggregator_domain_enrichment.parquet")
    dom_wide = pivot_domain(dom)
    dom_wide["surface"] = "aggregator"
    log(f"  v3 domain enrichment: {dom_wide.shape}")

    # --- merge ---
    merged = base.merge(top_genes_df, on=["surface", "feature_idx"], how="left")
    merged = merged.merge(enrich_wide, on=["surface", "feature_idx"], how="left")
    merged = merged.merge(dom_wide, on=["surface", "feature_idx"], how="left")

    log(f"  merged shape: {merged.shape}")
    log(f"  columns ({len(merged.columns)}): {list(merged.columns)}")

    out_pq = ATLAS / "feature_atlas_full.parquet"
    merged.to_parquet(out_pq, index=False)
    log(f"  wrote {out_pq}")

    # --- summary ---
    summary = {
        "n_rows": int(len(merged)),
        "n_columns": int(len(merged.columns)),
        "by_surface": merged["surface"].value_counts().to_dict(),
        "aggregator_columns_filled": {
            col: int(merged[merged.surface == "aggregator"][col].notna().sum())
            for col in merged.columns
            if col not in {"feature_idx", "surface", "alive"}
        },
    }
    json.dump(summary, open(ATLAS / "feature_atlas_full.summary.json", "w"), indent=2, default=str)
    log(json.dumps(summary, indent=2, default=str))
    log("DONE")


if __name__ == "__main__":
    main()
