#!/usr/bin/env python
"""Phase 3: Novel relationship discovery + gene-level predictions (GAPs 5, 10, 11, 12).

Uses the causal circuit edge table + enrichment data to:
- GAP 5: Compare circuit edges to STRING database reference graph
- GAP 10: Tissue-specific circuit enrichment
- GAP 11: Gene-level circuit predictions (feature edges → gene pairs)
- GAP 12: Disease gene set mapping
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
BIO_DIR = ROOT / "atlas" / "novae-human-0" / "bio"
LOG_PATH = ROOT / "logs" / "21_circuit_biology.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_feature_genes() -> dict[tuple[int, int], list[str]]:
    """Load top genes per (layer, feature) from all available sources."""
    fa = pd.read_parquet(ROOT / "atlas" / "novae-human-0" / "feature_atlas_full.parquet")
    genes = {}

    # From feature_atlas (aggregator + cell_embedder + conv layers with enrichment)
    for _, row in fa.iterrows():
        surface = row.get("surface", "")
        if not surface.startswith("conv_"):
            continue
        layer = int(surface.split("_")[1])
        fid = int(row["feature_idx"])
        tg = row.get("top_genes", "")
        if pd.notna(tg) and tg:
            genes[(layer, fid)] = [g.strip() for g in str(tg).split(",")[:5]]

    # From hub enrichments (has top_genes column)
    hub_path = OUT_DIR / "circuit_hub_enrichments.parquet"
    if hub_path.exists():
        hub = pd.read_parquet(hub_path)
        for _, row in hub.drop_duplicates(["layer", "feature_idx"]).iterrows():
            key = (int(row["layer"]), int(row["feature_idx"]))
            if key not in genes:
                tg = row.get("top_genes", "")
                if pd.notna(tg) and tg:
                    genes[key] = [g.strip().upper() for g in str(tg).split(",")[:5]]

    return genes


def load_feature_tissues() -> dict[tuple[int, int], str]:
    """Load top tissue per (layer, feature)."""
    fa = pd.read_parquet(ROOT / "atlas" / "novae-human-0" / "feature_atlas_full.parquet")
    tissues = {}
    for _, row in fa.iterrows():
        surface = row.get("surface", "")
        if not surface.startswith("conv_"):
            continue
        layer = int(surface.split("_")[1])
        fid = int(row["feature_idx"])
        t = row.get("top_tissue", "")
        if pd.notna(t) and t:
            tissues[(layer, fid)] = str(t)
    return tissues


def main():
    log("=" * 60)
    log("Phase 3: Circuit biology (GAPs 5, 10, 11, 12)")

    edges = pd.read_parquet(OUT_DIR / "causal_circuit_edges.parquet")
    log(f"loaded {len(edges)} circuit edges")

    feat_genes = load_feature_genes()
    feat_tissues = load_feature_tissues()
    log(f"feature genes: {len(feat_genes)} entries")
    log(f"feature tissues: {len(feat_tissues)} entries")

    results = {}

    # === GAP 11: Gene-level circuit predictions ===
    log("\n--- GAP 11: Gene-level circuit predictions ---")
    gene_pairs = []
    for _, e in edges.iterrows():
        src_key = (int(e["source_layer"]), int(e["source_feature"]))
        tgt_key = (int(e["target_layer"]), int(e["target_feature"]))
        src_genes = feat_genes.get(src_key, [])
        tgt_genes = feat_genes.get(tgt_key, [])
        if src_genes and tgt_genes:
            for sg, tg in product(src_genes, tgt_genes):
                sg_upper = sg.upper()
                tg_upper = tg.upper()
                if sg_upper != tg_upper:
                    gene_pairs.append({
                        "source_gene": sg_upper,
                        "target_gene": tg_upper,
                        "source_layer": int(e["source_layer"]),
                        "source_feature": int(e["source_feature"]),
                        "target_layer": int(e["target_layer"]),
                        "target_feature": int(e["target_feature"]),
                        "cohen_d": float(e["cohen_d"]),
                        "sign": int(e["sign"]),
                    })

    gp_df = pd.DataFrame(gene_pairs)
    if len(gp_df) > 0:
        # Deduplicate gene pairs (keep strongest |d|)
        gp_df["abs_d"] = gp_df["cohen_d"].abs()
        gp_dedup = gp_df.sort_values("abs_d", ascending=False).drop_duplicates(
            ["source_gene", "target_gene"], keep="first"
        )
        gp_dedup.to_parquet(OUT_DIR / "gene_level_predictions.parquet", index=False)
        log(f"  raw gene pairs: {len(gp_df)}")
        log(f"  unique gene pairs: {len(gp_dedup)}")
        log(f"  unique genes involved: {len(set(gp_dedup['source_gene']) | set(gp_dedup['target_gene']))}")
        results["gap11_gene_predictions"] = {
            "n_raw_pairs": len(gp_df),
            "n_unique_pairs": len(gp_dedup),
            "n_genes": len(set(gp_dedup["source_gene"]) | set(gp_dedup["target_gene"])),
            "top_pairs": [
                {"src": r["source_gene"], "tgt": r["target_gene"], "d": round(r["cohen_d"], 3)}
                for _, r in gp_dedup.head(20).iterrows()
            ],
        }
    else:
        log("  no gene pairs generated")
        results["gap11_gene_predictions"] = {"n_unique_pairs": 0}

    # === GAP 5: Compare to known biology ===
    log("\n--- GAP 5: Novel vs known relationships ---")
    # Build a set of known gene interactions from GO co-annotation
    # (Lightweight alternative to STRING download)
    # Two genes are "known" if they share a GO term in the enrichment tables
    fa = pd.read_parquet(ROOT / "atlas" / "novae-human-0" / "feature_atlas_full.parquet")

    # Collect GO terms per gene from the feature atlas top_genes
    gene_to_go = defaultdict(set)
    for _, row in fa[fa["surface"] == "aggregator"].iterrows():
        tg = row.get("top_genes", "")
        go = row.get("top_GO_BP_v2", "")
        if pd.notna(tg) and pd.notna(go) and tg and go:
            for g in str(tg).split(",")[:5]:
                gene_to_go[g.strip().upper()].add(str(go).strip().lower())

    if len(gp_dedup) > 0 and len(gene_to_go) > 0:
        n_known = 0
        n_novel = 0
        n_checkable = 0
        for _, r in gp_dedup.iterrows():
            sg = r["source_gene"]
            tg = r["target_gene"]
            sg_go = gene_to_go.get(sg, set())
            tg_go = gene_to_go.get(tg, set())
            if sg_go and tg_go:
                n_checkable += 1
                if sg_go & tg_go:
                    n_known += 1
                else:
                    n_novel += 1

        log(f"  checkable pairs (both have GO terms): {n_checkable}")
        log(f"  known (shared GO term): {n_known} ({n_known/max(n_checkable,1):.1%})")
        log(f"  novel (no shared GO term): {n_novel} ({n_novel/max(n_checkable,1):.1%})")
        results["gap5_novelty"] = {
            "n_checkable": n_checkable,
            "n_known": n_known,
            "n_novel": n_novel,
            "known_fraction": round(n_known / max(n_checkable, 1), 4),
            "novel_fraction": round(n_novel / max(n_checkable, 1), 4),
        }
    else:
        results["gap5_novelty"] = {"n_checkable": 0}

    # === GAP 10: Tissue-specific circuit enrichment ===
    log("\n--- GAP 10: Tissue-specific circuit enrichment ---")
    edge_tissues = []
    for _, e in edges.iterrows():
        src_key = (int(e["source_layer"]), int(e["source_feature"]))
        tgt_key = (int(e["target_layer"]), int(e["target_feature"]))
        src_tissue = feat_tissues.get(src_key)
        tgt_tissue = feat_tissues.get(tgt_key)
        if src_tissue and tgt_tissue:
            edge_tissues.append({
                "src_tissue": src_tissue,
                "tgt_tissue": tgt_tissue,
                "same_tissue": src_tissue == tgt_tissue,
            })

    if edge_tissues:
        et_df = pd.DataFrame(edge_tissues)
        same_frac = et_df["same_tissue"].mean()
        tissue_pair_counts = Counter(
            (r["src_tissue"], r["tgt_tissue"]) for _, r in et_df.iterrows()
        )
        log(f"  edges with tissue annotation: {len(et_df)}")
        log(f"  same-tissue edges: {et_df['same_tissue'].sum()} ({same_frac:.1%})")
        log(f"  top tissue pairs:")
        for (st, tt), count in tissue_pair_counts.most_common(10):
            log(f"    {st} → {tt}: {count}")

        results["gap10_tissue_circuits"] = {
            "n_annotated_edges": len(et_df),
            "same_tissue_fraction": round(float(same_frac), 4),
            "top_tissue_pairs": [
                {"source": st, "target": tt, "count": int(c)}
                for (st, tt), c in tissue_pair_counts.most_common(10)
            ],
        }
    else:
        results["gap10_tissue_circuits"] = {"n_annotated_edges": 0}

    # === GAP 12: Disease gene mapping ===
    log("\n--- GAP 12: Disease-associated circuit features ---")
    # Use KEGG disease pathways and cancer-related terms as proxy
    disease_keywords = [
        "cancer", "tumor", "carcinoma", "leukemia", "lymphoma",
        "inflammation", "autoimmune", "infection", "disease",
        "alzheimer", "diabetes", "fibrosis",
    ]

    hub_path = OUT_DIR / "circuit_hub_enrichments.parquet"
    disease_features = []
    if hub_path.exists():
        hub = pd.read_parquet(hub_path)
        for _, r in hub.iterrows():
            term = str(r.get("term", "")).lower()
            for kw in disease_keywords:
                if kw in term:
                    disease_features.append({
                        "layer": int(r["layer"]),
                        "feature": int(r["feature_idx"]),
                        "term": str(r["term"]),
                        "keyword": kw,
                        "fdr": float(r["fdr"]),
                    })
                    break

    if disease_features:
        dis_df = pd.DataFrame(disease_features)
        log(f"  disease-associated enrichments: {len(dis_df)}")
        n_disease_feats = dis_df.groupby(["layer", "feature"]).size().reset_index()
        log(f"  unique disease-associated features: {len(n_disease_feats)}")
        for _, r in dis_df.drop_duplicates(["layer", "feature"]).iterrows():
            log(f"    conv_{r['layer']}/F{r['feature']}: {r['term'][:60]}")
        results["gap12_disease"] = {
            "n_disease_enrichments": len(dis_df),
            "n_disease_features": len(n_disease_feats),
            "features": [
                {"layer": int(r["layer"]), "feature": int(r["feature"]), "term": r["term"]}
                for _, r in dis_df.drop_duplicates(["layer", "feature"]).iterrows()
            ],
        }
    else:
        results["gap12_disease"] = {"n_disease_features": 0}

    # Save
    json.dump(results, open(OUT_DIR / "circuit_biology.json", "w"), indent=2)
    log(f"\nwrote circuit_biology.json")
    log("DONE")


if __name__ == "__main__":
    main()
