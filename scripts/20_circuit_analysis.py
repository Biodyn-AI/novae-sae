#!/usr/bin/env python
"""Phase 2: Circuit graph analysis (GAPs 2, 3, 6, 7, 8).

Post-processes the causal_circuit_edges.parquet from script 19:
- GAP 2: Inhibitory/excitatory balance
- GAP 3: Biological coherence of edges (shared enrichments)
- GAP 6: Process hierarchy from layer position
- GAP 7: Effect attenuation curves
- GAP 8: Hub feature identification
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "20_circuit_analysis.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("Phase 2: Circuit graph analysis")

    edges = pd.read_parquet(OUT_DIR / "causal_circuit_edges.parquet")
    log(f"loaded {len(edges)} edges")

    # Load feature enrichments for biological coherence
    fa = pd.read_parquet(ROOT / "atlas" / "novae-human-0" / "feature_atlas_full.parquet")

    results = {}

    # === GAP 2: Inhibitory/excitatory balance ===
    log("\n--- GAP 2: Inhibitory/excitatory balance ---")
    n_exc = int((edges["sign"] > 0).sum())
    n_inh = int((edges["sign"] < 0).sum())
    inh_frac = n_inh / len(edges) if len(edges) > 0 else 0
    log(f"  excitatory: {n_exc} ({n_exc/len(edges):.1%})")
    log(f"  inhibitory: {n_inh} ({inh_frac:.1%})")
    log(f"  balance: {'balanced' if 0.4 < inh_frac < 0.6 else 'skewed'}")
    results["gap2_balance"] = {
        "n_excitatory": n_exc,
        "n_inhibitory": n_inh,
        "inhibitory_fraction": round(inh_frac, 4),
    }

    # === GAP 3: Biological coherence ===
    log("\n--- GAP 3: Biological coherence ---")
    # Build feature → enrichment term sets (for conv layers, use surface-specific rows)
    def get_terms(surface: str, feat_idx: int) -> set:
        row = fa[(fa["surface"] == surface) & (fa["feature_idx"] == feat_idx)]
        if row.empty:
            return set()
        r = row.iloc[0]
        terms = set()
        for col in ["top_GO_BP_v2", "top_KEGG_v2", "top_Reactome_v2",
                     "top_PanglaoDB_v2", "top_CellMarker_v2"]:
            val = r.get(col)
            if pd.notna(val) and val:
                terms.add(str(val).strip().lower())
        return terms

    n_coherent = 0
    n_testable = 0
    for _, edge in edges.iterrows():
        src_surface = f"conv_{int(edge['source_layer'])}"
        tgt_surface = f"conv_{int(edge['target_layer'])}"
        src_terms = get_terms(src_surface, int(edge["source_feature"]))
        tgt_terms = get_terms(tgt_surface, int(edge["target_feature"]))
        if src_terms and tgt_terms:
            n_testable += 1
            if src_terms & tgt_terms:
                n_coherent += 1

    coherence_rate = n_coherent / n_testable if n_testable > 0 else 0
    log(f"  testable edges (both annotated): {n_testable}")
    log(f"  biologically coherent (shared term): {n_coherent} ({coherence_rate:.1%})")
    results["gap3_coherence"] = {
        "n_testable": n_testable,
        "n_coherent": n_coherent,
        "coherence_rate": round(coherence_rate, 4),
    }

    # === GAP 6: Process hierarchy from layer position ===
    log("\n--- GAP 6: Process hierarchy from layer position ---")
    # For each feature in edges, get its top enrichment term + its layer
    term_layers = defaultdict(list)
    for _, edge in edges.iterrows():
        for role, layer_col, feat_col in [
            ("source", "source_layer", "source_feature"),
            ("target", "target_layer", "target_feature"),
        ]:
            surface = f"conv_{int(edge[layer_col])}"
            terms = get_terms(surface, int(edge[feat_col]))
            for t in terms:
                term_layers[t].append(int(edge[layer_col]))

    # Mean layer per term
    term_mean_layer = {t: np.mean(layers) for t, layers in term_layers.items() if len(layers) >= 3}
    sorted_terms = sorted(term_mean_layer.items(), key=lambda x: x[1])

    log(f"  terms with >= 3 occurrences: {len(sorted_terms)}")
    log(f"  earliest (layer ~0): {sorted_terms[:5] if sorted_terms else 'none'}")
    log(f"  latest   (layer ~9): {sorted_terms[-5:] if sorted_terms else 'none'}")

    results["gap6_hierarchy"] = {
        "n_terms": len(sorted_terms),
        "early_terms": [{"term": t, "mean_layer": round(l, 2)} for t, l in sorted_terms[:10]],
        "late_terms": [{"term": t, "mean_layer": round(l, 2)} for t, l in sorted_terms[-10:]],
    }

    # === GAP 7: Effect attenuation curves ===
    log("\n--- GAP 7: Effect attenuation curves ---")
    edges["layer_gap"] = edges["target_layer"] - edges["source_layer"]
    gap_counts = edges.groupby("layer_gap").size().to_dict()
    gap_mean_d = edges.groupby("layer_gap")["cohen_d"].apply(lambda x: x.abs().mean()).to_dict()

    log(f"  edge count by layer gap:")
    for gap in sorted(gap_counts.keys()):
        log(f"    gap {gap}: {gap_counts[gap]} edges, mean |d| = {gap_mean_d.get(gap, 0):.3f}")

    results["gap7_attenuation"] = {
        "edge_count_by_gap": {str(k): int(v) for k, v in gap_counts.items()},
        "mean_abs_d_by_gap": {str(k): round(float(v), 4) for k, v in gap_mean_d.items()},
    }

    # === GAP 8: Hub feature identification ===
    log("\n--- GAP 8: Hub feature identification ---")
    # Out-degree: number of downstream targets per source feature
    out_degree = edges.groupby(["source_layer", "source_feature"]).size().reset_index(name="out_degree")
    out_degree = out_degree.sort_values("out_degree", ascending=False)

    # In-degree: number of upstream sources per target feature
    in_degree = edges.groupby(["target_layer", "target_feature"]).size().reset_index(name="in_degree")
    in_degree = in_degree.sort_values("in_degree", ascending=False)

    log(f"  top-10 source hubs (by out-degree):")
    for _, row in out_degree.head(10).iterrows():
        surface = f"conv_{int(row['source_layer'])}"
        feat = int(row["source_feature"])
        sub = fa[(fa["surface"] == surface) & (fa["feature_idx"] == feat)]
        label = sub.iloc[0].get("top_PanglaoDB_v2", "?") if len(sub) > 0 else "?"
        log(f"    conv_{int(row['source_layer'])}/F{feat}: out={int(row['out_degree'])} | {label}")

    log(f"  top-10 target hubs (by in-degree):")
    for _, row in in_degree.head(10).iterrows():
        surface = f"conv_{int(row['target_layer'])}"
        feat = int(row["target_feature"])
        sub = fa[(fa["surface"] == surface) & (fa["feature_idx"] == feat)]
        label = sub.iloc[0].get("top_PanglaoDB_v2", "?") if len(sub) > 0 else "?"
        log(f"    conv_{int(row['target_layer'])}/F{feat}: in={int(row['in_degree'])} | {label}")

    # Save hub tables
    out_degree.to_parquet(OUT_DIR / "circuit_hubs_source.parquet", index=False)
    in_degree.to_parquet(OUT_DIR / "circuit_hubs_target.parquet", index=False)

    results["gap8_hubs"] = {
        "top_source_hubs": [
            {"layer": int(r["source_layer"]), "feature": int(r["source_feature"]), "out_degree": int(r["out_degree"])}
            for _, r in out_degree.head(10).iterrows()
        ],
        "top_target_hubs": [
            {"layer": int(r["target_layer"]), "feature": int(r["target_feature"]), "in_degree": int(r["in_degree"])}
            for _, r in in_degree.head(10).iterrows()
        ],
    }

    # GAP 14 bonus: hub annotation bias
    log("\n--- GAP 14: Hub annotation bias ---")
    top_hubs = out_degree.head(20)
    n_annotated = 0
    for _, row in top_hubs.iterrows():
        surface = f"conv_{int(row['source_layer'])}"
        feat = int(row["source_feature"])
        sub = fa[(fa["surface"] == surface) & (fa["feature_idx"] == feat)]
        if len(sub) > 0:
            panglao = sub.iloc[0].get("top_PanglaoDB_v2")
            if pd.notna(panglao) and panglao:
                n_annotated += 1
    log(f"  top-20 hubs annotated: {n_annotated}/20 ({n_annotated/20:.0%})")
    results["gap14_hub_bias"] = {
        "top_20_annotated": n_annotated,
        "fraction_annotated": round(n_annotated / 20, 2),
    }

    # Save results
    json.dump(results, open(OUT_DIR / "circuit_analysis.json", "w"), indent=2)
    log(f"\nwrote circuit_analysis.json")
    log(json.dumps(results, indent=2, default=str))
    log("DONE")


if __name__ == "__main__":
    main()
