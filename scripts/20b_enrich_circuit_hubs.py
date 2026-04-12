#!/usr/bin/env python
"""Enrich ONLY the circuit hub features at unenriched conv layers.

Only 44 features at conv_4/5/7/8 need enrichment (they appear in the
causal circuit edge table but lack biological annotations). This runs
in ~10 minutes vs ~20 hours for full-layer enrichment.
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
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "bio"
CAUSAL_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "20b_enrich_circuit_hubs.log"

TOP_CELL_FRAC = 0.001
MIN_TOP = 50
TOP_GENES = 20
LIBS = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022",
        "PanglaoDB_Augmented_2021", "CellMarker_Augmented_2021"]

SAE_SPECS = {
    **{f"conv_{i}": (128, 4096, 32) for i in range(9)},
    "conv_9": (64, 2048, 16),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    log("=" * 60)
    log("Enriching circuit hub features at unenriched layers")

    # Load circuit edges to find which features need enrichment
    edges = pd.read_parquet(CAUSAL_DIR / "causal_circuit_edges.parquet")
    circuit_feats = set()
    for _, r in edges.iterrows():
        circuit_feats.add((int(r["source_layer"]), int(r["source_feature"])))
        circuit_feats.add((int(r["target_layer"]), int(r["target_feature"])))

    # Already enriched layers
    enriched_layers = set()
    for l in range(10):
        if (OUT_DIR / f"conv_{l}_enrichment.parquet").exists():
            enriched_layers.add(l)
    log(f"already enriched layers: {sorted(enriched_layers)}")

    # Features needing enrichment
    need_enrichment = [(l, f) for l, f in circuit_feats if l not in enriched_layers]
    by_layer = defaultdict(list)
    for l, f in need_enrichment:
        by_layer[l].append(f)
    log(f"features needing enrichment: {len(need_enrichment)} across layers {sorted(by_layer.keys())}")
    for l in sorted(by_layer.keys()):
        log(f"  conv_{l}: {len(by_layer[l])} features")

    if not need_enrichment:
        log("nothing to enrich")
        return

    # Load manifest
    manifest = json.load(open(ACT_DIR / "manifest.json"))

    # For each unenriched layer: compute top genes for circuit features only
    all_enrichments = []

    for layer_idx in sorted(by_layer.keys()):
        features_to_enrich = by_layer[layer_idx]
        layer_name = f"conv_{layer_idx}"
        d_in, n_feat, k = SAE_SPECS[layer_name]

        log(f"\n=== {layer_name}: {len(features_to_enrich)} features ===")

        # Load SAE
        sae = TopKSAE(d_in=d_in, n_features=n_feat, k=k)
        sae.load_state_dict(torch.load(SAE_DIR / f"{layer_name}.pt", map_location="cpu"))
        sae.eval()

        # Accumulate top genes per feature across slides
        top_genes = defaultdict(lambda: defaultdict(float))
        top_counts = defaultdict(int)

        for slide in manifest["slides"]:
            tissue, name = slide["tissue"], slide["name"]
            act_path = ACT_DIR / "per_slide" / f"{tissue}__{name}" / f"{layer_name}_percell.npy"
            h5ad_path = DATA_DIR / tissue / f"{name}.h5ad"
            if not act_path.exists():
                continue

            act = np.load(act_path).astype(np.float32, copy=False)
            a = ad.read_h5ad(h5ad_path)
            sc.pp.normalize_total(a)
            sc.pp.log1p(a)
            genes = list(a.var_names)
            n_cells = act.shape[0]
            n_top = max(MIN_TOP, int(TOP_CELL_FRAC * n_cells))

            # SAE-encode to find top cells per feature
            with torch.no_grad():
                # Build sparse for only the features we need
                for fid in features_to_enrich:
                    # Find top cells for this feature
                    feat_vals = np.zeros(n_cells, dtype=np.float32)
                    for s in range(0, n_cells, 32768):
                        e = min(s + 32768, n_cells)
                        z, _ = sae.encode(torch.tensor(act[s:e]))
                        feat_vals[s:e] = np.abs(z[:, fid].numpy())

                    nonzero = np.flatnonzero(feat_vals)
                    if len(nonzero) < MIN_TOP:
                        continue
                    kk = min(n_top, len(nonzero))
                    top_local = nonzero[np.argpartition(feat_vals[nonzero], -kk)[-kk:]]

                    # Get gene expression for top cells
                    if sparse.issparse(a.X):
                        expr = np.asarray(a.X[top_local].mean(axis=0)).flatten()
                    else:
                        expr = a.X[top_local].mean(axis=0).flatten()
                    for gi, val in enumerate(expr):
                        if val > 0 and gi < len(genes):
                            top_genes[fid][genes[gi]] += float(val)
                    top_counts[fid] += 1

            del a, act
            gc.collect()

        # Run enrichment
        log(f"  enriching {len(top_genes)} features")
        t0 = time.time()
        for fid in sorted(top_genes.keys()):
            c = top_counts[fid]
            if c == 0:
                continue
            avg = {g: v / c for g, v in top_genes[fid].items()}
            gene_list = [g for g, _ in sorted(avg.items(), key=lambda x: -x[1])[:TOP_GENES]]

            for lib in LIBS:
                try:
                    res = gp.enrich(gene_list=gene_list, gene_sets=lib, outdir=None, no_plot=True)
                    if res and hasattr(res, "results") and len(res.results) > 0:
                        r = res.results.iloc[0]
                        all_enrichments.append({
                            "layer": layer_idx,
                            "feature_idx": fid,
                            "library": lib,
                            "term": str(r.get("Term", "")),
                            "p_value": float(r.get("P-value", 1.0)),
                            "fdr": float(r.get("Adjusted P-value", 1.0)),
                            "overlap": str(r.get("Overlap", "")),
                            "top_genes": ",".join(gene_list[:5]),
                        })
                except Exception:
                    pass

        dt = time.time() - t0
        log(f"  done in {dt:.0f}s")
        del sae
        gc.collect()

    # Save
    df = pd.DataFrame(all_enrichments)
    out_path = CAUSAL_DIR / "circuit_hub_enrichments.parquet"
    df.to_parquet(out_path, index=False)
    log(f"\nwrote {out_path} ({len(df)} rows)")

    # Summary per hub
    if len(df) > 0:
        sig = df[df["fdr"] < 0.05]
        log(f"significant enrichments (FDR<0.05): {len(sig)}")
        n_annotated = sig.groupby(["layer", "feature_idx"]).size().reset_index()
        log(f"features with >= 1 significant enrichment: {len(n_annotated)}")

        # Print hub annotations
        log("\nCircuit hub annotations:")
        for _, row in df.drop_duplicates(["layer", "feature_idx"]).iterrows():
            sub = df[(df["layer"] == row["layer"]) & (df["feature_idx"] == row["feature_idx"])]
            best = sub.loc[sub["fdr"].idxmin()]
            log(f"  conv_{int(row['layer'])}/F{int(row['feature_idx'])}: "
                f"{best['term'][:60]} (FDR={best['fdr']:.2e}) | genes: {best['top_genes']}")

    log("DONE")


if __name__ == "__main__":
    main()
