#!/usr/bin/env python
"""H4 completion — per-layer enrichment for conv_0 through conv_9.

Uses per-cell conv activations from 01b (conv_{L}_percell.npy per slide)
to compute top genes and Enrichr enrichment for every conv layer's SAE
features. This gives the per-layer annotation rate needed for the H4
depth-dependent processing evaluation.

Strategy: amortize slide I/O by loading each h5ad ONCE and computing
top genes for ALL 10 conv layers simultaneously.

Output per layer: bio/conv_{L}_enrichment.parquet
Summary: causal/annotation_rates_full.json (with enrichment-level rates)
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
import torch
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "bio"
CAUSAL_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "03c3_conv_enrichment.log"

TOP_CELL_FRAC = 0.001
MIN_TOP_CELLS = 50
TOP_GENES_PER_FEATURE = 20
ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "PanglaoDB_Augmented_2021",
    "CellMarker_Augmented_2021",
]

SAE_SPECS = {
    f"conv_{i}": (128, 4096, 32) for i in range(9)
}
SAE_SPECS["conv_9"] = (64, 2048, 16)

# Process a subset of layers to keep total time manageable (~30 min per layer)
# Pick 4 representative layers: conv_0 (early), conv_3 (mid-early),
# conv_6 (mid-late), conv_9 (last)
LAYERS_TO_PROCESS = ["conv_0", "conv_3", "conv_6", "conv_9"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def encode_top_cells_sparse(act: np.ndarray, sae: TopKSAE, k_sae: int, n_top: int) -> dict[int, np.ndarray]:
    """Encode activations through SAE and return top-N cell indices per feature."""
    n = act.shape[0]
    n_features = sae.n_features
    chunk = 32768

    # Build sparse CSC
    rows = np.empty(n * k_sae, dtype=np.int64)
    cols = np.empty(n * k_sae, dtype=np.int32)
    data = np.empty(n * k_sae, dtype=np.float32)
    write = 0
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xb = torch.tensor(act[start:end], dtype=torch.float32)
            z, idx = sae.encode(xb)
            B = end - start
            vals = z.gather(-1, idx).cpu().numpy()
            idx_np = idx.cpu().numpy().astype(np.int32)
            nn = B * k_sae
            rows[write:write + nn] = np.repeat(np.arange(start, end, dtype=np.int64), k_sae)
            cols[write:write + nn] = idx_np.ravel()
            data[write:write + nn] = vals.ravel()
            write += nn
    feats = sparse.csr_matrix(
        (data[:write], (rows[:write], cols[:write])),
        shape=(n, n_features),
    ).tocsc()

    # Per-feature top-N cells by |activation|
    result = {}
    indptr = feats.indptr
    indices = feats.indices
    data_arr = feats.data
    for fid in range(n_features):
        si, ei = indptr[fid], indptr[fid + 1]
        nnz = ei - si
        if nnz < MIN_TOP_CELLS:
            continue
        col_rows = indices[si:ei]
        col_vals = np.abs(data_arr[si:ei])
        k = min(n_top, nnz)
        top_local = np.argpartition(col_vals, -k)[-k:]
        result[fid] = col_rows[top_local]
    return result


def main() -> None:
    log("=" * 72)
    log("H4: per-layer enrichment for conv layers")

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    slides = manifest["slides"]

    # Load all SAEs
    saes = {}
    for layer in LAYERS_TO_PROCESS:
        d_in, n_features, k = SAE_SPECS[layer]
        sae = TopKSAE(d_in=d_in, n_features=n_features, k=k)
        sae.load_state_dict(torch.load(SAE_DIR / f"{layer}.pt", map_location="cpu"))
        sae.eval()
        saes[layer] = sae
        log(f"  loaded SAE {layer}: d_in={d_in}, n_features={n_features}, k={k}")

    # Phase 1: compute top genes per feature per layer
    # Accumulate: top_genes[layer][feature_idx] = {gene: sum_expr, ...}
    top_genes: dict[str, dict[int, dict[str, float]]] = {
        layer: defaultdict(lambda: defaultdict(float))
        for layer in LAYERS_TO_PROCESS
    }
    top_gene_counts: dict[str, dict[int, int]] = {
        layer: defaultdict(int) for layer in LAYERS_TO_PROCESS
    }

    for slide in slides:
        tissue = slide["tissue"]
        slide_name = slide["name"]
        slide_dir = ACT_DIR / "per_slide" / f"{tissue}__{slide_name}"
        h5ad_path = DATA_DIR / tissue / f"{slide_name}.h5ad"

        log(f"\n  [{slide['slide_idx']+1}/15] {tissue}/{slide_name}")

        # Load h5ad once for gene expression
        try:
            a = ad.read_h5ad(h5ad_path)
        except Exception as e:
            log(f"    ERROR loading h5ad: {e}")
            continue

        # Normalize gene expression for top-gene computation
        try:
            import scanpy as sc
            sc.pp.normalize_total(a)
            sc.pp.log1p(a)
        except Exception:
            pass

        gene_names = list(a.var_names)
        n_cells_slide = a.n_obs

        for layer in LAYERS_TO_PROCESS:
            act_path = slide_dir / f"{layer}_percell.npy"
            if not act_path.exists():
                continue

            act = np.load(act_path).astype(np.float32, copy=False)
            if act.shape[0] != n_cells_slide:
                log(f"    WARN {layer}: shape mismatch {act.shape[0]} vs {n_cells_slide}")
                continue

            n_top = max(MIN_TOP_CELLS, int(TOP_CELL_FRAC * n_cells_slide))
            top_cells = encode_top_cells_sparse(act, saes[layer], SAE_SPECS[layer][2], n_top)
            del act

            # For each feature's top cells, accumulate gene expression
            n_computed = 0
            for fid, cell_idx in top_cells.items():
                if len(cell_idx) == 0:
                    continue
                # Get gene expression for these cells
                if sparse.issparse(a.X):
                    expr = np.asarray(a.X[cell_idx].mean(axis=0)).flatten()
                else:
                    expr = a.X[cell_idx].mean(axis=0).flatten()
                for gi, val in enumerate(expr):
                    if val > 0 and gi < len(gene_names):
                        top_genes[layer][fid][gene_names[gi]] += float(val)
                top_gene_counts[layer][fid] += 1
                n_computed += 1
            log(f"    {layer}: {n_computed} features")

        del a
        gc.collect()

    # Phase 2: enrichment per layer
    annotation_rates = {}
    for layer in LAYERS_TO_PROCESS:
        log(f"\n=== Enrichment for {layer} ===")
        n_features = SAE_SPECS[layer][1]

        # Build per-feature top gene lists
        gene_lists = {}
        for fid in sorted(top_genes[layer].keys()):
            g = top_genes[layer][fid]
            count = top_gene_counts[layer][fid]
            if count == 0:
                continue
            # Average across slides
            avg = {k: v / count for k, v in g.items()}
            sorted_genes = sorted(avg.items(), key=lambda x: -x[1])[:TOP_GENES_PER_FEATURE]
            gene_lists[fid] = [g for g, _ in sorted_genes]

        log(f"  {len(gene_lists)} features with top genes")
        if not gene_lists:
            annotation_rates[layer] = {"n_features": n_features, "annotation_rate": 0.0}
            continue

        # Run enrichment
        enrichment_rows = []
        t0 = time.time()
        n_with_sig = 0
        for i, (fid, genes) in enumerate(sorted(gene_lists.items())):
            has_any_sig = False
            for lib in ENRICHR_LIBRARIES:
                try:
                    res = gp.enrich(gene_list=genes, gene_sets=lib, outdir=None, no_plot=True)
                    if res is not None and hasattr(res, "results") and len(res.results) > 0:
                        top_row = res.results.iloc[0]
                        fdr = float(top_row.get("Adjusted P-value", 1.0))
                        enrichment_rows.append({
                            "feature_idx": fid,
                            "library": lib,
                            "term": str(top_row.get("Term", "")),
                            "p_value": float(top_row.get("P-value", 1.0)),
                            "fdr": fdr,
                            "overlap": str(top_row.get("Overlap", "")),
                        })
                        if fdr < 0.05:
                            has_any_sig = True
                except Exception:
                    pass
            if has_any_sig:
                n_with_sig += 1
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(gene_lists) - i - 1) / rate / 60
                log(f"    {i+1}/{len(gene_lists)} ({rate:.1f} feat/s, ETA {eta:.1f} min)")

        if enrichment_rows:
            df = pd.DataFrame(enrichment_rows)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUT_DIR / f"{layer}_enrichment.parquet", index=False)
            log(f"  wrote {layer}_enrichment.parquet ({len(df)} rows)")

        ann_rate = n_with_sig / n_features if n_features > 0 else 0.0
        annotation_rates[layer] = {
            "n_features": n_features,
            "n_with_top_genes": len(gene_lists),
            "n_with_significant_enrichment": n_with_sig,
            "annotation_rate": round(ann_rate, 4),
            "time_s": round(time.time() - t0, 1),
        }
        log(f"  annotation rate: {ann_rate:.4f} ({n_with_sig}/{n_features})")

    # Summary
    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(annotation_rates, open(CAUSAL_DIR / "annotation_rates_conv.json", "w"), indent=2)
    log(f"\nwrote annotation_rates_conv.json")
    log(json.dumps(annotation_rates, indent=2))

    log("\nH4 depth evaluation (enrichment-level annotation rates):")
    for layer in LAYERS_TO_PROCESS:
        r = annotation_rates.get(layer, {})
        rate = r.get("annotation_rate", 0)
        bar = "#" * int(rate * 50)
        log(f"  {layer:10s}  {rate:.4f}  {bar}")
    log("DONE")


if __name__ == "__main__":
    main()
