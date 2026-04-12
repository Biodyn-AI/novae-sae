#!/usr/bin/env python
"""H7 depth half — spatial coherence (Moran's I) per conv layer.

Requires per-cell conv activations from 01b_extract_per_cell_conv.py
(files: per_slide/*/conv_{L}_percell.npy). For each layer 0-9 + aggregator,
encodes through the corresponding SAE, computes Moran's I per feature per
slide, and outputs spatial_coherence_layer_{L}.parquet.

The final output is a summary across layers for the H7 depth-dependence
plot: does mean Moran's I increase with layer depth?
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "08b_spatial_coherence_per_layer.log"

K_NEIGHBORS = 10
MIN_ACTIVE = 50
TOP_FRAC = 0.01

SAE_SPECS = {
    "conv_0": (128, 4096, 32),
    "conv_1": (128, 4096, 32),
    "conv_2": (128, 4096, 32),
    "conv_3": (128, 4096, 32),
    "conv_4": (128, 4096, 32),
    "conv_5": (128, 4096, 32),
    "conv_6": (128, 4096, 32),
    "conv_7": (128, 4096, 32),
    "conv_8": (128, 4096, 32),
    "conv_9": (64, 2048, 16),
    "aggregator": (64, 2048, 16),
}

# Process a subset of slides for speed (3 diverse tissues)
SLIDE_SUBSET = [
    ("tonsil", "Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs"),
    ("lymph_node", "Xenium_V1_hLymphNode_nondiseased_section_outs"),
    ("breast", "Xenium_V1_FFPE_Human_Breast_IDC_outs"),
    ("liver", "Xenium_V1_hLiver_nondiseased_section_FFPE_outs"),
    ("lung", "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs"),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def morans_i(values: np.ndarray, W: sparse.csr_matrix) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    v = values - values.mean()
    var = (v ** 2).sum()
    if var <= 0:
        return float("nan")
    Wv = W @ v
    num = float(v @ Wv)
    s = float(W.sum())
    if s <= 0:
        return float("nan")
    return (n / s) * (num / var)


def encode_sparse_csc(act: np.ndarray, sae: TopKSAE, k_sae: int) -> sparse.csc_matrix:
    n = act.shape[0]
    rows = np.empty(n * k_sae, dtype=np.int64)
    cols = np.empty(n * k_sae, dtype=np.int32)
    data = np.empty(n * k_sae, dtype=np.float32)
    write = 0
    chunk = 32768
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
    return sparse.csr_matrix(
        (data[:write], (rows[:write], cols[:write])),
        shape=(n, sae.n_features),
    ).tocsc()


def compute_morans_for_layer(layer_name: str) -> dict:
    """Compute mean Moran's I for one layer across slide subset."""
    d_in, n_features, k = SAE_SPECS[layer_name]
    sae = TopKSAE(d_in=d_in, n_features=n_features, k=k)
    sae.load_state_dict(torch.load(SAE_DIR / f"{layer_name}.pt", map_location="cpu"))
    sae.eval()

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    slide_lookup = {s["name"]: s for s in manifest["slides"]}

    all_mi = []  # list of (feature_idx, slide_name, morans_i_value)

    for tissue, slide_name in SLIDE_SUBSET:
        slide_dir = ACT_DIR / "per_slide" / f"{tissue}__{slide_name}"
        if layer_name == "aggregator":
            act_path = slide_dir / "aggregator.npy"
        else:
            act_path = slide_dir / f"{layer_name}_percell.npy"
        if not act_path.exists():
            log(f"    SKIP {tissue}/{slide_name} — {act_path.name} not found")
            continue

        act = np.load(act_path).astype(np.float32, copy=False)
        n_cells = act.shape[0]
        if n_cells < MIN_ACTIVE * 2:
            continue

        # Load spatial coordinates
        h5ad = DATA_DIR / tissue / f"{slide_name}.h5ad"
        try:
            a = ad.read_h5ad(h5ad)
        except Exception:
            continue
        if "spatial" not in a.obsm:
            del a
            continue
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        if len(coords) != n_cells:
            del a
            continue
        del a

        # Encode as sparse
        feats = encode_sparse_csc(act, sae, k)
        del act

        # Build k-NN spatial weights
        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, n_jobs=-1).fit(coords)
        _, idx = nbrs.kneighbors(coords)
        idx = idx[:, 1:]
        r = np.repeat(np.arange(n_cells), K_NEIGHBORS)
        c = idx.ravel()
        d = np.ones_like(r, dtype=np.float32)
        W = sparse.csr_matrix((d, (r, c)), shape=(n_cells, n_cells))
        W = (W + W.T) / 2

        # Per-feature Moran's I (using CSC column access)
        indptr = feats.indptr
        indices = feats.indices
        data_arr = feats.data
        n_computed = 0
        for fid in range(n_features):
            si, ei = indptr[fid], indptr[fid + 1]
            nnz = ei - si
            if nnz < MIN_ACTIVE:
                continue
            col = np.zeros(n_cells, dtype=np.float32)
            col[indices[si:ei]] = data_arr[si:ei]
            mi = morans_i(np.abs(col), W)
            if not np.isnan(mi):
                all_mi.append((fid, slide_name, mi))
                n_computed += 1
        log(f"    {tissue}/{slide_name}: {n_computed} features")
        del feats, W
        gc.collect()

    del sae
    gc.collect()

    if not all_mi:
        return {"layer": layer_name, "n_features_with_signal": 0, "mean_morans_i": None}

    df = pd.DataFrame(all_mi, columns=["feature_idx", "slide_name", "morans_i"])
    per_feat = df.groupby("feature_idx")["morans_i"].mean()
    return {
        "layer": layer_name,
        "n_features_with_signal": int(len(per_feat)),
        "mean_morans_i": float(per_feat.mean()),
        "median_morans_i": float(per_feat.median()),
        "std_morans_i": float(per_feat.std()),
    }


def main() -> None:
    log("=" * 72)
    log("H7 depth half — spatial coherence per layer")

    layers = [f"conv_{i}" for i in range(10)] + ["aggregator"]
    results = []
    for layer in layers:
        log(f"\n{layer}:")
        t0 = time.time()
        r = compute_morans_for_layer(layer)
        r["time_s"] = round(time.time() - t0, 1)
        results.append(r)
        log(f"  → mean I = {r.get('mean_morans_i', 'N/A')}, n = {r.get('n_features_with_signal', 0)}, {r['time_s']}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "spatial_coherence_depth.json"
    json.dump(results, open(out_path, "w"), indent=2)
    log(f"\nwrote {out_path}")

    log("\nH7 depth evaluation:")
    for r in results:
        mi = r.get("mean_morans_i")
        bar = "#" * int((mi or 0) * 100)
        log(f"  {r['layer']:15s}  I = {mi:.4f if mi else 'N/A':>8s}  {bar}")

    # Check trend
    conv_is = [r.get("mean_morans_i") for r in results if r["layer"].startswith("conv_") and r.get("mean_morans_i") is not None]
    if len(conv_is) >= 2:
        if conv_is[-1] > conv_is[0]:
            log("  → H7 depth SUPPORTED: Moran's I increases with layer depth")
        else:
            log("  → H7 depth NOT SUPPORTED: Moran's I does not increase with depth")
    log("DONE")


if __name__ == "__main__":
    main()
