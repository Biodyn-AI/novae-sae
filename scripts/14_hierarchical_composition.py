#!/usr/bin/env python
"""H10 — hierarchical composition: do later-layer SAE features decompose
linearly into combinations of earlier-layer features?

For each pair of layers (L, L+1) and selected non-adjacent pairs, encode
the shared 566K node activations through both layers' SAEs, compute the
cross-correlation matrix, and for each target feature in the later layer
find the max absolute Pearson correlation with any feature in the earlier
layer.

If H10 is true, the max-correlation should increase with layer depth —
later features should be more predictable from earlier features because
the GAT progressively composes simpler patterns into richer ones.

Outputs: atlas/novae-human-0/causal/hierarchical_composition.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "14_hierarchical_composition.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


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
}


def encode_sparse_csc(act_path: Path, sae_path: Path, d_in: int, n_features: int, k: int) -> sparse.csc_matrix:
    """Load activations and SAE, encode as sparse CSC."""
    agg = np.load(act_path).astype(np.float32, copy=False)
    n_cells = agg.shape[0]
    sae = TopKSAE(d_in=d_in, n_features=n_features, k=k)
    sae.load_state_dict(torch.load(sae_path, map_location="cpu"))
    sae.eval()

    rows = np.empty(n_cells * k, dtype=np.int64)
    cols = np.empty(n_cells * k, dtype=np.int32)
    data = np.empty(n_cells * k, dtype=np.float32)
    write = 0
    chunk = 32768
    with torch.no_grad():
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            xb = torch.tensor(agg[start:end], dtype=torch.float32)
            z, idx = sae.encode(xb)
            B = end - start
            vals = z.gather(-1, idx).cpu().numpy()
            idx_np = idx.cpu().numpy().astype(np.int32)
            n = B * k
            rows[write:write + n] = np.repeat(np.arange(start, end, dtype=np.int64), k)
            cols[write:write + n] = idx_np.ravel()
            data[write:write + n] = vals.ravel()
            write += n
    del agg, sae
    gc.collect()
    return sparse.csr_matrix(
        (data[:write], (rows[:write], cols[:write])),
        shape=(n_cells, n_features),
    ).tocsc()


def cross_max_corr(feats_src: sparse.csc_matrix, feats_tgt: sparse.csc_matrix) -> dict:
    """Compute per-target-feature max abs Pearson correlation with any source feature.

    Uses the sparse structure to compute column-wise dot products efficiently.
    Returns summary stats.
    """
    n_src = feats_src.shape[1]
    n_tgt = feats_tgt.shape[1]
    n = feats_src.shape[0]
    assert feats_tgt.shape[0] == n

    # Column norms for normalization
    src_norms = sparse.linalg.norm(feats_src, axis=0)  # (n_src,)
    tgt_norms = sparse.linalg.norm(feats_tgt, axis=0)  # (n_tgt,)

    # Compute cross-dot: C = feats_src.T @ feats_tgt → (n_src, n_tgt)
    # This is the key step; scipy handles sparse × sparse
    C = (feats_src.T @ feats_tgt).toarray().astype(np.float64)  # (n_src, n_tgt)

    # Normalize: cos = C / (src_norm * tgt_norm)
    src_norms_safe = np.where(src_norms > 0, src_norms, 1.0)
    tgt_norms_safe = np.where(tgt_norms > 0, tgt_norms, 1.0)
    C /= src_norms_safe[:, None]
    C /= tgt_norms_safe[None, :]

    # Per-target: max abs cosine across source features
    max_abs_cos = np.abs(C).max(axis=0)  # (n_tgt,)

    return {
        "n_source": int(n_src),
        "n_target": int(n_tgt),
        "max_abs_cos_mean": float(max_abs_cos.mean()),
        "max_abs_cos_median": float(np.median(max_abs_cos)),
        "max_abs_cos_p90": float(np.percentile(max_abs_cos, 90)),
        "max_abs_cos_p95": float(np.percentile(max_abs_cos, 95)),
        "n_above_0.3": int((max_abs_cos > 0.3).sum()),
        "n_above_0.5": int((max_abs_cos > 0.5).sum()),
        "frac_above_0.3": float((max_abs_cos > 0.3).mean()),
        "frac_above_0.5": float((max_abs_cos > 0.5).mean()),
    }


def main() -> None:
    log("=" * 72)
    log("H10: hierarchical composition — cross-layer feature alignment")

    layers = [f"conv_{i}" for i in range(10)]

    # Pairs to test: adjacent + a few long-range
    pairs = []
    for i in range(9):
        pairs.append((layers[i], layers[i + 1]))
    # Long-range
    for src, tgt in [("conv_0", "conv_5"), ("conv_0", "conv_9"),
                     ("conv_5", "conv_9"), ("conv_3", "conv_7")]:
        pairs.append((src, tgt))

    # Cache encoded features so we don't re-encode for shared layers
    cache: dict[str, sparse.csc_matrix] = {}

    results = []
    for src_name, tgt_name in pairs:
        log(f"\n{src_name} → {tgt_name}")

        if src_name not in cache:
            d_in, n_feat, k = SAE_SPECS[src_name]
            log(f"  encoding {src_name} (d_in={d_in}, n_feat={n_feat}, k={k})")
            cache[src_name] = encode_sparse_csc(
                ACT_DIR / f"{src_name}.npy",
                SAE_DIR / f"{src_name}.pt",
                d_in, n_feat, k,
            )
            log(f"    {cache[src_name].shape}, nnz={cache[src_name].nnz:,}")

        if tgt_name not in cache:
            d_in, n_feat, k = SAE_SPECS[tgt_name]
            log(f"  encoding {tgt_name}")
            cache[tgt_name] = encode_sparse_csc(
                ACT_DIR / f"{tgt_name}.npy",
                SAE_DIR / f"{tgt_name}.pt",
                d_in, n_feat, k,
            )
            log(f"    {cache[tgt_name].shape}, nnz={cache[tgt_name].nnz:,}")

        t0 = time.time()
        stats = cross_max_corr(cache[src_name], cache[tgt_name])
        stats["source"] = src_name
        stats["target"] = tgt_name
        stats["layer_gap"] = int(tgt_name.split("_")[1]) - int(src_name.split("_")[1])
        results.append(stats)
        log(f"  max_cos mean={stats['max_abs_cos_mean']:.4f} "
            f"median={stats['max_abs_cos_median']:.4f} "
            f"frac>0.3={stats['frac_above_0.3']:.3f} "
            f"frac>0.5={stats['frac_above_0.5']:.3f} "
            f"({time.time()-t0:.1f}s)")

        # Free memory for layers no longer needed as source in any remaining pair
        remaining_sources = set(p[0] for p in pairs[pairs.index((src_name, tgt_name)) + 1:])
        remaining_targets = set(p[1] for p in pairs[pairs.index((src_name, tgt_name)) + 1:])
        remaining_needed = remaining_sources | remaining_targets
        for k_cached in list(cache.keys()):
            if k_cached not in remaining_needed:
                del cache[k_cached]
                gc.collect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "hierarchical_composition.json"
    json.dump(results, open(out_path, "w"), indent=2)
    log(f"\nwrote {out_path} ({len(results)} pairs)")

    log("\nH10 evaluation: composition should increase with depth")
    adj = [r for r in results if r["layer_gap"] == 1]
    if len(adj) >= 2:
        early_frac = np.mean([r["frac_above_0.3"] for r in adj[:3]])
        late_frac = np.mean([r["frac_above_0.3"] for r in adj[-3:]])
        log(f"  early layers (conv_0→3) mean frac>0.3: {early_frac:.4f}")
        log(f"  late layers  (conv_6→9) mean frac>0.3: {late_frac:.4f}")
        if late_frac > early_frac:
            log("  → H10 SUPPORTED: later layer pairs have higher cross-layer alignment")
        else:
            log("  → H10 REFUTED: cross-layer alignment does not increase with depth")
    log("DONE")


if __name__ == "__main__":
    main()
