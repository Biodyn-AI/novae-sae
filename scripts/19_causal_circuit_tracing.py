#!/usr/bin/env python
"""GAP 1: Causal circuit tracing for Novae GATv2.

For each source feature F at source layer L, zero F's TopK coefficient
in the SAE code, decode back to hidden space, and let the modified state
propagate through subsequent GATv2 layers via graph attention. Measure
Cohen's d on downstream SAE features.

Protocol:
  1. Pick one small slide (brain, 24K cells) for tractable compute.
  2. Run baseline forward pass, capture conv outputs at all layers.
  3. For each source feature: run ablated forward pass with a hook that
     modifies conv[L]'s output (SAE-ablate feature F, decode back).
  4. Capture downstream conv outputs and SAE-encode them.
  5. Compute Cohen's d between baseline and ablated SAE features.
  6. Keep edges where |d| > 0.5 and consistency > 0.6.

Output: atlas/novae-human-0/causal/causal_circuit_edges.parquet
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import anndata as ad
import novae
import numpy as np
import pandas as pd
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "19_causal_circuit.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SLIDE = ("brain", "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs")
NUM_CONVS = 10
SOURCE_LAYERS = [0, 5, 9]  # early, mid, late
TOP_SOURCE_FEATURES = 30   # per source layer
D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.6

SAE_SPECS = {
    **{f"conv_{i}": (128, 4096, 32) for i in range(9)},
    "conv_9": (64, 2048, 16),
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_saes() -> dict[int, TopKSAE]:
    """Load SAEs for all conv layers."""
    saes = {}
    for layer_idx in range(NUM_CONVS):
        name = f"conv_{layer_idx}"
        d_in, n_feat, k = SAE_SPECS[name]
        sae = TopKSAE(d_in=d_in, n_features=n_feat, k=k)
        sae.load_state_dict(torch.load(SAE_DIR / f"{name}.pt", map_location="cpu"))
        sae.eval()
        saes[layer_idx] = sae
    return saes


def select_source_features(saes: dict, per_cell_dir: Path, n_top: int) -> dict[int, list[int]]:
    """Select top-N most active features per source layer."""
    result = {}
    for layer_idx in SOURCE_LAYERS:
        act_path = per_cell_dir / f"conv_{layer_idx}_percell.npy"
        if not act_path.exists():
            log(f"  WARN: {act_path} not found")
            continue
        act = np.load(act_path).astype(np.float32, copy=False)
        sae = saes[layer_idx]
        # Compute mean |activation| per feature
        chunk = 32768
        feat_sum = np.zeros(sae.n_features, dtype=np.float64)
        n_cells = act.shape[0]
        with torch.no_grad():
            for start in range(0, n_cells, chunk):
                end = min(start + chunk, n_cells)
                xb = torch.tensor(act[start:end])
                z, _ = sae.encode(xb)
                feat_sum += np.abs(z.numpy()).sum(axis=0)
        feat_mean = feat_sum / n_cells
        top_features = np.argsort(-feat_mean)[:n_top].tolist()
        result[layer_idx] = top_features
        log(f"  layer {layer_idx}: top-{n_top} features selected (max mean_abs={feat_mean[top_features[0]]:.4f})")
        del act
    return result


def run_forward_pass(model, adata, saes, ablate_layer=None, ablate_feature=None):
    """Run forward pass capturing SAE-encoded features at all conv layers.

    If ablate_layer and ablate_feature are set, hooks conv[ablate_layer]
    to zero that feature in the SAE code and decode back.

    Returns: dict[layer_idx] -> np.ndarray of shape (n_batches, batch_nodes, n_features)
    concatenated across batches.
    """
    gnn = model.encoder.gnn
    layer_outputs = {i: [] for i in range(NUM_CONVS)}
    hooks = []

    def make_capture_hook(layer_idx):
        def hook(module, inp, out):
            layer_outputs[layer_idx].append(out.detach().cpu().float())
        return hook

    def make_ablation_hook(layer_idx, feature_idx, sae):
        def hook(module, inp, out):
            with torch.no_grad():
                out_dev = out.float()
                # Move to CPU for SAE (SAE is on CPU)
                out_cpu = out_dev.cpu()
                z, idx = sae.encode(out_cpu)
                z[:, feature_idx] = 0.0
                out_ablated = sae.decode(z)
                # Put back on device
                out_new = out_ablated.to(out.device).to(out.dtype)
            layer_outputs[layer_idx].append(out_new.detach().cpu().float())
            return out_new
        return hook

    # Register hooks
    for i in range(NUM_CONVS):
        if ablate_layer is not None and i == ablate_layer:
            hooks.append(gnn.convs[i].register_forward_hook(
                make_ablation_hook(i, ablate_feature, saes[i])
            ))
        else:
            hooks.append(gnn.convs[i].register_forward_hook(make_capture_hook(i)))

    # Run forward pass
    model.compute_representations(adata, zero_shot=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    return layer_outputs


def compute_cohen_d(baseline_feats, ablated_feats):
    """Compute Cohen's d and consistency between baseline and ablated feature activations.

    baseline_feats, ablated_feats: (n_nodes, n_features) tensors
    Returns: (cohen_d, consistency, sign) arrays of shape (n_features,)
    """
    diff = ablated_feats - baseline_feats  # (n_nodes, n_features)
    mean_diff = diff.mean(axis=0)
    std_diff = diff.std(axis=0)
    std_diff = np.where(std_diff > 1e-9, std_diff, 1.0)
    d = mean_diff / std_diff

    # Consistency: fraction of nodes where sign(diff) matches sign(mean_diff)
    sign_mean = np.sign(mean_diff)
    sign_per_node = np.sign(diff)
    consistency = (sign_per_node == sign_mean[None, :]).mean(axis=0)
    sign = np.sign(d)

    return d, consistency, sign


def main():
    log("=" * 72)
    log("GAP 1: Causal circuit tracing for Novae GATv2")

    log("loading model + SAEs")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    saes = load_saes()
    log(f"  loaded {len(saes)} layer SAEs")

    # Load slide
    tissue, slide_name = SLIDE
    h5ad = DATA_DIR / tissue / f"{slide_name}.h5ad"
    log(f"loading slide: {tissue}/{slide_name}")
    adata = ad.read_h5ad(h5ad)
    novae.spatial_neighbors(adata)
    log(f"  {adata.n_obs:,} cells")

    # Select source features
    per_cell_dir = ROOT / "activations" / "novae-human-0" / "per_slide" / f"{tissue}__{slide_name}"
    log("selecting source features")
    source_features = select_source_features(saes, per_cell_dir, TOP_SOURCE_FEATURES)

    # Run baseline forward pass
    log("running baseline forward pass")
    t0 = time.time()
    baseline_outputs = run_forward_pass(model, adata, saes)
    log(f"  baseline done in {time.time()-t0:.0f}s")

    # SAE-encode baseline outputs
    log("SAE-encoding baseline outputs")
    baseline_encoded = {}
    for layer_idx in range(NUM_CONVS):
        tensors = baseline_outputs[layer_idx]
        if not tensors:
            continue
        cat = torch.cat(tensors, dim=0).numpy()
        sae = saes[layer_idx]
        chunks = []
        with torch.no_grad():
            for s in range(0, cat.shape[0], 32768):
                e = min(s + 32768, cat.shape[0])
                z, _ = sae.encode(torch.tensor(cat[s:e]))
                chunks.append(z.numpy())
        baseline_encoded[layer_idx] = np.vstack(chunks)
        log(f"  layer {layer_idx}: {baseline_encoded[layer_idx].shape}")
    del baseline_outputs
    gc.collect()

    # Circuit tracing: for each source feature, run ablated pass
    all_edges = []
    total_source = sum(len(v) for v in source_features.values())
    done = 0

    for src_layer, features in source_features.items():
        log(f"\n=== Source layer conv_{src_layer} ({len(features)} features) ===")

        for feat_idx in features:
            t0 = time.time()

            # Run ablated forward pass
            ablated_outputs = run_forward_pass(
                model, adata, saes,
                ablate_layer=src_layer,
                ablate_feature=feat_idx,
            )

            # SAE-encode ablated outputs at downstream layers
            for tgt_layer in range(src_layer + 1, NUM_CONVS):
                if tgt_layer not in baseline_encoded:
                    continue
                tensors = ablated_outputs.get(tgt_layer, [])
                if not tensors:
                    continue
                abl_cat = torch.cat(tensors, dim=0).numpy()

                # SAE-encode
                sae = saes[tgt_layer]
                abl_chunks = []
                with torch.no_grad():
                    for s in range(0, abl_cat.shape[0], 32768):
                        e = min(s + 32768, abl_cat.shape[0])
                        z, _ = sae.encode(torch.tensor(abl_cat[s:e]))
                        abl_chunks.append(z.numpy())
                abl_encoded = np.vstack(abl_chunks)

                base = baseline_encoded[tgt_layer]
                n_nodes = min(base.shape[0], abl_encoded.shape[0])
                base_trim = base[:n_nodes]
                abl_trim = abl_encoded[:n_nodes]

                # Cohen's d
                d, consistency, sign = compute_cohen_d(base_trim, abl_trim)

                # Filter significant edges
                mask = (np.abs(d) > D_THRESHOLD) & (consistency > CONSISTENCY_THRESHOLD)
                sig_features = np.where(mask)[0]

                for tgt_feat in sig_features:
                    all_edges.append({
                        "source_layer": src_layer,
                        "source_feature": feat_idx,
                        "target_layer": tgt_layer,
                        "target_feature": int(tgt_feat),
                        "cohen_d": float(d[tgt_feat]),
                        "consistency": float(consistency[tgt_feat]),
                        "sign": int(sign[tgt_feat]),
                        "n_nodes": n_nodes,
                    })

            del ablated_outputs
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

            done += 1
            if done % 5 == 0 or done == total_source:
                log(f"  [{done}/{total_source}] conv_{src_layer}/F{feat_idx}: "
                    f"{len(all_edges)} edges total, {time.time()-t0:.1f}s")

    # Save results
    df = pd.DataFrame(all_edges)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "causal_circuit_edges.parquet"
    df.to_parquet(out_path, index=False)
    log(f"\nwrote {out_path} ({len(df)} edges)")

    # Summary
    summary = {
        "n_edges": len(df),
        "n_source_features": total_source,
        "source_layers": SOURCE_LAYERS,
        "slide": f"{tissue}/{slide_name}",
        "n_cells": int(adata.n_obs),
        "d_threshold": D_THRESHOLD,
        "consistency_threshold": CONSISTENCY_THRESHOLD,
    }
    if len(df) > 0:
        summary["n_excitatory"] = int((df["sign"] > 0).sum())
        summary["n_inhibitory"] = int((df["sign"] < 0).sum())
        summary["inhibitory_fraction"] = float((df["sign"] < 0).mean())
        summary["median_abs_d"] = float(df["cohen_d"].abs().median())
        summary["mean_layer_gap"] = float((df["target_layer"] - df["source_layer"]).mean())

        # Per-source-layer breakdown
        for sl in SOURCE_LAYERS:
            sub = df[df["source_layer"] == sl]
            summary[f"edges_from_conv_{sl}"] = len(sub)

    json.dump(summary, open(OUT_DIR / "causal_circuit_edges.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
