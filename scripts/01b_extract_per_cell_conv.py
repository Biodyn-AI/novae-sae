#!/usr/bin/env python
"""Extract PER-CELL conv-layer activations (not subsampled).

The original 01_extract_activations.py subsampled 64 nodes per batch for
the conv layers, losing the cell-index mapping. This script re-runs the
model with hooks that aggregate conv outputs per cell (using the PyG
batch index), producing per-cell conv activations that can be used for
spatial Moran's I at each layer depth (H7 second half).

For each slide, for each conv layer L:
  per_cell_conv_L = scatter_mean(conv_L_output, index=data.batch)

This pools the per-node conv output to one vector per cell (graph) in
the batch, then concatenates across batches and reindexes via
valid_indices. The result is a (n_cells, dim_L) array per layer per slide.

To keep memory bounded, processes one slide at a time and saves to
per_slide/*/conv_{L}_percell.npy.
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
import torch
from torch_geometric.utils import scatter

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ACT_DIR = ROOT / "activations" / "novae-human-0"
PER_SLIDE_DIR = ACT_DIR / "per_slide"
LOG_PATH = ROOT / "logs" / "01b_extract_per_cell_conv.log"

NUM_CONVS = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("01b: extract per-cell conv activations (for H7 depth)")

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()
    log(f"model loaded on {DEVICE}")

    for slide in manifest["slides"]:
        tissue = slide["tissue"]
        slide_name = slide["name"]
        out_dir = PER_SLIDE_DIR / f"{tissue}__{slide_name}"
        done_marker = out_dir / "DONE_PERCELL_CONV"

        if done_marker.exists():
            log(f"SKIP (already done): {tissue}/{slide_name}")
            continue

        h5ad = DATA_DIR / tissue / f"{slide_name}.h5ad"
        log(f"\n{tissue}/{slide_name}")
        a = ad.read_h5ad(h5ad)
        novae.spatial_neighbors(a)
        n_cells = a.n_obs
        log(f"  {n_cells:,} cells")

        # Prepare adata for model
        adatas = model._prepare_adatas(a)
        a_prepared = adatas[0]
        dm = model._init_datamodule(a_prepared)
        valid_indices = dm.dataset.valid_indices[0]
        dl = dm.predict_dataloader()

        # Per-batch accumulator: per_conv[L] = list of (batch_size, dim) arrays
        per_conv_batches = [[] for _ in range(NUM_CONVS)]
        batch_state = {"batch_tensor": None}

        def make_conv_hook(layer_idx):
            def hook(_mod, _inp, out):
                if not isinstance(out, torch.Tensor):
                    return
                # Aggregate per-cell (per-graph) using scatter_mean
                bt = batch_state["batch_tensor"]
                if bt is None:
                    return
                n_graphs = int(bt.max().item()) + 1
                per_cell = scatter(out, bt, dim=0, dim_size=n_graphs, reduce="mean")
                per_conv_batches[layer_idx].append(per_cell.detach().cpu().float().numpy())
            return hook

        # Register hooks on all conv layers
        handles = []
        for i in range(NUM_CONVS):
            handles.append(model.encoder.gnn.convs[i].register_forward_hook(make_conv_hook(i)))

        # Hook on encoder.forward to capture data.batch before it's used
        orig_encoder_forward = model.encoder.forward

        def hooked_encoder_forward(data):
            batch_state["batch_tensor"] = data.batch
            return orig_encoder_forward(data)

        model.encoder.forward = hooked_encoder_forward

        t0 = time.time()
        # Run compute_representations (drives the dataloader + hooks)
        model.compute_representations(a, zero_shot=True)
        dt = time.time() - t0
        log(f"  encode {dt:.0f}s ({n_cells / dt:.0f} cells/s)")

        # Restore encoder forward
        model.encoder.forward = orig_encoder_forward
        for h in handles:
            h.remove()

        # Concatenate per-conv outputs and reindex via valid_indices
        for L in range(NUM_CONVS):
            stacked = np.concatenate(per_conv_batches[L], axis=0)  # (n_valid, dim)
            # Fill into full (n_cells, dim) array via valid_indices
            full = np.zeros((n_cells, stacked.shape[1]), dtype=np.float32)
            full[valid_indices[:len(stacked)]] = stacked[:len(valid_indices)]
            np.save(out_dir / f"conv_{L}_percell.npy", full)
            if L == 0 or L == 9:
                log(f"  conv_{L}_percell: {full.shape}")

        done_marker.touch()
        del a, a_prepared, dm, per_conv_batches
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    log("ALL SLIDES DONE")


if __name__ == "__main__":
    main()
