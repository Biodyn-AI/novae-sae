#!/usr/bin/env python
"""Phase 1 — extract Novae activations on a curated subset of slides.

Hooks every interpretability-relevant surface of `novae-human-0`:

  - cell_embedder           : per-node 512-dim embedded gene features
  - encoder.gnn.convs[0..9] : per-node GAT layer outputs (128 dim, 64 dim at conv 9)
  - encoder.node_aggregation: per-cell 64-dim "cell-in-niche" representation
                              (verified == adata.obsm['novae_latent'])

The aggregator output is saved fully (one row per cell). Per-node surfaces are
subsampled inside the hook (NODES_PER_BATCH_KEEP rows per batch) to keep peak
memory bounded.

Each slide is saved to its own `per_slide/<tissue>__<name>/` directory with a
`DONE` marker so the script is restartable. After all slides finish a final
consolidation pass writes global arrays + a manifest.json.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import anndata as ad
import novae
import numpy as np
import torch

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "activations" / "novae-human-0"
PER_SLIDE_DIR = OUT_DIR / "per_slide"
LOG_PATH = ROOT / "logs" / "01_extract_activations.log"

# 15 slides chosen to span every human tissue + 3 technologies (xenium /
# merscope / cosmx). Where both normal and disease slides are available the
# normal slide is preferred to keep niche/cell-type signal less confounded by
# tumor microenvironment. Cell counts come from datasets/metadata.csv.
SLIDES: list[tuple[str, str, str, int]] = [
    # tissue,        technology, slide_name,                                                            n_cells
    ("tonsil",        "xenium",   "Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs",  864388),
    ("lymph_node",    "xenium",   "Xenium_V1_hLymphNode_nondiseased_section_outs",                        377985),
    ("brain",         "xenium",   "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",                    24406),
    ("lung",          "xenium",   "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs",         295883),
    ("liver",         "xenium",   "Xenium_V1_hLiver_nondiseased_section_FFPE_outs",                       239271),
    ("colon",         "xenium",   "Xenium_V1_hColon_Non_diseased_Base_FFPE_outs",                         270984),
    ("breast",        "xenium",   "Xenium_V1_FFPE_Human_Breast_IDC_outs",                                 574852),
    ("kidney",        "xenium",   "Xenium_V1_hKidney_nondiseased_section_outs",                            97560),
    ("pancreas",      "xenium",   "Xenium_V1_hPancreas_nondiseased_section_outs",                         103901),
    ("skin",          "xenium",   "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs",                       68476),
    ("bone_marrow",   "xenium",   "Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs",           225906),
    ("ovarian",       "xenium",   "Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs",                       247636),
    ("prostate",      "xenium",   "Xenium_Prime_Human_Prostate_FFPE_outs",                                193000),
    ("uterine",       "merscope", "HumanUterineCancerPatient1_region_0",                                  843285),
    ("head_and_neck", "cosmx",    "14H007030716H0216858_down",                                             98220),
]

NODES_PER_BATCH_KEEP = 64        # per-batch reservoir size for per-node surfaces
NUM_CONVS = 10                   # GATv2 layers in novae-human-0
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def process_slide(model, tissue: str, technology: str, slide_name: str) -> dict | None:
    out_dir = PER_SLIDE_DIR / f"{tissue}__{slide_name}"
    if (out_dir / "DONE").exists():
        log(f"  SKIP (already DONE): {tissue}/{slide_name}")
        info = json.load(open(out_dir / "info.json"))
        return info

    out_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = DATA_DIR / tissue / f"{slide_name}.h5ad"
    if not h5ad_path.exists():
        log(f"  ERROR: file not found: {h5ad_path}")
        return None

    log(f"  loading {h5ad_path.name}")
    t0 = time.time()
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        log(f"  ERROR loading: {type(e).__name__}: {e}")
        return None
    log(f"    shape: {adata.shape}, load time: {time.time()-t0:.1f}s")

    try:
        novae.spatial_neighbors(adata)
    except Exception as e:
        log(f"  ERROR in spatial_neighbors: {type(e).__name__}: {e}")
        del adata
        gc.collect()
        return None

    rng = np.random.default_rng(SEED + abs(hash(slide_name)) % (2**31))

    state = {
        "agg": [],
        "convs": [[] for _ in range(NUM_CONVS)],
        "embed": [],
    }

    def hook_aggregator(_mod, _inp, out):
        if isinstance(out, torch.Tensor):
            state["agg"].append(out.detach().cpu().float().numpy().copy())

    def make_conv_hook(i: int):
        def hook(_mod, _inp, out):
            if not isinstance(out, torch.Tensor):
                return
            n = out.shape[0]
            k = min(NODES_PER_BATCH_KEEP, n)
            idx = rng.choice(n, size=k, replace=False)
            sample = out[idx].detach().cpu().float().numpy().copy()
            state["convs"][i].append(sample)
        return hook

    def hook_embedder(_mod, _inp, out):
        # cell_embedder.forward returns the modified PyG Data/Batch object;
        # the per-node embedded features live on `out.x`
        x = getattr(out, "x", None)
        if not isinstance(x, torch.Tensor):
            return
        n = x.shape[0]
        k = min(NODES_PER_BATCH_KEEP, n)
        idx = rng.choice(n, size=k, replace=False)
        sample = x[idx].detach().cpu().float().numpy().copy()
        state["embed"].append(sample)

    handles = [model.encoder.node_aggregation.register_forward_hook(hook_aggregator)]
    for i in range(NUM_CONVS):
        handles.append(model.encoder.gnn.convs[i].register_forward_hook(make_conv_hook(i)))
    handles.append(model.cell_embedder.register_forward_hook(hook_embedder))

    log(f"  encoding ...")
    t0 = time.time()
    try:
        model.compute_representations(adata, zero_shot=True)
    except Exception as e:
        log(f"  ERROR during encode: {type(e).__name__}: {e}")
        for h in handles:
            h.remove()
        del adata
        gc.collect()
        return None
    encode_time = time.time() - t0

    for h in handles:
        h.remove()

    agg = np.concatenate(state["agg"], axis=0)                       # (N_cells, 64)
    convs = [np.concatenate(state["convs"][i], axis=0) for i in range(NUM_CONVS)]  # (n_samp, dim)
    embed = np.concatenate(state["embed"], axis=0)                   # (n_samp, 512)

    # Sanity: aggregator stack should equal obsm['novae_latent']
    nl = np.asarray(adata.obsm.get("novae_latent"))
    sanity_diff = float(np.abs(agg - nl).max()) if nl is not None and nl.shape == agg.shape else float("nan")

    log(
        f"    n_cells={agg.shape[0]:,} | "
        f"agg{tuple(agg.shape)} conv0{tuple(convs[0].shape)} conv9{tuple(convs[9].shape)} "
        f"embed{tuple(embed.shape)} | "
        f"encode={encode_time:.1f}s ({adata.n_obs/encode_time:.0f} cells/s) | "
        f"agg-vs-novae_latent abs diff={sanity_diff:.2e}"
    )

    np.save(out_dir / "aggregator.npy", agg)
    for i in range(NUM_CONVS):
        np.save(out_dir / f"conv_{i}.npy", convs[i])
    np.save(out_dir / "cell_embedder.npy", embed)

    info = {
        "slide_name": slide_name,
        "tissue": tissue,
        "technology": technology,
        "n_cells": int(agg.shape[0]),
        "n_genes_in_panel": int(adata.n_vars),
        "encode_time_s": encode_time,
        "cells_per_sec": adata.n_obs / encode_time,
        "aggregator_shape": list(agg.shape),
        "conv_shapes": [list(c.shape) for c in convs],
        "cell_embedder_shape": list(embed.shape),
        "n_layer_samples": int(convs[0].shape[0]),
        "agg_vs_novae_latent_max_abs_diff": sanity_diff,
    }
    json.dump(info, open(out_dir / "info.json", "w"), indent=2)
    (out_dir / "DONE").touch()

    del adata, agg, convs, embed, state
    gc.collect()
    return info


def consolidate() -> None:
    log("\n=== consolidating per-slide arrays into global files ===")

    aggs: list[np.ndarray] = []
    embeds: list[np.ndarray] = []
    convs_all: list[list[np.ndarray]] = [[] for _ in range(NUM_CONVS)]
    cell_slide_id: list[int] = []
    layer_slide_id: list[int] = []

    manifest = {
        "checkpoint": "novae-human-0",
        "device": DEVICE,
        "nodes_per_batch_keep": NODES_PER_BATCH_KEEP,
        "seed": SEED,
        "slides": [],
    }

    cum_cells = 0
    cum_layer = 0
    sidx = 0
    for tissue, tech, name, _ in SLIDES:
        sdir = PER_SLIDE_DIR / f"{tissue}__{name}"
        if not (sdir / "DONE").exists():
            log(f"  WARNING: not done, skipping in consolidation: {sdir}")
            continue
        info = json.load(open(sdir / "info.json"))
        agg = np.load(sdir / "aggregator.npy")
        embed = np.load(sdir / "cell_embedder.npy")
        convs = [np.load(sdir / f"conv_{i}.npy") for i in range(NUM_CONVS)]

        aggs.append(agg)
        embeds.append(embed)
        for i in range(NUM_CONVS):
            convs_all[i].append(convs[i])
        cell_slide_id.extend([sidx] * agg.shape[0])
        layer_slide_id.extend([sidx] * convs[0].shape[0])

        manifest["slides"].append({
            "slide_idx": sidx,
            "name": info["slide_name"],
            "tissue": info["tissue"],
            "technology": info["technology"],
            "n_cells": info["n_cells"],
            "n_layer_samples": info["n_layer_samples"],
            "cells_per_sec": info["cells_per_sec"],
            "cell_offset_start": cum_cells,
            "cell_offset_end": cum_cells + info["n_cells"],
            "layer_offset_start": cum_layer,
            "layer_offset_end": cum_layer + info["n_layer_samples"],
        })
        cum_cells += info["n_cells"]
        cum_layer += info["n_layer_samples"]
        sidx += 1

    if not aggs:
        log("  nothing consolidated; aborting")
        return

    np.save(OUT_DIR / "aggregator.npy", np.concatenate(aggs, axis=0))
    np.save(OUT_DIR / "cell_embedder.npy", np.concatenate(embeds, axis=0))
    np.save(OUT_DIR / "cell_slide_id.npy", np.asarray(cell_slide_id, dtype=np.int32))
    np.save(OUT_DIR / "layer_slide_id.npy", np.asarray(layer_slide_id, dtype=np.int32))
    for i in range(NUM_CONVS):
        np.save(OUT_DIR / f"conv_{i}.npy", np.concatenate(convs_all[i], axis=0))

    json.dump(manifest, open(OUT_DIR / "manifest.json", "w"), indent=2)

    log(f"  consolidated {sidx} slides")
    log(f"  total cells: {cum_cells:,}")
    log(f"  total layer samples: {cum_layer:,}")


def main() -> None:
    log("=" * 72)
    log(f"Phase 1: activation extraction")
    log(f"  device     : {DEVICE}")
    log(f"  checkpoint : {CKPT_DIR}")
    log(f"  data       : {DATA_DIR}")
    log(f"  output     : {OUT_DIR}")
    log(f"  slides     : {len(SLIDES)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_SLIDE_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading checkpoint")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    t_global = time.time()
    for idx, (tissue, tech, name, est_n) in enumerate(SLIDES):
        log(f"\n[{idx+1}/{len(SLIDES)}] {tissue} / {tech} / {name} (est {est_n:,} cells)")
        process_slide(model, tissue, tech, name)

    log(f"\nAll slides processed. Total wall clock: {(time.time()-t_global)/60:.1f} min")
    consolidate()
    log("DONE")


if __name__ == "__main__":
    main()
