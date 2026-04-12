#!/usr/bin/env python
"""Phase 3d — per-slide assign_domains.

`assign_domains` requires `obs.novae_leaves`, which is only populated by a full
`compute_representations` call (the leaves are the SwAV prototype assignments
that depend on the slide-specific top-k prototype queue, so we cannot shortcut
this by just setting `obsm["novae_latent"]` from the cached aggregator).

This script therefore re-runs `compute_representations` for each of the 15
curated slides and then `assign_domains(level=7)`. Per-cell domain labels are
saved next to the cached activations:

  activations/novae-human-0/per_slide/<tissue>__<slide>/domains_level7.npy

Wall-clock on M2 Pro: ~2 hours total (matches Phase 1 throughput).
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

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ACT_DIR = ROOT / "activations" / "novae-human-0"
LOG_PATH = ROOT / "logs" / "03d_assign_domains.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DOMAIN_LEVELS = [7, 12, 20]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("Phase 3d: per-slide assign_domains")
    log(f"  device: {DEVICE}")

    manifest = json.load(open(ACT_DIR / "manifest.json"))

    log(f"loading model")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    t_global = time.time()
    for sid, slide in enumerate(manifest["slides"]):
        sdir = ACT_DIR / "per_slide" / f"{slide['tissue']}__{slide['name']}"
        out_files = {lvl: sdir / f"domains_level{lvl}.npy" for lvl in DOMAIN_LEVELS}
        out_leaves = sdir / "novae_leaves.npy"
        if all(p.exists() for p in out_files.values()) and out_leaves.exists():
            log(f"[{sid+1}/{len(manifest['slides'])}] {slide['name']} SKIP (all domain levels done)")
            continue

        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"\n[{sid+1}/{len(manifest['slides'])}] {slide['tissue']}/{slide['name']}")
        log(f"  loading {h5ad.name}")
        try:
            adata = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"  ERROR load: {type(e).__name__}: {e}")
            continue

        try:
            novae.spatial_neighbors(adata)
        except Exception as e:
            log(f"  ERROR neighbors: {type(e).__name__}: {e}")
            del adata
            continue

        log(f"  compute_representations ({adata.n_obs:,} cells)")
        t0 = time.time()
        try:
            model.compute_representations(adata, zero_shot=True)
        except Exception as e:
            log(f"  ERROR compute_representations: {type(e).__name__}: {e}")
            del adata
            continue
        log(f"    encoded in {time.time()-t0:.0f}s ({adata.n_obs/(time.time()-t0):.0f} cells/s)")

        for level in DOMAIN_LEVELS:
            try:
                model.assign_domains(adata, level=level)
                col = f"novae_domains_{level}"
                if col in adata.obs.columns:
                    domains = adata.obs[col].astype(str).values
                    np.save(out_files[level], domains)
                    n_unique = len(set(domains))
                    log(f"    level {level}: {n_unique} unique domains")
            except Exception as e:
                log(f"    ERROR level {level}: {type(e).__name__}: {e}")

        if "novae_leaves" in adata.obs.columns:
            np.save(out_leaves, adata.obs["novae_leaves"].astype(str).values)
        if "novae_sid" in adata.obs.columns:
            np.save(sdir / "novae_sid.npy", adata.obs["novae_sid"].astype(str).values)

        del adata
        gc.collect()

    log(f"\nDONE in {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
