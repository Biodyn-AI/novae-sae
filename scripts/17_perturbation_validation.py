#!/usr/bin/env python
"""§4.5/5 — spatial perturbation validation (Perturb-map).

Uses the Perturb-map dataset (Dhainaut, Rose et al., Cell 2022;
GSE193460): 4 mouse lung sections with 35 CRISPR targets profiled via
10x Visium.

IMPORTANT: this dataset is
  (a) MOUSE — requires novae-mouse-0, not novae-human-0;
  (b) VISIUM spot-level (~55 um spots, not single-cell);
  (c) small (~6K spots total).

This script currently PREPARES the data and DOCUMENTS the pipeline.
The actual Novae processing is blocked until novae-mouse-0 is downloaded.

When novae-mouse-0 is available:
  1. Load each Visium section as AnnData via scanpy.read_visium()
  2. Merge spot_annotation.csv into .obs (perturbation labels)
  3. Run novae.spatial_neighbors() + model.compute_representations()
  4. SAE-encode the aggregator output
  5. For each CRISPR target gene: compare SAE feature activations on
     target-perturbed spots vs control spots (Wilcoxon + BH)
  6. Report: how many targets have a target-specific SAE feature response?

Output: atlas/novae-mouse-0/causal/perturbation_validation.parquet
        (blocked until novae-mouse-0 is available)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
PERTURB_DIR = ROOT / "datasets" / "perturbation" / "perturb_map"
LOG_PATH = ROOT / "logs" / "17_perturbation_validation.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("§4.5/5 perturbation validation preparation (Perturb-map)")

    # Inventory downloaded files
    samples = ["KP_1", "KP_2", "KP_3", "KP_4"]
    for s in samples:
        prefix = PERTURB_DIR / s
        filt = list(PERTURB_DIR.glob(f"*{s}*filtered_feature_bc_matrix.h5"))
        annot = list(PERTURB_DIR.glob(f"*{s}*spot_annotation.csv"))
        pos = list(PERTURB_DIR.glob(f"*{s}*tissue_positions*"))
        log(f"  {s}: matrix={len(filt)} annotation={len(annot)} positions={len(pos)}")

    # Parse spot annotations to understand perturbation structure
    all_phenotypes = set()
    total_spots = 0
    for s in samples:
        annots = list(PERTURB_DIR.glob(f"*{s}*spot_annotation.csv"))
        if not annots:
            continue
        df = pd.read_csv(annots[0])
        log(f"  {s}: {len(df)} spots, columns={list(df.columns)[:10]}")
        if "phenotypes" in df.columns:
            phenos = df["phenotypes"].dropna().unique()
            all_phenotypes.update(phenos)
            total_spots += len(df)
            log(f"    phenotypes: {list(phenos)[:15]}")

    log(f"\ntotal spots: {total_spots}")
    log(f"unique phenotype labels: {len(all_phenotypes)}")

    # Identify CRISPR target genes (phenotypes that look like gene_N patterns)
    import re
    targets = set()
    for p in all_phenotypes:
        m = re.match(r"^([A-Za-z0-9]+)_\d+$", str(p))
        if m:
            targets.add(m.group(1))
    log(f"putative CRISPR target genes: {len(targets)}")
    log(f"  {sorted(targets)[:20]}")

    summary = {
        "dataset": "Perturb-map (Dhainaut et al., Cell 2022)",
        "accession": "GSE193460",
        "species": "mouse",
        "technology": "Visium (spot-level, ~55 um)",
        "n_samples": len(samples),
        "total_spots": total_spots,
        "n_perturbation_targets": len(targets),
        "targets": sorted(targets),
        "all_phenotype_labels": sorted(str(p) for p in all_phenotypes),
        "status": "DATA_READY_BLOCKED_ON_MOUSE_CHECKPOINT",
        "blocker": "novae-mouse-0 checkpoint not downloaded; mouse SAE not trained",
        "next_steps": [
            "1. Download novae-mouse-0: novae.Novae.from_pretrained('MICS-Lab/novae-mouse-0')",
            "2. Extract mouse activations on mouse Visium slides",
            "3. Train mouse aggregator SAE",
            "4. Run this script with --run flag to process Perturb-map through Novae + SAE",
            "5. Wilcoxon + BH test per target gene: SAE feature response",
        ],
    }

    out_dir = ROOT / "atlas" / "novae-human-0" / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "perturbation_validation_prep.json", "w"), indent=2)
    log(f"\nwrote {out_dir / 'perturbation_validation_prep.json'}")
    log(json.dumps(summary, indent=2))
    log("DONE (preparation only; actual validation blocked on novae-mouse-0)")


if __name__ == "__main__":
    main()
