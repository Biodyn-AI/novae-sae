#!/usr/bin/env python
"""Reviewer R2 control: R_ABC triplet-selection sensitivity.

The 10 'triplets' in combinatorial_ablation.parquet share the same
(feat_A, feat_B) = (2107, 2829); only feat_C varies. The 'bootstrap
CI' therefore measures a single pair's synergy profile against
different third features, not a general property of feature triplets.

This script exposes the finding and provides a cleaner sensitivity
decomposition:

  - Per-triplet R_AB is constant at 0.9927 (same pair, same pair-level
    synergy with itself) — this is the near-additive pair behavior.
  - R_AC varies with which C is chosen (range 0.992–1.014).
  - R_BC varies similarly.
  - R_ABC is dominated by which C is chosen, not by variation across
    independent triplets.

Output: reviewer_controls/P_triplet_sensitivity.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ATLAS = Path(__file__).resolve().parent.parent / "atlas" / "novae-human-0"
OUT = ATLAS / "causal" / "reviewer_controls" / "P_triplet_sensitivity.json"


def main() -> None:
    df = pd.read_parquet(ATLAS / "causal" / "combinatorial_ablation.parquet")
    fixed_pair = (int(df["feat_A"].iloc[0]), int(df["feat_B"].iloc[0]))
    unique_a = df["feat_A"].nunique()
    unique_b = df["feat_B"].nunique()
    unique_c = df["feat_C"].nunique()

    summary = {
        "n_triplets": int(len(df)),
        "unique_feat_A": unique_a,
        "unique_feat_B": unique_b,
        "unique_feat_C": unique_c,
        "fixed_pair_AB": list(fixed_pair),
        "independence_across_triplets": False,
        "R_AB_value": float(df["R_AB"].iloc[0]),
        "R_AB_std": float(df["R_AB"].std()),
        "R_ABC_median": float(df["R_ABC"].median()),
        "R_ABC_min": float(df["R_ABC"].min()),
        "R_ABC_max": float(df["R_ABC"].max()),
        "R_ABC_range_from_C_choice": float(df["R_ABC"].max() - df["R_ABC"].min()),
        "conclusion": (
            "The 10 triplets are not 10 independent samples: they share the "
            "same (A, B) pair with only the third feature varying. The "
            "reported bootstrap CI [0.957, 0.966] therefore reflects the "
            "variability of adding a third feature to ONE pair, not a "
            "general property of feature triplets in the SAE. The synergy "
            "claim holds in the narrow sense that the observed pair-plus-C "
            "triplets are near-additive but slightly synergistic; the claim "
            "does NOT generalize to arbitrary triplets from the SAE dictionary "
            "without additional ablation experiments on independent pairs."
        ),
        "needed_followup": (
            "Re-run combinatorial ablation on ≥3 independent (A, B) pairs "
            "with ≥5 C-choices each, to separate 'pair-level synergy' from "
            "'triplet-level synergy' and test generalization."
        ),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
