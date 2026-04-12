#!/usr/bin/env python
"""§4.7/6 — hierarchical null-model pooling.

Combines the four confound-control outputs (§4.7 items 1–4) into a single
per-feature survival scorecard:

  A) slide_shuffle        — from confounds_effect_size.parquet
                            (slide_max_ratio >= 2.0)
  B) tech_residualization — from confound_survival.parquet
                            (survives_tech)
  C) l20_residualization  — from confound_survival.parquet
                            (survives_l20)
  D) graph_rewire         — from graph_ablation_v2.parquet
                            (dep_self_loop_norm > 0.2 OR dep_random_rewire > 0.2)
                            i.e. the feature is at least somewhat context-
                            dependent under a graph-scale-fair protocol.

Output per feature:
  - survival_depth: count of 0..4 controls passed
  - survives_hierarchical: True if all four pass
  - verdict at each level

Plus an aggregate "manifesto 10% calibration" statistic:
  fraction of features passing ALL four controls.

Output: atlas/novae-human-0/causal/hierarchical_null.parquet +
        hierarchical_null.summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "13_hierarchical_null.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("§4.7/6 hierarchical null-model pooling")

    es = pd.read_parquet(OUT_DIR / "confounds_effect_size.parquet").set_index("feature_idx")
    log(f"  confounds_effect_size: {es.shape}")
    surv = pd.read_parquet(OUT_DIR / "confound_survival.parquet").set_index("feature_idx")
    log(f"  confound_survival: {surv.shape}")

    ga_v2_path = OUT_DIR / "graph_ablation_v2.parquet"
    if ga_v2_path.exists():
        ga2 = pd.read_parquet(ga_v2_path).set_index("feature_idx")
        log(f"  graph_ablation_v2: {ga2.shape}")
    else:
        ga2 = None
        log("  graph_ablation_v2: NOT FOUND — level (D) will be NaN")

    rows = []
    n_features = 2048
    for fid in range(n_features):
        row = {"feature_idx": fid}

        # Level A: slide_shuffle via effect-size max ratio
        if fid in es.index:
            slide_ratio = es.loc[fid].get("slide_max_ratio", np.nan)
            row["level_a_slide"] = bool(pd.notna(slide_ratio) and slide_ratio >= 2.0)
            row["slide_max_ratio"] = float(slide_ratio) if pd.notna(slide_ratio) else np.nan
        else:
            row["level_a_slide"] = False
            row["slide_max_ratio"] = np.nan

        # Level B: tech_residualization
        if fid in surv.index:
            row["level_b_tech_resid"] = bool(surv.loc[fid].get("survives_tech", False))
        else:
            row["level_b_tech_resid"] = False

        # Level C: l20_residualization
        if fid in surv.index:
            row["level_c_l20_resid"] = bool(surv.loc[fid].get("survives_l20", False))
        else:
            row["level_c_l20_resid"] = False

        # Level D: graph_rewire — dep > 0.2 on either normalized self-loop
        # or random rewire (whichever is more generous; require at least
        # one of the graph-fair regimes to show dependence).
        if ga2 is not None and fid in ga2.index:
            r = ga2.loc[fid]
            dep_norm = r.get("dep_self_loop_norm", np.nan)
            dep_rewire = r.get("dep_random_rewire", np.nan)
            dep_best = max(
                dep_norm if pd.notna(dep_norm) else -np.inf,
                dep_rewire if pd.notna(dep_rewire) else -np.inf,
            )
            row["level_d_graph_rewire"] = bool(dep_best > 0.2)
            row["dep_self_loop_norm"] = float(dep_norm) if pd.notna(dep_norm) else np.nan
            row["dep_random_rewire"] = float(dep_rewire) if pd.notna(dep_rewire) else np.nan
        else:
            row["level_d_graph_rewire"] = False
            row["dep_self_loop_norm"] = np.nan
            row["dep_random_rewire"] = np.nan

        # Cumulative survival levels (hierarchical)
        passes = [
            row["level_a_slide"],
            row["level_b_tech_resid"],
            row["level_c_l20_resid"],
            row["level_d_graph_rewire"],
        ]
        row["survival_depth"] = int(sum(passes))
        row["survives_all_4"] = bool(all(passes))
        row["survives_at_least_3"] = bool(sum(passes) >= 3)
        row["survives_at_least_2"] = bool(sum(passes) >= 2)
        rows.append(row)

    df = pd.DataFrame(rows)

    out_path = OUT_DIR / "hierarchical_null.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {out_path} ({len(df)} rows)")

    # Count breakdown
    depth_counts = df["survival_depth"].value_counts().sort_index().to_dict()
    summary = {
        "n_features": int(len(df)),
        "levels": {
            "A_slide_shuffle": "effect-size slide_max_ratio >= 2.0",
            "B_tech_residualization": "confound_survival.survives_tech",
            "C_l20_residualization": "confound_survival.survives_l20",
            "D_graph_rewire": "graph_ablation_v2 max(dep_self_loop_norm, dep_random_rewire) > 0.2",
        },
        "per_level_count": {
            "A_slide": int(df["level_a_slide"].sum()),
            "B_tech_resid": int(df["level_b_tech_resid"].sum()),
            "C_l20_resid": int(df["level_c_l20_resid"].sum()),
            "D_graph_rewire": int(df["level_d_graph_rewire"].sum()),
        },
        "survival_depth_histogram": {str(k): int(v) for k, v in depth_counts.items()},
        "n_survives_all_4": int(df["survives_all_4"].sum()),
        "fraction_survives_all_4": float(df["survives_all_4"].sum() / len(df)),
        "n_survives_at_least_3": int(df["survives_at_least_3"].sum()),
        "fraction_survives_at_least_3": float(df["survives_at_least_3"].sum() / len(df)),
        "n_survives_at_least_2": int(df["survives_at_least_2"].sum()),
        "fraction_survives_at_least_2": float(df["survives_at_least_2"].sum() / len(df)),
    }
    json.dump(summary, open(OUT_DIR / "hierarchical_null.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
