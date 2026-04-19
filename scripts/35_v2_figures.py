#!/usr/bin/env python
"""Generate v2 figures for the paper revision.

Figure F: Random-direction null scatter
  - x: real prototype reassignment rate per feature
  - y: null mean with 95% CI
  - coloring: above-null / indistinguishable / below-null

Figure H: Self-loop vs feature-perm dependency scatter
  - x: self-loop dep (from v2 graph_ablation.parquet)
  - y: feature-perm dep (from H_feature_perm_ablation.parquet)
  - quadrants at 0.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
ATLAS = ROOT / "atlas" / "novae-human-0"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_random_direction_null() -> None:
    df = pd.read_parquet(ATLAS / "causal" / "reviewer_controls" / "F_random_direction_null.parquet")
    real = df["real_proto_reassign_rate"].values
    null_mean = df["rand_proto_reassign_mean"].values
    null_lo = df["rand_proto_reassign_CI_lo"].values
    null_hi = df["rand_proto_reassign_CI_hi"].values
    above = real > null_hi
    below = real < null_lo
    indist = ~(above | below)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    # Diagonal: real == null_mean
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.4, label="real = null mean")

    # Plot null CI bars on y for each feature (horizontal orientation: x=real, y=null_mean)
    # Alternative: error bars on y showing null mean ± CI at x=real.
    for i, (r, m, lo, hi) in enumerate(zip(real, null_mean, null_lo, null_hi)):
        ax.plot([r, r], [lo, hi], color="#AAAAAA", alpha=0.5, lw=1, zorder=1)

    ax.scatter(real[indist], null_mean[indist], color="#9CA3AF", s=42,
               label=f"indistinguishable (n={int(indist.sum())})",
               edgecolor="white", lw=0.6, zorder=3)
    ax.scatter(real[above], null_mean[above], color="#059669", s=56,
               label=f"real > null CI (n={int(above.sum())})",
               edgecolor="white", lw=0.6, zorder=4)
    ax.scatter(real[below], null_mean[below], color="#E11D48", s=56,
               label=f"real < null CI (n={int(below.sum())})",
               edgecolor="white", lw=0.6, zorder=4, marker="v")
    ax.set_xlabel("Real prototype reassignment rate (feature ablation)")
    ax.set_ylabel("Null mean reassignment rate (random matched-magnitude direction)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_title("Matched-magnitude random-direction null (50 features)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "fig_v2_random_direction_null.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig_v2_random_direction_null.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_selfloop_vs_featureperm() -> None:
    fp = pd.read_parquet(ATLAS / "causal" / "reviewer_controls" / "H_feature_perm_vs_selfloop_v2.parquet")
    sl = fp["dep_self_loop"].to_numpy()
    fpv = fp["feature_perm_dependency"].to_numpy()
    # Clip extremes for visualization
    sl_clip = np.clip(sl, -2, 2)
    fp_clip = np.clip(fpv, -2, 2)
    # Quadrants at 0.5
    both_high = (sl > 0.5) & (fpv > 0.5)
    only_sl = (sl > 0.5) & (fpv <= 0.5)
    only_fp = (sl <= 0.5) & (fpv > 0.5)
    neither = (sl <= 0.5) & (fpv <= 0.5)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.axhline(0.5, color="black", lw=0.8, alpha=0.6)
    ax.axvline(0.5, color="black", lw=0.8, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.5, alpha=0.3)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.3)

    kw = dict(alpha=0.55, edgecolor="white", lw=0.3, s=14)
    ax.scatter(sl_clip[neither], fp_clip[neither], color="#9CA3AF",
               label=f"neither (n={int(neither.sum())})", **kw)
    ax.scatter(sl_clip[only_sl], fp_clip[only_sl], color="#F59E0B",
               label=f"self-loop only (n={int(only_sl.sum())})", **kw)
    ax.scatter(sl_clip[only_fp], fp_clip[only_fp], color="#3B82F6",
               label=f"feature-perm only (n={int(only_fp.sum())})", **kw)
    ax.scatter(sl_clip[both_high], fp_clip[both_high], color="#059669",
               label=f"robust: both > 0.5 (n={int(both_high.sum())})", **kw)

    ax.set_xlabel("Self-loop dependency (v1 / H8)")
    ax.set_ylabel("Feature-permutation dependency (v2)")
    ax.set_xlim(-2.05, 1.15)
    ax.set_ylim(-2.05, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.set_title("Context-dependence under two controls\n"
                 "(self-loop: architectural shift; feature-perm: content shuffle)",
                 fontsize=11)

    # Annotation boxes
    ax.text(0.95, 0.98, f"robust = {int(both_high.sum())}\n"
                        f"self-loop-only = {int(only_sl.sum())}\n"
                        f"feature-perm-only = {int(only_fp.sum())}\n"
                        f"neither = {int(neither.sum())}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="#F9FAFB", edgecolor="#D1D5DB"))

    fig.tight_layout()
    out = FIG_DIR / "fig_v2_selfloop_vs_featureperm.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig_v2_selfloop_vs_featureperm.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    fig_random_direction_null()
    fig_selfloop_vs_featureperm()


if __name__ == "__main__":
    main()
