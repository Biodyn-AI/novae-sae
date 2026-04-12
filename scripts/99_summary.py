#!/usr/bin/env python
"""Walks every output produced by Phases 1–4 and prints a single human-readable
summary table to stdout. Used as the final reporting step at the end of an
end-to-end pipeline run."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ATLAS_DIR = ROOT / "atlas" / "novae-human-0"


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    print("=" * 72)
    print("NOVAE MECHINTERP — PIPELINE SUMMARY (novae-human-0)")
    print("=" * 72)

    # ---- Phase 1: extraction ----
    section("Phase 1 — activation extraction")
    manifest_path = ACT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        slides = manifest["slides"]
        n_slides = len(slides)
        n_cells = sum(s["n_cells"] for s in slides)
        n_layer_samples = sum(s["n_layer_samples"] for s in slides)
        print(f"  slides:        {n_slides}")
        print(f"  total cells:   {n_cells:,}")
        print(f"  layer samples: {n_layer_samples:,}")
        tissues = sorted(set(s["tissue"] for s in slides))
        techs = sorted(set(s["technology"] for s in slides))
        print(f"  tissues ({len(tissues)}):    {', '.join(tissues)}")
        print(f"  technologies:  {', '.join(techs)}")
    else:
        print(f"  MISSING: {manifest_path}")

    # ---- Phase 2: SAEs ----
    section("Phase 2 — SAE training")
    sae_summary_path = SAE_DIR / "summary.json"
    if sae_summary_path.exists():
        sae_summary = json.load(open(sae_summary_path))
        rows = sae_summary["summaries"]
        print(f"  {'name':18s} {'d':>5s} {'F':>6s} {'k':>4s} {'val_mse':>9s} {'var_exp':>8s} {'alive':>15s}")
        for s in rows:
            print(
                f"  {s['name']:18s} {s['d_in']:>5d} {s['n_features']:>6d} {s['k']:>4d} "
                f"{s['val_mse_final']:>9.4f} {s['var_explained_final']:>8.3f} "
                f"{s['alive_final']:>5d}/{s['n_features']:<5d} ({s['alive_fraction']:.0%})"
            )
    else:
        print(f"  MISSING: {sae_summary_path}")

    # ---- Phase 3: characterization ----
    section("Phase 3 — feature characterization")
    char_summary_path = ATLAS_DIR / "summary.json"
    if char_summary_path.exists():
        char = json.load(open(char_summary_path))
        print(f"  {'name':18s} {'alive':>10s} {'var_exp':>8s} {'tech_conf':>10s} {'non_aligned':>12s}")
        sup_by_name = {s["name"]: s for s in char.get("superposition", [])}
        for s in char.get("surfaces", []):
            sup = sup_by_name.get(s["name"], {})
            print(
                f"  {s['name']:18s} "
                f"{s['n_alive']:>5d}/{s['n_features']:<4d} "
                f"{s['var_explained_full']:>8.3f} "
                f"{s['tech_confounded_count']:>5d} ({s['tech_confounded_fraction']:.1%}) "
                f"{sup.get('fraction_non_aligned', float('nan')):>12.3f}"
            )
    else:
        print(f"  MISSING: {char_summary_path}")

    mod_summary_path = ATLAS_DIR / "modules_summary.json"
    if mod_summary_path.exists():
        print()
        print("  Module discovery (Phase 3b):")
        mods = json.load(open(mod_summary_path))
        print(f"    {'name':18s} {'alive':>8s} {'modules':>8s} {'PMI thr':>10s}")
        for s in mods.get("surfaces", []):
            print(
                f"    {s['name']:18s} {s['n_alive']:>8d} {s['n_modules']:>8d} {s['pmi_threshold']:>10.3f}"
            )

    # ---- Phase 4: causal validation ----
    section("Phase 4 — causal validation")
    causal_summary_path = ATLAS_DIR / "causal" / "summary.json"
    if causal_summary_path.exists():
        cs = json.load(open(causal_summary_path))
        print(f"  features tested: {cs.get('n_features_tested')}")
        print(f"  slides:          {cs.get('slides')}")
        ab = cs.get("ablation_stats", {})
        if ab.get("mean_effect_score") is not None:
            print(f"  ablation effect (1 - mean cosine to baseline):")
            print(f"    mean   = {ab['mean_effect_score']:.4f}")
            print(f"    median = {ab['median_effect_score']:.4f}")
        mi = cs.get("morans_i_stats", {})
        if mi.get("mean") is not None:
            print(f"  Moran's I (spatial coherence) per feature × slide:")
            print(f"    mean   = {mi['mean']:.3f}")
            print(f"    median = {mi['median']:.3f}")
            print(f"    n>0.1  = {mi['n_significant_positive']}")
    else:
        print(f"  MISSING: {causal_summary_path}")

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
