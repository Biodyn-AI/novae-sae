#!/usr/bin/env python
"""Reviewer control: shuffled-cell-label SAE specificity test.

Reviewer concern: the sae_dissociation control (script 25) compares
annotation rate of trained SAE vs shuffled-cell-order SAE, both at 40–81%.
But high annotation rate with matched cells-to-features is not specificity
— it's base rate (100% in our corpus, see C_random_gene_null).

The correct control: preserve the SAE dictionary, shuffle the feature-
to-cell mapping, and recompute per-feature top-genes. If annotation rate
remains similar but *top-niche / top-Enrichr agreement* drops to chance,
we've isolated the specificity of the atlas.

Protocol:
  1. Load per-feature top cells + cell tissue/niche labels.
  2. For each feature, compute top-niche match and top-Enrichr term.
  3. Shuffle feature-to-cell assignments; re-derive top-genes from
     shuffled cells; compute top-niche match and top-Enrichr term.
  4. Report specificity (top Enrichr term's cell-type matches top-niche
     cell-type) for real vs shuffle.

Output: atlas/novae-human-0/causal/reviewer_controls/M_specificity.json
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
ATLAS = ROOT / "atlas" / "novae-human-0"
OUT_DIR = ATLAS / "causal" / "reviewer_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "34_shuffled_specificity.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("Shuffled-label specificity test")

    top_genes = pd.read_parquet(ATLAS / "bio" / "aggregator_top_genes.parquet")
    enr = pd.read_parquet(ATLAS / "bio" / "aggregator_enrichment.parquet")
    atlas = pd.read_parquet(ATLAS / "feature_atlas.parquet") \
        if (ATLAS / "feature_atlas.parquet").exists() else None

    # Per-feature top domain (niche)
    if atlas is not None:
        cols_candidates = [c for c in atlas.columns
                           if "domain" in c.lower() and "top" in c.lower()]
        log(f"  candidate domain cols: {cols_candidates}")

    # Per-feature best Enrichr term
    best = enr.sort_values("fdr").groupby("feature_idx").head(1)
    best = best[["feature_idx", "library", "top_term", "fdr"]]
    log(f"  best-Enrichr rows: {len(best)}")

    # "Matches niche" is ill-defined without a shared taxonomy. We use a
    # simple proxy: does the top Enrichr cell-type library (CellMarker or
    # PanglaoDB) contain a substring that overlaps with the feature's top
    # tissue/niche label?
    if atlas is None:
        log("atlas not present; producing scaffold")
        (OUT_DIR / "M_specificity.json").write_text(
            json.dumps({"status": "scaffold", "reason": "feature_atlas.parquet missing"},
                       indent=2)
        )
        return

    # Try to locate tissue column
    tissue_col = None
    for c in atlas.columns:
        if "tissue" in c.lower() and "top" in c.lower():
            tissue_col = c
            break
    if tissue_col is None:
        tissue_col = next((c for c in atlas.columns if "tissue" in c.lower()), None)

    if tissue_col is None:
        log("no tissue column in atlas; scaffold")
        (OUT_DIR / "M_specificity.json").write_text(
            json.dumps({"status": "scaffold", "reason": "no tissue col"}, indent=2)
        )
        return

    log(f"  tissue col: {tissue_col}")
    atl = atlas[["feature_idx", tissue_col]].copy()
    merged = best.merge(atl, on="feature_idx", how="left")

    def matches(term: str, tissue: str) -> bool:
        if not isinstance(term, str) or not isinstance(tissue, str):
            return False
        t = tissue.lower()
        s = term.lower()
        hits = [
            ("tonsil", "tonsil"),
            ("lymph", "lymph"),
            ("lung", "lung"),
            ("liver", "hepat"),
            ("liver", "liver"),
            ("colon", "colon"),
            ("colon", "intestin"),
            ("breast", "breast"),
            ("breast", "mammary"),
            ("kidney", "kidney"),
            ("kidney", "renal"),
            ("pancreas", "pancrea"),
            ("skin", "skin"),
            ("skin", "melano"),
            ("ovarian", "ovar"),
            ("prostate", "prostat"),
            ("uterine", "uter"),
            ("bone_marrow", "marrow"),
            ("brain", "brain"),
            ("brain", "neuro"),
            ("head_and_neck", "epithel"),
        ]
        for tk, sk in hits:
            if tk in t and sk in s:
                return True
        return False

    real_match = merged.apply(lambda r: matches(r["top_term"], str(r[tissue_col])),
                              axis=1).mean()
    log(f"real tissue-term match rate: {real_match*100:.1f}%")

    # Shuffle: permute tissue column across features
    rng = np.random.default_rng(2026)
    shuf_rates = []
    for _ in range(50):
        shuf_tissues = rng.permutation(merged[tissue_col].values)
        tmp = merged.copy()
        tmp[tissue_col] = shuf_tissues
        r = tmp.apply(lambda r: matches(r["top_term"], str(r[tissue_col])), axis=1).mean()
        shuf_rates.append(r)
    shuf_mean = float(np.mean(shuf_rates))
    shuf_ci = (float(np.percentile(shuf_rates, 2.5)),
               float(np.percentile(shuf_rates, 97.5)))
    log(f"shuffled rate: {shuf_mean*100:.2f}% (95% CI {shuf_ci[0]*100:.2f}--{shuf_ci[1]*100:.2f}%)")

    summary = {
        "n_features": int(merged["feature_idx"].nunique()),
        "real_tissue_term_match_rate": float(real_match),
        "shuffled_null_rate": shuf_mean,
        "shuffled_null_95CI": shuf_ci,
        "excess_over_null": float(real_match - shuf_mean),
        "interpretation": (
            "The excess above the shuffle null is the *specificity* signal. "
            "Annotation rate alone (100% in this corpus) is a ceiling artifact; "
            "specificity is what the paper should report alongside."
        ),
    }
    (OUT_DIR / "M_specificity.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
