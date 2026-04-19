#!/usr/bin/env python
"""Reviewer control: composition-regressed Perturb-map test.

Reviewer concern: Visium spots (~55μm) contain tens of cells, so
"SAE feature responds to perturbation X" may be driven by per-spot
cell-type composition shifts rather than the perturbation itself.

Protocol:
  1. Load Perturb-map mouse lung Visium data.
  2. Estimate per-spot cell-type composition (simple correlation-based
     deconvolution against a marker-gene panel).
  3. For each SAE feature × perturbation label, regress the feature
     activation on (i) perturbation indicator and (ii) composition
     covariates; report the partial Cohen's d after controlling for
     composition.
  4. Compare associations with and without composition control.

Output: atlas/novae-human-0/causal/reviewer_controls/L_perturbmap_composition.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal" / "reviewer_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "33_perturbmap_composition.log"

# Look for existing Perturb-map prep output
PREP_JSON = (ROOT / "atlas" / "novae-human-0" / "causal"
             / "perturbation_validation_prep.json")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def residualize(y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """OLS residualization: return y - C @ beta_hat."""
    beta, *_ = np.linalg.lstsq(C, y, rcond=None)
    return y - C @ beta


def main() -> None:
    log("=" * 72)
    log("Perturb-map composition-regressed association test")
    if not PREP_JSON.exists():
        log(f"PREP JSON not found: {PREP_JSON}")
        log("Rerun script 17 first to generate perturbation_validation_prep.json")
        return
    prep = json.load(open(PREP_JSON))
    log(f"prep keys: {list(prep.keys())}")

    # This control requires access to the raw Perturb-map h5ad + SAE-encoded
    # feature activations per spot. If the prep artifacts don't include
    # them, we stage a scaffold script that can be filled in when the
    # Visium file is available.
    pert_h5ad = prep.get("h5ad_path")
    feat_npy = prep.get("feature_activations_npy")
    label_col = prep.get("perturbation_label_col", "perturbation")

    if pert_h5ad is None or feat_npy is None:
        log("prep JSON missing h5ad_path/feature_activations_npy — scaffold only.")
        (OUT_DIR / "L_perturbmap_composition.summary.json").write_text(
            json.dumps({
                "status": "scaffold",
                "message": (
                    "This script requires the Perturb-map h5ad and "
                    "per-spot SAE feature activations. Populate "
                    "h5ad_path and feature_activations_npy in "
                    "perturbation_validation_prep.json and re-run."
                ),
            }, indent=2)
        )
        return

    a = ad.read_h5ad(pert_h5ad)
    feats = np.load(feat_npy)  # (n_spots, n_features)
    assert feats.shape[0] == a.n_obs
    log(f"{a.n_obs} spots, {feats.shape[1]} features")

    # Composition estimate: mean log1p expression of marker genes per
    # canonical cell type; use as regressor block.
    markers = {
        "macrophage": ["Cd68", "Csf1r", "Adgre1"],
        "tcell": ["Cd3d", "Cd3e", "Cd3g"],
        "bcell": ["Cd19", "Cd79a", "Ms4a1"],
        "fibroblast": ["Col1a1", "Col3a1", "Acta2"],
        "endothelial": ["Pecam1", "Cdh5", "Vwf"],
        "epithelial": ["Krt18", "Krt8", "Epcam"],
        "neutrophil": ["S100a8", "S100a9", "Ly6g"],
        "tumor": ["Kras", "Trp53"],
    }
    comp_cols = []
    comp_vals = []
    gene_list = [g for g in a.var_names]
    gene_index = {g.lower(): i for i, g in enumerate(gene_list)}
    X = a.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    for ct, genes in markers.items():
        idxs = [gene_index[g.lower()] for g in genes if g.lower() in gene_index]
        if not idxs:
            continue
        comp_cols.append(ct)
        comp_vals.append(X[:, idxs].mean(axis=1))
    C = np.stack(comp_vals, axis=1)  # (n_spots, n_markers)
    C = np.hstack([np.ones((C.shape[0], 1), dtype=C.dtype), C])  # intercept

    labels = a.obs[label_col].astype(str).values if label_col in a.obs else None
    if labels is None:
        log(f"no label column {label_col}; abort")
        return

    uniq = sorted(set(labels))
    rows = []
    for lbl in uniq:
        mask_pos = labels == lbl
        if mask_pos.sum() < 5 or (~mask_pos).sum() < 5:
            continue
        for fid in range(feats.shape[1]):
            y = feats[:, fid]
            if np.var(y) < 1e-10:
                continue
            y_res = residualize(y, C)
            try:
                u1, p1 = mannwhitneyu(y[mask_pos], y[~mask_pos])
                u2, p2 = mannwhitneyu(y_res[mask_pos], y_res[~mask_pos])
            except Exception:
                continue
            # Effect sizes
            def d(a, b):
                pooled = np.sqrt((a.var() + b.var()) / 2 + 1e-12)
                return float((a.mean() - b.mean()) / pooled)
            rows.append({
                "label": lbl,
                "feature_idx": int(fid),
                "cohen_d_raw": d(y[mask_pos], y[~mask_pos]),
                "cohen_d_composition_regressed": d(y_res[mask_pos], y_res[~mask_pos]),
                "p_raw": float(p1),
                "p_residualized": float(p2),
            })
    df = pd.DataFrame(rows)
    # BH correction on residualized p
    from statsmodels.stats.multitest import multipletests
    _, df["fdr_resid"], _, _ = multipletests(df["p_residualized"].fillna(1.0), method="fdr_bh")
    _, df["fdr_raw"], _, _ = multipletests(df["p_raw"].fillna(1.0), method="fdr_bh")
    df.to_parquet(OUT_DIR / "L_perturbmap_composition.parquet", index=False)
    summary = {
        "n_rows": len(df),
        "n_sig_raw": int((df["fdr_raw"] < 0.05).sum()),
        "n_sig_residualized": int((df["fdr_resid"] < 0.05).sum()),
        "attenuation": float(
            1 - (df["fdr_resid"] < 0.05).sum() / max((df["fdr_raw"] < 0.05).sum(), 1)
        ),
        "median_abs_d_drop": float(
            df["cohen_d_raw"].abs().median() - df["cohen_d_composition_regressed"].abs().median()
        ),
    }
    (OUT_DIR / "L_perturbmap_composition.summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
