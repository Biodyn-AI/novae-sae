#!/usr/bin/env python
"""§4.7 confound suite for the aggregator SAE.

Applies four controls per feature and emits a survival table:

1. **Slide-label shuffle**: are the feature's top cells really concentrated
   on a single slide/tissue beyond what random labeling would give?
   Implemented as a max-class-ratio test vs. the corpus baseline (already
   done in `09b_effect_size_confounds.py`; the result is merged in).

2. **Technology residualization**: subtract the per-technology mean of the
   feature's activation from every cell; re-select top cells on the
   residual; recompute the domain-l7 top-class fraction. If the niche
   signal vanishes, the feature was tech-driven.

3. **Cell-type-like residualization** using domain-l20 labels (fine-grained
   Leiden clusters, ~92 classes globally; these are the best proxy for
   cell-type in the absence of true per-cell annotations). Same recipe as
   tech residualization but with l20 class means. If the l7 (coarser
   niche) signal vanishes after l20 residualization, the feature is just
   a cell-type indicator.

4. **Degree-preserving graph rewire**: deferred to
   `10b_graph_ablation_v2.py` because it requires re-running
   `model.compute_representations` under a rewired graph.

Per-feature survival flags:
- `survives_slide`: `slide_max_ratio >= 2.0` (from 09b) i.e. top cells
  concentrate on a single slide ≥ 2× baseline share.
- `survives_tech`: after tech residualization, the feature's l7 top-class
  fraction is ≥ 50% of its pre-residualization value.
- `survives_l20`: same but residualizing l20.
- `survives_all`: all three.

Output: `atlas/novae-human-0/causal/confound_survival.parquet`.

Produces the H2 verdict (<20% of raw positives survive the strict
confound suite).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "12_confound_suite.log"

TOP_FRAC = 0.001
MIN_TOP = 50


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_global_domains(level: int) -> np.ndarray:
    """Stitch per-slide domains into a global (n_cells_global,) array."""
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    chunks = []
    for s in manifest["slides"]:
        path = ACT_DIR / "per_slide" / f"{s['tissue']}__{s['name']}" / f"domains_level{level}.npy"
        arr = np.load(path, allow_pickle=True).astype(object)
        assert arr.shape[0] == s["n_cells"], (s["name"], arr.shape, s["n_cells"])
        chunks.append(arr)
    return np.concatenate(chunks)


def encode_sparse(agg: np.ndarray, sae: TopKSAE, k_sae: int, chunk: int = 32768) -> sparse.csc_matrix:
    n_cells = agg.shape[0]
    rows = np.empty(n_cells * k_sae, dtype=np.int64)
    cols = np.empty(n_cells * k_sae, dtype=np.int32)
    data = np.empty(n_cells * k_sae, dtype=np.float32)
    write = 0
    with torch.no_grad():
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            xb = torch.tensor(agg[start:end], dtype=torch.float32)
            z, idx = sae.encode(xb)
            B = end - start
            vals = z.gather(-1, idx).cpu().numpy()
            idx_np = idx.cpu().numpy().astype(np.int32)
            n = B * k_sae
            rows[write:write + n] = np.repeat(np.arange(start, end, dtype=np.int64), k_sae)
            cols[write:write + n] = idx_np.ravel()
            data[write:write + n] = vals.ravel()
            write += n
    return sparse.csr_matrix(
        (data[:write], (rows[:write], cols[:write])),
        shape=(n_cells, sae.n_features),
    ).tocsc()


def residualize_feature_top_cells(
    col_rows: np.ndarray,
    col_vals: np.ndarray,
    class_ids: np.ndarray,
    n_classes: int,
    n_cells: int,
    n_top: int,
) -> np.ndarray:
    """For one SAE feature's sparse column, compute per-class mean activation
    across ALL cells (zeros count), subtract the per-cell class mean, and
    return the indices of the top-n_top cells by residual magnitude.
    """
    # per-class sums on nonzeros only; cells not in col_rows have activation 0
    class_sum = np.zeros(n_classes, dtype=np.float64)
    np.add.at(class_sum, class_ids[col_rows], col_vals.astype(np.float64))
    # counts of ALL cells per class (includes cells with zero activation)
    class_n = np.bincount(class_ids, minlength=n_classes).astype(np.float64)
    class_mean = np.where(class_n > 0, class_sum / class_n, 0.0)

    # residual per NONZERO cell = val - class_mean[class]
    resid_nz = col_vals.astype(np.float64) - class_mean[class_ids[col_rows]]

    # for ZERO-activation cells (implicit), residual = -class_mean[class]
    # Negative magnitude — those cells won't be in the top-|resid| unless
    # class_mean is unusually large. Include them via class-level precomputed
    # top candidates: for each class, any zero cell has |resid| = |class_mean[c]|.
    # Build a pool of candidates: (cell_idx, |resid|)

    # Nonzero cells
    abs_nz = np.abs(resid_nz)
    # Per-class "zero" magnitude
    zero_mag_by_class = np.abs(class_mean)

    # Combine: for the top-|resid| selection we want the cells with the
    # largest absolute residuals. Nonzero cells have their own residuals;
    # for each class c, ALL cells in that class with activation zero share
    # the same |residual| = zero_mag_by_class[c]. For classes where
    # zero_mag_by_class is small enough to be outside the top-n_top, we can
    # ignore the zeros entirely — they can't enter the top. For classes
    # where it IS large, the top cells from that class may be zero-activation
    # cells (which all have identical residuals, so the tie is broken
    # arbitrarily).
    #
    # Practical shortcut: use only the nonzero cells as candidates. For TopK
    # SAE with k=16, most features have few nonzero cells, so the top-N by
    # |residual_nz| is a close approximation to the true top-N residual set.
    # Any class whose zero-cells have |class_mean| larger than the smallest
    # kept |resid_nz| contributes some false negatives, but the effect on
    # the top-N l7 class-fraction is usually small because zero-cells dilute
    # toward the baseline l7 distribution — a conservative direction (makes
    # features look MORE confounded, not less).
    k = min(n_top, len(col_rows))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    top_local = np.argpartition(abs_nz, -k)[-k:]
    return col_rows[top_local]


def main() -> None:
    log("=" * 72)
    log("§4.7 confound suite")

    log("loading manifest + metadata")
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    slide_meta = {s["slide_idx"]: s for s in manifest["slides"]}
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")
    n_cells = cell_slide_id.shape[0]

    cell_tissue = np.array([slide_meta[sid]["tissue"] for sid in cell_slide_id])
    cell_tech = np.array([slide_meta[sid]["technology"] for sid in cell_slide_id])

    tissues, tissue_inv = np.unique(cell_tissue, return_inverse=True)
    techs, tech_inv = np.unique(cell_tech, return_inverse=True)
    log(f"  tissues={len(tissues)}, techs={len(techs)}")

    log("stitching per-slide domain labels")
    l7 = load_global_domains(7)
    l20 = load_global_domains(20)
    assert l7.shape[0] == n_cells and l20.shape[0] == n_cells
    l7_classes, l7_inv = np.unique(l7, return_inverse=True)
    l20_classes, l20_inv = np.unique(l20, return_inverse=True)
    log(f"  l7 classes={len(l7_classes)}, l20 classes={len(l20_classes)}")
    del l7, l20

    log("loading aggregator activations + SAE")
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    log("encoding aggregator → sparse CSC")
    t0 = time.time()
    feats = encode_sparse(agg, sae, k_sae=16)
    del agg
    gc.collect()
    log(f"  {feats.shape}, nnz={feats.nnz:,}, {time.time()-t0:.1f}s")

    # Load effect-size confounds for the slide-shuffle-equivalent result
    es_path = OUT_DIR / "confounds_effect_size.parquet"
    if es_path.exists():
        es = pd.read_parquet(es_path)[["feature_idx", "slide_max_ratio", "tech_max_ratio"]].rename(
            columns={"slide_max_ratio": "slide_ratio_09b", "tech_max_ratio": "tech_ratio_09b"}
        )
        log(f"  loaded {es_path.name} for slide-shuffle-equivalent stats")
    else:
        es = None
        log("  WARNING: confounds_effect_size.parquet missing; slide survival will be NaN")

    n_top = max(MIN_TOP, int(TOP_FRAC * n_cells))
    log(f"  n_top per feature: {n_top}")

    indptr = feats.indptr
    indices = feats.indices
    data = feats.data
    n_features = feats.shape[1]

    log("computing baseline + residualized top-l7 fractions per feature")
    rows = []
    t_start = time.time()
    for fid in range(n_features):
        start_ix = indptr[fid]
        end_ix = indptr[fid + 1]
        n_nonzero = end_ix - start_ix
        row = {"feature_idx": fid, "n_nonzero": int(n_nonzero)}

        if n_nonzero < MIN_TOP:
            row.update({
                "baseline_l7_top_class": None,
                "baseline_l7_top_frac": None,
                "tech_resid_same_class_frac": None,
                "l20_resid_same_class_frac": None,
                "tech_resid_l7_top_frac": None,
                "l20_resid_l7_top_frac": None,
                "survives_slide": False,
                "survives_tech": False,
                "survives_l20": False,
                "survives_all": False,
            })
            rows.append(row)
            continue

        col_rows = indices[start_ix:end_ix]
        col_vals = data[start_ix:end_ix]
        abs_vals = np.abs(col_vals)

        # Baseline top cells by |activation|
        k = min(n_top, n_nonzero)
        top_local = np.argpartition(abs_vals, -k)[-k:]
        top_cells = col_rows[top_local]

        # Baseline l7 top class + fraction
        l7_top = l7_inv[top_cells]
        base_counts = np.bincount(l7_top, minlength=len(l7_classes))
        base_top_class = int(base_counts.argmax())
        base_top_frac = float(base_counts[base_top_class] / k)

        # Tech residualization
        tech_top_cells = residualize_feature_top_cells(
            col_rows, col_vals, tech_inv, len(techs), n_cells, k
        )
        tech_l7_top = l7_inv[tech_top_cells]
        tech_counts = np.bincount(tech_l7_top, minlength=len(l7_classes))
        tech_top_frac = float(tech_counts.max() / max(len(tech_top_cells), 1))
        tech_same_class_frac = float(tech_counts[base_top_class] / max(len(tech_top_cells), 1))

        # l20 residualization
        l20_top_cells = residualize_feature_top_cells(
            col_rows, col_vals, l20_inv, len(l20_classes), n_cells, k
        )
        l20_l7_top = l7_inv[l20_top_cells]
        l20_counts = np.bincount(l20_l7_top, minlength=len(l7_classes))
        l20_top_frac = float(l20_counts.max() / max(len(l20_top_cells), 1))
        l20_same_class_frac = float(l20_counts[base_top_class] / max(len(l20_top_cells), 1))

        row.update({
            "baseline_l7_top_class": l7_classes[base_top_class],
            "baseline_l7_top_frac": base_top_frac,
            "tech_resid_same_class_frac": tech_same_class_frac,
            "l20_resid_same_class_frac": l20_same_class_frac,
            "tech_resid_l7_top_frac": tech_top_frac,
            "l20_resid_l7_top_frac": l20_top_frac,
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    # Merge with 09b slide-ratio data
    if es is not None:
        df = df.merge(es, on="feature_idx", how="left")
        df["survives_slide"] = df["slide_ratio_09b"] >= 2.0
    else:
        df["survives_slide"] = False

    # Survival criteria: residualized same-class frac must be >= 50% of baseline
    # AND baseline must exceed 0.25 (otherwise the feature had no meaningful
    # niche enrichment to begin with).
    min_baseline = 0.25
    df["survives_tech"] = (
        df["baseline_l7_top_frac"].fillna(0) >= min_baseline
    ) & (
        df["tech_resid_same_class_frac"].fillna(0)
        >= 0.5 * df["baseline_l7_top_frac"].fillna(0)
    )
    df["survives_l20"] = (
        df["baseline_l7_top_frac"].fillna(0) >= min_baseline
    ) & (
        df["l20_resid_same_class_frac"].fillna(0)
        >= 0.5 * df["baseline_l7_top_frac"].fillna(0)
    )
    df["survives_all"] = df["survives_slide"] & df["survives_tech"] & df["survives_l20"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "confound_survival.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {out_path} ({len(df)} rows, {time.time()-t_start:.1f}s per-feature loop)")

    active = df["baseline_l7_top_frac"].notna()
    n_active = int(active.sum())
    n_with_niche = int((df["baseline_l7_top_frac"] >= min_baseline).sum())
    summary = {
        "n_features": int(len(df)),
        "n_active_min_top_cells": n_active,
        "n_with_niche_signal_baseline": n_with_niche,
        "min_baseline_niche_frac": min_baseline,
        "n_survives_slide": int(df["survives_slide"].sum()),
        "n_survives_tech": int(df["survives_tech"].sum()),
        "n_survives_l20": int(df["survives_l20"].sum()),
        "n_survives_all": int(df["survives_all"].sum()),
        "fraction_survives_all_of_niche": (
            float(df["survives_all"].sum() / n_with_niche) if n_with_niche else None
        ),
    }
    json.dump(summary, open(OUT_DIR / "confound_survival.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))

    log("")
    log("H2 evaluation: <20% of features should survive strict confound suite")
    if n_with_niche:
        frac = summary["fraction_survives_all_of_niche"]
        log(f"  result: {frac*100:.2f}% of features with niche signal survive all three controls")
        if frac < 0.20:
            log("  → H2 CONFIRMED (causal poverty: few features are truly niche-specific)")
        else:
            log("  → H2 REFUTED (more features survive than expected)")
    log("DONE")


if __name__ == "__main__":
    main()
