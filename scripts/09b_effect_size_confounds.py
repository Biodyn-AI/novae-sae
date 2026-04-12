#!/usr/bin/env python
"""Block 1.3b — effect-size confound filter (fixes 1.3 overpower).

Block 1.3's chi-square is overpowered at n_top=4525: every feature passes
FDR<0.05 because any micro-deviation from baseline is "significant" at
that sample size. The useful statistic is *effect size*: how large is
the top-class share relative to the corpus baseline?

For each aggregator SAE feature, for each dimension in {slide, tissue,
technology}, compute:
  - top_class_share: fraction of top-N cells in the single largest class
  - baseline_max_share: the fraction of ALL cells in that same class
  - max_class_ratio = top_class_share / baseline_max_share
  - entropy_ratio = H(top) / H(baseline)   (structural vs. noise)

A feature is "slide-confounded" if its top cells pile up on one slide
much more than the baseline expects, i.e. max_class_ratio >= 2.0 on the
slide dimension. Same for tissue / technology.

Output: atlas/novae-human-0/causal/confounds_effect_size.parquet
Produces H9 verdict (≥5% tech-specific features).
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
LOG_PATH = ROOT / "logs" / "09b_effect_size_confounds.log"

TOP_FRAC = 0.001
MIN_TOP = 50
EFFECT_RATIO_THRESHOLD = 2.0  # "concentrated" if max class is 2× baseline


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def encode_sparse(agg: np.ndarray, sae: TopKSAE, k_sae: int, chunk: int = 32768):
    n_cells, n_features = agg.shape[0], sae.n_features
    rows_buf = np.empty(n_cells * k_sae, dtype=np.int64)
    cols_buf = np.empty(n_cells * k_sae, dtype=np.int32)
    data_buf = np.empty(n_cells * k_sae, dtype=np.float32)
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
            rows_buf[write:write + n] = np.repeat(np.arange(start, end, dtype=np.int64), k_sae)
            cols_buf[write:write + n] = idx_np.ravel()
            data_buf[write:write + n] = vals.ravel()
            write += n
    return sparse.csr_matrix(
        (data_buf[:write], (rows_buf[:write], cols_buf[:write])),
        shape=(n_cells, n_features),
    ).tocsc()


def main() -> None:
    log("=" * 72)
    log("Block 1.3b: effect-size confound filter")

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    log("loading activations + metadata")
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    cell_slide_id = np.load(ACT_DIR / "cell_slide_id.npy")
    log(f"  agg: {agg.shape}")

    slide_meta = {s["slide_idx"]: s for s in manifest["slides"]}
    cell_tissue = np.array([slide_meta[sid]["tissue"] for sid in cell_slide_id])
    cell_tech = np.array([slide_meta[sid]["technology"] for sid in cell_slide_id])

    tissues, tissue_counts = np.unique(cell_tissue, return_counts=True)
    techs, tech_counts = np.unique(cell_tech, return_counts=True)
    slides_uniq, slide_counts = np.unique(cell_slide_id, return_counts=True)

    tissue_baseline = tissue_counts / tissue_counts.sum()
    tech_baseline = tech_counts / tech_counts.sum()
    slide_baseline = slide_counts / slide_counts.sum()

    tissue_to_idx = {t: i for i, t in enumerate(tissues)}
    tech_to_idx = {t: i for i, t in enumerate(techs)}
    slide_to_idx = {s: i for i, s in enumerate(slides_uniq)}

    log(f"  {len(tissues)} tissues, {len(techs)} techs, {len(slides_uniq)} slides")
    log(f"  baseline max shares — tissue={tissue_baseline.max():.3f}, tech={tech_baseline.max():.3f}, slide={slide_baseline.max():.3f}")

    log("loading aggregator SAE")
    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    log("encoding aggregator → sparse features")
    t0 = time.time()
    feats = encode_sparse(agg, sae, k_sae=16)
    del agg
    gc.collect()
    log(f"  CSC {feats.shape}, nnz={feats.nnz:,}, {time.time()-t0:.1f}s")

    n_cells = feats.shape[0]
    n_features = feats.shape[1]
    n_top = max(MIN_TOP, int(TOP_FRAC * n_cells))
    log(f"  n_top per feature: {n_top}")

    log("computing per-feature effect-size statistics")
    indptr = feats.indptr
    indices = feats.indices
    data = feats.data

    def class_stats(top_ix: np.ndarray, cell_class: np.ndarray, to_idx: dict, baseline: np.ndarray):
        classes = cell_class[top_ix]
        counts = np.zeros(len(baseline), dtype=np.int64)
        for c in classes:
            counts[to_idx[c]] += 1
        fracs = counts / max(counts.sum(), 1)
        arg_max = int(fracs.argmax())
        max_share = float(fracs[arg_max])
        base_share = float(baseline[arg_max])
        ratio = max_share / max(base_share, 1e-12)
        # normalized Shannon entropy
        fracs_nz = fracs[fracs > 0]
        H_obs = float(-(fracs_nz * np.log(fracs_nz)).sum())
        base_nz = baseline[baseline > 0]
        H_base = float(-(base_nz * np.log(base_nz)).sum())
        entropy_ratio = H_obs / max(H_base, 1e-12)
        return {
            "max_share": max_share,
            "max_class_idx": arg_max,
            "max_share_ratio": ratio,
            "entropy_ratio": entropy_ratio,
        }

    rows = []
    for fid in range(n_features):
        start_ix = indptr[fid]
        end_ix = indptr[fid + 1]
        n_nonzero = end_ix - start_ix
        row = {"feature_idx": fid, "n_nonzero": int(n_nonzero)}
        if n_nonzero < MIN_TOP:
            row.update({
                "tissue_max_share": None, "tissue_max_class_idx": None, "tissue_max_ratio": None, "tissue_entropy_ratio": None,
                "tech_max_share": None, "tech_max_class_idx": None, "tech_max_ratio": None, "tech_entropy_ratio": None,
                "slide_max_share": None, "slide_max_class_idx": None, "slide_max_ratio": None, "slide_entropy_ratio": None,
                "n_top_used": 0,
            })
            rows.append(row)
            continue
        col_rows = indices[start_ix:end_ix]
        col_vals = np.abs(data[start_ix:end_ix])
        k = min(n_top, n_nonzero)
        top_local = np.argpartition(col_vals, -k)[-k:]
        top_ix = col_rows[top_local]

        t_stats = class_stats(top_ix, cell_tissue, tissue_to_idx, tissue_baseline)
        te_stats = class_stats(top_ix, cell_tech, tech_to_idx, tech_baseline)
        s_stats = class_stats(top_ix, cell_slide_id, slide_to_idx, slide_baseline)

        row.update({
            "tissue_max_share": t_stats["max_share"],
            "tissue_max_class_idx": int(t_stats["max_class_idx"]),
            "tissue_max_ratio": t_stats["max_share_ratio"],
            "tissue_entropy_ratio": t_stats["entropy_ratio"],
            "tech_max_share": te_stats["max_share"],
            "tech_max_class_idx": int(te_stats["max_class_idx"]),
            "tech_max_ratio": te_stats["max_share_ratio"],
            "tech_entropy_ratio": te_stats["entropy_ratio"],
            "slide_max_share": s_stats["max_share"],
            "slide_max_class_idx": int(s_stats["max_class_idx"]),
            "slide_max_ratio": s_stats["max_share_ratio"],
            "slide_entropy_ratio": s_stats["entropy_ratio"],
            "n_top_used": int(k),
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    # Human-readable class name columns
    df["tissue_top_class"] = df["tissue_max_class_idx"].map(
        {i: tissues[i] for i in range(len(tissues))}
    )
    df["tech_top_class"] = df["tech_max_class_idx"].map(
        {i: techs[i] for i in range(len(techs))}
    )
    df["slide_top_class_idx"] = df["slide_max_class_idx"]  # raw slide index

    # Effect-size flags
    thr = EFFECT_RATIO_THRESHOLD
    df["tissue_concentrated_es"] = df["tissue_max_ratio"] >= thr
    df["tech_concentrated_es"] = df["tech_max_ratio"] >= thr
    df["slide_concentrated_es"] = df["slide_max_ratio"] >= thr

    # Tech-specific = single-technology domination (share >= 0.9 AND ratio >= 2)
    df["tech_specific"] = (df["tech_max_share"] >= 0.9) & (df["tech_max_ratio"] >= thr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "confounds_effect_size.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {out_path} ({len(df)} rows)")

    total_active = int((df["n_top_used"] > 0).sum())
    summary = {
        "n_features": int(len(df)),
        "n_active": total_active,
        "effect_ratio_threshold": thr,
        "n_tissue_concentrated": int(df["tissue_concentrated_es"].sum()),
        "n_tech_concentrated": int(df["tech_concentrated_es"].sum()),
        "n_slide_concentrated": int(df["slide_concentrated_es"].sum()),
        "n_tech_specific (>=0.9 single-tech)": int(df["tech_specific"].sum()),
        "fraction_tech_specific": (
            float(df["tech_specific"].sum() / total_active) if total_active else None
        ),
        "tissue_max_ratio_p50": float(df["tissue_max_ratio"].median(skipna=True)),
        "tech_max_ratio_p50": float(df["tech_max_ratio"].median(skipna=True)),
        "slide_max_ratio_p50": float(df["slide_max_ratio"].median(skipna=True)),
    }
    json.dump(summary, open(OUT_DIR / "confounds_effect_size.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))

    # H9 verdict
    log("")
    log("H9 evaluation: >=5% of features should be technology-specific")
    if total_active:
        frac = summary["fraction_tech_specific"]
        log(f"  result: {frac*100:.2f}% of active features tech-specific")
        if frac >= 0.05:
            log("  → H9 CONFIRMED")
        else:
            log("  → H9 REFUTED")
    log("DONE")


if __name__ == "__main__":
    main()
