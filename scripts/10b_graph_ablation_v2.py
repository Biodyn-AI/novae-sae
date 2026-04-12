#!/usr/bin/env python
"""§4.5/2 full graph-ablation suite — three regimes + sanity control.

Fixes the scale-confound in `10_block1_graph_ablation.py` (see
`memory/feedback_graph_ablation_scale_confound.md`). For each slide, run
`novae.compute_representations` under four graph regimes:

  (a) **full**            — the original Delaunay graph from
                            `novae.spatial_neighbors`.
  (b) **self_loop**       — replace obsp with `sp.eye(n)` (weight 1.0 on
                            self). This is the H8 canonical protocol but
                            suffers from softmax re-normalization (iso
                            self-weight is ~6× the full-graph self-weight
                            because Delaunay has ~6 neighbors per node).
  (c) **self_loop_norm**  — replace obsp with `sp.eye(n) / mean_degree`
                            where `mean_degree` is the per-slide mean
                            Delaunay degree. Restores parity with the
                            full-graph self-attention weight, so dep
                            reflects context use, not rescaling.
  (d) **random_rewire**   — degree-preserving random rewire: randomly
                            permute the column indices of each row of
                            the full obsp while keeping the row lengths.
                            Tests whether the *specific* neighborhood
                            matters vs. any graph with the same degree
                            distribution.

Per feature: `dep_{regime} = (mean|full| - mean|regime|) / max(mean|full|, eps)`

Output: `atlas/novae-human-0/causal/graph_ablation_v2.parquet` with one
row per feature containing mean|full|, mean|self_loop|, mean|self_loop_norm|,
mean|random_rewire| and the three dep scores.

Small-slide subset only (brain, kidney, pancreas, skin, liver) to keep
the MPS wall time under ~25 min.
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
import pandas as pd
import scipy.sparse as sp
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "10b_graph_ablation_v2.log"

# Small-slide subset for tractable wall time on MPS
SLIDE_SUBSET_NAMES = [
    "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",       # brain 24k
    "Xenium_V1_hKidney_nondiseased_section_outs",                # kidney 97k
    "Xenium_V1_hPancreas_nondiseased_section_outs",              # pancreas 104k
    "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs",           # skin 68k
    "Xenium_V1_hLiver_nondiseased_section_FFPE_outs",            # liver 239k
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_FEATURES = 2048
RNG_SEED = 2026


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def encode_features(sae: TopKSAE, X: np.ndarray) -> np.ndarray:
    chunk = 32768
    out = []
    sae.to(DEVICE)
    with torch.no_grad():
        for start in range(0, X.shape[0], chunk):
            xb = torch.tensor(X[start:start + chunk], dtype=torch.float32, device=DEVICE)
            z, _ = sae.encode(xb)
            out.append(z.cpu().numpy())
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return np.vstack(out)


def snapshot_obsp(adata) -> dict[str, sp.spmatrix]:
    """Snapshot the spatial obsp entries so we can restore/replace them
    between regimes.
    """
    snap = {}
    for key in list(adata.obsp.keys()):
        if "spatial" in key.lower() or "connectivit" in key.lower() or "distance" in key.lower():
            snap[key] = adata.obsp[key].copy()
    return snap


def restore_obsp(adata, snap: dict[str, sp.spmatrix]) -> None:
    for key, mat in snap.items():
        adata.obsp[key] = mat.copy()


def install_obsp(adata, mat_by_key: dict[str, sp.spmatrix]) -> None:
    for key, mat in mat_by_key.items():
        adata.obsp[key] = mat.copy()
    # clear cached novae output so compute_representations recomputes
    if "novae_latent" in adata.obsm:
        del adata.obsm["novae_latent"]


def build_self_loop(n: int, scale: float = 1.0) -> sp.csr_matrix:
    """Scaled identity matrix as the spatial graph."""
    return (sp.eye(n, format="csr", dtype=np.float32) * float(scale)).tocsr()


def build_random_rewire(mat: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Degree-preserving random rewire: each row keeps its original number
    of nonzeros, but the column indices are replaced by a random
    permutation of the full index space (sampled without replacement per
    row). Values are preserved in order.
    """
    n = mat.shape[0]
    out = mat.copy().tolil()
    for i in range(n):
        row = mat.getrow(i)
        nnz = row.nnz
        if nnz == 0:
            continue
        new_cols = rng.choice(n, size=nnz, replace=False)
        out.rows[i] = new_cols.tolist()
        out.data[i] = row.data.tolist()
    return out.tocsr()


def mean_degree(mat: sp.spmatrix) -> float:
    """Mean nnz per row (= average neighbors including self loops)."""
    if mat.nnz == 0:
        return 1.0
    return float(mat.nnz / mat.shape[0])


def main() -> None:
    log("=" * 72)
    log("§4.5/2 graph ablation v2 — full / self_loop / self_loop_norm / random_rewire")

    log("loading model + SAE")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    sae = TopKSAE(d_in=64, n_features=N_FEATURES, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    manifest = json.load(open(ACT_DIR / "manifest.json"))
    slides_to_process = [s for s in manifest["slides"] if s["name"] in SLIDE_SUBSET_NAMES]
    log(f"  processing {len(slides_to_process)} slides")

    rng = np.random.default_rng(RNG_SEED)

    REGIMES = ["full", "self_loop", "self_loop_norm", "random_rewire"]
    sums = {r: np.zeros(N_FEATURES, dtype=np.float64) for r in REGIMES}
    cells_by_regime = {r: 0 for r in REGIMES}

    for slide in slides_to_process:
        h5ad = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"\n[{slide['slide_idx']+1}/{len(manifest['slides'])}] {slide['tissue']}/{slide['name']}")
        try:
            a = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"  ERROR load: {e}")
            continue

        try:
            novae.spatial_neighbors(a)
        except Exception as e:
            log(f"  ERROR spatial_neighbors: {e}")
            del a
            continue

        snap = snapshot_obsp(a)
        primary_key = None
        for key in snap:
            if "spatial_connectiv" in key.lower() or key.lower() == "spatial_distances":
                primary_key = key if primary_key is None else primary_key
        if primary_key is None and snap:
            primary_key = next(iter(snap))
        mean_deg = mean_degree(snap[primary_key]) if primary_key else 6.0
        log(f"  {a.n_obs:,} cells; mean_degree on '{primary_key}' = {mean_deg:.2f}")

        # ---- Regime (a): full ----
        try:
            install_obsp(a, snap)
            t0 = time.time()
            model.compute_representations(a, zero_shot=True)
            full_latent = a.obsm["novae_latent"].copy()
            log(f"  full pass {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  ERROR full compute: {type(e).__name__}: {e}")
            del a
            continue

        # ---- Regime (b): self_loop ----
        try:
            eye = build_self_loop(a.n_obs, scale=1.0)
            install_obsp(a, {k: eye for k in snap})
            t0 = time.time()
            model.compute_representations(a, zero_shot=True)
            iso_latent = a.obsm["novae_latent"].copy()
            log(f"  self_loop pass {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  ERROR self_loop: {type(e).__name__}: {e}")
            iso_latent = None

        # ---- Regime (c): self_loop_norm ----
        try:
            eye_norm = build_self_loop(a.n_obs, scale=1.0 / mean_deg)
            install_obsp(a, {k: eye_norm for k in snap})
            t0 = time.time()
            model.compute_representations(a, zero_shot=True)
            iso_norm_latent = a.obsm["novae_latent"].copy()
            log(f"  self_loop_norm (1/{mean_deg:.2f}) pass {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  ERROR self_loop_norm: {type(e).__name__}: {e}")
            iso_norm_latent = None

        # ---- Regime (d): random_rewire ----
        rewired = {k: build_random_rewire(mat, rng) for k, mat in snap.items()}
        try:
            install_obsp(a, rewired)
            t0 = time.time()
            model.compute_representations(a, zero_shot=True)
            rewire_latent = a.obsm["novae_latent"].copy()
            log(f"  random_rewire pass {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  ERROR random_rewire: {type(e).__name__}: {e}")
            rewire_latent = None

        # Encode all regimes via SAE and accumulate
        log("  SAE encoding all regimes")
        regime_latents = {
            "full": full_latent,
            "self_loop": iso_latent,
            "self_loop_norm": iso_norm_latent,
            "random_rewire": rewire_latent,
        }
        for r, lat in regime_latents.items():
            if lat is None:
                continue
            f = encode_features(sae, lat)
            sums[r] += np.abs(f).sum(axis=0)
            cells_by_regime[r] += f.shape[0]
            del f

        del a, full_latent, iso_latent, iso_norm_latent, rewire_latent, snap, rewired
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    log("")
    log(f"regime cell counts: {cells_by_regime}")
    means = {
        r: sums[r] / max(cells_by_regime[r], 1)
        for r in REGIMES
    }

    rows = []
    eps = 1e-9
    for fid in range(N_FEATURES):
        row = {"feature_idx": fid}
        for r in REGIMES:
            row[f"mean_abs_{r}"] = float(means[r][fid])
        f_full = means["full"][fid]
        for r in ["self_loop", "self_loop_norm", "random_rewire"]:
            f_r = means[r][fid]
            if f_full > eps:
                dep = (f_full - f_r) / f_full
            else:
                dep = float("nan")
            row[f"dep_{r}"] = dep
        rows.append(row)
    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "graph_ablation_v2.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {out_path} ({len(df)} rows)")

    has_signal = df["mean_abs_full"] > eps
    n_sig = int(has_signal.sum())
    summary = {
        "regimes": REGIMES,
        "n_features": int(len(df)),
        "n_with_full_signal": n_sig,
        "slides_processed": len(slides_to_process),
        "cells_by_regime": cells_by_regime,
    }
    for r in ["self_loop", "self_loop_norm", "random_rewire"]:
        dep = df.loc[has_signal, f"dep_{r}"].dropna()
        summary[f"{r}_dep_p50"] = float(dep.median())
        summary[f"{r}_dep_mean"] = float(dep.mean())
        summary[f"{r}_n_dep_gt_0.5"] = int((dep > 0.5).sum())
        summary[f"{r}_n_dep_gt_0.2"] = int((dep > 0.2).sum())
        summary[f"{r}_n_dep_lt_neg0.5"] = int((dep < -0.5).sum())
        summary[f"{r}_fraction_dep_gt_0.5"] = (
            float((dep > 0.5).sum() / n_sig) if n_sig else None
        )

    json.dump(summary, open(OUT_DIR / "graph_ablation_v2.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))

    log("")
    log("H8 evaluation (degree-normalized regime, canonical): >20% should have dep > 0.5")
    if n_sig:
        frac = summary["self_loop_norm_fraction_dep_gt_0.5"]
        log(f"  self_loop_norm dep > 0.5: {frac*100:.2f}% of active features")
        if frac > 0.20:
            log("  → H8 CONFIRMED (Novae uses spatial context)")
        else:
            log("  → H8 REFUTED")
    log("")
    log("H8 evaluation (random_rewire regime, strict): >20% should have dep > 0.5")
    if n_sig:
        frac = summary["random_rewire_fraction_dep_gt_0.5"]
        log(f"  random_rewire dep > 0.5: {frac*100:.2f}% of active features")
    log("DONE")


if __name__ == "__main__":
    main()
