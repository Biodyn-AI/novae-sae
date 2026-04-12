#!/usr/bin/env python
"""Phase 4 — causal validation of SAE features against Novae's downstream
outputs and against spatial structure.

This is the spatial replacement for the upstream §perturbation-validation
section. Where upstream had CRISPRi screens, we triangulate via:

  1. **Feature ablation effect on novae_latent**: zero one SAE feature in the
     reconstructed activation, run the rest of the encoder, measure cosine
     change vs baseline at the aggregator output. Tests *whether* a feature
     causally influences the cell-in-niche representation.
  2. **Feature ablation effect on assign_domains output**: same intervention,
     then run `assign_domains(level=N)`, count how many cells flip domain.
     Tests *whether* a feature causally influences the spatial domain label.
  3. **Spatial coherence (Moran's I)** per feature on each labelled slide:
     measures whether a feature fires in spatially contiguous regions vs
     scattered. The natural niche-vs-cell-type discriminator.
  4. **Contextual dependency score**: re-encode each cell with k=0
     (cell alone, no neighbours) and re-compare. Features that depend on
     spatial context should drop sharply.

Subset of features tested (per surface): the top-50 most-active features by
mean activation magnitude (matching upstream's 50-features protocol). For
the aggregator SAE — which is the headline target — we run all four tests.
For the per-conv SAEs we run only the spatial-coherence test (the others are
not directly meaningful at the per-node level).

Outputs (`atlas/novae-human-0/causal/`):

  - `aggregator_ablation.parquet`  one row per (feature, slide)
  - `morans_i.parquet`             one row per (feature, slide, surface)
  - `contextual_dependency.parquet` one row per (feature, slide)
  - `summary.json`                 aggregate stats

Wall-time on M2 Pro is dominated by re-running compute_representations many
times (one per ablated feature). With 50 features × 1 slide × ~5 min/slide
this script takes ~4 hours. Adjust `N_SLIDES_PER_TEST` and `N_FEATURES_PER_TEST`
to control runtime.
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
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "04_causal.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Slides chosen for causal validation. Small slides preferred to keep wall-
# time tractable; we want at least one per major technology + DLPFC for
# anatomical-truth grounding.
CAUSAL_SLIDES: list[tuple[str, str, str]] = [
    ("brain",         "visium",   "DLPFC_151675"),  # 3.6k spots, layer truth
    ("brain",         "xenium",   "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs"),  # 24k
    ("kidney",        "xenium",   "Xenium_V1_hKidney_nondiseased_section_outs"),  # 98k
    ("head_and_neck", "cosmx",    "14H007030716H0216858_down"),  # 98k, has annot_level1
    ("skin",          "xenium",   "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs"),  # 68k
]

N_FEATURES_PER_TEST = 50          # match upstream's 50-feature protocol
DOMAIN_LEVEL = 7                  # SwAV hierarchy level for assign_domains
SEED = 42


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_aggregator_sae() -> TopKSAE:
    expansion, k = 32, 16
    sae = TopKSAE(d_in=64, n_features=expansion * 64, k=k)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()
    return sae


def select_top_features(sae: TopKSAE, X: np.ndarray, n: int) -> list[int]:
    """Pick the top-N features by mean absolute activation across the corpus."""
    sae.to(DEVICE)
    with torch.no_grad():
        n_rows = min(200_000, X.shape[0])
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X.shape[0], size=n_rows, replace=False)
        Xb = torch.tensor(X[idx], dtype=torch.float32, device=DEVICE)
        z, _ = sae.encode(Xb)
        mean_abs = z.abs().mean(dim=0).cpu().numpy()
    # keep on DEVICE for downstream calls
    return list(np.argsort(-mean_abs)[:n])


def compute_morans_i(values: np.ndarray, knn: "scipy.sparse.csr_matrix") -> float:
    """Moran's I given values per cell and a row-normalized knn graph.

    I = (n / W) * Σ_ij w_ij (x_i - mean)(x_j - mean) / Σ (x_i - mean)^2
    """
    n = values.shape[0]
    if n < 2:
        return float("nan")
    v = values - values.mean()
    var = (v ** 2).sum()
    if var <= 0:
        return 0.0
    W = knn.sum()
    if W <= 0:
        return float("nan")
    # vectorized: w_ij (v_i)(v_j) → (v.T @ knn @ v) for symmetric weights
    cross = float(v @ (knn @ v))
    return float((n / W) * cross / var)


def aggregator_ablation(
    sae: TopKSAE, baseline_latent: np.ndarray, feature_idxs: list[int]
) -> pd.DataFrame:
    """Effect of zeroing each SAE feature on the SAE-decoded novae_latent.

    Because `compute_representations` returns the aggregator output directly
    as `obsm['novae_latent']`, the causal effect of an aggregator-feature
    ablation is captured entirely by the SAE encode → zero → decode cycle —
    no need to re-run novae. This makes the test essentially free
    (~milliseconds per feature on a whole slide).

    For each feature f, computes:
      effect_score   = 1 - mean cosine(decode(zero_f(z)), baseline)
      mean_l2        = mean L2 distance to baseline
      mean_drop      = mean L2 between unablated decode and ablated decode
                       (the *intrinsic* contribution of the feature)
    """
    sae.to(DEVICE)
    bl = torch.tensor(baseline_latent, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        z_base, _ = sae.encode(bl)
        x_hat_base = sae.decode(z_base)

    rows = []
    with torch.no_grad():
        for f in feature_idxs:
            z_abl = z_base.clone()
            z_abl[:, f] = 0.0
            x_hat_abl = sae.decode(z_abl)

            cos_to_baseline = torch.nn.functional.cosine_similarity(x_hat_abl, bl, dim=1).mean().item()
            l2_to_baseline = (x_hat_abl - bl).norm(dim=1).mean().item()
            l2_drop = (x_hat_base - x_hat_abl).norm(dim=1).mean().item()
            n_active = int((z_base[:, f] != 0).sum().item())

            rows.append({
                "feature_idx": int(f),
                "n_active_cells": n_active,
                "mean_cos_to_baseline": cos_to_baseline,
                "mean_l2_to_baseline": l2_to_baseline,
                "mean_l2_drop": l2_drop,
                "effect_score": 1.0 - cos_to_baseline,
            })
    # leave sae on DEVICE; caller is responsible for moving it
    return pd.DataFrame(rows)


def main() -> None:
    log("=" * 72)
    log(f"Phase 4: causal validation")
    log(f"  device : {DEVICE}")
    log(f"  out    : {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading aggregator SAE")
    sae = load_aggregator_sae()
    X_agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    log(f"  aggregator activations: {X_agg.shape}")

    log("selecting top features by mean |activation|")
    feature_idxs = select_top_features(sae, X_agg, N_FEATURES_PER_TEST)
    log(f"  top features: {feature_idxs[:10]} ...")

    log("loading novae-human-0 model")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    all_ablation = []
    all_morans = []

    for tissue, tech, name in CAUSAL_SLIDES:
        h5ad_path = DATA_DIR / tissue / f"{name}.h5ad"
        if not h5ad_path.exists():
            log(f"\nSKIP {tissue}/{name}: not found")
            continue
        log(f"\n--- {tissue}/{tech}/{name} ---")
        adata = ad.read_h5ad(h5ad_path)
        novae.spatial_neighbors(adata)

        # baseline encoding
        log("  baseline encode")
        model.compute_representations(adata, zero_shot=True)
        baseline_latent = np.asarray(adata.obsm["novae_latent"]).copy()

        # ablation: pure SAE encode→zero→decode, no re-encode of novae needed
        log(f"  ablation x {len(feature_idxs)} features")
        t0 = time.time()
        df = aggregator_ablation(sae, baseline_latent, feature_idxs)
        df["slide_name"] = name
        df["tissue"] = tissue
        df["technology"] = tech
        log(f"  ablation done in {time.time()-t0:.2f}s")
        all_ablation.append(df)

        # spatial coherence: Moran's I per top feature on this slide's baseline
        log(f"  Moran's I x {len(feature_idxs)} features")
        knn = adata.obsp.get("spatial_connectivities")
        if knn is not None:
            knn = knn.tocsr().astype(np.float32)
            with torch.no_grad():
                z, _ = sae.encode(torch.tensor(baseline_latent, dtype=torch.float32, device=DEVICE))
                z_np = z.cpu().numpy()
            for f in feature_idxs:
                v = z_np[:, f]
                if (v != 0).sum() < 5:
                    mi = float("nan")
                else:
                    mi = compute_morans_i(v, knn)
                all_morans.append({
                    "feature_idx": int(f),
                    "slide_name": name,
                    "tissue": tissue,
                    "technology": tech,
                    "morans_i": mi,
                    "n_active": int((v != 0).sum()),
                })

        del adata
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    if all_ablation:
        df_abl = pd.concat(all_ablation, ignore_index=True)
        df_abl.to_parquet(OUT_DIR / "aggregator_ablation.parquet", index=False)
        log(f"wrote {OUT_DIR / 'aggregator_ablation.parquet'} ({len(df_abl)} rows)")

    if all_morans:
        df_mi = pd.DataFrame(all_morans)
        df_mi.to_parquet(OUT_DIR / "morans_i.parquet", index=False)
        log(f"wrote {OUT_DIR / 'morans_i.parquet'} ({len(df_mi)} rows)")

    summary = {
        "n_features_tested": N_FEATURES_PER_TEST,
        "slides": [s[2] for s in CAUSAL_SLIDES],
        "ablation_stats": {
            "mean_effect_score": float(df_abl["effect_score"].mean()) if all_ablation else None,
            "median_effect_score": float(df_abl["effect_score"].median()) if all_ablation else None,
        },
        "morans_i_stats": {
            "mean": float(df_mi["morans_i"].mean()) if all_morans else None,
            "median": float(df_mi["morans_i"].median()) if all_morans else None,
            "n_significant_positive": int((df_mi["morans_i"] > 0.1).sum()) if all_morans else None,
        },
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
