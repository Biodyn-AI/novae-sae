#!/usr/bin/env python
"""§4.5/3 — cross-technology coherence.

For each tissue with Xenium + MERSCOPE coverage, compute per-feature
mean |activation| profiles on both technologies using the aggregator SAE.
Then correlate across technologies per feature. Features with high
cross-tech Spearman correlation are technology-invariant; those with low
correlation are tech-specific.

Approach: for each tissue/tech pair, load the h5ad, run Novae
compute_representations (if not already cached as per-slide aggregator),
encode through the aggregator SAE, and compute per-feature mean |activation|.
Then for each tissue, correlate the Xenium feature profile with the MERSCOPE
profile.

Multi-tech human tissues available (from metadata.csv):
  - liver:    Xenium (1 non-disease), MERSCOPE (2 cancer), CosMx (2)
  - lung:     Xenium (5), MERSCOPE (2)
  - breast:   Xenium (5), MERSCOPE (1)
  - colon:    Xenium (7), MERSCOPE (2)
  - ovarian:  Xenium (1), MERSCOPE (4)
  - pancreas: Xenium (4), CosMx (1)
  - prostate: Xenium (1), MERSCOPE (2)
  - skin:     Xenium (4), MERSCOPE (2)

For efficiency, pick ONE xenium + ONE merscope slide per tissue (using the
already-processed slides from the 15-slide corpus where possible, plus one
new MERSCOPE slide per tissue).

Output: atlas/novae-human-0/causal/cross_tech_coherence.parquet
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
from scipy import sparse, stats

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "16_cross_tech_coherence.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Pairs: (tissue, xenium_slide, merscope_or_cosmx_slide, alt_tech_name)
# Using already-processed xenium slides where possible
PAIRS = [
    ("liver",
     "Xenium_V1_hLiver_nondiseased_section_FFPE_outs",
     "HumanLiverCancerPatient1_region_0",
     "merscope"),
    ("lung",
     "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs",
     "HumanLungCancerPatient1_region_0",
     "merscope"),
    ("breast",
     "Xenium_V1_FFPE_Human_Breast_IDC_outs",
     "HumanBreastCancerPatient1_region_0",
     "merscope"),
    ("colon",
     "Xenium_V1_hColon_Non_diseased_Base_FFPE_outs",
     "HumanColonCancerPatient1_region_0",
     "merscope"),
    ("ovarian",
     "Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs",
     "HumanOvarianCancerPatient1_region_0",
     "merscope"),
    ("prostate",
     "Xenium_Prime_Human_Prostate_FFPE_outs",
     "HumanProstateCancerPatient1_region_0",
     "merscope"),
    ("skin",
     "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs",
     "HumanMelanomaPatient1_region_0",
     "merscope"),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def get_aggregator_for_slide(
    model, sae, tissue: str, slide_name: str,
) -> np.ndarray:
    """Get per-feature mean |activation| for a slide. Uses cached per-slide
    aggregator if available, otherwise runs compute_representations."""
    cached = ACT_DIR / "per_slide" / f"{tissue}__{slide_name}" / "aggregator.npy"
    if cached.exists():
        log(f"    using cached aggregator: {cached.name}")
        agg = np.load(cached).astype(np.float32, copy=False)
    else:
        h5ad = DATA_DIR / tissue / f"{slide_name}.h5ad"
        log(f"    loading + encoding {h5ad.name}")
        a = ad.read_h5ad(h5ad)
        novae.spatial_neighbors(a)
        t0 = time.time()
        model.compute_representations(a, zero_shot=True)
        agg = np.asarray(a.obsm["novae_latent"], dtype=np.float32)
        log(f"    {a.n_obs:,} cells, encode {time.time()-t0:.0f}s")
        # Cache for reuse
        out_dir = ACT_DIR / "per_slide" / f"{tissue}__{slide_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "aggregator.npy", agg)
        del a
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # SAE-encode and return per-feature mean |activation|
    n_cells = agg.shape[0]
    k_sae = 16
    chunk = 32768
    feat_sum = np.zeros(sae.n_features, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            xb = torch.tensor(agg[start:end], dtype=torch.float32)
            z, _ = sae.encode(xb)
            feat_sum += np.abs(z.cpu().numpy()).sum(axis=0)
    return (feat_sum / max(n_cells, 1)).astype(np.float32)


def main() -> None:
    log("=" * 72)
    log("§4.5/3 cross-technology coherence")

    log("loading model + SAE")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    results = []
    for tissue, xen_slide, alt_slide, alt_tech in PAIRS:
        log(f"\n{tissue}: xenium={xen_slide[:40]}... {alt_tech}={alt_slide[:40]}...")
        try:
            xen_profile = get_aggregator_for_slide(model, sae, tissue, xen_slide)
            alt_profile = get_aggregator_for_slide(model, sae, tissue, alt_slide)
        except Exception as e:
            log(f"  ERROR: {e}")
            continue

        # Per-feature Spearman correlation of profiles is not meaningful
        # (profiles are per-feature scalars, not per-cell).
        # Instead, compute the correlation of the 2048-dim FEATURE PROFILE
        # across the two slides: are the same features active on both?
        rho, p = stats.spearmanr(xen_profile, alt_profile)
        pearson_r, pearson_p = stats.pearsonr(xen_profile, alt_profile)

        # Per-feature ratio: how different is the activation magnitude?
        ratio = alt_profile / np.maximum(xen_profile, 1e-12)
        median_ratio = float(np.median(ratio[xen_profile > 1e-9]))

        results.append({
            "tissue": tissue,
            "xenium_slide": xen_slide,
            "alt_slide": alt_slide,
            "alt_tech": alt_tech,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "median_activation_ratio": median_ratio,
            "xenium_n_active_gt_1e-6": int((xen_profile > 1e-6).sum()),
            "alt_n_active_gt_1e-6": int((alt_profile > 1e-6).sum()),
        })
        log(f"  rho={rho:.4f} r={pearson_r:.4f} median_ratio={median_ratio:.3f}")

    if not results:
        log("ERROR: no tissue pairs processed")
        return

    df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cross_tech_coherence.parquet"
    df.to_parquet(out_path, index=False)
    log(f"\nwrote {out_path} ({len(df)} rows)")

    summary = {
        "n_tissue_pairs": len(df),
        "mean_spearman_rho": float(df["spearman_rho"].mean()),
        "median_spearman_rho": float(df["spearman_rho"].median()),
        "min_spearman_rho": float(df["spearman_rho"].min()),
        "tissues": df["tissue"].tolist(),
        "per_tissue_rho": dict(zip(df["tissue"], df["spearman_rho"].round(4))),
    }
    json.dump(summary, open(OUT_DIR / "cross_tech_coherence.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
