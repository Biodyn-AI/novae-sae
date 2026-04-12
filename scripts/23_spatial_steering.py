#!/usr/bin/env python
"""GAP 16: Trajectory-guided feature steering along spatial gradients.

NOVEL contribution not in either arXiv paper. Novae has spatial gradients
instead of pseudotime:
  - Crypt-to-villus in colon (intestinal zonation)
  - Tumor core to periphery in breast IDC

For each SAE feature at the aggregator, multiply its activation by
alpha (0.5x, 2x, 5x), decode back to aggregator space, and measure
the cosine shift toward/away from a spatial-gradient gene signature.

Features that push toward "villus" when amplified encode pro-maturity
spatial signals; those that push toward "crypt" encode stem/proliferative
signals.

Output: atlas/novae-human-0/causal/spatial_steering.parquet
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

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "23_spatial_steering.log"

# Spatial gradient gene signatures (manually curated from literature)
GRADIENTS = {
    "colon_crypt_to_villus": {
        "description": "Intestinal crypt (stem) → villus (differentiated)",
        "positive_genes": [  # villus / differentiated markers
            "FABP1", "APOA1", "APOA4", "APOB", "SLC26A3",
            "CA2", "SI", "ANPEP", "ACE2", "TFF3",
        ],
        "negative_genes": [  # crypt / stem / proliferative markers
            "LGR5", "OLFM4", "ASCL2", "SOX9", "MKI67",
            "PCNA", "TOP2A", "STMN1", "BIRC5", "CDK1",
        ],
    },
    "breast_tumor_vs_normal": {
        "description": "Normal breast → tumor microenvironment",
        "positive_genes": [  # tumor / cancer markers
            "ERBB2", "MKI67", "ESR1", "PGR", "TOP2A",
            "MMP2", "MMP9", "VEGFA", "HIF1A", "CD274",
        ],
        "negative_genes": [  # normal tissue markers
            "KRT14", "KRT5", "ACTA2", "MYH11", "MYLK",
            "DES", "CNN1", "KRT8", "KRT18", "EPCAM",
        ],
    },
}

ALPHAS = [0.5, 2.0, 5.0]
TOP_FEATURES = 100  # most active features to steer


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def compute_gradient_direction(sae: TopKSAE, gene_names: list[str],
                                pos_genes: list[str], neg_genes: list[str]) -> torch.Tensor:
    """Compute a direction in SAE-decoded space that points from negative
    to positive gene signatures.

    Uses the SAE decoder columns to project gene names into the
    64-dim aggregator space. This is approximate: the decoder maps
    FROM features TO aggregator space, so we use the decoder's
    implicit gene mapping via the top-gene associations.

    Simpler approach: compute mean aggregator embedding of cells
    expressing positive genes vs negative genes from the activation
    corpus, then use the difference as the gradient direction.
    """
    # The gradient direction is defined in aggregator (64-dim) space.
    # We approximate it by taking the mean activation of the top-1%
    # cells for each gene signature and computing the difference.
    # This requires the aggregator activations + per-cell gene info.
    # For now, use a random direction as placeholder and measure
    # steering magnitude rather than direction.
    #
    # Actually, we can use the SAE itself: if we know which features
    # correspond to positive vs negative genes (from the feature atlas),
    # we can define the gradient as the difference of their decoder
    # columns.
    return None  # Will use a different approach below


def main():
    log("=" * 60)
    log("GAP 16: Spatial gradient steering (NOVEL)")

    # Load SAE + aggregator activations
    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    n_cells = agg.shape[0]
    log(f"aggregator: {agg.shape}")

    # Load feature atlas for gene associations
    fa = pd.read_parquet(ROOT / "atlas" / "novae-human-0" / "feature_atlas_full.parquet")
    agg_fa = fa[fa["surface"] == "aggregator"].set_index("feature_idx")

    # Select top features by mean activation
    chunk = 32768
    feat_mean = np.zeros(2048, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n_cells, chunk):
            e = min(s + chunk, n_cells)
            z, _ = sae.encode(torch.tensor(agg[s:e]))
            feat_mean += np.abs(z.numpy()).sum(axis=0)
    feat_mean /= n_cells
    top_features = np.argsort(-feat_mean)[:TOP_FEATURES].tolist()
    log(f"selected top-{TOP_FEATURES} features by mean activation")

    results = []

    for grad_name, grad_info in GRADIENTS.items():
        log(f"\n=== Gradient: {grad_name} ===")
        log(f"  {grad_info['description']}")

        pos_genes = set(g.upper() for g in grad_info["positive_genes"])
        neg_genes = set(g.upper() for g in grad_info["negative_genes"])

        # Find features whose top genes overlap with gradient signatures
        feat_to_gradient_score = {}
        for fid in range(2048):
            if fid not in agg_fa.index:
                continue
            row = agg_fa.loc[fid]
            tg = row.get("top_genes", "")
            if pd.isna(tg) or not tg:
                continue
            genes = set(g.strip().upper() for g in str(tg).split(","))
            pos_overlap = len(genes & pos_genes)
            neg_overlap = len(genes & neg_genes)
            feat_to_gradient_score[fid] = pos_overlap - neg_overlap

        n_pos_feats = sum(1 for v in feat_to_gradient_score.values() if v > 0)
        n_neg_feats = sum(1 for v in feat_to_gradient_score.values() if v < 0)
        log(f"  features with positive gradient genes: {n_pos_feats}")
        log(f"  features with negative gradient genes: {n_neg_feats}")

        # For steering: sample 5000 random cells, for each feature and alpha,
        # multiply the feature's activation, decode, measure shift
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_cells, size=min(5000, n_cells), replace=False)
        x_sample = torch.tensor(agg[sample_idx], dtype=torch.float32)

        with torch.no_grad():
            z_baseline, _ = sae.encode(x_sample)
            x_hat_baseline = sae.decode(z_baseline)

        for fid in top_features:
            grad_score = feat_to_gradient_score.get(fid, 0)
            feat_label = ""
            if fid in agg_fa.index:
                feat_label = str(agg_fa.loc[fid].get("top_PanglaoDB_v2", ""))[:40]

            for alpha in ALPHAS:
                with torch.no_grad():
                    z_steered = z_baseline.clone()
                    z_steered[:, fid] = z_steered[:, fid] * alpha
                    x_hat_steered = sae.decode(z_steered)

                    # Measure shift: cosine distance from baseline
                    cos_shift = torch.nn.functional.cosine_similarity(
                        x_hat_baseline, x_hat_steered, dim=1
                    )
                    mean_cos = float(cos_shift.mean())

                    # L2 shift magnitude
                    l2_shift = float((x_hat_steered - x_hat_baseline).norm(dim=1).mean())

                    # Direction: does the shift correlate with gradient features?
                    # Compute the steered representation's projection onto gradient-
                    # associated decoder columns
                    delta = x_hat_steered - x_hat_baseline  # (5000, 64)
                    decoder_w = sae.decoder.weight.data  # (64, 2048)

                    # Project delta onto each feature's decoder direction
                    # and check if positive-gradient features get amplified
                    projections = (delta @ decoder_w).mean(dim=0).numpy()  # (2048,)
                    pos_proj = np.mean([projections[f] for f in feat_to_gradient_score
                                       if feat_to_gradient_score[f] > 0]) if n_pos_feats > 0 else 0
                    neg_proj = np.mean([projections[f] for f in feat_to_gradient_score
                                       if feat_to_gradient_score[f] < 0]) if n_neg_feats > 0 else 0

                results.append({
                    "gradient": grad_name,
                    "feature_idx": fid,
                    "feature_label": feat_label if feat_label != "nan" else "",
                    "alpha": alpha,
                    "mean_cos_shift": mean_cos,
                    "mean_l2_shift": l2_shift,
                    "gradient_score": grad_score,
                    "pos_gene_projection": float(pos_proj),
                    "neg_gene_projection": float(neg_proj),
                    "net_gradient_push": float(pos_proj - neg_proj),
                })

            if (top_features.index(fid) + 1) % 20 == 0:
                log(f"  {top_features.index(fid)+1}/{TOP_FEATURES} features done")

    df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "spatial_steering.parquet", index=False)
    log(f"\nwrote spatial_steering.parquet ({len(df)} rows)")

    # Summary per gradient
    summary = {}
    for grad_name in GRADIENTS:
        sub = df[df["gradient"] == grad_name]
        for alpha in ALPHAS:
            a_sub = sub[sub["alpha"] == alpha]
            n_push_pos = int((a_sub["net_gradient_push"] > 0).sum())
            n_push_neg = int((a_sub["net_gradient_push"] < 0).sum())
            summary[f"{grad_name}_alpha_{alpha}"] = {
                "n_features": len(a_sub),
                "n_push_positive": n_push_pos,
                "n_push_negative": n_push_neg,
                "fraction_positive": round(n_push_pos / max(len(a_sub), 1), 4),
                "mean_net_push": round(float(a_sub["net_gradient_push"].mean()), 6),
                "mean_l2_shift": round(float(a_sub["mean_l2_shift"].mean()), 4),
            }
            log(f"  {grad_name} alpha={alpha}: {n_push_pos} push+ / {n_push_neg} push- "
                f"(net={a_sub['net_gradient_push'].mean():.6f})")

    json.dump(summary, open(OUT_DIR / "spatial_steering.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
