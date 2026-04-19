#!/usr/bin/env python
"""Reviewer control: SAE seed stability.

Retrain the aggregator SAE with 2 additional random seeds and measure
feature-dictionary overlap. If the top-50 causally-important features
disappear or reshuffle across seeds, claims tied to specific feature IDs
are not reproducible.

Protocol:
  1. Load the existing aggregator activations (4.5M × 64).
  2. Train 2 new SAEs with different seeds (same hparams, same data).
  3. For each (seed1, seed2) pair, compute Hungarian-matched
     feature-to-feature cosine similarity on decoder columns.
  4. For the top-50 features from the primary atlas, find their best
     matches in each replicate and report.

Output: atlas/novae-human-0/causal/reviewer_controls/K_seed_stability.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
ATLAS = ROOT / "atlas" / "novae-human-0"
OUT_DIR = ATLAS / "causal" / "reviewer_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "32_sae_seed_stability.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = [1, 2]  # primary = implicit seed 0 in existing checkpoint
N_EPOCHS = 5
BATCH = 8192
LR = 1e-3


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def train_sae(X: np.ndarray, seed: int, d_in: int = 64, n_feat: int = 2048, k: int = 16) -> TopKSAE:
    torch.manual_seed(seed)
    sae = TopKSAE(d_in=d_in, n_features=n_feat, k=k).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    n = X_t.shape[0]
    for ep in range(N_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        mse_sum, nb = 0.0, 0
        for s in range(0, n, BATCH):
            batch = X_t[perm[s:s + BATCH]]
            x_hat, _, _ = sae(batch)
            loss = ((x_hat - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.renorm_decoder()
            mse_sum += loss.item()
            nb += 1
        log(f"  seed{seed} ep{ep+1}/{N_EPOCHS} mse={mse_sum/max(nb,1):.6f}")
    sae.eval()
    return sae


def feature_dict_overlap(sae_a: TopKSAE, sae_b: TopKSAE) -> dict:
    """Hungarian match decoder columns (cosine similarity)."""
    Wa = sae_a.decoder.weight.data.detach().cpu().numpy()  # (d, n)
    Wb = sae_b.decoder.weight.data.detach().cpu().numpy()
    # L2-normalize columns
    Wa = Wa / (np.linalg.norm(Wa, axis=0, keepdims=True) + 1e-8)
    Wb = Wb / (np.linalg.norm(Wb, axis=0, keepdims=True) + 1e-8)
    # Cosine-sim matrix (n, n)
    S = np.abs(Wa.T @ Wb)
    # Hungarian maximization
    row_ind, col_ind = linear_sum_assignment(-S)
    matched = S[row_ind, col_ind]
    return {
        "mean_match": float(matched.mean()),
        "median_match": float(np.median(matched)),
        "frac_match_gt_0.9": float((matched > 0.9).mean()),
        "frac_match_gt_0.7": float((matched > 0.7).mean()),
        "frac_match_gt_0.5": float((matched > 0.5).mean()),
    }


def main() -> None:
    log("=" * 72)
    log("SAE seed stability")
    log("loading aggregator activations (mmap)…")
    agg = np.load(ACT_DIR / "aggregator.npy", mmap_mode="r")
    # Subsample 500k cells for speed
    N = min(agg.shape[0], 500_000)
    rng = np.random.default_rng(2026)
    idx = rng.choice(agg.shape[0], size=N, replace=False)
    X = np.asarray(agg[idx]).astype(np.float32)
    log(f"  subsample: {X.shape}")

    # Load primary SAE (existing)
    log("loading primary (trained) SAE…")
    sae_primary = TopKSAE(d_in=64, n_features=2048, k=16)
    sae_primary.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae_primary.to(DEVICE).eval()

    # Train replicates
    saes_by_seed = {"primary": sae_primary}
    for seed in SEEDS:
        log(f"\ntraining SAE seed={seed}")
        t0 = time.time()
        sae_r = train_sae(X, seed)
        log(f"  done in {time.time()-t0:.0f}s")
        torch.save(sae_r.state_dict(), OUT_DIR / f"aggregator_seed{seed}.pt")
        saes_by_seed[f"seed{seed}"] = sae_r

    # Pairwise overlap
    overlaps = {}
    keys = list(saes_by_seed.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            stats = feature_dict_overlap(saes_by_seed[a], saes_by_seed[b])
            overlaps[f"{a}_VS_{b}"] = stats
            log(f"  {a} vs {b}: {stats}")

    # Top-50 features from primary: find best match in each replicate
    try:
        per_feat = pd.read_parquet(ATLAS / "causal" / "prototype_domain_ablation.per_feature.parquet")
        top50 = per_feat["feature_idx"].tolist()
    except FileNotFoundError:
        top50 = list(range(50))

    Wp = sae_primary.decoder.weight.data.detach().cpu().numpy()
    Wp = Wp / (np.linalg.norm(Wp, axis=0, keepdims=True) + 1e-8)
    top50_tracking = {}
    for seed in SEEDS:
        sae_r = saes_by_seed[f"seed{seed}"]
        Wr = sae_r.decoder.weight.data.detach().cpu().numpy()
        Wr = Wr / (np.linalg.norm(Wr, axis=0, keepdims=True) + 1e-8)
        # For each top-50 primary feature, find its best cosine match in replicate
        S = np.abs(Wp[:, top50].T @ Wr)  # (50, n)
        best_match = S.max(axis=1)
        top50_tracking[f"seed{seed}"] = {
            "mean_best_match": float(best_match.mean()),
            "frac_match_gt_0.7": float((best_match > 0.7).mean()),
            "frac_match_gt_0.5": float((best_match > 0.5).mean()),
            "per_feature_best_match": [
                {"feature_idx": int(f), "best_match": float(m)}
                for f, m in zip(top50, best_match)
            ],
        }

    summary = {
        "n_cells_trained_on": int(N),
        "n_epochs": N_EPOCHS,
        "pairwise_feature_overlap": overlaps,
        "top50_feature_tracking": top50_tracking,
    }
    (OUT_DIR / "K_sae_seed_stability.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({k: v for k, v in summary.items() if k != "top50_feature_tracking"}, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
