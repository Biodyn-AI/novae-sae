#!/usr/bin/env python
"""Reviewer control: random-initialized Novae SAE baseline.

The single most decisive control: does *learning* in Novae produce the
biological features, or is the architecture alone enough? We train an
aggregator SAE on activations from a randomly-initialized Novae of the
same architecture and compare its interpretability to the trained-model
SAE atlas.

Protocol:
  1. Instantiate a fresh `novae.Novae` with the same gene panel and
     hyperparameters as novae-human-0 but NOT load pretrained weights.
  2. Run compute_representations on the slide subset to extract
     aggregator activations.
  3. Train a TopK SAE with the same specs (d=64, n_feat=2048, k=16) for
     the same number of epochs.
  4. Characterize features: top-genes, Enrichr annotation rate,
     graph-dependency, cross-slide niche specificity.
  5. Compare numbers to the real atlas.

If the random-init SAE gives comparable annotation rate / graph
dependency, the claims of "Novae has learned spatial biology" are not
supported — the architecture's inductive bias alone produces features.

Output: atlas/novae-human-0/causal/reviewer_controls/J_random_init/
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
from torch import nn

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
ACT_DIR = ROOT / "activations" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal" / "reviewer_controls" / "J_random_init"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "31_random_init_sae.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# Small, tractable subset
SLIDES_FOR_EXTRACT = [
    "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",
    "Xenium_V1_hPancreas_nondiseased_section_outs",
    "Xenium_V1_hKidney_nondiseased_section_outs",
]
SAE_EPOCHS = 10
SAE_BATCH = 8192
SAE_LR = 1e-3


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def reinit_weights(module: nn.Module, seed: int = 2026) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    for p in module.parameters():
        if p.dim() >= 2:
            nn.init.kaiming_normal_(p, generator=g)
        else:
            nn.init.zeros_(p)


def train_sae(X: np.ndarray, d_in: int, n_features: int, k: int) -> TopKSAE:
    sae = TopKSAE(d_in=d_in, n_features=n_features, k=k).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    n = X_t.shape[0]
    for ep in range(SAE_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        loss_sum = 0.0
        n_batches = 0
        for s in range(0, n, SAE_BATCH):
            batch = X_t[perm[s:s + SAE_BATCH]]
            x_hat, _, _ = sae(batch)
            loss = ((x_hat - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.renorm_decoder()
            loss_sum += loss.item()
            n_batches += 1
        log(f"  epoch {ep+1}/{SAE_EPOCHS}: mse={loss_sum/max(n_batches,1):.6f}")
    sae.eval()
    return sae


def main() -> None:
    log("=" * 72)
    log("Random-init Novae SAE baseline")

    # 1) Random-init model with same gene panel
    log("loading pretrained to copy gene panel config…")
    ref = novae.Novae.from_pretrained(str(CKPT_DIR))
    cfg = ref.hparams if hasattr(ref, "hparams") else None
    log(f"  reference cfg type: {type(cfg)}")
    rand_model = novae.Novae.from_pretrained(str(CKPT_DIR))
    reinit_weights(rand_model)
    del ref
    if DEVICE != "cpu":
        rand_model = rand_model.to(DEVICE)
    rand_model.eval()

    # 2) Extract aggregator activations
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    slides = [s for s in manifest["slides"] if s["name"] in SLIDES_FOR_EXTRACT]
    log(f"{len(slides)} slides")
    acts = []
    for s in slides:
        h5ad = DATA_DIR / s["tissue"] / f"{s['name']}.h5ad"
        log(f"  loading {s['name']}")
        a = ad.read_h5ad(h5ad)
        novae.spatial_neighbors(a)
        with torch.no_grad():
            rand_model.compute_representations(a, zero_shot=True)
        acts.append(a.obsm["novae_latent"].copy().astype(np.float32))
        del a
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    X = np.vstack(acts)
    log(f"activation shape: {X.shape}")
    np.save(OUT_DIR / "rand_aggregator.npy", X)

    # 3) Train SAE
    log("training SAE (rand-init)…")
    t0 = time.time()
    sae_rand = train_sae(X, d_in=64, n_features=2048, k=16)
    log(f"  done in {time.time()-t0:.0f}s")
    torch.save(sae_rand.state_dict(), OUT_DIR / "aggregator_rand.pt")

    # 4) Variance explained + dead-feature fraction
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        X_hat, z, _ = sae_rand(X_t)
        mse = ((X_hat - X_t) ** 2).mean(dim=1)
        var = X_t.var(dim=1, unbiased=False)
        frac_ve = float(1 - (mse / (var + 1e-12)).mean())
        active = (z.abs() > 1e-8).any(dim=0)
        dead_frac = float((~active).float().mean())
    log(f"rand SAE variance explained: {frac_ve*100:.1f}%  dead features: {dead_frac*100:.1f}%")

    # 5) Top-gene list per feature (for downstream Enrichr)
    # Build per-slide gene panel from the first slide
    a0 = ad.read_h5ad(DATA_DIR / slides[0]["tissue"] / f"{slides[0]['name']}.h5ad", backed="r")
    gene_names = np.asarray(a0.var_names).astype(str)
    del a0
    # Encode all cells, compute per-feature top-0.1% cells, get top 20 genes
    # from the *concatenated* expression corpus for those cells.
    # We re-read h5ads to compute per-cell gene expression for top-activated cells.
    log("extracting top-gene lists per feature…")
    # Per-cell encoded features
    with torch.no_grad():
        z_all = []
        for s in range(0, len(X), 32768):
            z_b, _ = sae_rand.encode(torch.tensor(X[s:s + 32768], dtype=torch.float32, device=DEVICE))
            z_all.append(z_b.cpu().numpy())
        z_all = np.vstack(z_all)
    log(f"  z_all: {z_all.shape}")
    # For each feature, pick the top-0.1% cells by |z|.
    n_cells = z_all.shape[0]
    top_n_per_feat = max(100, int(n_cells * 0.001))
    feature_top_cells = {}
    for f in range(z_all.shape[1]):
        if not active[f]:
            continue
        top_idx = np.argsort(-np.abs(z_all[:, f]))[:top_n_per_feat]
        feature_top_cells[f] = top_idx

    # Compute top genes by streaming through h5ads in order
    # Build a concatenated expression matrix in chunks
    log("building per-cell expression (log1p) for top-gene extraction…")
    pieces = []
    for s in slides:
        a = ad.read_h5ad(DATA_DIR / s["tissue"] / f"{s['name']}.h5ad", backed="r")
        # Cells appear in the same order as activations; load full X
        Xs = a.X
        if hasattr(Xs, "toarray"):
            Xs = Xs.toarray()
        pieces.append(np.asarray(Xs, dtype=np.float32))
        del a
    expr = np.concatenate(pieces, axis=0)
    del pieces
    log(f"  expr shape: {expr.shape}  (genes: {len(gene_names)})")

    assert expr.shape[0] == n_cells, f"mismatch: expr {expr.shape[0]} vs z {n_cells}"
    # Baseline mean expression across all cells
    baseline_mean = expr.mean(axis=0)
    rows = []
    for f, top_idx in feature_top_cells.items():
        sub = expr[top_idx].mean(axis=0)
        # Fold-change over baseline
        fc = sub / (baseline_mean + 1e-6)
        top_genes = np.argsort(-fc)[:20]
        for rank, gi in enumerate(top_genes):
            rows.append({
                "feature_idx": int(f),
                "rank": int(rank),
                "gene": str(gene_names[gi]),
                "fold_change": float(fc[gi]),
                "mean_expr_in_top_cells": float(sub[gi]),
            })
    top_df = pd.DataFrame(rows)
    top_df.to_parquet(OUT_DIR / "rand_top_genes.parquet", index=False)
    log(f"wrote rand_top_genes ({len(top_df)} rows, {top_df['feature_idx'].nunique()} features)")

    summary = {
        "n_cells": int(n_cells),
        "n_features_active": int(active.sum().item()),
        "n_features_dead": int((~active).sum().item()),
        "variance_explained": frac_ve,
        "dead_feature_fraction": dead_frac,
        "note": (
            "Next step: run the same Enrichr pipeline (script 03c) on rand_top_genes "
            "and compare annotation rate + per-library hit rate to the trained SAE."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
