#!/usr/bin/env python
"""GAP 17: SAE-vs-cell-type dissociation experiment.

Test whether biological coherence of circuit edges depends on the SAE
(model-learned structure) or on the cell composition (trivial co-occurrence).

Protocol:
  1. Train a SHUFFLED SAE on the same aggregator activations but with
     randomly permuted cell order (breaking spatial/biological structure).
  2. Compute prototype alignment and top-gene enrichment for the
     shuffled SAE features.
  3. Compare annotation rate and prototype alignment to the real SAE.

If coherence is SAE-dependent: real SAE >> shuffled SAE.
If coherence is trivial (cell-composition-driven): real SAE ~ shuffled.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "25_sae_dissociation.log"

N_STEPS = 10000
BATCH_SIZE = 1024
LR = 3e-4
TOP_GENES = 20
TOP_CELL_FRAC = 0.001
MIN_TOP = 50
N_FEATURES_TO_TEST = 200  # test enrichment on a subset
ENRICHR_LIBS = ["PanglaoDB_Augmented_2021", "GO_Biological_Process_2023"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def train_shuffled_sae(agg_shuffled, d_in=64, n_features=2048, k=16):
    """Train a TopK SAE on shuffled aggregator activations."""
    sae = TopKSAE(d_in=d_in, n_features=n_features, k=k)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    dataset = torch.tensor(agg_shuffled, dtype=torch.float32)
    n = dataset.shape[0]
    rng = np.random.default_rng(99)

    sae.train()
    for step in range(N_STEPS):
        idx = rng.choice(n, size=BATCH_SIZE, replace=False)
        x = dataset[idx]
        x_hat, z, _ = sae(x)
        loss = ((x - x_hat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.renorm_decoder()
        if (step + 1) % 2500 == 0:
            log(f"    step {step+1}: loss={loss.item():.6f}")

    sae.eval()

    with torch.no_grad():
        x_all = dataset[:min(50000, n)]
        x_hat_all, _, _ = sae(x_all)
        var_exp = 1.0 - ((x_all - x_hat_all) ** 2).sum().item() / ((x_all - x_all.mean(0)) ** 2).sum().item()
    log(f"    var_explained = {var_exp:.4f}")
    return sae, var_exp


def compute_annotation_rate(sae, agg, slide_data, n_features_test):
    """Compute annotation rate for a subset of features."""
    n_cells = agg.shape[0]
    n_top = max(MIN_TOP, int(TOP_CELL_FRAC * n_cells))

    # Find top cells per feature
    feat_mean = np.zeros(sae.n_features, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n_cells, 32768):
            e = min(s + 32768, n_cells)
            z, _ = sae.encode(torch.tensor(agg[s:e]))
            feat_mean += np.abs(z.numpy()).sum(axis=0)
    feat_mean /= n_cells
    top_features = np.argsort(-feat_mean)[:n_features_test]

    n_annotated = 0
    for fi, fid in enumerate(top_features):
        # Find top cells
        feat_vals = np.zeros(n_cells, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n_cells, 32768):
                e = min(s + 32768, n_cells)
                z, _ = sae.encode(torch.tensor(agg[s:e]))
                feat_vals[s:e] = np.abs(z[:, fid].numpy())

        nonzero = np.flatnonzero(feat_vals)
        if len(nonzero) < MIN_TOP:
            continue
        kk = min(n_top, len(nonzero))
        top_local = nonzero[np.argpartition(feat_vals[nonzero], -kk)[-kk:]]

        # Get gene expression for top cells
        gene_names, X = slide_data
        from scipy import sparse as sp
        if sp.issparse(X):
            expr = np.asarray(X[top_local].mean(axis=0)).flatten()
        else:
            expr = X[top_local].mean(axis=0).flatten()

        top_genes = [gene_names[gi] for gi in np.argsort(-expr)[:TOP_GENES] if expr[gi] > 0]
        if len(top_genes) < 3:
            continue

        has_sig = False
        for lib in ENRICHR_LIBS:
            try:
                res = gp.enrich(gene_list=top_genes, gene_sets=lib, outdir=None, no_plot=True)
                if res and hasattr(res, "results") and len(res.results) > 0:
                    if float(res.results.iloc[0].get("Adjusted P-value", 1.0)) < 0.05:
                        has_sig = True
                        break
            except Exception:
                pass
        if has_sig:
            n_annotated += 1

        if (fi + 1) % 50 == 0:
            log(f"    {fi+1}/{n_features_test}: {n_annotated} annotated so far")

    return n_annotated, n_features_test


def compute_prototype_alignment(sae, protos):
    """Fraction of SAE features cosine-aligned with any prototype at |cos|>=0.7."""
    decoder_w = sae.decoder.weight.data  # (64, 2048)
    decoder_cols = F.normalize(decoder_w, dim=0)  # normalize each column
    protos_n = F.normalize(protos, dim=1)  # (512, 64)
    sim = (protos_n @ decoder_cols).abs()  # (512, 2048)
    max_cos = sim.max(dim=0).values  # (2048,)
    frac_aligned = float((max_cos >= 0.7).float().mean())
    median_cos = float(max_cos.median())
    return frac_aligned, median_cos


def main():
    log("=" * 60)
    log("GAP 17: SAE-vs-cell-type dissociation")

    # Load real SAE + aggregator
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    log(f"aggregator: {agg.shape}")

    real_sae = TopKSAE(d_in=64, n_features=2048, k=16)
    real_sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    real_sae.eval()

    # Load prototypes
    import novae
    model = novae.Novae.from_pretrained(str(ROOT / "checkpoints" / "novae-human-0"))
    protos = model.swav_head._prototypes.data.detach().clone()
    del model
    gc.collect()

    # Load one slide for gene expression
    import anndata as ad
    import scanpy as sc
    a = ad.read_h5ad(str(ROOT / "datasets" / "mics-lab-novae" / "human" / "brain" / "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.h5ad"))
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    # Use only brain cells (first 24406) for gene expression
    slide_data = (list(a.var_names), a.X)
    brain_agg = agg[:a.n_obs]
    log(f"brain subset: {brain_agg.shape}")
    del a

    # === Real SAE metrics ===
    log("\n--- Real SAE ---")
    real_aligned, real_med_cos = compute_prototype_alignment(real_sae, protos)
    log(f"  prototype alignment: {real_aligned:.4f} (median cos = {real_med_cos:.4f})")

    log("  computing annotation rate (200 features)...")
    real_ann, real_total = compute_annotation_rate(real_sae, brain_agg, slide_data, N_FEATURES_TO_TEST)
    real_rate = real_ann / max(real_total, 1)
    log(f"  annotation rate: {real_ann}/{real_total} = {real_rate:.4f}")

    # === Shuffled SAE ===
    log("\n--- Shuffled SAE ---")
    log("  shuffling cell order...")
    rng = np.random.default_rng(42)
    perm = rng.permutation(agg.shape[0])
    agg_shuffled = agg[perm]

    log("  training shuffled SAE...")
    shuffled_sae, shuffled_var_exp = train_shuffled_sae(agg_shuffled)

    shuffled_aligned, shuffled_med_cos = compute_prototype_alignment(shuffled_sae, protos)
    log(f"  prototype alignment: {shuffled_aligned:.4f} (median cos = {shuffled_med_cos:.4f})")

    # For annotation: use shuffled SAE on the REAL (unshuffled) brain cells
    log("  computing annotation rate (200 features)...")
    shuf_ann, shuf_total = compute_annotation_rate(shuffled_sae, brain_agg, slide_data, N_FEATURES_TO_TEST)
    shuf_rate = shuf_ann / max(shuf_total, 1)
    log(f"  annotation rate: {shuf_ann}/{shuf_total} = {shuf_rate:.4f}")

    # === Comparison ===
    summary = {
        "real_sae": {
            "annotation_rate": round(real_rate, 4),
            "n_annotated": real_ann,
            "n_tested": real_total,
            "prototype_aligned_fraction": round(real_aligned, 4),
            "prototype_median_cos": round(real_med_cos, 4),
        },
        "shuffled_sae": {
            "annotation_rate": round(shuf_rate, 4),
            "n_annotated": shuf_ann,
            "n_tested": shuf_total,
            "prototype_aligned_fraction": round(shuffled_aligned, 4),
            "prototype_median_cos": round(shuffled_med_cos, 4),
            "var_explained": round(shuffled_var_exp, 4),
        },
        "dissociation": {
            "annotation_rate_ratio": round(real_rate / max(shuf_rate, 1e-6), 2),
            "is_sae_dependent": real_rate > shuf_rate * 1.5,
        },
    }
    json.dump(summary, open(OUT_DIR / "sae_dissociation.summary.json", "w"), indent=2)
    log(f"\n{json.dumps(summary, indent=2)}")

    log(f"\nGAP 17 verdict:")
    if summary["dissociation"]["is_sae_dependent"]:
        log(f"  Coherence is SAE-DEPENDENT: real rate {real_rate:.1%} >> shuffled {shuf_rate:.1%}")
    else:
        log(f"  Coherence is NOT clearly SAE-dependent: real {real_rate:.1%} vs shuffled {shuf_rate:.1%}")
    log("DONE")


if __name__ == "__main__":
    main()
