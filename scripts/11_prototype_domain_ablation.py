#!/usr/bin/env python
"""§4.5/1 — feature hook ablation → prototype + domain reassignment.

For each of the top-50 highest-confidence aggregator SAE features, pick
200 active cells, ablate the feature in the SAE code, decode the
reconstructed (ablated) aggregator latent, and measure:

  1. cos(x_hat_full, x_hat_ablated)         — decoder-attributed impact
  2. prototype reassignment rate (SwAV head, 512 prototypes)
  3. domain-l7 reassignment rate
  4. domain-l20 reassignment rate

Domain reassignment uses a precomputed prototype → domain_l* mapping
derived from the per-cell baseline prototype assignments + per-cell
domain labels (stored in `activations/per_slide/*/domains_level{7,20}.npy`).
Each prototype is mapped to the most common domain among cells assigned
to it. Then ablated prototype → ablated domain via the same map.

"High-confidence" selection: aggregator features with `top_domain_l7_frac
>= 0.4` AND `n_active_cells >= 1000`, sorted by `top_domain_l7_frac`.

Output: `atlas/novae-human-0/causal/prototype_domain_ablation.parquet`
(one row per feature × cell, plus a per-feature summary in
`prototype_domain_ablation.summary.json`).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import novae
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "11_prototype_domain_ablation.log"

N_FEATURES_TO_TEST = 50
N_CELLS_PER_FEATURE = 200
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_global_domains(level: int) -> np.ndarray:
    manifest = json.load(open(ACT_DIR / "manifest.json"))
    chunks = []
    for s in manifest["slides"]:
        path = ACT_DIR / "per_slide" / f"{s['tissue']}__{s['name']}" / f"domains_level{level}.npy"
        chunks.append(np.load(path, allow_pickle=True).astype(object))
    return np.concatenate(chunks)


def main() -> None:
    log("=" * 72)
    log("§4.5/1 prototype + domain ablation")

    log("loading model + SAE")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    protos = model.swav_head._prototypes.data.detach().clone()  # (512, 64)
    protos_n = F.normalize(protos, dim=1)
    log(f"  prototypes: {protos.shape}")
    del model
    gc.collect()

    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()

    log("loading aggregator activations")
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)  # (N, 64)
    n_cells = agg.shape[0]
    log(f"  agg: {agg.shape}")

    log("loading domain labels (l7, l20)")
    l7_global = load_global_domains(7)
    l20_global = load_global_domains(20)
    l7_classes, l7_inv = np.unique(l7_global, return_inverse=True)
    l20_classes, l20_inv = np.unique(l20_global, return_inverse=True)
    log(f"  l7={len(l7_classes)} classes, l20={len(l20_classes)} classes")
    del l7_global, l20_global

    log("selecting high-confidence features")
    fa = pd.read_parquet(ROOT / "atlas" / "novae-human-0" / "feature_atlas_full.parquet")
    agg_fa = fa[fa["surface"] == "aggregator"].copy()
    candidates = agg_fa[
        (agg_fa["top_domain_l7_frac"] >= 0.4)
        & (agg_fa["n_active_cells"] >= 1000)
    ].sort_values("top_domain_l7_frac", ascending=False)
    selected = candidates.head(N_FEATURES_TO_TEST)
    feature_ids = selected["feature_idx"].astype(int).tolist()
    log(f"  selected {len(feature_ids)} features (top_domain_l7_frac >= 0.4, n_active >= 1000)")
    log(f"  min l7 frac: {selected['top_domain_l7_frac'].min():.3f}, max: {selected['top_domain_l7_frac'].max():.3f}")

    log("computing baseline prototype assignments for all cells")
    # Argmax cos to prototypes on the RAW aggregator output.
    t0 = time.time()
    protos_device_n = protos_n.to(DEVICE)
    cell_proto = np.empty(n_cells, dtype=np.int32)
    chunk = 65536
    with torch.no_grad():
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            x = torch.tensor(agg[start:end], dtype=torch.float32, device=DEVICE)
            x_n = F.normalize(x, dim=1)
            sim = x_n @ protos_device_n.t()
            cell_proto[start:end] = sim.argmax(dim=1).cpu().numpy().astype(np.int32)
    log(f"  baseline assignments done in {time.time()-t0:.1f}s")

    # Build prototype → domain-l7 and prototype → domain-l20 maps by majority vote
    log("building prototype → domain-l* mapping by majority vote")
    proto_to_l7 = np.full(protos.shape[0], -1, dtype=np.int32)
    proto_to_l20 = np.full(protos.shape[0], -1, dtype=np.int32)
    for p in range(protos.shape[0]):
        mask = cell_proto == p
        if not mask.any():
            continue
        l7_here = l7_inv[mask]
        l20_here = l20_inv[mask]
        proto_to_l7[p] = int(np.bincount(l7_here).argmax())
        proto_to_l20[p] = int(np.bincount(l20_here).argmax())

    cov_l7 = (proto_to_l7 >= 0).mean()
    log(f"  {cov_l7*100:.1f}% of prototypes have at least one assigned cell")

    log("running per-feature ablation")
    rows = []
    per_feature_stats = []

    sae.to(DEVICE)
    protos_dev = protos.to(DEVICE)
    protos_n_dev = protos_n.to(DEVICE)

    for i, fid in enumerate(feature_ids):
        # Find N_CELLS_PER_FEATURE cells with largest feature activation
        # Encode in chunks and track the top-N per feature.
        t0 = time.time()

        # Do one pass over agg to find top-N active cells for feature fid
        top_k = N_CELLS_PER_FEATURE
        top_vals = np.full(top_k, -np.inf, dtype=np.float32)
        top_ix = np.full(top_k, -1, dtype=np.int64)
        with torch.no_grad():
            for start in range(0, n_cells, chunk):
                end = min(start + chunk, n_cells)
                xb = torch.tensor(agg[start:end], dtype=torch.float32, device=DEVICE)
                z, _ = sae.encode(xb)
                fvals = z[:, fid].abs().cpu().numpy()
                # Find cells in this chunk whose fvals exceed top_vals.min()
                thr = top_vals.min()
                cand_mask = fvals > thr
                if not cand_mask.any():
                    continue
                cand_vals = fvals[cand_mask]
                cand_ix = np.nonzero(cand_mask)[0] + start
                # Combine with existing top and re-select
                combined_vals = np.concatenate([top_vals, cand_vals])
                combined_ix = np.concatenate([top_ix, cand_ix])
                keep = np.argpartition(combined_vals, -top_k)[-top_k:]
                top_vals = combined_vals[keep]
                top_ix = combined_ix[keep]
        # Order top_ix by descending value
        order = np.argsort(-top_vals)
        top_vals = top_vals[order]
        top_ix = top_ix[order]
        valid = top_ix >= 0
        top_ix = top_ix[valid]
        if len(top_ix) < 10:
            log(f"  [{i+1}/{len(feature_ids)}] feature {fid}: too few active cells, skipping")
            continue

        # Load those cells' aggregator latents
        x = torch.tensor(agg[top_ix], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            z_full, _ = sae.encode(x)
            x_hat_full = sae.decode(z_full)
            z_abl = z_full.clone()
            z_abl[:, fid] = 0.0
            x_hat_abl = sae.decode(z_abl)

            # cosine between full and ablated reconstruction
            cos_full_abl = F.cosine_similarity(x_hat_full, x_hat_abl, dim=1).cpu().numpy()

            # Prototype assignment for both
            x_hat_full_n = F.normalize(x_hat_full, dim=1)
            x_hat_abl_n = F.normalize(x_hat_abl, dim=1)
            sim_full = x_hat_full_n @ protos_n_dev.t()
            sim_abl = x_hat_abl_n @ protos_n_dev.t()
            proto_full = sim_full.argmax(dim=1).cpu().numpy()
            proto_abl = sim_abl.argmax(dim=1).cpu().numpy()

            # Top-1 prototype confidence (max sim value)
            conf_full = sim_full.max(dim=1).values.cpu().numpy()
            conf_abl = sim_abl.max(dim=1).values.cpu().numpy()

        # Map to domains
        l7_full = proto_to_l7[proto_full]
        l7_abl = proto_to_l7[proto_abl]
        l20_full = proto_to_l20[proto_full]
        l20_abl = proto_to_l20[proto_abl]

        # Per-cell row
        for j in range(len(top_ix)):
            rows.append({
                "feature_idx": fid,
                "cell_idx": int(top_ix[j]),
                "feat_val": float(top_vals[j]),
                "cos_full_vs_ablated": float(cos_full_abl[j]),
                "proto_full": int(proto_full[j]),
                "proto_abl": int(proto_abl[j]),
                "proto_changed": bool(proto_full[j] != proto_abl[j]),
                "proto_conf_full": float(conf_full[j]),
                "proto_conf_abl": float(conf_abl[j]),
                "l7_full": int(l7_full[j]) if l7_full[j] >= 0 else -1,
                "l7_abl": int(l7_abl[j]) if l7_abl[j] >= 0 else -1,
                "l7_changed": bool(l7_full[j] != l7_abl[j]) if (l7_full[j] >= 0 and l7_abl[j] >= 0) else False,
                "l20_full": int(l20_full[j]) if l20_full[j] >= 0 else -1,
                "l20_abl": int(l20_abl[j]) if l20_abl[j] >= 0 else -1,
                "l20_changed": bool(l20_full[j] != l20_abl[j]) if (l20_full[j] >= 0 and l20_abl[j] >= 0) else False,
            })

        proto_reassign_rate = float((proto_full != proto_abl).mean())
        l7_reassign_rate = float((l7_full != l7_abl).mean())
        l20_reassign_rate = float((l20_full != l20_abl).mean())
        mean_cos = float(cos_full_abl.mean())
        mean_conf_drop = float((conf_full - conf_abl).mean())

        per_feature_stats.append({
            "feature_idx": fid,
            "n_cells": len(top_ix),
            "mean_cos_full_vs_ablated": mean_cos,
            "mean_conf_drop": mean_conf_drop,
            "proto_reassign_rate": proto_reassign_rate,
            "l7_reassign_rate": l7_reassign_rate,
            "l20_reassign_rate": l20_reassign_rate,
        })

        if (i + 1) % 5 == 0 or i == len(feature_ids) - 1:
            log(
                f"  [{i+1}/{len(feature_ids)}] feat {fid}: "
                f"cos={mean_cos:.3f} proto={proto_reassign_rate*100:.1f}% "
                f"l7={l7_reassign_rate*100:.1f}% l20={l20_reassign_rate*100:.1f}% "
                f"({time.time()-t0:.1f}s)"
            )

    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "prototype_domain_ablation.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {out_path} ({len(df)} rows)")

    per_feat_df = pd.DataFrame(per_feature_stats)
    per_feat_path = OUT_DIR / "prototype_domain_ablation.per_feature.parquet"
    per_feat_df.to_parquet(per_feat_path, index=False)
    log(f"wrote {per_feat_path} ({len(per_feat_df)} rows)")

    summary = {
        "n_features_tested": int(len(per_feat_df)),
        "n_cells_per_feature": N_CELLS_PER_FEATURE,
        "mean_cos_full_vs_ablated": float(per_feat_df["mean_cos_full_vs_ablated"].mean()),
        "mean_proto_reassign_rate": float(per_feat_df["proto_reassign_rate"].mean()),
        "mean_l7_reassign_rate": float(per_feat_df["l7_reassign_rate"].mean()),
        "mean_l20_reassign_rate": float(per_feat_df["l20_reassign_rate"].mean()),
        "n_features_proto_reassign_gt_0.5": int(
            (per_feat_df["proto_reassign_rate"] > 0.5).sum()
        ),
        "n_features_l7_reassign_gt_0.3": int(
            (per_feat_df["l7_reassign_rate"] > 0.3).sum()
        ),
    }
    json.dump(summary, open(OUT_DIR / "prototype_domain_ablation.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
