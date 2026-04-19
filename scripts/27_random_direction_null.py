#!/usr/bin/env python
"""Reviewer control: random-direction null for prototype reassignment.

Reviewer concern: claims like "ablating F592 causes 100% prototype
reassignment" do not have a magnitude-matched null. If you ablate a
*random* unit-norm direction in the aggregator with the same L2 norm as
F592's decoder column, what reassignment rate do you get?

Protocol:
  1. Load aggregator SAE and the activations corpus.
  2. For each of the 50 tested features (as in
     prototype_domain_ablation.per_feature.parquet), pick the same cells
     used there (reuse the full parquet).
  3. For each such feature F:
       a. Compute the real ablation effect: subtract feature's contribution
          (z[F] * decoder[:, F]) from the reconstructed aggregator latent.
       b. Compute a matched-magnitude random null: for each cell, subtract
          a random unit-norm direction scaled to |z[F] * decoder[:, F]|_2.
          Repeat N_RAND times; average reassignment rates.
       c. Report real, null_mean, null_CI95, and real - null delta.

Output: atlas/novae-human-0/causal/reviewer_controls/F_random_direction_null.parquet
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
ATLAS = ROOT / "atlas" / "novae-human-0"
OUT_DIR = ATLAS / "causal" / "reviewer_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "27_random_direction_null.log"

N_RAND = 30               # random directions per feature
N_CELLS_PER_FEATURE = 200 # match the original experiment
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
    log("Reviewer control: random-direction null for prototype ablation")

    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    # SwAV prototypes
    protos = model.swav_head._prototypes.data.detach().clone().to(DEVICE)
    protos_n = F.normalize(protos, dim=1)  # (512, 64)

    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.to(DEVICE).eval()

    # Load full aggregator activations + domain labels for baseline proto ids
    log("loading aggregator activations (~1.1GB)")
    agg = np.load(ACT_DIR / "aggregator.npy", mmap_mode="r")     # (N, 64)
    N = agg.shape[0]
    log(f"  N={N}, d={agg.shape[1]}")

    # Load domains for reassignment analysis
    dom_l7 = load_global_domains(7)
    dom_l20 = load_global_domains(20)

    # The selected 50 features used in §4.5/1
    per_feat = pd.read_parquet(
        ATLAS / "causal" / "prototype_domain_ablation.per_feature.parquet"
    )
    feature_ids = per_feat["feature_idx"].tolist()
    log(f"testing {len(feature_ids)} features")

    rng = np.random.default_rng(2026)
    torch.set_grad_enabled(False)
    rows = []

    # Decoder weights (64, 2048)
    W = sae.decoder.weight.data.detach().clone().to(DEVICE)   # (64, 2048)
    preb = sae.pre_bias.data.detach().clone().to(DEVICE)       # (64,)
    db = sae.decoder.bias.data.detach().clone().to(DEVICE)     # (64,)

    # Assign baseline prototypes globally in chunks (cosine-sim argmax)
    log("computing baseline prototype assignments for all cells…")
    CHUNK = 200_000
    proto_of_cell = np.empty(N, dtype=np.int32)
    for start in range(0, N, CHUNK):
        xb = torch.tensor(agg[start:start + CHUNK], dtype=torch.float32, device=DEVICE)
        # Feed through SAE reconstruction (baseline is ideally = raw latent)
        xb_n = F.normalize(xb, dim=1)
        sim = xb_n @ protos_n.T  # (b, 512)
        proto_of_cell[start:start + CHUNK] = sim.argmax(dim=1).cpu().numpy()
    log("  baseline prototype map done")

    # Pick cells per feature: top-activation cells for each feature index
    # Match the original protocol: 200 highest-activating cells per feature.
    # (We approximate via feature decoder vs activation correlation; a full
    # redo would re-SAE-encode the corpus. We instead re-encode top-N cells
    # directly by streaming the SAE forward.)
    log("streaming aggregator through SAE to find top-N cells per target feature…")
    K_TOP = N_CELLS_PER_FEATURE
    # For each target feature we need the top K cells by |z|. Stream:
    top_by_feat = {fid: [] for fid in feature_ids}
    feat_idx_arr = np.asarray(feature_ids, dtype=np.int64)
    feat_idx_t = torch.tensor(feat_idx_arr, dtype=torch.long, device=DEVICE)
    # Heap-free approach: maintain top-K per feature with running mins.
    cur_topk_vals = {fid: np.full(K_TOP, -np.inf, dtype=np.float32) for fid in feature_ids}
    cur_topk_idx = {fid: np.full(K_TOP, -1, dtype=np.int64) for fid in feature_ids}

    for start in range(0, N, CHUNK):
        xb = torch.tensor(agg[start:start + CHUNK], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            z_dense = sae.encode_dense(xb).abs()             # (b, 2048) magnitudes
            z_sel = z_dense.index_select(1, feat_idx_t).detach().cpu().numpy()  # (b, F)
        rng_idx = np.arange(start, start + z_sel.shape[0], dtype=np.int64)
        for j, fid in enumerate(feature_ids):
            vals = z_sel[:, j]
            merged_vals = np.concatenate([cur_topk_vals[fid], vals])
            merged_idx = np.concatenate([cur_topk_idx[fid], rng_idx])
            order = np.argsort(-merged_vals)[:K_TOP]
            cur_topk_vals[fid] = merged_vals[order]
            cur_topk_idx[fid] = merged_idx[order]

    log("  top-K selection done")

    # For each feature: compute real ablation + random-direction ablation
    for fid in feature_ids:
        cells = cur_topk_idx[fid]
        cells = cells[cells >= 0]
        if len(cells) == 0:
            continue
        x_raw = torch.tensor(agg[cells], dtype=torch.float32, device=DEVICE)  # (K, 64)

      # all ops are inference-only
        torch.set_grad_enabled(False)
        # 1) Baseline SAE code + reconstruction
        z_dense = sae.encode_dense(x_raw)                    # (K, 2048)
        _, top_idx = z_dense.abs().topk(sae.k, dim=-1)
        z = torch.zeros_like(z_dense)
        z.scatter_(-1, top_idx, z_dense.gather(-1, top_idx))
        x_hat = sae.decode(z)                                # (K, 64)

        # 2) Real ablation: zero out feature fid in z (if present)
        z_abl = z.clone()
        z_abl[:, fid] = 0.0
        x_abl = sae.decode(z_abl)
        delta_real = x_abl - x_hat                          # (K, 64)
        delta_real_mag = delta_real.norm(dim=1).cpu().numpy()  # per-cell magnitude

        # 3) Random-direction null: for each cell, sample N_RAND unit
        # directions, scale to delta_real_mag, and subtract.
        reassign_real = _reassign_rate(x_hat, x_abl, protos_n, dom_l7[cells], dom_l20[cells])

        rand_rates_proto = []
        rand_rates_l7 = []
        for _ in range(N_RAND):
            dirs = torch.randn_like(x_hat)
            dirs = F.normalize(dirs, dim=1)
            scaled = dirs * torch.tensor(delta_real_mag, device=DEVICE, dtype=torch.float32).unsqueeze(1)
            x_rand = x_hat + scaled       # same magnitude, random direction
            rr = _reassign_rate(x_hat, x_rand, protos_n, dom_l7[cells], dom_l20[cells])
            rand_rates_proto.append(rr["proto"])
            rand_rates_l7.append(rr["l7"])

        rand_proto = float(np.mean(rand_rates_proto))
        rand_l7 = float(np.mean(rand_rates_l7))
        rand_proto_ci = (float(np.percentile(rand_rates_proto, 2.5)),
                         float(np.percentile(rand_rates_proto, 97.5)))
        rand_l7_ci = (float(np.percentile(rand_rates_l7, 2.5)),
                      float(np.percentile(rand_rates_l7, 97.5)))

        rows.append({
            "feature_idx": fid,
            "n_cells": int(len(cells)),
            "mean_ablation_magnitude": float(delta_real_mag.mean()),
            "real_proto_reassign_rate": reassign_real["proto"],
            "rand_proto_reassign_mean": rand_proto,
            "rand_proto_reassign_CI_lo": rand_proto_ci[0],
            "rand_proto_reassign_CI_hi": rand_proto_ci[1],
            "proto_excess_over_random": reassign_real["proto"] - rand_proto,
            "real_l7_reassign_rate": reassign_real["l7"],
            "rand_l7_reassign_mean": rand_l7,
            "rand_l7_reassign_CI_lo": rand_l7_ci[0],
            "rand_l7_reassign_CI_hi": rand_l7_ci[1],
            "l7_excess_over_random": reassign_real["l7"] - rand_l7,
        })

        if len(rows) % 5 == 0:
            log(f"  [{len(rows)}/{len(feature_ids)}] feat {fid}: "
                f"real_proto={reassign_real['proto']:.2f}  null_proto={rand_proto:.2f}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_DIR / "F_random_direction_null.parquet", index=False)
    summary = {
        "n_features_tested": int(len(rows)),
        "median_real_proto_reassign": float(out_df["real_proto_reassign_rate"].median()),
        "median_null_proto_reassign": float(out_df["rand_proto_reassign_mean"].median()),
        "median_excess_over_random": float(out_df["proto_excess_over_random"].median()),
        "n_features_real_above_null_upper_CI": int(
            (out_df["real_proto_reassign_rate"] >
             out_df["rand_proto_reassign_CI_hi"]).sum()
        ),
        "n_features_real_indistinguishable": int(
            (
                (out_df["real_proto_reassign_rate"] <=
                 out_df["rand_proto_reassign_CI_hi"]) &
                (out_df["real_proto_reassign_rate"] >=
                 out_df["rand_proto_reassign_CI_lo"])
            ).sum()
        ),
    }
    (OUT_DIR / "F_random_direction_null.summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log(json.dumps(summary, indent=2))
    log("DONE")


def _reassign_rate(x_full: torch.Tensor, x_abl: torch.Tensor,
                   protos_n: torch.Tensor,
                   l7_labels: np.ndarray, l20_labels: np.ndarray) -> dict:
    x_full_n = F.normalize(x_full, dim=1)
    x_abl_n = F.normalize(x_abl, dim=1)
    proto_full = (x_full_n @ protos_n.T).argmax(dim=1).cpu().numpy()
    proto_abl = (x_abl_n @ protos_n.T).argmax(dim=1).cpu().numpy()
    proto_rate = float((proto_full != proto_abl).mean())
    # Proto → domain map via l7 labels attached to the queried cells' original
    # assignments — this is an approximation; the original experiment uses
    # a global proto → domain map. Proto-rate itself is exact.
    return {"proto": proto_rate, "l7": float("nan")}


if __name__ == "__main__":
    main()
