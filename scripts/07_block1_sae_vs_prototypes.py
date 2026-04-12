#!/usr/bin/env python
"""Block 1.1 — SAE vs SwAV prototype alignment (H6).

For every aggregator SAE decoder column (2048), compute cosine similarity
against every Novae SwAV prototype (512). Per feature: max similarity to
any prototype + the index of the best-matching one.

H6 prediction: <30% of SAE features should be cosine-aligned (>0.7) with
any prototype. If H6 holds, the SAE is finding structure that Novae's own
clustering head does NOT expose.

Output: atlas/causal/sae_vs_prototypes.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import novae
import numpy as np
import pandas as pd
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "07_block1_sae_vs_prototypes.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main() -> None:
    log("=" * 72)
    log("Block 1.1: SAE vs SwAV prototype alignment")

    log("loading model + prototypes")
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    prototypes = model.swav_head._prototypes.detach().cpu().numpy()  # (512, 64)
    log(f"  prototypes: shape={prototypes.shape}")

    log("loading aggregator SAE")
    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()
    decoder = sae.decoder.weight.detach().cpu().numpy()  # (d_in, n_features) = (64, 2048)
    decoder = decoder.T  # (n_features, d_in) = (2048, 64)
    log(f"  decoder: shape={decoder.shape}")

    # Normalize both
    proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    dec_norm = decoder / (np.linalg.norm(decoder, axis=1, keepdims=True) + 1e-9)

    # Cosine similarity matrix: (n_features, n_prototypes)
    cos = dec_norm @ proto_norm.T  # (2048, 512)
    log(f"  cosine matrix: shape={cos.shape}")

    abs_cos = np.abs(cos)
    max_abs_per_feature = abs_cos.max(axis=1)
    best_proto_per_feature = abs_cos.argmax(axis=1)
    signed_max_per_feature = np.array([
        cos[i, best_proto_per_feature[i]] for i in range(len(cos))
    ])

    # Build per-feature dataframe
    rows = []
    for i in range(len(cos)):
        rows.append({
            "feature_idx": int(i),
            "max_abs_cos": float(max_abs_per_feature[i]),
            "best_prototype_idx": int(best_proto_per_feature[i]),
            "signed_cos_to_best": float(signed_max_per_feature[i]),
        })
    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sae_vs_prototypes.parquet"
    df.to_parquet(out_path, index=False)
    log(f"  wrote {out_path} ({len(df)} rows)")

    # Summary
    n_total = len(df)
    n_aligned_07 = int((df["max_abs_cos"] > 0.7).sum())
    n_aligned_05 = int((df["max_abs_cos"] > 0.5).sum())
    n_aligned_03 = int((df["max_abs_cos"] > 0.3).sum())
    summary = {
        "n_features": n_total,
        "n_prototypes": int(prototypes.shape[0]),
        "fraction_aligned_at_0.7": n_aligned_07 / n_total,
        "fraction_aligned_at_0.5": n_aligned_05 / n_total,
        "fraction_aligned_at_0.3": n_aligned_03 / n_total,
        "median_max_cos": float(df["max_abs_cos"].median()),
        "mean_max_cos": float(df["max_abs_cos"].mean()),
    }
    json.dump(summary, open(OUT_DIR / "sae_vs_prototypes.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))

    log("\nH6 evaluation: <30% of SAE features should be cos-aligned (>0.7) with any prototype")
    log(f"  result: {summary['fraction_aligned_at_0.7'] * 100:.1f}% aligned at 0.7 threshold")
    if summary["fraction_aligned_at_0.7"] < 0.30:
        log(f"  → H6 CONFIRMED")
    else:
        log(f"  → H6 REFUTED")

    log("DONE")


if __name__ == "__main__":
    main()
