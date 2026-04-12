#!/usr/bin/env python
"""Phase 2 — train one TopK SAE per Novae activation surface.

Following upstream `arXiv:2603.02952` recipe with the small-layer adaptations
described in `RESEARCH_AND_IMPLEMENTATION_PLAN.md` §4.3:

  - TopK SAE
  - 32x expansion factor for the small (d=64..128) layers, 16x for d=512
  - k=16 for d=64 surfaces, k=32 for the rest
  - Adam, lr 3e-4, batch 4096, 5 epochs, MSE loss, decoder column renorm
  - Trains on MPS when available, falls back to CPU otherwise

Reads activation arrays produced by `01_extract_activations.py` from
`activations/novae-human-0/{aggregator,conv_0..9,cell_embedder}.npy` and writes
trained weights to `saes/novae-human-0/<surface>.pt`, plus a `summary.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# make `src/topk_sae` importable when running this script directly
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
LOG_PATH = ROOT / "logs" / "02_train_saes.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Per-surface hyperparameters: (surface_name, expansion_factor, k)
SAE_CONFIGS: list[tuple[str, int, int]] = [
    ("aggregator",     32, 16),  # d=64,  F=2048   PRIMARY (cell-in-niche)
    ("conv_0",         32, 32),  # d=128, F=4096
    ("conv_1",         32, 32),
    ("conv_2",         32, 32),
    ("conv_3",         32, 32),
    ("conv_4",         32, 32),
    ("conv_5",         32, 32),
    ("conv_6",         32, 32),
    ("conv_7",         32, 32),
    ("conv_8",         32, 32),
    ("conv_9",         32, 16),  # d=64,  F=2048   compression layer
    ("cell_embedder",  16, 32),  # d=512, F=8192   pre-graph (scGPT space)
]

LR = 3e-4
EPOCHS = 5
BATCH_SIZE = 4096
VAL_FRACTION = 0.05
SEED = 42


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def train_one(name: str, X: np.ndarray, expansion: int, k: int) -> tuple[TopKSAE, dict]:
    d = X.shape[1]
    n_features = expansion * d
    n_total = X.shape[0]

    log(f"  config       : d={d}, F={n_features} ({expansion}x), k={k}, n={n_total:,}")

    # train/val split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_val = max(1024, int(VAL_FRACTION * n_total))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(DEVICE)
    val_var = X_val.var().item()

    sae = TopKSAE(d_in=d, n_features=n_features, k=k).to(DEVICE)

    # initialize pre_bias to the mean of training data (Anthropic recipe)
    with torch.no_grad():
        sae.pre_bias.copy_(X_train.mean(dim=0).to(DEVICE))

    optim = torch.optim.Adam(sae.parameters(), lr=LR)

    history: list[dict] = []
    n_train = X_train.shape[0]
    steps_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        t_epoch = time.time()
        sae.train()
        ep_perm = torch.randperm(n_train)
        ep_loss = 0.0
        for step in range(steps_per_epoch):
            batch_idx = ep_perm[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            batch = X_train[batch_idx].to(DEVICE)
            x_hat, _z, _idx = sae(batch)
            loss = F.mse_loss(x_hat, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sae.renorm_decoder()
            ep_loss += float(loss.detach().cpu())

        ep_loss /= steps_per_epoch

        sae.eval()
        with torch.no_grad():
            x_hat_v, _z, _idx = sae(X_val)
            val_mse = float(F.mse_loss(x_hat_v, X_val).cpu())
            var_explained = 1.0 - val_mse / val_var if val_var > 0 else float("nan")
            alive = sae.feature_alive_mask(X_val).sum().item()

        log(
            f"    epoch {epoch+1}/{EPOCHS}: train_mse={ep_loss:.4f}  "
            f"val_mse={val_mse:.4f}  var_exp={var_explained:.3f}  "
            f"alive={alive}/{n_features}  ({time.time()-t_epoch:.1f}s)"
        )
        history.append({
            "epoch": epoch + 1,
            "train_mse": ep_loss,
            "val_mse": val_mse,
            "var_explained": var_explained,
            "alive": int(alive),
        })

    final = history[-1]
    summary = {
        "name": name,
        "d_in": d,
        "n_features": n_features,
        "expansion": expansion,
        "k": k,
        "n_train": int(n_train),
        "n_val": int(X_val.shape[0]),
        "val_mse_final": final["val_mse"],
        "var_explained_final": final["var_explained"],
        "alive_final": final["alive"],
        "alive_fraction": final["alive"] / n_features,
        "history": history,
    }
    return sae, summary


def main() -> None:
    log("=" * 72)
    log(f"Phase 2: SAE training")
    log(f"  device : {DEVICE}")
    log(f"  in     : {ACT_DIR}")
    log(f"  out    : {SAE_DIR}")
    log(f"  configs: {len(SAE_CONFIGS)}")

    SAE_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    t_global = time.time()

    for name, expansion, k in SAE_CONFIGS:
        out_pt = SAE_DIR / f"{name}.pt"
        if out_pt.exists():
            log(f"\n=== {name} === SKIP (already trained at {out_pt})")
            continue

        path = ACT_DIR / f"{name}.npy"
        if not path.exists():
            log(f"\n=== {name} === MISSING activation file: {path}")
            continue

        log(f"\n=== {name} ===")
        log(f"  loading {path}")
        X = np.load(path).astype(np.float32, copy=False)

        sae, summary = train_one(name, X, expansion, k)

        torch.save(sae.state_dict(), out_pt)
        json.dump(summary, open(SAE_DIR / f"{name}.summary.json", "w"), indent=2)
        summaries.append(summary)

        del X, sae
        if DEVICE == "mps":
            torch.mps.empty_cache()

    json.dump(
        {"device": DEVICE, "lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
         "summaries": summaries},
        open(SAE_DIR / "summary.json", "w"),
        indent=2,
    )

    log(f"\nDONE in {(time.time()-t_global)/60:.1f} min")
    if summaries:
        log(f"\n{'name':18s} {'d':>5s} {'F':>6s} {'k':>4s} {'val_mse':>9s} {'var_exp':>8s} {'alive':>10s}")
        for s in summaries:
            log(
                f"{s['name']:18s} {s['d_in']:>5d} {s['n_features']:>6d} {s['k']:>4d} "
                f"{s['val_mse_final']:>9.4f} {s['var_explained_final']:>8.3f} "
                f"{s['alive_final']:>5d}/{s['n_features']:<5d}"
            )


if __name__ == "__main__":
    main()
