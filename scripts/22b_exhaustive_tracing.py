#!/usr/bin/env python
"""GAP 13: Exhaustive circuit tracing at conv_5.

Trace the top-200 most active features at conv_5 (instead of just 30
in the original run). This reveals the heavy-tailed hub distribution
and annotation bias predicted by arXiv:2603.11940.

Uses the same protocol as script 19 but with more source features
and only one source layer.
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

ROOT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal"
LOG_PATH = ROOT / "logs" / "22b_exhaustive.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SLIDE = ("brain", "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs")
SOURCE_LAYER = 5
TOP_FEATURES = 200
NUM_CONVS = 10
D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.6

SAE_SPECS = {
    **{f"conv_{i}": (128, 4096, 32) for i in range(9)},
    "conv_9": (64, 2048, 16),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    log("=" * 60)
    log(f"GAP 13: Exhaustive circuit tracing at conv_{SOURCE_LAYER}")

    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    # Load SAEs
    saes = {}
    for i in range(NUM_CONVS):
        name = f"conv_{i}"
        d_in, n_feat, k = SAE_SPECS[name]
        sae = TopKSAE(d_in=d_in, n_features=n_feat, k=k)
        sae.load_state_dict(torch.load(SAE_DIR / f"{name}.pt", map_location="cpu"))
        sae.eval()
        saes[i] = sae

    # Load slide
    tissue, slide_name = SLIDE
    adata = ad.read_h5ad(DATA_DIR / tissue / f"{slide_name}.h5ad")
    novae.spatial_neighbors(adata)
    log(f"slide: {adata.n_obs} cells")

    # Select top features at source layer
    per_cell_path = ACT_DIR / "per_slide" / f"{tissue}__{slide_name}" / f"conv_{SOURCE_LAYER}_percell.npy"
    act = np.load(per_cell_path).astype(np.float32, copy=False)
    src_sae = saes[SOURCE_LAYER]
    feat_mean = np.zeros(src_sae.n_features, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, act.shape[0], 32768):
            e = min(s + 32768, act.shape[0])
            z, _ = src_sae.encode(torch.tensor(act[s:e]))
            feat_mean += np.abs(z.numpy()).sum(axis=0)
    feat_mean /= act.shape[0]
    source_features = np.argsort(-feat_mean)[:TOP_FEATURES].tolist()
    del act
    log(f"selected top-{TOP_FEATURES} features at conv_{SOURCE_LAYER}")

    # Baseline forward pass
    gnn = model.encoder.gnn
    baseline_encoded = {}

    def make_capture(layer_idx, store):
        def hook(mod, inp, out):
            store[layer_idx].append(out.detach().cpu().float())
        return hook

    baseline_store = {i: [] for i in range(NUM_CONVS)}
    handles = [gnn.convs[i].register_forward_hook(make_capture(i, baseline_store)) for i in range(NUM_CONVS)]
    model.compute_representations(adata, zero_shot=True)
    for h in handles:
        h.remove()

    for i in range(SOURCE_LAYER + 1, NUM_CONVS):
        if baseline_store[i]:
            cat = torch.cat(baseline_store[i], dim=0).numpy()
            chunks = []
            with torch.no_grad():
                for s in range(0, cat.shape[0], 32768):
                    e = min(s + 32768, cat.shape[0])
                    z, _ = saes[i].encode(torch.tensor(cat[s:e]))
                    chunks.append(z.numpy())
            baseline_encoded[i] = np.vstack(chunks)
    del baseline_store
    gc.collect()
    log("baseline encoded")

    # Trace each source feature
    all_edges = []
    for fi, feat_idx in enumerate(source_features):
        ablated_store = {i: [] for i in range(NUM_CONVS)}

        def make_ablation_hook(sae_l, fid):
            def hook(mod, inp, out):
                with torch.no_grad():
                    z, _ = sae_l.encode(out.detach().cpu().float())
                    z[:, fid] = 0.0
                    mod_out = sae_l.decode(z).to(out.device).to(out.dtype)
                ablated_store[SOURCE_LAYER].append(mod_out.detach().cpu().float())
                return mod_out
            return hook

        handles = []
        for i in range(NUM_CONVS):
            if i == SOURCE_LAYER:
                handles.append(gnn.convs[i].register_forward_hook(make_ablation_hook(src_sae, feat_idx)))
            else:
                handles.append(gnn.convs[i].register_forward_hook(make_capture(i, ablated_store)))

        model.compute_representations(adata, zero_shot=True)
        for h in handles:
            h.remove()

        for tgt_layer in range(SOURCE_LAYER + 1, NUM_CONVS):
            if tgt_layer not in baseline_encoded or not ablated_store[tgt_layer]:
                continue
            abl_cat = torch.cat(ablated_store[tgt_layer], dim=0).numpy()
            sae_t = saes[tgt_layer]
            abl_chunks = []
            with torch.no_grad():
                for s in range(0, abl_cat.shape[0], 32768):
                    e = min(s + 32768, abl_cat.shape[0])
                    z, _ = sae_t.encode(torch.tensor(abl_cat[s:e]))
                    abl_chunks.append(z.numpy())
            abl_enc = np.vstack(abl_chunks)

            base = baseline_encoded[tgt_layer]
            n = min(base.shape[0], abl_enc.shape[0])
            diff = abl_enc[:n] - base[:n]
            mean_d = diff.mean(axis=0)
            std_d = diff.std(axis=0)
            std_d = np.where(std_d > 1e-9, std_d, 1.0)
            d = mean_d / std_d
            sign = np.sign(d)
            consistency = (np.sign(diff) == sign[None, :]).mean(axis=0)

            mask = (np.abs(d) > D_THRESHOLD) & (consistency > CONSISTENCY_THRESHOLD)
            for tgt_feat in np.where(mask)[0]:
                all_edges.append({
                    "source_layer": SOURCE_LAYER,
                    "source_feature": feat_idx,
                    "target_layer": tgt_layer,
                    "target_feature": int(tgt_feat),
                    "cohen_d": float(d[tgt_feat]),
                    "consistency": float(consistency[tgt_feat]),
                    "sign": int(sign[tgt_feat]),
                })

        del ablated_store
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

        if (fi + 1) % 10 == 0 or fi == len(source_features) - 1:
            log(f"  [{fi+1}/{TOP_FEATURES}] F{feat_idx}: {len(all_edges)} edges total")

    df = pd.DataFrame(all_edges)
    df.to_parquet(OUT_DIR / "exhaustive_circuit_conv5.parquet", index=False)
    log(f"\nwrote exhaustive_circuit_conv5.parquet ({len(df)} edges)")

    # Hub analysis
    if len(df) > 0:
        out_deg = df.groupby("source_feature").size().sort_values(ascending=False)
        in_deg = df.groupby(["target_layer", "target_feature"]).size().sort_values(ascending=False)

        summary = {
            "n_edges": len(df),
            "n_source_features": TOP_FEATURES,
            "source_layer": SOURCE_LAYER,
            "n_excitatory": int((df["sign"] > 0).sum()),
            "n_inhibitory": int((df["sign"] < 0).sum()),
            "median_abs_d": round(float(df["cohen_d"].abs().median()), 4),
            "max_out_degree": int(out_deg.iloc[0]) if len(out_deg) > 0 else 0,
            "out_degree_p50": round(float(out_deg.median()), 1),
            "out_degree_p90": round(float(out_deg.quantile(0.9)), 1),
            "top_source_hubs": [
                {"feature": int(f), "out_degree": int(d)}
                for f, d in out_deg.head(10).items()
            ],
        }
    else:
        summary = {"n_edges": 0}

    json.dump(summary, open(OUT_DIR / "exhaustive_circuit_conv5.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
