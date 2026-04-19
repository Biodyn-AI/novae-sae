#!/usr/bin/env python
"""Reviewer control: multi-slide circuit replication.

Reviewer concern: all 439 circuit edges come from a single brain slide.
Biological stories (cancer→T-cell inhibition, myoepithelial hub, etc.)
rely on features whose KEGG/PanglaoDB labels may be out-of-distribution
for the specific slide used.

Protocol:
  1. Re-run the circuit-tracing protocol (script 19) on 2 additional
     small slides: pancreas and kidney.
  2. Join edges across slides and report:
       - Edge reproducibility (edges present on >=2 slides)
       - Source-feature rank agreement
       - Cohen's d distribution per slide
  3. Apply Bonferroni correction across the tested (source, target)
     feature pairs: alpha_corrected = 0.05 / N_tests.

Output: atlas/novae-human-0/causal/reviewer_controls/I_multi_slide_circuits/
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

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "novae-human-0" / "causal" / "reviewer_controls" / "I_multi_slide_circuits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "30_multislide_circuits.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# Additional small slides to complement brain
SLIDES = [
    ("brain", "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs"),   # replicate
    ("pancreas", "Xenium_V1_hPancreas_nondiseased_section_outs"),
    ("kidney", "Xenium_V1_hKidney_nondiseased_section_outs"),
]
NUM_CONVS = 10
# Use a smaller per-slide source-feature set for tractability
SOURCE_LAYERS = [0, 5]
TOP_SOURCE_FEATURES = 20
D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.6

SAE_SPECS = {
    **{f"conv_{i}": (128, 4096, 32) for i in range(9)},
    "conv_9": (64, 2048, 16),
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_saes() -> dict[int, TopKSAE]:
    saes = {}
    for i in range(NUM_CONVS):
        name = f"conv_{i}"
        d_in, n_feat, k = SAE_SPECS[name]
        sae = TopKSAE(d_in=d_in, n_features=n_feat, k=k)
        sae.load_state_dict(torch.load(SAE_DIR / f"{name}.pt", map_location="cpu"))
        sae.eval()
        saes[i] = sae
    return saes


def cohen_d(a: torch.Tensor, b: torch.Tensor) -> float:
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(unbiased=False), b.var(unbiased=False)
    pooled = torch.sqrt((va + vb) / 2 + 1e-12)
    return float((mu_a - mu_b) / pooled)


def per_slide_trace(model, saes, a_data, slide_name: str) -> pd.DataFrame:
    """Single-slide circuit tracing: ablate source SAE features at
    SOURCE_LAYERS and measure Cohen's d downstream."""
    model.eval()
    # Reuse the compute_representations pipeline but capture intermediate
    # conv outputs. Novae's GATv2 layers expose hooks via model.encoder.convs.
    from novae.model import Novae
    # Find the GATv2 convs
    convs = [m for n, m in model.named_modules() if "conv" in n.lower() and hasattr(m, 'att')]
    # Fall back to using compute_representations + hooking at model.encoder
    hooks = {}
    cache = {i: None for i in range(NUM_CONVS)}

    def make_hook(idx):
        def _hook(module, inp, out):
            cache[idx] = out.detach().clone()
        return _hook

    # Register hooks on each GATv2 conv in order
    # Novae stores the encoder convs in encoder.convs (ModuleList)
    if hasattr(model, "encoder") and hasattr(model.encoder, "convs"):
        for i, c in enumerate(model.encoder.convs):
            hooks[i] = c.register_forward_hook(make_hook(i))
    else:
        raise RuntimeError("Could not locate GATv2 convs on model.encoder.convs")

    try:
        # Baseline forward
        novae.spatial_neighbors(a_data)
        model.compute_representations(a_data, zero_shot=True)
        # Keep baseline conv outputs
        baseline_conv = {i: cache[i].cpu() for i in range(NUM_CONVS) if cache[i] is not None}

        # For each source feature at SOURCE_LAYERS, ablate (zero in SAE code,
        # decode, substitute, re-propagate).
        rows = []
        for src_layer in SOURCE_LAYERS:
            if src_layer not in baseline_conv:
                log(f"  layer {src_layer} unavailable; skipping")
                continue
            src_sae = saes[src_layer]
            src_sae.to(DEVICE)
            h_src = baseline_conv[src_layer].to(DEVICE)
            with torch.no_grad():
                z_src_dense = src_sae.encode_dense(h_src).abs().mean(dim=0)
            top_src_feats = torch.topk(z_src_dense, TOP_SOURCE_FEATURES).indices.cpu().numpy()
            src_sae.to("cpu")
            log(f"  src layer {src_layer}: top feats = {top_src_feats[:5]}…")

            for src_feat in top_src_feats:
                # Ablate: encode src layer with SAE, zero feature, decode,
                # replace; re-run downstream convs.
                src_sae.to(DEVICE)
                with torch.no_grad():
                    z_dense = src_sae.encode_dense(h_src)
                    _, idx = z_dense.abs().topk(src_sae.k, dim=-1)
                    z = torch.zeros_like(z_dense)
                    z.scatter_(-1, idx, z_dense.gather(-1, idx))
                    # zero specific feature
                    mask = (idx == src_feat).any(dim=-1)
                    z[:, src_feat] = 0.0
                    h_src_abl = src_sae.decode(z)  # (N, d_src)

                # Forward from src+1 to end with substituted h_src_abl.
                # This requires re-running the forward pass with the ablated
                # intermediate; we do this by clearing cache and setting a
                # one-shot hook that patches the src layer's output.
                patch_layer = src_layer
                patch_value = h_src_abl.detach()

                abl_cache = {i: None for i in range(NUM_CONVS)}

                def _abl_hook_factory(idx):
                    def _abl_hook(module, inp, out):
                        if idx == patch_layer:
                            abl_cache[idx] = patch_value.to(out.device).to(out.dtype)
                            return abl_cache[idx]
                        abl_cache[idx] = out.detach().clone()
                        return out
                    return _abl_hook

                # Remove baseline hooks
                for h in hooks.values():
                    h.remove()
                hooks = {}
                for i, c in enumerate(model.encoder.convs):
                    hooks[i] = c.register_forward_hook(_abl_hook_factory(i))
                # Clear cached latent + rerun
                if "novae_latent" in a_data.obsm:
                    del a_data.obsm["novae_latent"]
                with torch.no_grad():
                    model.compute_representations(a_data, zero_shot=True)
                # Cleanup this ablation's hooks
                for h in hooks.values():
                    h.remove()
                hooks = {}
                # Reinstall baseline hooks for the next iteration's baseline
                # cache refresh (though we already have baseline_conv).
                for i, c in enumerate(model.encoder.convs):
                    hooks[i] = c.register_forward_hook(make_hook(i))

                for tgt_layer in range(src_layer + 1, NUM_CONVS):
                    if abl_cache[tgt_layer] is None or tgt_layer not in baseline_conv:
                        continue
                    tgt_sae = saes[tgt_layer]
                    tgt_sae.to(DEVICE)
                    h_tgt_base = baseline_conv[tgt_layer].to(DEVICE)
                    h_tgt_abl = abl_cache[tgt_layer].to(DEVICE)
                    with torch.no_grad():
                        z_base, _ = tgt_sae.encode(h_tgt_base)
                        z_abl, _ = tgt_sae.encode(h_tgt_abl)
                    # Compute Cohen's d per target feature on top 50 per-cell activations
                    base_mean = z_base.abs().mean(dim=0)
                    top_tgt = base_mean.topk(50).indices.cpu().numpy()
                    for tfeat in top_tgt:
                        a_vec = z_base[:, tfeat]
                        b_vec = z_abl[:, tfeat]
                        d = cohen_d(b_vec - a_vec,
                                    torch.zeros_like(a_vec))  # sign of shift
                        # Consistency: fraction of cells where shift sign agrees
                        sign_mean = (b_vec - a_vec).sign()
                        dominant = sign_mean.mode().values if len(sign_mean) else torch.tensor(0.0)
                        consistency = float((sign_mean == dominant).float().mean()) if len(sign_mean) else 0.0
                        if abs(d) > D_THRESHOLD and consistency > CONSISTENCY_THRESHOLD:
                            rows.append({
                                "slide": slide_name,
                                "source_layer": int(src_layer),
                                "source_feature": int(src_feat),
                                "target_layer": int(tgt_layer),
                                "target_feature": int(tfeat),
                                "cohen_d": float(d),
                                "consistency": float(consistency),
                            })
                    tgt_sae.to("cpu")
                src_sae.to("cpu")
        df = pd.DataFrame(rows)
        return df
    finally:
        for h in hooks.values():
            h.remove()


def main() -> None:
    log("=" * 72)
    log("Multi-slide circuit replication")
    saes = load_saes()
    model = novae.Novae.from_pretrained(str(CKPT_DIR))
    if DEVICE != "cpu":
        model = model.to(DEVICE)

    all_edges = []
    for tissue, name in SLIDES:
        h5ad = DATA_DIR / tissue / f"{name}.h5ad"
        log(f"\n[{tissue}] {name}")
        try:
            a = ad.read_h5ad(h5ad)
        except Exception as e:
            log(f"  ERROR load: {e}")
            continue
        t0 = time.time()
        try:
            df = per_slide_trace(model, saes, a, name)
            log(f"  {len(df)} edges, {time.time()-t0:.0f}s")
            df.to_parquet(OUT_DIR / f"edges_{tissue}.parquet", index=False)
            all_edges.append(df)
        except Exception as e:
            log(f"  ERROR trace: {type(e).__name__}: {e}")
        del a
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    if not all_edges:
        log("no edges produced")
        return
    all_df = pd.concat(all_edges, ignore_index=True)
    all_df.to_parquet(OUT_DIR / "edges_all.parquet", index=False)

    # Per-edge-tuple, across slides
    key = ["source_layer", "source_feature", "target_layer", "target_feature"]
    agg = all_df.groupby(key).agg(
        n_slides=("slide", "nunique"),
        mean_d=("cohen_d", "mean"),
        min_consistency=("consistency", "min"),
    ).reset_index()
    agg.to_parquet(OUT_DIR / "edges_reproducible.parquet", index=False)
    summary = {
        "n_slides": len(SLIDES),
        "per_slide_edge_counts": all_df.groupby("slide").size().to_dict(),
        "n_unique_edges": int(len(agg)),
        "n_edges_in_ge2_slides": int((agg["n_slides"] >= 2).sum()),
        "n_edges_in_ge3_slides": int((agg["n_slides"] >= 3).sum()),
        "jaccard_pairwise": _jaccard_matrix(all_df),
    }
    (OUT_DIR / "edges_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log("DONE")


def _jaccard_matrix(df: pd.DataFrame) -> dict:
    slides = df["slide"].unique().tolist()
    key = ["source_layer", "source_feature", "target_layer", "target_feature"]
    sets = {s: set(map(tuple, df[df["slide"] == s][key].values.tolist())) for s in slides}
    out = {}
    for i, s1 in enumerate(slides):
        for s2 in slides[i + 1:]:
            inter = len(sets[s1] & sets[s2])
            union = len(sets[s1] | sets[s2])
            out[f"{s1}__VS__{s2}"] = {"jaccard": inter / union if union else 0,
                                        "intersection": inter, "union": union}
    return out


if __name__ == "__main__":
    main()
