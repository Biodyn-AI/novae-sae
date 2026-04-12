#!/usr/bin/env python
"""Cross-checkpoint pipeline: extract activations + train SAEs + H5 CCA +
perturbation validation.

Runs end-to-end for novae-mouse-0 and novae-brain-0 checkpoints:
  1. Extract aggregator activations on selected slides
  2. Train aggregator SAE (TopK, same specs as human-0)
  3. Cross-checkpoint CCA (H5) on overlapping tissues
  4. Perturbation validation on Perturb-map (mouse-0 only)
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
from scipy import stats

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

DATA_DIR = ROOT / "datasets" / "mics-lab-novae"
LOG_PATH = ROOT / "logs" / "18_cross_checkpoint.log"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Slides per checkpoint
MOUSE_SLIDES = [
    ("brain", "mouse", "Xenium_V1_FFPE_wildtype_5_7_months_outs"),
    ("colon", "mouse", "Xenium_V1_mouse_Colon_FF_outs"),
    ("brain", "mouse", "Xenium_V1_FF_Mouse_Brain_MultiSection_1_outs"),
]

BRAIN_SLIDES = [
    ("brain", "human", "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs"),
    ("brain", "mouse", "Xenium_V1_FFPE_wildtype_5_7_months_outs"),
    ("brain", "human", "Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs"),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def extract_aggregator(ckpt_name: str, slides: list, species_map: dict | None = None) -> np.ndarray:
    """Extract aggregator activations for a set of slides using a checkpoint."""
    ckpt_dir = ROOT / "checkpoints" / ckpt_name
    act_dir = ROOT / "activations" / ckpt_name
    act_dir.mkdir(parents=True, exist_ok=True)

    model = novae.Novae.from_pretrained(str(ckpt_dir))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    all_agg = []
    slide_info = []

    for tissue, species, slide_name in slides:
        data_path = DATA_DIR / species / tissue / f"{slide_name}.h5ad"
        if not data_path.exists():
            # Try human/ dir (some mouse slides stored under human/ from earlier download)
            data_path = DATA_DIR / "human" / tissue / f"{slide_name}.h5ad"
        if not data_path.exists():
            log(f"  SKIP {slide_name} — not found at {data_path}")
            continue

        cached = act_dir / f"{slide_name}_aggregator.npy"
        if cached.exists():
            log(f"  cached: {slide_name}")
            agg = np.load(cached)
            all_agg.append(agg)
            slide_info.append({"name": slide_name, "tissue": tissue, "species": species, "n_cells": agg.shape[0]})
            continue

        log(f"  loading {slide_name}")
        a = ad.read_h5ad(data_path)
        novae.spatial_neighbors(a)
        t0 = time.time()
        model.compute_representations(a, zero_shot=True)
        agg = np.asarray(a.obsm["novae_latent"], dtype=np.float32)
        dt = time.time() - t0
        log(f"    {a.n_obs:,} cells, {dt:.0f}s")

        np.save(cached, agg)
        all_agg.append(agg)
        slide_info.append({"name": slide_name, "tissue": tissue, "species": species, "n_cells": agg.shape[0]})
        del a
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    if not all_agg:
        return np.empty((0, 64), dtype=np.float32), []
    combined = np.concatenate(all_agg, axis=0)
    np.save(act_dir / "aggregator.npy", combined)
    json.dump(slide_info, open(act_dir / "manifest.json", "w"), indent=2)
    log(f"  total: {combined.shape}")
    return combined, slide_info


def train_sae(agg: np.ndarray, ckpt_name: str, d_in: int = 64, n_features: int = 2048, k: int = 16, n_steps: int = 20000, batch_size: int = 1024, lr: float = 3e-4) -> TopKSAE:
    """Train a TopK SAE on aggregator activations."""
    sae_dir = ROOT / "saes" / ckpt_name
    sae_dir.mkdir(parents=True, exist_ok=True)
    sae_path = sae_dir / "aggregator.pt"

    if sae_path.exists():
        log(f"  SAE cached at {sae_path}")
        sae = TopKSAE(d_in=d_in, n_features=n_features, k=k)
        sae.load_state_dict(torch.load(sae_path, map_location="cpu"))
        return sae

    log(f"  training SAE: d_in={d_in}, n_features={n_features}, k={k}, steps={n_steps}")
    sae = TopKSAE(d_in=d_in, n_features=n_features, k=k)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = torch.tensor(agg, dtype=torch.float32)
    n = dataset.shape[0]
    rng = np.random.default_rng(42)

    sae.train()
    t0 = time.time()
    for step in range(n_steps):
        idx = rng.choice(n, size=batch_size, replace=False)
        x = dataset[idx]
        x_hat, z, _ = sae(x)
        loss = ((x - x_hat) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.renorm_decoder()

        if (step + 1) % 5000 == 0:
            log(f"    step {step+1}/{n_steps}: loss={loss.item():.6f}")

    sae.eval()
    torch.save(sae.state_dict(), sae_path)

    # Summary
    with torch.no_grad():
        x_all = dataset[:min(50000, n)]
        x_hat_all, _, _ = sae(x_all)
        var_exp = 1.0 - ((x_all - x_hat_all) ** 2).sum().item() / ((x_all - x_all.mean(0)) ** 2).sum().item()
    summary = {"d_in": d_in, "n_features": n_features, "k": k, "var_explained": var_exp, "n_steps": n_steps, "n_cells": n}
    json.dump(summary, open(sae_dir / "aggregator.summary.json", "w"), indent=2)
    log(f"  SAE trained: var_explained={var_exp:.4f}, {time.time()-t0:.0f}s")
    return sae


def sae_feature_profile(sae: TopKSAE, agg: np.ndarray) -> np.ndarray:
    """Compute per-feature mean |activation| profile."""
    n = agg.shape[0]
    feat_sum = np.zeros(sae.n_features, dtype=np.float64)
    chunk = 32768
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xb = torch.tensor(agg[start:end], dtype=torch.float32)
            z, _ = sae.encode(xb)
            feat_sum += np.abs(z.cpu().numpy()).sum(axis=0)
    return (feat_sum / max(n, 1)).astype(np.float32)


def cross_checkpoint_cca(profiles: dict[str, np.ndarray], top_k: int = 50) -> list[dict]:
    """CCA on top-k features between all pairs of checkpoints."""
    from sklearn.cross_decomposition import CCA

    names = list(profiles.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_prof, b_prof = profiles[a_name], profiles[b_name]

            # Select top-k features from each by mean activation
            a_top = np.argsort(-a_prof)[:top_k]
            b_top = np.argsort(-b_prof)[:top_k]

            # Profile correlation (rank correlation of the full 2048-dim profiles)
            rho, p = stats.spearmanr(a_prof, b_prof)

            # CCA on the top features' activation magnitudes
            # For this we'd need per-cell activations on shared cells, which we
            # don't have across checkpoints. Use profile-level correlation instead.

            results.append({
                "checkpoint_a": a_name,
                "checkpoint_b": b_name,
                "spearman_rho_full_profile": float(rho),
                "spearman_p": float(p),
                "pearson_r": float(np.corrcoef(a_prof, b_prof)[0, 1]),
                "top_k": top_k,
                "a_top_features": a_top.tolist(),
                "b_top_features": b_top.tolist(),
                "overlap_in_top_k": int(len(set(a_top) & set(b_top))),
            })
            log(f"  {a_name} vs {b_name}: rho={rho:.4f}, top-{top_k} overlap={results[-1]['overlap_in_top_k']}")

    return results


def perturbation_validation(ckpt_name: str, sae: TopKSAE) -> dict | None:
    """Run perturbation validation on Perturb-map Visium data."""
    perturb_dir = ROOT / "datasets" / "perturbation" / "perturb_map"
    if not perturb_dir.exists():
        log("  Perturb-map data not found")
        return None

    import scanpy as sc

    # Try to load and process each sample
    samples = ["KP_1", "KP_2", "KP_3", "KP_4"]
    model = novae.Novae.from_pretrained(str(ROOT / "checkpoints" / ckpt_name))
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()

    all_feats = []
    all_labels = []

    for s in samples:
        # Find filtered matrix
        h5_files = list(perturb_dir.glob(f"*{s}*filtered_feature_bc_matrix.h5"))
        annot_files = list(perturb_dir.glob(f"*{s}*spot_annotation.csv"))
        pos_files = list(perturb_dir.glob(f"*{s}*tissue_positions*"))

        if not h5_files:
            log(f"  SKIP {s}: no filtered matrix found")
            continue

        try:
            a = sc.read_10x_h5(str(h5_files[0]))
            a.var_names_make_unique()
            if pos_files:
                pos = pd.read_csv(pos_files[0], header=None)
                pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
                pos = pos.set_index("barcode")
                shared = a.obs_names.intersection(pos.index)
                a = a[shared].copy()
                a.obsm["spatial"] = pos.loc[shared, ["pxl_col", "pxl_row"]].values.astype(np.float32)

            if annot_files:
                annot = pd.read_csv(annot_files[0])
                if "barcode" in annot.columns:
                    annot = annot.set_index("barcode")
                    shared = a.obs_names.intersection(annot.index)
                    a = a[shared].copy()
                    if "phenotypes" in annot.columns:
                        a.obs["phenotype"] = annot.loc[shared, "phenotypes"].values

            if a.n_obs < 10:
                log(f"  SKIP {s}: too few cells ({a.n_obs})")
                continue

            # Preprocess
            sc.pp.normalize_total(a)
            sc.pp.log1p(a)

            novae.spatial_neighbors(a)
            model.compute_representations(a, zero_shot=True)
            agg = np.asarray(a.obsm["novae_latent"], dtype=np.float32)

            # SAE encode
            with torch.no_grad():
                z_list = []
                for start in range(0, agg.shape[0], 4096):
                    end = min(start + 4096, agg.shape[0])
                    xb = torch.tensor(agg[start:end])
                    z, _ = sae.encode(xb)
                    z_list.append(z.numpy())
                feats = np.vstack(z_list)

            labels = a.obs.get("phenotype", pd.Series(["NA"] * a.n_obs))
            all_feats.append(feats)
            all_labels.extend(labels.tolist())
            log(f"  {s}: {a.n_obs} spots, {len(labels.unique()) if hasattr(labels, 'unique') else '?'} phenotypes")

            del a
            gc.collect()

        except Exception as e:
            log(f"  ERROR {s}: {type(e).__name__}: {e}")
            continue

    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    if not all_feats:
        return None

    feats = np.vstack(all_feats)
    labels = np.array([str(l) if l is not None and str(l) != "nan" else "NA" for l in all_labels], dtype=object)
    log(f"  total: {feats.shape[0]} spots, {len(np.unique(labels))} unique labels")

    # Wilcoxon test: for each unique perturbation label, compare feature
    # activations to all other spots
    unique_labels = [l for l in np.unique(labels) if l not in ("NA", "periphery", None, "nan")]
    results = []
    for label in unique_labels:
        mask = labels == label
        n_label = int(mask.sum())
        if n_label < 5:
            continue
        for fid in range(feats.shape[1]):
            col = np.abs(feats[:, fid])
            a_vals = col[mask]
            b_vals = col[~mask]
            if a_vals.std() == 0 and b_vals.std() == 0:
                continue
            try:
                stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                results.append({"label": label, "feature_idx": fid, "u_stat": float(stat), "p_value": float(p), "n_label": n_label, "mean_label": float(a_vals.mean()), "mean_other": float(b_vals.mean())})
            except Exception:
                pass

    if not results:
        return {"n_labels_tested": 0, "n_significant": 0}

    df = pd.DataFrame(results)
    # BH correction
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["fdr"] = fdr
    df["significant"] = df["fdr"] < 0.05

    out_dir = ROOT / "atlas" / ckpt_name / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "perturbation_validation.parquet", index=False)

    n_sig = int(df["significant"].sum())
    n_labels = len(df["label"].unique())
    labels_with_sig = df[df["significant"]].groupby("label").size()
    summary = {
        "n_spots": int(feats.shape[0]),
        "n_labels_tested": n_labels,
        "n_significant_tests": n_sig,
        "n_labels_with_significant_features": int(len(labels_with_sig)),
        "fraction_labels_responding": float(len(labels_with_sig) / max(n_labels, 1)),
        "top_responding_labels": labels_with_sig.sort_values(ascending=False).head(10).to_dict(),
    }
    json.dump(summary, open(out_dir / "perturbation_validation.summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    log("=" * 72)
    log("Cross-checkpoint pipeline")

    # === Phase 1: Extract + Train for mouse-0 ===
    log("\n=== novae-mouse-0 ===")
    log("extracting aggregator activations")
    mouse_agg, mouse_info = extract_aggregator("novae-mouse-0", MOUSE_SLIDES)
    if mouse_agg.shape[0] > 0:
        log("training aggregator SAE")
        mouse_sae = train_sae(mouse_agg, "novae-mouse-0")
        mouse_profile = sae_feature_profile(mouse_sae, mouse_agg)
    else:
        log("ERROR: no mouse activations extracted")
        mouse_sae = None
        mouse_profile = None

    # === Phase 2: Extract + Train for brain-0 ===
    log("\n=== novae-brain-0 ===")
    log("extracting aggregator activations")
    brain_agg, brain_info = extract_aggregator("novae-brain-0", BRAIN_SLIDES)
    if brain_agg.shape[0] > 0:
        log("training aggregator SAE")
        brain_sae = train_sae(brain_agg, "novae-brain-0")
        brain_profile = sae_feature_profile(brain_sae, brain_agg)
    else:
        log("ERROR: no brain activations extracted")
        brain_sae = None
        brain_profile = None

    # === Phase 3: Cross-checkpoint CCA (H5) ===
    log("\n=== H5: cross-checkpoint convergence ===")
    # Load human profile
    human_sae = TopKSAE(d_in=64, n_features=2048, k=16)
    human_sae.load_state_dict(torch.load(ROOT / "saes" / "novae-human-0" / "aggregator.pt", map_location="cpu"))
    human_sae.eval()
    human_agg = np.load(ROOT / "activations" / "novae-human-0" / "aggregator.npy")
    human_profile = sae_feature_profile(human_sae, human_agg)

    profiles = {"human-0": human_profile}
    if mouse_profile is not None:
        profiles["mouse-0"] = mouse_profile
    if brain_profile is not None:
        profiles["brain-0"] = brain_profile

    if len(profiles) >= 2:
        cca_results = cross_checkpoint_cca(profiles, top_k=50)
        out_dir = ROOT / "atlas" / "novae-human-0" / "causal"
        json.dump(cca_results, open(out_dir / "cross_checkpoint_cca.json", "w"), indent=2)
        log(f"wrote cross_checkpoint_cca.json ({len(cca_results)} pairs)")

        # H5 verdict
        for r in cca_results:
            rho = r["spearman_rho_full_profile"]
            log(f"  {r['checkpoint_a']} vs {r['checkpoint_b']}: rho={rho:.4f}")
        log("H5: cross-checkpoint profile correlation (not full CCA on shared cells)")
    else:
        log("SKIP: fewer than 2 checkpoint profiles available")

    # === Phase 4: Perturbation validation (mouse-0 only) ===
    log("\n=== §4.5/5: perturbation validation ===")
    if mouse_sae is not None:
        perturb_result = perturbation_validation("novae-mouse-0", mouse_sae)
        if perturb_result:
            log(f"perturbation result: {perturb_result.get('n_labels_with_significant_features', 0)} labels with sig features")
        else:
            log("perturbation validation failed or no data")
    else:
        log("SKIP: no mouse SAE available")

    log("\nALL DONE")


if __name__ == "__main__":
    main()
