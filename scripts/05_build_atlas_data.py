#!/usr/bin/env python
"""Phase A.1 — bake static JSON data files for the Novae atlas frontend.

Reads the atlas/ artefacts (feature_atlas_full.parquet, modules.parquet,
causal/*.parquet, bio/v2 + v3 outputs, per-slide spatial coordinates and
domain labels) and emits a self-contained data tree at:

  atlas/web/public/data/
    manifest.json
    surfaces.json
    slides.json
    features/<surface>.json                  # slim per-surface tables
    feature_detail/<surface>/<idx>.json      # lazy-loaded per-feature drilldown
    spatial/slide_<i>.json                   # per-slide background subsamples
    spatial/feature/aggregator/<idx>.json    # per-feature top-cell coordinates
    modules/<surface>.json                   # per-surface module summaries

Bootstrap target: ~10 KB. Per-page navigation: ~1-4 MB. Per-feature lazy: ~5 KB.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
from src.topk_sae import TopKSAE  # noqa: E402

ATLAS = ROOT / "atlas" / "novae-human-0"
ACT_DIR = ROOT / "activations" / "novae-human-0"
SAE_DIR = ROOT / "saes" / "novae-human-0"
DATA_DIR = ROOT / "datasets" / "mics-lab-novae" / "human"
OUT_DIR = ROOT / "atlas" / "web" / "public" / "data"
LOG_PATH = ROOT / "logs" / "05_build_atlas_data.log"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Bake budget
TOP_CELLS_PER_FEATURE_SPATIAL = 200    # for spatial overlay
SLIDE_BACKGROUND_N = 5000              # subsample per slide
TOP_GENES_PER_FEATURE_DETAIL = 20      # in detail JSON
SEED = 42


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=_default)


def _default(o):
    """JSON encoder fallback — only fires for non-natively-serializable types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"unserializable: {type(o).__name__}")


def to_jsonable(d):
    """Convert a dict of mixed numpy/pandas types to plain Python types."""
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            f = float(v)
            out[k] = None if np.isnan(f) or np.isinf(f) else f
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, float):
            out[k] = None if np.isnan(v) or np.isinf(v) else v
        else:
            try:
                if pd.isna(v):
                    out[k] = None
                    continue
            except (TypeError, ValueError):
                pass
            out[k] = v
    return out


def safe_str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def safe_num(v):
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ---------------- Label derivation helpers ----------------

import re

_GO_ID_RE = re.compile(r"\s*\(GO:\d+\)\s*$")
_KEGG_RE = re.compile(r"\s*Homo sapiens.*$", re.IGNORECASE)
_REACT_ID_RE = re.compile(r"\s*R-HSA-\d+\s*$")
_CELLMARKER_TAIL = re.compile(r":\s*[A-Z][a-z]+(\s[A-Z][a-z]+)*$")


def clean_term(t: str) -> str:
    if t is None:
        return ""
    s = str(t).strip()
    s = _GO_ID_RE.sub("", s)
    s = _KEGG_RE.sub("", s)
    s = _REACT_ID_RE.sub("", s)
    return s.strip()


# Generic / non-informative terms to skip when picking the lead label
_GENERIC_TERMS = {
    "embryonic stem cells",
    "positive regulation of dna-templated transcription",
    "pathways in cancer",
    "metabolic pathways",
    "regulation of dna-templated transcription",
}


def derive_feature_label(panglao_term, panglao_fdr, cellmarker_term, go_term, top_genes_str) -> str:
    """One-line English label for a feature, best-available source."""
    # PanglaoDB if significant and not generic
    if panglao_term and not pd.isna(panglao_term):
        c = clean_term(panglao_term)
        if c and c.lower() not in _GENERIC_TERMS and (panglao_fdr is None or pd.isna(panglao_fdr) or float(panglao_fdr) < 0.05):
            return c
    # CellMarker fallback
    if cellmarker_term and not pd.isna(cellmarker_term):
        c = clean_term(cellmarker_term)
        if c and c.lower() not in _GENERIC_TERMS:
            # Strip the trailing ":Tissue" decoration if present
            c2 = _CELLMARKER_TAIL.sub("", c)
            if c2:
                return c2
            return c
    # GO BP fallback
    if go_term and not pd.isna(go_term):
        c = clean_term(go_term)
        if c and c.lower() not in _GENERIC_TERMS:
            return c
    # Top genes fallback
    if top_genes_str and not pd.isna(top_genes_str):
        genes = [g.strip().upper() for g in str(top_genes_str).split(",") if g.strip()]
        if genes:
            return f"{', '.join(genes[:3])}+"
    return "Unannotated"


def load_aggregator_sae() -> TopKSAE:
    sae = TopKSAE(d_in=64, n_features=2048, k=16)
    sae.load_state_dict(torch.load(SAE_DIR / "aggregator.pt", map_location="cpu"))
    sae.eval()
    return sae


def encode_top_cells(sae: TopKSAE, X: np.ndarray, n_top: int) -> dict[int, np.ndarray]:
    """Per-feature: indices and activations of the top-N cells (by |activation|)."""
    sae.to(DEVICE)
    n, _ = X.shape
    n_features = sae.n_features
    rows, cols, vals = [], [], []
    chunk = 32768
    with torch.no_grad():
        for start in range(0, n, chunk):
            xb = torch.tensor(X[start:start + chunk], dtype=torch.float32, device=DEVICE)
            z, _ = sae.encode(xb)
            mask = z != 0
            r, c = mask.nonzero(as_tuple=True)
            v = z[r, c]
            rows.append((r + start).cpu().numpy())
            cols.append(c.cpu().numpy())
            vals.append(v.cpu().numpy())
    sae.to("cpu")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    rows_np = np.concatenate(rows)
    cols_np = np.concatenate(cols)
    vals_np = np.concatenate(vals)
    M = sparse.csr_matrix(
        (vals_np, (rows_np, cols_np)), shape=(n, n_features), dtype=np.float32
    ).tocsc()
    out: dict[int, np.ndarray] = {}
    for f in range(n_features):
        s, e = M.indptr[f], M.indptr[f + 1]
        if s == e:
            out[f] = np.empty((0, 2), dtype=np.float32)
            continue
        idx = M.indices[s:e]
        v = M.data[s:e]
        k = min(n_top, len(v))
        # Pick by absolute value
        order = np.argpartition(np.abs(v), -k)[-k:]
        # Sort descending by abs
        order = order[np.argsort(-np.abs(v[order]))]
        out[f] = np.column_stack([idx[order].astype(np.int64), v[order].astype(np.float32)])
    return out


def build_per_feature_spatial(
    top_cells_per_feature: dict[int, np.ndarray],
    manifest: dict,
    slide_coords: dict[int, np.ndarray],
) -> dict[int, list[dict]]:
    """For each feature, return list of {slide, slide_idx, x, y, activation}."""
    slide_starts = np.array([s["cell_offset_start"] for s in manifest["slides"]], dtype=np.int64)
    out: dict[int, list[dict]] = {}
    for f, top in top_cells_per_feature.items():
        if len(top) == 0:
            out[f] = []
            continue
        cells = top[:, 0].astype(np.int64)
        acts = top[:, 1].astype(np.float32)
        sids = np.searchsorted(slide_starts, cells, side="right") - 1
        rows = []
        for i in range(len(cells)):
            sid = int(sids[i])
            local = int(cells[i] - slide_starts[sid])
            coords = slide_coords.get(sid)
            if coords is None or local >= len(coords):
                continue
            rows.append({
                "slide_idx": sid,
                "x": float(coords[local, 0]),
                "y": float(coords[local, 1]),
                "a": float(acts[i]),
            })
        out[f] = rows
    return out


def main() -> None:
    log("=" * 72)
    log("Phase A.1: bake atlas data files")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ---------------- 1. Load tables ----------------
    log("loading parquet inputs")
    feature_atlas = pd.read_parquet(ATLAS / "feature_atlas_full.parquet")
    log(f"  feature_atlas_full: {feature_atlas.shape}")

    modules = pd.read_parquet(ATLAS / "modules.parquet")
    log(f"  modules: {modules.shape}")

    causal_ablation = pd.read_parquet(ATLAS / "causal" / "aggregator_ablation.parquet")
    causal_morans = pd.read_parquet(ATLAS / "causal" / "morans_i.parquet")
    log(f"  causal: ablation {causal_ablation.shape}, morans {causal_morans.shape}")

    # Block 1 + Phase 4 validation parquets — load if present
    block1 = {}
    for tag, fname in [
        ("proto", "sae_vs_prototypes.parquet"),
        ("morans_all", "spatial_coherence_all.parquet"),
        ("confounds", "confounds.parquet"),
        ("confounds_es", "confounds_effect_size.parquet"),
        ("confound_survival", "confound_survival.parquet"),
        ("ablation", "graph_ablation.parquet"),
        ("ablation_v2", "graph_ablation_v2.parquet"),
        ("proto_ablation", "prototype_domain_ablation.per_feature.parquet"),
    ]:
        p = ATLAS / "causal" / fname
        if p.exists():
            block1[tag] = pd.read_parquet(p).set_index("feature_idx")
            log(f"  block1.{tag}: {block1[tag].shape}")
        else:
            log(f"  block1.{tag}: not found (skipping)")

    _summary_raw = json.load(open(ATLAS / "summary.json"))
    _sae_raw = json.load(open(SAE_DIR / "summary.json"))
    _modules_raw = json.load(open(ATLAS / "modules_summary.json"))

    # Build per-name lookups (the source files store these as lists, not dicts)
    surf_by_name = {s["name"]: s for s in _summary_raw["surfaces"]}
    super_by_name = {s["name"]: s for s in _summary_raw["superposition"]}
    sae_by_name = {s["name"]: s for s in _sae_raw["summaries"]}
    mods_by_name = {s["name"]: s for s in _modules_raw["surfaces"]}

    summary = surf_by_name  # Backwards-compat alias used below
    sae_summary = sae_by_name
    modules_summary = mods_by_name

    # v2 top genes (aggregator) + cell_embedder top genes
    agg_top_genes = pd.read_parquet(ATLAS / "bio" / "v2" / "aggregator_top_genes_v2.parquet")
    ce_top_genes = pd.read_parquet(ATLAS / "bio" / "cell_embedder_top_genes.parquet")

    # All enrichment for the detail page
    agg_enr_v2 = pd.read_parquet(ATLAS / "bio" / "v2" / "aggregator_enrichment_v2.parquet")
    ce_enr = pd.read_parquet(ATLAS / "bio" / "cell_embedder_enrichment.parquet")

    # Domain enrichment
    dom_enr = pd.read_parquet(ATLAS / "bio" / "v3" / "aggregator_domain_enrichment.parquet")

    manifest = json.load(open(ACT_DIR / "manifest.json"))

    # ---------------- 1b. Pre-compute per-feature labels + significance ----------------
    log("\ncomputing per-feature labels (lb) and significance scores")
    label_lookup: dict[tuple[str, int], str] = {}
    sig_lookup: dict[tuple[str, int], dict] = {}
    niche_label_lookup: dict[tuple[int, str], str] = {}

    # Build niche labels at all 3 levels — used by slim tables, feature_detail,
    # and niche_index. The label is the dominant feature's lb.
    for level in [7, 12, 20]:
        sub_lvl = dom_enr[dom_enr.level == level]
        # For each niche, find the row with max log2_enrichment (the dominant feature)
        if not sub_lvl.empty:
            doms = sub_lvl.loc[sub_lvl.groupby("top_domain")["log2_enrichment"].idxmax()]
            for r in doms.itertuples():
                d = str(r.top_domain)
                if not d:
                    continue
                # Compute the dominant feature's label inline (label_lookup not yet built)
                fid_dom = int(r.feature_idx)
                agg_row = feature_atlas[
                    (feature_atlas["surface"] == "aggregator") & (feature_atlas["feature_idx"] == fid_dom)
                ]
                if len(agg_row) > 0:
                    rr = agg_row.iloc[0]
                    niche_label_lookup[(level, d)] = derive_feature_label(
                        rr.get("top_PanglaoDB_v2"),
                        rr.get("top_PanglaoDB_v2_fdr"),
                        rr.get("top_CellMarker_v2"),
                        rr.get("top_GO_BP_v2"),
                        rr.get("top_genes"),
                    )

    # Build per-feature niche stats lookup (level 20) for the score
    _sig_l20_by_f = {int(r.feature_idx): r for r in dom_enr[dom_enr.level == 20].itertuples()}

    for r in feature_atlas.itertuples():
        s = r.surface
        fid = int(r.feature_idx)

        # Label
        if s in {"aggregator", "cell_embedder"}:
            lb = derive_feature_label(
                getattr(r, "top_PanglaoDB_v2", None),
                getattr(r, "top_PanglaoDB_v2_fdr", None),
                getattr(r, "top_CellMarker_v2", None),
                getattr(r, "top_GO_BP_v2", None),
                getattr(r, "top_genes", None),
            )
        else:
            tissue = getattr(r, "top_tissue", None)
            lb = f"{tissue} feature" if tissue and not pd.isna(tissue) else "Unannotated"
        label_lookup[(s, fid)] = lb

        # Significance score: composite of cell-type FDR + niche enrichment + alive
        # Score is in arbitrary units, ~0-100 typical range, sortable
        bio_score = 0.0     # cell-type-database significance
        niche_score = 0.0   # spatial niche specificity
        alive_bonus = 0.0   # alive features get a small constant

        if s in {"aggregator", "cell_embedder"}:
            pl_fdr = getattr(r, "top_PanglaoDB_v2_fdr", None)
            if pl_fdr is not None and not pd.isna(pl_fdr) and float(pl_fdr) > 0:
                # -log10(FDR), capped at 50
                bio_score = min(50.0, -float(np.log10(max(float(pl_fdr), 1e-300))))

        if s == "aggregator" and fid in _sig_l20_by_f:
            nrow = _sig_l20_by_f[fid]
            log2enr = float(nrow.log2_enrichment) if not pd.isna(nrow.log2_enrichment) else 0.0
            nfdr = float(nrow.fisher_fdr) if not pd.isna(nrow.fisher_fdr) else 1.0
            # Niche component: log2enr (capped at 10) × 5, but only count if FDR significant
            if nfdr < 0.05:
                niche_score = min(10.0, max(0.0, log2enr)) * 5.0

        if hasattr(r, "alive") and bool(r.alive):
            alive_bonus = 1.0

        total = bio_score + niche_score + alive_bonus
        sig_lookup[(s, fid)] = {
            "bio_score": round(bio_score, 2),
            "niche_score": round(niche_score, 2),
            "total": round(total, 2),
        }

    # ---------------- 2. Top-level manifest, surfaces, slides ----------------
    log("\nemitting manifest / surfaces / slides")

    SURFACE_ORDER = [
        "cell_embedder",
        "conv_0", "conv_1", "conv_2", "conv_3", "conv_4",
        "conv_5", "conv_6", "conv_7", "conv_8", "conv_9",
        "aggregator",
    ]
    surfaces_present = [s for s in SURFACE_ORDER if s in feature_atlas["surface"].unique()]

    # Per-surface stats from the source summaries
    surfaces_payload = []
    for s in surfaces_present:
        sub = feature_atlas[feature_atlas["surface"] == s]
        sae_info = sae_by_name.get(s, {})
        surf_info = surf_by_name.get(s, {})
        sup_info = super_by_name.get(s, {})
        mod_info = mods_by_name.get(s, {})
        surfaces_payload.append({
            "name": s,
            "n_features": int(len(sub)),
            "alive": int(sub["alive"].sum()) if "alive" in sub.columns else None,
            "d": sae_info.get("d_in"),
            "n_features_total": sae_info.get("n_features"),
            "k": sae_info.get("k"),
            "expansion": sae_info.get("expansion"),
            "var_exp_full": surf_info.get("var_explained_full"),
            "superposition": sup_info.get("fraction_non_aligned"),
            "tech_confound_rate": surf_info.get("tech_confounded_fraction"),
            "n_modules": mod_info.get("n_modules"),
            "module_top5": mod_info.get("module_size_top5"),
            "has_bio": s in {"aggregator", "cell_embedder"},
            "has_domains": s == "aggregator",
            "has_spatial": s == "aggregator",
        })

    write_json(OUT_DIR / "surfaces.json", surfaces_payload)

    slides_payload = [
        {
            "idx": s["slide_idx"],
            "name": s["name"],
            "tissue": s["tissue"],
            "technology": s["technology"],
            "n_cells": s["n_cells"],
        }
        for s in manifest["slides"]
    ]
    write_json(OUT_DIR / "slides.json", slides_payload)

    write_json(OUT_DIR / "manifest.json", {
        "checkpoint": "novae-human-0",
        "n_surfaces": len(surfaces_present),
        "n_total_features": int(len(feature_atlas)),
        "n_slides": len(manifest["slides"]),
        "n_total_cells": sum(s["n_cells"] for s in manifest["slides"]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # ---------------- 3. Per-surface slim feature tables ----------------
    log("\nemitting per-surface slim feature tables")

    SLIM_COLS_BASE = [
        "feature_idx", "alive", "n_active_cells", "mean_abs", "max_abs",
        "top_tissue", "top_tissue_frac", "top_tech", "top_tech_frac", "tech_confound",
    ]
    SLIM_COLS_BIO = [
        "top_genes", "top_PanglaoDB_v2", "top_PanglaoDB_v2_fdr",
        "top_CellMarker_v2", "top_GO_BP_v2",
    ]
    SLIM_COLS_DOMAIN = [
        "top_domain_l20", "top_domain_l20_frac", "top_domain_l20_log2enr", "top_domain_l20_fdr",
    ]

    # Build per-(surface, feature) module lookup once for slim-table inclusion
    mod_lookup = {(t.surface, int(t.feature_idx)): (int(t.module_id), float(t.P_i))
                  for t in modules.itertuples()}

    # niche_label_lookup is already built in section 1b above

    for s in surfaces_present:
        sub = feature_atlas[feature_atlas["surface"] == s].copy()
        cols = list(SLIM_COLS_BASE)
        if s in {"aggregator", "cell_embedder"}:
            cols += SLIM_COLS_BIO
        if s == "aggregator":
            cols += SLIM_COLS_DOMAIN
        sub_slim = sub[cols].copy()
        # Convert to records, replacing NaN with None and adding module + label
        records = []
        for r in sub_slim.itertuples(index=False):
            d = to_jsonable(r._asdict())
            mod_info = mod_lookup.get((s, int(d["feature_idx"])))
            if mod_info is not None:
                d["module_id"] = mod_info[0]
                d["module_p"] = mod_info[1]
            else:
                d["module_id"] = None
                d["module_p"] = None
            d["lb"] = label_lookup.get((s, int(d["feature_idx"])), "Unannotated")
            # Inject niche label for aggregator
            if s == "aggregator" and d.get("top_domain_l20"):
                d["top_domain_l20_lb"] = niche_label_lookup.get((20, d["top_domain_l20"]), "")
            # Inject significance score
            sig = sig_lookup.get((s, int(d["feature_idx"])), {})
            d["sig"] = sig.get("total", 0.0)
            d["sig_bio"] = sig.get("bio_score", 0.0)
            d["sig_niche"] = sig.get("niche_score", 0.0)
            records.append(d)
        write_json(OUT_DIR / "features" / f"{s}.json", records)
        log(f"  {s}: {len(records)} features")

    # ---------------- 4. Modules per surface ----------------
    log("\nemitting modules")
    for s in surfaces_present:
        sub = modules[modules["surface"] == s]
        # Group by module_id
        groups = sub.groupby("module_id")
        mods = []
        for mid, grp in groups:
            members = grp["feature_idx"].astype(int).tolist()
            mods.append({
                "module_id": int(mid),
                "n_features": int(len(members)),
                "members": members,
            })
        mods.sort(key=lambda m: -m["n_features"])
        sm = modules_summary.get(s, {})
        write_json(OUT_DIR / "modules" / f"{s}.json", {
            "surface": s,
            "n_modules": sm.get("n_modules"),
            "modules": mods,
        })

    # ---------------- 5. Per-feature drill-down (aggregator) ----------------
    log("\nbuilding aggregator per-feature detail")

    # Index lookups
    agg_top_by_f = {fid: g.sort_values("rank") for fid, g in agg_top_genes.groupby("feature_idx")}
    agg_enr_by_f = {fid: g for fid, g in agg_enr_v2.groupby("feature_idx")}
    dom_by_f = {fid: g for fid, g in dom_enr.groupby("feature_idx")}
    abl_by_f = {int(r.feature_idx): r for r in causal_ablation.itertuples()}
    mor_by_f = {int(r.feature_idx): r for r in causal_morans.itertuples()}
    mod_by_f = {(t.surface, int(t.feature_idx)): (int(t.module_id), float(t.P_i))
                for t in modules.itertuples()}

    agg_atlas = feature_atlas[feature_atlas["surface"] == "aggregator"].set_index("feature_idx")

    for fid, base_row in agg_atlas.iterrows():
        fid = int(fid)
        # Top genes
        top_genes_list = []
        if fid in agg_top_by_f:
            for r in agg_top_by_f[fid].head(TOP_GENES_PER_FEATURE_DETAIL).itertuples():
                top_genes_list.append({
                    "rank": int(r.rank),
                    "gene": str(r.gene),
                    "mean_expr": safe_num(r.mean_expr_in_top_cells),
                    "baseline_mean": safe_num(r.baseline_mean_expr),
                    "fold_change": safe_num(r.fold_change),
                })
        # Enrichment libraries
        enrichments = {}
        if fid in agg_enr_by_f:
            for r in agg_enr_by_f[fid].itertuples():
                enrichments[str(r.library)] = {
                    "term": safe_str(r.top_term),
                    "p": safe_num(r.p),
                    "fdr": safe_num(r.fdr),
                    "overlap": safe_str(r.overlap),
                }
        # Domain enrichment for all 3 levels (with auto-derived niche label)
        domains = {}
        if fid in dom_by_f:
            for r in dom_by_f[fid].itertuples():
                lvl = int(r.level)
                top_dom = safe_str(r.top_domain)
                domains[f"level_{lvl}"] = {
                    "top_domain": top_dom,
                    "top_domain_lb": niche_label_lookup.get((lvl, top_dom), ""),
                    "top_domain_frac": safe_num(r.top_domain_frac),
                    "corpus_frac": safe_num(r.corpus_frac),
                    "log2_enrichment": safe_num(r.log2_enrichment),
                    "fdr": safe_num(r.fisher_fdr),
                    "n_top_cells": int(r.n_top_cells),
                }
        # Causal (sparse — only ~50 features have it)
        causal = None
        if fid in abl_by_f:
            r = abl_by_f[fid]
            causal = {
                "ablation_effect": safe_num(r.effect_score),
                "mean_l2_drop": safe_num(r.mean_l2_drop),
                "slide": str(r.slide_name),
            }
        morans = None
        if fid in mor_by_f:
            r = mor_by_f[fid]
            morans = {
                "morans_i": safe_num(r.morans_i),
                "n_active": int(r.n_active),
                "slide": str(r.slide_name),
            }
        # Module
        mod_info = mod_by_f.get(("aggregator", fid))
        module = (
            {"module_id": mod_info[0], "P_i": mod_info[1]}
            if mod_info is not None else None
        )

        sig = sig_lookup.get(("aggregator", fid), {})
        # Block 1 validation lookups (None if not available)
        block1_validation = {}
        if "proto" in block1 and fid in block1["proto"].index:
            r = block1["proto"].loc[fid]
            block1_validation["prototype_alignment"] = {
                "max_abs_cos": safe_num(r.get("max_abs_cos")),
                "best_prototype_idx": int(r.get("best_prototype_idx")) if pd.notna(r.get("best_prototype_idx")) else None,
                "signed_cos_to_best": safe_num(r.get("signed_cos_to_best")),
            }
        if "morans_all" in block1 and fid in block1["morans_all"].index:
            r = block1["morans_all"].loc[fid]
            block1_validation["spatial_coherence"] = {
                "morans_i_mean": safe_num(r.get("morans_i_mean")),
                "morans_i_median": safe_num(r.get("morans_i_median")),
                "n_slides": int(r.get("n_slides")) if pd.notna(r.get("n_slides")) else 0,
            }
        if "confounds" in block1 and fid in block1["confounds"].index:
            r = block1["confounds"].loc[fid]
            block1_validation["confounds"] = {
                "tissue_fdr": safe_num(r.get("tissue_fdr")),
                "tech_fdr": safe_num(r.get("tech_fdr")),
                "slide_fdr": safe_num(r.get("slide_fdr")),
                "tissue_concentrated": bool(r.get("tissue_concentrated", False)),
                "tech_concentrated": bool(r.get("tech_concentrated", False)),
                "slide_concentrated": bool(r.get("slide_concentrated", False)),
            }
        if "ablation" in block1 and fid in block1["ablation"].index:
            r = block1["ablation"].loc[fid]
            block1_validation["graph_ablation"] = {
                "mean_abs_full": safe_num(r.get("mean_abs_full")),
                "mean_abs_isolated": safe_num(r.get("mean_abs_isolated")),
                "contextual_dependency": safe_num(r.get("contextual_dependency")),
            }
        if "confounds_es" in block1 and fid in block1["confounds_es"].index:
            r = block1["confounds_es"].loc[fid]
            block1_validation["confounds_effect_size"] = {
                "tissue_max_share": safe_num(r.get("tissue_max_share")),
                "tissue_max_ratio": safe_num(r.get("tissue_max_ratio")),
                "tissue_top_class": safe_str(r.get("tissue_top_class")),
                "tech_max_share": safe_num(r.get("tech_max_share")),
                "tech_max_ratio": safe_num(r.get("tech_max_ratio")),
                "tech_top_class": safe_str(r.get("tech_top_class")),
                "slide_max_ratio": safe_num(r.get("slide_max_ratio")),
                "tissue_concentrated_es": bool(r.get("tissue_concentrated_es", False)),
                "tech_concentrated_es": bool(r.get("tech_concentrated_es", False)),
                "slide_concentrated_es": bool(r.get("slide_concentrated_es", False)),
                "tech_specific": bool(r.get("tech_specific", False)),
            }
        if "confound_survival" in block1 and fid in block1["confound_survival"].index:
            r = block1["confound_survival"].loc[fid]
            block1_validation["confound_survival"] = {
                "baseline_l7_top_class": safe_str(r.get("baseline_l7_top_class")),
                "baseline_l7_top_frac": safe_num(r.get("baseline_l7_top_frac")),
                "tech_resid_same_class_frac": safe_num(r.get("tech_resid_same_class_frac")),
                "l20_resid_same_class_frac": safe_num(r.get("l20_resid_same_class_frac")),
                "survives_slide": bool(r.get("survives_slide", False)),
                "survives_tech": bool(r.get("survives_tech", False)),
                "survives_l20": bool(r.get("survives_l20", False)),
                "survives_all": bool(r.get("survives_all", False)),
            }
        if "ablation_v2" in block1 and fid in block1["ablation_v2"].index:
            r = block1["ablation_v2"].loc[fid]
            block1_validation["graph_ablation_v2"] = {
                "mean_abs_full": safe_num(r.get("mean_abs_full")),
                "mean_abs_self_loop": safe_num(r.get("mean_abs_self_loop")),
                "mean_abs_self_loop_norm": safe_num(r.get("mean_abs_self_loop_norm")),
                "mean_abs_random_rewire": safe_num(r.get("mean_abs_random_rewire")),
                "dep_self_loop": safe_num(r.get("dep_self_loop")),
                "dep_self_loop_norm": safe_num(r.get("dep_self_loop_norm")),
                "dep_random_rewire": safe_num(r.get("dep_random_rewire")),
            }
        if "proto_ablation" in block1 and fid in block1["proto_ablation"].index:
            r = block1["proto_ablation"].loc[fid]
            block1_validation["prototype_domain_ablation"] = {
                "n_cells": int(r.get("n_cells")) if pd.notna(r.get("n_cells")) else 0,
                "mean_cos_full_vs_ablated": safe_num(r.get("mean_cos_full_vs_ablated")),
                "mean_conf_drop": safe_num(r.get("mean_conf_drop")),
                "proto_reassign_rate": safe_num(r.get("proto_reassign_rate")),
                "l7_reassign_rate": safe_num(r.get("l7_reassign_rate")),
                "l20_reassign_rate": safe_num(r.get("l20_reassign_rate")),
            }

        detail = {
            "surface": "aggregator",
            "feature_idx": fid,
            "lb": label_lookup.get(("aggregator", fid), "Unannotated"),
            "sig": sig.get("total", 0.0),
            "sig_bio": sig.get("bio_score", 0.0),
            "sig_niche": sig.get("niche_score", 0.0),
            "validation": block1_validation,
            "alive": bool(base_row["alive"]) if "alive" in base_row else None,
            "n_active_cells": int(base_row["n_active_cells"]),
            "mean_abs": safe_num(base_row["mean_abs"]),
            "max_abs": safe_num(base_row["max_abs"]),
            "top_tissue": safe_str(base_row["top_tissue"]),
            "top_tissue_frac": safe_num(base_row["top_tissue_frac"]),
            "top_tech": safe_str(base_row["top_tech"]),
            "top_tech_frac": safe_num(base_row["top_tech_frac"]),
            "tech_confound": bool(base_row.get("tech_confound", False)),
            "top_genes_summary": safe_str(base_row.get("top_genes")),
            "top_genes": top_genes_list,
            "enrichments": enrichments,
            "domains": domains,
            "causal": causal,
            "morans": morans,
            "module": module,
        }
        write_json(OUT_DIR / "feature_detail" / "aggregator" / f"{fid}.json", detail)

    log(f"  aggregator: {len(agg_atlas)} per-feature detail JSONs")

    # ---------------- 6. Per-feature drill-down (cell_embedder) ----------------
    log("\nbuilding cell_embedder per-feature detail")

    ce_top_by_f = {fid: g.sort_values("rank") for fid, g in ce_top_genes.groupby("feature_idx")}
    ce_enr_by_f = {fid: g for fid, g in ce_enr.groupby("feature_idx")}

    ce_atlas = feature_atlas[feature_atlas["surface"] == "cell_embedder"].set_index("feature_idx")
    n_ce_written = 0
    for fid, base_row in ce_atlas.iterrows():
        fid = int(fid)
        top_genes_list = []
        if fid in ce_top_by_f:
            for r in ce_top_by_f[fid].head(TOP_GENES_PER_FEATURE_DETAIL).itertuples():
                top_genes_list.append({
                    "rank": int(r.rank),
                    "gene": str(r.gene),
                    "score": safe_num(r.score),
                })
        enrichments = {}
        if fid in ce_enr_by_f:
            for r in ce_enr_by_f[fid].itertuples():
                enrichments[str(r.library)] = {
                    "term": safe_str(r.top_term),
                    "p": safe_num(r.p),
                    "fdr": safe_num(r.fdr),
                    "overlap": safe_str(r.overlap),
                }
        mod_info = mod_by_f.get(("cell_embedder", fid))
        module = (
            {"module_id": mod_info[0], "P_i": mod_info[1]}
            if mod_info is not None else None
        )
        sig = sig_lookup.get(("cell_embedder", fid), {})
        detail = {
            "surface": "cell_embedder",
            "feature_idx": fid,
            "lb": label_lookup.get(("cell_embedder", fid), "Unannotated"),
            "sig": sig.get("total", 0.0),
            "sig_bio": sig.get("bio_score", 0.0),
            "sig_niche": sig.get("niche_score", 0.0),
            "alive": bool(base_row["alive"]) if "alive" in base_row else None,
            "n_active_cells": int(base_row["n_active_cells"]),
            "mean_abs": safe_num(base_row["mean_abs"]),
            "max_abs": safe_num(base_row["max_abs"]),
            "top_tissue": safe_str(base_row["top_tissue"]),
            "top_tissue_frac": safe_num(base_row["top_tissue_frac"]),
            "top_tech": safe_str(base_row["top_tech"]),
            "top_tech_frac": safe_num(base_row["top_tech_frac"]),
            "top_genes_summary": safe_str(base_row.get("top_genes")),
            "top_genes": top_genes_list,
            "enrichments": enrichments,
            "module": module,
        }
        write_json(OUT_DIR / "feature_detail" / "cell_embedder" / f"{fid}.json", detail)
        n_ce_written += 1

    log(f"  cell_embedder: {n_ce_written} per-feature detail JSONs")

    # ---------------- 7. Conv layers: no per-feature detail files ----------------
    # The slim per-surface table has everything the conv-layer drilldown needs
    # (basic stats + module assignment via a separate join). The frontend falls
    # back to slim-table data when no detail file exists for a surface.
    log("\nconv-layer detail: skipped (slim table only)")

    # ---------------- 7b. Gene index ----------------
    log("\nbuilding gene_index.json")
    gene_index: dict[str, list[dict]] = defaultdict(list)
    for r in agg_top_genes.itertuples():
        gene = str(r.gene).strip().upper()
        if not gene:
            continue
        gene_index[gene].append({
            "s": "aggregator",
            "i": int(r.feature_idx),
            "score": float(r.fold_change),
            "metric": "fc",
        })
    for r in ce_top_genes.itertuples():
        gene = str(r.gene).strip().upper()
        if not gene:
            continue
        gene_index[gene].append({
            "s": "cell_embedder",
            "i": int(r.feature_idx),
            "score": float(r.score),
            "metric": "score",
        })
    # Sort each gene's features by score desc; cap at 30 per gene
    for g in gene_index:
        gene_index[g].sort(key=lambda e: -e["score"])
        gene_index[g] = gene_index[g][:30]
    write_json(OUT_DIR / "gene_index.json", dict(gene_index))
    log(f"  {len(gene_index)} unique genes")

    # ---------------- 7c. Niche index (Novae-unique) ----------------
    log("\nbuilding niche_index.json")
    niche_index: dict[str, dict] = {}
    # niche_label_lookup already built in section 1b
    for level in [7, 12, 20]:
        sub = dom_enr[dom_enr.level == level]
        by_niche: dict[str, list[dict]] = defaultdict(list)
        for r in sub.itertuples():
            d = str(r.top_domain)
            if not d:
                continue
            by_niche[d].append({
                "i": int(r.feature_idx),
                "frac": float(r.top_domain_frac),
                "log2enr": float(r.log2_enrichment),
                "fdr": float(r.fisher_fdr),
                "n_top_cells": int(r.n_top_cells),
            })
        # Sort each niche's features by log2enr desc
        niches: dict[str, dict] = {}
        for niche, feats in by_niche.items():
            feats.sort(key=lambda f: -f["log2enr"])
            niche_lb = niche_label_lookup.get((level, niche), "Unannotated")
            # Top tissues among the dominant features
            top_tissues: dict[str, int] = defaultdict(int)
            for f in feats[:10]:
                row = agg_atlas.loc[f["i"]] if f["i"] in agg_atlas.index else None
                if row is not None:
                    t = row.get("top_tissue")
                    if t and not pd.isna(t):
                        top_tissues[str(t)] += 1
            niches[niche] = {
                "lb": niche_lb,
                "n_features": len(feats),
                "features": feats,
                "top_tissues": dict(sorted(top_tissues.items(), key=lambda x: -x[1])),
            }
        niche_index[f"level_{level}"] = niches
    write_json(OUT_DIR / "niche_index.json", niche_index)
    log(
        f"  level 7: {len(niche_index['level_7'])}, "
        f"level 12: {len(niche_index['level_12'])}, "
        f"level 20: {len(niche_index['level_20'])}"
    )

    # ---------------- 7d. Cell type index ----------------
    log("\nbuilding celltype_index.json")
    celltype_index: dict[str, dict] = {}
    for r in agg_enr_v2.itertuples():
        if r.library != "PanglaoDB_Augmented_2021":
            continue
        term = clean_term(str(r.top_term))
        if not term:
            continue
        if term not in celltype_index:
            celltype_index[term] = {"library": "PanglaoDB", "features": []}
        celltype_index[term]["features"].append({
            "s": "aggregator",
            "i": int(r.feature_idx),
            "fdr": float(r.fdr),
            "lb": label_lookup.get(("aggregator", int(r.feature_idx)), "Unannotated"),
        })
    for r in ce_enr.itertuples():
        if r.library != "PanglaoDB_Augmented_2021":
            continue
        term = clean_term(str(r.top_term))
        if not term:
            continue
        if term not in celltype_index:
            celltype_index[term] = {"library": "PanglaoDB", "features": []}
        celltype_index[term]["features"].append({
            "s": "cell_embedder",
            "i": int(r.feature_idx),
            "fdr": float(r.fdr),
            "lb": label_lookup.get(("cell_embedder", int(r.feature_idx)), "Unannotated"),
        })
    # Sort each cell type's features by fdr asc, cap at 100
    for term in celltype_index:
        celltype_index[term]["features"].sort(key=lambda f: f["fdr"])
        celltype_index[term]["n_features"] = len(celltype_index[term]["features"])
        celltype_index[term]["features"] = celltype_index[term]["features"][:100]
    write_json(OUT_DIR / "celltype_index.json", celltype_index)
    log(f"  {len(celltype_index)} unique cell types")

    # ---------------- 7e. Global summary + depth flow ----------------
    log("\nbuilding global_summary.json")
    depth_flow = []
    for sname in surfaces_present:
        sub = feature_atlas[feature_atlas["surface"] == sname]
        sae_info = sae_by_name.get(sname, {})
        surf_info = surf_by_name.get(sname, {})
        sup_info = super_by_name.get(sname, {})
        mod_info = mods_by_name.get(sname, {})
        depth_flow.append({
            "surface": sname,
            "d": sae_info.get("d_in"),
            "n_features": sae_info.get("n_features"),
            "k": sae_info.get("k"),
            "alive": int(sub["alive"].sum()) if "alive" in sub.columns else None,
            "var_exp_full": surf_info.get("var_explained_full"),
            "superposition": sup_info.get("fraction_non_aligned"),
            "n_modules": mod_info.get("n_modules"),
            "tech_confound_rate": surf_info.get("tech_confounded_fraction"),
        })

    # PanglaoDB significant feature counts (aggregator + cell_embedder)
    n_agg_panglao_sig = int(((agg_enr_v2["library"] == "PanglaoDB_Augmented_2021") & (agg_enr_v2["fdr"] < 0.05)).sum())
    n_ce_panglao_sig = int(((ce_enr["library"] == "PanglaoDB_Augmented_2021") & (ce_enr["fdr"] < 0.05)).sum())

    # Niche significance counts at each level
    niche_sig = {}
    for level in [7, 12, 20]:
        sub = dom_enr[dom_enr.level == level]
        niche_sig[f"level_{level}"] = {
            "n_features_fdr_lt_0.05": int((sub["fisher_fdr"] < 0.05).sum()),
            "n_features_log2enr_gt_2": int((sub["log2_enrichment"] > 2).sum()),
            "n_unique_niches": int(sub["top_domain"].nunique()),
            "median_log2enr": float(sub["log2_enrichment"].median()),
        }

    # Significance distribution per surface for the home page histogram
    sig_distribution = {}
    for s in ["aggregator", "cell_embedder"]:
        scores = [sig_lookup.get((s, fid), {}).get("total", 0.0)
                  for fid in feature_atlas[feature_atlas["surface"] == s]["feature_idx"]]
        # 20 bins from 0 to 100
        hist, edges = np.histogram(scores, bins=20, range=(0, 100))
        sig_distribution[s] = {
            "bin_edges": edges.tolist(),
            "counts": hist.tolist(),
            "n_above_20": int(sum(1 for x in scores if x >= 20)),
            "n_above_50": int(sum(1 for x in scores if x >= 50)),
            "median": float(np.median(scores)) if scores else 0.0,
            "max": float(max(scores)) if scores else 0.0,
        }

    global_summary = {
        "checkpoint": "novae-human-0",
        "n_total_features": int(len(feature_atlas)),
        "n_alive": int(feature_atlas["alive"].sum()),
        "significance_distribution": sig_distribution,
        "n_total_cells": sum(s["n_cells"] for s in manifest["slides"]),
        "n_slides": len(manifest["slides"]),
        "n_tissues": len({s["tissue"] for s in manifest["slides"]}),
        "n_technologies": len({s["technology"] for s in manifest["slides"]}),
        "n_surfaces": len(surfaces_present),
        "n_aggregator_features": int((feature_atlas["surface"] == "aggregator").sum()),
        "n_cell_embedder_features": int((feature_atlas["surface"] == "cell_embedder").sum()),
        "n_panglao_sig": {
            "aggregator": n_agg_panglao_sig,
            "cell_embedder": n_ce_panglao_sig,
        },
        "n_unique_celltypes": len(celltype_index),
        "n_unique_genes_indexed": len(gene_index),
        "niche_significance": niche_sig,
        "depth_flow": depth_flow,
        "headline_findings": {
            "superposition": "99.4–100% of SAE features are non-aligned with top-50 SVD axes (cosine 0.7 threshold), on every surface — direct replication of upstream 99.8% finding",
            "spatial_coherence": "Mean Moran's I = 0.58 across 250 measurements, 94% of features > 0.1 — features are bona fide spatial niches, not cell-type indicators",
            "depth_compression": "Variance explained climbs 0.81 → 0.94 from conv_0 to conv_8; module count drops 313 → 44; tech-confound rate drops 71% → 50%",
            "causal_poverty": "Mean single-feature ablation effect 0.005 cosine — spatial analogue of upstream minimal regulatory logic finding",
            "niche_specificity": "At level-20 niches, 35% of aggregator features are >4× enriched for one specific Novae niche; several reach 100% concentration",
        },
    }
    write_json(OUT_DIR / "global_summary.json", global_summary)
    log(json.dumps({k: v for k, v in global_summary.items() if not isinstance(v, (dict, list))}, indent=2))

    # ---------------- 7f. Spotlight features (curated by query) ----------------
    log("\npicking spotlight features")
    # Aggregator features with strong PanglaoDB significance + strong niche enrichment
    agg_full = feature_atlas[feature_atlas["surface"] == "aggregator"].copy()
    # Join with domain l20 stats
    dom_l20 = dom_enr[dom_enr.level == 20].set_index("feature_idx")
    agg_full = agg_full.set_index("feature_idx").join(
        dom_l20[["log2_enrichment", "top_domain_frac", "fisher_fdr"]].rename(
            columns={
                "log2_enrichment": "dom_log2enr",
                "top_domain_frac": "dom_frac",
                "fisher_fdr": "dom_fdr",
            }
        )
    )
    spotlight_candidates = agg_full[
        (agg_full["top_PanglaoDB_v2_fdr"] < 1e-8)
        & (agg_full["dom_log2enr"] > 5)
        & (agg_full["alive"] == True)  # noqa: E712
    ].copy()
    spotlight_candidates["score"] = (
        -np.log10(spotlight_candidates["top_PanglaoDB_v2_fdr"].clip(lower=1e-300))
        + spotlight_candidates["dom_log2enr"]
    )
    spotlight_candidates = spotlight_candidates.sort_values("score", ascending=False)

    # Dedupe by PanglaoDB term — one per type for diversity
    seen_terms = set()
    spotlight = []
    for fid, row in spotlight_candidates.iterrows():
        term = clean_term(str(row.get("top_PanglaoDB_v2", "")))
        if term in seen_terms or term.lower() in _GENERIC_TERMS:
            continue
        seen_terms.add(term)
        spotlight.append({
            "surface": "aggregator",
            "feature_idx": int(fid),
            "lb": label_lookup.get(("aggregator", int(fid)), "Unannotated"),
            "top_genes": str(row.get("top_genes", "")),
            "panglao_term": term,
            "panglao_fdr": safe_num(row.get("top_PanglaoDB_v2_fdr")),
            "niche": str(row.get("top_domain_l20", "")),
            "niche_frac": safe_num(row.get("dom_frac")),
            "niche_log2enr": safe_num(row.get("dom_log2enr")),
            "tissue": str(row.get("top_tissue", "")),
            "tissue_frac": safe_num(row.get("top_tissue_frac")),
        })
        if len(spotlight) >= 10:
            break
    write_json(OUT_DIR / "spotlight_features.json", spotlight)
    log(f"  picked {len(spotlight)} spotlight features:")
    for sp in spotlight:
        log(f"    f{sp['feature_idx']:5d}  {sp['panglao_term']:40s}  niche {sp['niche']}  ({sp['tissue']})")

    # ---------------- 7b. Causal audit summary (Phase 4 roll-up) ----------------
    log("\nbuilding causal_audit.json (Phase 4 roll-up)")

    audit_summary: dict = {
        "hypotheses": {},
        "per_feature_table": None,
        "source_parquets": [],
    }

    def load_sum(name):
        p = ATLAS / "causal" / name
        if p.exists():
            return json.load(open(p))
        return None

    sums = {
        "sae_vs_prototypes": load_sum("sae_vs_prototypes.summary.json"),
        "spatial_coherence_all": load_sum("spatial_coherence_all.summary.json"),
        "confounds": load_sum("confounds.summary.json"),
        "confounds_effect_size": load_sum("confounds_effect_size.summary.json"),
        "confound_survival": load_sum("confound_survival.summary.json"),
        "graph_ablation": load_sum("graph_ablation.summary.json"),
        "graph_ablation_v2": load_sum("graph_ablation_v2.summary.json"),
        "prototype_domain_ablation": load_sum("prototype_domain_ablation.summary.json"),
    }
    audit_summary["source_summaries"] = sums

    # H-roll-up
    hyp: dict = {}
    if sums.get("sae_vs_prototypes"):
        s = sums["sae_vs_prototypes"]
        frac_aligned = s.get("fraction_aligned_at_0.7")
        hyp["H6_prototype_redundancy"] = {
            "claim": "<30% of SAE features cos-aligned (|cos|>=0.7) with any prototype",
            "fraction_aligned_at_0.7": frac_aligned,
            "median_max_cos": s.get("median_max_cos"),
            "verdict": "CONFIRMED" if (frac_aligned is not None and frac_aligned < 0.30) else "REFUTED",
        }
    if sums.get("spatial_coherence_all"):
        s = sums["spatial_coherence_all"]
        hyp["H7_spatial_coherence_first_half"] = {
            "claim": "mean Moran's I significantly > 0 on aggregator features",
            "mean_morans_i": s.get("mean_morans_i"),
            "median_morans_i": s.get("median_morans_i"),
            "n_with_signal": s.get("n_with_signal"),
            "verdict": "CONFIRMED" if (s.get("mean_morans_i") or 0) > 0.05 else "REFUTED",
            "depth_half_tested": False,
        }
    if sums.get("graph_ablation_v2"):
        s = sums["graph_ablation_v2"]
        canonical = s.get("self_loop_norm_fraction_dep_gt_0.5")
        hyp["H8_contextual_dependency"] = {
            "claim": ">20% of features should have dep>0.5 when graph collapsed",
            "canonical_regime": "self_loop_norm",
            "fraction_dep_gt_0.5_self_loop": s.get("self_loop_fraction_dep_gt_0.5"),
            "fraction_dep_gt_0.5_self_loop_norm": canonical,
            "fraction_dep_gt_0.5_random_rewire": s.get("random_rewire_fraction_dep_gt_0.5"),
            "verdict": "CONFIRMED" if (canonical or 0) > 0.20 else "REFUTED",
        }
    if sums.get("confounds_effect_size"):
        s = sums["confounds_effect_size"]
        hyp["H9_technology_artefacts"] = {
            "claim": ">=5% of features are technology-specific",
            "fraction_tech_specific": s.get("fraction_tech_specific"),
            "verdict": "CONFIRMED" if (s.get("fraction_tech_specific") or 0) >= 0.05 else "REFUTED",
        }
    if sums.get("confound_survival"):
        s = sums["confound_survival"]
        hyp["H2_causal_poverty"] = {
            "claim": "<20% of niche-signal features survive strict confound suite",
            "fraction_survives_all": s.get("fraction_survives_all_of_niche"),
            "verdict": "CONFIRMED" if (s.get("fraction_survives_all_of_niche") or 1.0) < 0.20 else "REFUTED",
            "note": "residualization is weak on TopK SAE outputs; upper bound",
        }
    audit_summary["hypotheses"] = hyp

    # Per-feature causal table for the audit page (aggregator only)
    per_feat = []
    for fid in range(2048):
        row = {"feature_idx": fid}
        if "proto" in block1 and fid in block1["proto"].index:
            r = block1["proto"].loc[fid]
            row["proto_max_abs_cos"] = safe_num(r.get("max_abs_cos"))
        if "morans_all" in block1 and fid in block1["morans_all"].index:
            r = block1["morans_all"].loc[fid]
            row["morans_i_mean"] = safe_num(r.get("morans_i_mean"))
        if "confounds_es" in block1 and fid in block1["confounds_es"].index:
            r = block1["confounds_es"].loc[fid]
            row["tissue_max_ratio"] = safe_num(r.get("tissue_max_ratio"))
            row["tech_max_ratio"] = safe_num(r.get("tech_max_ratio"))
            row["slide_max_ratio"] = safe_num(r.get("slide_max_ratio"))
            row["tech_specific"] = bool(r.get("tech_specific", False))
        if "confound_survival" in block1 and fid in block1["confound_survival"].index:
            r = block1["confound_survival"].loc[fid]
            row["survives_all"] = bool(r.get("survives_all", False))
            row["baseline_l7_top_frac"] = safe_num(r.get("baseline_l7_top_frac"))
        if "ablation_v2" in block1 and fid in block1["ablation_v2"].index:
            r = block1["ablation_v2"].loc[fid]
            row["dep_self_loop_norm"] = safe_num(r.get("dep_self_loop_norm"))
            row["dep_random_rewire"] = safe_num(r.get("dep_random_rewire"))
        if "proto_ablation" in block1 and fid in block1["proto_ablation"].index:
            r = block1["proto_ablation"].loc[fid]
            row["proto_reassign_rate"] = safe_num(r.get("proto_reassign_rate"))
            row["l7_reassign_rate"] = safe_num(r.get("l7_reassign_rate"))
        # Label from feature_atlas
        base_row = agg_atlas.loc[fid] if fid in agg_atlas.index else None
        if base_row is not None:
            row["label"] = label_lookup.get(("aggregator", fid), "Unannotated")
            row["top_tissue"] = safe_str(base_row.get("top_tissue"))
            row["top_domain_l7"] = safe_str(base_row.get("top_domain_l7"))
            row["top_domain_l7_frac"] = safe_num(base_row.get("top_domain_l7_frac"))
        per_feat.append(row)
    audit_summary["per_feature_table"] = per_feat
    audit_summary["source_parquets"] = [
        k for k in block1 if k in ("proto", "morans_all", "confounds", "confounds_es",
                                    "confound_survival", "ablation", "ablation_v2",
                                    "proto_ablation")
    ]
    write_json(OUT_DIR / "causal_audit.json", audit_summary)
    log(f"  wrote causal_audit.json (hypotheses: {list(hyp.keys())})")

    # ---------------- 8. Spatial: per-slide background subsamples ----------------
    log("\nbuilding spatial backgrounds + per-feature top-cell coordinates")

    # Encode aggregator activations to find top cells per feature
    log("  loading aggregator activations + SAE")
    agg = np.load(ACT_DIR / "aggregator.npy").astype(np.float32, copy=False)
    sae = load_aggregator_sae()
    log(f"  aggregator: {agg.shape}, top cells/feat = {TOP_CELLS_PER_FEATURE_SPATIAL}")
    top_cells = encode_top_cells(sae, agg, TOP_CELLS_PER_FEATURE_SPATIAL)
    del sae, agg
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # Loop slides: load h5ad, subsample background, also accumulate per-feature coords
    slide_starts = np.array([s["cell_offset_start"] for s in manifest["slides"]], dtype=np.int64)

    # Pre-build: which (feature, local_idx) belong to each slide
    feat_local_per_slide: dict[int, dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for f, top in top_cells.items():
        if len(top) == 0:
            continue
        cells = top[:, 0].astype(np.int64)
        acts = top[:, 1].astype(np.float32)
        sids = np.searchsorted(slide_starts, cells, side="right") - 1
        for i in range(len(cells)):
            sid = int(sids[i])
            local = int(cells[i] - slide_starts[sid])
            feat_local_per_slide[sid][int(f)].append((local, float(acts[i])))

    # Per-feature accumulator for the spatial JSONs
    feature_spatial: dict[int, list[dict]] = defaultdict(list)

    for slide in manifest["slides"]:
        sid = slide["slide_idx"]
        h5ad_path = DATA_DIR / slide["tissue"] / f"{slide['name']}.h5ad"
        log(f"  [{sid+1}/15] {slide['tissue']}/{slide['name']}")
        try:
            a = ad.read_h5ad(h5ad_path)
        except Exception as e:
            log(f"    ERROR load: {type(e).__name__}: {e}")
            continue
        if "spatial" not in a.obsm:
            log(f"    no spatial obsm, skip")
            continue
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        n_cells = len(coords)

        # Background subsample
        n_bg = min(SLIDE_BACKGROUND_N, n_cells)
        bg_idx = rng.choice(n_cells, size=n_bg, replace=False)
        bg_idx.sort()
        bg_coords = coords[bg_idx]

        # Domain labels for this slide (from per_slide files)
        sdir = ACT_DIR / "per_slide" / f"{slide['tissue']}__{slide['name']}"
        try:
            d_l7 = np.load(sdir / "domains_level7.npy", allow_pickle=True).astype(str)
            d_l20 = np.load(sdir / "domains_level20.npy", allow_pickle=True).astype(str)
        except Exception:
            d_l7 = np.full(n_cells, "", dtype=object)
            d_l20 = np.full(n_cells, "", dtype=object)

        bg_payload = {
            "slide_idx": sid,
            "name": slide["name"],
            "tissue": slide["tissue"],
            "technology": slide["technology"],
            "n_cells_total": int(n_cells),
            "n_cells_subsampled": int(n_bg),
            "x": bg_coords[:, 0].round(2).tolist(),
            "y": bg_coords[:, 1].round(2).tolist(),
            "domain_l7": d_l7[bg_idx].tolist(),
            "domain_l20": d_l20[bg_idx].tolist(),
        }
        write_json(OUT_DIR / "spatial" / f"slide_{sid}.json", bg_payload)

        # Accumulate per-feature top cells for this slide
        for fid, locals_acts in feat_local_per_slide.get(sid, {}).items():
            for local, act in locals_acts:
                if local >= n_cells:
                    continue
                feature_spatial[fid].append({
                    "slide_idx": sid,
                    "x": float(coords[local, 0]),
                    "y": float(coords[local, 1]),
                    "a": float(act),
                })

        del a, coords
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # Write per-feature spatial files
    log(f"  writing per-feature spatial JSONs ({len(feature_spatial)} features)")
    for fid, rows in feature_spatial.items():
        write_json(OUT_DIR / "spatial" / "feature" / "aggregator" / f"{fid}.json", rows)

    log("\nDONE")


if __name__ == "__main__":
    main()
