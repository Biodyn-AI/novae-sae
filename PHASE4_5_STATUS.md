# Phase 4 + Phase 5 status (novae-human-0)

Tracking what of RESEARCH_AND_IMPLEMENTATION_PLAN.md §4.5 (causal validation) and
§4.6 (online SAE atlas) has been completed, what's partial, and what's blocked.
Last updated: 2026-04-10.

---

## Phase 4 — Causal validation (§4.5 + §4.7)

### §4.5/1 Feature ablation with prototype + domain reassignment

- **Status**: DONE via `scripts/11_prototype_domain_ablation.py`.
- **Protocol**: top-50 aggregator features (filtered by `top_domain_l7_frac ≥ 0.4`
  and `n_active_cells ≥ 1000`) × 200 highest-activation cells each. For each cell,
  ablate the feature in the SAE code (set to 0), decode, compute prototype
  argmax against the SwAV head's 512 unit-norm prototypes, and compute l7/l20
  domain reassignment via a precomputed prototype → majority-domain map.
- **Outputs**: `atlas/novae-human-0/causal/prototype_domain_ablation.parquet`
  (per-cell) + `.per_feature.parquet` (per-feature aggregate) +
  `.summary.json`.
- **Supersedes**: the old `04_causal.py` which only measured activation drop,
  not downstream reassignment.

### §4.5/2 Graph ablation suite (three regimes)

- **Status**: DONE via `scripts/10b_graph_ablation_v2.py`.
- **Regimes implemented**:
  - (a) **full** — original Delaunay graph.
  - (b) **self_loop** — `sp.eye()` with weight 1.0 (the original naive
    protocol; scale-confounded).
  - (c) **self_loop_norm** — `sp.eye() * (1 / mean_degree)` (canonical H8
    regime; restores parity with full-graph softmax self-weight).
  - (d) **random_rewire** — degree-preserving random rewire of the Delaunay
    edges.
- **Slide subset**: brain, kidney, pancreas, skin, liver (~533k cells total)
  to keep wall time under ~25 min on MPS.
- **Outputs**: `graph_ablation_v2.parquet` + `.summary.json`. Canonical H8
  verdict comes from the `dep_self_loop_norm` column; `dep_random_rewire` is
  the strict-spatial-structure control.
- **Supersedes**: the initial `10_block1_graph_ablation.py` which only ran
  regimes (a) + (b) and got scale-confounded results (54% of features had
  iso > 2× full, median dep = −1.29). See
  `memory/feedback_graph_ablation_scale_confound.md` for the lesson.

### §4.5/3 Cross-technology coherence (Xenium ↔ MERSCOPE paired tissues)

- **Status**: NOT FEASIBLE with the current dataset.
- **Reason**: the `MICS-Lab/novae` corpus has 15 slides across 15 unique
  tissues (1 slide per tissue). Tech distribution is xenium×13, merscope×1,
  cosmx×1. The merscope slide covers head_and_neck and the cosmx slide
  covers a different tissue — **no tissue is covered by more than one
  technology**, so there are no paired-tissue pairs to correlate.
- **Follow-up**: add at least one Xenium slide on head_and_neck (or extend
  the corpus with MERSCOPE slides that match existing Xenium tissues) to
  unblock this test. Until then, skip with prejudice — this is a corpus
  gap, not a pipeline gap.
- **Weaker alternative** (not implemented): per-feature sensitivity to
  technology label (already captured by
  `confounds_effect_size.tech_max_ratio` and the `tech_specific` flag in
  that table).

### §4.5/4 Cross-organism transfer (human ↔ mouse via ortholog map)

- **Status**: BLOCKED on `novae-mouse-0` checkpoint.
- **What's missing**:
  - `checkpoints/novae-mouse-0/` not downloaded.
  - `activations/novae-mouse-0/*.npy` not extracted.
  - `saes/novae-mouse-0/*.pt` not trained.
  - Ortholog mapping table (human ↔ mouse gene orthologs; typically from
    BioMart).
- **Unblock recipe**:
  1. `novae.Novae.from_pretrained("MICS-Lab/novae-mouse-0")` → save to
     `checkpoints/novae-mouse-0/`.
  2. Download matched MERSCOPE/Xenium mouse-brain slides into
     `datasets/mics-lab-novae/mouse/`.
  3. Rerun Phases 1A–3A on `novae-mouse-0` with the existing scripts 01–03e
     pointed at `--checkpoint novae-mouse-0` (scripts currently hard-code
     novae-human-0; one-line path parameter needed).
  4. Write `scripts/14_cross_organism_transfer.py` that cross-encodes
     human cells with the mouse model via an ortholog map, trains a small
     cross-SAE, and measures feature overlap.

### §4.5/5 Perturbation datasets

- **Status**: NOT ATTEMPTED — likely blocked by dataset availability.
- **Reason**: spatial-transcriptomics perturbation screens at the scale of
  Replogle CRISPRi (100+ targets × thousands of cells/target) are not yet
  in the public domain at the time of writing. The closest candidates are
  small-n spatial lineage-tracing studies.
- **Follow-up**: audit whether any of the 2024–2025 spatial perturbation
  preprints match the protocol (e.g. MERFISH-ex, Perturb-spatial). If yes,
  implement `scripts/15_perturbation_validation.py` mirroring upstream's
  Wilcoxon + BH + TF-target Fisher protocol.

### §4.5/6 Cross-checkpoint feature alignment (CCA)

- **Status**: BLOCKED on both `novae-mouse-0` and `novae-brain-0`
  checkpoints + full pipeline runs on each.
- **What's missing**: everything under §4.5/4 plus `novae-brain-0`
  variants.
- **Unblock recipe**: same as §4.5/4, then an additional script
  `scripts/16_cross_checkpoint_cca.py` that does CCA on top-50 features
  between each pair of checkpoints on orthologous tissues.

### §4.7 Confound controls (cross-cutting, mandatory)

- **Slide / batch shuffles**: DONE as effect-size max-class-ratio test in
  `scripts/09b_effect_size_confounds.py` (analytically equivalent to the
  empirical slide-shuffle null for the top-class-share statistic).
- **Cell-type residualization**: DONE via l20-domain residualization in
  `scripts/12_confound_suite.py`. **Caveat**: residualization is weak on
  TopK SAE outputs because most cells have exactly zero activation for
  any given feature, so subtracting per-class means barely changes the
  top-cell ranking. The resulting `survives_l20` flag rejects few
  features and is therefore a **loose upper bound** on true survival.
  A stronger test would re-train the SAE on activations with cell-type
  means regressed out at the aggregator level — that's ~1 additional day
  of SAE training compute.
- **Technology residualization**: DONE via the same residualization trick
  with tech labels. Same caveat applies; and note the corpus has
  13/15 slides on xenium, so tech is essentially the slide-level label
  for 87% of cells. `confounds_effect_size.tech_max_ratio` (median 1.08)
  already shows that tech is not a strong confound.
- **Degree-preserving graph rewire**: DONE as regime (d) of
  `scripts/10b_graph_ablation_v2.py`, producing `dep_random_rewire`.
- **Permutation null per PMI/Fisher**: PARTIAL — the per-cell-top-class
  test in `09b` is a permutation-equivalent proxy. True per-enrichment
  permutation nulls for the PMI/Fisher tests in `03c_bio_grounding.py`
  are NOT DONE. Follow-up: re-run `03c_bio_grounding.py` with K=100
  label shuffles and record the shuffle-null distribution per feature ×
  library.
- **Hierarchical null model** (label perm → rewire → cell-type matching →
  max-null): NOT DONE. This is a single pooling script that combines
  the four survival flags above (`confound_survival.parquet`) into a
  hierarchical survival count, reporting the manifesto's "~10%"
  calibration. Write `scripts/17_hierarchical_null.py` as a short
  post-processing pass.

---

## Phase 5 — Online SAE atlas (§4.6)

### What's already shipped

- React + Vite + Tailwind build in `atlas/web/` with pages:
  Home, Tour, Stories, Surfaces (index + detail), Feature Detail,
  Modules, Niches (browse), Cell types (browse), Gene search, Spatial,
  About. Data is baked static JSON from `scripts/05_build_atlas_data.py`.
- GitHub Pages deploy workflow targeting `Biodyn-AI/novae-atlas`.
- Auto-derived niche labels from the aggregator PanglaoDB map.
- Cell-cell communication computed by `scripts/06_cell_cell_communication.py`.

### New in this session (2026-04-10)

- **Causal Audit page** (`atlas/web/src/pages/CausalAuditPage.jsx` + route
  `/causal` + navbar link) — per-feature causal roll-up table with
  hypothesis verdict cards. Loads `/data/causal_audit.json`.
- **`causal_audit.json` builder** added to `scripts/05_build_atlas_data.py`
  consolidating all the §4.5 + §4.7 summary parquets into one file.
- **Per-feature detail pages** now include `confounds_effect_size`,
  `confound_survival`, `graph_ablation_v2`, and
  `prototype_domain_ablation` fields under `validation.*`.

### Still to do on Phase 5

| item | status | what's needed |
|---|---|---|
| Cross-Layer Flow (Sankey, H10) | NOT DONE | Requires H10 script (layer L features linearly decomposing layer L+k features); out of scope for this session. |
| Cross-checkpoint Compare page | BLOCKED | Needs novae-mouse-0 and novae-brain-0 pipelines; blocked on §4.5/4, §4.5/6. |
| `novae-mouse-atlas`, `novae-brain-atlas`, `novae-cross-atlas` deploys | BLOCKED | Needs the other two checkpoints. Template re-use is free once data lands. |
| GitHub Pages first-deploy to `Biodyn-AI/novae-atlas` | PENDING | `.github/workflows/` workflow exists but repo may not have been pushed yet; confirm with `gh repo view Biodyn-AI/novae-atlas`. |
| Zenodo DOI snapshot | NOT DONE | Do at release time, not during iteration. |

---

## Summary of blockers

The following items **cannot** be completed in this environment without new
inputs:

1. **novae-mouse-0 + novae-brain-0 checkpoints** → blocks H5 (cross-checkpoint
   convergence), §4.5/4, §4.5/6, Cross-checkpoint Compare page, and the
   three additional atlases (mouse, brain, cross).
2. **Paired-technology tissues** → blocks §4.5/3 cross-technology coherence.
3. **Spatial perturbation screens** → blocks §4.5/5 perturbation validation.

Everything else in §4.5 + §4.7 + §4.6 Phase 5 has a script and an output
artefact on disk as of this session, though several items have known
methodological caveats (confound residualization is loose; H8 needs the
`self_loop_norm` + `random_rewire` regimes to interpret; per-enrichment
permutation nulls for §4.7 item 5 are still a TODO).
