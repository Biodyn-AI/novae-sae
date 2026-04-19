# Novae SAE: Sparse Autoencoder Atlas for the Novae Spatial Foundation Model

**Paper**: *A sparse autoencoder atlas of the Novae spatial foundation model, with matched-null controls for interpretability claims*

Anonymized repository released with a double-blind venue submission.
Authorship and affiliation information, interactive-atlas URLs, and
upstream-work citations have been removed; see the paper's camera-
ready version for attribution.

---

## Overview

This repository contains the complete analysis codebase for applying sparse autoencoder (SAE) mechanistic interpretability to [Novae](https://www.nature.com/articles/s41592-025-02899-6), a GATv2 graph foundation model for spatial transcriptomics. We train TopK SAEs on every internal layer of the `novae-human-0` checkpoint, extract a dictionary of 49,152 features, and apply a comprehensive validation suite in which every claim is calibrated against an appropriate null: panel-saturation for annotation rate, matched-magnitude random directions for single-feature ablation, feature-permutation for graph ablation, and raw expression for cross-technology coherence.

### Key findings

- Aggregator features are pervasively superposed: 0.0% align with any SwAV prototype at |cos|≥0.7; ≥99.4% are non-aligned with the top-50 SVD axes per surface.
- Under a doubly-controlled graph-ablation protocol (self-loop + feature-permutation), **26%** of features are genuinely context-dependent. Self-loop alone flags 67%; feature-permutation (graph intact, neighbor content scrambled) flags 40%.
- Of 50 prototype-ablation tests against a matched-magnitude random-direction null, **17 features exceed the null's upper 95% CI** (direction-specific load-bearing); 7 fall below the null's lower CI; 26 are statistically indistinguishable from random.
- Cross-technology Spearman ρ_SAE = 0.56 between Xenium and MERSCOPE feature profiles is indistinguishable from a raw-expression baseline of ρ_raw = 0.57; only **0.95%** of features are strictly technology-specific. The continuous cross-tech signal is inherited from shared panel biology, not added by the SAE.
- Causal circuit tracing on one brain slide yields **439 directed edges**; combinatorial triplet ablation gives R_ABC ∈ [0.957, 0.966] (bootstrap 95% CI) for one tested feature pair, a near-additive regime with component effects d ≈ 0.04.
- Spatial-gradient steering on the colon crypt–villus axis produces a **22 pp** change (39% → 61%) between attenuation and amplification, the informative effect-size statement against an asymmetric unperturbed baseline.

### Scope caveats

Several claims common in SAE-atlas work do not survive these calibrated controls. Annotation rate (100% FDR<0.05 per library) is a panel-saturation ceiling: a random 20-gene draw from the 487-gene Xenium panel already hits ≥99.95% against each Enrichr library. Cross-checkpoint convergence (H5) is not identified at the profile level because the three Novae checkpoints use non-overlapping gene panels. The single-slide circuit trace does not support claims about attenuation with layer depth or the generality of the observed synergy pattern; multi-slide replication is outlined as a planned extension.

## Repository structure

```
novae-sae/
├── README.md                          # This file
├── src/
│   └── topk_sae.py                    # TopK sparse autoencoder module
├── scripts/
│   ├── 01_extract_activations.py      # Phase 1: hook all 12 Novae surfaces
│   ├── 01b_extract_per_cell_conv.py   # Per-cell conv activations (H7 depth)
│   ├── 02_train_saes.py               # Phase 2: train TopK SAEs (32× expansion)
│   ├── 03_characterize.py             # Phase 3: var explained, alive features, superposition
│   ├── 03b_modules.py                 # Leiden module discovery on PMI graph
│   ├── 03c_bio_grounding.py           # Gene markers + Enrichr enrichment
│   ├── 03c2_bio_grounding_v2.py       # Improved enrichment pipeline
│   ├── 03c3_conv_enrichment.py        # Per-layer enrichment (H4)
│   ├── 03c3_domain_enrichment.py      # Niche domain enrichment
│   ├── 03d_assign_domains.py          # Novae niche assignment at 3 hierarchy levels
│   ├── 03e_merge_atlas.py             # Merge all characterization into feature atlas
│   ├── 04_causal.py                   # Phase 4: initial causal validation (Moran's I + ablation)
│   ├── 05_build_atlas_data.py         # Phase 5: generate static JSON for the web atlas
│   ├── 06_cell_cell_communication.py  # Cell-cell communication analysis
│   ├── 07_block1_sae_vs_prototypes.py # H6: SAE vs SwAV prototype alignment
│   ├── 08_block1_spatial_coherence_all.py  # H7: Moran's I on all features
│   ├── 08b_spatial_coherence_per_layer.py  # H7 depth: Moran's I per conv layer
│   ├── 09_block1_confound_suite.py    # Block 1.3: chi-square confound test
│   ├── 09b_effect_size_confounds.py   # H9: effect-size confound filter
│   ├── 10_block1_graph_ablation.py    # H8: graph ablation (self-loop collapse)
│   ├── 10b_graph_ablation_v2.py       # H8 v2: 4 regimes (self-loop, norm, rewire)
│   ├── 11_prototype_domain_ablation.py # §4.5/1: prototype + domain reassignment
│   ├── 12_confound_suite.py           # §4.7: tech + l20 residualization
│   ├── 13_hierarchical_null.py        # §4.7/6: 4-level confound survival gate
│   ├── 14_hierarchical_composition.py # H10: cross-layer feature alignment
│   ├── 15_permutation_null.py         # §4.7/5: label-shuffle permutation null
│   ├── 16_cross_tech_coherence.py     # §4.5/3: Xenium vs MERSCOPE coherence
│   ├── 17_perturbation_validation.py  # §4.5/5: Perturb-map CRISPR screen
│   ├── 18_cross_checkpoint_pipeline.py # H5: human/mouse/brain checkpoint comparison
│   ├── 19_causal_circuit_tracing.py   # GAP 1: causal circuit tracing across layers
│   ├── 20_circuit_analysis.py         # GAPs 2,3,6,7,8,14: circuit graph analysis
│   ├── 20b_enrich_circuit_hubs.py     # Targeted enrichment of circuit hub features
│   ├── 21_circuit_biology.py          # GAPs 5,10,11,12: gene predictions + disease mapping
│   ├── 22_combinatorial_ablation.py   # GAP 15: 3-way combinatorial ablation (synergy)
│   ├── 22b_exhaustive_tracing.py      # GAP 13: exhaustive tracing (200 features at conv_5)
│   ├── 23_spatial_steering.py         # GAP 16: spatial gradient steering (NOVEL)
│   ├── 24_pmi_vs_causal.py            # GAP 9: PMI co-activation vs causal edges
│   ├── 25_sae_dissociation.py         # GAP 17: real vs shuffled SAE dissociation
│   ├── 26_reviewer_controls.py        # Bootstrap R_ABC, gap-normalized attenuation,
│   │                                  #  random-gene null, steering baseline, novelty baseline
│   ├── 27_random_direction_null.py    # Matched-magnitude random-direction null for
│   │                                  #  prototype-reassignment ablation (50 features, 30 seeds)
│   ├── 28_crosstech_rawexpr_baseline.py # Raw-expression Spearman ρ baseline for §3.6
│   ├── 29_feature_perm_graph_ablation.py # Feature-permutation graph ablation (brain, kidney,
│   │                                     #  pancreas; 225,867 cells) — the H8 second control
│   ├── 30_multislide_circuit_tracing.py # Multi-slide circuit replication (planned)
│   ├── 31_random_init_sae_baseline.py   # Random-init Novae SAE baseline (planned)
│   ├── 32_sae_seed_stability.py         # SAE seed-stability test (planned)
│   ├── 33_perturbmap_composition_control.py # Composition-regressed Perturb-map (planned)
│   ├── 34_shuffled_labels_specificity.py # Shuffle-null specificity on tissue/term matching
│   ├── 35_v2_figures.py                # Generate matched-null figures (12, 13)
│   ├── 36_triplet_sensitivity.py       # Triplet-selection structure analysis
│   └── 99_summary.py                  # Generate summary statistics
├── paper/
│   ├── main.tex                       # Paper source
│   └── figures/                       # Generated figures (PDF, PNG)
├── atlas/novae-human-0/causal/reviewer_controls/
│   ├── A_rabc_bootstrap.json          # Bootstrap CI on R_ABC triplet synergy
│   ├── B_gap_normalized.{parquet,json} # Gap-normalized circuit edge counts
│   ├── C_random_gene_null.json         # Per-library panel-saturation hit rates
│   ├── D_steering_baseline.json        # Steering α-dependent push distributions
│   ├── E_gene_pair_novelty.json        # 30× random-pair enrichment for the 426 pairs
│   ├── F_random_direction_null.{parquet,json} # Matched-magnitude null ablation (50 features)
│   ├── G_crosstech_rawexpr_baseline.{parquet,json} # ρ_raw vs ρ_SAE (7 tissue pairs)
│   ├── H_feature_perm_ablation.{parquet,json}   # Feature-permutation dependency (3 slides)
│   ├── H_feature_perm_vs_selfloop.{parquet,json} # Merged self-loop + feature-perm analysis
│   ├── M_specificity.json              # Tissue-term specificity vs shuffle null
│   ├── P_triplet_sensitivity.json      # Triplet-selection structure analysis
│   └── summary.json                    # Aggregate summary of the above
└── configs/
    └── environment.yml                 # Conda environment specification
```

## Quick start

### 1. Environment setup

```bash
# Create conda environment
conda create -n novae-mi python=3.11
conda activate novae-mi

# Install dependencies
pip install novae torch torchvision torch-geometric
pip install anndata scanpy gseapy pandas numpy scipy scikit-learn
pip install statsmodels matplotlib

# Verify
python -c "import novae; print(novae.__version__)"
```

### 2. Download model checkpoints

```python
import novae

# Human checkpoint (primary)
m = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
m.save_pretrained("checkpoints/novae-human-0")

# Mouse + brain (for cross-checkpoint analysis)
m = novae.Novae.from_pretrained("MICS-Lab/novae-mouse-0")
m.save_pretrained("checkpoints/novae-mouse-0")
m = novae.Novae.from_pretrained("MICS-Lab/novae-brain-0")
m.save_pretrained("checkpoints/novae-brain-0")
```

### 3. Download spatial transcriptomics data

The activation corpus uses slides from the [MICS-Lab/novae](https://huggingface.co/datasets/MICS-Lab/novae) HuggingFace dataset:

```python
from huggingface_hub import hf_hub_download

# Example: download one slide
hf_hub_download(
    repo_id="MICS-Lab/novae",
    filename="human/brain/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.h5ad",
    repo_type="dataset",
    local_dir="datasets/mics-lab-novae",
)
```

The full corpus uses 15 human slides (13 Xenium, 1 MERSCOPE, 1 CosMx) across 15 tissues. See `scripts/01_extract_activations.py` for the complete slide list.

### 4. Run the pipeline

The scripts are numbered in execution order:

```bash
# Phase 1: Extract activations from all 12 Novae surfaces
python scripts/01_extract_activations.py

# Phase 2: Train TopK SAEs (one per surface)
python scripts/02_train_saes.py

# Phase 3: Characterize features
python scripts/03_characterize.py
python scripts/03b_modules.py
python scripts/03c_bio_grounding.py
python scripts/03d_assign_domains.py
python scripts/03e_merge_atlas.py

# Phase 4: Causal validation suite
python scripts/07_block1_sae_vs_prototypes.py    # H6
python scripts/08_block1_spatial_coherence_all.py # H7
python scripts/10b_graph_ablation_v2.py           # H8
python scripts/09b_effect_size_confounds.py       # H9
python scripts/11_prototype_domain_ablation.py    # Prototype ablation
python scripts/12_confound_suite.py               # Confound survival
python scripts/14_hierarchical_composition.py     # H10
python scripts/16_cross_tech_coherence.py          # Cross-tech

# Circuit tracing (requires per-cell conv activations)
python scripts/01b_extract_per_cell_conv.py
python scripts/19_causal_circuit_tracing.py
python scripts/20_circuit_analysis.py
python scripts/21_circuit_biology.py
python scripts/22_combinatorial_ablation.py
python scripts/23_spatial_steering.py
python scripts/24_pmi_vs_causal.py
python scripts/25_sae_dissociation.py

# Matched-null controls (outputs under atlas/novae-human-0/causal/reviewer_controls/)
python scripts/26_reviewer_controls.py        # A/B/C/D/E: bootstrap, gap-norm, panel null, steering, novelty
python scripts/27_random_direction_null.py    # F: matched-magnitude ablation null
python scripts/28_crosstech_rawexpr_baseline.py  # G: raw-expression cross-tech baseline
python scripts/29_feature_perm_graph_ablation.py # H: feature-permutation graph ablation
python scripts/34_shuffled_labels_specificity.py # M: tissue-term specificity vs shuffle
python scripts/36_triplet_sensitivity.py        # P: triplet-selection structure
python scripts/35_v2_figures.py                 # Figures 12, 13

# Phase 5: Build atlas data
python scripts/05_build_atlas_data.py
```

### 5. Build the paper

```bash
cd paper
pdflatex main.tex && pdflatex main.tex  # Two passes for references
```

## Hardware requirements

- **GPU**: Apple MPS (M2 Pro tested) or NVIDIA CUDA
- **RAM**: 32 GB minimum (SAE training + activation encoding)
- **Storage**: ~165 GB for activations + SAE weights + datasets
- **Compute time**: ~3 days for the full pipeline on M2 Pro

## Data sources

| Source | URL | Usage |
|--------|-----|-------|
| Novae checkpoints | `MICS-Lab/novae-{human,mouse,brain}-0` on HuggingFace | Model weights |
| Spatial transcriptomics slides | [`MICS-Lab/novae`](https://huggingface.co/datasets/MICS-Lab/novae) HuggingFace dataset | Activation corpus (95 slides, human + mouse) |
| Perturb-map CRISPR screen | [GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460) | Perturbation validation (§4.5/5) |
| Gene Ontology | [geneontology.org](http://geneontology.org) | Enrichment analysis |
| KEGG | [genome.jp/kegg](https://www.genome.jp/kegg/) | Enrichment analysis |
| Reactome | [reactome.org](https://reactome.org) | Enrichment analysis |
| PanglaoDB | [panglaodb.se](https://panglaodb.se) | Cell-type annotation |
| CellMarker | [biocc.hrbmu.edu.cn/CellMarker](http://biocc.hrbmu.edu.cn/CellMarker/) | Cell-type annotation |
| Enrichr | [maayanlab.cloud/Enrichr](https://maayanlab.cloud/Enrichr/) | Gene set enrichment |

## Citation

Citation details are withheld during anonymous review. A full BibTeX
entry will be added to the camera-ready version.

## Related work

- [Novae](https://www.nature.com/articles/s41592-025-02899-6) — the spatial foundation model we interpret (Blampey et al., Nature Methods 2025).
- Prior work on SAE atlases for single-cell foundation models (Geneformer, scGPT), circuit-tracing methodology for transformer residual streams, and exhaustive / combinatorial ablation are referenced in the paper; citations are withheld here during double-blind review.

## License

BSD-3-Clause.
