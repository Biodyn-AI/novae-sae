# Novae SAE: Sparse Autoencoder Atlas for the Novae Spatial Foundation Model

**Paper**: *Sparse autoencoder atlas of the Novae spatial foundation model reveals 49,152 interpretable biological concepts*

**Interactive atlas**: https://biodyn-ai.github.io/novae-atlas/

**Atlas source code**: https://github.com/Biodyn-AI/novae-atlas

---

## Overview

This repository contains the complete analysis codebase for applying sparse autoencoder (SAE) mechanistic interpretability to [Novae](https://www.nature.com/articles/s41592-025-02899-6), a GATv2 graph foundation model for spatial transcriptomics. We train TopK SAEs on every internal layer of the `novae-human-0` checkpoint, characterize 49,152 learned features with gene markers, cell-type databases, and spatial niche assignments, and apply a comprehensive causal-validation suite including graph ablation, prototype-domain reassignment, confound survival, cross-technology coherence, causal circuit tracing, and spatial gradient steering.

### Key findings

- **100%** of aggregator features carry significant biological enrichment (H1)
- **67%** of features depend on spatial context — they vanish without the neighborhood graph (H8)
- **0.95%** are technology-specific — the model generalizes across Xenium, MERSCOPE, and CosMx (H9)
- **439 causal circuit edges** with balanced excitation/inhibition (51%) and no effect attenuation (unlike transformers)
- Feature triplets show **synergy** (R=0.96), the opposite of transformer redundancy
- **426 novel gene-pair predictions** (70% not in GO)
- **81% vs 40%** annotation rate for real vs shuffled SAE (coherence is SAE-dependent)

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
│   └── 99_summary.py                  # Generate summary statistics
├── paper/
│   ├── main.tex                       # Paper source (20 pages, 50 references)
│   └── figures/                       # Generated figures (PDF)
├── RESEARCH_AND_IMPLEMENTATION_PLAN.md # Original research plan
├── CIRCUIT_TRACING_PLAN.md            # Circuit tracing gap analysis + plan
├── PHASE4_5_STATUS.md                 # Phase 4/5 completion status
└── configs/
    └── environment.yml                # Conda environment specification
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

```bibtex
@article{kendiukhov2026novae_sae,
  title={Sparse autoencoder atlas of the Novae spatial foundation model
         reveals 49,152 interpretable biological concepts},
  author={Kendiukhov, Ihor},
  year={2026},
  note={University of T\"ubingen}
}
```

## Related work

- [Novae](https://www.nature.com/articles/s41592-025-02899-6) — the spatial foundation model we interpret (Blampey et al., Nature Methods 2025)
- [SAE Atlas methodology](https://arxiv.org/abs/2603.02952) — the upstream SAE pipeline for Geneformer/scGPT (Kendiukhov, 2026)
- [Causal Circuit Tracing](https://arxiv.org/abs/2603.01752) — causal circuit methodology adapted for GATv2 (Kendiukhov, 2026)
- [Exhaustive Circuit Mapping](https://arxiv.org/abs/2603.11940) — exhaustive tracing + combinatorial ablation (Kendiukhov, 2026)

## License

BSD-3-Clause (matching the Novae model license).
