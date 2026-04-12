# Novae Mechanistic Interpretability — Research & Implementation Plan

> **Mission.** Replicate the SAE-based interpretability pipeline of `arXiv:2603.02952`
> ("Sparse autoencoders reveal organized biological knowledge but minimal regulatory
> logic in single-cell foundation models") on **Novae** (Nature Methods 2025,
> `s41592-025-02899-6`), adapting the methodology to a graph-based, spatial
> foundation model whose unit of representation is a *cell-in-its-spatial-context*
> rather than a gene token in a transformer.
>
> Author of upstream pipeline & this plan target: **Ihor Kendiukhov**.

---

## 0. Executive summary

Novae is a 10-layer GATv2 graph encoder (hidden 128, output 64, 8 heads) trained
with a SwAV-style self-supervised contrastive objective on ~30M cells across 18
tissues. It produces (a) per-cell representations in spatial context and (b)
512-prototype soft assignments that hierarchically cluster into nested spatial
domains. Three checkpoints exist on Hugging Face (`MICS-Lab/novae-human-0`,
`-mouse-0`, `-brain-0`), each ~32M parameters.

The original pipeline applied **TopK SAEs** (4× expansion, k=32) at every layer of
Geneformer (18 layers, d=1152) and scGPT (12 layers, d=512), characterized
features against GO/KEGG/Reactome/STRING/TRRUST, validated against CRISPRi
perturbations, ran PMI/Leiden module discovery, layer-wise U-shape profiling, SVD
vs SAE superposition, and shipped an interactive React/Plotly atlas via GitHub
Pages (`biodyn-ai.github.io`).

The replication is non-trivial because **the natural unit shifts from "gene token"
to "cell node"** and because **there is no CRISPRi screen to validate against in
spatial transcriptomics**. The plan below reframes every component of the pipeline
under this shift and identifies *which* parts transfer verbatim, which need
substitution, and which create new opportunities (e.g., Novae's built-in
prototype hierarchy, multi-organism checkpoints, and cross-technology zero-shot
behaviour).

---

## 1. Novae — what we are interpreting

### 1.1 Architecture (from the actually loaded `novae-human-0` checkpoint)

```
AnnData (cell × gene, with spatial coords)
        │
        ▼
[novae.spatial_neighbors]   k-NN spatial graph per slide; 2-hop neighborhoods
                             (n_hops_local = n_hops_view = 2)
        │
        ▼
CellEmbedder
   ├── embedding              # FROZEN gene table (60 697 × 512), scGPT-init
   │                          # 31 076 864 params (frozen)
   └── linear                 # (512 → 512) + bias, trainable
        │  (N_cells, 512)
        ▼
GraphEncoder.gnn              # PyG GATv2 stack of 10 convs, 16 heads each,
                              # edge_dim=1, ELU activation
   ├── conv 0                 # (512 → 128)   16 heads × 8 dim/head
   ├── convs 1..8             # (128 → 128)   16 heads × 8 dim/head
   └── conv 9                 # (128 → 64)    16 heads × ?, output_size=64
        │  (N_cells, 64)
        ▼
encoder.node_aggregation
        └── attention_aggregation   # gate_nn (64→1) + nn (64→64),
                                    # softmax-weighted pool per subgraph
        │  (B_subgraphs, 64)
        ▼
encoder.mlp_fusion            # 3-layer MLP, input dim 114 = 64 + 50,
                              # exists even on the non-multimodal checkpoint
                              # (114 → 114 → 64 → 64). Histo branch is zero-
                              # padded when H&E is absent. To verify at runtime.
        │  (B_subgraphs, 64)
        ▼
swav_head
   ├── projection             # L2-normalize → @ _prototypes^T
   └── _prototypes            # (512, 64) trainable
        │  (B_subgraphs, 512) prototype logits
        ▼
   Sinkhorn-Knopp → swapped CE loss; hierarchical agglomerative cosine
   clustering on prototypes → multi-level spatial domain assignment.
```

Training objective: SwAV adapted to graphs. Two augmented views of the same
spatial subgraph are produced via `GraphAugmentation` (panel_subset_size=0.8,
background_noise_lambda=5, gene dropout, sensitivity noise), passed through
encoder + aggregator + SwavHead, and the loss is `-0.5 * (CE(q1, p2) + CE(q2,
p1))` where `q` are Sinkhorn-Knopp soft codes and `p` are softmaxed prototype
logits. A slide-specific top-k prototype queue with `min_prototypes_ratio = 0.75`
dynamically restricts the active prototype subset, which is what makes the
model robust to batch effects and gene-panel mismatch across technologies.

**Parameter accounting (verified on `novae-human-0`):**

| Module                 | Params       | Trainable? |
|------------------------|--------------|------------|
| `cell_embedder.embedding` (60 697 × 512) | 31 076 864 | **frozen** |
| `cell_embedder.linear`                   |    262 656 | yes        |
| `encoder.gnn.convs.0`  (512 → 128, 16 h) |    131 712 | yes        |
| `encoder.gnn.convs.1..8` × 8 (128→128)   |    264 192 | yes        |
| `encoder.gnn.convs.9`  (128 → 64, 16 h)  |    266 304 | yes        |
| `encoder.node_aggregation.attention_aggregation` | 4 225 | yes |
| `encoder.mlp_fusion`                     |     24 630 | yes        |
| `swav_head._prototypes` (512 × 64)       |     32 768 | yes        |
| **TOTAL**                                | **32 066 423** | **989 559 trainable** |

**Implication for interpretability**: only ~3 % of the model is learned end-
to-end on spatial data. The other 97 % is the frozen scGPT gene-embedding
table that the upstream paper already analysed in detail. SAE training should
therefore concentrate on the GAT layers, the node aggregator, the fusion MLP,
and the prototype head — *not* on the cell-embedder embedding output, which is
already covered by the upstream scGPT atlas.

### 1.2 Activation surfaces (where SAEs can attach)

Unlike a transformer, Novae has no canonical "residual stream". The relevant
hookable tensors, indexed against the loaded `novae-human-0` state dict, are:

| # | Tensor                                | Shape (per call) | Hook point                                       |
|---|---------------------------------------|------------------|--------------------------------------------------|
| 0 | Cell-embedder linear output           | (N, 512)         | `cell_embedder.linear`                           |
| 1.0 | GAT conv 0 output                   | (N, 128)         | `encoder.gnn.convs[0]`                           |
| 1.1..1.8 | GAT conv i output (i=1..8)     | (N, 128)         | `encoder.gnn.convs[i]`                           |
| 1.9 | GAT conv 9 output (compressed)      | (N, 64)          | `encoder.gnn.convs[9]`                           |
| 2 | Final GAT output                      | (N, 64)          | `encoder.gnn` output                             |
| 3 | Aggregated subgraph representation    | (B, 64)          | `encoder.node_aggregation.attention_aggregation` |
| 4 | Fusion MLP output                     | (B, 64)          | `encoder.mlp_fusion`                             |
| 5 | Pre-prototype L2-normalized rep       | (B, 64)          | inside `swav_head.projection`                    |
| 6 | Prototype logits                      | (B, 512)         | `swav_head` output                               |
| 7 | Sinkhorn-Knopp soft codes (training)  | (B, 512)         | `swav_head.sinkhorn_knopp` output                |

The cell-embedder *embedding* layer (60 697 × 512) is **deliberately not on this
list**: it is the frozen scGPT gene table and is exhaustively covered by the
existing upstream scGPT atlas. Hooking it would just rediscover those features.

PyG's `GAT` does not expose per-layer outputs by default; the hooks must be
registered directly on each `encoder.gnn.convs[i]` module. This preserves
weight identity with the released checkpoint (no re-instantiation needed).

**Confirmed at runtime (Phase 0 smoke test on `novae-human-0` with DLPFC slide):**

- `compute_representations(zero_shot=True)` runs `cell_embedder → encoder.gnn → encoder.node_aggregation` and **returns the aggregator output directly** as `obsm["novae_latent"]`. Verified: `np.abs(stacked_aggregator - novae_latent).max() == 0.0`.
- `encoder.mlp_fusion` is **dead code** in non-multimodal mode (`forward` only runs it `if hasattr(data, "histo_embeddings")`). For `novae-human-0`, drop surface (4) — it never fires.
- `swav_head` is **only invoked by `assign_domains()`**, not by `compute_representations`. To collect prototype-space activations, hook during `assign_domains` (which adds `obs.novae_sid`, `obs.novae_leaves`, `obs.novae_domains_<level>`).
- `cell_embedder` returns the modified `data` object — capture per-cell embeddings via `out.x` from a hook on the parent `cell_embedder` module, **not** on `cell_embedder.linear` (which fires on the gene-table, not per-cell).
- Per-batch GAT activations have shape `(N_nodes_in_batch, dim)` where `N_nodes ≈ B_centers × 19` (2-hop neighborhood expansion). For `DLPFC_151675` (3592 cells): 8 batches × ~450 centers/batch × ~19 nodes/subgraph ≈ 9000 node-rows per batch. **Not all rows correspond to center cells** — the tail rows are neighbor cells that appear because some center cell's subgraph included them.
- Inference throughput on Apple M2 Pro MPS: ~4 s for a 3592-cell slide → ~1000 cells/sec. A 1M-cell slide takes ~17 minutes; the full 27M-cell human corpus takes ~7.5 wall-clock hours.

**Two SAE training modes** consistent with the above:

1. **Per-cell SAEs** (recommended primary): hook the **aggregator output** `(B_centers, 64)`, accumulate one row per cell across the corpus → `(N_total_cells, 64)`. This is the canonical "cell-in-niche" representation. Train **one SAE** here. d=64.
2. **Per-node SAEs** (for layer-wise / U-shape analysis): hook each `gnn.convs[i]`, accumulate the full `(N_nodes, dim)` tensor. This includes neighbors *as observed in someone else's subgraph*; features are "what this node looks like to a graph that contains it". Train **one SAE per GAT layer** (×10).
3. **Center-only per-layer SAEs** (cleanest, conditional on PyG batch convention): if PyG places centers first in each batch, slice the per-layer tensor `[: B_centers]` to get per-center activations. Otherwise use `data.batch` to mask. **To verify in Phase 1**.

**SAE sizing for these surfaces:**

| Surface | d   | F (32×) | k  | Notes                                          |
|---------|-----|---------|----|------------------------------------------------|
| 0 (cell embed `data.x`) | 512 | 4096 | 32 | Pre-graph; cell-as-bag-of-genes in scGPT space |
| 1.0 (conv 0)   | 128 | 4096   | 32 | First spatial integration layer                |
| 1.1..1.8       | 128 | 4096 each | 32 | Mid GAT stack                                |
| 1.9 (conv 9)   | 64  | 2048   | 16 | Compression to output_size                     |
| 3 (aggregator) | 64  | 2048   | 16 | **Primary target** — equals `novae_latent`     |
| 6 (prototypes) | 512 | 4096   | 32 | Prototype logits (only via `assign_domains`)   |

Total for `novae-human-0`: **13 SAEs** (1 cell-embed + 10 GAT layers + 1 aggregator + 1 prototype). Of these, the aggregator SAE is the *headline* artefact; the others are diagnostic.

### 1.3 What kinds of "concepts" Novae could plausibly encode

Because Novae sees a **cell + its k-nearest spatial neighbours**, plausible
features that an SAE could surface include:

- Cell-type identity (T cell, hepatocyte, GABAergic neuron, …).
- Niche identity (germinal center light zone, hepatic zonation layer, cortical
  layer 4, tumor margin, fibrotic shell, …).
- Cell-type *interaction* signatures (T-cell next to dendritic cell, tumor cell
  next to CAF, …).
- Anatomical region (Allen Brain reference areas for `novae-brain-0`).
- Tissue / organ identifiers (multi-tissue checkpoints).
- Technology / batch artefacts (Xenium-vs-MERFISH-vs-CosMx panels): a *failure
  mode* worth detecting.
- Gene-program features (highly variable gene modules), inherited from the
  scGPT-initialized embedding.
- Spatial gradients (distance to vessel, distance to lumen) — speculative.

These hypotheses determine which ontology databases and validation datasets are
appropriate (§5).

---

## 2. The upstream pipeline — what we are replicating

Source: **Kendiukhov, *Sparse autoencoders reveal organized biological knowledge
but minimal regulatory logic in single-cell foundation models*, arXiv:2603.02952.**

The pipeline has the following stages and parameter choices:

| Stage | Choice in upstream paper | Rationale |
|-------|--------------------------|-----------|
| Activation extraction | PyTorch hooks at the **output of every transformer layer (post-residual)**, every gene position, stored as memory-mapped float32 numpy | Captures the residual stream, which is the canonical interpretability locus |
| Activation corpus (Geneformer) | 2 000 K562 control cells from Replogle CRISPRi → 4.06 M positions/layer (336 GB total) | Matches the perturbation validation set |
| Activation corpus (scGPT)     | 3 000 Tabula Sapiens cells (1 k immune across 43 cell types, 1 k kidney, 1 k lung) → 3.56 M positions/layer | Maximizes cell-type diversity at fixed budget |
| SAE family | **TopK SAE**, 4× overcomplete dictionary, k=32 | Cleanest sparsity guarantees and avoids dead-feature collapse |
| SAE size | Geneformer 1152→4608, scGPT 512→2048 | 4× expansion factor at every layer |
| SAE training | Adam, lr 3e-4, batch 4096, 5 epochs, MSE recon loss, decoder columns unit-normalized after each step; Geneformer subsamples to 1 M positions/layer for tractability | Standard TopK setup |
| Feature characterization | Top 20 genes by mean activation magnitude per feature; one-sided Fisher's exact vs **GO BP, KEGG, Reactome, STRING, TRRUST**; BH FDR α=0.05 | Multi-database to triangulate |
| Layer profiling | Per-layer annotation rate; identifies **U-shape**: 57–59 % at L0–L1, min 45.4 % at L8, recovery 55–56 % at L10–L11 | Reveals "molecular → abstract → re-specialization" hierarchy |
| Module discovery | PMI between feature pairs across positions, permutation null at p<0.001, Leiden at resolution 1.0; reports 141 modules in Geneformer (6–12/layer), 76 in scGPT | Functional grouping of features |
| Cross-layer highways | PMI between SAE features at source vs target layer; PMI > 3 = highway | Tracks how concepts are routed through depth |
| Cell-type enrichment | Mean activation per cell type, Fisher exact + BH | Biological grounding of features |
| Superposition quantification | Per layer: cosine similarity of every SAE decoder column vs top-50 SVD axes; threshold 0.7. **99.8 %** of features are non-aligned. SAE explains 77–85 % activation variance vs SVD 31–38 % | Establishes that SAE captures structure invisible to linear methods |
| Causal validation | Forward-hook ablation at L11 of one SAE feature at a time across 200 cells/feature (50 features tested); measure logit disruption | Causal specificity baseline |
| Perturbation validation | 100 CRISPRi targets (48 TRRUST TFs + 52 others), 20 cells/target vs 100 k control positions, Wilcoxon + BH; for TFs check Fisher enrichment of responding-feature top-genes for known targets | Tests regulatory specificity. **Result: 6.2 %** of TFs show target-specific response |
| Online atlases | React 18 + TypeScript + Vite 6 + Tailwind + Plotly.js, static JSON via GitHub Pages, hosted at `biodyn-ai.github.io/{geneformer,scgpt}-atlas/` | Layer Explorer, Feature Detail, Module Explorer, Cross-Layer Flow, Gene Search, Ontology Search |

The **central upstream finding** to remember while replicating is: SAE features
on Geneformer/scGPT *encode rich organized biological knowledge* (29–59 %
annotate), but *fail to encode causal regulatory logic* (only 6.2 % of TFs show
target-specific response). The Novae replication needs to ask the analogous
question for *spatial* biology: "do the features encode true niche/region
knowledge, or are they merely cell-type co-occurrence statistics?"

---

## 3. The adaptation problem — what changes for Novae

### 3.1 Unit of representation: gene-token → cell-in-context

In Geneformer/scGPT a "position" is a gene token; SAE features are then
*gene-program* features. In Novae a "position" is a **cell node** whose
representation already integrates its spatial neighborhood. SAE features will
therefore be **cell-state-in-niche features**, more analogous to ImageNet
"concept neurons" than to Anthropic-style language features.

Implication: dictionary size scales with the number of *meaningful cell-niche
states* (probably hundreds to low thousands), not with the number of genes. We
should not blindly inherit "4× expansion" as the right ratio.

### 3.2 No residual stream → multi-surface SAE

The cleanest interpretability target is the per-GAT-layer node hidden state
(item #1 in §1.2). Layers are only 128-dimensional, so 4× expansion gives
512-feature dictionaries — too small to expose superposition meaningfully. The
pipeline should:

1. Train one SAE per GAT layer at the per-node hidden state.
2. Train one SAE at the L2-normalized aggregated representation (item #4),
   which is the "cell-in-niche embedding" everyone downstream actually uses.
3. **Increase expansion factor to 16× or 32×** at the small layers to give
   superposition room to be visible (so e.g. 128 → 4096, 64 → 2048).
4. Optionally train an SAE on the *concatenated* representation across all 10
   layers to look for cross-layer features.

### 3.3 Activation corpus

K562 / Replogle does not exist for Novae. We need a heterogeneous spatial corpus
that exercises Novae's full domain coverage. Concrete options (in order of
preference):

1. **Reuse Novae's own training corpus** via `huggingface.co/datasets/MICS-Lab/novae`
   (95 files, mixed Xenium / MERFISH / CosMx / Visium / spot-resolution data).
   This guarantees in-distribution activations and full tissue diversity.
2. **Held-out spatial atlases**: 10x Genomics Xenium demos (breast, lung, liver,
   brain), Allen Brain MERFISH atlas (mouse brain), HuBMAP CODEX/Xenium tissue
   atlas, NanoString CosMx FFPE atlas. Use these for the human checkpoint.
3. **Brain-specific**: Allen Brain Cell Atlas MERFISH + Vizgen MERFISH mouse
   brain atlas for `novae-brain-0`. Cross-reference with Allen anatomical
   ontology — this is the analogue of "TRRUST for transcription".
4. **Mouse-specific**: Stereo-seq mouse embryo for `novae-mouse-0`.

Concrete budget: aim for ~1–3 M cell-node positions per checkpoint, comparable
to upstream's 3.5–4 M positions per layer. This is **per-cell**, so storage is
much smaller than upstream (~64 floats × 3 M cells × 10 layers ≈ 7.5 GB, vs
upstream's 336 GB).

### 3.4 Validation databases — replacing GO/KEGG/Reactome/STRING/TRRUST

The biology of spatial niches lives in different databases. Concrete substitutions:

| Upstream DB | Spatial-equivalent DB | Use |
|-------------|----------------------|-----|
| GO BP (gene-set enrichment) | **GO BP** (still applies via top-gene readout per feature) | Fallback gene-program annotation |
| KEGG / Reactome | **KEGG / Reactome** (still applies) | Fallback pathway annotation |
| STRING (PPI)   | **STRING** (still applies) | Fallback PPI annotation |
| TRRUST (TF→target) | **CellChatDB / OmniPath / LIANA / NicheNet** (ligand–receptor pairs) | The spatial analogue of regulatory logic — does feature X correspond to a known L–R pair active in a niche? |
| — | **PanglaoDB / CellMarker / CellOntology (CL)** | Cell-type identity feature characterization |
| — | **UBERON (anatomy ontology)** | Tissue/region identity |
| — | **Allen Brain Atlas reference regions** | For `novae-brain-0` only — gold-standard region labels |
| — | **TabulaSapiens / HCA marker gene panels** | Cross-validation of cell-type markers |
| — | **Nichenet / SpatialDM ground-truth niches** | Niche-level validation |

Each feature's "top-20 genes" interpretation (upstream protocol) is replaced by
**three interpretation modes**:

1. **Top-gene mode**: project the SAE decoder column back into gene space via
   `CellEmbedder`'s pseudo-inverse and take the top genes (preserves the
   upstream protocol exactly, enables KEGG/Reactome/PanglaoDB tests).
2. **Top-cell mode**: rank cells by feature activation, look at which cell types
   / anatomical labels / tissue IDs are over-represented (Fisher exact + BH).
3. **Top-neighborhood mode**: rank *graphs* (k-hop subgraphs) by mean feature
   activation; look at which neighborhood compositions are over-represented
   (e.g. "CD8 T cell adjacent to tumor and macrophage").

Mode (1) preserves upstream comparability; modes (2) and (3) are net-new and
exploit spatial structure.

### 3.5 The perturbation-validation problem

There is no Replogle-equivalent perturbation screen for spatial transcriptomics.
We must construct a **causal-validation suite** out of available manipulations:

1. **Graph ablation**: remove all neighbors of a cell (k=0) and re-run; features
   that depend on niche context should drop, features that depend on intrinsic
   cell state should not. This separates *intrinsic* from *contextual* features.
2. **Neighborhood swap**: replace a cell's neighborhood with a different
   tissue's neighborhood; well-localized niche features should disappear.
3. **Cell-type knockout**: remove all cells of a given type from a slide before
   building the graph; observe which features collapse.
4. **Cross-technology consistency**: re-encode the same biological tissue with
   Xenium vs MERFISH vs CosMx (subset of Novae HF dataset has paired panels);
   *true* biological features should fire on the same tissues; technology-
   specific features expose batch artefacts.
5. **Cross-organism transfer**: apply the human-trained SAE to mouse data via
   ortholog gene mapping (Novae's `CellEmbedder` already handles unknown genes
   via PCA-init nearest-neighbor lookup); features that survive are
   evolutionarily-conserved cell-niche concepts.
6. **Perturb-seq spatial datasets**: where they exist (Perturb-FISH, MERFISH-
   based screens, the Spatial Genomics datasets in HF), use them as the proper
   analogue of upstream's CRISPRi protocol.
7. **Slide-level perturbations**: tumor-vs-adjacent-normal pairs (10x Xenium
   breast/lung tumor demos); features that change reliably between paired
   slides correspond to disease-state niches.

This suite is the spatial replacement for upstream §perturbation testing and is
where the scientific novelty of the replication lives.

### 3.6 Novae-specific opportunities (not in upstream)

Things that exist *only* because Novae is what it is:

1. **Compare SAE features against Novae's own prototype hierarchy.** SwavHead
   produces 512 prototypes that hierarchically cluster into spatial domains.
   Are SAE features (a) just rediscovering prototypes, (b) finer than
   prototypes, (c) orthogonal? Compute cosine similarity matrix of SAE decoder
   columns vs prototype vectors; expect a small overlap and large complement.
2. **GATv2 attention analysis.** Each GAT layer has explicit per-edge attention
   weights. We can do *attention pattern analysis* exactly like the
   transformer-style "which neighbour attends to which" — but here it is
   "which neighbour cell influences which target cell". Compare to LIANA / Cell-
   Chat ligand-receptor ground truth as the spatial analogue of TRRUST analysis.
3. **Multi-checkpoint cross-model analysis.** Three released models (`-human-0`,
   `-mouse-0`, `-brain-0`) trained on overlapping but not identical data.
   Train SAEs on all three and compute feature-level agreement (CCA / mutual
   nearest-neighbor on top-cell-type signatures). This is the analogue of the
   scGPT↔Geneformer convergence finding in upstream §VII.
4. **Hierarchy-conditioned SAEs.** Train one SAE conditioned on each level of
   Novae's domain hierarchy (coarse → fine) and ask whether features at finer
   levels are *compositions* of coarser features.
5. **Spatial coherence metric.** For every feature, compute Moran's I of its
   activation across the spatial coordinates of a slide. Features that are
   spatially coherent (high Moran's I) are *bona fide* niche features; spatially
   random features are likely cell-intrinsic identity features. This metric has
   no analogue in the gene-token setting and is a free novel axis.

---

## 4. Implementation plan

### 4.1 Phase 0 — Infrastructure & sanity (Week 0–1)

- Create a clean Python 3.11 conda env with `novae`, `torch`, `torch-geometric`,
  `scanpy`, `anndata`, `huggingface-hub`, `safetensors`, `numpy`, `scipy`,
  `scikit-learn`, `leidenalg`, `umap-learn`, `pyarrow`, `gseapy`, `decoupler`.
  Pin versions in `pyproject.toml`.
- Pull the three Novae checkpoints from Hugging Face into a local cache:
  `MICS-Lab/novae-human-0`, `-mouse-0`, `-brain-0`.
- Pull a small subset of `MICS-Lab/novae` HF dataset (say 5 slides covering the
  major technologies) and confirm `novae.spatial_neighbors` + `compute_repre-
  sentations` + `assign_domains` round-trip without error on each checkpoint.
- Reproduce the published novae-human-0 zero-shot domain inference numbers on
  one tutorial dataset to confirm we are loading weights correctly.
- Snapshot the `novae` source tree at a specific commit so the hook surface is
  reproducible.

Deliverable: a notebook `00_infra_sanity.ipynb` that loads each checkpoint,
encodes one slide, and prints the shapes of all activation surfaces in §1.2.

### 4.2 Phase 1 — Activation extraction (Week 1–3)

- Implement `novae_hooks.py`: forward hooks on every `GraphEncoder.gnn.convs[i]`
  output (10 hooks, one per GAT layer), plus hooks on `CellEmbedder` output,
  `AttentionAggregation` output, `SwavHead.projection` pre- and post-prototype.
- Run extraction over a curated activation corpus:
  - **human checkpoint**: ~3 M cell-nodes drawn from 30–60 slides spanning
    breast, lung, liver, kidney, colon (Xenium + MERFISH + CosMx).
  - **brain checkpoint**: ~3 M cell-nodes from Allen Brain MERFISH + Vizgen
    MERFISH mouse brain (cortex, striatum, hippocampus, cerebellum).
  - **mouse checkpoint**: ~3 M cell-nodes from Stereo-seq mouse embryo + Vizgen
    mouse brain.
- Storage: per-checkpoint, per-layer memory-mapped float32 numpy arrays under
  `/Volumes/Crucial X6/.../novae/activations/{checkpoint}/{layer}.npy`.
  Total storage budget: ~10 GB per checkpoint (much smaller than upstream's
  336 GB because Novae layers are 64–128-dim, not 1152-dim).
- Save a parallel `metadata.parquet` with per-cell (cell_id, slide_id,
  technology, organ, panel, cell_type_label_if_known, x, y, original_index)
  to enable downstream cell-type and spatial enrichment.

Deliverable: `01_extract_activations.py` and the activation arrays.

### 4.3 Phase 2 — SAE training (Week 3–5)

- Implement `topk_sae.py` mirroring upstream:
  - encoder: `Linear(d, F) → subtract pre_bias → TopK(k)`.
  - decoder: `Linear(F, d, bias=True)`, columns L2-normalized after each step.
  - loss: MSE recon, optional auxiliary "dead-feature revival" loss as in
    OpenAI's TopK reference.
- Hyperparameters (tentative, to be tuned):
  - Layer SAEs (per-node hidden, d=128): F = 4096 (32×), k = 32.
  - Aggregated SAE (cell-in-niche, d=64): F = 2048 (32×), k = 16.
  - Optimizer: Adam, lr 3e-4, batch 4096, 5 epochs, decoder column renorm.
- Train one SAE per (checkpoint, layer) — that is **3 checkpoints × ~12 surfaces
  ≈ 36 SAEs**, not the 30 in upstream. Should fit on a single A100 in <1 day
  total because the dim is small.
- Track standard SAE diagnostics: variance explained, fraction of features
  alive, mean L0, k-active distribution, dead-feature count over training,
  reconstruction MSE.

Deliverable: `02_train_saes.py`, trained SAE weights under
`novae/saes/{checkpoint}/{surface}.pt`, and a `sae_quality.csv`.

### 4.4 Phase 3 — Feature characterization (Week 5–8)

For each (checkpoint, surface) SAE:

1. **Top-gene mode (upstream-compatible)**: pseudo-invert through `CellEmbedder`
   to obtain a per-gene loading per feature; take top-20 genes; run Fisher's
   exact one-sided enrichment vs GO BP, KEGG, Reactome, STRING, TRRUST,
   PanglaoDB, CellMarker, CellOntology marker panels, NicheNet, OmniPath L-R.
   BH-FDR α = 0.05. Same code path as upstream, just with extra databases.
2. **Top-cell mode**: take the top 1 % of cells by feature activation; Fisher
   exact for over-representation by `cell_type_label`, `tissue`, `technology`,
   `organism`, `slide_id`; flag slide- or technology-confounded features.
3. **Top-neighborhood mode**: for each top cell, build the k=10 neighborhood
   composition vector; cluster these vectors; assign each feature its dominant
   neighborhood archetype; cross-reference with known niche definitions.
4. **Spatial coherence**: compute Moran's I of feature activation per slide,
   averaged across slides; report per-feature spatial-vs-intrinsic score.
5. **Layer profiling**: aggregate annotation rates per layer; look for the
   U-shape (or its absence — could be a real finding either way). Also profile
   spatial coherence per layer.
6. **Hierarchy comparison**: cosine similarity of every decoder column against
   Novae's 512 prototypes; report alignment matrix.
7. **Module discovery**: PMI between feature pairs across cell positions,
   permutation null at p<0.001, Leiden clustering at resolution 1.0; report
   module count per layer/checkpoint.
8. **Cross-layer information highways**: PMI between SAE features at source vs
   target layer; threshold PMI > 3.
9. **Superposition**: top-50 SVD axes per layer, cosine 0.7 threshold, fraction
   of non-aligned features; reproduce the upstream 99.8 % statistic in the
   spatial setting.

Deliverable: `03_characterize.py`, `feature_atlas.parquet` with one row per
feature × surface × checkpoint and ~50 columns of annotations, plus
`module_atlas.parquet` and `superposition_summary.parquet`.

### 4.5 Phase 4 — Causal validation (Week 8–11)

This is the spatial replacement for upstream's CRISPRi validation. Implement:

1. **Feature ablation**: forward-hook ablation that zeros one SAE feature in the
   reconstructed activation, projects back, runs Novae's prototype head, and
   measures the change in (i) prototype assignments, (ii) downstream domain
   labels at every hierarchy level. Per upstream protocol, sample 200 cells per
   feature, test 50 highest-confidence features per checkpoint.
2. **Graph ablation suite**: for each SAE feature, measure activation in three
   regimes: (a) full graph, (b) k=0 (cell alone), (c) random-graph rewire. Report
   the feature's *contextual dependency score* (a−b)/a. This is novel.
3. **Cross-technology coherence**: pick 10 paired tissues with both Xenium and
   MERFISH coverage; compute per-feature activation profiles; correlate. The
   distribution of correlations is the *technology-invariance audit*.
4. **Cross-organism transfer**: encode human cells with `novae-mouse-0` and vice
   versa via the ortholog-mapping path; train an SAE on the cross-encoded data;
   measure feature overlap.
5. **Perturbation datasets** (if available): replicate upstream protocol on any
   spatial perturbation screen we can find. Report fraction of perturbed
   targets with target-specific SAE feature responses (the spatial analogue of
   "6.2 % of TFs").
6. **Cross-checkpoint feature alignment**: linear-CCA of SAE feature top-cell-
   type signatures between `-human-0` and `-mouse-0` (orthologous tissues
   only) and between `-human-0` and `-brain-0`. This is the spatial analogue
   of upstream's scGPT-Geneformer convergence finding.

Deliverable: `04_causal.py`, `causal_validation.parquet`, and a one-page
summary table of validation rates that mirrors the upstream "6.2 %" headline
statistic.

### 4.6 Phase 5 — Online SAE atlas (Week 11–14)

Replicate upstream's `biodyn-ai.github.io/{geneformer,scgpt}-atlas/` stack:

- Frontend: React 18 + TypeScript + Vite 6 + Tailwind CSS + Plotly.js, identical
  template to existing atlases for visual continuity.
- Backend: static JSON files generated from `feature_atlas.parquet`,
  `module_atlas.parquet`, etc., served via GitHub Pages from a new repository
  `biodyn-ai/novae-atlas` (or `MICS-Lab/novae-atlas` if collaboration is
  desired).
- Pages to ship (mirroring upstream where possible, plus spatial-only pages):
  - **Layer Explorer** — annotation rate, module count, U-shape plot per layer.
  - **Feature Detail** — per-feature top genes, top cells, top niches, spatial
    coherence (Moran's I), prototype alignment, ablation effect, module
    membership, layer position, cross-layer highways.
  - **Module Explorer** — Leiden modules and their gene/cell/niche signatures.
  - **Cross-Layer Flow** — Sankey of cross-layer feature highways.
  - **Gene / Ontology / Cell-type / Niche Search** — query-driven entry into
    the feature space.
  - **(NEW) Slide Viewer** — 2D scatter of cells colored by feature activation,
    overlayed on slide coordinates. This is uniquely possible for Novae and is
    the visually compelling thing for spatial.
  - **(NEW) Cross-checkpoint Compare** — side-by-side feature alignment between
    `-human-0`, `-mouse-0`, `-brain-0`.
  - **(NEW) Causal Audit** — per-feature ablation effects, contextual
    dependency score, technology invariance score.
- Three deployable atlases: `novae-human-atlas`, `novae-mouse-atlas`,
  `novae-brain-atlas`, plus a unified `novae-cross-atlas` for the comparative
  analysis.

Deliverable: live atlas URLs and a frozen DOI snapshot via Zenodo.

### 4.7 Cross-cutting — confound controls (continuous)

Per the user's manifesto §IV the central methodological challenge is
distinguishing real model-learned structure from confounds. The following
controls are **mandatory** at every stage and follow the user's prior practice:

- **Slide / batch shuffles**: re-run characterization with shuffled slide labels;
  features whose enrichments survive are slide-confounded artefacts.
- **Cell-type residualization**: regress out cell-type identity from feature
  activations and re-run niche enrichment; features whose niche signal vanishes
  are *just* cell-type indicators.
- **Technology residualization**: regress out technology label and re-run.
- **Degree-preserving graph rewire**: re-build spatial graphs with degree-
  preserving rewire and re-encode; features that survive are graph-structure-
  dependent vs degree-trivial.
- **Permutation null** for every reported PMI / Fisher / module statistic.
- **Hierarchical null model framework** at the strict end (label permutation →
  rewire → cell-type matching → max-null), reporting survival rate analogous
  to the manifesto's "10 %" calibration. Expect <50 % of raw positives to
  survive; report both raw and corrected.

---

## 5. Hypotheses worth pre-registering

Pre-registering predictions before the data come in is the cleanest way to make
this a real test rather than a fishing expedition.

1. **H1 (annotation rate)**: ≥30 % of SAE features at the cell-in-niche surface
   will significantly enrich for at least one cell-type or niche label. Mirrors
   upstream's 29–59 % range.
2. **H2 (causal poverty)**: <20 % of SAE features will exhibit "true" niche
   specificity under the strict confound suite (cell-type residualized,
   technology residualized, degree-preserving rewire). Mirrors upstream's
   "minimal regulatory logic" finding.
3. **H3 (superposition)**: ≥95 % of SAE features will be non-aligned with the
   top-50 SVD axes of the corresponding surface, reproducing the 99.8 %
   superposition statistic in the spatial setting.
4. **H4 (depth-dependent processing)**: Annotation rate will be U-shaped or
   monotonically *non-increasing* across the 10 GAT layers, with niche-level
   features peaking in the middle layers and cell-type-level features peaking
   in early/late layers.
5. **H5 (cross-checkpoint convergence)**: SAE features from `-human-0` and
   `-mouse-0` will align significantly above chance on orthologous tissues
   (CCA r > 0.5 on the top-50 features), but `-brain-0` will diverge from both
   on non-brain tissue, indicating tissue-specialization.
6. **H6 (prototype redundancy)**: <30 % of SAE features will be cosine-aligned
   (>0.7) with any of the 512 SwAV prototypes, meaning the SAE recovers
   structure that the model's own clustering head does not expose.
7. **H7 (spatial coherence)**: Mean Moran's I of SAE features will be
   significantly above zero, and will increase with layer depth (later layers
   integrate more spatial context).
8. **H8 (contextual dependency)**: A non-trivial fraction (>20 %) of SAE
   features will show >50 % activation drop when neighborhoods are zeroed out,
   confirming Novae actually uses spatial context (vs being a glorified
   cell-type classifier).
9. **H9 (technology artefacts)**: ≥5 % of SAE features will be technology-
   specific (single-technology activation only), exposing batch artefacts that
   the SwAV training did not fully iron out.
10. **H10 (hierarchical composition)**: SAE features from later GAT layers will
    decompose linearly into combinations of earlier-layer features, mirroring
    information-highway findings in upstream.

Each hypothesis has a defined statistic, threshold, and confound control. The
*surprising* findings will come from where these hypotheses fail.

---

## 6. Risks, decisions, and open questions

### 6.1 Decisions that need resolution before coding starts

1. **SAE expansion factor**: 16× or 32× at d=128. Higher gives more superposition
   visibility but slows training and worsens dead-feature rates. Recommendation:
   **32×** for layer SAEs, **32×** for aggregated SAE, monitor dead-feature
   fraction and step down if >20 % die.
2. **Activation corpus composition**: how to balance tissues / technologies
   across the ~3 M cell-node budget. Recommendation: stratified sampling so each
   technology contributes ≥15 % and each major organ ≥10 %.
3. **Whether to train on per-node hidden states or per-cell aggregated states
   first**. Recommendation: **per-node first** because it preserves the
   layer-by-layer pipeline structure of upstream and enables U-shape analysis.
4. **Whether to include the multimodal histo-fusion path** in the first pass.
   Recommendation: **no**, ship single-modality SAE atlas first; histo fusion
   is a follow-on paper.
5. **License of the released atlas code and data**. Recommendation: BSD-3 to
   match upstream Novae license, plus CC-BY-4.0 for the JSON data dumps.

### 6.2 Risks

- **Gene-space pseudo-inverse may be ill-conditioned** because `CellEmbedder`
  uses an identity-init linear and the gene vocabulary is large (~30 k). Top-
  gene mode may be noisy. Mitigation: use the actual `CellEmbedder` weights
  directly to compute decoder→gene-space projections, or fall back to top-cell
  mode as the primary interpretation.
- **Small layer dimension (64–128) may not exhibit superposition at all**, in
  which case H3 fails and we should report it as a *negative* finding — itself
  scientifically valuable.
- **Zero-shot data variability** means slide-level batch effects may dominate
  the SAE features. Mitigation: rigorous slide-shuffle confound controls (§4.7)
  and refusal to ship features that fail them.
- **No CRISPRi-equivalent ground truth** means causal validation is the weakest
  link. Mitigation: triangulate via the six-pronged suite in §4.5 rather than
  relying on a single perturbation source.
- **Three checkpoints multiply the work by 3×**. Mitigation: prototype
  everything on `novae-human-0` first, then re-run scripts on the other two
  checkpoints once stable.
- **Compute budget**: should fit comfortably on a single A100 day for SAE
  training and a few GPU-days for activation extraction; the constraint will
  be storage and human time on characterization.

### 6.3 Open scientific questions

- **Does a graph foundation model exhibit the same "abstract middle, biological
  edges" U-shape that transformers do?** Plausible but not obvious — graph
  message-passing has different inductive biases.
- **Is Novae's prototype hierarchy a *bottleneck* on what the model can
  represent**, or do SAE features find structure orthogonal to the prototypes?
  This goes to the heart of "is built-in clustering enough or do we need
  post-hoc decomposition?".
- **Cross-organism conservation of niche features** is the spatial analogue of
  the manifesto's cross-model convergence finding. If `-human-0` and `-mouse-0`
  agree on niche features for orthologous tissues, that's strong evidence that
  the model learned biology rather than dataset artefacts.
- **Does the SwAV objective produce more or less interpretable features than
  the masked-language objective?** This is a *novel* methodological question
  that the comparison with upstream's Geneformer/scGPT atlases enables.

---

## 7. Concrete deliverables

| # | Deliverable | Path |
|---|-------------|------|
| 1 | This plan | `novae/RESEARCH_AND_IMPLEMENTATION_PLAN.md` |
| 2 | Infra sanity notebook | `novae/notebooks/00_infra_sanity.ipynb` |
| 3 | Activation extractor | `novae/scripts/01_extract_activations.py` |
| 4 | Activation arrays | `novae/activations/{checkpoint}/{layer}.npy` |
| 5 | TopK SAE module | `novae/src/topk_sae.py` |
| 6 | SAE training script | `novae/scripts/02_train_saes.py` |
| 7 | SAE checkpoints | `novae/saes/{checkpoint}/{surface}.pt` |
| 8 | Characterization pipeline | `novae/scripts/03_characterize.py` |
| 9 | Feature atlas tables | `novae/atlas/feature_atlas.parquet` etc. |
| 10 | Causal validation pipeline | `novae/scripts/04_causal.py` |
| 11 | Causal validation tables | `novae/atlas/causal_validation.parquet` |
| 12 | Online atlases (3 + 1) | `biodyn-ai/novae-{human,mouse,brain,cross}-atlas` |
| 13 | Paper draft | `novae/paper/main.tex` |
| 14 | Frozen Zenodo snapshot | external |

The paper draft (#13) should follow the structure of arxiv:2603.02952 closely,
with the explicit framing "we extend the comparative atlas methodology to
spatial graph foundation models" so that the two papers form a coherent series.

---

## 8. Resolved decisions (2026-04-09)

All five open decisions are resolved:

1. **Activation corpus**: **both** — `MICS-Lab/novae` HF dataset for SAE
   training (in-distribution) plus external held-out atlases for causal
   validation (true generalization test).
2. **Scope of first pass**: **`novae-human-0` only**, then expand to mouse
   and brain after the human pipeline is shown to work end-to-end.
3. **Atlas hosting**: **same `biodyn-ai` GitHub org** as the existing
   Geneformer and scGPT atlases, for visual and methodological continuity.
4. **Paper**: **separate paper**, framed as an extension of the
   comparative-atlas methodology to spatial graph foundation models.
5. **Compute**: **local Apple M2 Pro MacBook (32 GB unified memory)**, no
   CUDA. Implications below.

### 8.1 Compute revision: M2 Pro / 32 GB / MPS

The original plan was budgeted for an A100. The local-MacBook constraint forces
the following revisions:

| Budget item | Original assumption | Revised for M2 Pro |
|---|---|---|
| GPU backend | CUDA single A100 | PyTorch MPS, fall back to CPU where MPS is missing ops |
| SAE training time per (checkpoint, surface) | minutes | tens of minutes to a few hours |
| Total SAE training time | <1 GPU-day | ~1–3 wall-clock days for `novae-human-0` (12 surfaces) |
| Activation extraction | a few GPU-hours | a few wall-clock hours per slide-batch |
| SAE batch size | 4096 | 1024 (drop to 512 if MPS OOMs at d=128, F=4096) |
| Mixed precision | bf16 OK on A100 | float32 (MPS bf16 is unreliable; float16 has precision risk on TopK-grad) |
| Parallel SAEs | several at once on A100 | sequential, one at a time |
| Activation storage as memory-mapped float32 | uncritical | strict — see §8.2 |

The model itself (32M params) and the activation matrices (small dim) easily
fit in 32 GB RAM. The bottleneck is SAE training throughput on MPS. We mitigate
by sequential training, smaller batches, and float32.

### 8.2 Storage revision: 252 GB free on Crucial X6

The disk is 87% full with **only ~252 GB free**. The upstream pipeline used
336 GB *just for Geneformer activations*. We cannot replicate that footprint.
The revised storage budget:

| Item | Original | Revised |
|---|---|---|
| Novae checkpoints | n/a | ~400 MB total (3 × 32M × float32 + tokenizer/config) |
| Activation arrays per layer per checkpoint | 30+ GB | **≤ 5 GB** per surface for human (target ~1.5 M cell-nodes × 128 dim × float32 ≈ 0.77 GB; budget ~5 GB to allow oversampling) |
| Total activations for human checkpoint (12 surfaces) | — | **≤ 50 GB** |
| Activation corpus (raw spatial slides cached locally) | unbounded | **≤ 80 GB** total — strict cap; subset slides aggressively |
| Trained SAE weights | small | <1 GB total |
| External validation atlases | unbounded | **≤ 30 GB**, only what each causal experiment needs, deletable after use |
| Atlas frontend builds and JSON dumps | small | <2 GB |
| **Total budget** | — | **≤ 165 GB** of the 252 GB free, leaving headroom |

Strict policy: every download is checked against the running used-space delta;
if any single download would push free space below 80 GB, it is rejected and
the corpus is re-curated.

### 8.3 Phase ordering revised for human-first

Original Phase 1–5 timing assumed all three checkpoints in parallel. Revised:

- **Phase 0 + 1A**: setup + `novae-human-0` activation extraction.
- **Phase 2A**: SAE training for `novae-human-0` only (12 surfaces).
- **Phase 3A**: characterization for `novae-human-0` only.
- **Phase 4A**: causal validation for `novae-human-0` only.
- **Phase 5A**: ship the `novae-human-atlas` standalone.
- **Phase 6 (later)**: re-run Phases 1–5 for `novae-mouse-0` and `novae-brain-0`
  using the now-stable scripts; ship `novae-mouse-atlas`, `novae-brain-atlas`,
  and `novae-cross-atlas` after both are trained.

This ordering minimizes wasted work if a methodological choice has to be revised
mid-flight (since the second and third checkpoints inherit any fix made on
the first one).

---

*Drafted 2026-04-09. Decisions resolved 2026-04-09.*
