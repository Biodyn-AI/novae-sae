# Causal Circuit Tracing Plan for Novae

## Motivation

Papers arXiv:2603.01752 ("Causal Circuit Tracing") and arXiv:2603.11940
("Exhaustive Circuit Mapping") established a methodology for discovering
directed causal relationships between SAE features across layers of
single-cell foundation models. Both papers worked on transformer
architectures (Geneformer, scGPT) where causal effects propagate through
the additive residual stream.

Novae's GATv2 graph architecture creates a fundamentally different
setting: causal effects propagate through **attention-weighted spatial
aggregation**. When a feature is ablated at layer N, the effect at layer
N+1 depends on which cells' neighbors were affected. This means circuit
edges in Novae are **spatially mediated** — a novel property not present
in transformer circuits. This is the primary scientific contribution of
the Novae circuit tracing extension.

## Gap Analysis (17 gaps, 3 tiers)

### Tier 1 — Falls out of causal circuit tracing (GAP 1)

| Gap | Description | Dependency |
|-----|-------------|------------|
| **GAP 1** | Causal circuit tracing: zero feature at layer L, measure Cohen's d on downstream SAE features | Master prerequisite |
| GAP 2 | Inhibitory/excitatory balance: fraction of edges with negative vs positive d | GAP 1 |
| GAP 3 | Biological coherence: fraction of edges where source and target share GO/KEGG terms | GAP 1 + existing enrichments |
| GAP 6 | Process hierarchy from layer position: mean layer of each GO domain in circuit graph | GAP 1 |
| GAP 7 | Effect attenuation curves: edge count vs layer distance | GAP 1 |
| GAP 8 | Hub feature identification: rank features by in-degree/out-degree in circuit graph | GAP 1 |

### Tier 2 — Requires GAP 1 + moderate additional work

| Gap | Description | Extra work |
|-----|-------------|------------|
| GAP 5 | Novel relationship discovery: compare circuit edges to STRING/GO reference graph | Download STRING, build reference graph |
| GAP 9 | PMI co-activation vs causal comparison: validate circuit edges against statistical co-activation | Compute PMI between feature pairs |
| GAP 10 | Tissue-specific circuit enrichment: Fisher test for tissue keywords in circuit edges | Use existing tissue annotations |
| GAP 11 | Gene-level circuit predictions: translate feature edges to gene pairs via top-loading genes | Use existing top-gene tables |
| GAP 12 | Disease gene set mapping: map circuit edges to disease-relevant domains | Download disease gene sets |

### Tier 3 — Independent or blocked

| Gap | Description | Status |
|-----|-------------|--------|
| GAP 13 | Exhaustive tracing: trace ALL active features at one layer (not just top-50) | Independent, compute-heavy |
| GAP 14 | Hub annotation bias: check if top hubs are disproportionately unannotated | Independent, quick |
| GAP 15 | Higher-order combinatorial ablation: ablate 2-3 features, measure redundancy/synergy | Independent |
| **GAP 16** | **Trajectory-guided feature steering along spatial gradients** | **Novel — not in either paper** |
| GAP 4 | Cross-model consensus domain pairs | Blocked: needs second spatial model with SAEs |
| GAP 17 | SAE-vs-cell-type dissociation | Needs SAEs trained on single-tissue subsets |

---

## Implementation Plan

### Phase 1: Causal Circuit Tracing (GAP 1)

**Script:** `scripts/19_causal_circuit_tracing.py`

**Protocol:**
1. Select source features: top-50 most active features at each of 3
   source layers (conv_0, conv_5, conv_9). Total: 150 source features.
2. For each source feature F at source layer L:
   a. Load a batch of 200 cells where F is highly active (top-1% by
      activation magnitude on the shared 566K node corpus).
   b. Run the **baseline** forward pass: encode cells through layers
      L..aggregator, capturing SAE features at each downstream layer.
   c. Run the **ablated** forward pass: zero feature F's TopK coefficient
      at layer L, decode back to hidden space, then propagate through
      layers L+1..aggregator, capturing SAE features at each downstream
      layer.
   d. For each downstream feature G at each downstream layer M:
      compute Cohen's d = (mean_baseline - mean_ablated) / pooled_std.
      Keep edges where |d| > 0.5 and consistency > 0.6 (fraction of
      cells where the sign of the effect matches the mean sign).
3. Output: `causal_circuit_edges.parquet` with columns:
   source_layer, source_feature, target_layer, target_feature,
   cohen_d, sign (+1/-1), consistency, n_cells.

**Architectural adaptation for GATv2:**
In a transformer, you can ablate a feature in the residual stream and
let the modified state flow through subsequent layers. In Novae's GATv2,
the hidden state at layer N feeds into the attention mechanism at layer
N+1 — but the attention operates over the GRAPH, meaning the ablated
cell's modified hidden state also affects its neighbors' attention
weights. This spatial propagation is the key difference.

For the initial implementation, we use a simplified approach:
- Hook each GATv2 conv layer's output
- At the source layer L, replace the output for the target cells with
  the SAE-ablated version (zero one feature, decode back)
- Let this modified output flow through subsequent conv layers naturally
  (including graph attention propagation to neighbors)
- Measure effects on downstream SAE features for the same target cells

This captures both direct computational effects AND spatially-mediated
effects (neighbors receiving ablated attention messages).

**Compute estimate:** 150 source features × 200 cells × 2 passes
(baseline + ablated) × ~1s per pass ≈ 5 hours on MPS. Can be
parallelized by processing source layers independently.

### Phase 2: Circuit Graph Analysis (GAPs 2, 3, 6, 7, 8)

**Script:** `scripts/20_circuit_analysis.py`

Uses the edge table from Phase 1 to compute:
- Inhibitory/excitatory balance (GAP 2): fraction of edges with d < 0
- Biological coherence (GAP 3): for each edge, check if source and
  target features share at least one GO/KEGG term from the enrichment
  tables
- Process hierarchy (GAP 6): compute mean source layer and mean target
  layer for each GO domain; plot domain ordering
- Effect attenuation (GAP 7): histogram of edge count vs |target_layer -
  source_layer|; fit exponential decay
- Hub identification (GAP 8): rank features by out-degree and in-degree;
  characterize top-10 hubs biologically

**Compute estimate:** <5 min (pure post-processing on the edge table).

### Phase 3: Novel Relationship Discovery (GAPs 5, 10, 11, 12)

**Script:** `scripts/21_circuit_biology.py`

- Download STRING database (human) for reference graph of known gene
  interactions
- For each circuit edge, translate source and target features to their
  top-5 genes; check if any gene pair exists in STRING → "known" edge
- Edges NOT in STRING are "novel" predictions
- Tissue-specific enrichment: for edges between tissue-concentrated
  features, Fisher test for tissue-keyword enrichment
- Disease mapping: use DisGeNET or OMIM gene-disease associations

**Compute estimate:** ~30 min (STRING download + matching).

### Phase 4: Advanced Analyses (GAPs 13, 14, 15)

**Script:** `scripts/22_exhaustive_tracing.py`

- Exhaustive tracing: trace ALL ~4000 active features at conv_5 (not
  just top-50). Very compute-heavy (~40 hours for full exhaustive).
  Compromise: trace top-500 at one layer.
- Hub annotation bias: compare annotation rate of top-20 hubs vs
  random features
- Higher-order ablation: pick 10 triplets of co-active features,
  ablate in all 7 combinations (A, B, C, AB, AC, BC, ABC), compute
  redundancy ratio R_ABC = median(pairwise_effect) / joint_effect

**Compute estimate:** ~8 hours for top-500 exhaustive; ~2 hours for
combinatorial ablation.

### Phase 5: Spatial Gradient Steering (GAP 16) — NOVEL

**Script:** `scripts/23_spatial_steering.py`

This is a **new contribution** not in either paper. Novae doesn't have
a pseudotime/differentiation axis, but it HAS spatial gradients:
- Crypt-to-villus in colon (zonation axis)
- Cortex-to-medulla in kidney
- Follicle center to mantle zone in lymph node
- Tumor core to periphery in breast IDC

Protocol:
1. For each spatial gradient, define a "maturity" gene signature (e.g.,
   villus genes vs crypt genes in colon).
2. For each SAE feature at each layer, multiply its activation by
   alpha (0.5x, 2x, 5x), propagate through remaining layers, and
   measure the cosine shift of the aggregator output toward/away from
   the maturity signature.
3. Features that push toward maturity when amplified are "pro-maturity";
   those that push away are "anti-maturity". This reveals which internal
   features encode position along spatial gradients.

**Compute estimate:** ~4 hours for 3 gradients × 50 features × 3 alphas.

---

## Output Artifacts

| Artifact | Path |
|----------|------|
| Circuit edge table | `atlas/novae-human-0/causal/causal_circuit_edges.parquet` |
| Circuit analysis summary | `atlas/novae-human-0/causal/circuit_analysis.json` |
| Hub features | `atlas/novae-human-0/causal/circuit_hubs.parquet` |
| Novel predictions | `atlas/novae-human-0/causal/novel_predictions.parquet` |
| Exhaustive tracing | `atlas/novae-human-0/causal/exhaustive_circuit_conv5.parquet` |
| Combinatorial ablation | `atlas/novae-human-0/causal/combinatorial_ablation.parquet` |
| Spatial steering | `atlas/novae-human-0/causal/spatial_steering.parquet` |

## Atlas Pages

- **Circuit Explorer** page showing the directed graph of causal edges,
  filterable by source/target layer, biology theme, and edge strength.
- **Spatial Steering** page showing per-feature steering effects along
  tissue gradients.

## Paper Sections

The circuit tracing results would form the core of a follow-up paper:
"Causal circuits in a spatial graph foundation model reveal spatially
mediated regulatory logic." The novel contribution vs. the transformer
papers is the spatial propagation of causal effects through the graph
attention mechanism.

---

## Execution Order

1. **Phase 1** (GAP 1): `19_causal_circuit_tracing.py` — ~5 hours
2. **Phase 2** (GAPs 2,3,6,7,8): `20_circuit_analysis.py` — ~5 min
3. **Phase 3** (GAPs 5,10,11,12): `21_circuit_biology.py` — ~30 min
4. **Phase 4** (GAPs 13,14,15): `22_exhaustive_tracing.py` — ~10 hours
5. **Phase 5** (GAP 16): `23_spatial_steering.py` — ~4 hours
6. **Atlas + Paper update** — ~2 hours

Total estimated compute: ~22 hours.

---

*Plan drafted 2026-04-12. Based on gap analysis of arXiv:2603.01752 and
arXiv:2603.11940 vs. current Novae atlas.*
