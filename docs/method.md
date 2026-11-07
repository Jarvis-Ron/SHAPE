# SHAPE mechanism

The current model uses opportunity-weighted rank and neutral personal-scale parameters. Layer semantic supervision, jump/POI auxiliary supervision and the additive Pareto distance-cost term have been removed. Original candidate construction is retained. Bundled weights and reported metrics belong to the earlier mechanism and require fresh training before current generation.

## Stage 1: static grid representations

A shared grid encoder aligns the 20-dimensional static feature vectors:
13 POI categories, one density feature and six transport features. Its output
is a 64-dimensional latent associated with an explicit int64 grid ID. Sparse
grid geometry comes from GridSpec; raw IDs are never assumed to be dense row
indices. Fresh preparation uses the complete static vocabulary.

## Stage 2: source activity prototypes

The source event encoder combines cyclic time, time gaps, the shared grid
latent, a learned POI-category embedding, and relative jump distance. Residual
FSQ uses three layers of 64 codes each (`levels=[4,4,4]`). The decoder yields
64-dimensional reconstructed event features for the landing head.

Each FSQ layer receives a learned projection of the remaining shared latent
residual. There is no feature-path-to-layer assignment, layer semantic MSE,
semantic projection warm-start, or layer-targeted auxiliary routing. Layers
learn jointly from reconstruction, quantization regularization, temporal/future
objectives. Jump, POI and time auxiliary weights are zero; production training
and checkpoint loading reject auxiliary classifier parameters. Earlier v2
checkpoints with jump/POI auxiliary supervision must be retrained. POI and
relative-jump inputs and reconstruction targets remain available; this change
does not remove those features or prove particular emergent layer semantics.

The fixed training entry point uses no time auxiliary, no encoder surprisal,
IID pre-quantization noise with sigma 0.1, seed 42, 50 epochs, batch size 196 and
an 8,388,608-cell attention budget. Long sequences are kept intact. The
reconstruction weight named `lambda_sem` is separate from the removed layer
semantic loss; it does not assign a target to a quantizer layer.

Venue Word2Vec columns are excluded by the encoder; fresh preprocessing leaves
them zero. A regression test perturbs every excluded column and checks that
the encoder output remains exactly equal. Historical predictions remain usable
for evaluation. Tokenizer loading, source export, landing training and generation
require the new `residual_fsq_no_layer_semantics_no_aux_v3` training contract.

## Stage 3: target trajectory generation

1. **Initialization and source semantics.** Fit code-conditioned movement rings
   and attraction semantics on source training days. Assign target homes to
   source residents by static neighborhood similarity while respecting the
   supplied target population home marginal (Hamilton integer quotas).
2. **Hard opportunity ring.** Draw a coarse opportunity-rank ring for each event
   from the code-conditioned distribution. If a ring has no target cells,
   expand to adjacent rings using the retained rule. Candidates come from all
   emitted static target grids.
3. **Candidate scoring.** Combine reconstructed-feature matching, the learned
   fine-ring density, transferred POI attraction and
   generated-visit memory. Sample the next grid at temperature 0.75; update
   memory using generated visits only. Chronological source-day traversal
   prevents future generated days from entering earlier memory.

For grid `g`, opportunity mass is the sidecar's **raw POI count**,
`w_g = poi_count_raw(g) / mean(poi_count_raw)`. Both cities use

```text
rank(o,d) = sum(w_g for g where distance(o,g) < distance(o,d))
```

Equal-distance grids are excluded together. Unit weights recover the original
unweighted rank exactly. Coarse edges remain `[0,1,5,20,infinity]`; fine edges
and code backoff remain unchanged. Rows are computed with stable sorting and
prefix sums and cached with a bound, avoiding an N×N table for California.

Within the selected ring, the removed additive term is

```text
-gamma * log(max(distance_km(origin, destination), xmin) / xmin)
```

The source `gamma`/`xmin` fit and its enable option are also removed. Other
Stage 3 distance mechanisms retain their original behavior. In the basic
retrieval path, candidates are the union of attribute top-k, the previous
location's physical neighborhood (radius = `candidate_radius_multiplier *
continuity_sigma`), and visit memory. The radius supplements candidates; it
does not exclude top-k or remembered locations outside that neighborhood.

Code-conditioned paths, including the fixed opportunity-rank model, already
score the full static vocabulary and bypass that radius routine. This remains
unchanged. The separate low-level physical-distance kernels, basic continuity
penalty and legacy quantile-remapping helpers are retained. The fixed
four-direction runner explicitly selects rank coordinates and uses no target
distance calibration. Geometry still determines ranks and home contexts, and
distance-based evaluation metrics remain unchanged.

At λ=κ=0, per-resident ring tilting and per-resident beta rescaling are neutral.
Global `revisit_beta=1` remains active. Source rank/revisit profiles are retained
for provenance; they no longer fit any physical cost and do not require positive
source jumps. Inactive profile outputs do not affect sampling. This variant
does not demonstrate an effect of personal-scale modulation.

The generated object is a target-grid trajectory on an observed **source**
resident's schedule. Generation does not use a target person's trajectory
prefix. The paper should describe this as source-behavior-conditioned transfer;
there is no separate behavior-code generator in this release.
