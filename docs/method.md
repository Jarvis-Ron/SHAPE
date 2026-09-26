# SHAPE mechanism

The released model uses opportunity-weighted rank, physical distance cost, and neutral personal-scale parameters. The entry point runs only this fixed mechanism.

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

The historical tokenizer has layered semantic injection, jump/POI auxiliary
weights 0.125/0.075, no time auxiliary, no surprisal, and IID pre-quantization
noise with sigma 0.1. It was trained with seed 42, 50 epochs, batch size 196 and
an 8,388,608-cell attention budget in the current training entry point (historical CA training used 51,380,224 cells). Long sequences are kept intact.

Venue Word2Vec columns are excluded by the encoder; fresh preprocessing leaves
them zero. A regression test perturbs every excluded column and checks that
the encoder output remains exactly equal. Historical archives remain usable.

## Stage 3: target trajectory generation

1. **Initialization and source semantics.** Fit code-conditioned movement rings
   and attraction semantics on source training days. Assign target homes to
   source residents by static neighborhood similarity while respecting the
   supplied target population home marginal (Hamilton integer quotas).
2. **Hard distance ring.** Draw a coarse opportunity-rank ring for each event
   from the code-conditioned distribution. If a ring has no target cells,
   expand to adjacent rings using the retained rule. Candidates come from all
   emitted static target grids.
3. **Candidate scoring.** Combine reconstructed-feature matching, the learned
   fine-ring density, transferred POI attraction, physical distance cost and
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

Within the selected ring, the additional physical cost is

```text
cost(o,d) = -gamma * log(max(distance_km(o,d), xmin) / xmin)
gamma = 1 + n / sum(log(source_positive_train_jump / xmin))
```

`xmin` is the smallest positive source-training jump. This additive log-score
term favors shorter moves among otherwise eligible opportunities. Target
distance statistics do not enter this fit.

At λ=κ=0, per-resident ring tilting and per-resident beta rescaling are neutral.
Global `revisit_beta=1` remains active. The historical profile fitter is retained
as the source-cost estimator and for provenance compatibility; inactive profile
outputs do not affect sampling. This variant does not demonstrate an effect of
personal-scale modulation.

The generated object is a target-grid trajectory on an observed **source**
resident's schedule. Generation does not use a target person's trajectory
prefix. The paper should describe this as source-behavior-conditioned transfer;
there is no separate behavior-code generator in this release.
