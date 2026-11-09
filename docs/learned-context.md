This optional V2 extends the source-trained landing scorer at `75cd8e0` with
seven additional scalar parameters. Existing Stage 1/2 checkpoints, the landing
network and the V1 scorer are frozen. V1 remains available by omitting
`learned_context_checkpoint` from the generation spec.

The new parameters are the residential density/diversity mix used in home
contexts, continuous effective neighborhood size, relative residential/POI
matching cost weight, separate initial home and anchor counts, and two ordered
global quantile levels for landing attributes. Exact target home quotas and the
Hamilton assignment algorithm remain constraints, not learned quantities.

Home-context fitting uses source training-day Layer-0 code histograms grouped by
source modal-home cell. A soft neighborhood distribution based on behavior JSD
is fitted by a distribution based on context cost. Twenty percent of source
home cells are held out for checkpoint selection; a cell cannot match itself.
No cross-city pair labels or target behavior labels are supplied. This is a
source metric-learning surrogate, not a proof of transferable home assignments.

The hard residential percentile is replaced by a sigmoid empirical-CDF estimate
using at most 1,024 deterministic static-map probes (width .05). The hard 25-cell
neighborhood is replaced by sigmoid rank weights (width 2) over at most 100
nearest cells; its learned effective size lies in [1,100]. Relative context
weights sum to two, removing an unidentifiable global scale in greedy matching.
The soft matching likelihood temperature is .1. These are fixed design choices.

Count and quantile fitting uses the same source conditional ring likelihood as
V1. The observed first source event initializes teacher forcing and is excluded
from the loss; generation starts at the assigned home. Past training days alone
seed training/validation cross-day memory. Initial home count wins if the same
location is also an anchor; subsequent visits increment counts by one. The
existing train/generation history distinction remains a limitation.

Quantile levels are differentiable via piecewise-linear empirical interpolation
on the same pooled static maps used by the original landing head. The lower level
lies in (0,.5), the upper in (.5,1), initially .05/.95. The fitted numerical bounds
are checkpointed and applied unchanged at generation. No target trajectories
select the levels. Static map access is an existing protocol condition.

Training seed is 42. Home fitting uses Adam lr .03, at most 100 epochs,
patience 10; count/quantile fitting uses Adam lr .01, batch 256, at most 20
epochs, patience 5. Both retain the best source-validation checkpoint including
initialization, with minimum improvement 1e-6. A parameter is trainable even if
the validation rule selects its initial state; report such outcomes honestly.

```bash
python -m SHAPE.stage3.learned_context --codes SOURCE_CODES \
  --landing-checkpoint LANDING --source-grid SOURCE_MAP \
  --learned-landing-checkpoint V1_WEIGHTS \
  --pooled-grids NYC_MAP CA_MAP PARIS_MAP --output FIT_DIR --device cuda:0
```

Add `learned_context_checkpoint` and its expected SHA256 to the existing V1
generation spec. Checkpoint loading binds source codes, map, neural head and
V1 weights; provenance records learned values and normalization bounds.

The V1 fit implementation also materializes rank-bin rows only for observed
source origins. It still includes every destination and preserves the exact
full-matrix probabilities. This avoids allocating a full CA-by-CA matrix and
does not change the statistical model.
