# Released model

Stage 1 uses a shared grid encoder over 20-dimensional static features.
Stage 2 learns three residual FSQ code layers by reconstruction and its
regularization objectives. Layer semantic supervision and jump/POI/time
auxiliary heads are disabled in the release training interface.

Stage 3 fits a neural query head on source codes and source static attributes.
It then learns 68 landing parameters and seven additional context parameters
on source data. Both learned checkpoints are required by the released runner.
The source schedule is retained during target grid selection.

The learned components cover opportunity-score strength, POI attraction,
revisit strength, residential mixtures/gates, Home context construction and
matching, initial Home/anchor counts, and two attribute quantile levels.
Architecture, optimizer settings, coarse rank boundaries and numerical
stability settings remain specified choices. Trainable parameters can retain
their initial values when source validation selects initialization.

There is no additive absolute-distance penalty. Geographic ordering and
feasibility still use geometry. Target static attributes and public Home quotas
from the GHS-POP R2023A 2010 residential population layer are inference inputs.
Stage 3 reads the precomputed quotas from `dataset/home_quotas/` and allocates
Home slots in proportion to them. This is zero-shot source-schedule transfer
conditioned on static maps and public population data. Target trajectories
are used only for evaluation; they supply no Home quotas, training signals,
calibration statistics or checkpoint-selection outcomes.

See the two learned-component documents for objectives and fixed settings.
