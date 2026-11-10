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
feasibility still use geometry. Target static attributes and a supplied
aggregate training-home quota are inference inputs. This is source-schedule
transfer under those conditions, not unconditional learning of target activity
schedules. No target test outcomes select parameters or checkpoints.

See the two learned-component documents for objectives and fixed settings.
