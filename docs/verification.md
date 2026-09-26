# Executed verification

The packaged native scorer recomputed all four historical rows from the supplied
original generated trajectories and target references. Across 11,025 primary
metric values, the largest absolute discrepancy from the archived original
scores was below 2e-16. All five reported means and sample standard deviations
match at four decimal places. This checks trajectories, not just summary JSON.
Per-direction counts and errors are in `release_validation.json`.

The local test suite passed 64 tests: static-grid training/extraction, source
feature isolation, FSQ and auxiliary-loss backpropagation, attention batching,
weighted-rank ties, distance cost, target-home boundaries, pipeline stage
routing, archived-draw reproduction, and distinct seed/shell uncertainty units.
A synthetic integration test exports source codes, trains a source landing
head, saves its bindings, and reloads the checkpoint. A real two-user CA export
also succeeded using the bundled tokenizer, prepared features, and raw calendar.

NYC → Paris was additionally regenerated for all five seeds on CPU. Three
seeds reproduced every grid ID; seeds 43 and 45 differed in 15 and 1 events,
respectively, out of 22,772 events each. The regenerated spatiotemporal mean
rounds to 0.2501 instead of the archived 0.2500; the other reported entries match
to four decimals. Frozen probabilities near a sampling boundary can produce
hardware-dependent trajectories. Original predictions remain the reference
for exact historical scoring and were not replaced by the CPU rerun.

The earlier NYC → CA native generation regression is retained in
`provenance/nyc_to_ca/refactor_regression.json`. It reproduced all three original
prediction arrays exactly on the original CUDA software stack. That historical
receipt is distinct from the current local CPU checks.

Metadata anonymization checked 5,964 numerical arrays/tensors against their
archived values, with no value changes. The NYC → Paris representative loss was
obtained by a real 20-epoch source landing-head rerun. All other plotted loss
histories are original records. The representative rerun does not replace the
frozen evaluation weights.

Fresh full-data three-stage training for all directions has not been executed
as part of this release check. Its commands, source-only supervision, synthetic
I/O, and prepared-CA export path were checked. Historical upstream embeddings
and current raw preparation are distinct workflows; the release does not claim
bitwise reconstruction of unavailable upstream histories. Tested package
versions are recorded in `environment.json`.
