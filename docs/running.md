# Running SHAPE

Install from the repository root with `python -m pip install -e '.[test]'`.
The checked-in configurations resolve paths relative to their own location.
Keep `dataset/` alongside `SHAPE/` when cloning this release.
Start with `--mode train`; the checked-in configurations bind data inputs.

`python run_full_pipeline.py --help` describes four modes:

- `verify`: check data bindings and the evaluation populations; when given a
  trained model specification, also check its model bindings.
- `generate`: use `--spec PATH_TO_FIT/transfer.json` to generate five seeds with
  a trained model, with no target reference or evaluation-plan reads.
- `reproduce`: use the same `--spec` to generate, then evaluate using equal
  complete-user counts.
- `train`: reuse prepared Stage 1 events, retrain Stage 2 and all three Stage 3
  components, then generate and evaluate. Add `--fresh-grid` to refit Stage 1.

Select one of `nyc_to_ca`, `nyc_to_paris`, `ca_to_nyc`, `ca_to_paris`, or `all`.
`--dry-run` shows the command plan without creating a run. `--lanes` controls
generation concurrency on the chosen device; it does not change model settings.
Training runs synchronously and stops on the first failed command.

Stage 3 loads the configuration's `home_quota` file from `dataset/home_quotas/`:
the GHS-POP R2023A 2010 residential population quotas for that direction. The
runner checks its SHA256, source metadata and target grid alignment, and records
the file path, hash and provenance in `generation/runtime.json` and
`generation/manifest.json`. Missing quotas or target-trajectory inputs are
rejected; generation cannot reconstruct quotas from target check-ins.

Older saved `fit/transfer.json` files may bind the previous quota metadata hash.
To reuse existing model weights, update only their `home_quota` path and
`expected_sha256.home_quota` to the corresponding checked-in configuration.
The quota grid IDs and numeric weights are unchanged by this provenance
correction; learned model bindings do not depend on quota-file metadata.

For example, train once, then use the resulting specification for separate
generation and evaluation:

```bash
python run_full_pipeline.py --mode train --direction nyc_to_paris \
  --output results/refit --device cuda:0
python -m SHAPE run generate --spec results/refit/nyc_to_paris/fit/transfer.json \
  --output results/paris-generation --device cuda:0
python -m SHAPE run evaluate --spec results/refit/nyc_to_paris/fit/transfer.json \
  --predictions results/paris-generation/predictions.npz \
  --output results/paris-evaluation
```

The individual stage commands are listed by `python -m SHAPE --help`.
`train-landing-weights` fits the source landing coefficients; `train-context`
fits the additional Home context, initial memory-count and quantile parameters.
Both are fitted and bound automatically in `fit/transfer.json`. The public
runner has no fallback to the older fixed-coefficient model.

Retraining may produce different parameters across platforms. Checkpoints are
selected on source validation data, never on target test metrics. The released
user-selection plans are valid only for the released source/target cohorts;
evaluation rejects altered populations rather than silently misaligning users.
