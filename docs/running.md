# Running SHAPE

Install from the repository root with `python -m pip install -e '.[test]'`.
The checked-in configurations resolve paths relative to their own location.
Keep `dataset/` and `checkpoints/` alongside `SHAPE/` when cloning this release.

`python run_full_pipeline.py --help` describes four modes:

- `verify`: check checkpoint/data bindings and the evaluation populations.
- `generate`: use the fitted model to generate five seeds, with no target
  reference or evaluation-plan reads.
- `reproduce`: generate, then evaluate using equal complete-user counts.
- `train`: reuse prepared Stage 1 events, retrain Stage 2 and all three Stage 3
  components, then generate and evaluate. Add `--fresh-grid` to refit Stage 1.

Select one of `nyc_to_ca`, `nyc_to_paris`, `ca_to_nyc`, `ca_to_paris`, or `all`.
`--dry-run` shows the command plan without creating a run. `--lanes` controls
generation concurrency on the chosen device; it does not change model settings.
Training runs synchronously and stops on the first failed command.

For separate generation and evaluation:

```bash
python -m SHAPE run generate --spec configs/nyc_to_paris.json \
  --output results/paris-generation --device cuda:0
python -m SHAPE run evaluate --spec configs/nyc_to_paris.json \
  --predictions results/paris-generation/predictions.npz \
  --output results/paris-evaluation
```

The individual stage commands are listed by `python -m SHAPE --help`.
`train-landing-weights` fits the source landing coefficients; `train-context`
fits the additional Home context, initial memory-count and quantile parameters.
Both must be fitted and bound in the configuration. The public runner has no
fallback to the older fixed-coefficient model.

Retraining may produce different parameters across platforms. Checkpoints are
selected on source validation data, never on target test metrics. The released
user-selection plans are valid only for the released source/target cohorts;
evaluation rejects altered populations rather than silently misaligning users.
