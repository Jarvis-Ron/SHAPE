# SHAPE

SHAPE transfers source-city activity schedules to a target city's grid cells.
This repository contains the final model code with source-learned landing and
Home context parameters, and the data needed for the four
transfer directions: NYC → CA, NYC → Paris, CA → NYC, and CA → Paris.

The model implementation comes from `cc4faf5`, with training, generation, and
evaluation entry points. Train the model before generating trajectories.

## Install and verify

Use Python 3.10 or later. Run commands from the repository root:

```bash
python -m pip install -e '.[test]'
python run_full_pipeline.py --mode verify --direction all
python -m pytest -q
```

Verification checks the four data configurations, file hashes, source-code
training contract, and evaluation user populations. It does not train a model.

## Train, generate and evaluate

```bash
# Preview the commands without starting training.
python run_full_pipeline.py --mode train --direction nyc_to_paris --dry-run

# Reuse the prepared Stage 1 event embeddings and fit Stage 2 and Stage 3.
python run_full_pipeline.py --mode train --direction nyc_to_paris \
  --device cuda:0 --output results/refit
```

Use `--direction all` for all four directions. CPU execution uses `--device cpu`.
Each run uses generation seeds 42–46 and writes predictions plus provenance.
Evaluation writes `evaluation/result.json`, `metric_draws.npz`, and `table.md`
with OD CPC, Spatiotemporal JSD, RoG JSD, and Jump-length JSD.
Output directories must be new to prevent accidental overwrites.

The training chain fits the tokenizer, exports source codes, fits the neural
landing head, then fits the landing weights and Home context parameters.
It writes a bound model specification to `fit/transfer.json`, then generates
and evaluates five seeds. `--fresh-grid` additionally retrains Stage 1 and
rebuilds source events. Training results may vary across hardware and platforms.

To generate again using your trained model, without reading target evaluation
data:

```bash
python run_full_pipeline.py --mode generate \
  --spec results/refit/nyc_to_paris/fit/transfer.json \
  --device cuda:0 --output results/generated
```

Use `--mode reproduce` with the same `--spec` to generate and evaluate together.

## Model and information conditions

- Stage 1 encodes 20-dimensional static grid attributes with shared weights.
- Stage 2 uses three residual FSQ layers without layer-level semantic targets
  or jump/POI/time auxiliary supervision. Reconstruction and other training
  regularizers remain part of its objective.
- Stage 3 loads **both** `learned_landing.pth` (68 parameters) and
  `learned_context.pth` (7 additional parameters), alongside the neural landing
  head. Missing learned checkpoints are rejected by the public runner.
- The additive absolute-distance penalty is absent. Geographic distances still
  determine opportunity order and geographic feasibility; this is not a claim
  that the model never uses geometry.

Generation preserves source times and activity categories. Target static maps
and an **aggregate target training-home quota** are allowed inputs. The quota
is derived from target training check-ins and is supplied as a generation
condition; it is not inferred from static residential attributes alone. Target
test trajectories are used only for evaluation. This release therefore does
not represent an unconditional, target-trajectory-free setting.

Evaluation matches the number of complete eligible users on each side for 200
fixed selections. It does not match day or event counts. Each seed is scored
separately, then results report the mean and sample SD across the five seed
means. Five generation seeds do not represent five independent training runs
or five times as many independent people.

## Contents

| Directory | Contents |
|---|---|
| `SHAPE/` | Current three-stage model, training and evaluation code |
| `configs/` | Four training-data and evaluation configurations |
| `dataset/` | Static maps, source data, processed reference data, Home quotas and evaluation selections |
| `tests/` | Model and release verification |

NYC-source directions share the same prepared source events. The two CA-source
directions retain their corresponding prepared embeddings.
Only the current model and supporting data are included in the working tree;
comparison models, ablation runners, old predictions and historical score
tables are omitted. Git history is retained.

See [running](docs/running.md), [data](docs/datasets.md),
[evaluation](docs/data-and-evaluation.md), [learned landing](docs/learned-landing.md),
[learned context](docs/learned-context.md), and [data attribution](THIRD_PARTY_DATA.md).
