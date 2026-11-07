# SHAPE

Research code for cross-city trajectory generation. SHAPE transfers source-city activity sequences to a target city's static grid using activity prototypes, opportunity-weighted rings, and generated-visit memory. The current implementation removes layer-specific semantic supervision and jump/POI auxiliary supervision from residual FSQ, and the additive Pareto distance-cost term from Stage 3. Original candidate selection, including the physical-radius neighborhood in the basic retrieval path, is retained.

**The bundled weights, predictions, loss histories and four reported rows are historical artifacts from the earlier model with layer semantic supervision and distance cost.** They are retained for scoring and verification. Train new Stage 2 and landing checkpoints to use the current model; old tokenizer checkpoints and source-code archives are rejected by training/export/generation rather than silently relabeled.

## Installation

Use Python 3.10 or later in a virtual environment. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

CPU is sufficient for scoring the supplied outputs. CUDA is recommended for full training and large-target generation. Install the PyTorch build appropriate for your hardware first if necessary. Dependencies and tested local versions are recorded in `pyproject.toml` and `docs/environment.json`.

## Score the four historical rows

Datasets, historical Stage 3 inputs, and original generated trajectories are included. No external model download or private source repository is required for historical scoring.

```bash
# Recompute metrics from original generated grid sequences.
bash run_full_pipeline.sh --direction all --mode score --output results/scored

# Check saved metric draws and summaries; no new generation.
bash run_full_pipeline.sh --direction all --mode verify
```

Select one direction with `--direction nyc_to_ca`, `nyc_to_paris`, `ca_to_nyc`, or `ca_to_paris`. Use `--workers 4` to control scoring workers. Output directories must be new. Failures stop the pipeline and retain completed outputs.

| Direction | OD CPC ↑ | Spatiotemporal JSD ↓ | RoG JSD ↓ | Jump-length JSD ↓ | POI-category JSD ↓ |
|---|---:|---:|---:|---:|---:|
| NYC → CA | 0.8330 ± 0.0054 | 0.4058 ± 0.0024 | 0.1777 ± 0.0032 | 0.1248 ± 0.0013 | 0.0015 ± 0.0001 |
| NYC → Paris | 0.5806 ± 0.0033 | 0.2500 ± 0.0027 | 0.0034 ± 0.0004 | 0.0292 ± 0.0013 | 0.0002 ± 0.0000 |
| CA → NYC | 0.6242 ± 0.0092 | 0.3673 ± 0.0020 | 0.2152 ± 0.0116 | 0.0692 ± 0.0032 | 0.0015 ± 0.0001 |
| CA → Paris | 0.5663 ± 0.0030 | 0.2855 ± 0.0027 | 0.0025 ± 0.0003 | 0.0226 ± 0.0015 | 0.0002 ± 0.0000 |

**The rows use distinct evaluation protocols.** NYC → CA pools three generated populations before each of 200 matched-shell scores; its SD is over those 200 scores on 8,870 common-support target days. Both Paris rows average 200 shell scores within each of five generation seeds and report the mean and SD of the five seed means. CA → NYC directly scores each of five generated populations, independently normalizing OD distributions to unit mass. These are sampling seeds with fixed learned weights, not independent training runs. See [evaluation details](docs/data-and-evaluation.md).

## Train the model

```bash
# Inspect the commands before allocating training resources.
bash run_full_pipeline.sh --mode train --direction nyc_to_ca --device cuda:0 --dry-run

# Static encoder → source event preparation → tokenizer → landing head
# → target generation → evaluation.
bash run_full_pipeline.sh --mode train --direction nyc_to_ca --device cuda:0 --output results/fresh

# The same complete workflow for all four directions.
bash run_full_pipeline.sh --mode train --direction all --device cuda:0 --output results/fresh_all

# Optionally reuse the supplied historical CA source-event embeddings.
bash run_full_pipeline.sh --mode train --direction ca_to_paris --prepared-events --device cuda:0 --output results/ca_prepared
```

Fresh training produces the current model, with no layer semantic loss, semantic warm-start, jump/POI auxiliary losses, or Stage 3 Pareto distance cost. The fixed four-direction runner retains opportunity-rank generation without target-distance calibration. Its Stage 1 uses the complete static map vocabulary. Historical CA event embeddings use a smaller source cohort and can still be used to train a new tokenizer with `--prepared-events`. The table above remains the archived baseline; separately executed experiments must record their current source revision, training contract, inputs and evaluation protocol.

To sample again after fresh training, use the resulting bound spec:

```bash
python -m SHAPE.reproduce generate --spec results/fresh/nyc_to_ca/fit/transfer.json --device cuda:0 --output results/fresh_resampled
```

`--mode reproduce` with the bundled historical Stage 3 inputs now fails with a retraining instruction. For the original generator, use repository revision `fb1f69aa79073e0769a7686fbba6b4eedcde349f`; exact historical **metric** reproduction remains available on this revision with `--mode score`. Floating-point kernels and hardware can affect sampled sequences.

Target check-ins do not train the behavioral tokenizer or landing head. The recorded experiments use target training-home population marginals for initialization. Matched-shell evaluation separately uses target user/day structure after generation. Each direction's information boundary is documented.

## Included artifacts

- `dataset/`: static grids and provenance, NYC and CA raw check-ins, and prepared CA event datasets.
- `inputs/<direction>/`: source codes, frozen landing weights, target reference sequences, original predictions, and fixed shell plans or home quotas.
- `reference_results/<direction>/`: original metric samples and summaries.
- `training_loss/`: one landing-head training/validation curve per direction, CSV values, a four-panel plot, and available tokenizer/grid histories.

The NYC → Paris landing curve is a new seed-42 rerun on the frozen source codes with the recorded hyperparameters. The other three curves are original logs. The new curve does not replace historical weights or predictions. See [data contents](docs/datasets.md), [dataset credits](THIRD_PARTY_DATA.md), and [loss provenance](training_loss/README.md).

## Model and code

| Stage | Input → output | Implementation |
|---|---|---|
| Static representation | 20 static attributes → 64-dimensional grid embeddings | `SHAPE/stage1/` |
| Activity prototypes | Source events → three residual FSQ codes and reconstructed features | `SHAPE/stage2/` |
| Spatial generation | Source behavior, target map, home marginal → target-grid trajectories | `SHAPE/stage3/` |

The fixed model uses opportunity-weighted rank with personal-scale parameters λ = κ = 0 and global generated-visit memory. Physical distance is used to order static opportunities, not as an additional candidate penalty or a target-trajectory calibration. Stage 2 learns from reconstruction, quantization regularization and temporal/future objectives. Jump, POI and time auxiliary weights are zero and their classifier heads are absent; no prescribed semantic targets are assigned to individual quantizer layers. See [method details](docs/method.md) and [the code map](docs/code-map.md). `run_full_pipeline.py` is the four-direction entry point; `SHAPE/reproduce.py` performs generation and evaluation. New tokenizer/source archives carry the `residual_fsq_no_layer_semantics_no_aux_v3` training contract.

The removed Stage 3 term is `-gamma * log(max(distance_km, xmin) / xmin)`.
The basic sampler still combines attribute top-k candidates, a physical-radius
neighborhood and visit memory. The fixed opportunity-rank sampler retains its
original full-vocabulary candidate path, so this revision does not introduce a
new hard radius cutoff into rank generation. Other legacy distance kernels and
calibration helpers remain available at the low level; see [running details](docs/running.md).

Outputs include `predictions.npz`, input hashes, runtime provenance, `metric_draws.csv`, `metrics.csv`, `five_metrics.csv`, and `result.json`. The scorer computes values from trajectories; it never replaces them with the reference table. Executed checks and their limits are in [verification.md](docs/verification.md).
