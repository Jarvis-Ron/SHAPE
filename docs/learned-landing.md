# Source-learned landing parameters

This optional Stage 3 variant learns 68 parameters on source-city trajectories:
three positive scoring strengths (revisit, POI attraction and opportunity rank),
one density/diversity mixing proportion, and 64 code-specific residential gates.
Stage 1, Stage 2, the pretrained landing head, its logit scale and both sampling
temperatures remain frozen. There is no new Stage 2 semantic supervision or
absolute-distance penalty.

```sh
python -m SHAPE train-landing-weights \
  --codes /path/to/source_codes.npz \
  --landing-checkpoint /path/to/landing.pth \
  --source-grid dataset/NYC_grid_merged_features.csv \
  --output /path/to/learned_fit --device cuda:0
```

The source grid must have its existing static provenance CSV beside it. The
source codes must match the frozen landing checkpoint and tokenizer. No target
trajectory or reference argument is accepted by the trainer.

The objective is source next-location cross entropy conditional on the true
coarse opportunity ring. The gold location is never removed by a randomly
sampled training ring. Fine-bin probability/cardinality, attribute scores,
revisit counts and attraction follow the existing sampler. Within-ring
attraction normalization adds a constant and therefore cancels in this loss.
Coarse-bin probabilities and boundaries are unchanged. The training objective
does not claim to optimize unconditional trajectory metrics directly.

Training is teacher forced: the first observed event initializes a day, then
only its observed prefix supplies within-day memory. Earlier training days
supply familiar-location anchors with count one. Validation/test days never
update training memory. Validation uses earlier training days and its own
within-day prefix. In generation the first event is still the supplied home,
and all subsequent memory uses generated locations, so a separate source
autoregressive check remains useful.

Initial strengths are 1, 1 and 0.8. Softplus keeps them positive. The density
mixture starts at 0.625. Each gate starts at
`(source_train_home_night_support + 1) / (source_train_support + 2)` and uses a
sigmoid. Residential and ordinary POI distributions are each normalized before
mixing; the final code attraction has unit mean. Gates learn location likelihood,
not predefined Stage 2 semantic labels. An unsupported code receives no direct
likelihood evidence for its gate.

Defaults are Adam, learning rate 0.01, batch size 512, at most 20 epochs, patience
5, and seed 42. The checkpoint is selected by source validation NLL, including
the initialization as epoch zero. Training writes `history.json`, `report.json`
and `learned_landing.pth`. These optimizer/budget choices remain fixed settings;
making scoring parameters learnable does not eliminate all hyperparameters.

To generate, add `learned_landing_checkpoint` to the normal transfer spec and
add its SHA-256 under `expected_sha256` with the same key. `python -m SHAPE.reproduce
generate --help` documents generation arguments. Omitting this optional field
retains the original fixed scoring behavior. A learned checkpoint is rejected
when source codes, source static table, frozen head or source statistics differ.
Generation manifests record the variant and learned checkpoint hash.

Target static map attributes determine the available opportunities and
attraction table. Target home quotas remain the existing protocol input; the
learned gate does not infer or replace that population. Target trajectories
must not choose parameters or checkpoints. Five generation seeds are five
sampling repeats of a fixed fitted model, not five independently trained models
or five times as many independent people.
