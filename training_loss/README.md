# Representative training losses

`landing_loss.csv`, `landing_loss.png`, and `landing_loss.pdf` show the source
landing-head cross-entropy over 20 epochs, with training seed 42. Every plotted
point comes from an actual logged epoch. Curves are not smoothed or interpolated.
The loss is measured on source training/validation events, not target test data.

| Direction | Selected curve | Provenance |
|---|---|---|
| NYC → CA | Landing-head train/validation loss | Original training log |
| NYC → Paris | Landing-head train/validation loss | New representative CPU rerun on the frozen source codes |
| CA → NYC | Landing-head train/validation loss | Original training log |
| CA → Paris | Landing-head train/validation loss | Original training log |

The NYC → Paris original per-epoch log was not recovered. The representative
rerun uses the recorded 20 epochs, batch size 4,096, learning rate 0.001, weight
decay 0.00001, seed 42, and pooled static normalization. Its best validation loss
is 6.9034576684; the archived checkpoint records 6.9034589316. The rerun does not
replace that checkpoint, its generated outputs, or its reported metrics.

Available original tokenizer and grid-encoder histories are also included in
the corresponding direction directories. Training source can be shared across
target directions; these plots are illustrative histories, not four independent
training-seed uncertainty estimates.

Regenerate the CSV and plot with `python scripts/plot_training_loss.py`.
