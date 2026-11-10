# Verification

Run `python run_full_pipeline.py --mode verify --direction all` after cloning.
The command validates data file hashes, the source-code training contract, and
the exact ordered user populations for the evaluation selections. It reports
`scope: data` for the checked-in configurations.

After training, validate the resulting model and data together with:

```bash
python -m SHAPE verify --spec results/refit/nyc_to_paris/fit/transfer.json
```

This additionally checks tokenizer/head lineage, disabled Stage 2 auxiliary
losses, and both learned Stage 3 components and their source bindings. The
report has `scope: model_and_data` and includes their parameter counts.

Run `python -m pytest -q` for numerical model tests and release-interface tests.
These cover disabled auxiliary supervision, real checkpoint/export/head I/O,
rank-bin acceleration, learned context gradients, Home quota preservation,
source-only training inputs and the evaluation sampling unit.

Source statistics remain bound to the exact code, map and head files. For
attraction statistics, the loader also accepts an exact fingerprint of the
effective arrays and role labels. This avoids a false mismatch when BLAS
implementations differ only in discarded float64 intermediate bits. There is
no tolerance or rounding of the arrays used by the model; rank-statistic
checks remain unchanged.

Generation writes a manifest that binds its inputs and output predictions.
Evaluation additionally records all 200 metric draws for each of the five
seeds. These generated outputs belong under ignored `results/` directories.
