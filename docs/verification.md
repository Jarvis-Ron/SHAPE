# Verification

Run `python run_full_pipeline.py --mode verify --direction all` after cloning.
The command validates file hashes, source tokenizer/head lineage, both learned
checkpoint schemas and bindings, and the exact ordered user populations for
the evaluation selections. It reports 68 landing and seven context parameters
for each direction.

Run `python -m pytest -q` for numerical model tests and release-interface tests.
These cover disabled auxiliary supervision, real checkpoint/export/head I/O,
rank-bin acceleration, learned context gradients, Home quota preservation,
source-only training inputs and the evaluation sampling unit.

`checkpoints/manifest.json` records the original and published artifact hashes.
The export removes private path metadata and optimizer resume state, updates
dependent checksum bindings, and preserves model tensors/data arrays. These
are the fitted final-model artifacts, not weights from a new training run.

Source statistics remain bound to the exact code, map and head files. For
attraction statistics, the loader also accepts an exact fingerprint of the
effective arrays and role labels. This avoids a false mismatch when BLAS
implementations differ only in discarded float64 intermediate bits. There is
no tolerance or rounding of the arrays used by the model; rank-statistic
checks remain unchanged.

Generation writes a manifest that binds its inputs and output predictions.
Evaluation additionally records all 200 metric draws for each of the five
seeds. These generated outputs belong under ignored `results/` directories.
