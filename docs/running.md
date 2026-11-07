# Running SHAPE

The supported four-direction entry point is `run_full_pipeline.py` (or the shell wrapper).

| Mode | Work performed | Requires training |
|---|---|---|
| `verify` | Verify fixed input hashes and recompute summaries of archived metric draws | No |
| `score` | Recompute metrics from the original generated trajectories | No |
| `reproduce` | Generate from current-contract Stage 3 inputs, then score; bundled historical inputs are rejected | Requires new Stage 2/landing artifacts |
| `train` | Train static encoder, prepare source events, train tokenizer and landing head, generate, score | Yes |

Run `python run_full_pipeline.py --help`. Four-direction configs resolve paths relative to their containing directory. Their SHA-256 bindings protect against accidentally mixing inputs. Fresh-training outputs receive new bindings. The full runner invokes stages synchronously and stops on failure; each training stage has its own log. Use a fresh output directory for each execution.

Use `--mode train` for the current model. The four bundled configs bind older
supervised-tokenizer archives, so their `--mode reproduce` path intentionally
fails before generation. Reuse a new run with
`python -m SHAPE.reproduce generate --spec <fit/transfer.json> --device cpu --output <new-directory>`.
Changing metadata on old weights is not a substitute for retraining: export
checks the tokenizer training contract and Stage 3 binds the new codes to the
retrained landing head. Removed semantic supervision CLI/config options raise
an error, including a zero semantic coefficient; there is no enable switch.
The old `cost` configuration key and physical-distance sampler arguments are
also unsupported. Geometry for opportunity rank and distance metrics remain.

For CA-source training, `--prepared-events` skips static training and event preparation, using the bundled historical event features instead. This is only available for one CA-source direction at a time. Fresh static preparation uses the complete static vocabulary and does not reconstruct the older source-active embeddings. The raw-to-training path and frozen-input replay therefore answer different reproduction questions.

The lower-level commands are `python -m SHAPE train-grid`, `extract-grid`, `prepare-events`, `train-tokenizer`, `encode-source`, and `train-landing`. Append `--help` for stage-specific options. Three-seed NYC-to-CA wrappers use the same current training contract; the root runner covers all four directions. Historical scoring and verification do not load a tokenizer and remain available.
