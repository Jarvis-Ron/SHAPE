# Running SHAPE

The supported four-direction entry point is `run_full_pipeline.py` (or the shell wrapper).

| Mode | Work performed | Requires training |
|---|---|---|
| `verify` | Verify fixed input hashes and recompute summaries of archived metric draws | No |
| `score` | Recompute metrics from the original generated trajectories | No |
| `reproduce` | Generate new trajectories from frozen Stage 3 inputs, then score | No |
| `train` | Train static encoder, prepare source events, train tokenizer and landing head, generate, score | Yes |

Run `python run_full_pipeline.py --help`. Four-direction configs resolve paths relative to their containing directory. Their SHA-256 bindings protect against accidentally mixing inputs. Fresh-training outputs receive new bindings. The full runner invokes stages synchronously and stops on failure; each training stage has its own log. Use a fresh output directory for each execution.

For CA-source training, `--prepared-events` skips static training and event preparation, using the bundled historical event features instead. This is only available for one CA-source direction at a time. Fresh static preparation uses the complete static vocabulary and does not reconstruct the older source-active embeddings. The raw-to-training path and frozen-input replay therefore answer different reproduction questions.

The lower-level commands are `python -m SHAPE train-grid`, `extract-grid`, `prepare-events`, `train-tokenizer`, `encode-source`, and `train-landing`. Append `--help` for stage-specific options. Historical three-seed NYC-to-CA wrappers remain for compatibility, but the new root runner must be used for the four-direction table.
