# Code map

| Code | Purpose |
|---|---|
| `SHAPE/stage1/` | Static feature alignment, shared grid encoder, embedding extraction |
| `SHAPE/stage2/` | Source event preparation and no-auxiliary-loss residual FSQ training |
| `SHAPE/stage3/prepare_codes.py` | Calendar-aware source encoding |
| `SHAPE/stage3/landing_head.py` | Neural query/landing head |
| `SHAPE/stage3/learned_landing.py` | Source-fitted landing coefficients and residential gates |
| `SHAPE/stage3/home_quota.py` | Load and validate GHS-POP R2023A 2010 residential population Home quotas |
| `SHAPE/stage3/learned_context.py` | Source-fitted Home context, initial counts and quantiles |
| `SHAPE/stage3/runtime.py`, `sampler.py`, `context_memory.py` | Complete model runtime and autoregressive target cell sampling |
| `SHAPE/stage3/opportunity_rank.py` | Exact opportunity-rank geometry and accelerated bins |
| `SHAPE/reproduce.py` | Bound configuration loading and five-seed generation |
| `SHAPE/evaluation/matched_users.py` | Four-metric evaluation with equal complete-user budgets |
| `SHAPE/verify.py` | Dataset, checkpoint and evaluation-population verification |
| `SHAPE/pipeline.py` | Synchronous source training, generation and evaluation commands |

Some internal schema and task identifiers retain historical names for artifact
compatibility. They are not additional comparison models. Shared feature
utilities in `evaluation/prototype_atlas.py` are required by source encoding;
the file contains no atlas experiment runner.
