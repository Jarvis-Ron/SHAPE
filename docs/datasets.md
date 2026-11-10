# Included datasets

| Path | Role |
|---|---|
| `*_grid_merged_features.csv`, `*_grid_provenance.csv` | Complete static map vocabulary and POI/transport features for NYC, CA and Paris |
| `grid_spec.json`, `feature_spec.json`, producer receipts | Grid geometry and static preprocessing definitions |
| `checkin_NY_TY2014/` | Original NYC check-in file and provider readme |
| `Gowalla-CA/` | Supplied California check-in partitions |
| `prepared/nyc/`, `prepared/ca_to_nyc/`, `prepared/ca_to_paris/` | Source events, int64 grid IDs, user splits and codes from the released no-auxiliary-loss tokenizer |
| `reference/{nyc,ca,paris}.npz` | Processed calendar-aware reference events used by evaluation |
| `home_quotas/*.json` | Target training-home histograms supplied as inference conditions |
| `evaluation/*_users.npz`, `evaluation/*.json` | 200 complete-user selection plans and population/hash bindings |

Paris is distributed here as processed reference data, not as the full original
global Foursquare raw archive. Its provider readme remains in `dataset/`.
Source data are pseudonymous check-in records; consult the original providers'
terms and attribution in [THIRD_PARTY_DATA.md](../THIRD_PARTY_DATA.md).

Stage 2 training consumes prepared source-event embeddings. Stage 3 generation
consumes the corresponding source code archive, static maps, fitted weights
and aggregate target Home quota. Target reference trajectories and user
selection plans are opened only by verification/evaluation, not generation.

Prepared CA embeddings are direction-specific reused Stage 1 artifacts. The
two NYC transfers share the same prepared source events.
File checksums in each configuration bind these choices explicitly.
