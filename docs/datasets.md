# Included datasets

| Path | Role |
|---|---|
| `*_grid_merged_features.csv`, `*_grid_provenance.csv` | Complete static map vocabulary and POI/transport features for NYC, CA and Paris |
| `grid_spec.json`, `feature_spec.json`, producer receipts | Grid geometry and static preprocessing definitions |
| `checkin_NY_TY2014/` | Original NYC check-in file and provider readme |
| `Gowalla-CA/` | Supplied California check-in partitions |
| `prepared/nyc/`, `prepared/ca_to_nyc/`, `prepared/ca_to_paris/` | Source events, int64 grid IDs, user splits and codes from the released no-auxiliary-loss tokenizer |
| `reference/{nyc,ca,paris}.npz` | Processed calendar-aware reference events used by evaluation |
| `home_quotas/*.json` | Public population Home quotas derived from the GHS-POP R2023A 2010 residential population layer |
| `evaluation/*_users.npz`, `evaluation/*.json` | 200 complete-user selection plans and population/hash bindings |

Paris is distributed here as processed reference data, not as the full original
global Foursquare raw archive. Its provider readme remains in `dataset/`.
Source data are pseudonymous check-in records; consult the original providers'
terms and attribution in [THIRD_PARTY_DATA.md](../THIRD_PARTY_DATA.md).

Stage 2 training consumes prepared source-event embeddings. Stage 3 generation
consumes the corresponding source code archive, static maps, fitted weights
and the GHS-POP R2023A 2010 residential population Home quota. Target reference trajectories and user
selection plans are opened only by verification/evaluation, not generation.

Each configuration points to `dataset/home_quotas/<direction>.json` and binds
its SHA256. These files use `ghs_pop_home_quota_v1`, with `grid_ids` aligned to
the complete target static vocabulary and `home_counts` holding precomputed
integer allocation weights. Stage 3 normalizes the weights and uses Hamilton
allocation to obtain exact Home capacities for the source users being transferred.
`quota_total` is the sum of allocation weights, not a count of observed target
users or a raw city population total. The quota source is public residential
population data, not target training/test check-ins or the POI-based residential
context score used to match users to available Home slots.

The dataset provenance is supplied by the repository maintainer. Raw GHS-POP
rasters and the raster-to-grid preprocessing are not bundled; the verifier checks
the declared source, year, city, grid alignment, counts and file hashes, rather
than reconstructing the upstream population aggregation. See the
[quota README](../dataset/home_quotas/README.md) and
[data attribution](../THIRD_PARTY_DATA.md).

Prepared CA embeddings are direction-specific reused Stage 1 artifacts. The
two NYC transfers share the same prepared source events.
File checksums in each configuration bind these choices explicitly.
