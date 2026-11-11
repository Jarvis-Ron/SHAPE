# Public population Home quotas

These are precomputed Home-allocation quotas derived from the **GHS-POP R2023A
2010 residential population layer**, as confirmed by the repository maintainer.
They are public static population inputs, not home histograms inferred from
target-city check-ins. R2023A is the release and 2010 is the population epoch.

| File | Target city |
|---|---|
| `nyc_to_ca.json` | California |
| `nyc_to_paris.json` | Paris |
| `ca_to_nyc.json` | New York City |
| `ca_to_paris.json` | Paris |

The `ghs_pop_home_quota_v1` schema contains:

- `grid_ids`: ordered IDs of every cell in the target static grid vocabulary.
- `home_counts`: aligned, nonnegative integer allocation weights.
- `provenance`: dataset, release, population year/layer, target city and
  `quota_total` (the sum of weights), with `target_checkins_used: false`.

The counts are processed quota weights, not raw city population totals or
observed target-user counts. Stage 3 normalizes them to probabilities, then
uses Hamilton allocation to scale them to the source-user count. Static Home
contexts determine which source user occupies which slot while preserving
these capacities exactly. The POI-based residential context score is separate
from the public population prior.

`configs/<direction>.json` selects the corresponding file via `home_quota` and
binds its SHA256. `SHAPE.reproduce.generate` loads it, and
`SHAPE.stage3.runtime.prepare_transfer_runtime` validates alignment and uses it
for Home assignment. Missing or incompatible quotas fail explicitly; target
reference trajectories cannot supply a fallback. Runtime and generation
manifests record the quota path, hash and source metadata.

The provenance correction preserves the existing `grid_ids` and `home_counts`
arrays. The repository does not bundle the raw population rasters or the
raster-to-grid preprocessing, so verification covers the supplied quotas and
their declared provenance, not a reconstruction of that upstream processing.
Provider citation and links are in [THIRD_PARTY_DATA.md](../../THIRD_PARTY_DATA.md).
