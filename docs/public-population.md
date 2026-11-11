# Public population as the home prior

All four directions use the **2010 epoch of GHS-POP R2023A**, published by the
European Commission Joint Research Centre. Its residential population counts
are derived from census/administrative population and GHSL built-up information,
not the SHAPE target check-ins. The selected WGS84 product has 3-arcsecond cells
(approximately 100 m at the equator). It is an estimate, not a register of
individual homes. [Official product description](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php).

The year was fixed before measuring new trajectory scores: the original
Gowalla data cover 2009–2010, and NYC/Paris Foursquare data cover 2012–2013.
The common 2010 layer is close to those observation periods and avoids using
a current 2025/2030 population distribution for historical trajectories.
R2023A is a retrospective product; this is not a claim that this exact raster
was available in 2010, or a temporally causal forecasting experiment.

## Conversion to the SHAPE grids

1. Read the city bounding box from `dataset/grid_spec.json`. Download only its
   window from the official JRC cloud-optimized GeoTIFF. No check-ins or user
   selection files are inputs to preprocessing.
2. Intersect every existing emitted static grid with the population pixels.
   Apportion each pixel's **population count** by the intersecting spherical
   rectangle area, assuming uniform population inside the input pixel. This
   preserves mass across a complete partition; bilinear density interpolation
   is not used. Nodata is masked and counted in the audit.
3. Retain only the already published static-grid vocabulary. Report the
   population excluded by its gaps; do not move that mass to a nearby cell or
   fill gaps using target visits. Normalize the retained counts to sum to one.
4. For N synthetic users, apportion N times those probabilities with Hamilton's
   largest-remainder rule. Source-learned home-context matching decides which
   user occupies each available slot. The final quotas sum to exactly N.

Fractional population estimates are kept until user-capacity apportionment.
The population total does not set the number of synthetic users. The same
Paris prior is used for NYC→Paris and CA→Paris; different user budgets can
produce different integer capacities.

## Geographic coverage

These totals describe **the configured study bounding boxes**, not official
municipal/state census totals. CA is the broad California study region, not
only Los Angeles. The final prior is conditional on the existing emitted-grid
support.

| Target | Existing grids | Grids with positive population | Population retained within the study bounding box |
|---|---:|---:|---:|
| NYC | 7,376 | 7,015 | 99.0621% |
| CA | 105,640 | 96,856 | 88.7739% |
| Paris | 709 | 709 | 99.9858% |

The excluded share is material for CA (11.2261%). It is recorded explicitly in
the data rather than hidden by renormalization. Expanding the static vocabulary
would be a separate model/data change. Residential population also need not
follow the spatial distribution of Foursquare/Gowalla users, so this substitution
does not guarantee better trajectory metrics. Previous check-in-quota results
are not measurements of this revision.

## Files and validation

`dataset/population_quotas/{nyc,ca,paris}.json` contain sorted grid IDs,
fractional `population_counts` and provenance under
`target_public_population_quota_v1`. Provenance records the dataset DOI,
epoch, license, remote object ETag, crop checksum, static-grid checksum,
aggregation rule, retained/excluded mass and information-use flags.
Each direction's config binds its population file by SHA-256. The runtime
rejects old training-home schemas, mismatched grids/cities, negative/nonfinite
counts and target check-in inputs. New run manifests report
`target_checkin_generation_use: none` and no target training statistics.

To rebuild the three population files (network access is required only here):

```bash
python -m pip install -e '.[population]'
python -m SHAPE prepare-population --city nyc --year 2010 \
  --grid dataset/NYC_grid_merged_features.csv --cache data_cache/population \
  --output dataset/population_quotas/nyc.json
python -m SHAPE prepare-population --city ca --year 2010 \
  --grid dataset/CA_grid_merged_features.csv --cache data_cache/population \
  --output dataset/population_quotas/ca.json
python -m SHAPE prepare-population --city paris --year 2010 \
  --grid dataset/PARIS_grid_merged_features.csv --cache data_cache/population \
  --output dataset/population_quotas/paris.json
```

Bundled files need no rasterio or network access for normal generation.
Raster compression/library changes can change crop bytes and thus provenance
hashes; deliberate rebuilds require rebinding the config SHA-256 after review.
Population values and mass audits should agree within floating-point precision.

Existing source-fitted weights can be reused. To migrate a saved training spec,
copy `home_quota`, its `expected_sha256.home_quota`, and `home_prior` from the
matching current config into a **new** spec; resolve the quota path relative to
that new spec. Preserve all source/model hashes. The old spec is intentionally
rejected so historical generation cannot silently use check-in quotas.

Source-learned environment matching, daily home initialization and home memory
pseudocounts remain unchanged. Home therefore remains both an initial location
and a memory anchor. The new input removes the target check-in-derived home
distribution; it does not make generation unconditional.

## Attribution

Schiavina M., Freire S., Carioli A., MacManus K. (2023).
*GHS-POP R2023A – GHS population grid multitemporal (1975–2030).* European
Commission, Joint Research Centre.
DOI: [10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE).
The [JRC-managed AWS distribution](https://registry.opendata.aws/ghs-pop/)
lists CC BY 4.0. Derived files record this license and the spatial aggregation.
