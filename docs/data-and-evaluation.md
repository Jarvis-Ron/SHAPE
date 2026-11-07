# Data and evaluation contracts

All four archived experiments used the earlier mechanism with layer semantic supervision and physical distance cost. Their recorded source cohorts, generation seeds, target references, scores and scoring units are retained. They are not results for the current model, which removes those mechanisms and requires new training/generation.

| Direction | Generation seeds | Scoring | Mean and sample SD (ddof=1) |
|---|---|---|---|
| NYC → CA | 42, 43, 44 | Pool seed population statistics before scoring; 200 matched shells; 8,870/8,915 target test days | Across 200 shell scores |
| NYC → Paris | 42–46 | 200 matched shells per seed; full eligible target test population | Across five within-seed shell means |
| CA → NYC | 42–46 | Direct source-population scoring against all 9,400 eligible NYC test days; no resampling | Across five generation-seed scores |
| CA → Paris | 42–46 | 200 matched shells per seed; full eligible target test population | Across five within-seed shell means |

The five primary metrics use days with at least two events. Archives also retain one-event days. The shell seed is 42031. Regenerated shell plans must equal the original arrays. Strict protocols fail if a target day length has no source donor. NYC → CA's original common-support protocol excludes 45 days containing 1,882 events and 1,837 transitions; it is not a full-CA-test result. Error bars are sample SDs, not confidence intervals or independent-training SDs.

## Metrics

| Metric | Definition |
|---|---|
| OD CPC | 100 × 100 coarse-zone transitions. Matched shells use equal-mass counts; direct CA → NYC independently normalizes both histograms before computing 2 sum(min(p,q))/(sum(p)+sum(q)). |
| Spatiotemporal JSD | Mean spatial JSD over four periods: 06–11, 11–17, 17–22, 22–06. |
| RoG JSD | Histogram of per-user radius of gyration, with 2-km bins to 100 km plus overflow. |
| Jump-length JSD | Histogram of consecutive within-day grid-center jumps, with the same bins. |
| POI-category JSD | Visit-weighted shares of 13 POI categories from repaired static group features. |

JSD uses base-2 logarithms. Shell evaluation assigns source donor days to target user/day slots. Shell RoG follows those reassigned users; direct evaluation retains the transplanted source-user grouping. These are distinct evaluation units. Shell-score CSVs retain auxiliary diagnostics.

## Information boundaries

- Static training uses map attributes. Behavioral tokenizer and landing-head supervision use source-city check-ins.
- Generated trajectories follow encoded observed source activities and timing; this release has no separate behavior-code generator.
- The three shell experiments initialize from target training-day modal homes for reference-eligible users. Only training grid/user/split arrays enter the home extractor; the sampler receives the aggregate marginal and static context. Eligibility uses existing train/test split membership.
- CA → NYC generation reads an exported aggregate home quota over all 1,083 NYC training users. It does not open the target reference during generation; evaluation reads it afterward.
- Current code-conditioned opportunity-rank/attraction semantics use source training data. Physical distance cost and target-distance calibration have been removed; the archived results had a source-fitted distance cost. Revisit memory uses generated past visits. Target test locations are scored after generation. Matched-shell evaluation separately uses target user/day structure.

Here, zero-shot denotes no target check-in training of the behavioral model. Target population marginals and evaluation shells are additional information and are reported explicitly.

## Artifact formats

`source_codes.npz` includes source train/validation/test event codes, reconstructed features, grid IDs, minutes, offsets, dates, users, split labels, and tokenizer binding. New exports additionally require `tokenizer_training_contract=residual_fsq_no_layer_semantics_v2`. `landing.pth` binds the learned query head to that source archive. The new generation path rejects bundled archives without that contract; historical scoring does not require it.

`target_reference.npz` contains exactly eight arrays: `grid_id`, `minute`, `day_offsets`, `day_user_ids`, `day_dates`, `day_splits`, `event_user_ids`, `event_splits`. Target tokenizer features are unnecessary.

`predictions.npz` maps `seed_<integer>` to the generated target-grid sequence, aligned with `source_test_real` after native user eligibility selection. That source payload supplies times and user/day boundaries. Configurations bind inputs by SHA-256. Metadata paths have been anonymized; numerical arrays and model tensors were checked against archived originals.
