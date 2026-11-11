# Evaluation protocol

The released protocol is `equal_complete_users_v1` for all four directions.
Generation first produces every eligible source user's transferred schedule for
each of seeds 42–46. Target test users are not used to construct those schedules.
Home allocation uses the public GHS-POP R2023A 2010 residential population
quotas in `dataset/home_quotas/`. Target check-ins enter only verification and
evaluation; the equal-user evaluation budget does not set the generation quota.

Evaluation retains days with at least two events. Eligible users have at least
one such day. For each direction, the user budget is the smaller of the
synthetic and target-test user populations: 1,070 for each NYC-source direction
and 1,032 for each CA-source direction.

Each of 200 fixed draws samples that many users without replacement from each
side and retains all their eligible days. These exact plans are bundled and
reused across seeds. Users may occur in multiple draws; the draws are repeated
subsamples, not independent populations. Day, event and jump counts can differ
between synthetic and real samples despite equal user counts.

For each draw and each seed, the evaluator computes four metrics:

| Metric | Preferred direction | Interpretation |
|---|---|---|
| OD CPC | Higher | Agreement of normalized OD distributions |
| Spatiotemporal JSD | Lower | Discrepancy of period-specific spatial distributions |
| RoG JSD | Lower | Discrepancy of user radius-of-gyration distributions |
| Jump-length JSD | Lower | Discrepancy of consecutive-event displacement distributions |

OD matrices are normalized to unit mass before CPC, so differing numbers of
transitions are not interpreted as a difference in total flow volume. Binning
and spatial aggregation remain those of the original metric implementation.

The 200 scores are averaged **within each seed**. The reported mean and sample
standard deviation (`ddof=1`) are computed over the five seed means. Seeds are
never pooled into five times as many users. This SD describes generation
variability for a fixed fitted model and evaluation plan, not uncertainty over
independently trained models or unseen cities.

Output files contain per-seed means, all draws and input hashes. No copied
target daily shells, target-event-count matching, or pooled seed populations
are used by this evaluator.
