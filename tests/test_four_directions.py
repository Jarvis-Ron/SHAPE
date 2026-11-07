"""Protocol and reproducibility checks for the public four-direction wrapper."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from SHAPE.reproduce import generate, read_spec, summarize
from SHAPE.stage3.run import FIVE
from SHAPE.stage3.runtime import quota_probability
from run_full_pipeline import DIRECTIONS, ROOT, training_plan, verify_reference


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_original_metric_samples_recover_all_four_reference_rows(direction):
    result = verify_reference(direction)
    assert set(result) == set(FIVE)


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_new_generation_rejects_historical_source_codes_before_target_access(direction, tmp_path):
    spec = read_spec(ROOT / 'configs' / f'{direction}.json')
    # If generation touches the target reference before checking training
    # lineage, this nonexistent path causes the wrong exception.
    spec['target_reference'] = str(tmp_path / 'must-not-be-read.npz')
    destination = tmp_path / 'generation'
    with pytest.raises(ValueError, match='Retrain Stage 2'):
        generate(spec, destination, 'cpu')
    assert not destination.exists()


def test_landing_training_rejects_historical_supervised_codes(tmp_path):
    from SHAPE.stage3.landing_head import train_landing_head
    spec = read_spec(ROOT / 'configs' / 'nyc_to_ca.json')
    with pytest.raises(ValueError, match='Retrain Stage 2'):
        train_landing_head(spec['source_codes'], None, spec['source_grid'], tmp_path / 'landing.pth')
    assert not (tmp_path / 'landing.pth').exists()


def test_seed_variability_is_not_confused_with_shell_variability():
    # Large within-seed variation, equal means across seeds: seed SD is zero.
    from SHAPE.stage3.metrics import TEN_METRICS
    rows = [dict(seed=seed, metric=m, value=float(i % 2))
            for seed in range(42, 47) for m in TEN_METRICS for i in range(200)]
    summary, per_seed = summarize(rows, 'per_seed_shell', list(range(42, 47)))
    assert all(row['mean'] == .5 and row['std'] == 0 for row in summary)
    assert len(per_seed) == 5 * len(TEN_METRICS)
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(rows[:-1], 'per_seed_shell', list(range(42, 47)))


def test_quota_api_accepts_only_an_aligned_population_histogram():
    quota = dict(schema='target_train_home_quota_v1', grid_ids=[2, 4],
                 home_counts=[1, 3], provenance={})
    np.testing.assert_array_equal(quota_probability(quota, np.array([2, 4])), [.25, .75])
    for changed in [dict(quota, individual_trajectories=[]), dict(quota, home_counts=[1.5, 2.5]),
                    dict(quota, grid_ids=[4, 2]), dict(quota, home_counts=[0, 0])]:
        with pytest.raises(ValueError):
            quota_probability(changed, np.array([2, 4]))


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_fresh_training_fits_behavior_on_source_only(direction, tmp_path):
    spec = read_spec(ROOT / 'configs' / f'{direction}.json')
    commands, paths = training_plan(spec, tmp_path, 'cpu')
    assert [c[4] for c in commands] == ['train-grid', 'extract-grid', 'prepare-events',
                                     'train-tokenizer', 'encode-source', 'train-landing']
    for command in commands:
        for option in ('--city', '--source-city'):
            if option in command:
                assert command[command.index(option) + 1] == spec['source_city']
    assert set(paths) == {spec['source_city']}
    assert all(spec['target_reference'] not in command for command in commands)
    assert all('--injection-mode' not in command for command in commands)
