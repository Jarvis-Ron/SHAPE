"""Population mass, geometry and information-boundary regressions."""
import json
from pathlib import Path

import numpy as np
import pytest

from SHAPE.population import SCHEMA, population_probabilities, rectangle_population
from SHAPE.stage3.resident_index import proportional_integer_capacities
from SHAPE.stage3.context_memory import context_quota_home_assignment
from SHAPE.stage3.runtime import prepare_transfer_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_only_positive_capacity_cells_need_contexts():
    rng = np.random.default_rng(7)
    source_context = rng.uniform(size=(5, 14)).astype(np.float32)
    target_context = rng.uniform(size=(100, 14)).astype(np.float32)
    source_context[:, 1:] /= source_context[:, 1:].sum(axis=1, keepdims=True)
    target_context[:, 1:] /= target_context[:, 1:].sum(axis=1, keepdims=True)
    probability = rng.uniform(size=100)
    probability /= probability.sum()
    capacities = proportional_integer_capacities(probability, 5)
    sparse_context = target_context.copy()
    sparse_context[capacities == 0] = 0
    for seed in range(42, 47):
        full = context_quota_home_assignment(np.arange(5), source_context,
            target_context, probability, seed, np.arange(5), np.arange(100))
        sparse = context_quota_home_assignment(np.arange(5), source_context,
            sparse_context, probability, seed, np.arange(5), np.arange(100))
        np.testing.assert_array_equal(sparse[0], full[0])
        assert sparse[1] == full[1]


def fixture_quota():
    return dict(schema=SCHEMA, city='paris', grid_ids=[2, 4], population_counts=[0.25, 1.75],
                provenance=dict(source_kind='public_residential_population', dataset='test',
                                epoch=2010, target_checkins_used=False, grid_sha256='expected'))


def test_overlap_apportionment_preserves_mass_and_fractional_boundary_pixels():
    # Two 1-degree pixels in each of two rows, with known population totals.
    counts = np.array([[10., 20.], [30., 40.]])
    transform = (1., 0., 0., 0., -1., 2.)
    bounds = [[0, 2, 0, .5], [0, 2, .5, 1.5], [0, 2, 1.5, 2]]
    result = rectangle_population(counts, transform, bounds)
    np.testing.assert_allclose(result, [20, 50, 30], rtol=1e-12)
    assert result.sum() == counts.sum()
    # Splitting at a latitude inside a pixel uses surface area, not half of its count.
    horizontal = rectangle_population(counts, transform, [[0, .5, 0, 2], [.5, 2, 0, 2]])
    expected_south = 70*np.sin(np.radians(.5))/np.sin(np.radians(1.))
    np.testing.assert_allclose(horizontal, [expected_south, 100-expected_south], rtol=1e-12)


def test_population_projection_rejects_invalid_values_or_missing_coverage():
    transform = (1., 0., 0., 0., -1., 2.)
    for counts in (np.array([[-1., 0.], [2., 3.]]), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            rectangle_population(counts, transform, [[0, 2, 0, 2]])
    with pytest.raises(ValueError, match='cover'):
        rectangle_population(np.ones((2, 2)), transform, [[0, 2, -1, 2]])


def test_float_population_weights_keep_exact_user_totals_and_zero_mass_exclusions():
    q = fixture_quota()
    p = population_probabilities(q, [2, 4], city='paris', grid_sha256='expected')
    np.testing.assert_allclose(p, [.125, .875])
    for total in [1, 17, 1070]:
        counts = proportional_integer_capacities(np.r_[p, 0.], total)
        assert counts.sum() == total and counts[-1] == 0
        assert np.max(np.abs(counts-total*np.r_[p, 0.])) < 1.
    q['population_counts'] = [250., 1750.]
    np.testing.assert_array_equal(population_probabilities(q), p)


def test_city_grid_identity_and_training_quota_cannot_be_silently_substituted():
    q = fixture_quota()
    for kwargs in [dict(city='nyc'), dict(grid_sha256='other'), dict(vocabulary=[2, 5])]:
        with pytest.raises(ValueError):
            population_probabilities(q, **kwargs)
    with pytest.raises(ValueError, match='public'):
        population_probabilities(dict(schema='target_train_home_quota_v1', grid_ids=[2, 4],
                                      home_counts=[1, 7], provenance={}))
    with pytest.raises(ValueError, match='required'):
        prepare_transfer_runtime({}, {}, {}, None, None, 'unused', ['cpu'])
    with pytest.raises(ValueError, match='rejects target checkins'):
        prepare_transfer_runtime({}, {}, {}, {'grid_id': [2]}, None, 'unused', ['cpu'],
                                 target_home_quota=q, personal_scale={})


def test_four_configs_share_target_city_weights_and_report_excluded_population():
    paris_paths = set()
    for path in sorted((ROOT/'configs').glob('*.json')):
        spec = json.loads(path.read_text())
        quota_path = (path.parent/spec['home_quota']).resolve()
        q = json.loads(quota_path.read_text())
        p = population_probabilities(q, city=spec['target_city'])
        provenance = q['provenance']
        assert provenance['epoch'] == 2010
        assert provenance['target_test_used'] is False
        np.testing.assert_allclose(p.sum(), 1., rtol=1e-12)
        np.testing.assert_allclose(sum(q['population_counts']), provenance['retained_population'])
        np.testing.assert_allclose(provenance['retained_population']+provenance['excluded_population'],
                                   provenance['bbox_population'])
        assert 0 < provenance['retained_population_fraction'] <= 1
        if spec['target_city'] == 'paris':
            paris_paths.add(quota_path)
    assert len(paris_paths) == 1
