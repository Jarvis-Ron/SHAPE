"""Public release bindings, source-only training, and population matching."""
from pathlib import Path

import numpy as np
import pytest

from SHAPE.pipeline import DIRECTIONS, ROOT, training_plan
from SHAPE.reproduce import check_inputs, read_spec
from SHAPE.evaluation.matched_users import summarize, validate_plans, user_hash, take_users
from SHAPE.stage3.runtime import quota_probability
from SHAPE.verify import verify


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_bundled_model_and_population_bindings(direction):
    report = verify(ROOT / 'configs' / f'{direction}.json')
    assert report['learned_landing_parameters'] == 68
    assert report['learned_context_parameters'] == 7
    assert report['users_per_side'] == (1070 if direction.startswith('nyc') else 1032)


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_generation_checks_do_not_read_target_reference_or_evaluation_plans(direction, tmp_path):
    spec = read_spec(ROOT / 'configs' / f'{direction}.json')
    for k in ('target_reference', 'user_selection', 'evaluation_metadata'):
        spec[k] = str(tmp_path / 'must-not-be-read')
    check_inputs(spec, generation=True)


@pytest.mark.parametrize('missing', ['learned_landing_checkpoint', 'learned_context_checkpoint'])
def test_missing_fitted_weights_never_silently_select_older_model(missing):
    spec = read_spec(ROOT / 'configs/nyc_to_paris.json')
    spec.pop(missing)
    with pytest.raises(ValueError, match='Latest SHAPE requires'):
        check_inputs(spec, generation=True)


def test_seed_sd_uses_five_means_and_never_pools_user_populations():
    draws = np.tile([0., 1.], (5, 100))
    result = summarize({'test': draws}, list(range(42,47)))['test']
    assert result['mean'] == .5 and result['seed_std'] == 0
    with pytest.raises(ValueError):
        summarize({'test': draws[:, :-1]}, list(range(42,47)))


def test_equal_user_plan_rejects_replacement_or_changed_population():
    su, tu = np.array(['a','b']), np.array(['c','d','e'])
    sp = np.tile([0,1], (200,1))
    tp = sp.copy()
    meta = dict(users_per_side=2, source_user_order_sha256=user_hash(su),
                target_user_order_sha256=user_hash(tu))
    validate_plans(su, tu, sp, tp, meta)
    with pytest.raises(ValueError, match='User order'):
        validate_plans(su[::-1], tu, sp, tp, meta)
    sp[0] = [0,0]
    with pytest.raises(ValueError, match='without replacement'):
        validate_plans(su, tu, sp, tp, meta)


def test_selecting_a_user_preserves_all_eligible_days_and_events():
    sub = dict(offsets=np.array([0,2,5,7]), users=np.array(['a','b','a']),
               minute=np.arange(7), values=np.arange(7)+10)
    selected = take_users(sub, np.array(['a','b']), np.array([0]))
    np.testing.assert_array_equal(selected['offsets'], [0,2,4])
    np.testing.assert_array_equal(selected['values'], [10,11,15,16])


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_training_graph_fits_both_learned_components_on_source_only(direction, tmp_path):
    spec = read_spec(ROOT / 'configs' / f'{direction}.json')
    commands, paths = training_plan(spec, tmp_path)
    assert [c[4] for c in commands] == ['train-tokenizer', 'encode-source', 'train-landing',
                                       'train-landing-weights', 'train-context']
    for command in commands:
        for option in ('--city', '--source-city'):
            if option in command:
                assert command[command.index(option)+1] == spec['source_city']
        assert spec['target_reference'] not in command
        assert spec['home_quota'] not in command
    for option in ('--jump-aux-weight','--poi-aux-weight','--time-aux-weight'):
        assert commands[0][commands[0].index(option)+1] == '0'
    assert set(paths) == {spec['source_city']}
    fresh, _ = training_plan(spec, tmp_path, fresh_grid=True)
    assert [c[4] for c in fresh[:3]] == ['train-grid','extract-grid','prepare-events']


def test_quota_api_accepts_only_an_aligned_population_histogram():
    quota = dict(schema='target_train_home_quota_v1', grid_ids=[2,4], home_counts=[1,3], provenance={})
    np.testing.assert_array_equal(quota_probability(quota, np.array([2,4])), [.25,.75])
    for changed in [dict(quota, individual_trajectories=[]), dict(quota, home_counts=[1.5,2.5]),
                    dict(quota, grid_ids=[4,2]), dict(quota, home_counts=[0,0])]:
        with pytest.raises(ValueError):
            quota_probability(changed, np.array([2,4]))
