"""Public release bindings, source-only training, and population matching."""
from pathlib import Path

import numpy as np
import pytest

from SHAPE.pipeline import DIRECTIONS, ROOT, main as pipeline_main, train, training_plan
from SHAPE.reproduce import MODEL_INPUTS, check_inputs, generate, read_spec
from SHAPE.evaluation.matched_users import summarize, validate_plans, user_hash, take_users
from SHAPE.stage3.runtime import quota_probability
from SHAPE.stage3.contracts import sha256_file
from SHAPE.verify import verify


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_bundled_data_and_population_bindings(direction):
    report = verify(ROOT / 'configs' / f'{direction}.json')
    assert report['scope'] == 'data'
    assert 'learned_landing_parameters' not in report
    assert report['users_per_side'] == (1070 if direction.startswith('nyc') else 1032)


def bind_model_files(spec, tmp_path):
    """Hash-only fixtures for input validation; model I/O has separate tests."""
    for key in MODEL_INPUTS:
        path = tmp_path / f'{key}.pth'
        path.write_bytes(key.encode())
        spec[key] = str(path)
        spec['expected_sha256'][key] = sha256_file(path)
    return spec


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_generation_checks_do_not_read_target_reference_or_evaluation_plans(direction, tmp_path):
    spec = read_spec(ROOT / 'configs' / f'{direction}.json', require_fitted=False)
    bind_model_files(spec, tmp_path)
    for k in ('target_reference', 'user_selection', 'evaluation_metadata'):
        spec[k] = str(tmp_path / 'must-not-be-read')
    check_inputs(spec, generation=True)


@pytest.mark.parametrize('missing', MODEL_INPUTS)
def test_missing_fitted_weights_never_silently_select_older_model(missing, tmp_path):
    spec = read_spec(ROOT / 'configs/nyc_to_paris.json', require_fitted=False)
    bind_model_files(spec, tmp_path)
    spec.pop(missing)
    with pytest.raises(ValueError, match='Latest SHAPE requires'):
        check_inputs(spec, generation=True)


def test_data_only_spec_rejects_generation_before_creating_output(tmp_path):
    path = ROOT / 'configs/nyc_to_paris.json'
    with pytest.raises(ValueError, match='Train first'):
        read_spec(path)
    spec = read_spec(path, require_fitted=False)
    output = tmp_path / 'generation'
    with pytest.raises(ValueError, match='Train first'):
        generate(spec, output, 'cpu')
    assert not output.exists()


def test_training_binds_new_model_files_for_generation(tmp_path, monkeypatch):
    spec = read_spec(ROOT / 'configs/nyc_to_paris.json', require_fitted=False)

    def fake_fit(command, **kwargs):
        stage = command[4]
        if stage == 'train-tokenizer':
            path = Path(command[command.index('--save-dir') + 1]) / 'tokenizer.pth'
        else:
            path = Path(command[command.index('--output') + 1])
            if stage == 'train-landing-weights':
                path /= 'learned_landing.pth'
            elif stage == 'train-context':
                path /= 'learned_context.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stage.encode())

    monkeypatch.setattr('SHAPE.pipeline.subprocess.run', fake_fit)
    fitted = train(spec, tmp_path / 'fit', 'cpu')
    reloaded = read_spec(tmp_path / 'fit/transfer.json')
    bindings = check_inputs(reloaded, generation=True)
    assert set(MODEL_INPUTS) <= set(bindings)
    assert reloaded == fitted
    assert bindings['source_codes'] != spec['expected_sha256']['source_codes']
    assert reloaded['source_events'] == spec['source_events']
    assert 'grid_checkpoint' not in reloaded

    generated = []
    monkeypatch.setattr('SHAPE.pipeline.generate', lambda spec, output, device, lanes:
                        generated.append(spec))
    pipeline_main(['--mode', 'generate', '--spec', str(tmp_path / 'fit/transfer.json'),
                   '--output', str(tmp_path / 'rerun')])
    assert generated == [fitted]


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
    spec = read_spec(ROOT / 'configs' / f'{direction}.json', require_fitted=False)
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
    quota = dict(schema='target_public_population_quota_v1', city='paris',
                 grid_ids=[2,4], population_counts=[1.5,4.5],
                 provenance=dict(source_kind='public_residential_population',
                                 dataset='test population', epoch=2010, target_checkins_used=False))
    np.testing.assert_array_equal(quota_probability(quota, np.array([2,4])), [.25,.75])
    for changed in [dict(quota, individual_trajectories=[]),
                    dict(quota, schema='target_train_home_quota_v1'),
                    dict(quota, grid_ids=[4,2]), dict(quota, population_counts=[0,0]),
                    dict(quota, population_counts=[-1,2]),
                    dict(quota, population_counts=[np.nan,2]),
                    dict(quota, provenance=dict(quota['provenance'], target_checkins_used=True))]:
        with pytest.raises(ValueError):
            quota_probability(changed, np.array([2,4]))
