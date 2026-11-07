import copy
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from SHAPE.pipeline import build_plan
from SHAPE.stage3.run import train_home_reference, config_id, summarize, FIVE
from SHAPE.stage3.tasks import FIXED_VARIANT, load_contract, make_x1_hra_tasks
from SHAPE.verify import verify

ROOT = Path(__file__).resolve().parents[1]


def test_every_retained_module_imports_without_deleted_repository():
    for path in (ROOT/'SHAPE').rglob('*.py'):
        name = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
        importlib.import_module(name)
    assert not (ROOT/'CIVIC').exists()


def test_historical_draws_recover_the_requested_row_and_sample_sd():
    row = verify(ROOT/'provenance/nyc_to_ca')
    assert list(row.values())[1:] == [
        '0.8330 ± 0.0054','0.4058 ± 0.0024','0.1777 ± 0.0032',
        '0.1248 ± 0.0013','0.0015 ± 0.0001',
    ]
    with (ROOT/'provenance/nyc_to_ca/metric_draws.csv').open() as handle:
        draws = [dict(row, value=float(row['value'])) for row in csv.DictReader(handle)]
    expected = json.loads((ROOT/'provenance/nyc_to_ca/result.json').read_text())
    for result in summarize(draws,'SHAPE'):
        assert result['mean'] == pytest.approx(expected['means'][result['metric']],abs=1e-14)
        assert result['std'] == pytest.approx(expected['stds'][result['metric']],abs=1e-14)
        assert result['draw_count'] == 200
        assert result['std_ddof'] == 1


def test_historical_verifier_rejects_missing_draws(tmp_path):
    for name in ('result.json','metric_draws.csv'):
        (tmp_path/name).write_bytes((ROOT/'provenance/nyc_to_ca'/name).read_bytes())
    path=tmp_path/'metric_draws.csv'
    lines=path.read_text().splitlines(True)
    path.write_text(''.join(lines[:-10]))
    with pytest.raises(ValueError,match='Missing or duplicate'):
        verify(tmp_path)


def test_target_test_values_cannot_cross_home_boundary():
    artifact=dict(grid_id=np.array([1,999,2]),event_user_ids=np.array(['a','a','b']),
                  event_splits=np.array(['train','test','train']),codes=np.ones((3,3)))
    before=train_home_reference(artifact)
    artifact['grid_id'][1]=-100
    after=train_home_reference(artifact)
    assert set(before)=={'grid_id','event_user_ids','event_splits'}
    for key in before:
        np.testing.assert_array_equal(before[key],after[key])


def test_release_refuses_other_mechanisms_and_changed_contract(tmp_path):
    assert config_id(FIXED_VARIANT)=='SHAPE'
    for key,value in [('weighted_rank',False),('lambda',.5),('kappa',1),('cost',True)]:
        with pytest.raises(ValueError):
            config_id(dict(FIXED_VARIANT,**{key:value}))
    contract=load_contract()
    contract['frozen_landing']['selection_temperature']=.9
    path=tmp_path/'changed.json';path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        load_contract(path)


def test_only_source_observed_codes_enter_three_sampling_lanes():
    tasks=make_x1_hra_tasks({'source_train_real':{},'source_test_real':{}},load_contract())
    assert [task['sampling_seed'] for task in tasks]==[42,43,44]
    assert all(task['generator_seed'] is None and task['scale_mode']=='rank' for task in tasks)
    with pytest.raises(ValueError):
        make_x1_hra_tasks({'target_test_real':{}},load_contract())


def test_training_graph_has_no_baseline_or_target_tokenizer_fit(tmp_path):
    spec=json.loads((ROOT/'configs/pipeline_nyc_to_ca.json').read_text())
    commands,run_spec=build_plan(spec,tmp_path)
    assert [command[4] for command in commands]==['train-tokenizer','encode-source','train-landing','run']
    train=commands[0]
    assert train[train.index('--city')+1]=='nyc'
    assert train[train.index('--encoder-perturb-sigma')+1]=='0.1'
    assert run_spec['target_reference']==spec['target_reference']
    assert '--configs' not in commands[-1]


def test_raw_pipeline_includes_all_three_stages(tmp_path):
    spec=json.loads((ROOT/'configs/pipeline_nyc_to_ca.json').read_text())
    spec['prepare_static_and_source_events']=True
    spec.pop('target_reference')
    commands,_=build_plan(spec,tmp_path)
    assert [command[4] for command in commands]==[
        'train-grid','extract-grid','prepare-events','train-tokenizer',
        'encode-source','train-landing','prepare-reference','run',
    ]


def test_pipeline_forwards_one_city_path_config_to_all_data_adapters(tmp_path):
    spec=json.loads((ROOT/'configs/pipeline_nyc_to_ca.json').read_text())
    spec['prepare_static_and_source_events']=True
    spec['city_paths_config']='configs/city_paths.example.json'
    spec.pop('target_reference')
    commands,_=build_plan(spec,tmp_path)
    for command in commands:
        if command[4] in ('prepare-events','encode-source','prepare-reference'):
            assert spec['city_paths_config'] in command
