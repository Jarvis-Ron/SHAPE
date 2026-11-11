import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from SHAPE.stage3.run import config_id
from SHAPE.stage3.runtime import prepare_transfer_runtime
from SHAPE.stage3.tasks import FIXED_VARIANT, load_contract, make_x1_hra_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_every_retained_module_imports():
    for path in (ROOT/'SHAPE').rglob('*.py'):
        importlib.import_module('.'.join(path.relative_to(ROOT).with_suffix('').parts))


def test_stage3_requires_public_quota_without_target_trajectory_fallback():
    with pytest.raises(ValueError, match='requires a GHS-POP home quota'):
        prepare_transfer_runtime({}, {}, {}, None, None, None, ['cpu'],
                                 personal_scale=dict(FIXED_VARIANT))
    for reference, cohort in (({}, None), (None, {}), ({}, {})):
        with pytest.raises(ValueError, match='rejects target checkins/cohort'):
            prepare_transfer_runtime({}, {}, {}, reference, cohort, None, ['cpu'],
                                     personal_scale=dict(FIXED_VARIANT), target_home_quota={})
    with pytest.raises(ValueError, match='fixed opportunity-rank'):
        prepare_transfer_runtime({}, {}, {}, None, None, None, ['cpu'],
                                 target_home_quota={})


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


def test_only_source_observed_codes_enter_five_sampling_lanes():
    tasks=make_x1_hra_tasks({'source_train_real':{},'source_test_real':{}},load_contract())
    assert [t['sampling_seed'] for t in tasks]==[42,43,44,45,46]
    assert all(t['generator_seed'] is None and t['scale_mode']=='rank' for t in tasks)
    with pytest.raises(ValueError):
        make_x1_hra_tasks({'target_test_real':{}},load_contract())


def test_public_encoder_has_no_depth_ablation_controls(capsys):
    from SHAPE.stage3.prepare_codes import parse_args
    for option in ('--analysis-only-variable-depth', '--pad-analysis-codes-to-three'):
        with pytest.raises(SystemExit):
            parse_args([option])
        assert 'unrecognized arguments' in capsys.readouterr().err
