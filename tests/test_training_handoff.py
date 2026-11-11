"""Exercise actual checkpoint -> source archive -> trained landing head I/O."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from SHAPE.gridspec import StaticGrid
from SHAPE.stage2.config import MODULE2_CONFIG
from SHAPE.stage2.prepare_city_event_data import REPO_ROOT, _absolute, load_city_paths, CITY_CONFIG
from SHAPE.stage2.sparse_event_data import save_sparse_event_dataset
from SHAPE.stage2.tokenizer import build_tokenizer
from SHAPE.stage2.trainer import train_vqvae
from torch.utils.data import DataLoader
from SHAPE.stage2.train_tokenizer import parse_args, build_training_config
from SHAPE.stage3 import prepare_codes as exporter
from SHAPE.stage3.contracts import load_npz, sha256_file
from SHAPE.stage3.landing_head import train_landing_head, load_landing_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_raw_paths_resolve_inside_repository_and_overrides_merge(tmp_path, monkeypatch):
    assert REPO_ROOT == ROOT
    assert _absolute('dataset/grid_spec.json').is_file()
    monkeypatch.setitem(CITY_CONFIG, 'nyc', dict(CITY_CONFIG['nyc']))
    path=tmp_path/'paths.json'
    path.write_text(json.dumps({'nyc':{'checkin_file':'new-checkins.txt'}}))
    load_city_paths(path)
    assert CITY_CONFIG['nyc']['checkin_file']=='new-checkins.txt'
    assert CITY_CONFIG['nyc']['grid_csv']=='dataset/NYC_grid_merged_features.csv'


def test_public_training_cli_only_accepts_fixed_mechanism():
    config=build_training_config(parse_args([]))
    assert config['quantizer_types']==['fsq']*3
    assert config['jump_aux_weight']==0
    assert config['poi_aux_weight']==0
    assert config['encoder_perturb_sigma']==.1
    assert config['max_attention_cells']==8388608
    for option in ('--injection-mode', '--semantic-loss-weight', '--semantic-warm-start-steps'):
        with pytest.raises(SystemExit):
            parse_args([option, '0'])
    with pytest.raises(SystemExit):
        parse_args(['--quantizer-type','ema'])
    for option, value in (('--jump-aux-weight', '.125'), ('--poi-aux-weight', '.075')):
        with pytest.raises(SystemExit):
            parse_args([option, value])


def test_codes_export_and_source_landing_training_are_connected(tmp_path, monkeypatch):
    torch.manual_seed(42)
    grid_path=ROOT/'dataset/NYC_grid_merged_features.csv'
    grid=StaticGrid.load(grid_path,city='nyc')
    ids=grid.grid_ids[:3]
    calendar={}
    sequences={}
    metadata=[]
    for user in ('u0','u1','u2','u3'):
        instants=[dt.datetime(2020,1,1)+dt.timedelta(days=day,hours=8+event)
                  for day in range(6) for event in range(3)]
        calendar[user]=instants
        x=np.zeros((len(instants),136),dtype=np.float32)
        x[:,17]=[value.weekday()/7 for value in instants]
        x[:,18]=[value.hour for value in instants]
        x[:,19:83]=np.random.default_rng(42).normal(size=(len(instants),64))
        x[:,123]=np.tile(ids,6)
        x[:,124]=6
        sequences[user]=x
        metadata.append(dict(user_id=user,length=len(instants)))
    data=tmp_path/'events'
    save_sparse_event_dataset(sequences,metadata,
        {'train':{'u0','u3'},'val':{'u1'},'test':{'u2'}},data,{'city':'nyc'})
    config=dict(MODULE2_CONFIG,city='nyc',d_model=8,nhead=2,
                num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=16,dropout=0.0,
                epochs=1,patience=3,fsq_min_active_codes=1,fsq_min_perplexity=1,
                fsq_require_all_buckets=False,
                save_dir=str(tmp_path),model_name='tokenizer.pth')
    tokenizer=build_tokenizer(config)
    assert tokenizer.jump_head is None and tokenizer.poi_head is None
    checkpoint=tmp_path/'tokenizer.pth'
    def loader(user):
        return DataLoader([dict(x=torch.from_numpy(sequences[user]),
            attention_mask=torch.ones(len(sequences[user]),dtype=torch.bool))],batch_size=1)
    _, tokenizer_history=train_vqvae(tokenizer,loader('u0'),loader('u1'),config=config)
    assert len(tokenizer_history['train_loss'])==1
    assert 'train_semantic' not in tokenizer_history
    assert 'val_semantic' not in tokenizer_history
    for name in ('train_jump_aux', 'val_jump_aux', 'train_poi_aux', 'val_poi_aux'):
        assert tokenizer_history[name] == [0.0]
    # Calendar recovery is isolated from real private check-ins; all export,
    # serialization, feature validation and landing training below are real.
    monkeypatch.setattr(exporter,'load_raw_calendar',lambda *a,**kw:calendar)
    codes=tmp_path/'source_codes.npz'
    receipt=exporter.prepare_codes(checkpoint,data,None,codes,torch.device('cpu'),
        batch_size=1,source_city='nyc',grid_csv=grid_path,
        expected_jump_aux_weight=0,expected_poi_aux_weight=0,
        expected_time_aux_weight=0,require_no_time_aux=True)
    archive=load_npz(codes)
    assert receipt['event_count']==72
    assert archive['codes'].shape==(72,3)
    assert archive['reconstructed_h'].shape==(72,64)
    assert archive['grid_id'].dtype==np.int64
    assert set(archive['event_splits'])=={'train','val','test'}
    assert str(archive['tokenizer_training_contract'])==config['tokenizer_training_contract']
    # The separate evaluation-only event exporter calls the quantizer directly
    # too; exercise it so a removed semantic-loss return cannot break it.
    from SHAPE.stage3.prepare_landing_events import prepare_landing_events
    events=tmp_path/'landing_events.npz'
    prepare_landing_events(checkpoint,data,'nyc',events,batch_size=1)
    evaluated=load_npz(events)
    assert evaluated['reconstructed_h'].shape==(18,64)
    assert np.isfinite(evaluated['reconstructed_h']).all()
    assert str(evaluated['tokenizer_training_contract'])==config['tokenizer_training_contract']
    landing=tmp_path/'landing.pth'
    history=train_landing_head(codes,None,grid_path,landing,
        epochs=1,batch_size=16,mode='l2_attributes',normalization_grid_csvs=[grid_path])
    assert history['source_city']=='nyc'
    assert history['codes_sha256']==sha256_file(codes)
    assert history['best_epoch']==1
    runtime=load_landing_runtime(landing,None,grid_path,torch.device('cpu'),
                                 vocabulary_policy='all_emitted')
    assert runtime is not None
    # Exercise the production Stage-3 boundary on the new tokenizer/landing pair.
    # Use the actual bundled public-population quota and static Paris map.
    from SHAPE.stage3.source_cohort import load_g0_cohort
    from SHAPE.stage3.runtime import build_source_payloads, prepare_transfer_runtime
    from SHAPE.stage3.context_memory import _run_e2_tasks
    from SHAPE.stage3.tasks import FIXED_VARIANT, load_contract, make_x1_hra_tasks
    from SHAPE.stage3.home_quota import load_home_quota
    from SHAPE.stage3.resident_index import _unique_user_values, proportional_integer_capacities
    cohort=load_g0_cohort(codes,{},None,require_complete_user_splits=True,reference_only=True)
    payloads=build_source_payloads(cohort)
    contract=load_contract()
    params=contract['frozen_landing']
    target_path=ROOT/'dataset/PARIS_grid_merged_features.csv'
    target=StaticGrid.load(target_path,city='paris')
    quota=load_home_quota(ROOT/'dataset/home_quotas/nyc_to_paris.json', target_city='paris')
    source=dict(reference_codes=codes,checkpoint=landing,grid_csv=grid_path,
        binding={},arms={'D1A1_raw':{'parameters':{key:params[key] for key in
            ('selection_mode','selection_temperature','distance_mode','distance_tau',
             'attraction_mode','attraction_strength')}}})
    transferred=prepare_transfer_runtime(source,dict(selected_beta=1.,binding={}),cohort,
        None,None,target_path,['cpu'],source_city='nyc',target_city='paris',
        personal_scale=dict(FIXED_VARIANT),target_home_quota=quota,sampling_seeds=(42,))
    audit=transferred['provenance']
    assert audit['pareto_distance_cost'] is False
    assert audit['target_train_distance_quantiles_used_by_distance_scale'] is False
    assert audit['target_train_statistics']==[]
    assert audit['target_checkins_used_by_generation'] is False
    assert audit['target_home']['provenance']==quota['provenance']
    probability=np.asarray(quota['home_counts'],dtype=float)
    probability/=probability.sum()
    np.testing.assert_array_equal(audit['target_home_probability'],probability)
    users, homes=_unique_user_values(cohort['users'],transferred['source_context_homes'][42])
    realized=np.bincount(np.searchsorted(target.grid_ids,homes),minlength=len(target.grid_ids))
    np.testing.assert_array_equal(realized,proportional_integer_capacities(probability,len(users)))
    tasks=make_x1_hra_tasks(payloads,contract)[:1]
    results=_run_e2_tasks(tasks,payloads,transferred,stage_label='rank-only-fixture',
        **{key:params[key] for key in ('landing_temperature','candidate_top_k',
            'candidate_omitted_mass_tolerance','candidate_audit_steps')},
        sequence_batch_size=2,landing_event_batch_size=16)
    prediction=results[0]['prediction']
    assert len(prediction)==len(payloads['source_test_real']['features'])
    assert np.isin(prediction,target.grid_ids).all()
