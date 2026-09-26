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
    assert config['jump_aux_weight']==.125
    assert config['poi_aux_weight']==.075
    assert config['encoder_perturb_sigma']==.1
    assert config['max_attention_cells']==8388608
    with pytest.raises(SystemExit):
        parse_args(['--quantizer-type','ema'])
    with pytest.raises(SystemExit):
        parse_args(['--jump-aux-weight','0'])


def test_codes_export_and_source_landing_training_are_connected(tmp_path, monkeypatch):
    torch.manual_seed(42)
    grid_path=ROOT/'dataset/NYC_grid_merged_features.csv'
    grid=StaticGrid.load(grid_path,city='nyc')
    ids=grid.grid_ids[:3]
    calendar={}
    sequences={}
    metadata=[]
    for user in ('u0','u1','u2'):
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
        {'train':{'u0'},'val':{'u1'},'test':{'u2'}},data,{'city':'nyc'})
    config=dict(MODULE2_CONFIG,city='nyc',d_model=8,nhead=2,
                num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=16,dropout=0.0)
    tokenizer=build_tokenizer(config)
    checkpoint=tmp_path/'tokenizer.pth'
    torch.save({'config':config,'model_state_dict':tokenizer.state_dict()},checkpoint)
    # Calendar recovery is isolated from real private check-ins; all export,
    # serialization, feature validation and landing training below are real.
    monkeypatch.setattr(exporter,'load_raw_calendar',lambda *a,**kw:calendar)
    codes=tmp_path/'source_codes.npz'
    receipt=exporter.prepare_codes(checkpoint,data,None,codes,torch.device('cpu'),
        batch_size=1,source_city='nyc',grid_csv=grid_path,
        expected_jump_aux_weight=.125,expected_poi_aux_weight=.075,
        expected_time_aux_weight=0,require_no_time_aux=True)
    archive=load_npz(codes)
    assert receipt['event_count']==54
    assert archive['codes'].shape==(54,3)
    assert archive['reconstructed_h'].shape==(54,64)
    assert archive['grid_id'].dtype==np.int64
    assert set(archive['event_splits'])=={'train','val','test'}
    landing=tmp_path/'landing.pth'
    history=train_landing_head(codes,None,grid_path,landing,
        epochs=1,batch_size=16,mode='l2_attributes',normalization_grid_csvs=[grid_path])
    assert history['source_city']=='nyc'
    assert history['codes_sha256']==sha256_file(codes)
    assert history['best_epoch']==1
    runtime=load_landing_runtime(landing,None,grid_path,torch.device('cpu'),
                                 vocabulary_policy='all_emitted')
    assert runtime is not None
