import numpy as np
import pytest
from SHAPE.stage3.opportunity_rank import SpatialRankIndex,SpatialRankBinIndex
from SHAPE.stage3.rank_semantics import RANK_PIECEWISE_EDGES


@pytest.mark.parametrize('mode',['uniform','weighted','zeros','ties','small'])
def test_saturated_index_preserves_every_bin_including_ties_and_zero_mass(mode):
    rng=np.random.default_rng(8);n=19 if mode=='small' else 900
    centers=np.column_stack((rng.uniform(20,60,n),rng.uniform(-130,-70,n)))
    weights=None if mode=='uniform' else rng.integers(0,20,n).astype(float)
    if mode=='zeros':weights[:850]=0
    if mode=='ties':
        centers[:300]=centers[0];weights[:300]=1
    full=SpatialRankIndex(centers,weights,cache_rows=0)
    fast=SpatialRankBinIndex(centers,weights,cache_rows=2)
    for origin in [0,1,18,n-1]:
        a,b=full.row(origin),fast.row(origin)
        for edges in [RANK_PIECEWISE_EDGES,RANK_PIECEWISE_EDGES[[0,4,8,12,16]]]:
            np.testing.assert_array_equal(np.searchsorted(edges[1:-1],a,side='right'),np.searchsorted(edges[1:-1],b,side='right'))
        mask=a<250
        np.testing.assert_array_equal(a[mask],b[mask])
        np.testing.assert_array_equal(b,fast.row(origin))


def test_exact_edge_and_tail_both_enter_final_bin():
    centers=np.column_stack((np.zeros(600),np.arange(600)*.001))
    full=SpatialRankIndex(centers);fast=SpatialRankBinIndex(centers)
    a,b=full.row(0),fast.row(0)
    assert a[250]==250 and b[250]==250
    np.testing.assert_array_equal(np.searchsorted(RANK_PIECEWISE_EDGES[1:-1],a,side='right'),np.searchsorted(RANK_PIECEWISE_EDGES[1:-1],b,side='right'))


def test_seeded_multi_event_generation_is_identical_with_fast_bins():
    import torch
    from SHAPE.stage3.sampler import group_autoregressive_landing
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();self.logit_scale=torch.nn.Parameter(torch.tensor(1.))
        def encode_query(self,h):return h
    rng=np.random.default_rng(21);n=900;days=15;length=7;events=days*length
    centers=np.column_stack((rng.uniform(25,45,n),rng.uniform(110,125,n)))
    weights=rng.integers(0,4,n).astype(float);table=rng.random((n,4)).astype(np.float32)
    h=rng.random((events,4)).astype(np.float32)
    args=dict(revisit_beta=2.,target_grid_ids=np.arange(n),centers=centers,
        code_distance_bin_probabilities=np.tile([.2,.3,.3,.2],(events,1)),
        code_distance_piecewise_probabilities=np.tile(np.repeat([.2,.3,.3,.2],4)/4,(events,1)),
        physical_distance_piecewise_edges_km=RANK_PIECEWISE_EDGES,
        distance_coordinate='spatial_rank',distance_kernel_tau=1.3,
        selection_mode='sample',selection_seed=42,selection_temperature=.75,
        hard_distance_ring_constraint=True,hard_distance_ring_seed=42,hard_distance_ring_empty_policy='expand_adjacent')
    predictions=[]
    for index in [SpatialRankIndex(centers,weights),SpatialRankBinIndex(centers,weights)]:
        predictions.append(group_autoregressive_landing(Head(),table,np.zeros((n,20)),{},h,
            np.arange(0,events+1,length),np.arange(days),torch.device('cpu'),prepared_spatial_rank_tensor=index,**args))
    np.testing.assert_array_equal(*predictions)
