import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from SHAPE.stage3.opportunity_rank import (
    SpatialRankIndex, distance_row_km, opportunity_weights,
    fit_personal_profiles, profile_event_parameters, tilt_ring_probabilities,
)
from SHAPE.stage3.sampler import spatial_rank_table, group_autoregressive_landing
class _FlatLanding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
    def encode_query(self, h):
        return torch.ones((len(h), 2), dtype=torch.float32, device=h.device)



def test_uniform_weights_and_lazy_rows_equal_original_rank():
    centers = np.array([[0, 0], [0, .01], [0, -.01], [0, .02]])
    expected = np.stack([np.searchsorted(np.sort(distance_row_km(c, centers)),
                                         distance_row_km(c, centers), side='left') for c in centers])
    actual = spatial_rank_table(centers, np.ones(4))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(SpatialRankIndex(centers).dense(), expected)


def test_weighted_rank_excludes_equal_distance_and_retains_zero_mass_grids():
    centers = np.array([[0, 0], [0, .01], [0, -.01], [0, .02]])
    weights = opportunity_weights([2, 0, 6, 8])
    index = SpatialRankIndex(centers, weights)
    np.testing.assert_array_equal(index.row(0), [0, .5, .5, 2.0])
    assert index.shape == (4, 4)
    assert weights.mean() == 1


def fixture():
    grids = np.array(([0,1,0]*8)+([0,1,2]*8)+([3,3]*3))
    data = {'day_offsets': np.array([0,24,48,54]),
            'day_splits': np.array(['train','train','test']),
            'day_user_ids': np.array(['a','b','a']),
            'day_dates': np.array(['2020-01-01','2020-01-01','2020-01-02'])}
    centers = np.array([[0,0],[0,.01],[0,.02],[0,.2]])
    return data, grids, centers


def test_profiles_ignore_test_and_shrink_sparse_residents():
    data, grids, centers = fixture()
    first = fit_personal_profiles(data, grids, centers, np.arange(4), rank_index=SpatialRankIndex(centers))
    changed = grids.copy(); changed[48:] = [0,3,0,3,0,3]
    second = fit_personal_profiles(data, changed, centers, np.arange(4), rank_index=SpatialRankIndex(centers))
    assert first == second
    assert first['residents']['a']['ring_count'] == 23
    assert first['residents']['a']['revisit_observations'] == 14
    pop = np.asarray(first['population_ring_probability'])
    for row in first['residents'].values():
        expected = (np.asarray(row['ring_counts']) + 20*pop)/(row['ring_count']+20)
        np.testing.assert_allclose(row['shrunk_ring_probability'], expected)


def test_zero_exponents_and_missing_users_are_exactly_neutral():
    data, grids, centers = fixture()
    profile = fit_personal_profiles(data, grids, centers, np.arange(4), rank_index=SpatialRankIndex(centers))
    multiplier, beta = profile_event_parameters(profile, ['a','missing'], [0,3,7], 0, 0)
    np.testing.assert_array_equal(multiplier, np.ones((7,4)))
    np.testing.assert_array_equal(beta, np.ones(2))
    p = np.tile(np.array([.1,.2,.3,.4],dtype=np.float32), (7,1))
    np.testing.assert_array_equal(tilt_ring_probabilities(p,multiplier),p)
    multiplier, beta = profile_event_parameters(profile, ['missing'], [0,2], 1.5, 1.5)
    np.testing.assert_array_equal(multiplier,np.ones((2,4)))
    np.testing.assert_array_equal(beta,[1])


def test_profile_tilt_preserves_zero_probability_and_normalization():
    out = tilt_ring_probabilities([[.5,0,.25,.25]], [[2,100,1,.5]])
    np.testing.assert_allclose(out,[[1/1.375,0,.25/1.375,.125/1.375]])
    assert out[0,1] == 0


def test_large_rank_index_has_bounded_rows_not_square_storage():
    centers = np.column_stack((np.linspace(30,40,100000),np.zeros(100000)))
    index = SpatialRankIndex(centers, cache_rows=2)
    for origin in [0,1,2,3]: index.row(origin)
    assert len(index.cache) == 2
    assert sum(v.nbytes for v in index.cache.values()) == 2*100000*4


def test_parallel_landing_lanes_share_a_bounded_rank_cache_safely():
    centers=np.column_stack((np.linspace(30,40,50),np.zeros(50)))
    reference=SpatialRankIndex(centers).dense()
    shared=SpatialRankIndex(centers,cache_rows=2)
    origins=list(range(50))*10
    with ThreadPoolExecutor(8) as pool:
        for origin,row in zip(origins,pool.map(shared.row,origins)):
            np.testing.assert_array_equal(row,reference[origin])
    assert len(shared.cache)<=2


def test_full_landing_neutral_profiles_and_lazy_rank_preserve_predictions():
    centers=np.array([[0,0],[0,.01],[0,-.01],[0,.02],[0,.04]],dtype=np.float32)
    probability=np.tile(np.array([.1,.4,.3,.2],dtype=np.float32),(8,1))
    args=(_FlatLanding(),np.ones((5,2),dtype=np.float32),np.zeros((5,20),dtype=np.float32),
          {'lower':[0.]*20,'upper':[1.]*20},np.ones((8,2),dtype=np.float32),
          np.array([0,4,8]),np.array([0,1]),torch.device('cpu'))
    kwargs=dict(revisit_beta=1.,target_grid_ids=np.arange(5),centers=centers,
                code_distance_bin_probabilities=probability,distance_coordinate='spatial_rank',
                rank_bin_edges=(0,1,5,20,np.inf),
                selection_mode='sample',selection_seed=42,hard_distance_ring_constraint=True,
                hard_distance_ring_empty_policy='expand_adjacent',return_audit=True)
    original,audit=group_autoregressive_landing(*args,**kwargs,
        prepared_spatial_rank_tensor=torch.from_numpy(spatial_rank_table(centers)))
    neutral,check=group_autoregressive_landing(*args,**kwargs,
        prepared_spatial_rank_tensor=SpatialRankIndex(centers,np.ones(5)),
        personal_ring_multiplier=np.ones((8,4)),sequence_revisit_beta=np.ones(2))
    np.testing.assert_array_equal(neutral,original)
    assert audit['hard_distance_ring_pass'] == check['hard_distance_ring_pass'] is True


def test_invalid_static_weights_fail_closed():
    with pytest.raises(ValueError): opportunity_weights([0,0])
    with pytest.raises(ValueError): opportunity_weights([1,-1])


def test_sampling_is_invariant_to_distance_scale_with_the_same_opportunity_order():
    centers=np.array([[0,0],[0,.2],[0,.02]],dtype=np.float32)
    args=(_FlatLanding(),np.ones((3,2),dtype=np.float32),np.zeros((3,20),dtype=np.float32),
          {'lower':[0.]*20,'upper':[1.]*20},np.ones((8,2),dtype=np.float32),
          np.array([0,4,8]),np.array([0,0]),torch.device('cpu'))
    kwargs=dict(revisit_beta=0.,target_grid_ids=np.arange(3),
        code_distance_bin_probabilities=np.tile([0.,1.,0.,0.],(8,1)),
        distance_coordinate='spatial_rank',rank_bin_edges=(0,1,5,20,np.inf),
        selection_mode='sample',selection_seed=42,hard_distance_ring_constraint=True,
        hard_distance_ring_empty_policy='expand_adjacent',return_audit=True)
    baseline,audit=group_autoregressive_landing(*args,**kwargs,centers=centers,
        prepared_spatial_rank_tensor=SpatialRankIndex(centers))
    scaled,check=group_autoregressive_landing(*args,**kwargs,centers=centers*10,
        prepared_spatial_rank_tensor=SpatialRankIndex(centers*10))
    np.testing.assert_array_equal(scaled, baseline)
    assert audit['hard_distance_ring_pass'] and check['hard_distance_ring_pass']
    assert audit['physical_distance_correction'] is False
    with pytest.raises(ValueError,match="Only spatial_rank"):
        group_autoregressive_landing(*args,**dict(kwargs,distance_coordinate='physical_km'),centers=centers)
    for option in ('distance_cost_gamma','distance_cost_xmin_km','code_target_distance_km'):
        with pytest.raises(TypeError):
            group_autoregressive_landing(*args,**kwargs,centers=centers,**{option:1.0})


def test_source_profiles_need_no_positive_physical_jumps():
    data, grids, centers = fixture()
    grids[:] = 0
    profile = fit_personal_profiles(data, grids, centers, np.arange(4),
                                   rank_index=SpatialRankIndex(centers))
    assert profile['population_ring_probability'] == [1.,0.,0.,0.]
    assert 'gamma' not in profile and 'cost_xmin_km' not in profile


def test_personal_tilt_selects_different_rings_for_two_residents():
    centers=np.array([[0,0],[0,.01],[0,.02]],dtype=np.float32)
    args=(_FlatLanding(),np.ones((3,2),dtype=np.float32),np.zeros((3,20),dtype=np.float32),
          {'lower':[0.]*20,'upper':[1.]*20},np.ones((8,2),dtype=np.float32),
          np.array([0,4,8]),np.array([0,0]),torch.device('cpu'))
    multiplier=np.array([[1,0,1,1]]*4+[[0,1,1,1]]*4,dtype=np.float32)
    prediction,audit=group_autoregressive_landing(*args,revisit_beta=0.,
        target_grid_ids=np.arange(3),centers=centers,
        code_distance_bin_probabilities=np.tile([.5,.5,0.,0.],(8,1)),
        distance_coordinate='spatial_rank',rank_bin_edges=(0,1,5,20,np.inf),
        selection_mode='sample',selection_seed=42,hard_distance_ring_constraint=True,
        hard_distance_ring_empty_policy='expand_adjacent',return_audit=True,
        personal_ring_multiplier=multiplier,prepared_spatial_rank_tensor=SpatialRankIndex(centers))
    assert prediction[:4].tolist()==[0]*4
    assert np.all(np.diff(prediction[4:])!=0)
    assert audit['hard_distance_ring_pass'] is True
