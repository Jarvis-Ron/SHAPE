import copy
import hashlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from SHAPE.stage3.learned_landing import (
    LearnedLanding, SCHEMA, attraction_inputs, conditional_logits,
    teacher_forced_records, load_learned_landing,
)
from SHAPE.stage3.sampler import group_autoregressive_landing


def test_source_likelihood_updates_every_requested_parameter_group():
    rng = np.random.default_rng(42)
    raw, group = rng.random((9, 20)), rng.random((9, 20))
    category = rng.random((3, 13)); category /= category.sum(axis=1, keepdims=True)
    model = LearnedLanding([.2, .5, .8])
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    attraction = model.attraction_table(*attraction_inputs(raw, group, category))
    logits = conditional_logits(torch.tensor(rng.random((3, 9)), dtype=torch.float32),
        torch.tensor(rng.normal(size=(3, 9)), dtype=torch.float32),
        torch.tensor(rng.integers(0, 5, (3, 9)), dtype=torch.float32),
        attraction, model.strengths(), torch.zeros(3, 9, dtype=torch.long), torch.tensor([0, 4, 8]))
    F.cross_entropy(logits, torch.tensor([0, 4, 8])).backward()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.all(parameter.grad.abs() > 1e-9), name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.all(parameter.detach() != before[name]), name
    assert torch.all(model.strengths() > 0)
    assert 0 < model.export_values()['density_mix'] < 1
    np.testing.assert_allclose(attraction.detach().mean(1), 1., atol=1e-6)


def test_attraction_limits_match_legacy_residential_and_poi_paths():
    from SHAPE.stage3.context_memory import map_marginal_rescaled_attraction
    rng = np.random.default_rng(3)
    raw, group = rng.random((17, 20)), rng.random((17, 20))
    categories = rng.random((2, 13)); categories /= categories.sum(1, keepdims=True)
    roles = dict(role=['home_night_residential_proxy', 'ambiguous_general_poi_mixture'],
                 category_probabilities=categories, dominant_poi_category=[0, 1], table_sha256='test')
    legacy, _ = map_marginal_rescaled_attraction(roles, raw, group)
    model = LearnedLanding([.5, .5])
    with torch.no_grad():
        model.gate_logits.copy_(torch.tensor([50., -50.]))
        result = model.attraction_table(*attraction_inputs(raw, group, categories)).numpy()
    np.testing.assert_allclose(result, legacy, rtol=3e-6, atol=2e-6)


def test_training_memory_excludes_future_and_heldout_observations():
    ref = dict(day_offsets=np.arange(0, 11, 2), day_user_ids=np.array(['u']*5),
        day_dates=np.array(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05']),
        day_splits=np.array(['train','val','test','train','train']))
    grids = np.array([10,20, 30,40, 50,60, 10,20, 70,70])
    vocab = np.arange(10, 81, 10)
    records = teacher_forced_records(ref, grids, vocab)
    # Second training day remembers first train day, but neither val nor test.
    event, origin, target, memory, counts = records['train'][1]
    assert event == 7 and origin == 0 and target == 1
    np.testing.assert_array_equal(memory, [0, 1])
    np.testing.assert_array_equal(counts, [1, 1])
    # Current gold and future days never enter this step's memory.
    assert 1 not in records['train'][0][3]
    assert 6 not in memory
    changed = grids.copy(); changed[2:6] = [80,80,80,80]
    other = teacher_forced_records(ref, changed, vocab)
    for first, second in zip(records['train'], other['train']):
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


class FlatHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.))

    def encode_query(self, h):
        return torch.ones((len(h), 2), device=h.device)


def test_conditional_training_probabilities_match_unchanged_sampler_choices():
    # Two-event days give an exact sampler oracle: known uniform draws and the
    # same finite candidate distribution. Includes excluded coarse-ring grids.
    from SHAPE.stage3.rank_semantics import RANK_PIECEWISE_EDGES
    n, days = 8, 200
    ranks = np.tile(np.array([0., .2, .3, .6, .8, 1.5, 3., 8.], np.float32), (n, 1))
    fine = np.searchsorted(RANK_PIECEWISE_EDGES[1:-1], ranks[0], side='right')
    mass = np.array([.15,.25,.35,.25]+[0.]*12, dtype=np.float32)
    cardinality = np.bincount(fine, minlength=16)
    rank_density = np.log(mass[fine].clip(min=1e-12)) - np.log(cardinality[fine])
    attraction = np.array([[.2,.6,1.5,2.,1.,.3,1.,1.]], dtype=np.float32)
    strengths = torch.tensor([1.3, .7, 1.1])
    logits = conditional_logits(torch.ones(1, n), torch.tensor(rank_density[None], dtype=torch.float32),
        torch.tensor([[1.]+[0.]*(n-1)]), torch.from_numpy(attraction), strengths,
        torch.tensor(fine[None]//4), torch.tensor([0]))
    prob = logits.softmax(1).numpy()[0]
    uniforms = np.random.default_rng(42).random(days*2).astype(np.float32)[1::2]
    expected = np.searchsorted(np.cumsum(prob), uniforms, side='left')
    prediction, audit = group_autoregressive_landing(
        FlatHead(), np.ones((n, 2), np.float32), np.zeros((n,20),np.float32), {},
        np.ones((days*2,2),np.float32), np.arange(0,days*2+1,2), np.zeros(days,int), torch.device('cpu'),
        revisit_beta=float(strengths[0]), target_grid_ids=np.arange(n),
        centers=np.column_stack((np.zeros(n), np.arange(n)*.01)),
        code_distance_bin_probabilities=np.tile([1.,0,0,0],(days*2,1)),
        code_distance_piecewise_probabilities=np.tile(mass,(days*2,1)),
        physical_distance_piecewise_edges_km=RANK_PIECEWISE_EDGES,
        distance_kernel_tau=1/float(strengths[2]), distance_coordinate='spatial_rank',
        prepared_spatial_rank_tensor=torch.from_numpy(ranks),
        selection_mode='sample', selection_seed=42, selection_temperature=.75,
        code_attraction_table=attraction, event_attraction_code=np.zeros(days*2,int),
        attraction_mode='raw', attraction_strength=float(strengths[1]), attraction_epsilon=1e-6,
        hard_distance_ring_constraint=True, hard_distance_ring_seed=42,
        hard_distance_ring_empty_policy='expand_adjacent', return_audit=True)
    np.testing.assert_array_equal(prediction[1::2], expected)
    assert audit['hard_distance_ring_pass']
    assert audit['pareto_distance_cost'] is False


def test_checkpoint_roundtrip_and_wrong_baseline_rejection(tmp_path):
    paths = {k: tmp_path / k for k in ('source_codes','landing_checkpoint','source_grid')}
    for k, p in paths.items(): p.write_text(k)
    bindings = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    model = LearnedLanding([.2,.8])
    state = dict(schema=SCHEMA, source_city='nyc', codebook_size=2,
                 state_dict=model.state_dict(), bindings=bindings)
    path = tmp_path / 'learned.pth'; torch.save(state, path)
    restored, _ = load_learned_landing(path, **paths, source_city='nyc')
    assert model.export_values() == restored.export_values()
    paths['source_codes'].write_text('different model')
    with pytest.raises(ValueError, match='binding mismatch: source_codes'):
        load_learned_landing(path, **paths, source_city='nyc')


def test_generation_rejects_unhashed_learned_checkpoint():
    from SHAPE.reproduce import check_inputs
    with pytest.raises(ValueError, match='expected checksum'):
        check_inputs({'learned_landing_checkpoint': 'unknown', 'expected_sha256': {}})
