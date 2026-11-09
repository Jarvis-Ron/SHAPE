import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from SHAPE.stage3.learned_context import LearnedContext, context_inputs, memory_records, jsd, load_context, SCHEMA
from SHAPE.stage3.learned_landing import teacher_forced_records
from SHAPE.stage3.context_memory import _context_cost, context_quota_home_assignment
from SHAPE.stage3.sampler import group_autoregressive_landing


def test_home_metric_updates_all_three_context_parameters_and_matches_runtime_cost():
    rng=np.random.default_rng(12)
    raw=rng.uniform(.01,3,(110,20));group=rng.uniform(.01,1,(110,20))
    centers=np.column_stack((np.linspace(30,31,110),rng.uniform(120,121,110)))
    inp=context_inputs(raw,group,centers,np.arange(8))
    model=LearnedContext();before=copy.deepcopy(model.state_dict())
    ctx=model.context(inp)
    logits=-model.cost(ctx[:4],ctx[4:])/.1
    F.cross_entropy(logits,torch.tensor([3,0,2,1])).backward()
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    for name in ('home_density_logit','neighborhood_logit','context_mix_logit'):
        g=getattr(model,name).grad
        assert g is not None and torch.isfinite(g) and abs(g)>1e-7
    optimizer.step()
    for name in ('home_density_logit','neighborhood_logit','context_mix_logit'):
        assert not torch.equal(before[name],model.state_dict()[name])
    with torch.no_grad():
        context=model.context(inp);cost=model.cost(context[:1],context[1:]).numpy()[0];v=model.export_values()
    expected=_context_cost(context[0].numpy(),context[1:].numpy(),(v['residential_cost_weight'],v['poi_cost_weight']))
    np.testing.assert_allclose(cost,expected,atol=1e-7)


def test_quantile_values_and_both_parameter_gradients():
    model=LearnedContext().double()
    # Check a differentiable interval, away from empirical-quantile knots.
    with torch.no_grad():model.quantile_logits.add_(.037)
    pool=torch.stack((torch.linspace(0,1,201),torch.linspace(0,2,201)**2),1).double()
    low,high=model.quantiles(pool)
    np.testing.assert_allclose(torch.stack((low,high)).detach(),np.quantile(pool.numpy(),model.levels().detach().numpy(),axis=0),rtol=1e-6,atol=1e-7)
    x=torch.tensor([[.3,.9],[.7,2.1]],dtype=torch.float64)
    loss=(model.normalize(x,pool)*torch.tensor([[1.,2.],[3.,1.]])).sum()
    loss.backward();gradient=model.quantile_logits.grad.clone()
    assert torch.all(gradient.abs()>1e-7)
    for j in range(2):
        delta=1e-5
        with torch.no_grad():
            model.quantile_logits[j]+=delta
            plus=(model.normalize(x,pool)*torch.tensor([[1.,2.],[3.,1.]])).sum()
            model.quantile_logits[j]-=2*delta
            minus=(model.normalize(x,pool)*torch.tensor([[1.,2.],[3.,1.]])).sum()
            model.quantile_logits[j]+=delta
        assert abs(float((plus-minus)/(2*delta)-gradient[j]))<1e-4


def test_count_records_equal_v1_at_initialization_and_ignore_heldout_history():
    ref=dict(day_offsets=np.arange(0,11,2),day_user_ids=np.array(['u']*5),
        day_dates=np.array(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05']),
        day_splits=np.array(['train','val','test','train','train']))
    grids=np.array([10,20,30,40,50,60,10,20,70,70]);ids=np.arange(10,81,10)
    old=teacher_forced_records(ref,grids,ids);new=memory_records(ref,grids,ids)
    for split in old:
        for a,b in zip(old[split],new[split]):
            for x,y in zip(a[:4],b[:4]):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(a[4],b[4]+b[5]+b[6])
    changed=grids.copy();changed[2:6]=80
    other=memory_records(ref,changed,ids)
    for a,b in zip(new['train'],other['train']):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


class FlatHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.logit_scale=torch.nn.Parameter(torch.tensor(0.))
    def encode_query(self,h):return torch.ones((len(h),2))


def test_learned_initial_counts_match_sampler_probabilities_without_duplicate_home_boost():
    n=4;days=400;home,anchor=2.3,.4
    count=torch.tensor([[home,anchor,0.,0.]])
    logits=torch.log1p(count)/.75;prob=logits.softmax(1).numpy()[0]
    u=np.random.default_rng(42).random(days*2).astype(np.float32)[1::2]
    expected=np.searchsorted(np.cumsum(prob),u)
    pred=group_autoregressive_landing(FlatHead(),np.ones((n,2),np.float32),np.zeros((n,20)),{},
        np.ones((days*2,2),np.float32),np.arange(0,days*2+1,2),np.zeros(days,int),torch.device('cpu'),
        revisit_beta=1.,target_grid_ids=np.arange(n),centers=np.column_stack((np.zeros(n),np.arange(n)*.001)),
        candidate_top_k=n,selection_mode='sample',selection_seed=42,selection_temperature=.75,
        code_distance_bin_probabilities=np.tile([1.,0,0,0],(days*2,1)),
        distance_coordinate='spatial_rank',prepared_spatial_rank_tensor=torch.zeros(n,n),
        history_seed_grid_ids=np.tile([0,1,1],(days,1)),initial_home_count=home,initial_anchor_count=anchor)
    np.testing.assert_array_equal(pred[1::2],expected)
    model=LearnedContext();counts=model.counts()
    F.cross_entropy(torch.log1p(torch.stack((counts[0],counts[1],counts.new_tensor(0.)))[None]),torch.tensor([1])).backward()
    assert torch.all(model.raw_counts.grad.abs()>1e-6)


def test_context_checkpoint_rejects_changed_binding(tmp_path):
    from SHAPE.stage3.contracts import sha256_file
    paths={k:tmp_path/k for k in ['source_codes','landing_checkpoint','source_grid','learned_landing_checkpoint']}
    for p in paths.values():p.write_text('bound')
    m=LearnedContext();path=tmp_path/'model.pth'
    torch.save(dict(schema=SCHEMA,source_city='nyc',bindings={k:sha256_file(v) for k,v in paths.items()},state_dict=m.state_dict()),path)
    loaded,_=load_context(path,**paths,source_city='nyc')
    assert loaded.export_values()==m.export_values()
    paths['source_grid'].write_text('changed')
    with pytest.raises(ValueError,match='binding'):load_context(path,**paths,source_city='nyc')


def test_context_weights_keep_exact_home_quotas():
    source=np.array([[.2,.8,.2]+[0.]*11,[.8,.2,.8]+[0.]*11])
    target=source[::-1].copy()
    assignments,audit=context_quota_home_assignment(np.array([0,1,0,1]),source,target,np.array([.5,.5]),42,
        np.array([0,1]),np.array([10,20]),context_weights=(.3,1.7))
    np.testing.assert_array_equal(np.bincount(assignments//10)[1:],[2,2]);assert audit['capacity_exact']
