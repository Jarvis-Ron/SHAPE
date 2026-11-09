"""Source-trained home contexts, memory priors and static quantile levels.

Home metric learning uses source home-cell behavior profiles, with disjoint
home cells for validation. Landing calibration uses causal source histories.
No target trajectories are accepted by either fit objective.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = 'shape_learned_context_v2'


def logit(x):
    return math.log(x/(1-x))


def jsd(a, b):
    """Broadcast-compatible JSD, natural logs."""
    a, b = a.clamp(min=1e-12), b.clamp(min=1e-12)
    m = (a+b)*.5
    return .5*((a*(a/m).log()).sum(-1)+(b*(b/m).log()).sum(-1))


class LearnedContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.home_density_logit = nn.Parameter(torch.tensor(logit(.625)))
        self.neighborhood_logit = nn.Parameter(torch.tensor(logit(24/99)))
        self.context_mix_logit = nn.Parameter(torch.tensor(0.))
        self.raw_counts = nn.Parameter(torch.log(torch.expm1(torch.ones(2))))
        self.quantile_logits = nn.Parameter(torch.tensor([logit(.1),logit(.9)]))

    def levels(self):
        p=self.quantile_logits.sigmoid()
        return torch.stack((.5*p[0], .5+.5*p[1]))

    def counts(self):
        return F.softplus(self.raw_counts)

    def quantiles(self, sorted_pool):
        z=self.levels()*(len(sorted_pool)-1)
        left=z.floor().long();right=(left+1).clamp(max=len(sorted_pool)-1)
        fraction=(z-left).unsqueeze(1)
        bounds=sorted_pool[left]*(1-fraction)+sorted_pool[right]*fraction
        low,high=bounds.unbind()
        high=torch.where(high-low<1e-6,low+1,high)
        return low,high

    def normalize(self, raw_attributes, sorted_pool):
        low,high=self.quantiles(sorted_pool)
        return ((raw_attributes-low)/(high-low)).clamp(0,1)

    def context(self, inputs):
        density,diversity,probe_density,probe_diversity,near_poi=inputs
        a=self.home_density_logit.sigmoid()
        score=a*density+(1-a)*diversity
        probes=a*probe_density+(1-a)*probe_diversity
        residential=((score[:,None]-probes[None,:])/.05).sigmoid().mean(1)
        k=1+99*self.neighborhood_logit.sigmoid()
        ranks=torch.arange(1,near_poi.shape[1]+1,device=near_poi.device)
        weights=((k+.5-ranks)/2).sigmoid()
        poi=(near_poi*weights[None,:,None]).sum(1)+1e-6
        poi=poi/poi.sum(1,keepdim=True)
        return torch.cat((residential[:,None],poi),dim=1)

    def cost(self, left, right):
        mix=self.context_mix_logit.sigmoid()
        return 2*mix*(left[:,None,0]-right[None,:,0]).abs()+2*(1-mix)*jsd(left[:,None,1:],right[None,:,1:])

    def export_values(self):
        a=float(self.context_mix_logit.sigmoid().detach().cpu())
        counts=self.counts().detach().cpu().tolist()
        return dict(home_density_mix=float(self.home_density_logit.sigmoid().detach().cpu()),
            effective_neighborhood=float((1+99*self.neighborhood_logit.sigmoid()).detach().cpu()),
            residential_cost_weight=2*a,poi_cost_weight=2*(1-a),
            home_pseudocount=counts[0],anchor_pseudocount=counts[1],
            quantile_levels=self.levels().detach().cpu().tolist())


def context_inputs(raw,group,centers,required,device='cpu'):
    from scipy.spatial import cKDTree
    raw=np.maximum(np.asarray(raw,dtype=np.float64),0)
    group=np.asarray(group,dtype=np.float64);required=np.asarray(required,dtype=np.int64)
    p=group[:,:13]**2;p/=p.sum(1,keepdims=True).clip(min=1e-12)
    diversity=-(p*np.log(p.clip(min=1e-12))).sum(1)/math.log(13)
    radians=np.radians(centers);lat,lon=radians.T
    sphere=np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    _,nearest=cKDTree(sphere).query(sphere[required],k=min(100,len(raw)))
    if nearest.ndim==1:nearest=nearest[:,None]
    probes=np.linspace(0,len(raw)-1,min(1024,len(raw)),dtype=np.int64)
    return tuple(torch.as_tensor(x,dtype=torch.float32,device=device) for x in
        (group[required,13],diversity[required],group[probes,13],diversity[probes],raw[nearest,:13]))


def source_homes(reference,projected):
    totals=defaultdict(Counter)
    for u,s,a,b in zip(reference['day_user_ids'],reference['day_splits'],reference['day_offsets'][:-1],reference['day_offsets'][1:]):
        if s=='train':totals[str(u)].update(projected[a:b].tolist())
    return {u:min(c,key=lambda g:(-c[g],g)) for u,c in totals.items()},totals


def memory_records(reference,projected,ids):
    """Affine memory = observed visits + c_home*home + c_anchor*anchor.

    Home wins on duplicate initial-location/anchor entries. As in V1, the
    observed first event initializes source teacher forcing and is not scored.
    Generation initializes at its assigned home. Only past train days seed anchors.
    """
    from SHAPE.stage3.context_memory import _memory_anchor_ids
    homes,totals=source_homes(reference,projected)
    offsets=reference['day_offsets'];users=reference['day_user_ids'];splits=reference['day_splits']
    positions=np.searchsorted(ids,projected);states={};records={'train':[],'val':[]}
    for i in sorted(range(len(users)),key=lambda i:(str(users[i]),str(reference['day_dates'][i]))):
        split=str(splits[i]);u=str(users[i])
        if split not in records:continue
        a,b=map(int,offsets[i:i+2])
        if b<=a:continue
        home=int(positions[a]);current=home
        anchors={int(np.searchsorted(ids,g)) for g in _memory_anchor_ids(states.get(u),len(totals[u]))}-{home}
        visits=Counter()
        for e in range(a+1,b):
            keys=np.array(sorted(set(visits)|anchors|{home}),dtype=np.int32)
            records[split].append((e,current,int(positions[e]),keys,
                np.array([visits[k] for k in keys],np.float32),
                (keys==home).astype(np.float32),np.array([k in anchors for k in keys],np.float32)))
            current=int(positions[e]);visits[current]+=1
        if split=='train':
            s=states.setdefault(u,dict(counts=Counter(),last={},clock=0))
            for g in projected[a:b]:
                g=int(g);s['clock']+=1;s['counts'][g]+=1;s['last'][g]=s['clock']
            retain=_memory_anchor_ids(s,len(totals[u]))
            s['counts']=Counter({g:s['counts'][g] for g in retain});s['last']={g:s['last'][g] for g in retain}
    return records


def pack_memory(records,reference,device):
    from SHAPE.stage3.learned_landing import _pack_records
    data=_pack_records(records,reference,device)
    for col,name in [(5,'home'),(6,'anchor')]:
        x=np.zeros(data['memory_counts'].shape,np.float32)
        for i,r in enumerate(records):x[i,:len(r[col])]=r[col]
        data[name]=torch.as_tensor(x,device=device)
    return data


def fit_home(model,reference,projected,ids,raw,group,centers,device):
    homes,_=source_homes(reference,projected)
    hids=np.array(sorted(set(homes.values())),np.int64);hp={g:i for i,g in enumerate(hids)}
    hist=np.zeros((len(hids),64),np.float32)
    for u,s,a,b in zip(reference['day_user_ids'],reference['day_splits'],reference['day_offsets'][:-1],reference['day_offsets'][1:]):
        if s=='train':hist[hp[homes[str(u)]]]+=np.bincount(reference['codes'][a:b,0],minlength=64)
    hist+=1e-6;hist/=hist.sum(1,keepdims=True)
    inputs=context_inputs(raw,group,centers,np.searchsorted(ids,hids),device)
    labels=torch.as_tensor(hist,device=device)
    order=np.random.default_rng(42).permutation(len(hids));n=max(1,int(.2*len(hids)))
    val=torch.as_tensor(order[:n],device=device);train=torch.as_tensor(order[n:],device=device)
    if len(train)<2:raise ValueError('insufficient independent source home cells')
    params=[model.home_density_logit,model.neighborhood_logit,model.context_mix_logit]
    optimizer=torch.optim.Adam(params,lr=.03);history=[];best=math.inf;stale=0;grads={}
    def objective(rows,training):
        total=0.
        for idx in rows.split(128):
            with torch.set_grad_enabled(training):
                context=model.context(inputs)
                logits=-model.cost(context[idx],context[train])/.1
                same=idx[:,None]==train[None,:]
                logits=logits.masked_fill(same,-1e9)
                with torch.no_grad():
                    target=(-jsd(labels[idx,None,:],labels[None,train,:])/.1).masked_fill(same,-1e9).softmax(1)
                loss=-(target*logits.log_softmax(1)).sum(1).mean()
                if training:
                    optimizer.zero_grad();loss.backward()
                    for name in ('home_density_logit','neighborhood_logit','context_mix_logit'):
                        g=getattr(model,name).grad
                        assert g is not None and torch.isfinite(g).all(),name
                        grads[name]=max(grads.get(name,0),float(g.abs().max().cpu()))
                    optimizer.step()
                total+=float(loss.detach())*len(idx)
        return total/len(rows)
    for epoch in range(101):
        if epoch:objective(train,True)
        loss=objective(val,False)
        history.append(dict(epoch=epoch,val_loss=loss,values=model.export_values()))
        if loss<best-1e-6:best=loss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if epoch%10==0:print('Home fit',epoch,loss,flush=True)
        if stale>=10:break
    model.load_state_dict(best_state)
    return dict(best_epoch=best_epoch,history=history,gradient_max=grads,train_home_cells=len(train),val_home_cells=len(val),
        label_source='source train code histogram; validation home cells disjoint',
        home_ids_sha256=__import__('hashlib').sha256(hids.tobytes()).hexdigest())


def load_context(path,*,source_codes,landing_checkpoint,source_grid,learned_landing_checkpoint,source_city):
    from SHAPE.stage3.contracts import sha256_file
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('schema')!=SCHEMA or state['source_city']!=source_city:raise ValueError('context schema/city mismatch')
    for k,p in dict(source_codes=source_codes,landing_checkpoint=landing_checkpoint,source_grid=source_grid,
                    learned_landing_checkpoint=learned_landing_checkpoint).items():
        if sha256_file(p)!=state['bindings'][k]:raise ValueError('context binding mismatch: '+k)
    model=LearnedContext();model.load_state_dict(state['state_dict'],strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite context weights')
    return model.eval(),state


def fit(codes,landing_checkpoint,source_grid,learned_landing_checkpoint,pooled_grids,output,device='cpu'):
    from SHAPE.gridspec import StaticGrid
    from SHAPE.stage3.contracts import load_npz,require_current_tokenizer_training,sha256_file,write_json
    from SHAPE.stage3.grid_projection import _project_nominal_ids_to_vocabulary
    from SHAPE.stage3.group_index import repair_group_index,load_v4_transport_provenance
    from SHAPE.stage3.landing_head import load_landing_runtime,load_canonical_grid_attributes
    from SHAPE.stage3.opportunity_rank import SpatialRankIndex,static_poi_weights
    from SHAPE.stage3.prototype_semantics import fit_train_code_attraction_semantics
    from SHAPE.stage3.rank_semantics import fit_train_rank_distance_semantics,RANK_PIECEWISE_EDGES
    from SHAPE.stage3.static_features import _load_raw_table
    from SHAPE.stage3.learned_landing import load_learned_landing,attraction_inputs,conditional_logits
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(1);device=torch.device(device)
    ref=load_npz(codes);require_current_tokenizer_training(ref);city=str(ref['source_city'].item())
    head,_,_,centers,ids,_,frozen=load_landing_runtime(landing_checkpoint,None,source_grid,device,vocabulary_policy='all_emitted')
    head.requires_grad_(False)
    scorer,_=load_learned_landing(learned_landing_checkpoint,source_codes=codes,landing_checkpoint=landing_checkpoint,source_grid=source_grid,source_city=city)
    scorer=scorer.to(device).requires_grad_(False)
    static=StaticGrid.load(source_grid,city=city)
    projected,_=_project_nominal_ids_to_vocabulary(ref['grid_id'],ids,static.spec)
    raw,_=_load_raw_table(source_grid);group,_,_=repair_group_index(raw,city,transport_provenance=load_v4_transport_provenance(source_grid,ids))
    weights,_=static_poi_weights(source_grid,ids);ranks=SpatialRankIndex(centers,weights,cache_rows=8192)
    semantics=fit_train_rank_distance_semantics(ref,projected,centers,ids,rank_index=ranks,weights=weights)
    roles=fit_train_code_attraction_semantics(ref,projected,raw,ids,residential_grid_table=group)
    model=LearnedContext().to(device);initial=model.export_values()
    home_report=fit_home(model,ref,projected,ids,raw,group,centers,device)
    for p in (model.home_density_logit,model.neighborhood_logit,model.context_mix_logit):p.requires_grad_(False)
    pool=np.concatenate([load_canonical_grid_attributes(g)[0] for g in pooled_grids])
    if sorted(map(sha256_file,pooled_grids))!=sorted(frozen['normalization']['fit_grid_csv_sha256']):
        raise ValueError('pooled static maps changed from original landing head')
    sorted_pool=torch.as_tensor(np.sort(pool,axis=0).copy(),device=device)
    attributes=torch.as_tensor(load_canonical_grid_attributes(source_grid)[0],device=device)
    print('Building causal count records',flush=True)
    records=memory_records(ref,projected,ids);data={s:pack_memory(rows,ref,device) for s,rows in records.items()};del records
    origins=np.unique(np.concatenate([d['origin'].cpu().numpy() for d in data.values()]))
    print('Exact rank rows',len(origins),'destinations',len(ids),flush=True)
    bins=np.stack([np.searchsorted(RANK_PIECEWISE_EDGES[1:-1],ranks.row(int(i)),side='right') for i in origins]).astype(np.uint8)
    count=np.stack([np.bincount(b,minlength=16) for b in bins]);fine=torch.as_tensor(bins,device=device)
    lookup=torch.as_tensor(origins,device=device);log_count=torch.as_tensor(np.log(count.clip(min=1)),dtype=torch.float32,device=device)
    log_probability=torch.as_tensor(np.log(semantics['piecewise_probabilities'].clip(min=1e-12)),device=device)
    with torch.no_grad():
        h=ref['reconstructed_h'];queries=torch.cat([F.normalize(head.encode_query(torch.as_tensor(h[i:i+8192],device=device)),dim=-1) for i in range(0,len(h),8192)])
        scale=head.logit_scale.exp().clamp(max=100)
        attraction=scorer.attraction_table(*attraction_inputs(raw,group,roles['category_probabilities'],device))
    optimizer=torch.optim.Adam([model.raw_counts,model.quantile_logits],lr=.01)
    rng=np.random.default_rng(42);history=[];best=math.inf;stale=0;grads={}
    def epoch(split,training):
        d=data[split];n=len(d['gold']);order=rng.permutation(n) if training else np.arange(n);total=0.
        for left in range(0,n,256):
            idx=torch.as_tensor(order[left:left+256],device=device);origin,code,gold=(d[k][idx].long() for k in ['origin','code','gold'])
            row=torch.searchsorted(lookup,origin);b=fine[row].long()
            with torch.set_grad_enabled(training):
                normalized=F.normalize(model.normalize(attributes,sorted_pool),dim=-1)
                attribute=scale*queries[d['event'][idx]]@normalized.T
                c=model.counts();counts=d['memory_counts'][idx]+c[0]*d['home'][idx]+c[1]*d['anchor'][idx]
                memory=torch.zeros_like(attribute).scatter_add_(1,d['memory_positions'][idx].long(),counts)
                rd=log_probability[code].gather(1,b)-log_count[row].gather(1,b)
                logits=conditional_logits(attribute,rd,memory,attraction[code],scorer.strengths(),b//4,gold)
                loss=F.cross_entropy(logits,gold)
                if not torch.isfinite(loss):raise ValueError('nonfinite landing loss')
                if training:
                    optimizer.zero_grad();loss.backward()
                    for name in ('raw_counts','quantile_logits'):
                        g=getattr(model,name).grad
                        assert g is not None and torch.isfinite(g).all(),name
                        grads[name]=np.maximum(grads.get(name,np.zeros(2)),g.detach().abs().cpu().numpy()).tolist()
                    optimizer.step()
                total+=float(loss.detach())*len(idx)
        return total/n
    for e in range(21):
        train_loss=epoch('train',True) if e else None;loss=epoch('val',False)
        history.append(dict(epoch=e,train_nll=train_loss,val_nll=loss,values=model.export_values()))
        if loss<best-1e-6:best=loss;best_state=copy.deepcopy(model.state_dict());best_epoch=e;stale=0
        else:stale+=1
        print(json.dumps(history[-1]),flush=True);write_json(output/'history.json',history)
        if stale>=5:break
    model.load_state_dict(best_state)
    with torch.no_grad():low,high=model.quantiles(sorted_pool)
    norm=dict(contract='source_learned_ordered_quantile_levels_v2',lower=low.cpu().tolist(),upper=high.cpu().tolist(),
        quantile_low=model.export_values()['quantile_levels'][0],quantile_high=model.export_values()['quantile_levels'][1],
        fit_grid_csv_sha256=list(map(sha256_file,pooled_grids)))
    bindings={k:sha256_file(v) for k,v in dict(source_codes=codes,landing_checkpoint=landing_checkpoint,source_grid=source_grid,learned_landing_checkpoint=learned_landing_checkpoint).items()}
    state=dict(schema=SCHEMA,source_city=city,state_dict={k:v.cpu() for k,v in best_state.items()},normalization=norm,bindings=bindings,
        best_epoch=best_epoch,home_best_epoch=home_report['best_epoch'],values=model.export_values(),training_seed=42)
    torch.save(state,output/'learned_context.pth')
    write_json(output/'report.json',dict(status='complete',**{k:v for k,v in state.items() if k!='state_dict'},
        initial_values=initial,home_fit=home_report,landing_gradient_max=grads,initial_val_nll=history[0]['val_nll'],selected_val_nll=best,
        checkpoint_sha256=sha256_file(output/'learned_context.pth'),target_trajectory_training=False,
        trainable_parameter_count=7,history=history))


def main():
    p=argparse.ArgumentParser();p.add_argument('--codes',required=True);p.add_argument('--landing-checkpoint',required=True)
    p.add_argument('--source-grid',required=True);p.add_argument('--learned-landing-checkpoint',required=True)
    p.add_argument('--pooled-grids',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    fit(**vars(p.parse_args()))


if __name__=='__main__':main()
