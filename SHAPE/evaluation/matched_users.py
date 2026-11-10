"""Four metrics with equal complete-user budgets, separately for each seed."""
import hashlib
import json
from pathlib import Path

import numpy as np

from SHAPE.stage3.contracts import sha256_file, write_json
from SHAPE.stage3.metrics import aggregate_subset, primary_subset
from SHAPE.stage3.rank_semantics import _subset_sequences

FOUR = {
    'M2_od_cpc': 'OD CPC',
    'M3_period_space_jsd_mean': 'Spatiotemporal JSD',
    'A_radius_of_gyration_distribution_jsd': 'RoG JSD',
    'A_jump_distance_spectrum_jsd': 'Jump-length JSD',
}


def user_hash(users):
    return hashlib.sha256(json.dumps(users.tolist()).encode()).hexdigest()


def validate_plans(source_users, target_users, sp, tp, metadata):
    """Reject a reused plan if either population or the sampling unit changed."""
    budget = min(len(source_users), len(target_users))
    if budget < 1 or metadata['users_per_side'] != budget:
        raise ValueError('User budget differs from the eligible populations')
    for users, plan, field in ((source_users, sp, 'source_user_order_sha256'),
                               (target_users, tp, 'target_user_order_sha256')):
        if user_hash(users) != metadata[field]:
            raise ValueError('User order changed; prepare a new evaluation plan')
        if plan.shape != (200, budget) or plan.dtype.kind not in 'iu':
            raise ValueError('Expected 200 complete-user selections at the matched budget')
        if plan.min() < 0 or plan.max() >= len(users):
            raise ValueError('User index outside population')
        if any(len(np.unique(row)) != budget for row in plan):
            raise ValueError('Users must be sampled without replacement within each draw')


def take_users(subset, users, indices):
    """Keep all eligible days of each selected user; never resample events."""
    return _subset_sequences(subset['offsets'], subset['users'], subset['minute'],
                             subset['values'], np.isin(subset['users'], users[indices]))


def summarize(draws, seeds):
    result = {}
    for metric, values in draws.items():
        values = np.asarray(values)
        if values.shape != (len(seeds), 200) or not np.isfinite(values).all():
            raise ValueError('Expected 200 finite metric values for each generation seed')
        means = values.mean(axis=1)
        result[metric] = dict(mean=float(means.mean()), seed_std=float(means.std(ddof=1)),
                             per_seed={str(seed): float(v) for seed, v in zip(seeds, means)})
    return result


def evaluate(spec, prediction_file, output):
    from SHAPE.reproduce import (check_inputs, load_source, load_reference,
                                 scoring_runtime, direct_scores, announce)
    bindings = check_inputs(spec)
    _, payloads = load_source(spec)
    source = payloads['source_test_real']
    _, _, target, static = load_reference(spec)
    runtime = scoring_runtime(spec, static)
    real = primary_subset(target, target['values'])
    seeds = spec['seeds']
    with np.load(prediction_file, allow_pickle=False) as z:
        if set(z.files) != {f'seed_{s}' for s in seeds}:
            raise ValueError('Predictions must contain exactly the five generation seeds')
        subs = []
        for seed in seeds:
            predicted = z[f'seed_{seed}']
            if predicted.shape != (len(source['features']),):
                raise ValueError('Predictions do not match source events')
            if not np.isin(predicted, static.grid_ids).all():
                raise ValueError('Predictions contain cells outside the target vocabulary')
            subs.append(primary_subset(source, predicted))
    su, tu = np.unique(subs[0]['users']), np.unique(real['users'])
    metadata = json.loads(Path(spec['evaluation_metadata']).read_text())
    if sha256_file(spec['user_selection']) != metadata['user_selection_sha256']:
        raise ValueError('Evaluation selection checksum changed')
    with np.load(spec['user_selection'], allow_pickle=False) as z:
        sp, tp = z['synthetic_indices'], z['real_indices']
    validate_plans(su, tu, sp, tp, metadata)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    full_source = len(su) == sp.shape[1]
    full_target = len(tu) == tp.shape[1]
    sa = [aggregate_subset(s, runtime) for s in subs] if full_source else None
    ta = aggregate_subset(real, runtime) if full_target else None
    draws = {m: np.zeros((len(seeds), len(sp))) for m in FOUR}
    for repeat in range(len(sp)):
        truth = ta if full_target else aggregate_subset(take_users(real, tu, tp[repeat]), runtime)
        for i, sub in enumerate(subs):
            gen = sa[i] if full_source else aggregate_subset(take_users(sub, su, sp[repeat]), runtime)
            point = direct_scores(gen, truth, runtime['target_group'])
            for m in FOUR:
                v = point[m]
                if not np.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError('Undefined/out-of-range metric: ' + m)
                draws[m][i, repeat] = v
        if (repeat + 1) % 50 == 0:
            announce(f'Evaluated {repeat + 1}/{len(sp)} matched user draws')
    result = dict(direction=spec['direction'], protocol=spec['protocol'],
                  users_per_side=sp.shape[1], draws=200, generation_seeds=seeds,
                  training_seed=42, metrics=summarize(draws, seeds),
                  input_sha256=bindings, prediction_sha256=sha256_file(prediction_file),
                  interpretation='Each seed is evaluated separately. Mean and sample SD across five seed means; users are never pooled across seeds.')
    np.savez_compressed(out / 'metric_draws.npz', **draws)
    write_json(out / 'result.json', result)
    rows = ['| Metric | Mean | Seed SD |', '|---|---:|---:|']
    for m, label in FOUR.items():
        v = result['metrics'][m]
        rows.append(f"| {label} | {v['mean']:.4f} | {v['seed_std']:.4f} |")
    (out / 'table.md').write_text('\n'.join(rows) + '\n')
    return result
