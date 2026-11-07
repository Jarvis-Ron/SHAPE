"""Execute the four fixed SHAPE experiments with their recorded evaluation units."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from SHAPE.gridspec import StaticGrid
from SHAPE.stage3.sampler import _compare_aggregates
from SHAPE.stage3.context_memory import _run_e2_tasks
from SHAPE.stage2.config import TOKENIZER_TRAINING_CONTRACT
from SHAPE.stage3.contracts import require_current_tokenizer_training, sha256_file, write_json
from SHAPE.stage3.grid_projection import _project_nominal_ids_to_vocabulary
from SHAPE.stage3.group_index import load_v4_transport_provenance, repair_group_index
from SHAPE.stage3.landing_head import load_canonical_grid_attributes
from SHAPE.stage3.matched_shell import _resampled_payload
from SHAPE.stage3.metrics import TEN_METRICS, aggregate_subset, primary_subset, score_prediction_group
from SHAPE.stage3.rank_semantics import _jsonable
from SHAPE.stage3.reference import reference_cohort, reference_payload
from SHAPE.stage3.run import FIVE, announce, train_home_reference
from SHAPE.stage3.runtime import build_source_payloads, prepare_transfer_runtime
from SHAPE.stage3.shell_plan import shell_plan, write_csv
from SHAPE.stage3.shell_support import subset_target_test_days
from SHAPE.stage3.source_cohort import load_g0_cohort
from SHAPE.stage3.static_features import _load_raw_table
from SHAPE.stage3.tasks import FIXED_VARIANT, load_contract, make_x1_hra_tasks

ROOT = Path(__file__).resolve().parents[1]
STATE = None
PROTOCOLS = {'pooled_three_seed_shell', 'per_seed_shell', 'direct_source_population'}


def read_spec(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec['configuration'] != FIXED_VARIANT or spec['protocol'] not in PROTOCOLS:
        raise ValueError('Unsupported mechanism or evaluation protocol')
    expected = [42, 43, 44] if spec['protocol'] == 'pooled_three_seed_shell' else [42, 43, 44, 45, 46]
    if spec['seeds'] != expected:
        raise ValueError('The reference generation seeds must not change')
    for key in ('source_codes', 'landing_checkpoint', 'source_grid', 'target_grid',
                'target_reference', 'home_quota', 'original_shell_plans', 'archived_predictions'):
        if spec.get(key):
            spec[key] = str((path.parent / spec[key]).resolve())
    return spec


def check_inputs(spec, *, generation=False):
    bindings = {}
    for key, expected in spec['expected_sha256'].items():
        if generation and spec['protocol'] == 'direct_source_population' and key == 'target_reference':
            continue
        actual = sha256_file(spec[key])
        if actual != expected:
            raise ValueError(f'Input checksum mismatch: {key}')
        bindings[key] = actual
    return bindings


def load_source(spec):
    cohort = load_g0_cohort(spec['source_codes'], {}, None,
                           require_complete_user_splits=True, reference_only=True)
    if cohort['city'] != spec['source_city'] or cohort['generated_binding']:
        raise ValueError('Source behavior binding changed')
    return cohort, build_source_payloads(cohort)


def load_reference(spec):
    with np.load(spec['target_reference'], allow_pickle=False) as archive:
        reference = {k: archive[k] for k in ('grid_id', 'minute', 'day_offsets', 'day_user_ids',
                                            'day_dates', 'day_splits', 'event_user_ids', 'event_splits')}
    cohort = reference_cohort(reference)
    static = StaticGrid.load(spec['target_grid'], city=spec['target_city'])
    projected, _ = _project_nominal_ids_to_vocabulary(reference['grid_id'], static.grid_ids, static.spec)
    cohort['real_grid'] = projected[cohort['event_indices']]
    return reference, cohort, reference_payload(cohort), static


def sampling_tasks(payloads, contract, seeds):
    template = make_x1_hra_tasks(payloads, contract)[0]
    return [dict(deepcopy(template), sampling_seed=seed) for seed in seeds]


def generate(spec, output, device, lanes=1):
    with np.load(spec['source_codes'], allow_pickle=False) as archive:
        require_current_tokenizer_training(archive)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bindings = check_inputs(spec, generation=True)
    cohort, payloads = load_source(spec)
    contract = load_contract()
    landing = contract['frozen_landing']
    source = dict(reference_codes=spec['source_codes'], checkpoint=spec['landing_checkpoint'],
                  grid_csv=spec['source_grid'], generated_by_seed={}, binding=bindings,
                  arms={'D1A1_raw': {'parameters': {k: landing[k] for k in
                        ('selection_mode', 'selection_temperature', 'distance_mode', 'distance_tau',
                         'attraction_mode', 'attraction_strength')}}})
    direct = spec['protocol'] == 'direct_source_population'
    if direct:
        # The generator receives only an aggregate histogram for the target.
        homes, target_users = None, None
        quota = json.loads(Path(spec['home_quota']).read_text())
    else:
        reference, target_cohort, _, _ = load_reference(spec)
        homes = train_home_reference(reference)
        target_users = {'users': target_cohort['users']}
        quota = None
    runtime = prepare_transfer_runtime(source,
        dict(selected_beta=landing['revisit_beta'], binding={'parameters_from': 'fixed_SHAPE'}),
        cohort, homes, target_users, spec['target_grid'], [device] * lanes,
        source_city=spec['source_city'], target_city=spec['target_city'],
        personal_scale=dict(FIXED_VARIANT), sampling_seeds=tuple(spec['seeds']),
        target_home_quota=quota, progress=announce)
    results = _run_e2_tasks(sampling_tasks(payloads, contract, spec['seeds']), payloads, runtime,
        stage_label='SHAPE', progress=lambda done, total, task, dev: announce(f'Completed seed {task["sampling_seed"]}: {done}/{total}'),
        **{k: landing[k] for k in ('landing_temperature', 'candidate_top_k',
                                 'candidate_omitted_mass_tolerance', 'candidate_audit_steps')},
        sequence_batch_size=32, landing_event_batch_size=64)
    results.sort(key=lambda r: r['sampling_seed'])
    if [r['sampling_seed'] for r in results] != spec['seeds']:
        raise RuntimeError('Missing or repeated generated seed')
    np.savez_compressed(output / 'predictions.npz', **{f'seed_{r["sampling_seed"]}': r['prediction'] for r in results})
    provenance = dict(runtime['provenance'])
    probability = provenance.pop('target_home_probability')
    np.savez_compressed(output / 'home_assignments.npz', probability=probability,
                       **{f'seed_{s}': a for s, a in runtime['source_context_homes'].items()})
    write_json(output / 'runtime.json', _jsonable(provenance))
    write_json(output / 'manifest.json', dict(status='complete', direction=spec['direction'],
        configuration=FIXED_VARIANT, seeds=spec['seeds'], input_sha256=bindings,
        mechanism='shape_no_layer_semantics_no_aux_rank_only_v3',
        tokenizer_training_contract=TOKENIZER_TRAINING_CONTRACT,
        target_reference_read_by_generation=not direct,
        target_checkin_generation_use='aggregate_home_quota_only' if direct else 'training_home_marginal_and_eligible_user_set',
        prediction_sha256=sha256_file(output / 'predictions.npz')))
    return output / 'predictions.npz'


def scoring_runtime(spec, static):
    _, centers, ids, _ = load_canonical_grid_attributes(spec['target_grid'])
    raw, _ = _load_raw_table(spec['target_grid'])
    group, _, _ = repair_group_index(raw, spec['target_city'],
        transport_provenance=load_v4_transport_provenance(spec['target_grid'], ids))
    return dict(target_centers=centers, target_grid_ids=ids, target_static=static, target_group=group)


def direct_scores(aggregate, truth, group):
    pm, rm = float(aggregate['od'].sum()), float(truth['od'].sum())
    if min(pm, rm) <= 0:
        raise ValueError('OD distributions must have positive mass')
    pred, ref = dict(aggregate), dict(truth)
    pred['od'], ref['od'] = aggregate['od'] / pm, truth['od'] / rm
    result = _compare_aggregates(pred, ref, group)
    assert abs(result['M2_od_cpc'] - (1 - .5 * np.abs(pred['od'] - ref['od']).sum())) < 1e-12
    return {k: result[k] for k in FIVE}


def score_init(source, target, plans, runtime, predictions, protocol):
    global STATE
    torch.set_num_threads(1)
    truth = aggregate_subset(primary_subset(target, target['values']), runtime)
    STATE = source, target, plans, runtime, predictions, protocol, truth


def score_repeat(repeat):
    source, target, plans, runtime, predictions, protocol, truth = STATE
    indices, aligned = _resampled_payload(source, target, plans[repeat])
    if protocol == 'pooled_three_seed_shell':
        point, _ = score_prediction_group([predictions[s][indices] for s in sorted(predictions)], aligned, runtime, truth)
        return [dict(repeat=repeat, metric=m, value=point[m]) for m in TEN_METRICS]
    rows = []
    for seed in sorted(predictions):
        point, _ = score_prediction_group([predictions[seed][indices]], aligned, runtime, truth)
        rows.extend(dict(seed=seed, repeat=repeat, metric=m, value=point[m]) for m in TEN_METRICS)
    return rows


def summarize(rows, protocol, seeds):
    summaries, per_seed = [], []
    metrics = list(FIVE) if protocol == 'direct_source_population' else TEN_METRICS
    for metric in metrics:
        if protocol == 'pooled_three_seed_shell':
            values = [r['value'] for r in rows if r['metric'] == metric]
            if len(values) != 200:
                raise ValueError('Expected exactly 200 pooled shell scores')
            axis = 'shell_resamples_of_pooled_seed_populations'
        else:
            values = []
            for seed in seeds:
                draws = [r['value'] for r in rows if r['metric'] == metric and r['seed'] == seed]
                expected = 1 if protocol == 'direct_source_population' else 200
                if len(draws) != expected:
                    raise ValueError('Incomplete seed scores')
                value = float(np.mean(draws))
                values.append(value)
                per_seed.append(dict(seed=seed, metric=metric, mean=value, sample_count=expected))
            axis = 'generation_seed_scores' if protocol == 'direct_source_population' else 'generation_seed_shell_means'
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite scores')
        summaries.append(dict(metric=metric, mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                              sample_count=len(values), sample_axis=axis, std_ddof=1))
    return summaries, per_seed


def evaluate(spec, prediction_file, output, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    check_inputs(spec)
    _, payloads = load_source(spec)
    source = payloads['source_test_real']
    _, _, target, static = load_reference(spec)
    runtime = scoring_runtime(spec, static)
    with np.load(prediction_file, allow_pickle=False) as archive:
        if set(archive.files) != {f'seed_{s}' for s in spec['seeds']}:
            raise ValueError('Prediction archive seed set differs from the experiment')
        predictions = {s: archive[f'seed_{s}'] for s in spec['seeds']}
    for prediction in predictions.values():
        if len(prediction) != sum(source['lengths']) or not np.isin(prediction, runtime['target_grid_ids']).all():
            raise ValueError('Prediction shape or target grid vocabulary mismatch')
    rows = []
    if spec['protocol'] == 'direct_source_population':
        truth = aggregate_subset(primary_subset(target, target['values']), runtime)
        for seed, prediction in predictions.items():
            aggregate = aggregate_subset(primary_subset(source, prediction), runtime)
            point = direct_scores(aggregate, truth, runtime['target_group'])
            rows.extend(dict(seed=seed, metric=m, value=point[m]) for m in FIVE)
        plan_hash = None
    else:
        plans, _, retained = shell_plan(source, target, 200, output, length_policy=spec['shell_length_policy'])
        target = subset_target_test_days(target, retained)
        if spec.get('original_shell_plans'):
            with np.load(spec['original_shell_plans'], allow_pickle=False) as archive:
                if not np.array_equal(plans, archive['plans']):
                    raise ValueError('Evaluation shell plans differ from the recorded experiment')
        plan_hash = hashlib.sha256(plans.tobytes()).hexdigest()
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=score_init,
                initargs=(source, target, plans, runtime, predictions, spec['protocol'])) as pool:
            for batch in pool.map(score_repeat, range(200), chunksize=4):
                rows.extend(batch)
    summaries, per_seed = summarize(rows, spec['protocol'], spec['seeds'])
    write_csv(output / 'metric_draws.csv', rows)
    write_csv(output / 'metrics.csv', summaries)
    if per_seed:
        write_csv(output / 'per_seed_metrics.csv', per_seed)
    lookup = {r['metric']: r for r in summaries}
    five = {'direction': spec['direction'], **{label: f'{lookup[k]["mean"]:.4f} ± {lookup[k]["std"]:.4f}' for k, label in FIVE.items()}}
    write_csv(output / 'five_metrics.csv', [five])
    result = dict(direction=spec['direction'], protocol=spec['protocol'], seeds=spec['seeds'],
                  means={k: lookup[k]['mean'] for k in FIVE}, stds={k: lookup[k]['std'] for k in FIVE},
                  prediction_sha256=sha256_file(prediction_file), shell_plan_array_sha256=plan_hash,
                  target_days=len(target['lengths']), source_days=len(source['lengths']), status='complete')
    archived = spec.get('archived_predictions')
    result['prediction_origin'] = (
        'historical_semantic_supervision_and_distance_cost'
        if archived and result['prediction_sha256'] == sha256_file(archived)
        else 'supplied_predictions'
    )
    write_json(output / 'result.json', result)
    print(json.dumps(five), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['generate', 'evaluate'])
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    spec = read_spec(args.spec)
    if args.action == 'generate':
        generate(spec, args.output, args.device, args.lanes)
    else:
        evaluate(spec, args.predictions or spec['archived_predictions'], args.output, args.workers)


if __name__ == '__main__':
    main()
