"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from SHAPE.gridspec import StaticGrid
from SHAPE.stage3.shell_plan import shell_plan, write_csv
from SHAPE.stage3.contracts import sha256_file, write_json
from SHAPE.stage3.tasks import DEFAULT_CONTRACT, FIXED_VARIANT, load_contract, make_x1_hra_tasks
from SHAPE.stage3.source_cohort import load_g0_cohort
from SHAPE.stage3.rank_semantics import _jsonable
from SHAPE.stage3.context_memory import _run_e2_tasks
from SHAPE.stage3.matched_shell import _resampled_payload
from SHAPE.stage3.grid_projection import _project_nominal_ids_to_vocabulary
from SHAPE.stage3.metrics import TEN_METRICS, aggregate_subset, primary_subset, score_prediction_group
from SHAPE.stage3.reference import reference_cohort, reference_payload
from SHAPE.stage3.runtime import build_source_payloads, prepare_transfer_runtime
from SHAPE.stage3.shell_support import subset_target_test_days


FIVE = {
    'M2_od_cpc': 'OD CPC',
    'M3_period_space_jsd_mean': 'Spatiotemporal JSD',
    'A_radius_of_gyration_distribution_jsd': 'RoG JSD',
    'A_jump_distance_spectrum_jsd': 'Jump-length JSD',
    'A_visited_poi_category_share_jsd': 'POI-category JSD',
}


SCORE = None


def announce(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def config_id(config):
    if config != FIXED_VARIANT:
        raise ValueError('Only the frozen SHAPE variant is supported')
    return 'SHAPE'


def train_home_reference(artifact):
    """Explicit narrow boundary: generation receives no target test locations."""
    chosen = np.asarray(artifact['event_splits']) == 'train'
    return {key: np.asarray(artifact[key])[chosen] for key in
            ('grid_id', 'event_user_ids', 'event_splits')}


def score_init(source, target, plans, runtime):
    global SCORE
    torch.set_num_threads(1)
    reference = aggregate_subset(primary_subset(target, target['values']), runtime)
    SCORE = (source, target, plans, runtime, reference)


def score_one(job):
    repeat, prediction_file = job
    source, target, plans, runtime, reference = SCORE
    indices, aligned = _resampled_payload(source, target, plans[repeat])
    with np.load(prediction_file, allow_pickle=False) as archive:
        predictions = [archive['seed_%d' % seed][indices] for seed in (42, 43, 44)]
    point, audit = score_prediction_group(predictions, aligned, runtime, reference)
    return [{'repeat': repeat, 'metric': metric, 'value': point[metric], **audit}
            for metric in TEN_METRICS]


def summarize(draws, arm):
    result = []
    for metric in TEN_METRICS:
        values = np.array([row['value'] for row in draws if row['metric'] == metric])
        if len(values) < 2 or not np.isfinite(values).all():
            raise RuntimeError('undefined/insufficient metric: ' + metric)
        result.append(dict(arm=arm, metric=metric, mean=float(values.mean()),
                           std=float(values.std(ddof=1)), draw_count=len(values),
                           uncertainty='shell_SD_fixed_models', std_ddof=1))
    return result


def run(args):
    torch.set_num_threads(args.cpu_threads)
    spec = json.loads(Path(args.spec).read_text())
    for key in ('source_codes','landing_checkpoint','target_reference','source_grid','target_grid','original_shell_plans'):
        if spec.get(key) and not Path(spec[key]).is_absolute():
            spec[key] = str((Path(args.spec).resolve().parent / spec[key]).resolve())
    if spec.get('source_city') not in ('nyc','ca') or spec.get('target_city') not in ('nyc','ca','paris') or spec['source_city'] == spec['target_city']:
        raise ValueError('Unsupported source/target direction')
    if spec.get('shell_length_policy') not in ('strict','common-support'):
        raise ValueError('Explicit shell_length_policy is required')
    configs = [dict(FIXED_VARIANT)]
    if args.shell_repeats != 200:
        raise ValueError('formal personal-scale protocol requires 200 shared shells')
    if len({config_id(c) for c in configs}) != len(configs):
        raise ValueError('duplicate configuration')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'worker_manifest.json').exists():
        raise FileExistsError('worker output already exists; never silently reuse changed inputs')
    source_city, target_city = spec['source_city'], spec['target_city']
    inputs = {key: dict(path=str(Path(spec[key]).resolve()), sha256=sha256_file(spec[key]))
              for key in ('source_codes', 'landing_checkpoint', 'target_reference',
                          'source_grid', 'target_grid')}
    for key, expected in spec.get('expected_sha256', {}).items():
        if key not in inputs or inputs[key]['sha256'] != expected:
            raise ValueError('Frozen reference input hash mismatch: ' + key)
    contract_path = DEFAULT_CONTRACT
    contract = load_contract(contract_path)
    landing = contract['frozen_landing']
    manifest = dict(status='running', direction=source_city+'_to_'+target_city,
                    configs=configs, inputs=inputs, device=args.device,
                    seeds=[42,43,44], shell_repeats=args.shell_repeats,
                    full_source_cohort=True, source_model_refit=False,
                    mechanism_contract_sha256=sha256_file(DEFAULT_CONTRACT),
                    base_rank_contract_sha256=sha256_file(contract_path),
                    test_unchanged=True, started_at=time.time(),
                    information_audit={
                        'personal_profile_and_gamma':'source_city_train_only',
                        'rank_and_candidates':'complete_static_grid_vocabulary',
                        'target_train_prior':'T2_home_marginal_only',
                        'target_distance_quantiles_used':False,
                        'target_test_locations':'scorer_only',
                        'target_test_structure':'unchanged_person_day_shell',
                        'model_selection':'none_frozen_user_selected_variant'})
    write_json(output/'worker_manifest.json', manifest)
    announce('source and shell preflight '+manifest['direction'])
    cohort = load_g0_cohort(spec['source_codes'], {}, None,
                           require_complete_user_splits=True, reference_only=True)
    if cohort['city'] != source_city or cohort['generated_binding']:
        raise RuntimeError('frozen source-city observed-code binding mismatch')
    payloads = build_source_payloads(cohort)
    source_payload = payloads['source_test_real']
    with np.load(spec['target_reference'], allow_pickle=False) as archive:
        reference = {key: archive[key] for key in ('grid_id','minute','day_offsets',
            'day_user_ids','day_dates','day_splits','event_user_ids','event_splits')}
    target_cohort = reference_cohort(reference)
    target_static = StaticGrid.load(spec['target_grid'], city=target_city)
    projected, projection = _project_nominal_ids_to_vocabulary(
        reference['grid_id'], target_static.grid_ids, target_static.spec)
    target_cohort['real_grid'] = projected[target_cohort['event_indices']]
    full_target = reference_payload(target_cohort)
    plans, shell_audit, retained = shell_plan(source_payload, full_target,
        args.shell_repeats, output, length_policy=spec['shell_length_policy'])
    shell_audit.update(source_city=source_city,target_city=target_city,
                       legacy_NYC_CA_fields_are_algorithm_identifiers=True)
    write_json(output/'preflight/shell_audit.json',_jsonable(shell_audit))
    target = subset_target_test_days(full_target, retained)
    if spec.get('original_shell_plans'):
        with np.load(spec['original_shell_plans'], allow_pickle=False) as archive:
            if not np.array_equal(plans, archive['plans'][:args.shell_repeats]):
                raise RuntimeError('new shell plans differ from original immutable evaluation')
    target_hash = hashlib.sha256()
    for key in ('values','lengths'):
        target_hash.update(np.asarray(target[key]).tobytes())
    for key in ('offsets','users','dates','minute'):
        target_hash.update(np.asarray(target['view'][key]).tobytes())
    manifest.update(test_array_sha256=target_hash.hexdigest(),
        shell_plan_sha256=hashlib.sha256(plans.tobytes()).hexdigest(),
        target_test_days=len(target['lengths']), target_test_events=len(target['values']),
        measurement_scope=shell_audit['measurement_scope'], projection=projection)
    write_json(output/'worker_manifest.json', _jsonable(manifest))
    # Only train home events cross the generation interface. Test locations are
    # retained by this separate evaluator, never in the landing runtime.
    homes_reference = train_home_reference(reference)
    source = dict(reference_codes=spec['source_codes'], checkpoint=spec['landing_checkpoint'],
                  grid_csv=spec['source_grid'], generated_by_seed={}, binding=inputs,
                  arms={'D1A1_raw': {'parameters': {key:landing[key] for key in
                    ('selection_mode','selection_temperature','distance_mode','distance_tau',
                     'attraction_mode','attraction_strength')}}})
    closeout = dict(selected_beta=landing['revisit_beta'],
                    binding={'parameters_from':'master_frozen_rank_T2','target_selection':False})
    runtime = None
    previous_weighted = None
    pool = None
    try:
        for config in configs:
            name = config_id(config)
            root = output/name
            root.mkdir()
            started = time.monotonic()
            write_json(root/'configuration.json', config)
            if previous_weighted != config['weighted_rank']:
                if runtime is not None:
                    del runtime
                    torch.cuda.empty_cache()
                runtime = prepare_transfer_runtime(source, closeout, cohort, homes_reference,
                    {'users':target_cohort['users']}, spec['target_grid'], [args.device]*args.lanes,
                    source_city=source_city, target_city=target_city,
                    personal_scale=dict(config), progress=announce)
                previous_weighted = config['weighted_rank']
                write_json(root/'source_profile.json', _jsonable(runtime['personal_profiles']))
                provenance = dict(runtime['provenance'])
                probability = provenance.pop('target_home_probability')
                write_json(root/'runtime_provenance.json', _jsonable(provenance))
                np.savez_compressed(root/'home_assignments.npz', probability=probability,
                    **{'seed_%d'%seed:value for seed,value in runtime['source_context_homes'].items()})
                if pool is None:
                    scoring_runtime = {key:runtime[key] for key in
                        ('target_centers','target_grid_ids','target_static','target_group')}
                    pool = ProcessPoolExecutor(args.score_workers,
                        mp_context=mp.get_context('spawn'), initializer=score_init,
                        initargs=(source_payload,target,plans,scoring_runtime))
            runtime['personal_scale'] = dict(config)
            announce('landing '+name)
            results = _run_e2_tasks(make_x1_hra_tasks(payloads, contract), payloads, runtime,
                stage_label=name,
                progress=lambda done,total,task,device: announce(
                    f'{name} seeds {done}/{total}, seed={task["sampling_seed"]}'),
                **{key:landing[key] for key in ('landing_temperature','candidate_top_k',
                    'candidate_omitted_mass_tolerance','candidate_audit_steps')},
                sequence_batch_size=args.sequence_batch_size,
                landing_event_batch_size=args.event_batch_size)
            results.sort(key=lambda x:x['sampling_seed'])
            if [x['sampling_seed'] for x in results] != [42,43,44]:
                raise RuntimeError('missing/repeated landing seed')
            prediction_path = root/'predictions.npz'
            np.savez_compressed(prediction_path, **{'seed_%d'%x['sampling_seed']:x['prediction']
                                                   for x in results})
            write_json(root/'landing_audit.json', _jsonable([
                {k:v for k,v in x.items() if k!='prediction'} for x in results]))
            announce('scoring '+name)
            draws = []
            for index, rows in enumerate(pool.map(score_one,
                    [(i,str(prediction_path)) for i in range(len(plans))], chunksize=4)):
                draws.extend(rows)
                if (index+1)%50 == 0:
                    announce(f'{name} shells {index+1}/{len(plans)}')
            summaries = summarize(draws, name)
            write_csv(root/'metric_draws.csv', draws)
            write_csv(root/'metrics.csv', summaries)
            by_metric={row['metric']:row for row in summaries}
            write_csv(root/'five_metrics.csv',[{'method':name,**{
                label:f"{by_metric[key]['mean']:.4f} ± {by_metric[key]['std']:.4f}"
                for key,label in FIVE.items()}}])
            result = dict(config=config, means={x['metric']:x['mean'] for x in summaries},
                          stds={x['metric']:x['std'] for x in summaries},
                          elapsed_seconds=time.monotonic()-started,
                          prediction_sha256=sha256_file(prediction_path),
                          test_array_sha256=manifest['test_array_sha256'],
                          shell_plan_sha256=manifest['shell_plan_sha256'])
            for key,record in inputs.items():
                if sha256_file(record['path']) != record['sha256']:
                    raise RuntimeError('immutable experiment input changed: '+key)
            write_json(root/'result.json', result)
            announce('completed '+name+' '+json.dumps(result))
        manifest.update(status='complete', finished_at=time.time())
    except Exception:
        error = traceback.format_exc()
        (output/'error.txt').write_text(error, encoding='utf-8')
        manifest.update(status='failed', error=error, finished_at=time.time())
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        write_json(output/'worker_manifest.json', _jsonable(manifest))


def main(argv=None):
    parser=argparse.ArgumentParser(description='SHAPE: frozen opportunity rank + distance cost, lambda=kappa=0')
    for key in ('spec','output','device'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--shell-repeats',type=int,default=200)
    parser.add_argument('--lanes',type=int,default=3)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--score-workers',type=int,default=8)
    parser.add_argument('--sequence-batch-size',type=int,default=32)
    parser.add_argument('--event-batch-size',type=int,default=64)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
