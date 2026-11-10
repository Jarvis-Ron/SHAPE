"""Generate the released learned-context SHAPE model and evaluate complete users."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
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
from SHAPE.stage3.metrics import TEN_METRICS, aggregate_subset, primary_subset, score_prediction_group
from SHAPE.stage3.rank_semantics import _jsonable
from SHAPE.stage3.reference import reference_cohort, reference_payload
from SHAPE.stage3.run import FIVE, announce, train_home_reference
from SHAPE.stage3.runtime import build_source_payloads, prepare_transfer_runtime
from SHAPE.stage3.source_cohort import load_g0_cohort
from SHAPE.stage3.static_features import _load_raw_table
from SHAPE.stage3.tasks import FIXED_VARIANT, load_contract, make_x1_hra_tasks

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {'equal_complete_users_v1'}
MODEL_INPUTS = ('tokenizer_checkpoint', 'landing_checkpoint',
                'learned_landing_checkpoint', 'learned_context_checkpoint')


def require_bindings(spec, keys):
    for key in keys:
        if not spec.get(key) or key not in spec.get('expected_sha256', {}):
            hint = ' Train first and use the resulting fit/transfer.json.' if key in MODEL_INPUTS else ''
            raise ValueError('Latest SHAPE requires a bound input with an expected checksum: ' + key + hint)


def read_spec(path, *, require_fitted=True):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec['configuration'] != FIXED_VARIANT or spec['protocol'] not in PROTOCOLS:
        raise ValueError('Unsupported mechanism or evaluation protocol')
    expected = [42, 43, 44, 45, 46]
    if spec['seeds'] != expected:
        raise ValueError('The reference generation seeds must not change')
    for key in ('source_codes', 'landing_checkpoint', 'source_grid', 'target_grid',
                'target_reference', 'home_quota', 'source_events', 'tokenizer_checkpoint',
                'grid_checkpoint', 'user_selection', 'evaluation_metadata',
                'learned_landing_checkpoint', 'learned_context_checkpoint'):
        if spec.get(key):
            spec[key] = str((path.parent / spec[key]).resolve())
    require_bindings(spec, ('home_quota', 'user_selection', 'evaluation_metadata'))
    if require_fitted:
        require_bindings(spec, MODEL_INPUTS)
    return spec


def check_inputs(spec, *, generation=False, require_fitted=True):
    bindings = {}
    require_bindings(spec, ('home_quota',))
    if require_fitted:
        require_bindings(spec, MODEL_INPUTS)
    for key, expected in spec['expected_sha256'].items():
        if generation and key in ('target_reference', 'user_selection', 'evaluation_metadata'):
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
    bindings = check_inputs(spec, generation=True)
    with np.load(spec['source_codes'], allow_pickle=False) as archive:
        require_current_tokenizer_training(archive)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cohort, payloads = load_source(spec)
    contract = load_contract()
    landing = contract['frozen_landing']
    if spec.get('learned_landing_checkpoint') and landing['landing_temperature'] != 1.:
        raise ValueError('Learned landing requires its fixed landing temperature')
    source = dict(reference_codes=spec['source_codes'], checkpoint=spec['landing_checkpoint'],
                  grid_csv=spec['source_grid'], generated_by_seed={}, binding=bindings,
                  arms={'D1A1_raw': {'parameters': {k: landing[k] for k in
                        ('selection_mode', 'selection_temperature', 'distance_mode', 'distance_tau',
                         'attraction_mode', 'attraction_strength')}}})
    # Only the explicit aggregate home histogram enters target conditioning.
    homes, target_users = None, None
    quota = json.loads(Path(spec['home_quota']).read_text())
    runtime = prepare_transfer_runtime(source,
        dict(selected_beta=landing['revisit_beta'], binding={'parameters_from': 'fixed_SHAPE'}),
        cohort, homes, target_users, spec['target_grid'], [device] * lanes,
        source_city=spec['source_city'], target_city=spec['target_city'],
        personal_scale=dict(FIXED_VARIANT), sampling_seeds=tuple(spec['seeds']),
        target_home_quota=quota, progress=announce,
        learned_landing_checkpoint=spec.get('learned_landing_checkpoint'),
        learned_context_checkpoint=spec.get('learned_context_checkpoint'))
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
        mechanism='shape_source_learned_context_v2',
        tokenizer_training_contract=TOKENIZER_TRAINING_CONTRACT,
        target_reference_read_by_generation=False,
        target_checkin_generation_use='aggregate_home_quota_only',
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



def evaluate(spec, prediction_file, output, workers=1):
    from SHAPE.evaluation.matched_users import evaluate as score
    return score(spec, prediction_file, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'evaluate'))
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--predictions')
    args = parser.parse_args(argv)
    if args.lanes < 1:
        parser.error('--lanes must be positive')
    torch.set_num_threads(1)
    spec = read_spec(args.spec)
    if args.action == 'generate':
        generate(spec, args.output, args.device, args.lanes)
    else:
        if not args.predictions:
            parser.error('evaluate requires --predictions from a completed generation')
        evaluate(spec, args.predictions, args.output)


if __name__ == '__main__':
    main()
