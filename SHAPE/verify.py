"""Check data, matched-user plans, and trained model bindings when provided."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SHAPE.reproduce import MODEL_INPUTS, check_inputs, load_source, load_reference, read_spec
from SHAPE.evaluation.matched_users import validate_plans
from SHAPE.stage3.contracts import require_current_tokenizer_training, sha256_file
from SHAPE.stage3.learned_context import load_context
from SHAPE.stage3.learned_landing import load_learned_landing
from SHAPE.stage3.metrics import primary_subset
from SHAPE.stage3.home_quota import load_home_quota, quota_probability


def verify_model(spec, tokenizer_hash):
    if tokenizer_hash != sha256_file(spec['tokenizer_checkpoint']):
        raise ValueError('Tokenizer/source archive binding changed')
    tokenizer = torch.load(spec['tokenizer_checkpoint'], map_location='cpu', weights_only=False)
    if any(tokenizer['config'].get(k, 0) != 0 for k in ('jump_aux_weight', 'poi_aux_weight', 'time_aux_weight')):
        raise ValueError('Stage 2 auxiliary supervision must be disabled')
    head = torch.load(spec['landing_checkpoint'], map_location='cpu', weights_only=False)
    if head['tokenizer_sha256'] != tokenizer_hash or head['codes_sha256'] != sha256_file(spec['source_codes']):
        raise ValueError('Landing/source archive binding changed')
    common = {k: spec[k] for k in ('source_codes', 'landing_checkpoint', 'source_grid', 'source_city')}
    landing, _ = load_learned_landing(spec['learned_landing_checkpoint'], **common)
    context, _ = load_context(spec['learned_context_checkpoint'],
                              learned_landing_checkpoint=spec['learned_landing_checkpoint'], **common)
    return dict(learned_landing_parameters=sum(p.numel() for p in landing.parameters()),
                learned_context_parameters=sum(p.numel() for p in context.parameters()))


def verify(path):
    spec = read_spec(path, require_fitted=False)
    has_model = any(spec.get(key) for key in MODEL_INPUTS)
    check_inputs(spec, require_fitted=has_model)
    with np.load(spec['source_codes'], allow_pickle=False) as a:
        require_current_tokenizer_training(a)
        tokenizer_hash = str(a['tokenizer_sha256'].item())
    model_report = verify_model(spec, tokenizer_hash) if has_model else {}
    _, payloads = load_source(spec)
    _, _, target, target_static = load_reference(spec)
    quota = load_home_quota(spec['home_quota'], target_city=spec['target_city'])
    quota_probability(quota, target_static.grid_ids, target_city=spec['target_city'])
    source = payloads['source_test_real']
    source_sub = primary_subset(source, source['features'][:, 0])
    target_sub = primary_subset(target, target['values'])
    with np.load(spec['user_selection'], allow_pickle=False) as a:
        sp, tp = a['synthetic_indices'], a['real_indices']
    metadata = json.loads(Path(spec['evaluation_metadata']).read_text())
    validate_plans(np.unique(source_sub['users']), np.unique(target_sub['users']), sp, tp, metadata)
    result = dict(direction=spec['direction'], status='verified', users_per_side=sp.shape[1],
                  scope='model_and_data' if has_model else 'data',
                  home_quota_schema=quota['schema'], home_quota_provenance=quota['provenance'],
                  **model_report)
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    verify(args.spec)
