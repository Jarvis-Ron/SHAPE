#!/usr/bin/env python3
"""Run the four SHAPE experiments synchronously, with explicit evidence modes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch

from SHAPE.reproduce import check_inputs, evaluate, generate, read_spec
from SHAPE.stage2.config import TOKENIZER_TRAINING_CONTRACT
from SHAPE.stage3.contracts import sha256_file, write_json

ROOT = Path(__file__).resolve().parent
DIRECTIONS = ('nyc_to_ca', 'nyc_to_paris', 'ca_to_nyc', 'ca_to_paris')


def verify_reference(direction):
    """Check archived draws, never substitute them for generated scores."""
    spec = read_spec(ROOT / 'configs' / f'{direction}.json')
    check_inputs(spec)
    record = json.loads((ROOT / 'reference_results' / direction / 'reference.json').read_text())
    samples = record['metric_samples']
    result = {}
    for metric, reference in record['metrics'].items():
        if spec['protocol'] == 'pooled_three_seed_shell':
            values = samples['pooled_seed_shell_values'][metric]
            if len(values) != 200:
                raise ValueError('The pooled reference requires 200 shell scores')
        elif spec['protocol'] == 'per_seed_shell':
            draws = [samples['seed_shell_values'][str(seed)][metric] for seed in spec['seeds']]
            if any(len(v) != 200 for v in draws):
                raise ValueError('Each seed requires 200 shell scores')
            values = [np.mean(v) for v in draws]
        else:
            values = [samples['direct_seed_values'][str(seed)][metric] for seed in spec['seeds']]
        mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
        if not np.isclose(mean, reference['mean'], rtol=0, atol=1e-12) or not np.isclose(
                std, reference['sample_std'], rtol=0, atol=1e-12):
            raise ValueError(f'Reference summary differs from recorded draws: {direction}/{metric}')
        result[metric] = dict(mean=mean, std=std)
    print(json.dumps(dict(direction=direction, verification='archived_draws',
        artifact_origin='historical_semantic_supervision_and_distance_cost', metrics=result)), flush=True)
    return result


def training_plan(spec, output, device, prepared_events=False):
    """Construct source-only training; static alignment can use all static maps."""
    output = Path(output).resolve()
    source = spec['source_city']
    training, data = output / 'training', output / 'data'
    event_dir = ROOT / 'dataset' / 'prepared' / spec['direction'] if prepared_events else data / 'events'
    paths = {source: dict(latent_file=str(data / 'grid_latents' / f'{source}_latent.npz'),
                          out_dir=str(event_dir))}
    commands = []
    def add(*args):
        commands.append([sys.executable, '-u', '-m', 'SHAPE', *map(str, args)])
    if not prepared_events:
        add('train-grid', '--checkpoint', training / 'grid.pth', '--history', training / 'grid_history.json',
            '--device', device, '--epochs', 200, '--seed', 42)
        add('extract-grid', '--checkpoint', training / 'grid.pth', '--output-dir', data / 'grid_latents', '--device', device)
        add('prepare-events', '--city', source, '--config', output / 'city_paths.json', '--seed', 42)
    add('train-tokenizer', '--city', source, '--data-dir', event_dir, '--device', device,
        '--quantizer-type', 'fsq', '--num-quantizers', 3,
        '--codebook-size', 64, '--fsq-levels', '4,4,4', '--jump-aux-weight', .125,
        '--poi-aux-weight', .075, '--time-aux-weight', 0, '--encoder-perturb-mode', 'iid',
        '--encoder-perturb-sigma', .1, '--epochs', 50, '--batch-size', 196,
        '--max-attention-cells', 8388608, '--seed', 42, '--save-dir', training,
        '--model-name', 'tokenizer.pth', '--history-name', 'tokenizer_history.json',
        '--plot-path', training / 'tokenizer.png')
    add('encode-source', '--source-city', source, '--data-dir', event_dir,
        '--checkpoint', training / 'tokenizer.pth', '--grid-csv', spec['source_grid'],
        '--output', data / 'source_codes.npz', '--device', device, '--batch-size', 1, '--seed', 42,
        '--expected-jump-aux-weight', .125, '--expected-poi-aux-weight', .075,
        '--expected-time-aux-weight', 0, '--require-no-time-aux', '--city-config', output / 'city_paths.json')
    add('train-landing', '--codes', data / 'source_codes.npz', '--grid-csv', spec['source_grid'],
        '--output', training / 'landing.pth', '--mode', 'l2_attributes', '--device', device,
        '--epochs', 20, '--batch-size', 4096, '--seed', 42)
    for city in ('CA', 'NYC', 'PARIS'):
        commands[-1] += ['--normalization-grid-csv', str(ROOT / 'dataset' / f'{city}_grid_merged_features.csv')]
    return commands, paths


def train(spec, output, device, prepared_events=False):
    if prepared_events and spec['source_city'] != 'ca':
        raise ValueError('Prepared source-event datasets are bundled only for the CA-source directions')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data').mkdir()
    (output / 'training').mkdir()
    commands, paths = training_plan(spec, output, device, prepared_events)
    write_json(output / 'city_paths.json', paths)
    receipt = dict(status='running', direction=spec['direction'], commands=commands, completed_stages=0,
                   run_type='fresh_training', uses_frozen_target_reference=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        for index, command in enumerate(commands, 1):
            print(shlex.join(command), flush=True)
            with (output / f'stage_{index:02d}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, check=True)
            receipt['completed_stages'] = index
        new_spec = dict(spec, source_codes=str(output / 'data' / 'source_codes.npz'),
                        landing_checkpoint=str(output / 'training' / 'landing.pth'))
        new_spec.pop('original_shell_plans', None)
        new_spec.pop('archived_predictions', None)
        new_spec.pop('bundled_artifacts', None)
        new_spec['tokenizer_training_contract'] = TOKENIZER_TRAINING_CONTRACT
        new_spec['artifact_origin'] = 'fresh_training'
        new_spec['expected_sha256'] = {key: sha256_file(new_spec[key]) for key in spec['expected_sha256']}
        write_json(output / 'transfer.json', new_spec)
        receipt['status'] = 'complete'
        return new_spec
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(output / 'training_manifest.json', receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=(*DIRECTIONS, 'all'), default='all')
    parser.add_argument('--mode', choices=('verify', 'score', 'reproduce', 'train'), default='score',
                        help='verify: historical draws; score: historical predictions; reproduce: generation from current-contract inputs (bundled historical weights are rejected); train: fresh training and generation')
    parser.add_argument('--output', default='results/shape')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepared-events', action='store_true', help='For CA-source training only: use the bundled historical event embeddings')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.prepared_events and (args.mode != 'train' or args.direction not in ('ca_to_nyc', 'ca_to_paris')):
        parser.error('--prepared-events requires --mode train and one CA-source direction')
    torch.set_num_threads(1)
    directions = DIRECTIONS if args.direction == 'all' else (args.direction,)
    output = Path(args.output).resolve()
    for direction in directions:
        spec = read_spec(ROOT / 'configs' / f'{direction}.json')
        destination = output / direction
        if args.dry_run:
            print(f'{direction}: mode={args.mode}; protocol={spec["protocol"]}; seeds={spec["seeds"]}')
            if args.mode == 'train':
                commands, _ = training_plan(spec, destination / 'fit', args.device, args.prepared_events)
                for command in commands:
                    print(shlex.join(command))
            if args.mode in ('reproduce', 'train'):
                print(f'generate -> {destination / "generation"}; evaluate -> {destination / "evaluation"}')
            continue
        if args.mode == 'verify':
            verify_reference(direction)
            continue
        destination.mkdir(parents=True, exist_ok=False)
        if args.mode == 'train':
            spec = train(spec, destination / 'fit', args.device, args.prepared_events)
        predictions = spec.get('archived_predictions')
        if args.mode in ('reproduce', 'train'):
            predictions = generate(spec, destination / 'generation', args.device)
        evaluate(spec, predictions, destination / 'evaluation', args.workers)


if __name__ == '__main__':
    main()
