"""Run only the final learned-context SHAPE model, synchronously."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import torch

from SHAPE.reproduce import MODEL_INPUTS, check_inputs, evaluate, generate, read_spec
from SHAPE.stage3.contracts import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ('nyc_to_ca', 'nyc_to_paris', 'ca_to_nyc', 'ca_to_paris')


def training_plan(spec, output, device='cpu', fresh_grid=False):
    """Reuse supplied Stage 1 events by default, then fit all later components."""
    output = Path(output).resolve()
    source = spec['source_city']
    training, data = output / 'training', output / 'data'
    event_dir = data / 'events' if fresh_grid else Path(spec['source_events'])
    paths = {source: dict(latent_file=str(data / 'grid_latents' / f'{source}_latent.npz'),
                          out_dir=str(event_dir))}
    commands = []
    def add(*args):
        commands.append([sys.executable, '-u', '-m', 'SHAPE', *map(str, args)])
    if fresh_grid:
        add('train-grid', '--checkpoint', training / 'grid.pth', '--history', training / 'grid_history.json',
            '--device', device, '--epochs', 200, '--seed', 42)
        add('extract-grid', '--checkpoint', training / 'grid.pth', '--output-dir', data / 'grid_latents', '--device', device)
        add('prepare-events', '--city', source, '--config', output / 'city_paths.json', '--seed', 42)
    add('train-tokenizer', '--city', source, '--data-dir', event_dir, '--device', device,
        '--quantizer-type', 'fsq', '--num-quantizers', 3, '--codebook-size', 64,
        '--fsq-levels', '4,4,4', '--jump-aux-weight', 0, '--poi-aux-weight', 0,
        '--time-aux-weight', 0, '--encoder-perturb-mode', 'iid', '--encoder-perturb-sigma', .1,
        '--epochs', 50, '--batch-size', 196, '--max-attention-cells', 8388608, '--seed', 42,
        '--save-dir', training, '--model-name', 'tokenizer.pth',
        '--history-name', 'tokenizer_history.json', '--plot-path', training / 'tokenizer.png')
    add('encode-source', '--source-city', source, '--data-dir', event_dir,
        '--checkpoint', training / 'tokenizer.pth', '--grid-csv', spec['source_grid'],
        '--output', data / 'source_codes.npz', '--device', device, '--batch-size', 1, '--seed', 42,
        '--expected-jump-aux-weight', 0, '--expected-poi-aux-weight', 0,
        '--expected-time-aux-weight', 0, '--require-no-time-aux', '--city-config', output / 'city_paths.json')
    pooled = [ROOT / 'dataset' / f'{city}_grid_merged_features.csv' for city in ('CA', 'NYC', 'PARIS')]
    add('train-landing', '--codes', data / 'source_codes.npz', '--grid-csv', spec['source_grid'],
        '--output', training / 'landing.pth', '--mode', 'l2_attributes', '--device', device,
        '--epochs', 20, '--batch-size', 4096, '--seed', 42)
    for path in pooled:
        commands[-1] += ['--normalization-grid-csv', str(path)]
    add('train-landing-weights', '--codes', data / 'source_codes.npz',
        '--landing-checkpoint', training / 'landing.pth', '--source-grid', spec['source_grid'],
        '--output', training / 'landing_weights', '--device', device, '--seed', 42)
    add('train-context', '--codes', data / 'source_codes.npz',
        '--landing-checkpoint', training / 'landing.pth', '--source-grid', spec['source_grid'],
        '--learned-landing-checkpoint', training / 'landing_weights/learned_landing.pth',
        '--pooled-grids', *pooled, '--output', training / 'context', '--device', device)
    return commands, paths


def train(spec, output, device, fresh_grid=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data').mkdir()
    (output / 'training').mkdir()
    commands, paths = training_plan(spec, output, device, fresh_grid)
    write_json(output / 'city_paths.json', paths)
    receipt = dict(status='running', commands=commands, completed_stages=0,
                   target_trajectories_used_for_training=False, training_seed=42)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        for index, command in enumerate(commands, 1):
            print(shlex.join(command), flush=True)
            with (output / f'stage_{index:02d}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env=env, check=True)
            receipt['completed_stages'] = index
        fitted = dict(spec, source_codes=str(output / 'data/source_codes.npz'),
                      tokenizer_checkpoint=str(output / 'training/tokenizer.pth'),
                      landing_checkpoint=str(output / 'training/landing.pth'),
                      learned_landing_checkpoint=str(output / 'training/landing_weights/learned_landing.pth'),
                      learned_context_checkpoint=str(output / 'training/context/learned_context.pth'))
        if fresh_grid:
            fitted.update(grid_checkpoint=str(output / 'training/grid.pth'),
                          source_events=str(output / 'data/events'))
        bound_keys = set(spec['expected_sha256']) | set(MODEL_INPUTS)
        if fresh_grid:
            bound_keys.add('grid_checkpoint')
        fitted['expected_sha256'] = {key: sha256_file(fitted[key]) for key in sorted(bound_keys)}
        write_json(output / 'transfer.json', fitted)
        receipt['status'] = 'complete'
        return fitted
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(output / 'training_manifest.json', receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--direction', choices=(*DIRECTIONS, 'all'), default='all')
    inputs.add_argument('--spec', help='Use a transfer.json produced by training for generation or evaluation')
    parser.add_argument('--mode', choices=('verify', 'generate', 'reproduce', 'train'), default='verify',
                        help='verify assets; generate five seeds; reproduce generates and evaluates; train refits Stage 2 and all Stage 3 weights, then generates and evaluates')
    parser.add_argument('--output', default='results/shape')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--fresh-grid', action='store_true', help='Also refit Stage 1 and prepare new source events; this is a fresh experiment')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.lanes < 1:
        parser.error('--lanes must be positive')
    if args.fresh_grid and args.mode != 'train':
        parser.error('--fresh-grid requires --mode train')
    torch.set_num_threads(1)
    directions = DIRECTIONS if args.direction == 'all' else (args.direction,)
    spec_paths = [Path(args.spec)] if args.spec else [ROOT / 'configs' / f'{d}.json' for d in directions]
    for spec_path in spec_paths:
        spec = read_spec(spec_path, require_fitted=args.mode in ('generate', 'reproduce'))
        direction = spec['direction']
        destination = Path(args.output).resolve() / direction
        if args.dry_run:
            print(f'{direction}: {args.mode}; latest learned context; five generation seeds')
            if args.mode == 'train':
                commands, _ = training_plan(spec, destination / 'fit', args.device, args.fresh_grid)
                for command in commands:
                    print(shlex.join(command))
            continue
        if args.mode == 'verify':
            from SHAPE.verify import verify
            verify(spec_path)
            continue
        check_inputs(spec, generation=args.mode == 'generate', require_fitted=args.mode != 'train')
        destination.mkdir(parents=True, exist_ok=False)
        if args.mode == 'train':
            spec = train(spec, destination / 'fit', args.device, args.fresh_grid)
        predictions = generate(spec, destination / 'generation', args.device, args.lanes)
        if args.mode != 'generate':
            evaluate(spec, predictions, destination / 'evaluation')


if __name__ == '__main__':
    main()
