"""Explicit, synchronous training-to-evaluation command graph for SHAPE.

The historical result starts from prepared source event embeddings. Fresh raw
preparation is also supported; it produces new bindings and is not presented as
a bit-for-bit recreation of those historical upstream embeddings.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def build_plan(spec, output, device='cpu'):
    output = Path(output).resolve()
    source, target = spec['source_city'], spec['target_city']
    if source not in ('nyc','ca') or target not in ('nyc','ca','paris') or source == target:
        raise ValueError('Unsupported transfer direction')
    data = output/'data'
    training = output/'training'
    commands = []
    def add(*arguments):
        commands.append([sys.executable,'-u','-m','SHAPE',*map(str,arguments)])
    if spec.get('prepare_static_and_source_events', False):
        add('train-grid','--checkpoint',training/'grid.pth','--history',training/'grid_history.json','--device',device)
        add('extract-grid','--checkpoint',training/'grid.pth','--output-dir','SHAPE/stage1/output','--device',device)
        add('prepare-events','--city',source)
        if spec.get('city_paths_config'):
            commands[-1].extend(['--config',str(spec['city_paths_config'])])
    event_dir = spec['source_event_dir']
    tokenizer = training/'tokenizer.pth'
    codes = data/'source_codes.npz'
    landing = training/'landing.pth'
    add('train-tokenizer','--city',source,'--data-dir',event_dir,'--device',device,
        '--quantizer-type','fsq','--injection-mode','layered','--num-quantizers','3',
        '--codebook-size','64','--fsq-levels','4,4,4','--jump-aux-weight','0.125',
        '--poi-aux-weight','0.075','--time-aux-weight','0',
        '--encoder-perturb-mode','iid','--encoder-perturb-sigma','0.1',
        '--epochs','50','--batch-size','196','--max-attention-cells','8388608',
        '--seed','42','--save-dir',training,'--model-name','tokenizer.pth',
        '--history-name','tokenizer_history.json','--plot-path',training/'tokenizer.png')
    add('encode-source','--source-city',source,'--data-dir',event_dir,
        '--checkpoint',tokenizer,'--grid-csv',spec['source_grid'],'--output',codes,
        '--device',device,'--batch-size','1','--seed','42',
        '--expected-jump-aux-weight','0.125','--expected-poi-aux-weight','0.075',
        '--expected-time-aux-weight','0','--require-no-time-aux')
    if spec.get('city_paths_config'):
        commands[-1].extend(['--city-config',str(spec['city_paths_config'])])
    add('train-landing','--codes',codes,'--grid-csv',spec['source_grid'],
        '--output',landing,'--mode','l2_attributes','--device',device,
        '--epochs','20','--batch-size','4096','--seed','42')
    for path in spec['normalization_grids']:
        commands[-1].extend(['--normalization-grid-csv',str(path)])
    target_reference = spec.get('target_reference')
    if not target_reference:
        target_reference = str(data/'target_reference.npz')
        add('prepare-reference','--city',target,'--grid-csv',spec['target_grid'],
            '--output',target_reference)
        if spec.get('city_paths_config'):
            commands[-1].extend(['--config',str(spec['city_paths_config'])])
        if spec.get('target_raw_checkins'):
            commands[-1].extend(['--raw-checkins',*spec['target_raw_checkins']])
    run_spec = dict(source_city=source,target_city=target,source_codes=str(codes),
        landing_checkpoint=str(landing),source_grid=spec['source_grid'],target_grid=spec['target_grid'],
        target_reference=target_reference,shell_length_policy=spec['shell_length_policy'])
    add('run','--spec',output/'transfer.json','--output',output/'transfer',
        '--device',device,'--lanes','3','--score-workers',str(spec.get('score_workers',8)))
    return commands, run_spec


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    spec=json.loads(Path(args.spec).read_text())
    for key in ('source_event_dir','source_grid','target_grid','target_reference','city_paths_config'):
        if spec.get(key):
            spec[key]=str(Path(spec[key]).resolve())
    for key in ('normalization_grids','target_raw_checkins'):
        if spec.get(key):
            spec[key]=[str(Path(value).resolve()) for value in spec[key]]
    commands,run_spec=build_plan(spec,args.output,args.device)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new pipeline output directory')
    (output/'data').mkdir(parents=True)
    (output/'training').mkdir()
    (output/'transfer.json').write_text(json.dumps(run_spec,indent=2)+'\n')
    manifest=dict(status='running',commands=commands,completed_stages=0)
    try:
        for index,command in enumerate(commands):
            print(shlex.join(command),flush=True)
            with (output/f'stage_{index+1:02d}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            manifest['completed_stages']=index+1
        manifest['status']='complete'
    except BaseException as error:
        manifest.update(status='failed',error=repr(error))
        raise
    finally:
        (output/'pipeline_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
