"""Run the documented SHAPE stages: python -m SHAPE <command> --help."""
import importlib
import sys

COMMANDS = {
    'train-grid': ('stage1.train_grid_embeddings', 'main'),
    'extract-grid': ('stage1.extract_embeddings', 'main'),
    'prepare-events': ('stage2.prepare_city_event_data', 'main'),
    'train-tokenizer': ('stage2.train_tokenizer', 'main'),
    'encode-source': ('stage3.prepare_codes', 'main'),
    'train-landing': ('stage3.landing_head', 'train_main'),
    'train-landing-weights': ('stage3.learned_landing', 'main'),
    'prepare-reference': ('reference', 'main'),
    'prepare-population': ('population', 'main'),
    'train-context': ('stage3.learned_context', 'main'),
    'run': ('reproduce', 'main'),
    'pipeline': ('pipeline', 'main'),
    'verify': ('verify', 'main'),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print('SHAPE code: residual FSQ without layer supervision; opportunity-weighted rank without the Pareto distance-cost term (lambda=kappa=0)')
        print('Usage: python -m SHAPE <command> [options]\nCommands:')
        for name in COMMANDS:
            print('  ' + name)
        return
    name = argv.pop(0)
    if name not in COMMANDS:
        raise SystemExit('Unknown SHAPE command: ' + name)
    module, entry = COMMANDS[name]
    return getattr(importlib.import_module('SHAPE.' + module), entry)(argv)


if __name__ == '__main__':
    main()
