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
    'prepare-reference': ('reference', 'main'),
    'run': ('stage3.run', 'main'),
    'pipeline': ('pipeline', 'main'),
    'verify-reference': ('verify', 'main'),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print('SHAPE code: opportunity-weighted rank + distance cost (lambda=kappa=0)')
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
