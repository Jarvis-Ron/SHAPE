"""Export recorded landing losses without smoothing or invented epochs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ('nyc_to_ca', 'nyc_to_paris', 'ca_to_nyc', 'ca_to_paris')
LABELS = ('NYC → CA', 'NYC → Paris', 'CA → NYC', 'CA → Paris')


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), constrained_layout=True)
    rows = []
    for direction, label, ax in zip(DIRECTIONS, LABELS, axes.flat):
        data = json.loads((ROOT / 'training_loss' / direction / 'landing_history.json').read_text())
        history = data['history']
        origin = 'representative_rerun' if direction == 'nyc_to_paris' else 'original_training_log'
        epochs = [row['epoch'] for row in history]
        for split, color in [('train', '#526d82'), ('val', '#ae8267')]:
            ax.plot(epochs, [row[split]['loss'] for row in history], label=split.capitalize(),
                    color=color, linewidth=2)
        for row in history:
            rows.append(dict(direction=direction, epoch=row['epoch'], train_loss=row['train']['loss'],
                             validation_loss=row['val']['loss'], history_origin=origin, training_seed=42))
        ax.set(title=label + (' (rerun)' if direction == 'nyc_to_paris' else ''),
               xlabel='Epoch', ylabel='Landing cross-entropy')
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    folder = ROOT / 'training_loss'
    with (folder / 'landing_loss.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig.savefig(folder / 'landing_loss.png', dpi=220, metadata={'Software': 'SHAPE'})
    fig.savefig(folder / 'landing_loss.pdf', metadata={'Author': '', 'Creator': 'SHAPE',
                                                    'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
