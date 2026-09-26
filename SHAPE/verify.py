"""Recompute the published row from stored draws, without inventing a rerun."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

FIVE = {
    'M2_od_cpc': 'OD CPC',
    'M3_period_space_jsd_mean': 'Spatiotemporal JSD',
    'A_radius_of_gyration_distribution_jsd': 'RoG JSD',
    'A_jump_distance_spectrum_jsd': 'Jump-length JSD',
    'A_visited_poi_category_share_jsd': 'POI-category JSD',
}


def verify(root):
    root = Path(root)
    expected = json.loads((root/'result.json').read_text())
    with (root/'metric_draws.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    output = {'method': 'SHAPE nyc→ca'}
    for key, label in FIVE.items():
        selected = [row for row in rows if row['metric'] == key]
        if sorted(int(row['repeat']) for row in selected) != list(range(200)):
            raise ValueError('Missing or duplicate shell draws: ' + key)
        values = np.asarray([float(row['value']) for row in selected])
        mean, std = float(values.mean()), float(values.std(ddof=1))
        if not np.isfinite(values).all() or not np.isclose(mean, expected['means'][key], atol=1e-14, rtol=0) or not np.isclose(std, expected['stds'][key], atol=1e-14, rtol=0):
            raise ValueError('Historical numerical reference changed: ' + key)
        output[label] = f'{mean:.4f} ± {std:.4f}'
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=str(Path(__file__).parents[1]/'provenance/nyc_to_ca'))
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.reference), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
