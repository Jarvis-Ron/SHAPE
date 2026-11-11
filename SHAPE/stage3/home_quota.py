"""GHS-POP R2023A (2010 residential population) home-allocation quotas."""
import json
from pathlib import Path

import numpy as np


HOME_QUOTA_SCHEMA = 'ghs_pop_home_quota_v1'
GHS_POP_PROVENANCE = {
    'dataset': 'GHS-POP',
    'release': 'R2023A',
    'population_year': 2010,
    'population_layer': 'residential',
    'target_checkins_used': False,
}


def quota_probability(quota, vocabulary, *, target_city=None):
    """Normalize precomputed public-population quotas aligned to the static map.

    ``home_counts`` are integer allocation weights, not observed target users
    or raw raster population totals. Hamilton allocation later scales their
    normalized mass to the number of source users being transferred.
    """
    if not isinstance(quota, dict) or set(quota) != {
            'schema', 'grid_ids', 'home_counts', 'provenance'} \
            or quota['schema'] != HOME_QUOTA_SCHEMA:
        raise ValueError('Expected a GHS-POP home quota; use dataset/home_quotas/*.json')
    provenance = quota['provenance']
    if not isinstance(provenance, dict) or any(
            type(provenance.get(key)) is not type(value) or provenance[key] != value
            for key, value in GHS_POP_PROVENANCE.items()):
        raise ValueError('Home quota must identify GHS-POP R2023A 2010 residential population')
    if target_city is not None and provenance.get('target_city') != target_city:
        raise ValueError('Home quota target city does not match the static map')
    raw_ids = np.asarray(quota['grid_ids'])
    ids = np.asarray(quota['grid_ids'], dtype=np.int64)
    counts = np.asarray(quota['home_counts'], dtype=np.float64)
    if ids.ndim != 1 or not len(ids) or raw_ids.dtype.kind not in 'iu' \
            or not np.array_equal(raw_ids, ids) or np.any(np.diff(ids) <= 0) \
            or not np.array_equal(ids, vocabulary) or counts.shape != ids.shape \
            or not np.isfinite(counts).all() or np.any(counts < 0) \
            or np.any(counts != np.floor(counts)) or not np.isfinite(counts.sum()) \
            or counts.sum() <= 0 or provenance.get('quota_total') != counts.sum():
        raise ValueError('Invalid or unaligned GHS-POP home quota')
    return counts / counts.sum()


def load_home_quota(path, *, target_city):
    """Read the configured quota file without opening any target trajectories."""
    quota = json.loads(Path(path).read_text(encoding='utf-8'))
    vocabulary = quota.get('grid_ids', []) if isinstance(quota, dict) else []
    quota_probability(quota, vocabulary, target_city=target_city)
    return quota
