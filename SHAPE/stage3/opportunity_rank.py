"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from collections import OrderedDict
import hashlib
from pathlib import Path
from threading import RLock
import numpy as np


def distance_row_km(center, centers):
    centers = np.asarray(centers, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    mean_lat = np.radians(0.5 * (centers[:, 0] + center[0]))
    north = (centers[:, 0] - center[0]) * 111.32
    east = (centers[:, 1] - center[1]) * 111.32 * np.cos(mean_lat)
    return np.sqrt(north * north + east * east)


def opportunity_weights(mass):
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1 or not len(mass) or not np.isfinite(mass).all() \
            or np.any(mass < 0) or mass.mean() <= 0:
        raise ValueError('opportunity mass must be finite, nonnegative and nonempty')
    return mass / mass.mean()


def static_poi_weights(grid_csv, grid_ids):
    """Use actual static POI counts, not the percentile-transformed density."""
    import pandas as pd
    path = Path(grid_csv)
    provenance = path.with_name(path.name.replace('_merged_features.csv', '_provenance.csv'))
    table = pd.read_csv(provenance)
    if 'poi_count_raw' not in table or 'grid_id' not in table:
        raise ValueError('static provenance must contain grid_id and poi_count_raw')
    if table.grid_id.duplicated().any():
        raise ValueError('duplicate static provenance grid ID')
    mass = table.set_index('grid_id').loc[np.asarray(grid_ids), 'poi_count_raw'].to_numpy(float)
    weights = opportunity_weights(mass)
    return weights, dict(source=str(provenance), column='poi_count_raw',
                         sha256=hashlib.sha256(provenance.read_bytes()).hexdigest(),
                         rows=len(weights), positive_mass_rows=int((mass > 0).sum()),
                         total_POIs=float(mass.sum()), mean_POIs=float(mass.mean()),
                         weight_mean=float(weights.mean()), target_checkins_used=False)


class SpatialRankIndex:
    """Exact bounded-memory origin rows; no N by N allocation.

    Stable sorting and a prefix sum retain exact strictly-closer tie semantics.
    Uniform weights follow the old searchsorted algorithm bit for bit.
    """
    def __init__(self, centers, weights=None, cache_rows=64):
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.ndim != 2 or self.centers.shape[1] != 2 \
                or not len(self.centers) or not np.isfinite(self.centers).all():
            raise ValueError('rank requires finite [grid,2] centers')
        self.weights = np.ones(len(self.centers), dtype=np.float64) if weights is None \
            else np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (len(self.centers),) or not np.isfinite(self.weights).all() \
                or np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise ValueError('rank weights must align with the static vocabulary')
        self.uniform = bool(np.all(self.weights == 1))
        self.cache_rows = max(0, int(cache_rows))
        self.cache = OrderedDict()
        self.cache_lock = RLock()
        self.shape = (len(self.centers), len(self.centers))
        self.computed_rows = 0
        self.observed_pairs = {}

    def row(self, origin):
        origin = int(origin)
        if not 0 <= origin < len(self.centers):
            raise IndexError('rank origin outside static vocabulary')
        with self.cache_lock:
            if origin in self.cache:
                self.cache.move_to_end(origin)
                return self.cache[origin]
        d = distance_row_km(self.centers[origin], self.centers)
        if self.uniform:
            values = np.searchsorted(np.sort(d, kind='stable'), d, side='left').astype(np.float32)
        else:
            order = np.argsort(d, kind='stable')
            positions = np.searchsorted(d[order], d, side='left')
            prefix = np.concatenate(([0.0], np.cumsum(self.weights[order], dtype=np.float64)))
            values = prefix[positions].astype(np.float32)
        with self.cache_lock:
            self.computed_rows += 1
            if self.cache_rows:
                self.cache[origin] = values
                while len(self.cache) > self.cache_rows:
                    self.cache.popitem(last=False)
        return values

    def __getitem__(self, item):
        origin, destination = item
        a, b = np.broadcast_arrays(np.asarray(origin), np.asarray(destination))
        if a.ndim == 0:
            key = (int(a), int(b))
            if key not in self.observed_pairs:
                self.observed_pairs[key] = self.row(key[0])[key[1]]
            return self.observed_pairs[key]
        out = np.empty(a.shape, dtype=np.float32)
        for value in np.unique(a):
            mask = a == value
            out[mask] = self.row(value)[b[mask]]
        return out

    def gather(self, origins, destinations):
        origins = np.asarray(origins, dtype=np.int64)
        destinations = np.asarray(destinations, dtype=np.int64)
        return np.stack([self.row(o)[d] for o, d in zip(origins, destinations)])

    def dense(self):
        return np.stack([self.row(i) for i in range(len(self.centers))])


def fit_personal_profiles(artifact, projected_grid, centers, grid_ids, *,
                         rank_index, bin_edges=(0, 1, 5, 20, np.inf), n0=20):
    """Estimate profiles from chronologically ordered source train events only."""
    from SHAPE.evaluation.prototype_atlas import event_revisit_flags
    offsets = np.asarray(artifact['day_offsets'], dtype=np.int64)
    splits = np.asarray(artifact['day_splits'], dtype=np.str_)
    users = np.asarray(artifact['day_user_ids'], dtype=np.str_)
    dates = np.asarray(artifact['day_dates'], dtype=np.str_)
    vocabulary = np.asarray(grid_ids, dtype=np.int64)
    positions = np.searchsorted(vocabulary, projected_grid)
    if np.any(positions >= len(vocabulary)) or not np.array_equal(vocabulary[positions], projected_grid):
        raise ValueError('source profile grid outside static vocabulary')
    counts, histories, distances = {}, {}, []
    coarse_total = np.zeros(4, dtype=np.int64)
    fine_edges = np.array([0, 1, 5, 20, 50, 100, 500, 1000, 5000, 20000, np.inf])
    diagnostic = np.zeros(len(fine_edges)-1, dtype=np.int64)
    selected = [i for i in range(len(users)) if splits[i] == 'train']
    selected.sort(key=lambda i: (str(users[i]), str(dates[i]), i))
    for day in selected:
        user = str(users[day])
        start, stop = int(offsets[day]), int(offsets[day + 1])
        counts.setdefault(user, np.zeros(4, dtype=np.int64))
        histories.setdefault(user, []).extend(np.asarray(projected_grid[start:stop]).tolist())
        for event in range(start + 1, stop):
            left, right = positions[event-1], positions[event]
            value = float(rank_index[left, right])
            bucket = min(3, int(np.searchsorted(np.asarray(bin_edges)[1:], value, side='right')))
            counts[user][bucket] += 1
            coarse_total[bucket] += 1
            diagnostic[min(len(diagnostic)-1, int(np.searchsorted(fine_edges[1:], value, side='right')))] += 1
            distances.append(float(distance_row_km(centers[left], np.asarray(centers)[[right]])[0]))
    if coarse_total.sum() == 0:
        raise ValueError('no source training transitions for personal profiles')
    population = coarse_total / float(coarse_total.sum())
    revisits = {}
    for user, history in histories.items():
        flags = event_revisit_flags(np.asarray(history))
        valid = flags[flags >= 0]
        revisits[user] = (int(valid.sum()), len(valid))
    estimable = [a/b for a,b in revisits.values() if b]
    mean_return = float(np.mean(estimable)) if estimable else 0.0
    residents = {}
    for user, count in counts.items():
        n = int(count.sum())
        shrunk = (count + n0 * population) / (n + n0)
        ratio = np.divide(shrunk, population, out=np.ones(4), where=population > 0)
        returns, observations = revisits[user]
        r_shrunk = (returns + n0 * mean_return) / (observations + n0)
        r_ratio = r_shrunk / mean_return if mean_return > 0 else 1.0
        residents[user] = dict(ring_counts=count.tolist(), ring_count=n,
            shrunk_ring_probability=shrunk.tolist(), ring_ratio=ratio.tolist(),
            revisit_count=returns, revisit_observations=observations,
            shrunk_revisit_rate=r_shrunk, revisit_ratio=r_ratio)
    positive = np.asarray(distances)[np.asarray(distances) > 0]
    if not len(positive):
        raise ValueError('cannot fit distance cost without positive source jumps')
    xmin = float(positive.min())
    denominator = float(np.log(positive / xmin).sum())
    gamma = 1.0 + len(positive)/denominator if denominator > 0 else None
    return dict(schema='personal_scale_profile_v1', fit_split='source_train_only',
                n0=n0, population_ring_counts=coarse_total.tolist(),
                population_ring_probability=population.tolist(), mean_return_rate=mean_return,
                revisit_definition='window20_burn10_per_resident_train_chronology',
                residents=residents, gamma=gamma, cost_xmin_km=xmin,
                positive_cost_observations=len(positive),
                diagnostic_rank_edges=[float(x) if np.isfinite(x) else 'inf' for x in fine_edges],
                diagnostic_rank_counts=diagnostic.tolist())


def profile_event_parameters(profile, users, offsets, exponent, kappa, beta0=1.0):
    """Expand immutable resident parameters onto this landing call's events."""
    users = np.asarray(users, dtype=np.str_)
    offsets = np.asarray(offsets, dtype=np.int64)
    ring = np.ones((int(offsets[-1]), 4), dtype=np.float32)
    beta = np.full(len(users), float(beta0), dtype=np.float32)
    for index, user in enumerate(users):
        row = profile['residents'].get(str(user))
        if row is None:
            continue
        if exponent != 0:
            ring[offsets[index]:offsets[index+1]] = np.asarray(row['ring_ratio']) ** float(exponent)
        if kappa != 0:
            beta[index] *= float(row['revisit_ratio']) ** float(kappa)
    return ring, beta


def tilt_ring_probabilities(probability, multiplier):
    probability = np.asarray(probability, dtype=np.float32)
    multiplier = np.asarray(multiplier, dtype=np.float32)
    if probability.shape != multiplier.shape or np.any(multiplier < 0) or not np.isfinite(multiplier).all():
        raise ValueError('personal ring multipliers must be aligned, finite and nonnegative')
    if np.all(multiplier == 1):
        return probability.copy()
    weighted = probability.astype(np.float64) * multiplier
    denominator = weighted.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError('personal profile removed every supported ring')
    return (weighted / denominator).astype(np.float32)
