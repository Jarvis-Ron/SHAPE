"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import OrderedDict
import math
import numpy as np
from SHAPE.stage3.sampler import _compare_aggregates, _period_index
from SHAPE.stage3.rank_semantics import _subset_sequences
from SHAPE.stage3.spatial_metrics import _coarse_zone_lookup, _metrics_with_ratios


TEN_METRICS = OrderedDict((
    ("M2_od_cpc", "M2 OD CPC ↑"),
    ("M2_od_pearson", "M2 Pearson ↑"),
    ("M1_visit_jsd", "M1 visit JSD ↓"),
    ("M3_period_space_jsd_mean", "Spatiotemporal JSD ↓"),
    ("A_visit_entropy_ratio", "Entropy ratio →1"),
    ("A_grid_coverage_ratio", "Coverage ratio →1"),
    ("A_radius_of_gyration_distribution_jsd", "RoG JSD ↓"),
    ("A_jump_distance_spectrum_jsd", "Jump-length JSD ↓"),
    ("A_jump_bin_distribution_jsd", "Jump-bin JSD ↓"),
    ("A_visited_poi_category_share_jsd", "POI-category JSD ↓"),
))


def population_aggregate(grid, offsets, minute, users, centers, vocabulary,
                         grid_spec=None):
    grid = np.asarray(grid, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    minute = np.asarray(minute, dtype=np.int64)
    users = np.asarray(users, dtype=np.str_)
    centers = np.asarray(centers, dtype=np.float64)
    vocabulary = np.asarray(vocabulary, dtype=np.int64)
    if len(offsets) != len(users) + 1 or offsets[0] != 0 \
            or offsets[-1] != len(grid) or len(minute) != len(grid) \
            or np.any(np.diff(offsets) < 1) \
            or np.any((minute < 0) | (minute >= 1440)):
        raise ValueError("population events, day offsets or local minutes are invalid")
    if not len(vocabulary) or np.any(np.diff(vocabulary) <= 0) \
            or centers.shape != (len(vocabulary), 2):
        raise ValueError("population static vocabulary is invalid")
    positions = np.searchsorted(vocabulary, grid)
    if np.any(positions >= len(vocabulary)) \
            or not np.array_equal(vocabulary[positions], grid):
        raise ValueError("population grid IDs left the emitted vocabulary")
    width = len(vocabulary)
    output = {
        "visits": np.bincount(positions, minlength=width).astype(np.float64),
        "periods": np.bincount(
            _period_index(minute).astype(np.int64) * width + positions,
            minlength=4 * width,
        ).reshape(4, width).astype(np.float64),
        "hours": np.bincount(minute // 60, minlength=24).astype(np.float64),
    }
    valid_right = np.ones(len(grid), dtype=bool)
    valid_right[offsets[:-1]] = False
    right = np.flatnonzero(valid_right)
    zones = _coarse_zone_lookup(vocabulary, grid_spec)
    od_ids = zones[positions[right - 1]] * 100 + zones[positions[right]]
    output["od"] = np.bincount(od_ids, minlength=10000).astype(np.float64)
    points = centers[positions]
    left_points, right_points = points[right - 1], points[right]
    mean_lat = np.radians(0.5 * (left_points[:, 0] + right_points[:, 0]))
    dy = (right_points[:, 0] - left_points[:, 0]) * 111.32
    dx = (right_points[:, 1] - left_points[:, 1]) * 111.32 * np.cos(mean_lat)
    distances = np.sqrt(dx * dx + dy * dy)
    edges = np.r_[np.linspace(0.0, 100.0, 51), np.inf]
    output["jump_hist"] = np.histogram(distances, bins=edges)[0].astype(np.float64)
    output["jump_bin_hist"] = np.histogram(
        distances, bins=[0.0, 1.0, 5.0, 20.0, np.inf]
    )[0].astype(np.float64)

    # Preserve the original per-user arithmetic and point order, including
    # users whose shell days came from different NYC donors.
    _, day_user = np.unique(users, return_inverse=True)
    event_user = np.repeat(day_user, np.diff(offsets))
    order = np.argsort(event_user, kind="stable")
    counts = np.bincount(event_user)
    ends = np.r_[0, np.cumsum(counts)]
    radii = []
    for start, end in zip(ends[:-1], ends[1:]):
        user_points = points[order[start:end]]
        centroid = user_points.mean(axis=0)
        north = (user_points[:, 0] - centroid[0]) * 111.32
        east = (user_points[:, 1] - centroid[1]) * 111.32 * math.cos(
            math.radians(float(centroid[0]))
        )
        radii.append(float(np.sqrt(np.mean(north * north + east * east))))
    output["radius_hist"] = np.histogram(radii, bins=edges)[0].astype(np.float64)
    return output


def primary_subset(payload, values):
    view = payload["view"]
    result = _subset_sequences(
        view["offsets"], view["users"], view["minute"], values,
        np.asarray(payload["lengths"]) >= 2,
    )
    if result is None:
        raise RuntimeError("the events_ge_2 primary stratum is empty")
    return result


def aggregate_subset(subset, runtime):
    return population_aggregate(
        subset["values"], subset["offsets"], subset["minute"], subset["users"],
        runtime["target_centers"], runtime["target_grid_ids"],
        runtime["target_static"].spec,
    )


def score_prediction_group(predictions, payload, runtime, reference):
    """E4 seed-sketch mean, then metric; coverage is never a seed union."""
    if not predictions:
        raise ValueError("prediction group is empty")
    pooled = None
    coverage = []
    for prediction in predictions:
        current = aggregate_subset(primary_subset(payload, prediction), runtime)
        coverage.append(np.count_nonzero(current["visits"]) / len(current["visits"]))
        if pooled is None:
            pooled = current
        else:
            for key in pooled:
                pooled[key] += current[key]
    for key in pooled:
        pooled[key] /= len(predictions)
    audit = {}
    point = _compare_aggregates(pooled, reference, runtime["target_group"],
                                mass_audit=audit)
    point["A_grid_coverage_predicted"] = float(np.mean(coverage))
    point = _metrics_with_ratios(point)
    if not np.isclose(audit["pred_mass"], audit["ref_mass"], rtol=1e-12, atol=1e-8):
        raise RuntimeError("exact shell failed to preserve OD transition mass")
    return point, audit
