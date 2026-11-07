"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np
from SHAPE.stage3.sampler import _distance_row_km, attraction_distribution
from SHAPE.stage3.source_semantics import _load_codes_artifact


ATTRACTION_SMOOTHING = 1e-6


ATTRACTION_PURITY_THRESHOLD = 0.5


HOME_NIGHT_THRESHOLD = 0.5


def cohort_split_view(cohort, split):
    """Return complete sequences and event indices for one frozen split."""
    split = str(split)
    if split not in ("train", "test"):
        raise ValueError("P3-B2 split view must be train or test")
    splits = np.asarray(cohort["splits"], dtype=np.str_)
    users = np.asarray(cohort["users"], dtype=np.str_)
    dates = np.asarray(cohort["dates"], dtype=np.str_)
    offsets = np.asarray(cohort["offsets"], dtype=np.int64)
    if len(offsets) != len(splits) + 1 or len(users) != len(splits) \
            or len(dates) != len(splits):
        raise RuntimeError("P3-B2 cohort arrays are not day-aligned")
    selected = np.flatnonzero(splits == split)
    if not len(selected):
        raise RuntimeError("P3-B2 cohort has no %s days" % split)
    pieces = [
        np.arange(int(offsets[index]), int(offsets[index + 1]), dtype=np.int64)
        for index in selected
    ]
    event_indices = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    lengths = np.diff(offsets)[selected]
    result_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)
    ))
    selected_users = users[selected]
    if split == "test" and set(selected_users.tolist()) != set(users.tolist()):
        raise RuntimeError("P3-B2 every selected user must have heldout days")
    key_payload = "\n".join(
        "%s\t%s\t%s" % (users[index], dates[index], split)
        for index in selected
    ).encode("utf-8")
    return {
        "split": split,
        "sequence_indices": selected,
        "event_indices": event_indices,
        "offsets": result_offsets,
        "users": selected_users,
        "dates": dates[selected],
        "minute": np.asarray(cohort["minute"], dtype=np.int64)[event_indices],
        "day_count": int(len(selected)),
        "user_count": int(len(set(selected_users.tolist()))),
        "event_count": int(len(event_indices)),
        "key_sha256": hashlib.sha256(key_payload).hexdigest(),
    }




def apply_code_distance_mixture(codes, semantics, offsets=None):
    """Map event layer-0 codes to frozen four-bin and piecewise densities."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2 or codes.shape[1] != 3:
        raise RuntimeError("P3-B2 requires event-aligned three-layer codes")
    code0 = codes[:, 0]
    physical = np.asarray(semantics["physical_bin_probabilities"], dtype=np.float32)
    piecewise = np.asarray(semantics["piecewise_probabilities"], dtype=np.float32)
    if np.any(code0 < 0) or np.any(code0 >= len(physical)):
        raise RuntimeError("P3-B2 code is outside the frozen codebook")
    starts = np.zeros(len(codes), dtype=bool)
    if offsets is not None:
        starts[np.asarray(offsets, dtype=np.int64)[:-1]] = True
    return physical[code0], piecewise[code0], {
        "event_count": int(len(codes)),
        "transition_event_count": int(np.count_nonzero(~starts)),
        "mapping_sha256": semantics["table_sha256"],
        "source_fit_split": semantics["fit_split"],
    }


def fit_train_code_attraction_semantics(
    reference_codes, projected_grid, raw_grid_table, grid_ids,
    residential_grid_table=None,
    codebook_size=64,
):
    """Fit code-to-map attraction without heldout flow or personal features."""
    reference = _load_codes_artifact(reference_codes)
    required = ("event_user_ids", "event_splits", "minute", "poi_category")
    if any(name not in reference for name in required):
        raise RuntimeError("P3-B2 attraction fit is missing reference event fields")
    users = np.asarray(reference["event_user_ids"], dtype=np.str_)
    splits = np.asarray(reference["event_splits"], dtype=np.str_)
    minute = np.asarray(reference["minute"], dtype=np.int64)
    category = np.asarray(reference["poi_category"], dtype=np.int64)
    codes = np.asarray(reference["codes"], dtype=np.int64)
    projected_grid = np.asarray(projected_grid, dtype=np.int64)
    raw = np.maximum(np.asarray(raw_grid_table, dtype=np.float64), 0.0)
    grid_ids = np.asarray(grid_ids, dtype=np.int64)
    if codes.shape != (len(users), 3) or len(projected_grid) != len(users) \
            or len(minute) != len(users) or len(category) != len(users) \
            or raw.shape != (len(grid_ids), 20):
        raise RuntimeError("P3-B2 attraction arrays are not aligned")
    if np.any(category < 0) or np.any(category >= 13):
        raise RuntimeError("P3-B2 POI category must be in [0, 13)")

    home_counts = defaultdict(Counter)
    for user, split, grid in zip(users, splits, projected_grid):
        if str(split) == "train":
            home_counts[str(user)][int(grid)] += 1
    home_by_user = {
        user: sorted(values.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for user, values in home_counts.items() if values
    }
    category_counts = np.zeros((int(codebook_size), 13), dtype=np.int64)
    home_night = np.zeros(int(codebook_size), dtype=np.int64)
    support = np.zeros(int(codebook_size), dtype=np.int64)
    train_mask = splits == "train"
    for user, code_row, grid, event_minute, poi in zip(
        users[train_mask], codes[train_mask], projected_grid[train_mask],
        minute[train_mask], category[train_mask],
    ):
        code = int(code_row[0])
        if not 0 <= code < int(codebook_size):
            raise RuntimeError("P3-B2 layer-0 code is out of range")
        category_counts[code, int(poi)] += 1
        support[code] += 1
        is_night = int(event_minute) < 6 * 60 or int(event_minute) >= 22 * 60
        if is_night or int(grid) == int(home_by_user[str(user)]):
            home_night[code] += 1
    population_category = category_counts.sum(axis=0).astype(np.float64)
    if population_category.sum() <= 0:
        raise RuntimeError("P3-B2 attraction fit has no source-train events")
    population_category /= population_category.sum()
    poi = raw[:, :13]
    _, residential_score = attraction_distribution(
        raw if residential_grid_table is None else residential_grid_table,
        include_transit=False,
    )
    table = np.zeros((int(codebook_size), len(grid_ids)), dtype=np.float64)
    roles, dominant_categories, purity, category_weights = [], [], [], []
    for code in range(int(codebook_size)):
        counts = category_counts[code].astype(np.float64)
        weights = population_category if counts.sum() <= 0 else counts / counts.sum()
        dominant = int(np.argmax(weights))
        positive = weights > 0
        entropy = -float(np.sum(weights[positive] * np.log(weights[positive])))
        code_purity = float(1.0 - entropy / math.log(13.0))
        home_share = float(home_night[code] / max(1, support[code]))
        if support[code] > 0 and home_share >= HOME_NIGHT_THRESHOLD:
            role = "home_night_residential_proxy"
            score = residential_score
        elif support[code] > 0 and code_purity >= ATTRACTION_PURITY_THRESHOLD:
            role = "dominant_poi_column"
            score = poi[:, dominant]
        else:
            role = "ambiguous_general_poi_mixture"
            score = poi @ weights
        score = np.maximum(np.asarray(score, dtype=np.float64), 0.0)
        table[code] = score + ATTRACTION_SMOOTHING
        roles.append(role)
        dominant_categories.append(dominant)
        purity.append(code_purity)
        category_weights.append(weights)
    zero_mask = (poi.sum(axis=1) <= 1e-12) & (raw[:, 13] <= 1e-12)
    payload = {
        "roles": roles,
        "dominant_categories": dominant_categories,
        "support": support.tolist(),
        "table": table.tolist(),
    }
    return {
        "contract": (
            "source_train_layer0_code_to_map_only_poi_or_residential_"
            "attraction_with_general_mixture_v1"
        ),
        "fit_split": "train_only",
        "excluded_splits": ["val", "test"],
        "uses_personal_features": False,
        "uses_heldout_flow": False,
        "map_columns": ["poi_prop_%d" % index for index in range(13)],
        "residential_proxy": "existing_density_plus_poi_richness_no_transit",
        "smoothing": ATTRACTION_SMOOTHING,
        "purity_threshold": ATTRACTION_PURITY_THRESHOLD,
        "home_night_threshold": HOME_NIGHT_THRESHOLD,
        "support": support,
        "home_night_support": home_night,
        "dominant_poi_category": np.asarray(dominant_categories, dtype=np.int8),
        "category_probabilities": np.asarray(
            category_weights, dtype=np.float32
        ),
        "category_purity": np.asarray(purity, dtype=np.float32),
        "role": roles,
        "role_counts": dict(sorted(Counter(roles).items())),
        "attraction_table": table.astype(np.float32),
        "zero_attraction_grid_mask": zero_mask,
        "zero_attraction_grid_count": int(np.count_nonzero(zero_mask)),
        "zero_attraction_grid_fraction": float(np.mean(zero_mask)),
        "table_sha256": hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }
