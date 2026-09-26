"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import numpy as np


FEATURE_COLUMNS = {
    "weekday": 17,
    "hour": 18,
    "grid_id": 123,
    "poi_category": 124,
    "jump_distance": 127,
}


POI_LABELS = (
    "shop",
    "professional_service",
    "personal_care",
    "medical",
    "transport",
    "education",
    "food_social",
    "arts_entertainment",
    "civic_residential",
    "outdoor_recreation",
    "historic_place",
    "neighborhood_nature",
    "home_lodging",
)


DEFAULT_JUMP_THRESHOLDS = (0.05, 0.15, 0.30)


REVISIT_WINDOW = 20


REVISIT_BURN_IN = 10


def decode_weekday(raw_weekday):
    """Decode the event schema's weekday/7 representation to integers 0..6.

    Older derived datasets occasionally stored raw weekday ids.  Supporting
    both forms keeps the evaluator honest across archived checkpoints while
    retaining strict range validation.
    """
    values = np.asarray(raw_weekday, dtype=np.float64)
    finite = np.isfinite(values)
    decoded = np.full(values.shape, -1, dtype=np.int16)
    if not finite.any():
        return decoded
    finite_values = values[finite]
    if finite_values.min() >= -1e-6 and finite_values.max() <= 1.0 + 1e-6:
        finite_values = finite_values * 7.0
    rounded = np.rint(finite_values).astype(np.int16)
    valid = (rounded >= 0) & (rounded <= 6)
    finite_indices = np.flatnonzero(finite)
    decoded[finite_indices[valid]] = rounded[valid]
    return decoded


def event_revisit_flags(
        grid_ids, window=REVISIT_WINDOW, burn_in=REVISIT_BURN_IN):
    """Label bounded revisits and mark the burn-in prefix as invalid.

    An event is a revisit when its grid occurs among the preceding ``window``
    events in the same trajectory. The first ``burn_in`` events carry ``-1``
    so they cannot contribute to a revisit profile.
    """
    if int(window) != window or int(window) < 1:
        raise ValueError("revisit window must be a positive integer")
    if int(burn_in) != burn_in or int(burn_in) < 0:
        raise ValueError("revisit burn-in must be a non-negative integer")
    window = int(window)
    burn_in = int(burn_in)
    grids = np.asarray(grid_ids)
    revisit = np.full(len(grids), -1, dtype=np.int8)
    for index, raw_grid in enumerate(grids):
        if index < burn_in or not np.isfinite(raw_grid):
            continue
        grid = int(round(float(raw_grid)))
        if grid < 0:
            continue
        history = grids[max(0, index - window):index]
        valid_history = history[np.isfinite(history)]
        if valid_history.size:
            valid_history = np.rint(valid_history).astype(np.int64)
            valid_history = valid_history[valid_history >= 0]
        revisit[index] = int(np.any(valid_history == grid))
    return revisit


def extract_event_attributes(sequence):
    """Extract atlas attributes from one valid, unpadded trajectory."""
    sequence = np.asarray(sequence)
    if sequence.ndim != 2 or sequence.shape[1] <= max(FEATURE_COLUMNS.values()):
        raise ValueError("sequence must have shape [events, >=128]")

    hours = np.rint(sequence[:, FEATURE_COLUMNS["hour"]]).astype(np.int16)
    hours[(hours < 0) | (hours > 23)] = -1
    weekdays = decode_weekday(sequence[:, FEATURE_COLUMNS["weekday"]])
    poi = np.rint(sequence[:, FEATURE_COLUMNS["poi_category"]]).astype(np.int16)
    poi[(poi < 0) | (poi >= len(POI_LABELS))] = -1
    jump = sequence[:, FEATURE_COLUMNS["jump_distance"]].astype(np.float32)
    jump[~np.isfinite(jump)] = np.nan
    jump[(jump < 0.0) | (jump > 1.0 + 1e-5)] = np.nan
    grids = np.rint(sequence[:, FEATURE_COLUMNS["grid_id"]]).astype(np.int64)
    grids[grids < 0] = -1

    return {
        "hour": hours,
        "weekday": weekdays,
        "poi_category": poi,
        "jump_distance": jump,
        "grid_id": grids,
        "revisit": event_revisit_flags(grids),
    }


def categorize_jump_distances(distances, thresholds=DEFAULT_JUMP_THRESHOLDS):
    """Map normalized grid jump distances to five shared, fixed bins.

    Zero has its own stationary bin.  Positive distances are split at 0.05,
    0.15, and 0.30 of the GridSpec-normalized city diagonal, so cities use
    exactly the same support instead of city-specific quantiles.
    """
    thresholds = tuple(float(value) for value in thresholds)
    if len(thresholds) != 3 or any(value <= 0 for value in thresholds):
        raise ValueError("jump thresholds must contain three positive values")
    if tuple(sorted(thresholds)) != thresholds:
        raise ValueError("jump thresholds must be strictly increasing")

    values = np.asarray(distances, dtype=np.float64)
    bins = np.full(values.shape, -1, dtype=np.int16)
    finite = np.isfinite(values) & (values >= 0.0)
    stationary = finite & (values <= 1e-8)
    moving = finite & ~stationary
    bins[stationary] = 0
    bins[moving] = 1 + np.digitize(
        values[moving], thresholds, right=True
    )
    return bins
