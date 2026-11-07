"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from SHAPE.stage3.contracts import CODES_SCHEMA, load_npz, require_scalar
from SHAPE.stage3.sampler import _distance_row_km


PHYSICAL_JUMP_EDGES_KM = np.asarray(
    [0.0, 1.0, 5.0, 20.0, np.inf], dtype=np.float64
)


DISTANCE_LOG_EPSILON_KM = 0.05


def _load_codes_artifact(reference_codes):
    artifact = (
        load_npz(reference_codes)
        if isinstance(reference_codes, (str, os.PathLike, Path))
        else reference_codes
    )
    require_scalar(artifact, "schema", CODES_SCHEMA)
    return artifact


def fit_train_code_distance_semantics(
    reference_codes,
    projected_grid,
    centers,
    grid_ids,
    codebook_size=64,
    log_epsilon_km=DISTANCE_LOG_EPSILON_KM,
):
    """Freeze d_code from within-day source-train transitions only.

    The destination event's layer-0 code indexes a robust geometric median of
    physical kilometres.  Unseen codes receive the source-train population
    median.  No validation/test grid or jump label participates.
    """
    artifact = _load_codes_artifact(reference_codes)
    offsets = np.asarray(artifact["day_offsets"], dtype=np.int64)
    day_splits = np.asarray(artifact["day_splits"], dtype=np.str_)
    codes = np.asarray(artifact["codes"], dtype=np.int64)
    projected_grid = np.asarray(projected_grid, dtype=np.int64)
    vocabulary = np.asarray(grid_ids, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64)
    if codes.shape != (len(projected_grid), 3) \
            or offsets[0] != 0 or offsets[-1] != len(codes) \
            or len(day_splits) != len(offsets) - 1:
        raise RuntimeError("P3-B1 reference code arrays are not day-aligned")
    if not np.isfinite(log_epsilon_km) or log_epsilon_km <= 0:
        raise ValueError("distance log epsilon must be finite and positive")
    positions = np.searchsorted(vocabulary, projected_grid)
    safe = np.clip(positions, 0, max(0, len(vocabulary) - 1))
    if np.any(projected_grid < 0) or not np.all(vocabulary[safe] == projected_grid):
        raise RuntimeError("P3-B1 projected source grids changed vocabulary")

    by_code = [[] for _ in range(int(codebook_size))]
    bin_counts = np.zeros((int(codebook_size), 4), dtype=np.int64)
    all_distances = []
    transition_count = 0
    for split, start, end in zip(day_splits, offsets[:-1], offsets[1:]):
        if str(split) != "train":
            continue
        start, end = int(start), int(end)
        for event in range(start + 1, end):
            code = int(codes[event, 0])
            if not 0 <= code < int(codebook_size):
                raise RuntimeError("P3-B1 source-train layer-0 code is out of range")
            distance = float(_distance_row_km(
                centers[positions[event - 1]], centers[[positions[event]]]
            )[0])
            by_code[code].append(distance)
            all_distances.append(distance)
            jump_bin = int(np.searchsorted(
                PHYSICAL_JUMP_EDGES_KM[1:], distance, side="right"
            ))
            bin_counts[code, min(jump_bin, 3)] += 1
            transition_count += 1
    if not all_distances:
        raise RuntimeError("P3-B1 needs source-train within-day transitions")

    def geometric_median(values):
        values = np.asarray(values, dtype=np.float64)
        return max(0.0, float(
            np.exp(np.median(np.log(values + float(log_epsilon_km))))
            - float(log_epsilon_km)
        ))

    fallback_distance = geometric_median(all_distances)
    fallback_bins = bin_counts.sum(axis=0).astype(np.float64)
    fallback_bins /= fallback_bins.sum()
    target = np.full(int(codebook_size), fallback_distance, dtype=np.float32)
    conditional = np.tile(fallback_bins, (int(codebook_size), 1))
    support = np.asarray([len(values) for values in by_code], dtype=np.int64)
    for code, values in enumerate(by_code):
        if not values:
            continue
        target[code] = geometric_median(values)
        conditional[code] = bin_counts[code] / float(bin_counts[code].sum())
    dominant_bin = np.argmax(conditional, axis=1).astype(np.int8)
    table_payload = {
        "target_distance_km": target.tolist(),
        "support": support.tolist(),
        "physical_bin_probabilities": conditional.tolist(),
    }
    table_sha = hashlib.sha256(json.dumps(
        table_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "contract": (
            "source_train_within_day_destination_layer0_code_to_"
            "physical_km_geometric_median_v1"
        ),
        "fit_split": "train_only",
        "excluded_splits": ["val", "test"],
        "cross_day_transitions_excluded": True,
        "first_event_excluded": True,
        "codebook_size": int(codebook_size),
        "distance_log_epsilon_km": float(log_epsilon_km),
        "physical_jump_edges_km": [0.0, 1.0, 5.0, 20.0, "inf"],
        "transition_count": int(transition_count),
        "seen_code_count": int(np.count_nonzero(support)),
        "unseen_code_count": int(np.count_nonzero(support == 0)),
        "population_fallback_distance_km": float(fallback_distance),
        "population_fallback_bin_probabilities": fallback_bins.tolist(),
        "target_distance_km": target,
        "support": support,
        "physical_bin_probabilities": conditional,
        "dominant_physical_bin": dominant_bin,
        "table_sha256": table_sha,
    }


def derive_projected_train_homes(
    reference_codes, projected_grid, sequence_users
):
    """Resolve modal train homes after vocabulary projection with stable ties."""
    artifact = _load_codes_artifact(reference_codes)
    event_users = np.asarray(artifact["event_user_ids"], dtype=np.str_)
    event_splits = np.asarray(artifact["event_splits"], dtype=np.str_)
    projected_grid = np.asarray(projected_grid, dtype=np.int64)
    if len(projected_grid) != len(event_users):
        raise RuntimeError("P3-B1 projected home arrays are not event-aligned")
    counts = defaultdict(Counter)
    for user, split, grid in zip(event_users, event_splits, projected_grid):
        if str(split) == "train":
            counts[str(user)][int(grid)] += 1
    homes = []
    mapping = {}
    for user in np.asarray(sequence_users, dtype=np.str_):
        ranking = sorted(
            counts[str(user)].items(), key=lambda item: (-item[1], item[0])
        )
        if not ranking:
            raise RuntimeError(
                "P3-B1 selected user has no projected train home: %s" % user
            )
        home = int(ranking[0][0])
        homes.append(home)
        mapping[str(user)] = home
    homes = np.asarray(homes, dtype=np.int64)
    return homes, {
        "contract": "projected_train_mode_count_desc_grid_id_asc_v1",
        "fit_split": "train_only",
        "sequence_count": int(len(homes)),
        "unique_user_count": int(len(mapping)),
        "user_home_sha256": hashlib.sha256(json.dumps(
            mapping, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }
