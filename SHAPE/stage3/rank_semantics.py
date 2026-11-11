"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import hashlib
import json
import math
import numpy as np
import torch
from SHAPE.stage3.sampler import _distance_row_km, spatial_rank_table
from SHAPE.stage3.source_semantics import DISTANCE_LOG_EPSILON_KM, _load_codes_artifact
from SHAPE.stage3.prototype_semantics import apply_code_distance_mixture


SOURCE_CITY = "nyc"


TARGET_CITY = "paris"


PRIMARY_STRATUM = "events_ge_2"


RANK_BIN_EDGES = np.asarray((0.0, 1.0, 5.0, 20.0, np.inf), dtype=np.float64)


RANK_PIECEWISE_EDGES = np.asarray((
    0.0, 0.25, 0.50, 0.75, 1.0,
    2.0, 3.0, 4.0, 5.0,
    8.75, 12.50, 16.25, 20.0,
    50.0, 100.0, 250.0, np.inf,
), dtype=np.float64)


EVENT_COUNT_STRATA = (
    ("events_eq_1", "=1", 1, 1, "descriptive"),
    ("events_eq_2", "=2", 2, 2, "descriptive"),
    ("events_3_to_5", "3-5", 3, 5, "descriptive"),
    ("events_ge_6", ">=6", 6, None, "descriptive"),
    ("events_ge_2", ">=2", 2, None, "primary"),
    ("all", "all", 1, None, "composition_sensitive_descriptive"),
)


DETAILED_STRATA = tuple(item[0] for item in EVENT_COUNT_STRATA[:4])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _subset_sequences(offsets, users, minute, values, mask):
    offsets = np.asarray(offsets, dtype=np.int64)
    users = np.asarray(users, dtype=np.str_)
    minute = np.asarray(minute, dtype=np.int64)
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(users),) or len(offsets) != len(users) + 1:
        raise RuntimeError("P3-E1 sequence mask is not aligned")
    chosen = np.flatnonzero(mask)
    if not len(chosen):
        return None
    pieces = [
        np.arange(int(offsets[index]), int(offsets[index + 1]), dtype=np.int64)
        for index in chosen
    ]
    event_indices = np.concatenate(pieces)
    lengths = np.diff(offsets)[chosen]
    return {
        "offsets": np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum(lengths, dtype=np.int64),
        )),
        "users": users[chosen],
        "minute": minute[event_indices],
        "values": values[event_indices],
        "sequence_indices": chosen,
        "event_indices": event_indices,
        "day_count": int(len(chosen)),
        "user_count": int(len(set(users[chosen].tolist()))),
        "event_count": int(len(event_indices)),
    }


def fit_train_rank_distance_semantics(
        reference_codes, projected_grid, centers, grid_ids, codebook_size=64,
        *, weights=None, rank_index=None):
    """Fit code-conditioned Noulas spatial-rank mixtures on source train."""
    artifact = _load_codes_artifact(reference_codes)
    offsets = np.asarray(artifact["day_offsets"], dtype=np.int64)
    splits = np.asarray(artifact["day_splits"], dtype=np.str_)
    codes = np.asarray(artifact["codes"], dtype=np.int64)
    projected_grid = np.asarray(projected_grid, dtype=np.int64)
    vocabulary = np.asarray(grid_ids, dtype=np.int64)
    positions = np.searchsorted(vocabulary, projected_grid)
    safe = np.clip(positions, 0, max(0, len(vocabulary) - 1))
    if codes.shape != (len(projected_grid), 3) \
            or not np.all(vocabulary[safe] == projected_grid):
        raise RuntimeError("P3-E1 rank-semantics arrays are not aligned")
    ranks = rank_index if rank_index is not None else spatial_rank_table(
        centers, weights=weights, lazy=True)
    coarse = np.zeros((int(codebook_size), 4), dtype=np.int64)
    fine = np.zeros((int(codebook_size), 16), dtype=np.int64)
    support = np.zeros(int(codebook_size), dtype=np.int64)
    for split, start, end in zip(splits, offsets[:-1], offsets[1:]):
        if str(split) != "train":
            continue
        for event in range(int(start) + 1, int(end)):
            code = int(codes[event, 0])
            if not 0 <= code < int(codebook_size):
                raise RuntimeError("P3-E1 rank code is outside the codebook")
            rank = float(ranks[positions[event - 1], positions[event]])
            coarse_bin = int(np.searchsorted(
                RANK_BIN_EDGES[1:], rank, side="right"
            ))
            fine_bin = int(np.searchsorted(
                RANK_PIECEWISE_EDGES[1:], rank, side="right"
            ))
            coarse[code, min(coarse_bin, 3)] += 1
            fine[code, min(fine_bin, 15)] += 1
            support[code] += 1
    population_coarse = coarse.sum(axis=0).astype(np.float64)
    if population_coarse.sum() <= 0:
        raise RuntimeError("P3-E1 rank fit has no source-train transitions")
    population_coarse /= population_coarse.sum()
    probability = np.tile(population_coarse, (int(codebook_size), 1))
    for code in range(int(codebook_size)):
        if coarse[code].sum() > 0:
            probability[code] = coarse[code] / float(coarse[code].sum())
    piecewise = np.zeros_like(fine, dtype=np.float64)
    backoffs = 0
    population_fine = fine.sum(axis=0)
    for code in range(int(codebook_size)):
        for coarse_bin in range(4):
            left, right = coarse_bin * 4, (coarse_bin + 1) * 4
            local = fine[code, left:right].astype(np.float64)
            if local.sum() <= 0:
                local = population_fine[left:right].astype(np.float64)
                backoffs += 1
            if local.sum() <= 0:
                local = np.ones(4, dtype=np.float64)
            piecewise[code, left:right] = (
                probability[code, coarse_bin] * local / local.sum()
            )
    payload = {
        "probability": probability.tolist(),
        "piecewise": piecewise.tolist(),
        "support": support.tolist(),
    }
    return {
        "contract": (
            "source_train_destination_layer0_code_to_Noulas_per_origin_"
            "strictly_closer_spatial_rank_mixture_v1"
        ),
        "fit_split": "train_only",
        "excluded_splits": ["val", "test"],
        "first_event_excluded": True,
        "cross_day_transitions_excluded": True,
        "rank_definition": ("number_of_active_grid_centers_strictly_closer" if weights is None
                            else "sum_of_mean_normalized_static_POI_mass_strictly_closer"),
        "codebook_size": int(codebook_size),
        "rank_bin_edges": ["inf" if np.isposinf(x) else float(x)
                           for x in RANK_BIN_EDGES],
        "piecewise_rank_edges": ["inf" if np.isposinf(x) else float(x)
                                 for x in RANK_PIECEWISE_EDGES],
        "transition_count": int(support.sum()),
        "support": support,
        "physical_bin_probabilities": probability.astype(np.float32),
        "piecewise_probabilities": piecewise.astype(np.float32),
        "population_backoff_code_bin_count": int(backoffs),
        "table_sha256": hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }


def _payload_mixture(payload, scale_mode, runtime):
    spec = (
        runtime["target_native_spec"]
        if scale_mode == "target_native" else runtime["scale_specs"][scale_mode]
    )
    cached = payload["mixtures"].get(scale_mode)
    if cached is not None and cached["spec"] is spec:
        return cached
    physical, piecewise, audit = apply_code_distance_mixture(
        payload["codes"], spec["semantics"], payload["view"]["offsets"]
    )
    cached = {
        "physical": physical,
        "piecewise": piecewise,
        "audit": audit,
        "spec": spec,
    }
    payload["mixtures"][scale_mode] = cached
    return cached
