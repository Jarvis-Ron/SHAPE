"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import numpy as np


def _unique_user_values(sequence_users, sequence_values):
    users = np.asarray(sequence_users, dtype=np.str_)
    values = np.asarray(sequence_values, dtype=np.int64)
    if users.shape != values.shape:
        raise RuntimeError("user/value arrays must be sequence-aligned")
    mapping = {}
    for user, value in zip(users, values):
        user, value = str(user), int(value)
        if user in mapping and mapping[user] != value:
            raise RuntimeError("one user has inconsistent train-modal homes")
        mapping[user] = value
    ordered_users = np.asarray(sorted(mapping), dtype=np.str_)
    ordered_values = np.asarray(
        [mapping[str(user)] for user in ordered_users], dtype=np.int64
    )
    return ordered_users, ordered_values


def _broadcast_user_values(sequence_users, unique_users, unique_values):
    unique_users = np.asarray(unique_users, dtype=np.str_)
    unique_values = np.asarray(unique_values, dtype=np.int64)
    if unique_users.shape != unique_values.shape \
            or len(set(unique_users.tolist())) != len(unique_users):
        raise RuntimeError("unique-user assignments are invalid")
    mapping = dict(zip(unique_users.tolist(), unique_values.tolist()))
    try:
        return np.asarray(
            [mapping[str(user)] for user in np.asarray(sequence_users)],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError("home assignment lacks a selected user") from exc


def proportional_integer_capacities(probabilities, total):
    """Hamilton allocation with stable grid-index ties and exact total."""
    probability = np.maximum(np.asarray(probabilities, dtype=np.float64), 0.0)
    total = int(total)
    if probability.ndim != 1 or not len(probability) or total < 1 \
            or not np.isfinite(probability).all() or probability.sum() <= 0:
        raise ValueError("quota probabilities/total are invalid")
    probability /= probability.sum()
    expected = probability * total
    capacities = np.floor(expected).astype(np.int64)
    remainder = total - int(capacities.sum())
    if remainder:
        order = np.lexsort((np.arange(len(probability)), -(expected - capacities)))
        capacities[order[:remainder]] += 1
    if int(capacities.sum()) != total:
        raise RuntimeError("proportional quota allocation lost capacity")
    return capacities
