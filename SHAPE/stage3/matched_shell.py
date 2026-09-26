"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import hashlib
import json
from collections import Counter
import numpy as np


def build_composition_resample_plan(
    source_lengths, target_lengths, *, repeats, seed, target_users=None
):
    """Sample whole source days into the exact target shell structure."""
    source_lengths = np.asarray(source_lengths, dtype=np.int64)
    target_lengths = np.asarray(target_lengths, dtype=np.int64)
    repeats = int(repeats)
    if (
        source_lengths.ndim != 1
        or target_lengths.ndim != 1
        or not len(source_lengths)
        or not len(target_lengths)
        or np.any(source_lengths < 1)
        or np.any(target_lengths < 1)
        or repeats < 1
    ):
        raise ValueError("composition-resample lengths are invalid")
    if target_users is not None:
        target_users = np.asarray(target_users, dtype=np.str_)
        if target_users.shape != target_lengths.shape:
            raise ValueError("target users and target day lengths are misaligned")
        person_day_counts = Counter(target_users.tolist())
    else:
        person_day_counts = None
    donors = {
        int(length): np.flatnonzero(source_lengths == length)
        for length in np.unique(target_lengths)
    }
    missing = [length for length, values in donors.items() if not len(values)]
    if missing:
        raise RuntimeError(
            f"X1 has no whole-day donor for Paris event counts: {missing}"
        )
    plans = np.empty((repeats, len(target_lengths)), dtype=np.int64)
    for repeat in range(repeats):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                (
                    int(seed),
                    int(repeat),
                    8849,
                )
            )
        )
        for length, pool in donors.items():
            positions = np.flatnonzero(target_lengths == length)
            plans[repeat, positions] = rng.choice(
                pool, size=len(positions), replace=True
            )
    if not np.all(source_lengths[plans] == target_lengths[None, :]):
        raise RuntimeError("composition resampling changed an event count")
    usage = np.bincount(plans.ravel(), minlength=len(source_lengths))
    return plans, {
        "contract": (
            "whole_X1_predicted_day_exact_event_count_bootstrap_into_"
            "Paris_test_person_day_shell_v1"
        ),
        "repeats": repeats,
        "seed": int(seed),
        "source_day_count": len(source_lengths),
        "target_day_count_per_repeat": len(target_lengths),
        "target_event_count_per_repeat": int(target_lengths.sum()),
        "sampling_with_replacement": True,
        "same_plan_shared_across_landing_sampling_seeds": True,
        "formal_cross_city_evaluation_protocol": (
            "predicted_day_bootstrap_into_target_person_day_shell_v1"
        ),
        "target_city_information_used": "day_length_histogram_only",
        "target_grid_or_outcome_used": False,
        "Paris_grid_or_outcome_used": False,
        "exact_event_count_match": True,
        "Paris_test_person_day_count_preserved_exactly": (
            person_day_counts is not None
        ),
        "Paris_test_person_count": (
            None if person_day_counts is None else len(person_day_counts)
        ),
        "Paris_test_person_day_count_sha256": (
            None
            if person_day_counts is None
            else hashlib.sha256(
                json.dumps(
                    sorted(person_day_counts.items()),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ),
        "target_length_histogram": dict(
            sorted(Counter(map(int, target_lengths)).items())
        ),
        "used_source_day_count": int(np.count_nonzero(usage)),
        "maximum_source_day_reuse": int(usage.max()),
        "max_source_day_reuse_reported": True,
        "plan_sha256": hashlib.sha256(
            plans.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def _resampled_payload(source_payload, target_payload, plan_row):
    source_offsets = np.asarray(source_payload["view"]["offsets"], dtype=np.int64)
    pieces = [
        np.arange(source_offsets[index], source_offsets[index + 1], dtype=np.int64)
        for index in np.asarray(plan_row, dtype=np.int64)
    ]
    event_indices = np.concatenate(pieces)
    target_lengths = np.asarray(target_payload["lengths"], dtype=np.int64)
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(target_lengths, dtype=np.int64),
        )
    )
    if len(event_indices) != offsets[-1]:
        raise RuntimeError("Q4 selected-day event count changed")
    return event_indices, {
        "key": "Q4_X1_Paris_shell_reaggregation",
        "lengths": target_lengths,
        "view": {
            "offsets": offsets,
            "sequence_indices": np.arange(len(target_lengths)),
            "users": np.asarray(target_payload["view"]["users"], dtype=np.str_),
            "minute": np.asarray(source_payload["view"]["minute"], dtype=np.int64)[
                event_indices
            ],
            "day_count": len(target_lengths),
            "user_count": len(set(target_payload["view"]["users"].tolist())),
            "event_count": int(offsets[-1]),
        },
    }
