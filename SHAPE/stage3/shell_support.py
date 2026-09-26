"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from collections import Counter
import numpy as np


SHELL_LENGTH_POLICIES = ("strict", "common-support")


def _population_counts(lengths, users, mask):
    lengths = lengths[mask]
    return {
        "days": int(len(lengths)),
        "events": int(lengths.sum()),
        "within_day_transitions": int(np.maximum(lengths - 1, 0).sum()),
        "users": int(len(np.unique(users[mask]))),
    }


def audit_length_support(source_lengths, target_lengths, target_users, policy):
    """Return an auditable selection based solely on exact NYC test support."""
    if policy not in SHELL_LENGTH_POLICIES:
        raise ValueError("unknown shell length policy: " + str(policy))
    source_lengths = np.asarray(source_lengths, dtype=np.int64)
    target_lengths = np.asarray(target_lengths, dtype=np.int64)
    users = np.asarray(target_users, dtype=np.str_)
    if source_lengths.ndim != 1 or target_lengths.ndim != 1 \
            or not len(source_lengths) or not len(target_lengths) \
            or users.shape != target_lengths.shape \
            or np.any(source_lengths < 1) or np.any(target_lengths < 1):
        raise ValueError("shell lengths/users are invalid")
    source_counts = Counter(map(int, source_lengths))
    target_counts = Counter(map(int, target_lengths))
    supported = np.isin(target_lengths, source_lengths)
    missing = {length: count for length, count in sorted(target_counts.items())
               if not source_counts[length]}
    full = np.ones(len(target_lengths), dtype=bool)
    primary = target_lengths >= 2
    counts = {name: _population_counts(target_lengths, users, mask) for name, mask in (
        ("full", full), ("retained", supported), ("excluded", ~supported),
    )}
    primary_counts = {name: _population_counts(target_lengths, users, mask & primary)
                      for name, mask in (("full", full), ("retained", supported),
                                         ("excluded", ~supported))}
    fractions = {
        key: (counts["retained"][key] / counts["full"][key]
              if counts["full"][key] else None)
        for key in ("days", "events", "within_day_transitions", "users")
    }
    retained_users = set(users[supported].tolist())
    excluded_users = set(users[~supported].tolist())
    scope = "full_target_test_days" if not missing else "common_support_target_days"
    return np.flatnonzero(supported), {
        "schema": "civic_target_test_length_support_v1",
        "requested_policy": policy,
        "exact_length_support": not missing,
        "all_target_days_retained": not missing,
        "measurement_scope_if_run": scope,
        "selection_applied": policy == "common-support" and bool(missing),
        "selection_inputs": ["NYC_test_day_lengths", "CA_test_day_lengths"],
        "counts": {str(length): {"NYC_donor_days": source_counts[length],
                                 "CA_test_days": count}
                   for length, count in sorted(target_counts.items())},
        "missing_CA_lengths": missing,
        "population": counts,
        "primary_events_ge_2_population": primary_counts,
        "retained_fractions": fractions,
        "users_lost_entirely": len(excluded_users - retained_users),
        "users_with_both_retained_and_excluded_days": len(excluded_users & retained_users),
        "user_count_note": "excluded users means users with any excluded day; user counts can overlap",
        "no_replacement_Q4_feasible": all(source_counts[k] >= count
                                           for k, count in target_counts.items()),
        "target_train_calibration": "unchanged_full_eligible_cohort",
        "claim_limit": ("All target days have exact donors."
                        if not missing else
                        "Common-support scores exclude unsupported lengths; they do not estimate full CA performance."),
    }


def subset_target_test_days(payload, indices):
    """Select complete reference days after calibration, without changing events."""
    indices = np.asarray(indices, dtype=np.int64)
    lengths = np.asarray(payload["lengths"], dtype=np.int64)
    if not len(indices) or indices.ndim != 1 or np.any(np.diff(indices) <= 0) \
            or np.any((indices < 0) | (indices >= len(lengths))):
        raise ValueError("target day selection must be nonempty, ordered and unique")
    if np.array_equal(indices, np.arange(len(lengths))):
        return payload
    view = payload["view"]
    offsets = np.asarray(view["offsets"], dtype=np.int64)
    if not np.array_equal(np.diff(offsets), lengths):
        raise ValueError("target day offsets and lengths disagree")
    events = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices])
    selected = {
        "lengths": lengths[indices],
        "values": np.asarray(payload["values"])[events],
        "view": {"offsets": np.r_[0, np.cumsum(lengths[indices])],
                 "users": np.asarray(view["users"])[indices],
                 "minute": np.asarray(view["minute"])[events]},
    }
    if "dates" in view:
        selected["view"]["dates"] = np.asarray(view["dates"])[indices]
    return selected
