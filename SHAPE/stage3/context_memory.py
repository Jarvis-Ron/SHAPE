"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date as calendar_date
import hashlib
import json
from threading import Lock
import numpy as np
import torch
from SHAPE.stage3.sampler import _encode_landing_queries, attraction_distribution, group_autoregressive_landing
from SHAPE.stage3.resident_index import proportional_integer_capacities
from SHAPE.stage3.rank_semantics import DISTANCE_LOG_EPSILON_KM, _jsonable, _payload_mixture
from SHAPE.stage3.prototype_semantics import ATTRACTION_SMOOTHING


HOME_CONTEXT_NEIGHBORS = 25


HOME_CONTEXT_RESIDENTIAL_WEIGHT = 1.0


HOME_CONTEXT_POI_WEIGHT = 1.0


def _stable_sha(payload):
    return hashlib.sha256(json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _percentile_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("home-context residential values are invalid")
    ordered = np.sort(values, kind="stable")
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    if len(values) == 1:
        return np.asarray([0.5], dtype=np.float64)
    return (0.5 * (left + right - 1)) / float(len(values) - 1)


def _exact_nearest_indices(distance, count):
    """Return legacy ``(distance, grid_index)`` top-k without a full sort."""
    distance = np.asarray(distance, dtype=np.float64)
    count = int(count)
    if distance.ndim != 1 or not 1 <= count <= len(distance) \
            or not np.isfinite(distance).all():
        raise ValueError("nearest-grid inputs are invalid")
    grid_order = np.arange(len(distance), dtype=np.int64)
    if count == len(distance):
        return np.lexsort((grid_order, distance))
    candidate = np.argpartition(distance, count - 1)[:count]
    boundary = float(distance[candidate].max())
    closer = np.flatnonzero(distance < boundary)
    tied = np.flatnonzero(distance == boundary)
    needed = count - len(closer)
    chosen = np.concatenate((closer, tied[:needed]))
    return chosen[np.lexsort((chosen, distance[chosen]))]


def _context_cost(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    residential = np.abs(target[:, 0] - source[0])
    left = np.clip(source[1:], 1e-12, None)
    right = np.clip(target[:, 1:], 1e-12, None)
    middle = 0.5 * (right + left[None, :])
    poi_jsd = 0.5 * (
        np.sum(left[None, :] * np.log(left[None, :] / middle), axis=1)
        + np.sum(right * np.log(right / middle), axis=1)
    )
    return (
        HOME_CONTEXT_RESIDENTIAL_WEIGHT * residential
        + HOME_CONTEXT_POI_WEIGHT * poi_jsd
    )


def context_quota_home_assignment(
    source_homes, source_context, target_context, target_probability, seed,
    source_grid_ids, target_grid_ids,
):
    """Greedily minimize map-context cost under exact Hamilton capacities."""
    source_homes = np.asarray(source_homes, dtype=np.int64)
    source_ids = np.asarray(source_grid_ids, dtype=np.int64)
    target_ids = np.asarray(target_grid_ids, dtype=np.int64)
    source_context = np.asarray(source_context, dtype=np.float64)
    target_context = np.asarray(target_context, dtype=np.float64)
    positions = np.searchsorted(source_ids, source_homes)
    safe = np.clip(positions, 0, max(0, len(source_ids) - 1))
    if source_context.shape != (len(source_ids), 14) \
            or target_context.shape != (len(target_ids), 14) \
            or not np.all(source_ids[safe] == source_homes):
        raise RuntimeError("P3-E2 context-home inputs are not grid aligned")
    capacities = proportional_integer_capacities(
        target_probability, len(source_homes)
    )
    active = np.flatnonzero(capacities > 0)
    remaining = capacities[active].copy()
    selected = np.empty(len(source_homes), dtype=np.int64)
    costs = np.empty(len(source_homes), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(source_homes))
    for user_index in order:
        available = remaining > 0
        row = _context_cost(
            source_context[positions[user_index]], target_context[active]
        )
        row = np.where(available, row, np.inf)
        choice = int(np.argmin(row))
        if not np.isfinite(row[choice]):
            raise RuntimeError("context-home quota exhausted early")
        selected[user_index] = active[choice]
        costs[user_index] = row[choice]
        remaining[choice] -= 1
    realized = np.bincount(selected, minlength=len(target_ids))
    if not np.array_equal(realized, capacities):
        raise RuntimeError("context-home assignment violated exact quotas")

    # A preregistered marginal-only pairing diagnostic uses the same exact
    # capacities and seed, but shuffles slots without looking at context.
    slots = np.repeat(np.arange(len(target_ids), dtype=np.int64), capacities)
    random_slots = slots[rng.permutation(len(slots))]
    random_costs = np.asarray([
        _context_cost(source_context[position], target_context[[target]])[0]
        for position, target in zip(positions, random_slots)
    ])
    return target_ids[selected], {
        "contract": (
            "exact_Paris_train_home_Hamilton_quota_then_static_residential_"
            "percentile_and_local_POI_context_match_v1"
        ),
        "seed": int(seed),
        "capacity_exact": True,
        "assigned_user_count": int(len(source_homes)),
        "positive_capacity_grid_count": int(np.count_nonzero(capacities)),
        "maximum_grid_capacity": int(capacities.max()),
        "capacity_sha256": hashlib.sha256(capacities.tobytes()).hexdigest(),
        "assigned_context_cost_mean": float(costs.mean()),
        "assigned_context_cost_q95": float(np.quantile(costs, 0.95)),
        "marginal_only_pairing_context_cost_mean": float(random_costs.mean()),
        "context_cost_improvement": float(random_costs.mean() - costs.mean()),
        "uses_target_heldout": False,
        "uses_target_trajectory_for_matching": False,
    }


def map_marginal_rescaled_attraction(semantics, target_raw, target_group):
    """Keep code roles, but put every target-map category on unit mass."""
    raw = np.maximum(np.asarray(target_raw, dtype=np.float64), 0.0)
    weights = np.asarray(
        semantics.get("category_probabilities"), dtype=np.float64
    )
    roles = list(semantics.get("role", ()))
    dominant = np.asarray(
        semantics.get("dominant_poi_category"), dtype=np.int64
    )
    if raw.ndim != 2 or raw.shape[1] != 20 \
            or weights.shape != (len(roles), 13) \
            or dominant.shape != (len(roles),):
        raise RuntimeError("P3-E2 attraction semantics are incomplete")
    poi = raw[:, :13] + ATTRACTION_SMOOTHING
    poi /= poi.sum(axis=0, keepdims=True).clip(min=1e-12)
    _, residential = attraction_distribution(
        target_group, include_transit=False
    )
    residential = np.maximum(residential, 0.0) + ATTRACTION_SMOOTHING
    residential /= residential.sum()
    table = np.empty((len(roles), len(raw)), dtype=np.float64)
    for code, role in enumerate(roles):
        if role == "home_night_residential_proxy":
            score = residential
        elif role == "dominant_poi_column":
            # Keep the dominant role label, but retain the complete frozen
            # train-fit category distribution instead of collapsing it to a
            # one-hot column.  Unit-mass target columns make those source
            # probabilities comparable on the Paris map.
            score = poi @ weights[code]
        elif role == "ambiguous_general_poi_mixture":
            score = poi @ weights[code]
        else:
            raise RuntimeError("unknown attraction role: %s" % role)
        # Unit mean preserves a common numerical scale while attraction is
        # normalized within each sampled distance ring by the core decoder.
        score = np.maximum(score, 0.0) + ATTRACTION_SMOOTHING
        table[code] = score / score.mean()
    table = table.astype(np.float32)
    return table, {
        "contract": (
            "preserve_source_code_POI_role_rescale_each_target_static_map_"
            "category_to_unit_mass_then_unit_mean_v1"
        ),
        "source_semantics_sha256": semantics["table_sha256"],
        "target_map_only": True,
        "uses_target_flow": False,
        "uses_target_personal_features": False,
        "poi_column_mass_after_normalization": poi.sum(axis=0).tolist(),
        "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
    }


def derive_familiar_location_capacity(reference, projected_grid, users):
    """Read only each NYC user's number of distinct train locations."""
    event_users = np.asarray(reference["event_user_ids"], dtype=np.str_)
    event_splits = np.asarray(reference["event_splits"], dtype=np.str_)
    projected_grid = np.asarray(projected_grid, dtype=np.int64)
    users = np.asarray(users, dtype=np.str_)
    if len(event_users) != len(projected_grid):
        raise RuntimeError("familiar-capacity events are not aligned")
    locations = defaultdict(set)
    for user, split, grid in zip(event_users, event_splits, projected_grid):
        if str(split) == "train":
            locations[str(user)].add(int(grid))
    unique_users = sorted(set(users.tolist()))
    missing = [user for user in unique_users if not locations[user]]
    if missing:
        raise RuntimeError(
            "P3-E2 familiar capacity is missing NYC train users: %s"
            % missing[:3]
        )
    mapping = {user: len(locations[user]) for user in unique_users}
    values = np.asarray([mapping[str(user)] for user in users], dtype=np.int64)
    if np.any(values < 1):
        raise RuntimeError("P3-E2 familiar capacity must be positive")
    unique_values = np.asarray(list(mapping.values()), dtype=np.int64)
    return values, {
        "contract": (
            "source_person_distinct_NYC_train_location_count_as_target_"
            "familiar_set_capacity_only_v1"
        ),
        "fit_split": "NYC_train_only",
        "uses_source_location_identity": False,
        "uses_target_heldout": False,
        "user_count": int(len(mapping)),
        "minimum": int(unique_values.min()),
        "median": float(np.median(unique_values)),
        "mean": float(unique_values.mean()),
        "maximum": int(unique_values.max()),
        "user_capacity_sha256": _stable_sha(mapping),
    }


def _parse_iso_day(value):
    try:
        return calendar_date.fromisoformat(str(value))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "P3-E2 rolling memory requires ISO calendar dates: %s" % value
        ) from error


def validate_interleaved_split_chronology(cohort):
    """Validate a merged per-user timeline without assuming a time split.

    ``assign_day_splits`` deliberately samples train/test days within each
    user, so a valid heldout day may precede a train day.  The no-leakage
    invariant is therefore enforced by the rolling scheduler: it merges both
    splits and consumes them in strict date order instead of preloading every
    train day.
    """
    users = np.asarray(cohort["users"], dtype=np.str_)
    dates = np.asarray(cohort["dates"], dtype=np.str_)
    splits = np.asarray(cohort["splits"], dtype=np.str_)
    if users.shape != dates.shape or users.shape != splits.shape:
        raise RuntimeError("P3-E2 chronology keys are not day aligned")
    if not len(users) or np.any(~np.isin(splits, ("train", "test"))):
        raise RuntimeError(
            "P3-E2 rolling memory requires train/test days only"
        )
    per_user = defaultdict(list)
    seen = set()
    for user, date, split in zip(users, dates, splits):
        key = (str(user), str(date))
        if key in seen:
            raise RuntimeError(
                "P3-E2 rolling memory has a duplicate user/date: %s" % (key,)
            )
        seen.add(key)
        parsed = _parse_iso_day(date)
        per_user[str(user)].append((parsed, str(date), str(split)))
    invalid = []
    timelines = {}
    interleaved = []
    for user, values in per_user.items():
        observed = {item[2] for item in values}
        if observed != {"train", "test"}:
            invalid.append(user)
            continue
        ordered = sorted(values, key=lambda item: item[0])
        if any(left[0] >= right[0]
               for left, right in zip(ordered, ordered[1:])):
            invalid.append(user)
            continue
        train_days = [item[0] for item in ordered if item[2] == "train"]
        test_days = [item[0] for item in ordered if item[2] == "test"]
        if max(train_days) >= min(test_days):
            interleaved.append(user)
        timelines[user] = [(item[1], item[2]) for item in ordered]
    if invalid:
        raise RuntimeError(
            "P3-E2 rolling memory requires unique ISO-dated train and test "
            "days for every user: %s" % sorted(invalid)[:3]
        )
    return {
        "contract": (
            "per_user_train_test_days_merged_then_strict_ISO_date_replay_v2"
        ),
        "user_count": int(len(timelines)),
        "day_count": int(len(users)),
        "interleaved_user_count": int(len(interleaved)),
        "all_train_days_before_all_test_days": bool(not interleaved),
        "future_train_preload_forbidden": True,
        "future_heldout_used": False,
        "pass": True,
        "timeline_sha256": _stable_sha(timelines),
    }


def _memory_anchor_ids(item, capacity):
    if item is None:
        return []
    capacity = int(capacity)
    ranking = sorted(
        item["counts"],
        key=lambda grid: (
            -float(item["counts"][grid]),
            -int(item["last"].get(grid, -1)),
            int(grid),
        ),
    )
    return ranking[:capacity]


def chronological_interleaved_split_landing(
    train_payload, test_payload, train_homes, test_homes,
    train_capacities, test_capacities, landing_by_split, *, selection_seed,
):
    """Replay train and heldout days on one past-only per-user timeline.

    The upstream day split is an intentionally randomized within-user
    heldout split, not a chronological cutoff.  Preloading all train days
    would therefore leak locations landed on a future train date into an
    earlier heldout day.  This scheduler merges the two split payloads,
    orders every user by ISO date, and updates memory only after that day's
    prediction.  Split-specific callbacks retain the exact frozen inputs for
    real/generated train and heldout landing.
    """
    payload_by_split = OrderedDict((
        ("train", train_payload), ("test", test_payload),
    ))
    homes_by_split = {
        "train": np.asarray(train_homes, dtype=np.int64),
        "test": np.asarray(test_homes, dtype=np.int64),
    }
    capacities_by_split = {
        "train": np.asarray(train_capacities, dtype=np.int64),
        "test": np.asarray(test_capacities, dtype=np.int64),
    }
    if set(landing_by_split) != {"train", "test"} \
            or any(not callable(item) for item in landing_by_split.values()):
        raise RuntimeError(
            "interleaved rolling memory needs train and test landing callbacks"
        )

    records_by_user = defaultdict(list)
    capacity_by_user = {}
    seen = set()
    split_sequence_counts = {}
    for split, payload in payload_by_split.items():
        view = payload["view"]
        offsets = np.asarray(view["offsets"], dtype=np.int64)
        users = np.asarray(view["users"], dtype=np.str_)
        dates = np.asarray(view["dates"], dtype=np.str_)
        homes = homes_by_split[split]
        capacities = capacities_by_split[split]
        if len(offsets) != len(users) + 1 or dates.shape != users.shape \
                or homes.shape != users.shape or capacities.shape != users.shape \
                or offsets[0] != 0 or np.any(np.diff(offsets) <= 0) \
                or np.any(capacities < 1):
            raise RuntimeError(
                "interleaved rolling-memory %s arrays are not sequence aligned"
                % split
            )
        split_sequence_counts[split] = int(len(users))
        for sequence, (user_value, date_value) in enumerate(zip(users, dates)):
            user, date = str(user_value), str(date_value)
            key = (user, date)
            if key in seen:
                raise RuntimeError(
                    "interleaved rolling memory has a duplicate user/date: %s"
                    % (key,)
                )
            seen.add(key)
            parsed = _parse_iso_day(date)
            capacity = int(capacities[sequence])
            if user in capacity_by_user \
                    and capacity_by_user[user] != capacity:
                raise RuntimeError(
                    "interleaved rolling-memory capacity changed for user %s"
                    % user
                )
            capacity_by_user[user] = capacity
            records_by_user[user].append({
                "user": user,
                "date": date,
                "parsed_date": parsed,
                "split": split,
                "sequence": int(sequence),
            })

    invalid_users = []
    interleaved_users = []
    for user, records in records_by_user.items():
        records.sort(key=lambda item: item["parsed_date"])
        observed = {item["split"] for item in records}
        if observed != {"train", "test"} or any(
                left["parsed_date"] >= right["parsed_date"]
                for left, right in zip(records, records[1:])):
            invalid_users.append(user)
            continue
        train_days = [item["parsed_date"] for item in records
                      if item["split"] == "train"]
        test_days = [item["parsed_date"] for item in records
                     if item["split"] == "test"]
        if max(train_days) >= min(test_days):
            interleaved_users.append(user)
    if invalid_users:
        raise RuntimeError(
            "interleaved rolling memory requires unique train and test dates "
            "for every user: %s" % sorted(invalid_users)[:3]
        )

    ordered_users = sorted(records_by_user)
    width = max(len(records_by_user[user]) for user in ordered_users)
    waves = [
        [records_by_user[user][rank] for user in ordered_users
         if rank < len(records_by_user[user])]
        for rank in range(width)
    ]
    test_event_count = int(np.asarray(
        test_payload["view"]["offsets"], dtype=np.int64
    )[-1])
    test_prediction = np.empty(test_event_count, dtype=np.int64)
    test_filled = np.zeros(test_event_count, dtype=bool)
    state = {}
    last_date = {}
    anchor_counts = []
    chronology_rows = []
    wave_audits = []
    split_tokens = {"train": 0x54524149, "test": 0x54455354}

    for wave_index, wave in enumerate(waves):
        staged = []
        for split in ("train", "test"):
            members = [item for item in wave if item["split"] == split]
            if not members:
                continue
            payload = payload_by_split[split]
            offsets = np.asarray(payload["view"]["offsets"], dtype=np.int64)
            sequences = np.asarray(
                [item["sequence"] for item in members], dtype=np.int64
            )
            pieces = [
                np.arange(int(offsets[index]), int(offsets[index + 1]),
                          dtype=np.int64)
                for index in sequences
            ]
            event_indices = np.concatenate(pieces)
            lengths = np.asarray([len(item) for item in pieces], dtype=np.int64)
            local_offsets = np.concatenate((
                np.asarray([0], dtype=np.int64),
                np.cumsum(lengths, dtype=np.int64),
            ))
            anchors = [
                _memory_anchor_ids(
                    state.get(item["user"]), capacity_by_user[item["user"]]
                )
                for item in members
            ]
            history_width = max(1, max(map(len, anchors), default=0))
            history = np.full(
                (len(members), history_width), -1, dtype=np.int64
            )
            for row, values in enumerate(anchors):
                history[row, :len(values)] = values
                anchor_counts.append(len(values))
            derived_seed = int(np.random.SeedSequence((
                int(selection_seed), 0x50334532, split_tokens[split],
                int(wave_index),
            )).generate_state(1, dtype=np.uint32)[0])
            prediction, landing_audit = landing_by_split[split](
                event_indices, local_offsets,
                homes_by_split[split][sequences], history, derived_seed,
            )
            prediction = np.asarray(prediction, dtype=np.int64)
            if prediction.shape != event_indices.shape:
                raise RuntimeError(
                    "interleaved rolling-memory %s prediction is misaligned"
                    % split
                )
            if split == "test":
                if np.any(test_filled[event_indices]):
                    raise RuntimeError(
                        "interleaved rolling memory predicted a test event twice"
                    )
                test_prediction[event_indices] = prediction
                test_filled[event_indices] = True
            staged.append((
                split, members, local_offsets, prediction, anchors,
                landing_audit,
            ))

        # Each user appears at most once in a wave.  Updating only after all
        # callbacks makes the past-only boundary explicit even when a wave
        # contains both train and test records for different users.
        for split, members, local_offsets, prediction, anchors, audit in staged:
            wave_audits.append({
                "wave": int(wave_index), "split": split,
                "sequence_count": int(len(members)), "landing": audit,
            })
            for row, record in enumerate(members):
                user = record["user"]
                if user in last_date \
                        and record["parsed_date"] <= last_date[user]:
                    raise RuntimeError(
                        "interleaved rolling memory replayed a non-future date"
                    )
                start = int(local_offsets[row])
                end = int(local_offsets[row + 1])
                item = state.setdefault(user, {
                    "counts": Counter(), "last": {}, "clock": 0,
                })
                for grid in prediction[start:end]:
                    grid = int(grid)
                    item["clock"] += 1
                    item["counts"][grid] += 1.0
                    item["last"][grid] = item["clock"]
                retained = _memory_anchor_ids(item, capacity_by_user[user])
                item["counts"] = Counter({
                    grid: item["counts"][grid] for grid in retained
                })
                item["last"] = {
                    grid: item["last"][grid] for grid in retained
                }
                last_date[user] = record["parsed_date"]
                chronology_rows.append((
                    user, record["date"], split, int(record["sequence"]),
                    tuple(map(int, anchors[row])),
                    tuple(map(int, prediction[start:end])),
                ))

    if not np.all(test_filled):
        raise RuntimeError(
            "interleaved rolling memory did not predict every heldout event"
        )
    retained_counts = [len(item["counts"]) for item in state.values()]
    same_code_audits = [
        item["landing"]["same_code_revisit"] for item in wave_audits
        if "same_code_revisit" in item["landing"]
    ]
    same_code_totals = {
        key: int(sum(int(item.get(key, 0)) for item in same_code_audits))
        for key in (
            "trigger_count", "choice_count", "ring_prebias_veto_count",
            "ring_postbias_veto_count", "ring_target_changed_count",
        )
    }
    audit = {
        "contract": (
            "merged_random_train_test_split_strict_per_user_date_replay_"
            "past_landed_Paris_memory_v2"
        ),
        "selection_seed": int(selection_seed),
        "user_count": int(len(records_by_user)),
        "wave_count": int(len(waves)),
        "sequence_count": int(sum(split_sequence_counts.values())),
        "split_sequence_counts": split_sequence_counts,
        "interleaved_user_count": int(len(interleaved_users)),
        "all_train_days_before_all_test_days": bool(not interleaved_users),
        "past_anchor_count_mean": float(np.mean(anchor_counts)),
        "past_anchor_count_max": int(max(anchor_counts, default=0)),
        "retained_location_count_mean": float(np.mean(retained_counts)),
        "retained_location_count_max": int(max(retained_counts, default=0)),
        "strict_user_date_chronology": True,
        "future_train_preloaded": False,
        "future_train_used": False,
        "future_heldout_used": False,
        "test_prediction_complete": True,
        "derived_seed_contract": (
            "numpy_SeedSequence_sampling_seed_P3E2_split_wave_v2"
        ),
        "chronology_sha256": _stable_sha(chronology_rows),
        "wave_audit_sha256": _stable_sha(wave_audits),
        "all_hard_ring_pass": bool(all(
            item["landing"].get("hard_distance_ring_pass", True)
            for item in wave_audits
        )),
    }
    if same_code_audits:
        audit["same_code_revisit"] = {
            "enabled": bool(any(
                item.get("enabled", False) for item in same_code_audits
            )),
            **same_code_totals,
            "choice_fraction_given_trigger": float(
                same_code_totals["choice_count"]
                / max(1, same_code_totals["trigger_count"])
            ),
            "wave_count": int(len(same_code_audits)),
        }
    return test_prediction, state, audit


def _task_configuration_id(task):
    values = [
        str(task["arm"]), str(task["scheme"]), str(task["split"]),
        str(task["scale_mode"]), str(task["home_strategy"]),
        str(task["memory_mode"]), str(task["attraction_mode"]),
    ]
    if task.get("generator_seed") is not None:
        values.append("g%d" % int(task["generator_seed"]))
    values.append("s%d" % int(task["sampling_seed"]))
    return "_".join(values)


def _task_homes(task, payload, runtime):
    sequences = payload["view"]["sequence_indices"]
    key = task["home_strategy"]
    if key == "Paris_true_train_modal":
        return runtime["target_true_homes"][sequences]
    if key == "Paris_T2_context_quota":
        return runtime["target_context_homes"][task["sampling_seed"]][sequences]
    if key == "E1_T1_population_marginal":
        return runtime["home_by_strategy"]["T1"][task["sampling_seed"]][sequences]
    if key == "T2_context_quota":
        return runtime["source_context_homes"][task["sampling_seed"]][sequences]
    raise RuntimeError("unknown P3-E2 home strategy: %s" % key)


def _task_attraction(task, runtime):
    key = task["attraction_mode"]
    tables = {
        "target_native_legacy": "target_native_attraction",
        "source_role_target_raw_legacy": "transferred_attraction",
        "target_native_map_rescaled": "target_native_attraction_rescaled",
        "source_role_target_map_rescaled": "transferred_attraction_rescaled",
    }
    try:
        return runtime[tables[key]]
    except KeyError as error:
        raise RuntimeError("unknown P3-E2 attraction mode: %s" % key) from error


def _core_landing(
    lane, runtime, payload, homes, history, attraction, scale_mode,
    selection_seed, encoded_queries, *, landing_temperature,
    candidate_top_k, candidate_omitted_mass_tolerance,
    candidate_audit_steps, sequence_batch_size, landing_event_batch_size,
    event_indices=None, offsets=None, event_code_keys=None,
    same_code_revisit_beta=0.0, same_code_key_layers="L0",
    same_code_ring_bias=0.0,
):
    mixture = _payload_mixture(payload, scale_mode, runtime)
    spec = mixture["spec"]
    if event_indices is None:
        event_indices = np.arange(len(payload["features"]), dtype=np.int64)
        local_offsets = payload["view"]["offsets"]
    else:
        event_indices = np.asarray(event_indices, dtype=np.int64)
        local_offsets = np.asarray(offsets, dtype=np.int64)
    features = payload["features"][event_indices]
    codes = payload["codes"][event_indices]
    if torch.is_tensor(encoded_queries):
        query_indices = torch.as_tensor(
            event_indices, dtype=torch.long, device=encoded_queries.device
        )
        queries = encoded_queries[query_indices]
    else:
        queries = np.asarray(encoded_queries)[event_indices]
    local_code_keys = None
    if event_code_keys is not None:
        event_code_keys = np.asarray(event_code_keys, dtype=np.int64)
        if event_code_keys.shape[0] != len(payload["features"]):
            raise ValueError("P3-E2 same-code keys are not payload-aligned")
        local_code_keys = event_code_keys[event_indices]
    personal_options = {}
    if runtime.get("personal_scale") is not None:
        from SHAPE.stage3.opportunity_rank import profile_event_parameters
        config = runtime["personal_scale"]
        parent_sequence = np.searchsorted(
            payload["view"]["offsets"][1:], event_indices[local_offsets[:-1]], side="right")
        local_users = payload["view"]["users"][parent_sequence]
        multiplier, betas = profile_event_parameters(
            runtime["personal_profiles"], local_users, local_offsets,
            config["lambda"], config["kappa"], runtime["selected_beta"])
        if config["lambda"] != 0:
            personal_options["personal_ring_multiplier"] = multiplier
        if config["kappa"] != 0:
            personal_options["sequence_revisit_beta"] = betas
        if config.get("cost", False):
            gamma = runtime["personal_profiles"]["gamma"]
            if gamma is None:
                raise ValueError("source distances do not admit the declared Pareto cost fit")
            personal_options.update(distance_cost_gamma=gamma,
                distance_cost_xmin_km=runtime["personal_profiles"]["cost_xmin_km"])
    return group_autoregressive_landing(
        lane["model"], runtime["target_l2"], runtime["target_raw"],
        runtime["normalization"], features, local_offsets, homes,
        lane["device"], landing_temperature, 1.0, runtime["selected_beta"],
        target_grid_ids=runtime["target_grid_ids"],
        centers=runtime["target_centers"], candidate_top_k=candidate_top_k,
        omitted_mass_tolerance=candidate_omitted_mass_tolerance,
        omitted_mass_audit_steps=candidate_audit_steps,
        sequence_batch_size=sequence_batch_size, return_audit=True,
        encoded_queries=queries, normalized_target_tensor=lane["target"],
        prepared_center_tensor=lane["centers"], neighborhood_cache={},
        event_batch_size=landing_event_batch_size,
        code_distance_bin_probabilities=mixture["physical"][event_indices],
        physical_distance_bin_edges_km=spec["coarse_edges"],
        code_distance_piecewise_probabilities=(
            mixture["piecewise"][event_indices]
        ),
        physical_distance_piecewise_edges_km=spec["piecewise_edges"],
        distance_kernel_tau=float(runtime["base_parameters"]["distance_tau"]),
        history_seed_grid_ids=history,
        distance_log_epsilon_km=DISTANCE_LOG_EPSILON_KM,
        selection_mode="sample", selection_seed=int(selection_seed),
        selection_temperature=float(
            runtime["base_parameters"]["selection_temperature"]
        ),
        code_attraction_table=attraction,
        event_attraction_code=codes[:, 0], attraction_mode="raw",
        attraction_strength=float(
            runtime["base_parameters"]["attraction_strength"]
        ),
        attraction_epsilon=ATTRACTION_SMOOTHING,
        hard_distance_ring_constraint=True,
        hard_distance_ring_seed=int(selection_seed),
        hard_distance_ring_empty_policy="expand_adjacent",
        distance_coordinate=spec["coordinate"],
        prepared_spatial_rank_tensor=(
            lane["rank"] if spec["coordinate"] == "spatial_rank" else None
        ),
        event_code_keys=local_code_keys,
        same_code_revisit_beta=float(same_code_revisit_beta),
        same_code_key_layers=same_code_key_layers,
        same_code_ring_bias=float(same_code_ring_bias),
        **personal_options,
    )


def _landing_task_cost(task, payloads):
    payload = payloads[task["payload_key"]]
    cost = int(len(payload["features"]))
    warmup_key = task.get("warmup_payload_key")
    if task.get("memory_mode") == "rolling_Paris_past" and warmup_key:
        cost += int(len(payloads[warmup_key]["features"]))
    return max(1, cost)


def _balanced_lane_assignments(tasks, payloads, lane_count):
    """Greedily balance event work while keeping every task deterministic."""
    lane_count = int(lane_count)
    if lane_count < 1:
        raise ValueError("landing needs at least one device lane")
    assignments = [[] for _ in range(lane_count)]
    loads = [0 for _ in range(lane_count)]
    ranked = sorted(
        enumerate(tasks),
        key=lambda item: (-_landing_task_cost(item[1], payloads), item[0]),
    )
    for _, task in ranked:
        lane_index = min(range(lane_count), key=lambda item: (loads[item], item))
        cost = _landing_task_cost(task, payloads)
        assignments[lane_index].append(task)
        loads[lane_index] += cost
    return assignments, {
        "contract": "largest_event_work_first_greedy_device_balance_v1",
        "task_count": int(len(tasks)),
        "estimated_event_work_by_lane": loads,
        "estimated_event_work_imbalance": int(max(loads) - min(loads)),
    }


def _cached_landing_queries(lane, payload):
    cache = lane.setdefault("encoded_query_cache", {})
    stats = lane.setdefault(
        "encoded_query_cache_stats", {"hits": 0, "misses": 0}
    )
    cache_key = str(payload.get("query_cache_key", payload["key"]))
    if cache_key not in cache:
        cache[cache_key] = _encode_landing_queries(
            lane["model"], payload["features"], lane["device"]
        )
        stats["misses"] += 1
    else:
        stats["hits"] += 1
    return cache[cache_key]


def _prepare_task_mixtures(tasks, payloads, runtime):
    seen = set()
    for task in tasks:
        for key in (task["payload_key"], task.get("warmup_payload_key")):
            if key is None or (key, task["scale_mode"]) in seen:
                continue
            _payload_mixture(
                payloads[key], task["scale_mode"], runtime
            )
            seen.add((key, task["scale_mode"]))


def _run_e2_tasks(tasks, payloads, runtime, *, progress=None,
                  stage_label="E2", **landing_options):
    lanes = runtime["lanes"]
    _prepare_task_mixtures(tasks, payloads, runtime)
    assignments, schedule = _balanced_lane_assignments(
        tasks, payloads, len(lanes)
    )
    schedule["stage"] = str(stage_label)
    runtime.setdefault("runtime_optimization", {}).setdefault(
        "landing_schedules", {}
    )[str(stage_label)] = schedule
    completed = 0
    completion_lock = Lock()

    def worker(lane, lane_tasks):
        def queries(payload):
            return _cached_landing_queries(lane, payload)

        nonlocal completed
        output = []
        for task in lane_tasks:
            payload = payloads[task["payload_key"]]
            homes = _task_homes(task, payload, runtime)
            attraction = _task_attraction(task, runtime)
            rolling = task["memory_mode"] == "rolling_Paris_past"
            if rolling:
                train_payload = payloads[task["warmup_payload_key"]]
                train_homes = _task_homes(task, train_payload, runtime)
                train_capacities = runtime["source_familiar_capacity"][
                    train_payload["view"]["sequence_indices"]
                ]

                def train_wave(
                        indices, offsets, local_homes, history, seed,
                        _payload=train_payload, _attraction=attraction,
                        _scale_mode=task["scale_mode"]):
                    return _core_landing(
                        lane, runtime, _payload, local_homes, history,
                        _attraction, _scale_mode, seed, queries(_payload),
                        event_indices=indices, offsets=offsets,
                        **landing_options,
                    )

                test_capacities = runtime["source_familiar_capacity"][
                    payload["view"]["sequence_indices"]
                ]

                def test_wave(
                        indices, offsets, local_homes, history, seed,
                        _payload=payload, _attraction=attraction,
                        _scale_mode=task["scale_mode"]):
                    return _core_landing(
                        lane, runtime, _payload, local_homes, history,
                        _attraction, _scale_mode, seed, queries(_payload),
                        event_indices=indices, offsets=offsets,
                        **landing_options,
                    )

                prediction, _, rolling_audit = \
                    chronological_interleaved_split_landing(
                        train_payload, payload, train_homes, homes,
                        train_capacities, test_capacities,
                        {"train": train_wave, "test": test_wave},
                        selection_seed=task["sampling_seed"],
                    )
                landing_audit = {
                    "contract": (
                        "P3_E2_interleaved_train_test_strict_date_memory_v2"
                    ),
                    "train_warmup": {
                        "mode": "merged_with_test_by_per_user_date",
                        "future_train_preloaded": False,
                        "sequence_count": rolling_audit[
                            "split_sequence_counts"
                        ]["train"],
                    },
                    "test": rolling_audit,
                }
            else:
                history = payload["history"]
                prediction, landing_audit = _core_landing(
                    lane, runtime, payload, homes, history, attraction,
                    task["scale_mode"], task["sampling_seed"], queries(payload),
                    **landing_options,
                )
            output.append({
                **task,
                "configuration_id": _task_configuration_id(task),
                "prediction": prediction,
                "landing_audit": landing_audit,
                "distance_application": _payload_mixture(
                    payload, task["scale_mode"], runtime
                )["audit"],
                "device": str(lane["device"]),
            })
            if progress is not None:
                with completion_lock:
                    completed += 1
                    progress(
                        completed, len(tasks), task, str(lane["device"])
                    )
        return output

    if len(lanes) == 1:
        results = worker(lanes[0], assignments[0])
    else:
        with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
            futures = [
                executor.submit(worker, lane, items)
                for lane, items in zip(lanes, assignments, strict=True)
            ]
            results = [item for future in futures for item in future.result()]
    return sorted(results, key=lambda item: item["configuration_id"])
