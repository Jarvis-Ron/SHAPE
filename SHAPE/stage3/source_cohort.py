"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np
from SHAPE.stage3.contracts import CODES_SCHEMA, load_npz, require_scalar, sha256_file
from SHAPE.stage3.prepare_generated_landing import GENERATED_LANDING_SCHEMA


MAIN_GRANULARITY = "od_100x100"


REPORT_METRICS = (
    "M1_visit_jsd",
    "M2_od_cpc",
    "M2_od_pearson",
    "M2_od_nrmse",
    "M3_period_space_jsd_mean",
    "M3_morning_06_11_jsd",
    "M3_afternoon_11_17_jsd",
    "M3_evening_17_22_jsd",
    "M3_night_22_06_jsd",
    "M4_rhythm_24h_jsd_descriptive",
    "A_visit_entropy_predicted",
    "A_visit_entropy_reference",
    "A_grid_coverage_predicted",
    "A_grid_coverage_reference",
    "A_radius_of_gyration_distribution_jsd",
    "A_jump_distance_spectrum_jsd",
    "A_jump_bin_distribution_jsd",
    "A_visited_poi_category_share_jsd",
)


def _record_keys(users, dates):
    keys = tuple(zip(
        np.asarray(users, dtype=np.str_).tolist(),
        np.asarray(dates, dtype=np.str_).tolist(),
    ))
    if len(keys) != len(set(keys)):
        raise RuntimeError("daily archive contains duplicate user/date keys")
    return keys


def _slice_by_key(artifact, key_to_index, key, event_name):
    index = key_to_index[key]
    offsets = np.asarray(artifact["day_offsets"], dtype=np.int64)
    start, end = int(offsets[index]), int(offsets[index + 1])
    return np.asarray(artifact[event_name])[start:end]


def _load_generated_records(paths):
    records = {}
    tokenizer_sha, city = None, None
    source_paths = []
    for path in paths:
        artifact = load_npz(path)
        require_scalar(artifact, "schema", GENERATED_LANDING_SCHEMA)
        archive_sha = str(require_scalar(artifact, "tokenizer_sha256"))
        archive_city = str(require_scalar(artifact, "source_city")).lower()
        if tokenizer_sha is None:
            tokenizer_sha, city = archive_sha, archive_city
        if archive_sha != tokenizer_sha or archive_city != city:
            raise RuntimeError(
                "generated train/heldout landing archives changed tokenizer or city"
            )
        offsets = np.asarray(artifact["sequence_offsets"], dtype=np.int64)
        users = np.asarray(artifact["sequence_user_ids"], dtype=np.str_)
        dates = np.asarray(artifact["sequence_dates"], dtype=np.str_)
        keys = _record_keys(users, dates)
        features = np.asarray(artifact["reconstructed_h"], dtype=np.float32)
        codes = (
            None if "codes" not in artifact else
            np.asarray(artifact["codes"], dtype=np.int64)
        )
        minutes = np.asarray(artifact["minute"], dtype=np.int64)
        real_grid = np.asarray(artifact["reference_grid_id"], dtype=np.int64)
        homes = np.asarray(artifact["home_grid_id"], dtype=np.int64)
        source_generation_binding = {}
        for binding_key in (
            "codes_sha256", "profiles_sha256", "generator_checkpoint_sha256",
            "generator_seed", "generation_contract", "word_dropout",
            "user_conditioning", "user_context_pooling",
            "user_context_max_days", "user_shrinkage",
            "user_conditional_prior", "user_discriminative_loss",
            "variation_matching_loss",
        ):
            if binding_key not in artifact:
                continue
            binding_value = np.asarray(artifact[binding_key])
            if binding_value.shape != ():
                raise RuntimeError(
                    "generated landing binding field must be scalar: %s"
                    % binding_key
                )
            source_generation_binding[binding_key] = binding_value.item()
        if len(offsets) != len(keys) + 1 or len(homes) != len(keys) \
                or offsets[0] != 0 or offsets[-1] != len(features) \
                or len(minutes) != len(features) or len(real_grid) != len(features) \
                or (codes is not None and (
                    codes.shape != (len(features), 3)
                    or np.any(codes < 0) or np.any(codes >= 64)
                )):
            raise RuntimeError("generated landing archive arrays are not aligned")
        for index, key in enumerate(keys):
            if key in records:
                raise RuntimeError(
                    "generated train/heldout archives overlap at %s/%s" % key
                )
            start, end = int(offsets[index]), int(offsets[index + 1])
            records[key] = {
                "features": features[start:end],
                "codes": None if codes is None else codes[start:end],
                "minute": minutes[start:end],
                "reference_grid": real_grid[start:end],
                "home": int(homes[index]),
            }
        source_paths.append({
            "path": str(path), "sha256": sha256_file(path),
            "source_generation_binding": source_generation_binding,
        })
    if not records:
        raise RuntimeError("generated landing inputs contain no days")
    return records, {
        "tokenizer_sha256": tokenizer_sha,
        "source_city": city,
        "archives": source_paths,
    }


def _reference_train_test_eligibility(day_users, day_splits):
    """Select users by split availability only and return an auditable record."""
    day_users = np.asarray(day_users, dtype=np.str_)
    day_splits = np.asarray(day_splits, dtype=np.str_)
    if day_users.shape != day_splits.shape:
        raise RuntimeError("reference day users and splits are not aligned")
    splits_by_user = defaultdict(set)
    for user, split in zip(day_users, day_splits):
        split = str(split)
        if split in ("train", "test"):
            splits_by_user[str(user)].add(split)
    required = {"train", "test"}
    eligible = sorted(
        user for user, splits in splits_by_user.items() if splits == required
    )
    excluded = sorted(set(splits_by_user) - set(eligible))
    exclusions = [
        {
            "user_id": user,
            "observed_splits": sorted(splits_by_user[user]),
            "missing_splits": sorted(required - splits_by_user[user]),
        }
        for user in excluded
    ]
    payload = json.dumps(
        exclusions, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return set(eligible), {
        "contract": "reference_split_availability_train_and_test_only_v1",
        "selection_uses_outcome_values": False,
        "required_splits": ["train", "test"],
        "candidate_user_count": int(len(splits_by_user)),
        "eligible_user_count": int(len(eligible)),
        "excluded_incomplete_user_count": int(len(excluded)),
        "excluded_incomplete_users": exclusions,
        "excluded_incomplete_users_sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_g0_cohort(
    reference_codes, generated_by_seed, max_users=None,
    require_complete_user_splits=False, *, reference_only=False,
):
    """Load a real-code cohort, optionally intersected across generator seeds.

    Historical G0/P3 experiments require generated archives and retain only
    user-days shared by every generator seed.  The frozen X1 real-source model
    has no generated-code lane at all; ``reference_only=True`` selects the
    leakage-safe train/test cohort directly from the observed code archive.
    Keeping that mode explicit prevents an accidentally empty generator map
    from silently changing an experiment cohort.
    """
    reference = load_npz(reference_codes)
    require_scalar(reference, "schema", CODES_SCHEMA)
    city = str(require_scalar(reference, "source_city")).lower()
    tokenizer_sha = str(require_scalar(reference, "tokenizer_sha256"))
    day_users = np.asarray(reference["day_user_ids"], dtype=np.str_)
    day_dates = np.asarray(reference["day_dates"], dtype=np.str_)
    day_splits = np.asarray(reference["day_splits"], dtype=np.str_)
    keys = _record_keys(day_users, day_dates)
    key_to_index = {key: index for index, key in enumerate(keys)}
    all_reference_keys = {
        key for key, split in zip(keys, day_splits)
        if str(split) in ("train", "test")
    }
    eligible_users, eligibility_audit = _reference_train_test_eligibility(
        day_users, day_splits
    )
    selected_reference_keys = (
        {
            key for key, split in zip(keys, day_splits)
            if str(split) in ("train", "test") and key[0] in eligible_users
        }
        if require_complete_user_splits else all_reference_keys
    )

    generated_by_seed = dict(generated_by_seed or {})
    if reference_only and generated_by_seed:
        raise ValueError(
            "reference-only G0 cohort forbids generated archives"
        )
    if not reference_only and not generated_by_seed:
        raise RuntimeError(
            "G0 requires generated archives unless reference_only=True"
        )

    generated_records, generated_binding = {}, {}
    key_set = set(selected_reference_keys) if reference_only else None
    for seed in sorted(generated_by_seed):
        records, binding = _load_generated_records(generated_by_seed[seed])
        if binding["source_city"] != city \
                or binding["tokenizer_sha256"] != tokenizer_sha:
            raise RuntimeError(
                "G0 generated archives disagree with reference city/tokenizer"
            )
        current = set(records)
        if not current <= all_reference_keys:
            unexpected = sorted(current - all_reference_keys)[:3]
            raise RuntimeError(
                "G0 generated keys are absent from real train+test days: %s"
                % unexpected
            )
        if require_complete_user_splits:
            current = {
                key for key in current if key[0] in eligible_users
            }
        if key_set is None:
            key_set = current
        elif current != key_set:
            raise RuntimeError("G0 day cohort changed across generator seeds")
        generated_records[int(seed)] = records
        generated_binding[int(seed)] = binding

    ordered_keys = sorted(key_set or ())
    selected_users = sorted({key[0] for key in ordered_keys})
    if max_users is not None:
        keep_users = set(selected_users[:int(max_users)])
        ordered_keys = [key for key in ordered_keys if key[0] in keep_users]
        selected_users = sorted(keep_users)
    if len(selected_users) < 4 or not ordered_keys:
        raise RuntimeError("G0 requires at least four users and one shared day")

    train_home_counts = defaultdict(Counter)
    event_users = np.asarray(reference["event_user_ids"], dtype=np.str_)
    event_splits = np.asarray(reference["event_splits"], dtype=np.str_)
    event_grid = np.asarray(reference["grid_id"], dtype=np.int64)
    for user, split, grid in zip(event_users, event_splits, event_grid):
        if str(user) in selected_users and str(split) == "train" and int(grid) >= 0:
            train_home_counts[str(user)][int(grid)] += 1
    home_by_user = {}
    for user in selected_users:
        if not train_home_counts[user]:
            raise RuntimeError("G0 user has no train-only real home: %s" % user)
        home_by_user[user] = int(train_home_counts[user].most_common(1)[0][0])

    real_features, real_codes, real_grids, minutes = [], [], [], []
    users, dates, splits, true_homes, lengths = [], [], [], [], []
    generated_features = {seed: [] for seed in generated_records}
    generated_codes = {seed: [] for seed in generated_records}
    for key in ordered_keys:
        index = key_to_index[key]
        split = str(day_splits[index])
        h = _slice_by_key(reference, key_to_index, key, "reconstructed_h").astype(
            np.float32, copy=False
        )
        grid = _slice_by_key(reference, key_to_index, key, "grid_id").astype(
            np.int64, copy=False
        )
        codes = _slice_by_key(reference, key_to_index, key, "codes").astype(
            np.int64, copy=False
        )
        minute = _slice_by_key(reference, key_to_index, key, "minute").astype(
            np.int64, copy=False
        )
        if not len(h) or len(grid) != len(h) or len(minute) != len(h) \
                or codes.shape != (len(h), 3) \
                or np.any(codes < 0) or np.any(codes >= 64):
            raise RuntimeError("real G0 day arrays are empty or misaligned")
        for seed, records in generated_records.items():
            generated = records[key]
            if len(generated["features"]) != len(h) \
                    or not np.array_equal(generated["minute"], minute) \
                    or not np.array_equal(generated["reference_grid"], grid) \
                    or int(generated["home"]) != home_by_user[key[0]]:
                raise RuntimeError(
                    "G0 real/generated schedule, grid, or home mismatch at seed "
                    "%d for %s/%s" % (seed, key[0], key[1])
                )
            generated_features[seed].append(generated["features"])
            if generated_codes[seed] is not None:
                if generated["codes"] is None:
                    generated_codes[seed] = None
                else:
                    generated_codes[seed].append(generated["codes"])
        real_features.append(h)
        real_codes.append(codes)
        real_grids.append(grid)
        minutes.append(minute)
        users.append(key[0])
        dates.append(key[1])
        splits.append(split)
        true_homes.append(home_by_user[key[0]])
        lengths.append(len(h))

    offsets = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(lengths, dtype=np.int64),
    ))
    split_counts = Counter(splits)
    if not split_counts.get("train") or not split_counts.get("test"):
        raise RuntimeError("G0 full cohort must include both train and heldout days")
    return {
        "city": city,
        "tokenizer_sha256": tokenizer_sha,
        "reference_codes": str(reference_codes),
        "reference_codes_sha256": sha256_file(reference_codes),
        "offsets": offsets,
        "users": np.asarray(users, dtype=np.str_),
        "dates": np.asarray(dates, dtype=np.str_),
        "splits": np.asarray(splits, dtype=np.str_),
        "true_homes": np.asarray(true_homes, dtype=np.int64),
        "real_features": np.concatenate(real_features),
        "real_codes": np.concatenate(real_codes),
        "real_grid": np.concatenate(real_grids),
        "minute": np.concatenate(minutes),
        "generated_features": {
            seed: np.concatenate(rows) for seed, rows in generated_features.items()
        },
        "generated_codes": {
            seed: (None if rows is None else np.concatenate(rows))
            for seed, rows in generated_codes.items()
        },
        "generated_binding": generated_binding,
        "cohort": {
            "contract": (
                "reference_train_test_eligible_users_real_codes_only_v1"
                if reference_only else
                "reference_train_test_eligible_users_then_exact_cross_seed_"
                "user_date_alignment_no_val_v1"
                if require_complete_user_splits else
                "exact_cross_seed_intersection_required_then_real_train_plus_"
                "heldout_user_date_alignment_no_val"
            ),
            "input_mode": (
                "observed_reference_codes_only"
                if reference_only else "real_generated_shared_schedule"
            ),
            "user_count": len(selected_users),
            "day_count": len(ordered_keys),
            "event_count": int(offsets[-1]),
            "split_day_counts": dict(sorted(split_counts.items())),
            "key_sha256": hashlib.sha256(
                "\n".join("%s\t%s" % key for key in ordered_keys).encode("utf-8")
            ).hexdigest(),
            "reference_train_test_days_available": len(selected_reference_keys),
            "generated_day_coverage_fraction": float(
                len(ordered_keys) / max(1, len(selected_reference_keys))
            ),
            **(
                {"reference_user_eligibility": eligibility_audit}
                if require_complete_user_splits else {}
            ),
        },
    }
