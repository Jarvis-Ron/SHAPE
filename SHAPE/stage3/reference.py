"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from SHAPE.gridspec import StaticGrid
from SHAPE.stage2.prepare_city_event_data import load_ca_checkins
from SHAPE.stage2.sparse_event_data import map_records_to_grid
from SHAPE.stage3.contracts import sha256_file
from SHAPE.stage3.prepare_codes import assign_day_splits


REFERENCE_SCHEMA = "civic_transfer_reference_days_v1"


def prepare_ca_reference(raw_files, grid_csv, output, *, seed=42,
                         verify_release=True):
    """Match Module-2 filtering and prepare_codes calendar/day splitting.

    Keep complete users with >=3 in-bbox events, including missing-static
    events.  Neither active-grid membership nor latent availability filters
    an event.  Sparse geometry projection happens later, as in P3-E1.
    """
    raw_files = [Path(path).resolve() for path in raw_files]
    if not raw_files or len(set(raw_files)) != len(raw_files):
        raise ValueError("provide distinct CA raw checkin files")
    missing = [str(path) for path in raw_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("CA raw checkins are missing: " + ", ".join(missing))
    static = StaticGrid.load(grid_csv, city="ca", verify_release=verify_release)
    records, mapping = map_records_to_grid(load_ca_checkins(raw_files), static)
    users = defaultdict(list)
    for record in records:
        users[str(record["user_id"])].append(record)
    days = []
    excluded_users = 0
    for user in sorted(users):
        rows = sorted(users[user], key=lambda row: row["local_dt"])
        if len(rows) < 3:
            excluded_users += 1
            continue
        dates = [row["date"] for row in rows]
        splits = assign_day_splits(user, dates, seed=seed)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["date"]].append(row)
        for date in sorted(grouped):
            days.append((user, date, splits[date], grouped[date]))
    if not days:
        raise RuntimeError("CA preprocessing retained no complete user days")
    lengths = np.asarray([len(day[3]) for day in days], dtype=np.int64)
    day_users = np.asarray([day[0] for day in days], dtype=np.str_)
    day_splits = np.asarray([day[2] for day in days], dtype=np.str_)
    receipt = {
        "schema": REFERENCE_SCHEMA,
        "city": "ca", "seed": int(seed),
        "daily_split_contract": "per_user_random_80_10_10_v1",
        "local_time_contract": "iana_timezone_v1",
        "timezone": "America/Los_Angeles",
        "raw_files": [{"path": str(path), "sha256": sha256_file(path)}
                      for path in raw_files],
        "grid_csv": str(Path(grid_csv).resolve()),
        "grid_csv_sha256": sha256_file(grid_csv),
        "mapping": mapping,
        "users_below_three_events_excluded": excluded_users,
        "prepared_user_count": len(set(day_users.tolist())),
        "prepared_day_count": len(days),
        "prepared_event_count": int(lengths.sum()),
        "active_grid_filter_used": False,
        "target_tokenizer_or_latent_required_for_X1": False,
        "reference_true_codes_created": False,
        "duplicates_policy": "retain_like_original_preprocessing",
    }
    arrays = {
        "schema": np.asarray(REFERENCE_SCHEMA),
        "source_city": np.asarray("ca"),
        "metadata_json": np.asarray(json.dumps(receipt, ensure_ascii=False)),
        "day_offsets": np.r_[0, np.cumsum(lengths)],
        "day_user_ids": day_users,
        "day_dates": np.asarray([day[1] for day in days], dtype=np.str_),
        "day_splits": day_splits,
        "event_user_ids": np.repeat(day_users, lengths),
        "event_splits": np.repeat(day_splits, lengths),
        "grid_id": np.asarray([row["_grid_id"] for day in days for row in day[3]],
                              dtype=np.int64),
        "minute": np.asarray([row["minute"] for day in days for row in day[3]],
                             dtype=np.int16),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return arrays, receipt


def reference_cohort(artifact, *, max_users=None):
    """Identical train/test eligibility and day ordering to P3-E1."""
    offsets = np.asarray(artifact["day_offsets"], dtype=np.int64)
    users = np.asarray(artifact["day_user_ids"], dtype=np.str_)
    dates = np.asarray(artifact["day_dates"], dtype=np.str_)
    splits = np.asarray(artifact["day_splits"], dtype=np.str_)
    if len(offsets) != len(users) + 1 or len(dates) != len(users) \
            or len(splits) != len(users) or offsets[0] != 0 \
            or np.any(np.diff(offsets) <= 0) \
            or offsets[-1] != len(artifact["grid_id"]) \
            or len(artifact["minute"]) != offsets[-1]:
        raise ValueError("target reference day arrays are not aligned")
    if len(set(zip(users.tolist(), dates.tolist()))) != len(users):
        raise ValueError("target reference has duplicate user/date keys")
    by_user = defaultdict(set)
    for user, split in zip(users, splits):
        if split in ("train", "test"):
            by_user[str(user)].add(str(split))
    eligible = sorted(user for user, values in by_user.items()
                      if values == {"train", "test"})
    if max_users is not None:
        eligible = eligible[:int(max_users)]
    if len(eligible) < 4:
        raise RuntimeError("target cohort needs at least four users with train and test days")
    selected = set(eligible)
    indices = np.asarray([i for i, (user, split) in enumerate(zip(users, splits))
                          if user in selected and split in ("train", "test")],
                         dtype=np.int64)
    events = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices])
    lengths = np.diff(offsets)[indices]
    return {
        "users": users[indices], "dates": dates[indices],
        "splits": splits[indices], "offsets": np.r_[0, np.cumsum(lengths)],
        "minute": np.asarray(artifact["minute"])[events],
        "real_grid": np.asarray(artifact["grid_id"], dtype=np.int64)[events],
        "event_indices": events, "sequence_indices": indices,
        "eligible_users": eligible,
    }


def reference_payload(cohort, split="test"):
    chosen = np.flatnonzero(cohort["splits"] == split)
    offsets = cohort["offsets"]
    if not len(chosen):
        raise RuntimeError("target reference split is empty: " + split)
    events = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in chosen])
    lengths = np.diff(offsets)[chosen]
    return {
        "lengths": lengths,
        "values": cohort["real_grid"][events],
        "view": {
            "offsets": np.r_[0, np.cumsum(lengths)],
            "users": cohort["users"][chosen],
            "dates": cohort["dates"][chosen],
            "minute": cohort["minute"][events],
        },
    }


def target_train_homes(artifact, projected_grid, sequence_users):
    """The original projected train mode, with count-desc/grid-ID-asc ties."""
    counts = defaultdict(Counter)
    for user, split, grid in zip(artifact["event_user_ids"],
                                 artifact["event_splits"], projected_grid):
        if str(split) == "train":
            counts[str(user)][int(grid)] += 1
    mapping = {}
    for user in map(str, sequence_users):
        ranking = sorted(counts[user].items(), key=lambda item: (-item[1], item[0]))
        if not ranking:
            raise RuntimeError("target user has no train home: " + user)
        mapping[user] = ranking[0][0]
    return np.asarray([mapping[str(user)] for user in sequence_users], dtype=np.int64), {
        "contract": "projected_train_mode_count_desc_grid_id_asc_v1",
        "fit_split": "CA_day_split_train_only",
        "unique_user_count": len(mapping),
        "user_home_sha256": hashlib.sha256(json.dumps(
            sorted(mapping.items()), separators=(",", ":")
        ).encode()).hexdigest(),
    }


def target_train_distances(artifact, projected_grid, centers, vocabulary):
    """Physical same-day train transitions; target code labels are unnecessary."""
    positions = np.searchsorted(vocabulary, projected_grid)
    if np.any(positions >= len(vocabulary)) \
            or not np.array_equal(vocabulary[positions], projected_grid):
        raise ValueError("target train distances left the static vocabulary")
    offsets = np.asarray(artifact["day_offsets"], dtype=np.int64)
    pieces = [np.arange(start + 1, end) for split, start, end in zip(
        artifact["day_splits"], offsets[:-1], offsets[1:]
    ) if str(split) == "train" and end - start >= 2]
    if not pieces:
        raise RuntimeError("CA train has no within-day transitions for rescale")
    right = np.concatenate(pieces)
    left_points, right_points = centers[positions[right - 1]], centers[positions[right]]
    north = (right_points[:, 0] - left_points[:, 0]) * 111.32
    mean_lat = np.radians(0.5 * (left_points[:, 0] + right_points[:, 0]))
    east = (right_points[:, 1] - left_points[:, 1]) * 111.32 * np.cos(mean_lat)
    return np.sqrt(north * north + east * east)
