"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Mapping
import numpy as np
from sklearn.cluster import KMeans
from SHAPE.gridspec import SparseLatentTable, StaticGrid, write_active_grid_ids


EVENT_SCHEMA = "civic_event_driven_sparse_grid_v2"


EVENT_DIM = 136


GRID_ID_COLUMN = 123


MISSING_LATENT_COLUMN = 135


GRID_EMBEDDING_SLICE = slice(19, 83)


N_POI_CATEGORIES = 13


SPATIAL_VOCAB_SIZE = 64


def map_records_to_grid(records, static_grid: StaticGrid):
    """Attach int64 nominal IDs, dropping (not clipping) bbox violations."""
    if not records:
        return [], {
            "input_events": 0, "kept_events": 0, "oob_dropped": 0,
            "in_bbox_missing_static": 0, "emitted_static_events": 0,
        }
    lat = np.asarray([row["lat"] for row in records], dtype=np.float64)
    lng = np.asarray([row["lng"] for row in records], dtype=np.float64)
    grid_ids = np.asarray(static_grid.spec.coords_to_grid(lat, lng), dtype=np.int64)
    kept = []
    missing_static = 0
    for record, grid_id in zip(records, grid_ids):
        if grid_id < 0:
            continue
        row = dict(record)
        row["_grid_id"] = np.int64(grid_id)
        row["_grid_index"] = np.int64(static_grid.indices(grid_id))
        row["_missing_static"] = bool(row["_grid_index"] < 0)
        missing_static += int(row["_missing_static"])
        kept.append(row)
    return kept, {
        "input_events": int(len(records)),
        "kept_events": int(len(kept)),
        "oob_dropped": int(len(records) - len(kept)),
        "in_bbox_missing_static": int(missing_static),
        "emitted_static_events": int(len(kept) - missing_static),
    }


def write_ca_active_grid_set(
    mapped_records,
    static_grid: StaticGrid,
    output: str | Path,
    min_checkins: int = 1,
):
    counts = Counter(
        int(row["_grid_id"]) for row in mapped_records
        if not row["_missing_static"]
    )
    active = sorted(
        grid_id for grid_id, count in counts.items()
        if count >= int(min_checkins)
    )
    write_active_grid_ids(output, active, min_checkins, counts)
    return {
        "path": str(output),
        "min_checkins": int(min_checkins),
        "active_grid_count": int(len(active)),
        "active_event_count": int(sum(counts[grid_id] for grid_id in active)),
        "anchor_plus_ring_count": int(len(static_grid.anchor_ids(active))),
    }


def _time2vec_minute(minute):
    values = []
    for frequency in range(1, 9):
        phase = 2.0 * math.pi * float(minute) / 1440.0 * frequency
        values.extend((math.sin(phase), math.cos(phase)))
    return np.asarray(values, dtype=np.float32)


def _poi_embedding(category):
    values = []
    for frequency in range(1, 5):
        phase = 2.0 * math.pi * int(category) / N_POI_CATEGORIES * frequency
        values.extend((math.sin(phase), math.cos(phase)))
    return np.asarray(values, dtype=np.float32)


def _normalized_delta(hours):
    return min(1.0, math.log1p(max(0.0, float(hours))) / math.log1p(720.0))


def _delta_bin(hours):
    if 0 < hours <= 2:
        return 0
    if 2 < hours <= 6:
        return 1
    if hours > 6:
        return 2
    return None


def _rank_surprisal(counts):
    counts = np.asarray(counts, dtype=np.float64) + 1e-5
    probabilities = counts / counts.sum(axis=-1, keepdims=True)
    result = np.ones_like(probabilities, dtype=np.float32)
    for bucket in range(len(probabilities)):
        matrix = probabilities[bucket]
        off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
        values = matrix[off_diagonal]
        order = np.argsort(values)
        ranks = np.zeros(len(values), dtype=np.float32)
        ranks[order] = np.arange(1, len(values) + 1, dtype=np.float32)
        result[bucket][off_diagonal] = ranks / max(1, len(values))
    return result


def build_spatial_tokens(latent_table: SparseLatentTable, n_clusters=64):
    clusters = min(int(n_clusters), len(latent_table.embeddings))
    if clusters < 1:
        raise RuntimeError("Spatial vocabulary needs at least one latent row")
    model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    labels = model.fit_predict(latent_table.embeddings).astype(np.int32)
    return {
        int(grid_id): int(label)
        for grid_id, label in zip(latent_table.grid_ids, labels)
    }, model


def build_surprisal_tables(mapped_records, token_by_grid, category_mapper):
    spatial = np.zeros((3, SPATIAL_VOCAB_SIZE, SPATIAL_VOCAB_SIZE), dtype=np.float64)
    semantic = np.zeros((3, N_POI_CATEGORIES, N_POI_CATEGORIES), dtype=np.float64)
    groups = defaultdict(list)
    for row in mapped_records:
        groups[(row["user_id"], row["date"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["local_dt"])
        for previous, current in zip(rows[:-1], rows[1:]):
            delta = (current["local_dt"] - previous["local_dt"]).total_seconds() / 3600.0
            bucket = _delta_bin(delta)
            if bucket is None:
                continue
            previous_token = token_by_grid.get(int(previous["_grid_id"]))
            current_token = token_by_grid.get(int(current["_grid_id"]))
            if previous_token is not None and current_token is not None:
                spatial[bucket, previous_token, current_token] += 1.0
            previous_category = int(category_mapper(previous["cat_name"]))
            current_category = int(category_mapper(current["cat_name"]))
            if (0 <= previous_category < N_POI_CATEGORIES and
                    0 <= current_category < N_POI_CATEGORIES):
                semantic[bucket, previous_category, current_category] += 1.0
    return {"spa": _rank_surprisal(spatial), "sem": _rank_surprisal(semantic)}


def _city_diagonal_km(static_grid):
    bbox = static_grid.spec.bbox
    south_west = np.asarray([[bbox["lat_min"], bbox["lng_min"]]])
    north_east = np.asarray([[bbox["lat_max"], bbox["lng_max"]]])
    lat1, lng1 = np.radians(south_west[0])
    lat2, lng2 = np.radians(north_east[0])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return max(1e-6, 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, value))))


def build_sparse_event_sequences(
    mapped_records,
    static_grid: StaticGrid,
    latent_table: SparseLatentTable,
    category_mapper,
    venue_embeddings: Mapping[str, np.ndarray],
    token_by_grid: Mapping[int, int],
    surprisal_tables,
):
    """Build v2 event arrays; missing emitted/latent cells remain in sequence."""
    groups = defaultdict(list)
    for row in mapped_records:
        value = dict(row)
        value["_poi_cat"] = int(category_mapper(value["cat_name"]))
        groups[value["user_id"]].append(value)
    for rows in groups.values():
        rows.sort(key=lambda row: row["local_dt"])
    groups = {user: rows for user, rows in groups.items() if len(rows) >= 3}
    diagonal_km = _city_diagonal_km(static_grid)
    sequences, metadata = {}, []
    for user in sorted(groups, key=lambda value: (-len(groups[value]), value)):
        rows = groups[user]
        output = np.zeros((len(rows), EVENT_DIM), dtype=np.float32)
        grid_counter = Counter()
        weekday_counter = defaultdict(Counter)
        weekday_total = Counter()
        hour_counter = defaultdict(Counter)
        hour_total = Counter()
        last_visit = {}
        intervals = defaultdict(list)
        visited_centers = []
        for index, row in enumerate(rows):
            grid_id = int(row["_grid_id"])
            poi = max(0, min(N_POI_CATEGORIES - 1, int(row["_poi_cat"])))
            latent, latent_present = latent_table.lookup(grid_id)
            output[index, 0:16] = _time2vec_minute(row["minute"])
            if index:
                delta_hours = (
                    row["local_dt"] - rows[index - 1]["local_dt"]
                ).total_seconds() / 3600.0
                output[index, 16] = _normalized_delta(delta_hours)
            output[index, 17] = row["weekday"] / 7.0
            output[index, 18] = row["hour"]
            output[index, GRID_EMBEDDING_SLICE] = latent
            output[index, 83:91] = _poi_embedding(poi)
            output[index, 91:123] = venue_embeddings.get(
                row["venue_id"], np.zeros(32, dtype=np.float32)
            )
            output[index, GRID_ID_COLUMN] = grid_id
            output[index, 124] = poi
            token = token_by_grid.get(grid_id)
            if index == 0:
                output[index, 125:127] = 1.0
            else:
                previous = rows[index - 1]
                previous_token = token_by_grid.get(int(previous["_grid_id"]))
                delta_hours = (
                    row["local_dt"] - previous["local_dt"]
                ).total_seconds() / 3600.0
                bucket = _delta_bin(delta_hours)
                if bucket is not None and previous_token is not None and token is not None:
                    output[index, 125] = surprisal_tables["spa"][
                        bucket, previous_token, token
                    ]
                    output[index, 126] = surprisal_tables["sem"][
                        bucket,
                        max(0, min(N_POI_CATEGORIES - 1, int(previous["_poi_cat"]))),
                        poi,
                    ]
                else:
                    output[index, 125:127] = 1.0
                distance_km = float(static_grid.spec.distance_km(
                    int(previous["_grid_id"]), grid_id
                ))
                output[index, 127] = min(1.0, distance_km / diagonal_km)
            total = max(1, index)
            output[index, 128] = grid_counter[grid_id] / total
            output[index, 129] = (
                weekday_counter[row["weekday"]][grid_id]
                / max(1, weekday_total[row["weekday"]])
            )
            output[index, 130] = (
                hour_counter[row["hour"]][grid_id]
                / max(1, hour_total[row["hour"]])
            )
            center = static_grid.spec.center_of(grid_id)
            visited_centers.append(center)
            if len(visited_centers) > 1:
                points = np.asarray(visited_centers)
                centroid = points.mean(axis=0)
                lat_km = (points[:, 0] - centroid[0]) * 111.32
                lng_km = (points[:, 1] - centroid[1]) * 111.32 * math.cos(
                    math.radians(float(centroid[0]))
                )
                output[index, 131] = min(
                    1.0, float(np.sqrt(lat_km ** 2 + lng_km ** 2).mean()) / 50.0
                )
            if index:
                counts = np.asarray(list(grid_counter.values()), dtype=np.float64)
                probabilities = counts / counts.sum()
                entropy = -float(np.sum(probabilities * np.log(probabilities)))
                output[index, 132] = entropy / max(math.log(max(2, len(static_grid))), 1.0)
                output[index, 133] = 1.0 - len(grid_counter) / max(1, index)
            if grid_id in last_visit:
                elapsed = (
                    row["local_dt"] - rows[last_visit[grid_id]]["local_dt"]
                ).total_seconds() / 3600.0
                intervals[grid_id].append(elapsed)
            if len(intervals[grid_id]) >= 2:
                values = np.asarray(intervals[grid_id])
                output[index, 134] = 1.0 - min(
                    1.0, float(values.std() / max(values.mean(), 1e-8)) / 3.0
                )
            output[index, MISSING_LATENT_COLUMN] = float(
                row["_missing_static"] or not latent_present
            )
            grid_counter[grid_id] += 1
            weekday_counter[row["weekday"]][grid_id] += 1
            weekday_total[row["weekday"]] += 1
            hour_counter[row["hour"]][grid_id] += 1
            hour_total[row["hour"]] += 1
            last_visit[grid_id] = index
        sequences[user] = output
        metadata.append({
            "user_id": user, "length": len(rows), "seq_idx": len(metadata),
            "missing_latent_events": int(output[:, MISSING_LATENT_COLUMN].sum()),
        })
    return sequences, metadata


def split_users(sequences, seed=42, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.default_rng(int(seed))
    ordered = sorted(sequences, key=lambda user: len(sequences[user]))
    layers = np.array_split(np.asarray(ordered, dtype=object), 3)
    result = {"train": set(), "val": set(), "test": set()}
    for layer in layers:
        layer = layer.copy()
        rng.shuffle(layer)
        train_end = int(len(layer) * train_ratio)
        val_end = train_end + int(len(layer) * val_ratio)
        result["train"].update(map(str, layer[:train_end]))
        result["val"].update(map(str, layer[train_end:val_end]))
        result["test"].update(map(str, layer[val_end:]))
    return result


def save_sparse_event_dataset(
    sequences, metadata, splits, output_dir: str | Path, contract,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    users = sorted(sequences)
    index = {user: value for value, user in enumerate(users)}
    np.savez_compressed(
        output_dir / "event_driven_data.npz",
        **{"arr_%d" % index[user]: sequences[user] for user in users},
    )
    np.savez_compressed(
        output_dir / "grid_ids_int64.npz",
        **{
            "arr_%d" % index[user]: np.rint(
                sequences[user][:, GRID_ID_COLUMN]
            ).astype(np.int64)
            for user in users
        },
    )
    metadata_by_user = {row["user_id"]: row for row in metadata}
    rows = []
    for user in users:
        split = next(name for name, values in splits.items() if user in values)
        rows.append({**metadata_by_user[user], "seq_idx": index[user], "split": split})
    (output_dir / "user_meta.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for split, values in splits.items():
        np.save(
            output_dir / (split + "_user_ids.npy"),
            np.asarray(sorted(index[user] for user in values), dtype=np.int32),
        )
    payload = {
        "schema": EVENT_SCHEMA,
        "event_dim": EVENT_DIM,
        "grid_id_contract": "raw_nominal_int64_in_grid_ids_int64.npz",
        "float_feature_grid_id_column": GRID_ID_COLUMN,
        "missing_latent_flag_column": MISSING_LATENT_COLUMN,
        **contract,
    }
    (output_dir / "data_contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
