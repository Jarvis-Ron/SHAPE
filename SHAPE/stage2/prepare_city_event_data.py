"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
import ast
import csv
import datetime
import json
from pathlib import Path
import random
from zoneinfo import ZoneInfo
import numpy as np
import torch
from SHAPE.gridspec import SparseLatentTable, StaticGrid, sha256_file
from SHAPE.stage2.poi_category_mapping import POI_CATEGORY_MAP
from SHAPE.stage2.sparse_event_data import build_sparse_event_sequences, build_spatial_tokens, build_surprisal_tables, map_records_to_grid, save_sparse_event_dataset, split_users, write_ca_active_grid_set


REPO_ROOT = Path(__file__).resolve().parents[2]


CA_TIMEZONE = ZoneInfo("America/Los_Angeles")


CA_LOCAL_TIME_CONTRACT = "iana_timezone_v1"


CA_ACTIVE_PATH = REPO_ROOT / "dataset" / "CA_active_grids.csv"


CITY_CONFIG = {
    "nyc": {
        "label": "NYC",
        "grid_csv": "dataset/NYC_grid_merged_features.csv",
        "checkin_file": "dataset/checkin_NY_TY2014/dataset_TSMC2014_NYC.txt",
        "latent_file": "SHAPE/stage1/output/nyc_latent.npz",
        "out_dir": "SHAPE/stage2/nyc_event_driven",
    },
    "paris": {
        "label": "Paris",
        "grid_csv": "dataset/PARIS_grid_merged_features.csv",
        "checkin_file": "dataset/checkin_global2015/dataset_TIST2015_Checkins.txt",
        "poi_file": "dataset/checkin_global2015/dataset_TIST2015_POIs.txt",
        "latent_file": "SHAPE/stage1/output/paris_latent.npz",
        "out_dir": "SHAPE/stage2/paris_event_driven",
    },
    "ca": {
        "label": "CA",
        "grid_csv": "dataset/CA_grid_merged_features.csv",
        "checkin_files": [
            "dataset/Gowalla-CA/gowalla_train.csv",
            "dataset/Gowalla-CA/gowalla_val.csv",
            "dataset/Gowalla-CA/gowalla_test.csv",
        ],
        "latent_file": "SHAPE/stage1/output/ca_latent.npz",
        "out_dir": "SHAPE/stage2/ca_event_driven",
    },
}


def set_seed(seed=42):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _absolute(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_city_paths(path):
    """Override raw/static/output locations without changing feature semantics."""
    overrides = json.loads(Path(path).read_text())
    for city, values in overrides.items():
        if city not in CITY_CONFIG or not isinstance(values, dict):
            raise ValueError("City-path configuration requires nyc/ca/paris dictionaries")
        unknown = set(values) - set(CITY_CONFIG[city])
        if unknown:
            raise ValueError("Unknown city-path fields: %s" % sorted(unknown))
        CITY_CONFIG[city].update(values)


def load_nyc_checkins(path):
    records = []
    with _absolute(path).open("r", encoding="latin1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            utc = datetime.datetime.strptime(parts[7], "%a %b %d %H:%M:%S %z %Y")
            offset = int(parts[6])
            local = utc + datetime.timedelta(minutes=offset)
            records.append({
                "user_id": parts[0], "venue_id": parts[1],
                "cat_name": parts[3], "lat": float(parts[4]),
                "lng": float(parts[5]), "offset": offset,
                "local_dt": local, "hour": local.hour,
                "minute": local.hour * 60 + local.minute,
                "date": local.strftime("%Y-%m-%d"), "weekday": local.weekday(),
            })
    return records


def load_paris_venues(path, bbox):
    venues = {}
    with _absolute(path).open("r", encoding="latin1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            latitude, longitude = float(parts[1]), float(parts[2])
            if (parts[4] == "FR" and bbox["lat_min"] <= latitude <= bbox["lat_max"]
                    and bbox["lng_min"] <= longitude <= bbox["lng_max"]):
                venues[parts[0]] = {
                    "lat": latitude, "lng": longitude, "cat_name": parts[3],
                }
    return venues


def load_paris_checkins(path, venues):
    records = []
    with _absolute(path).open("r", encoding="latin1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[1] not in venues:
                continue
            utc = datetime.datetime.strptime(parts[2], "%a %b %d %H:%M:%S %z %Y")
            offset = int(parts[3])
            local = utc + datetime.timedelta(minutes=offset)
            venue = venues[parts[1]]
            records.append({
                "user_id": parts[0], "venue_id": parts[1],
                "cat_name": venue["cat_name"], "lat": venue["lat"],
                "lng": venue["lng"], "offset": offset,
                "local_dt": local, "hour": local.hour,
                "minute": local.hour * 60 + local.minute,
                "date": local.strftime("%Y-%m-%d"), "weekday": local.weekday(),
            })
    return records


def load_ca_checkins(csv_files):
    records = []
    for csv_path in csv_files:
        with _absolute(csv_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                category = row["POI_catname"]
                if category.startswith("[{") and "'name':" in category:
                    try:
                        parsed = ast.literal_eval(category.replace(";", ","))
                        category = parsed[0]["name"] if isinstance(parsed, list) else parsed["name"]
                    except (SyntaxError, ValueError, KeyError, TypeError):
                        category = "Other"
                utc = datetime.datetime.strptime(
                    row["checkin_time"], "%Y-%m-%d %H:%M:%S%z"
                )
                local = utc.astimezone(CA_TIMEZONE)
                records.append({
                    "user_id": row["user_id"], "venue_id": row["POI_id"],
                    "cat_name": category, "lat": float(row["latitude"]),
                    "lng": float(row["longitude"]),
                    "offset": int(local.utcoffset().total_seconds() // 60),
                    "local_dt": local, "hour": local.hour,
                    "minute": local.hour * 60 + local.minute,
                    "date": local.strftime("%Y-%m-%d"), "weekday": local.weekday(),
                })
    return records


def load_shared_venue_w2v(records=None):
    """Venue channels 91:123 are excluded by SHAPE's event encoder.

    Leave them zero instead of fitting an unused cross-city trajectory model.
    Archived event data remain supported without modification.
    """
    return {}


def map_poi_category(name):
    return int(POI_CATEGORY_MAP.get(name, 0))


def _load_city_records(city, static_grid):
    config = CITY_CONFIG[city]
    if city == "nyc":
        return load_nyc_checkins(config["checkin_file"])
    if city == "paris":
        venues = load_paris_venues(config["poi_file"], static_grid.spec.bbox)
        return load_paris_checkins(config["checkin_file"], venues)
    return load_ca_checkins(config["checkin_files"])


def prepare_active_grids(min_checkins=1):
    config = CITY_CONFIG["ca"]
    static_grid = StaticGrid.load(
        _absolute(config["grid_csv"]), city="ca", verify_release=True,
    )
    mapped, mapping = map_records_to_grid(
        _load_city_records("ca", static_grid), static_grid
    )
    active = write_ca_active_grid_set(
        mapped, static_grid, CA_ACTIVE_PATH, min_checkins=min_checkins
    )
    manifest = {
        "schema": "civic_ca_active_grid_manifest_v1",
        "city": "ca", "mapping": mapping, "active_set": active,
        "grid_csv_sha256": sha256_file(static_grid.csv_path),
        "grid_spec_schema": static_grid.spec.schema_version,
    }
    CA_ACTIVE_PATH.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def process_city(city, min_checkins=1, seed=42):
    city = str(city).lower()
    config = CITY_CONFIG[city]
    static_grid = StaticGrid.load(
        _absolute(config["grid_csv"]), city=city, verify_release=True,
    )
    mapped, mapping = map_records_to_grid(
        _load_city_records(city, static_grid), static_grid
    )
    active = None
    if city == "ca":
        active = write_ca_active_grid_set(
            mapped, static_grid, CA_ACTIVE_PATH, min_checkins=min_checkins
        )
    latent_path = _absolute(config["latent_file"])
    latent = SparseLatentTable.load(latent_path)
    if not np.all(static_grid.contains(latent.grid_ids)):
        raise RuntimeError("M1 latent table contains IDs outside the released static grid")
    token_by_grid, kmeans = build_spatial_tokens(latent)
    surprisal = build_surprisal_tables(mapped, token_by_grid, map_poi_category)
    sequences, metadata = build_sparse_event_sequences(
        mapped, static_grid, latent, map_poi_category,
        load_shared_venue_w2v(mapped), token_by_grid, surprisal,
    )
    splits = split_users(sequences, seed=seed)
    return save_sparse_event_dataset(
        sequences, metadata, splits, _absolute(config["out_dir"]), {
            "city": city, "mapping_counts": mapping, "active_set": active,
            "static_grid_rows": len(static_grid),
            "latent_rows": len(latent.grid_ids),
            "latent_path": str(latent_path),
            "latent_sha256": sha256_file(latent_path),
            "grid_csv_sha256": sha256_file(static_grid.csv_path),
            "grid_spec_schema": static_grid.spec.schema_version,
            "spatial_vocabulary_size": int(kmeans.n_clusters),
            "local_time_contract": (
                CA_LOCAL_TIME_CONTRACT if city == "ca"
                else "dataset_utc_offset_minutes_v1"
            ),
            "timezone": "America/Los_Angeles" if city == "ca" else None,
            "seed": int(seed),
        },
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("ca", "nyc", "paris", "all"), default="all")
    parser.add_argument("--config", help="JSON mapping of city names to raw/static/latent/output paths")
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--min-checkins", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.min_checkins < 1:
        parser.error("--min-checkins must be positive")
    if args.active_only and args.city not in ("ca", "all"):
        parser.error("--active-only is only valid for CA")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.config:
        load_city_paths(args.config)
    set_seed(args.seed)
    if args.active_only:
        result = prepare_active_grids(args.min_checkins)
    else:
        cities = ("ca", "nyc", "paris") if args.city == "all" else (args.city,)
        result = {
            city: process_city(city, args.min_checkins, args.seed) for city in cities
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
