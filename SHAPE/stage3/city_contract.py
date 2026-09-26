"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import defaultdict
from SHAPE.gridspec import StaticGrid
from SHAPE.stage2.prepare_city_event_data import CITY_CONFIG, _load_city_records, load_ca_checkins
from SHAPE.stage2.sparse_event_data import map_records_to_grid


SOURCE_CITIES = ("ca", "nyc", "paris")


def canonical_source_city(city):
    value = str(city).strip().lower()
    if value not in SOURCE_CITIES:
        raise ValueError("source city must be ca, nyc, or paris")
    return value


def city_paths(city):
    city = canonical_source_city(city)
    config = CITY_CONFIG[city]
    return {
        "city": city,
        "data_dir": str(config["out_dir"]),
        "grid_csv": str(config["grid_csv"]),
        "grid_latents": str(config["latent_file"]),
    }


def load_raw_calendar(city, raw_checkins=None, grid_csv=None):
    """Rebuild event-local timestamps with the exact Module-2 parsers.

    The raw records are used only to recover date and minute fields missing
    from the prepared event archive.  Geometry filtering and chronological
    ordering intentionally mirror ``prepare_city_event_data``.
    """
    city = canonical_source_city(city)
    config = CITY_CONFIG[city]
    grid_csv = str(grid_csv or config["grid_csv"])
    static_grid = StaticGrid.load(grid_csv, city=city, verify_release=True)
    if raw_checkins is not None:
        if city != "ca":
            raise ValueError(
                "--raw-checkins overrides are supported only for CA; NYC and "
                "Paris use their registered raw data plus POI metadata paths"
            )
        records = load_ca_checkins(list(raw_checkins))
    else:
        records = _load_city_records(city, static_grid)
    mapped, _ = map_records_to_grid(records, static_grid)
    by_user = defaultdict(list)
    for row in mapped:
        by_user[str(row["user_id"])].append(row["local_dt"])
    for values in by_user.values():
        values.sort()
    if not by_user:
        raise RuntimeError("No %s check-ins were loaded" % city.upper())
    return by_user
