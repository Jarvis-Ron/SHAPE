"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


FEATURE_NAMES = (
    tuple("poi_prop_%d" % index for index in range(13))
    + ("poi_density_level", "total_stations", "metro_stations",
       "bus_stations", "total_frequency", "metro_frequency",
       "bus_frequency")
)


def _load_raw_table(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Static grid table is empty: %s" % path)
    rows.sort(key=lambda row: int(row["grid_id"]))
    table = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    missing = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    seen = set()
    grid_ids = []
    for row_index, row in enumerate(rows):
        try:
            grid_id = int(row["grid_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Invalid grid_id in %s" % path) from error
        if grid_id < 0 or grid_id in seen:
            raise RuntimeError("Grid IDs must be unique non-negative int64 in %s" % path)
        seen.add(grid_id)
        grid_ids.append(grid_id)
        for index, name in enumerate(FEATURE_NAMES):
            value = row.get(name)
            try:
                number = float(value) if value not in (None, "") else float("nan")
            except (TypeError, ValueError):
                number = float("nan")
            if not np.isfinite(number):
                missing[index] += 1
                number = 0.0
            table[row_index, index] = number
    rates = {
        name: float(missing[index] / len(rows))
        for index, name in enumerate(FEATURE_NAMES)
    }
    return table, rates
