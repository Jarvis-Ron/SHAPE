"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import math
from collections import OrderedDict
import numpy as np
from SHAPE.stage3.source_cohort import MAIN_GRANULARITY, REPORT_METRICS
from SHAPE.stage3.sampler import SPATIAL_GRANULARITIES


X1_ANALYSIS = "X1_HRA_map_rescaled_attraction"


def _metrics_with_ratios(metrics):
    output = OrderedDict((metric, metrics.get(metric)) for metric in REPORT_METRICS)
    output["A_visit_entropy_ratio"] = float(
        output["A_visit_entropy_predicted"]
        / max(float(output["A_visit_entropy_reference"]), 1e-12)
    )
    output["A_grid_coverage_ratio"] = float(
        output["A_grid_coverage_predicted"]
        / max(float(output["A_grid_coverage_reference"]), 1e-12)
    )
    if any(
        value is None or not math.isfinite(float(value)) for value in output.values()
    ):
        missing = [
            key
            for key, value in output.items()
            if value is None or not math.isfinite(float(value))
        ]
        raise RuntimeError("P3-E4 metric is undefined: " + ", ".join(missing))
    return output


def _coarse_zone_lookup(vocabulary, grid_spec):
    """Match ``build_population_sketch`` coarse OD zone assignment exactly."""
    vocabulary = np.asarray(vocabulary, dtype=np.int64)
    row_bins, column_bins = SPATIAL_GRANULARITIES[MAIN_GRANULARITY]
    if grid_spec is None:
        fixture_width = max(1, int(np.ceil(np.sqrt(len(vocabulary)))))
        rows, columns = np.divmod(np.arange(len(vocabulary)), fixture_width)
        fixture_rows = max(1, int(np.ceil(len(vocabulary) / fixture_width)))
        coarse_rows = np.minimum(
            row_bins - 1, rows * row_bins // fixture_rows
        )
        coarse_columns = np.minimum(
            column_bins - 1, columns * column_bins // fixture_width
        )
    else:
        rows, columns = grid_spec.row_col(vocabulary)
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if np.any(rows < 0) or np.any(columns < 0):
            raise RuntimeError("P3-E4 target vocabulary left the frozen grid spec")
        widths = grid_spec.n_cols_by_row[rows]
        coarse_rows = np.minimum(
            row_bins - 1, rows * row_bins // grid_spec.n_rows
        )
        coarse_columns = np.minimum(
            column_bins - 1, columns * column_bins // widths
        )
    return (
        coarse_rows * int(column_bins) + coarse_columns
    ).astype(np.int64, copy=False)
