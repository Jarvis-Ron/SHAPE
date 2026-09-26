"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import numpy as np


SOURCE_VOCABULARY_PROJECTION_CONTRACT = (
    "nearest_source_vocabulary_cell_in_nominal_row_col_space_v1"
)


def _project_nominal_ids_to_vocabulary(values, vocabulary, spec):
    """Project sparse nominal IDs to the nearest representable source cell.

    The landing tables contain only emitted/active cells, while generated
    homes and reference events retain stable nominal GridSpec IDs.  Missing
    IDs are therefore a normal sparse-grid condition rather than corrupt
    data.  Ties are resolved by the smaller stable grid ID so reruns are
    deterministic.
    """
    input_values = np.asarray(values, dtype=np.int64)
    flat_values = input_values.reshape(-1)
    vocabulary = np.asarray(vocabulary, dtype=np.int64).reshape(-1)
    if len(vocabulary) == 0:
        raise RuntimeError("Landing vocabulary is empty")
    if len(np.unique(vocabulary)) != len(vocabulary) \
            or np.any(np.diff(vocabulary) <= 0):
        raise RuntimeError("Landing vocabulary must be strictly increasing and unique")

    vocabulary_rows, vocabulary_cols = spec.row_col(vocabulary)
    vocabulary_rows = np.asarray(vocabulary_rows, dtype=np.int64)
    vocabulary_cols = np.asarray(vocabulary_cols, dtype=np.int64)
    if np.any(vocabulary_rows < 0) or np.any(vocabulary_cols < 0):
        raise RuntimeError("Landing vocabulary contains non-nominal grid IDs")

    value_rows, value_cols = spec.row_col(flat_values)
    value_rows = np.asarray(value_rows, dtype=np.int64)
    value_cols = np.asarray(value_cols, dtype=np.int64)
    invalid = (value_rows < 0) | (value_cols < 0)
    if np.any(invalid):
        bad = np.unique(flat_values[invalid])[:5].tolist()
        raise RuntimeError(
            "Source artifact contains non-nominal grid IDs: %s" % bad
        )

    positions = np.searchsorted(vocabulary, flat_values)
    safe = np.clip(positions, 0, len(vocabulary) - 1)
    present = (positions < len(vocabulary)) & (vocabulary[safe] == flat_values)
    projected = flat_values.copy()
    missing_values = np.unique(flat_values[~present])
    replacement_ids = np.empty(len(missing_values), dtype=np.int64)
    projection_distances = np.empty(len(missing_values), dtype=np.float64)

    row_candidates = {}
    for row in np.unique(vocabulary_rows):
        mask = vocabulary_rows == row
        order = np.argsort(vocabulary_cols[mask], kind="stable")
        row_candidates[int(row)] = (
            vocabulary_cols[mask][order], vocabulary[mask][order]
        )

    for index, value in enumerate(missing_values):
        row, col = spec.row_col(int(value))
        best = None
        for candidate_row, (candidate_cols, candidate_ids) in row_candidates.items():
            row_distance = int(candidate_row - row) ** 2
            if best is not None and row_distance > best[0]:
                continue
            insertion = int(np.searchsorted(candidate_cols, int(col)))
            for candidate_index in (insertion - 1, insertion):
                if not 0 <= candidate_index < len(candidate_cols):
                    continue
                distance = row_distance + int(
                    int(candidate_cols[candidate_index]) - int(col)
                ) ** 2
                candidate = (distance, int(candidate_ids[candidate_index]))
                if best is None or candidate < best:
                    best = candidate
        if best is None:  # guarded by the non-empty vocabulary check
            raise RuntimeError("Unable to project source grid ID %d" % int(value))
        replacement_ids[index] = best[1]
        projection_distances[index] = np.sqrt(float(best[0]))

    projected_distances = np.empty(0, dtype=np.float64)
    if len(missing_values):
        missing_positions = np.searchsorted(missing_values, flat_values[~present])
        projected[~present] = replacement_ids[missing_positions]
        projected_distances = projection_distances[missing_positions]
    projected_count = int((~present).sum())
    report = {
        "contract": SOURCE_VOCABULARY_PROJECTION_CONTRACT,
        "input_values": int(len(flat_values)),
        "input_unique_ids": int(len(np.unique(flat_values))),
        "vocabulary_size": int(len(vocabulary)),
        "projected_values": projected_count,
        "projected_fraction": (
            float(projected_count / len(flat_values)) if len(flat_values) else 0.0
        ),
        "projected_unique_ids": int(len(missing_values)),
        "projection_distance_cells_mean": (
            float(projected_distances.mean()) if len(projected_distances) else 0.0
        ),
        "projection_distance_cells_max": (
            float(projected_distances.max()) if len(projected_distances) else 0.0
        ),
        "examples": [
            {"source_grid_id": int(source), "projected_grid_id": int(target)}
            for source, target in zip(missing_values[:5], replacement_ids[:5])
        ],
    }
    return projected.reshape(input_values.shape), report
