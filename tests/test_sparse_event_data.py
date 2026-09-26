import datetime as dt
import json
from pathlib import Path

import numpy as np

from SHAPE.gridspec import SparseLatentTable, StaticGrid
from SHAPE.stage2.sparse_event_data import (
    EVENT_DIM,
    GRID_ID_COLUMN,
    MISSING_LATENT_COLUMN,
    build_sparse_event_sequences,
    map_records_to_grid,
    save_sparse_event_dataset,
    split_users,
    write_ca_active_grid_set,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(latitude, longitude, minute, user="u0"):
    instant = dt.datetime(2020, 1, 1, 8, minute, tzinfo=dt.timezone.utc)
    return {
        "user_id": user, "venue_id": "v%d" % minute, "cat_name": "Food",
        "lat": float(latitude), "lng": float(longitude), "offset": 0,
        "local_dt": instant, "hour": instant.hour,
        "minute": instant.hour * 60 + instant.minute,
        "date": "2020-01-01", "weekday": instant.weekday(),
    }


def _paris_grid():
    return StaticGrid.load(
        REPO_ROOT / "dataset" / "PARIS_grid_merged_features.csv",
        city="paris", verify_release=True,
    )


def test_m2_mapping_drops_oob_and_keeps_in_bbox_non_emitted_cell():
    grid = _paris_grid()
    nominal = {
        row * grid.spec.id_stride + column
        for row, width in enumerate(grid.spec.n_cols_by_row)
        for column in range(int(width))
    }
    absent = min(nominal - set(grid.grid_ids.tolist()))
    emitted_center = grid.spec.center_of(int(grid.grid_ids[0]))
    absent_center = grid.spec.center_of(absent)
    records = [
        _record(*emitted_center, 0), _record(*absent_center, 1),
        _record(grid.spec.bbox["lat_max"] + 0.01, emitted_center[1], 2),
    ]
    mapped, report = map_records_to_grid(records, grid)
    assert report == {
        "input_events": 3, "kept_events": 2, "oob_dropped": 1,
        "in_bbox_missing_static": 1, "emitted_static_events": 1,
    }
    assert [int(row["_grid_id"]) for row in mapped] == [
        int(grid.grid_ids[0]), absent,
    ]


def test_m2_missing_latent_is_zero_flagged_without_breaking_sequence(tmp_path):
    grid = _paris_grid()
    nominal = {
        row * grid.spec.id_stride + column
        for row, width in enumerate(grid.spec.n_cols_by_row)
        for column in range(int(width))
    }
    absent = min(nominal - set(grid.grid_ids.tolist()))
    ids = [int(grid.grid_ids[0]), int(grid.grid_ids[1]), absent]
    records = [
        _record(*grid.spec.center_of(grid_id), index)
        for index, grid_id in enumerate(ids)
    ]
    mapped, _ = map_records_to_grid(records, grid)
    latent = SparseLatentTable(
        [ids[0]], np.ones((1, 64), dtype=np.float32)
    )
    tokens = {ids[0]: 1}
    surprisal = {
        "spa": np.ones((3, 64, 64), dtype=np.float32),
        "sem": np.ones((3, 13, 13), dtype=np.float32),
    }
    sequences, metadata = build_sparse_event_sequences(
        mapped, grid, latent, lambda _: 6, {}, tokens, surprisal,
    )
    sequence = sequences["u0"]
    assert sequence.shape == (3, EVENT_DIM)
    assert np.rint(sequence[:, GRID_ID_COLUMN]).astype(np.int64).tolist() == ids
    assert sequence[:, MISSING_LATENT_COLUMN].tolist() == [0.0, 1.0, 1.0]
    assert np.all(sequence[1:, 19:83] == 0.0)
    assert metadata[0]["missing_latent_events"] == 2

    splits = split_users(sequences)
    contract = save_sparse_event_dataset(
        sequences, metadata, splits, tmp_path, {"city": "paris"}
    )
    with np.load(tmp_path / "grid_ids_int64.npz") as archive:
        assert archive["arr_0"].dtype == np.int64
        assert archive["arr_0"].tolist() == ids
    assert contract["missing_latent_flag_column"] == MISSING_LATENT_COLUMN
    assert json.loads((tmp_path / "data_contract.json").read_text())["event_dim"] == EVENT_DIM


def test_ca_active_set_has_explicit_threshold_and_emitted_ring(tmp_path):
    grid = StaticGrid.load(
        REPO_ROOT / "dataset" / "CA_grid_merged_features.csv",
        city="ca", verify_release=True,
    )
    grid_id = int(grid.grid_ids[0])
    center = grid.spec.center_of(grid_id)
    records = [_record(*center, value, user="ca") for value in range(3)]
    mapped, _ = map_records_to_grid(records, grid)
    output = tmp_path / "CA_active_grids.csv"
    report = write_ca_active_grid_set(mapped, grid, output, min_checkins=2)
    assert report["active_grid_count"] == 1
    assert report["min_checkins"] == 2
    assert report["anchor_plus_ring_count"] >= 1
    assert "min_checkins" in output.read_text(encoding="utf-8").splitlines()[0]
