"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence
import numpy as np


FEATURE_NAMES = (
    tuple("poi_prop_%d" % index for index in range(13))
    + (
        "poi_density_level",
        "total_stations",
        "metro_stations",
        "bus_stations",
        "total_frequency",
        "metro_frequency",
        "bus_frequency",
    )
)


DEFAULT_GRID_SPEC = Path(__file__).resolve().parents[1] / "dataset" / "grid_spec.json"


DEFAULT_RELEASE_MANIFEST = (
    Path(__file__).resolve().parents[1] / "dataset" / "candidate_release_v4.json"
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_city(city: str) -> str:
    value = str(city).strip().upper()
    aliases = {"NEW_YORK": "NYC", "NEW YORK": "NYC", "LOS_ANGELES": "CA"}
    value = aliases.get(value, value)
    if value not in {"CA", "NYC", "PARIS"}:
        raise ValueError("city must be CA, NYC, or PARIS")
    return value


def infer_city(path: str | Path) -> str:
    name = Path(path).name.upper()
    for city in ("PARIS", "NYC", "CA"):
        if city in name:
            return city
    raise ValueError("Cannot infer city from path %s; pass city explicitly" % path)


def _as_array(values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    return array, array.ndim == 0


def _scalar_or_array(value, scalar):
    if scalar:
        return np.asarray(value).reshape(()).item()
    return value


@dataclass(frozen=True)
class GridSpec:
    """Frozen nominal geometry for one city, with south-origin row IDs."""

    city: str
    schema_version: str
    n_rows: int
    id_stride: int
    n_cols_by_row: np.ndarray
    dlat: float
    dlng_by_row: np.ndarray
    southwest: Mapping[str, float]
    bbox: Mapping[str, float]
    L_m: float
    emitted_rows: int
    row_direction: str = "south_to_north"
    col_direction: str = "west_to_east"

    @classmethod
    def load(
        cls,
        city: str,
        path: str | Path = DEFAULT_GRID_SPEC,
    ) -> "GridSpec":
        with Path(path).open("r", encoding="utf-8") as handle:
            root = json.load(handle)
        city_key = canonical_city(city)
        try:
            payload = root["cities"][city_key]
        except KeyError as exc:
            raise RuntimeError("grid_spec.json is missing city %s" % city_key) from exc
        n_rows = int(payload["n_rows"])
        n_cols = np.asarray(payload["n_cols_by_row"], dtype=np.int64)
        dlng = np.asarray(payload["dlng_by_row"], dtype=np.float64)
        if n_cols.shape != (n_rows,) or dlng.shape != (n_rows,):
            raise RuntimeError("%s grid spec has inconsistent row arrays" % city_key)
        if payload.get("row_direction") != "south_to_north" \
                or payload.get("col_direction") != "west_to_east":
            raise RuntimeError("Unsupported grid axis direction for %s" % city_key)
        spec = cls(
            city=city_key,
            schema_version=str(root["schema_version"]),
            n_rows=n_rows,
            id_stride=int(payload["id_stride"]),
            n_cols_by_row=n_cols,
            dlat=float(payload["dlat"]),
            dlng_by_row=dlng,
            southwest={key: float(value) for key, value in payload["southwest"].items()},
            bbox={key: float(value) for key, value in payload["bbox"].items()},
            L_m=float(payload.get("L_m", root["L_m"])),
            emitted_rows=int(payload["emitted_rows"]),
            row_direction=str(payload["row_direction"]),
            col_direction=str(payload["col_direction"]),
        )
        if spec.id_stride < int(spec.n_cols_by_row.max()):
            raise RuntimeError("id_stride is smaller than an emitted nominal row")
        return spec

    @property
    def nominal_grid_count(self) -> int:
        return int(self.n_cols_by_row.sum())

    @property
    def max_grid_id(self) -> int:
        return (self.n_rows - 1) * self.id_stride + int(self.n_cols_by_row[-1]) - 1

    def row_col(self, grid_ids):
        ids, scalar = _as_array(grid_ids, np.int64)
        rows = np.floor_divide(ids, self.id_stride)
        cols = np.remainder(ids, self.id_stride)
        valid = (
            (ids >= 0)
            & (rows >= 0)
            & (rows < self.n_rows)
        )
        safe_rows = np.clip(rows, 0, self.n_rows - 1)
        valid &= cols < self.n_cols_by_row[safe_rows]
        rows = np.where(valid, rows, -1).astype(np.int64)
        cols = np.where(valid, cols, -1).astype(np.int64)
        return _scalar_or_array(rows, scalar), _scalar_or_array(cols, scalar)

    def valid_nominal_ids(self, grid_ids):
        rows, cols = self.row_col(grid_ids)
        return (np.asarray(rows) >= 0) & (np.asarray(cols) >= 0)

    def coords_to_grid(self, lats, lngs):
        """Map coordinates to nominal IDs; out-of-bounds points return ``-1``.

        Boundary points on the frozen north/east edge belong to the final
        clipped cell.  Values outside the bbox are never clipped inward.
        """
        lat, lat_scalar = _as_array(lats, np.float64)
        lng, lng_scalar = _as_array(lngs, np.float64)
        lat, lng = np.broadcast_arrays(lat, lng)
        scalar = lat_scalar and lng_scalar
        finite = np.isfinite(lat) & np.isfinite(lng)
        inside = (
            finite
            & (lat >= self.bbox["lat_min"])
            & (lat <= self.bbox["lat_max"])
            & (lng >= self.bbox["lng_min"])
            & (lng <= self.bbox["lng_max"])
        )
        safe_lat = np.where(finite, lat, self.southwest["lat"])
        safe_lng = np.where(finite, lng, self.southwest["lng"])
        rows = np.floor(
            (safe_lat - self.southwest["lat"]) / self.dlat
        ).astype(np.int64)
        rows = np.where(lat == self.bbox["lat_max"], self.n_rows - 1, rows)
        safe_rows = np.clip(rows, 0, self.n_rows - 1)
        cols = np.floor(
            (safe_lng - self.southwest["lng"]) / self.dlng_by_row[safe_rows]
        ).astype(np.int64)
        east = lng == self.bbox["lng_max"]
        cols = np.where(east, self.n_cols_by_row[safe_rows] - 1, cols)
        valid = (
            inside
            & (rows >= 0)
            & (rows < self.n_rows)
            & (cols >= 0)
            & (cols < self.n_cols_by_row[safe_rows])
        )
        ids = np.where(valid, rows * self.id_stride + cols, -1).astype(np.int64)
        return _scalar_or_array(ids, scalar)

    def bounds_of(self, grid_ids):
        ids, scalar = _as_array(grid_ids, np.int64)
        rows, cols = self.row_col(ids)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        if np.any(rows < 0):
            raise ValueError("bounds_of received a non-nominal grid ID")
        lat_min = self.southwest["lat"] + rows * self.dlat
        lat_max = np.minimum(lat_min + self.dlat, self.bbox["lat_max"])
        lng_min = self.southwest["lng"] + cols * self.dlng_by_row[rows]
        lng_max = np.minimum(lng_min + self.dlng_by_row[rows], self.bbox["lng_max"])
        result = np.stack((lat_min, lat_max, lng_min, lng_max), axis=-1)
        return result.reshape(4) if scalar else result

    def center_of(self, grid_ids):
        bounds = self.bounds_of(grid_ids)
        scalar = np.asarray(bounds).ndim == 1
        values = np.asarray(bounds).reshape(-1, 4)
        centers = np.column_stack((
            0.5 * (values[:, 0] + values[:, 1]),
            0.5 * (values[:, 2] + values[:, 3]),
        ))
        return centers[0] if scalar else centers

    def neighbors(self, grid_id: int, diagonals: bool = True) -> np.ndarray:
        row, col = self.row_col(int(grid_id))
        if row < 0:
            return np.empty(0, dtype=np.int64)
        result = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if not diagonals and abs(dr) + abs(dc) != 1:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols_by_row[nr]:
                    result.append(nr * self.id_stride + nc)
        return np.asarray(result, dtype=np.int64)

    def distance_km(self, left_ids, right_ids):
        left = np.asarray(self.center_of(left_ids), dtype=np.float64)
        right = np.asarray(self.center_of(right_ids), dtype=np.float64)
        left, right = np.broadcast_arrays(left, right)
        lat1, lng1 = np.radians(left[..., 0]), np.radians(left[..., 1])
        lat2, lng2 = np.radians(right[..., 0]), np.radians(right[..., 1])
        dlat, dlng = lat2 - lat1, lng2 - lng1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
        return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.minimum(1.0, a)))


@dataclass(frozen=True)
class GridTile:
    core_indices: np.ndarray
    context_indices: np.ndarray
    core_ids: np.ndarray
    context_ids: np.ndarray
    relative_rows: np.ndarray
    relative_cols: np.ndarray


class StaticGrid:
    """Sorted emitted grid rows plus sparse ID/index operations."""

    def __init__(
        self,
        spec: GridSpec,
        grid_ids: np.ndarray,
        features: np.ndarray,
        centers: np.ndarray,
        bounds: np.ndarray,
        csv_path: str | Path,
        active_ids: Optional[np.ndarray] = None,
    ):
        order = np.argsort(np.asarray(grid_ids, dtype=np.int64), kind="stable")
        self.spec = spec
        self.grid_ids = np.asarray(grid_ids, dtype=np.int64)[order]
        self.features = np.asarray(features, dtype=np.float32)[order]
        self.centers = np.asarray(centers, dtype=np.float64)[order]
        self.bounds = np.asarray(bounds, dtype=np.float64)[order]
        self.csv_path = Path(csv_path)
        if len(self.grid_ids) != spec.emitted_rows:
            raise RuntimeError(
                "%s emitted-row mismatch: CSV=%d spec=%d"
                % (spec.city, len(self.grid_ids), spec.emitted_rows)
            )
        if len(np.unique(self.grid_ids)) != len(self.grid_ids):
            raise RuntimeError("Static grid contains duplicate grid IDs")
        if not np.all(spec.valid_nominal_ids(self.grid_ids)):
            raise RuntimeError("Static grid contains IDs outside nominal geometry")
        expected_bounds = np.asarray(spec.bounds_of(self.grid_ids), dtype=np.float64)
        if not np.allclose(
            self.bounds, expected_bounds, rtol=0.0, atol=1e-8, equal_nan=False
        ):
            maximum = float(np.max(np.abs(self.bounds - expected_bounds)))
            raise RuntimeError(
                "Static CSV cell bounds disagree with GridSpec (max error %.3g)"
                % maximum
            )
        if self.features.shape != (len(self.grid_ids), len(FEATURE_NAMES)):
            raise RuntimeError("Static feature table must have shape [N, 20]")
        self.rows, self.cols = spec.row_col(self.grid_ids)
        self.zero_mask = np.isclose(np.abs(self.features).sum(axis=1), 0.0)
        self.valid_mask = ~self.zero_mask
        if active_ids is None:
            self.active_ids = None
        else:
            values = np.unique(np.asarray(active_ids, dtype=np.int64))
            present = self.contains(values)
            if not np.all(present):
                missing = values[~present]
                raise RuntimeError(
                    "Active-grid file contains %d non-emitted IDs (first=%s)"
                    % (len(missing), missing[:5].tolist())
                )
            self.active_ids = values

    @classmethod
    def load(
        cls,
        csv_path: str | Path,
        city: Optional[str] = None,
        grid_spec_path: str | Path = DEFAULT_GRID_SPEC,
        release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST,
        verify_release: bool = False,
        active_ids_path: Optional[str | Path] = None,
        min_checkins: int = 1,
    ) -> "StaticGrid":
        csv_path = Path(csv_path)
        city_key = canonical_city(city or infer_city(csv_path))
        spec = GridSpec.load(city_key, grid_spec_path)
        if verify_release:
            verify_candidate_release(
                city_key, csv_path, grid_spec_path, release_manifest_path
            )
        ids, features, centers, bounds = [], [], [], []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_columns = [
                name for name in ("grid_id", *FEATURE_NAMES, "grid_lat_min",
                                  "grid_lat_max", "grid_lng_min", "grid_lng_max")
                if name not in (reader.fieldnames or ())
            ]
            if missing_columns:
                raise RuntimeError("Static CSV is missing columns: %s" % missing_columns)
            for row in reader:
                grid_id = int(row["grid_id"])
                ids.append(grid_id)
                features.append([float(row[name] or 0.0) for name in FEATURE_NAMES])
                cell_bounds = [
                    float(row["grid_lat_min"]), float(row["grid_lat_max"]),
                    float(row["grid_lng_min"]), float(row["grid_lng_max"]),
                ]
                bounds.append(cell_bounds)
                centers.append([
                    0.5 * (cell_bounds[0] + cell_bounds[1]),
                    0.5 * (cell_bounds[2] + cell_bounds[3]),
                ])
        active_ids = None
        if active_ids_path is not None:
            active_ids = load_active_grid_ids(active_ids_path, min_checkins)
        return cls(
            spec, np.asarray(ids, dtype=np.int64),
            np.asarray(features, dtype=np.float32),
            np.asarray(centers, dtype=np.float64),
            np.asarray(bounds, dtype=np.float64), csv_path, active_ids,
        )

    def __len__(self) -> int:
        return len(self.grid_ids)

    def indices(self, grid_ids, missing: int = -1):
        values, scalar = _as_array(grid_ids, np.int64)
        flat = values.ravel()
        positions = np.searchsorted(self.grid_ids, flat)
        safe = np.clip(positions, 0, max(0, len(self.grid_ids) - 1))
        found = (positions < len(self.grid_ids)) & (self.grid_ids[safe] == flat)
        result = np.where(found, positions, int(missing)).reshape(values.shape)
        return _scalar_or_array(result.astype(np.int64), scalar)

    def contains(self, grid_ids):
        return np.asarray(self.indices(grid_ids)) >= 0

    def ids_from_indices(self, indices):
        values, scalar = _as_array(indices, np.int64)
        if np.any(values < 0) or np.any(values >= len(self)):
            raise IndexError("Static-grid index out of range")
        result = self.grid_ids[values]
        return _scalar_or_array(result, scalar)

    def features_for(self, grid_ids):
        values, scalar = _as_array(grid_ids, np.int64)
        indices = np.asarray(self.indices(values))
        found = indices >= 0
        result = np.zeros(values.shape + (self.features.shape[1],), dtype=np.float32)
        result[found] = self.features[indices[found]]
        if scalar:
            return result.reshape(-1), bool(found.reshape(()))
        return result, found

    def centers_for(self, grid_ids, missing_value=np.nan):
        values, scalar = _as_array(grid_ids, np.int64)
        indices = np.asarray(self.indices(values))
        found = indices >= 0
        result = np.full(values.shape + (2,), missing_value, dtype=np.float64)
        result[found] = self.centers[indices[found]]
        return result.reshape(2) if scalar else result

    def emitted_neighbors(self, grid_id: int, diagonals: bool = True) -> np.ndarray:
        nominal = self.spec.neighbors(grid_id, diagonals=diagonals)
        return nominal[self.contains(nominal)]

    def anchor_ids(self, active_ids: Optional[Sequence[int]] = None) -> np.ndarray:
        active = self.active_ids if active_ids is None else np.asarray(active_ids, dtype=np.int64)
        if active is None:
            return self.grid_ids.copy()
        anchors = set(int(value) for value in active)
        for grid_id in active:
            anchors.update(int(value) for value in self.emitted_neighbors(int(grid_id)))
        return np.asarray(sorted(anchors), dtype=np.int64)

    def tiles(
        self,
        tile_size: int = 32,
        halo: int = 4,
        anchor_ids: Optional[Sequence[int]] = None,
    ) -> Iterator[GridTile]:
        """Yield deterministic emitted-cell tiles with an emitted halo."""
        if tile_size < 1 or halo < 0:
            raise ValueError("tile_size must be positive and halo non-negative")
        anchor = self.grid_ids if anchor_ids is None else np.asarray(anchor_ids, dtype=np.int64)
        core_mask = self.contains(anchor)
        anchor = np.unique(anchor[core_mask])
        anchor_indices = np.asarray(self.indices(anchor), dtype=np.int64)
        anchor_rows = np.asarray(self.rows)[anchor_indices]
        anchor_cols = np.asarray(self.cols)[anchor_indices]
        keys = np.column_stack((anchor_rows // tile_size, anchor_cols // tile_size))
        for tile_row, tile_col in sorted(set(map(tuple, keys.tolist()))):
            r0, c0 = tile_row * tile_size, tile_col * tile_size
            core = (
                (anchor_rows >= r0) & (anchor_rows < r0 + tile_size)
                & (anchor_cols >= c0) & (anchor_cols < c0 + tile_size)
            )
            context = (
                (np.asarray(self.rows) >= r0 - halo)
                & (np.asarray(self.rows) < r0 + tile_size + halo)
                & (np.asarray(self.cols) >= c0 - halo)
                & (np.asarray(self.cols) < c0 + tile_size + halo)
            )
            context_indices = np.flatnonzero(context).astype(np.int64)
            core_indices = anchor_indices[core]
            yield GridTile(
                core_indices=core_indices,
                context_indices=context_indices,
                core_ids=self.grid_ids[core_indices],
                context_ids=self.grid_ids[context_indices],
                relative_rows=np.asarray(self.rows)[context_indices] - (r0 - halo),
                relative_cols=np.asarray(self.cols)[context_indices] - (c0 - halo),
            )


class SparseLatentTable:
    """A latent matrix explicitly keyed by raw int64 grid IDs."""

    def __init__(self, grid_ids, embeddings):
        grid_ids = np.asarray(grid_ids, dtype=np.int64)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or len(grid_ids) != len(embeddings):
            raise ValueError("Sparse latent table requires ids[N] and embeddings[N,D]")
        order = np.argsort(grid_ids, kind="stable")
        self.grid_ids = grid_ids[order]
        self.embeddings = embeddings[order]
        if len(np.unique(self.grid_ids)) != len(self.grid_ids):
            raise ValueError("Sparse latent table contains duplicate IDs")

    @classmethod
    def load(cls, path: str | Path, static_grid: Optional[StaticGrid] = None):
        path = Path(path)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                ids = np.asarray(archive["grid_ids"], dtype=np.int64)
                key = "embeddings" if "embeddings" in archive.files else "latents"
                embeddings = np.asarray(archive[key], dtype=np.float32)
        else:
            embeddings = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            if static_grid is None:
                raise ValueError("Legacy .npy latents require a StaticGrid row binding")
            ids = static_grid.grid_ids
        table = cls(ids, embeddings)
        if static_grid is not None and not np.array_equal(table.grid_ids, static_grid.grid_ids):
            raise RuntimeError("Latent IDs do not exactly match the emitted static grid")
        return table

    def lookup(self, grid_ids):
        values, scalar = _as_array(grid_ids, np.int64)
        flat = values.ravel()
        positions = np.searchsorted(self.grid_ids, flat)
        safe = np.clip(positions, 0, max(0, len(self.grid_ids) - 1))
        found = (positions < len(self.grid_ids)) & (self.grid_ids[safe] == flat)
        result = np.zeros(values.shape + (self.embeddings.shape[1],), dtype=np.float32)
        result.reshape(-1, self.embeddings.shape[1])[found] = self.embeddings[positions[found]]
        return (result.reshape(-1), bool(found[0])) if scalar else (result, found.reshape(values.shape))


def load_active_grid_ids(path: str | Path, min_checkins: int = 1) -> np.ndarray:
    ids = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "grid_id" not in (reader.fieldnames or ()):
            raise RuntimeError("Active-grid CSV must contain grid_id")
        count_column = next(
            (name for name in ("checkin_count", "n_checkins", "count")
             if name in (reader.fieldnames or ())), None,
        )
        for row in reader:
            count = int(row[count_column]) if count_column is not None else 1
            if count >= int(min_checkins):
                ids.append(int(row["grid_id"]))
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def write_active_grid_ids(
    path: str | Path,
    grid_ids,
    min_checkins: int,
    counts: Optional[Mapping[int, int]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("grid_id", "checkin_count", "min_checkins"))
        writer.writeheader()
        for grid_id in sorted(set(map(int, grid_ids))):
            writer.writerow({
                "grid_id": grid_id,
                "checkin_count": int((counts or {}).get(grid_id, min_checkins)),
                "min_checkins": int(min_checkins),
            })


def verify_candidate_release(
    city: str,
    csv_path: str | Path,
    grid_spec_path: str | Path = DEFAULT_GRID_SPEC,
    release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> Mapping[str, object]:
    city_key = canonical_city(city)
    with Path(release_manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        receipt = manifest["primary_files"][city_key]
    except KeyError as exc:
        raise RuntimeError("Release manifest has no %s receipt" % city_key) from exc
    actual_csv = sha256_file(csv_path)
    if actual_csv != receipt["sha256"]:
        raise RuntimeError(
            "%s static CSV hash mismatch: expected %s, got %s"
            % (city_key, receipt["sha256"], actual_csv)
        )
    expected_spec = manifest.get("supporting_files", {}).get("dataset/grid_spec.json")
    actual_spec = sha256_file(grid_spec_path)
    if expected_spec is not None and actual_spec != expected_spec:
        raise RuntimeError("grid_spec.json hash does not match candidate release")
    return {
        "release_tag": manifest.get("release_tag"),
        "data_commit": manifest.get("data_commit"),
        "city": city_key,
        "csv_sha256": actual_csv,
        "grid_spec_sha256": actual_spec,
        "rows": int(receipt["data_rows"]),
    }


def verify_all_candidate_files(
    repository_root: str | Path,
    release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> Mapping[str, Mapping[str, object]]:
    repository_root = Path(repository_root)
    with Path(release_manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        city: verify_candidate_release(
            city,
            repository_root / receipt["path"],
            repository_root / "dataset" / "grid_spec.json",
            release_manifest_path,
        )
        for city, receipt in manifest["primary_files"].items()
    }
