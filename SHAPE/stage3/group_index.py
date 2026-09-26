"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from SHAPE.stage3.contracts import sha256_file


GROUP_INDEX_CONTRACT = (
    "hellinger_poi13+city_rank_density+city_rank_v4_transport6"
)


GROUP_INDEX_STAGE_CONTRACTS = {
    1: "city_rank_density_and_raw_transport7+poi_share13",
    2: "stage1+hellinger_poi13",
    3: "stage2+CA_imputation_retired_missing_invalid",
    4: GROUP_INDEX_CONTRACT,
}


def _rank01(values):
    """Stable average-tie empirical ranks in [0, 1], without SciPy."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return (ranks / float(len(values) - 1)).astype(np.float32)


def _v4_transport(values):
    """Select density plus the six released-v4 transport semantics.

    Keeping this adapter shared by the static table and query path prevents a
    silent train/inference contract split.  The legacy column labels are kept
    in the source CSV for compatibility, but no cross-semantic arithmetic is
    permitted here.
    """
    values = np.asarray(values)
    return values[:, 13:20]


def _safe_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(_rank01(left), _rank01(right))[0, 1])


def _paired_support_audit(left, right, valid_rows):
    left = np.asarray(left, dtype=np.float64)[valid_rows]
    right = np.asarray(right, dtype=np.float64)[valid_rows]
    left_zero = np.isclose(left, 0.0)
    right_zero = np.isclose(right, 0.0)
    both_nonzero = ~left_zero & ~right_zero
    return {
        "zero_support_exact_match": bool(np.array_equal(left_zero, right_zero)),
        "zero_support_mismatch_rate": float(np.mean(left_zero != right_zero))
        if len(left) else None,
        "joint_nonzero_rows": int(both_nonzero.sum()),
        "joint_nonzero_spearman": _safe_spearman(
            left[both_nonzero], right[both_nonzero]
        ),
    }


def load_v4_transport_provenance(grid_csv, grid_ids=None):
    """Load released-v4 distance provenance aligned to requested grid IDs.

    The ranked merged columns cannot by themselves prove that the source join
    was correct.  Their pinned provenance file can: the two proximity columns
    must preserve the order of negative nearest-station distance exactly.
    """
    grid_path = Path(grid_csv)
    suffix = "_grid_merged_features.csv"
    if not grid_path.name.endswith(suffix):
        return None
    provenance_path = grid_path.with_name(
        grid_path.name[:-len(suffix)] + "_grid_provenance.csv"
    )
    if not provenance_path.is_file():
        return None
    with provenance_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"grid_id", "nearest_station_m", "nearest_rail_m"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(
            "v4 transport provenance lacks required columns: %s"
            % provenance_path
        )
    by_grid = {}
    for row in rows:
        grid_id = int(row["grid_id"])
        if grid_id in by_grid:
            raise RuntimeError(
                "Duplicate grid_id in v4 transport provenance: %s"
                % provenance_path
            )
        try:
            station = float(row["nearest_station_m"])
            rail = float(row["nearest_rail_m"])
        except (TypeError, ValueError):
            station, rail = float("nan"), float("nan")
        by_grid[grid_id] = (station, rail)
    requested = (
        np.asarray(sorted(by_grid), dtype=np.int64)
        if grid_ids is None else np.asarray(grid_ids, dtype=np.int64)
    )
    missing = [int(grid_id) for grid_id in requested if int(grid_id) not in by_grid]
    if missing:
        raise RuntimeError(
            "v4 transport provenance is missing %d requested grid IDs: %s"
            % (len(missing), provenance_path)
        )
    values = np.asarray([by_grid[int(grid_id)] for grid_id in requested])
    return {
        "path": str(provenance_path),
        "sha256": sha256_file(provenance_path),
        "grid_ids": requested,
        "nearest_station_m": values[:, 0],
        "nearest_rail_m": values[:, 1],
    }


def _v4_transport_audit(raw, originally_empty, provenance=None):
    """Audit v4 semantic consistency without treating sparsity as a bad join."""
    valid_rows = ~np.asarray(originally_empty, dtype=bool)
    total_proximity = raw[:, 14]
    rail_proximity = raw[:, 15]
    all_invalid = (
        ~np.isfinite(total_proximity)
        | (total_proximity <= 0.0)
        | (total_proximity > 1.0 + 1e-9)
    )
    rail_invalid = (
        ~np.isfinite(rail_proximity)
        | (rail_proximity <= 0.0)
        | (rail_proximity > 1.0 + 1e-9)
    )
    rail_service = raw[:, 18] > 0.0
    rail_share = raw[:, 19] > 0.0

    def rate(mask):
        selected = np.asarray(mask, dtype=bool)[valid_rows]
        return float(np.mean(selected)) if len(selected) else None

    all_invalid_rate = rate(all_invalid)
    rail_invalid_rate = rate(rail_invalid)
    support_mismatch_rate = rate(rail_service != rail_share)
    column_contract_failure = not valid_rows.any() or any(
        value is not None and value > 0.0
        for value in (
            all_invalid_rate, rail_invalid_rate, support_mismatch_rate,
        )
    )
    provenance_report = {
        "available": provenance is not None,
        "status": "not_evaluable_without_pinned_distance_provenance",
        "path": None,
        "sha256": None,
        "nearest_station_missing_rate": None,
        "nearest_rail_missing_rate": None,
        "all_mode_rank_vs_negative_distance_spearman": None,
        "rail_rank_vs_negative_distance_spearman": None,
        "exact_order_preservation_pass": None,
    }
    join_failure = False
    if provenance is not None:
        nearest_station = np.asarray(
            provenance["nearest_station_m"], dtype=np.float64
        )
        nearest_rail = np.asarray(
            provenance["nearest_rail_m"], dtype=np.float64
        )
        if len(nearest_station) != len(raw) or len(nearest_rail) != len(raw):
            raise ValueError("v4 transport provenance must align with raw rows")
        station_missing = ~np.isfinite(nearest_station)
        rail_missing = ~np.isfinite(nearest_rail)
        station_rho = _safe_spearman(
            total_proximity[valid_rows & ~station_missing],
            -nearest_station[valid_rows & ~station_missing],
        )
        rail_rho = _safe_spearman(
            rail_proximity[valid_rows & ~rail_missing],
            -nearest_rail[valid_rows & ~rail_missing],
        )
        exact_order = (
            station_rho is not None and rail_rho is not None
            and abs(station_rho - 1.0) <= 1e-9
            and abs(rail_rho - 1.0) <= 1e-9
        )
        join_failure = bool(
            np.any(station_missing[valid_rows])
            or np.any(rail_missing[valid_rows])
            or not exact_order
        )
        provenance_report = {
            "available": True,
            "status": "pass" if not join_failure else "fail",
            "path": provenance.get("path"),
            "sha256": provenance.get("sha256"),
            "nearest_station_missing_rate": rate(station_missing),
            "nearest_rail_missing_rate": rate(rail_missing),
            "all_mode_rank_vs_negative_distance_spearman": station_rho,
            "rail_rank_vs_negative_distance_spearman": rail_rho,
            "exact_order_preservation_pass": bool(exact_order),
        }
    rail_service_zero_rate = rate(~rail_service)
    return {
        "contract": "v4_ranked_proximity_provenance_and_rail_service_support",
        "audited_nonempty_rows": int(valid_rows.sum()),
        "all_mode_proximity_invalid_rate": all_invalid_rate,
        "rail_proximity_invalid_rate": rail_invalid_rate,
        "all_mode_proximity_zero_rate_descriptive_only": rate(
            np.isclose(total_proximity, 0.0)
        ),
        "rail_proximity_zero_rate_descriptive_only": rate(
            np.isclose(rail_proximity, 0.0)
        ),
        "rail_service_zero_rate": rail_service_zero_rate,
        "rail_share_zero_rate": rate(~rail_share),
        "rail_service_share_support_mismatch_rate": support_mismatch_rate,
        "distance_provenance_audit": provenance_report,
        "suspicious_join_or_column_contract_failure": bool(
            column_contract_failure or join_failure
        ),
        "sparse_rail_service_warning": bool(
            rail_service_zero_rate is not None and rail_service_zero_rate > 0.80
        ),
        "interpretation": (
            "ranked proximity zero rate is not a join test; pinned distance "
            "provenance validates order preservation, while rail service zero "
            "rate measures coverage sparsity"
        ),
        "action": "validate_v4_semantics_source_table_not_mutated",
    }


def _poi_diagnostics(poi):
    poi = np.asarray(poi, dtype=np.float64)
    totals = poi.sum(axis=1, keepdims=True)
    shares = np.divide(poi, totals, out=np.zeros_like(poi), where=totals > 0)
    nonempty = totals[:, 0] > 0
    max_share = shares.max(axis=1)
    herfindahl = np.square(shares).sum(axis=1)
    return shares, {
        "nonempty_grids": int(nonempty.sum()),
        "empty_grids": int((~nonempty).sum()),
        "max_p_mean": float(max_share[nonempty].mean()) if nonempty.any() else 0.0,
        "max_p_median": float(np.median(max_share[nonempty])) if nonempty.any() else 0.0,
        "herfindahl_mean": float(herfindahl[nonempty].mean()) if nonempty.any() else 0.0,
        "herfindahl_median": float(np.median(herfindahl[nonempty])) if nonempty.any() else 0.0,
    }


def repair_group_index(
        raw_table, city, through_step=4, drop_metro_frequency=False,
        transport_provenance=None):
    """Return an emitted-N x 20 ablation table and invalid-row mask."""
    raw = np.asarray(raw_table, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 20 or len(raw) == 0:
        raise ValueError("group-index input must have shape [N, 20]")
    city = str(city).lower()
    if city not in ("ca", "nyc", "paris"):
        raise ValueError("city must be ca, nyc, or paris")
    through_step = int(through_step)
    if through_step not in GROUP_INDEX_STAGE_CONTRACTS:
        raise ValueError("group-index repair step must be 1, 2, 3, or 4")
    repaired_raw = raw.copy()
    imputed = np.zeros(len(raw), dtype=bool)
    originally_empty = np.isclose(np.abs(raw).sum(axis=1), 0.0)

    shares, poi_diagnostic = _poi_diagnostics(repaired_raw[:, :13])
    table = np.zeros((len(raw), 20), dtype=np.float32)
    table[:, :13] = (
        np.sqrt(shares) if through_step >= 2 else shares
    ).astype(np.float32)

    # Step 4 used to subtract bus columns from total columns.  That operation
    # was valid for the old count schema but is invalid under v4.  Direct v4
    # semantics are deliberately a numerical no-op relative to step 2/3; the
    # stage remains in the ablation graph to make retirement of the harmful
    # residual explicit and reproducible.
    transport = _v4_transport(repaired_raw)
    table[:, 13:] = np.column_stack([
        _rank01(transport[:, index]) for index in range(transport.shape[1])
    ])
    if drop_metro_frequency:
        # One-variable M0 validation.  Keep the frozen 20-D interface while
        # removing only the 99%+-zero metro-frequency signal.
        table[:, 18] = 0.0

    report = {
        "city": city,
        "contract": GROUP_INDEX_STAGE_CONTRACTS[through_step] + (
            "+metro_frequency_rank_removed"
            if drop_metro_frequency else ""
        ),
        "through_step": through_step,
        "shape": [int(len(raw)), 20],
        "poi_transform": (
            "sqrt(row_normalized_poi13)_hellinger"
            if through_step >= 2 else "row_normalized_poi13"
        ),
        "poi_diagnostic": poi_diagnostic,
        "scale_transform": "stable_average_tie_within_city_empirical_rank_0_1",
        "transport_columns": [
            "poi_density_rank", "all_mode_station_proximity_rank",
            "rail_station_proximity_rank", "departures_per_station_rank",
            "all_mode_departures_rank", "rail_departures_rank",
            "rail_share_rank",
        ],
        "transport_step4_semantic_repair": {
            "applied": bool(through_step >= 4),
            "legacy_total_minus_bus_residual": "retired",
            "reason": (
                "released_v4_total_stations_is_proximity_and_bus_stations_is_"
                "departures_per_station; total_frequency_is_absolute_and_"
                "bus_frequency_is_a_share"
            ),
            "transform": (
                "direct_v4_semantic_columns_then_city_rank"
                if through_step >= 4 else "not_yet_applied"
            ),
        },
        "transport_redundancy_audit": {
            "departures_per_station_vs_all_mode_departures":
                _paired_support_audit(raw[:, 16], raw[:, 17], ~originally_empty),
            "rail_departures_vs_rail_share":
                _paired_support_audit(raw[:, 18], raw[:, 19], ~originally_empty),
            "decision": "report_nonzero_subset_dependence_without_residualization",
        },
        "metro_join_audit": {
            **_v4_transport_audit(
                raw, originally_empty, transport_provenance
            ),
            "one_variable_removal_enabled": bool(drop_metro_frequency),
        },
        "ca_empty_grid_imputation": {
            "enabled": False,
            "original_empty_count": int(originally_empty.sum()),
            "imputed_count": int(imputed.sum()),
            "unresolved_empty_count": int(np.sum(originally_empty & ~imputed)),
            "neighbourhood": None,
            "contract": "retired_missing_rows_are_invalid_not_imputed",
        },
    }
    return table, imputed, report


def query_to_group_index(
        query, normalization, source_raw_table, through_step=4,
        drop_metro_frequency=False):
    """Parameter-free adapter from the frozen L2 output to group-index space."""
    if normalization is None:
        raise RuntimeError("group-index query calibration needs frozen L2 normalization")
    lower = np.asarray(normalization["lower"], dtype=np.float32)
    upper = np.asarray(normalization["upper"], dtype=np.float32)
    values = query * (upper - lower) + lower
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(values, dtype=np.float32)
    poi = np.maximum(values[:, :13], 0.0)
    totals = poi.sum(axis=1, keepdims=True)
    through_step = int(through_step)
    if through_step not in GROUP_INDEX_STAGE_CONTRACTS:
        raise ValueError("group-index query step must be 1, 2, 3, or 4")
    poi_share = np.divide(
        poi, totals, out=np.zeros_like(poi), where=totals > 0
    )
    result[:, :13] = np.sqrt(poi_share) if through_step >= 2 else poi_share
    source = np.asarray(source_raw_table, dtype=np.float32)
    # Empty source rows remain invalid; no CA neighbourhood imputation.
    # Keep query calibration exactly aligned with the static step-4 contract.
    # There is intentionally no total-minus-bus arithmetic under v4.
    source_transport = _v4_transport(source)
    query_transport = _v4_transport(values)
    for column in range(7):
        sorted_source = np.sort(source_transport[:, column])
        result[:, 13 + column] = np.searchsorted(
            sorted_source, query_transport[:, column], side="right"
        ) / float(len(sorted_source))
    if drop_metro_frequency:
        result[:, 18] = 0.0
    return result
