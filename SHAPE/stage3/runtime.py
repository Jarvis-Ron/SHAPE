"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import OrderedDict
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from SHAPE.gridspec import StaticGrid
from SHAPE.stage2.config import TOKENIZER_TRAINING_CONTRACT
from SHAPE.stage3.contracts import require_current_tokenizer_training
from SHAPE.stage3.tasks import FIXED_VARIANT
from SHAPE.stage3.group_index import load_v4_transport_provenance, repair_group_index
from SHAPE.stage3.landing_head import load_canonical_grid_attributes, load_landing_runtime
from SHAPE.stage3.sampler import _distance_row_km, attraction_distribution
from SHAPE.stage3.source_semantics import _load_codes_artifact, derive_projected_train_homes
from SHAPE.stage3.prototype_semantics import ATTRACTION_SMOOTHING, cohort_split_view, fit_train_code_attraction_semantics
from SHAPE.stage3.seeds import SELECTION_SEEDS
from SHAPE.stage3.resident_index import _broadcast_user_values, _unique_user_values
from SHAPE.stage3.rank_semantics import _target_home_probability, fit_train_rank_distance_semantics, RANK_PIECEWISE_EDGES
from SHAPE.stage3.opportunity_rank import SpatialRankIndex, static_poi_weights, fit_personal_profiles
from SHAPE.stage3.context_memory import _exact_nearest_indices, _percentile_ranks, context_quota_home_assignment, derive_familiar_location_capacity, map_marginal_rescaled_attraction, validate_interleaved_split_chronology
from SHAPE.stage3.static_features import _load_raw_table
from SHAPE.stage3.grid_projection import _project_nominal_ids_to_vocabulary
from SHAPE.stage3.reference import target_train_homes


def exact_nearest_rows(centers, origin, count, latitude_order=None):
    """Same neighbors/ties as E2, using the exact north-distance lower bound.

    A small latitude window gives an upper bound on the kth distance.  Every
    point with |latitude delta| <= that radius/111.32 is then scored with the
    original distance function; no approximate nearest-neighbor search.
    """
    centers = np.asarray(centers, dtype=np.float64)
    count = min(int(count), len(centers))
    if count < 1:
        raise ValueError("nearest-grid count must be positive")
    order = (np.argsort(centers[:, 0], kind="stable") if latitude_order is None
             else np.asarray(latitude_order))
    latitudes = centers[order, 0]
    index = int(np.searchsorted(latitudes, centers[origin, 0]))
    initial = order[max(0, index - count):min(len(order), index + count)]
    distances = _distance_row_km(centers[origin], centers[initial])
    radius = float(np.partition(distances, count - 1)[count - 1])
    delta = (radius + 1e-9) / 111.32
    lower = np.searchsorted(latitudes, centers[origin, 0] - delta, side="left")
    upper = np.searchsorted(latitudes, centers[origin, 0] + delta, side="right")
    candidates = np.sort(order[lower:upper])
    distance = _distance_row_km(centers[origin], centers[candidates])
    return candidates[_exact_nearest_indices(distance, count)]


def selected_home_context(raw, group, centers, required_positions, neighbors=25):
    """Exact E2 contexts only where Hamilton assignment can inspect them.

    Residential percentiles still use the entire static vocabulary.  Other
    POI rows remain zero and are never supplied as eligible home candidates.
    """
    raw = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    centers = np.asarray(centers, dtype=np.float64)
    _, residential = attraction_distribution(np.asarray(group, dtype=np.float64),
                                              include_transit=False)
    context = np.zeros((len(raw), 14), dtype=np.float32)
    context[:, 0] = _percentile_ranks(residential)
    required = np.unique(np.asarray(required_positions, dtype=np.int64))
    if np.any((required < 0) | (required >= len(raw))):
        raise ValueError("home context request is outside static rows")
    order = np.argsort(centers[:, 0], kind="stable")
    for origin in required:
        nearest = exact_nearest_rows(centers, origin, neighbors, order)
        mass = raw[nearest, :13].sum(axis=0) + ATTRACTION_SMOOTHING
        context[origin, 1:] = mass / mass.sum()
    return context, {
        "contract": "E2_static_context_exact_queried_rows_v1",
        "vocabulary_rows": len(raw), "computed_POI_rows": len(required),
        "neighborhood_size": int(neighbors),
        "percentile_population": "all_emitted_static_grids",
        "nearest_algorithm": "exact_latitude_lower_bound_then_E2_tie_repair",
        "computed_row_indices_sha256": hashlib.sha256(required.tobytes()).hexdigest(),
        "computed_context_sha256": hashlib.sha256(context[required].tobytes()).hexdigest(),
        "uses_trajectory_for_context_values": False,
    }


def build_source_payloads(cohort):
    """Keep the original observed-code train/test payload shape for X1."""
    payloads = OrderedDict()
    for split in ("train", "test"):
        view = cohort_split_view(cohort, split)
        events = view["event_indices"]
        key = f"source_{split}_real"
        payloads[key] = {
            "key": key, "cohort_name": "source", "split": split,
            "view": view, "lengths": np.diff(view["offsets"]),
            "features": np.asarray(cohort["real_features"][events], dtype=np.float32),
            "codes": np.asarray(cohort["real_codes"][events], dtype=np.int64),
            "generator_seed": None, "mixtures": {},
        }
    return payloads


def quota_probability(quota, vocabulary):
    """Accept a population histogram, never individual target trajectories."""
    if set(quota) != {"schema", "grid_ids", "home_counts", "provenance"} \
            or quota["schema"] != "target_train_home_quota_v1":
        raise ValueError("aggregate home quota schema changed")
    ids = np.asarray(quota["grid_ids"], dtype=np.int64)
    counts = np.asarray(quota["home_counts"], dtype=np.float64)
    if not np.array_equal(ids, vocabulary) or counts.shape != ids.shape \
            or not np.isfinite(counts).all() or np.any(counts < 0) \
            or np.any(counts != np.floor(counts)) or counts.sum() <= 0:
        raise ValueError("invalid aggregate home quota")
    return counts / counts.sum()


def prepare_transfer_runtime(source, closeout, source_cohort, target_reference,
                             target_cohort, target_grid_csv, devices, *,
                             verify_release=True, progress=lambda text: None,
                             source_city="nyc", target_city="ca", personal_scale=None,
                             rank_cache_rows=4096, sampling_seeds=SELECTION_SEEDS,
                             target_home_quota=None):
    if target_home_quota is not None and (
            target_reference is not None or target_cohort is not None):
        raise ValueError("quota-only generation rejects target checkins/cohort and distance calibration")
    sampling_seeds = tuple(sampling_seeds)
    if not sampling_seeds or len(set(sampling_seeds)) != len(sampling_seeds) \
            or any(type(seed) is not int or seed < 0 for seed in sampling_seeds):
        raise ValueError("sampling seeds must be distinct nonnegative integers")
    personal_scale = dict(FIXED_VARIANT) if personal_scale is None else personal_scale
    if personal_scale != FIXED_VARIANT:
        raise ValueError("Only opportunity ranks without distance correction are supported")
    source_reference = _load_codes_artifact(source["reference_codes"])
    require_current_tokenizer_training(source_reference)
    progress(f"loading frozen {source_city} landing head and full {target_city} static vocabulary")
    source_static = StaticGrid.load(source["grid_csv"], city=source_city,
                                    verify_release=verify_release)
    target_static = StaticGrid.load(target_grid_csv, city=target_city,
                                    verify_release=verify_release)
    loaded = load_landing_runtime(source["checkpoint"], None, target_grid_csv,
                                  torch.device(devices[0]), vocabulary_policy="all_emitted")
    model, target_tensor, _, target_centers, target_ids, target_binding, state = loaded
    if state.get("landing_mode") != "l2_attributes" \
            or str(state.get("source_city", "")).lower() != source_city \
            or state.get("tokenizer_sha256") != source_cohort["tokenizer_sha256"] \
            or state.get("codes_sha256") != source_cohort["reference_codes_sha256"]:
        raise RuntimeError("source frozen landing/tokenizer/reference binding changed")
    _, source_centers, source_ids, _ = load_canonical_grid_attributes(source["grid_csv"])
    fitted_source_ids = np.asarray(state.get("source_grid_ids"), dtype=np.int64)
    if not np.array_equal(fitted_source_ids, source_ids):
        # Historical CA heads use a source-training vocabulary. The new rank
        # geometry includes the complete static map without changing the head.
        if source_city != "ca" or not len(fitted_source_ids) \
                or not np.isin(fitted_source_ids, source_ids).all():
            raise RuntimeError("frozen landing source grid vocabulary is incompatible")
    if len(target_ids) != len(target_static) or not np.array_equal(target_ids, target_static.grid_ids):
        raise RuntimeError("target landing did not retain the full emitted static vocabulary")
    source_raw, _ = _load_raw_table(source["grid_csv"])
    target_raw, _ = _load_raw_table(target_grid_csv)
    source_group, _, source_repair = repair_group_index(source_raw, source_city,
        transport_provenance=load_v4_transport_provenance(source["grid_csv"], source_ids))
    target_group, _, target_repair = repair_group_index(target_raw, target_city,
        transport_provenance=load_v4_transport_provenance(target_grid_csv, target_ids))
    source_projected, source_projection = _project_nominal_ids_to_vocabulary(
        source_reference["grid_id"], source_ids, source_static.spec)
    if target_home_quota is None:
        target_projected, target_projection = _project_nominal_ids_to_vocabulary(
            target_reference["grid_id"], target_ids, target_static.spec)
        if not np.all(np.asarray(target_reference["event_splits"]) == "train"):
            raise ValueError("rank runtime accepts target train homes only; no target test locations")
    else:
        target_projection = {"computed": False, "input": "aggregate_home_quota_only"}
    source_homes, source_home_audit = derive_projected_train_homes(
        source_reference, source_projected, source_cohort["users"])
    source_users, unique_source_homes = _unique_user_values(source_cohort["users"], source_homes)
    if target_home_quota is None:
        target_homes, target_home_audit = target_train_homes(
            target_reference, target_projected, target_cohort["users"])
        target_home_audit["fit_split"] = target_city + "_day_split_train_only"
        _, unique_target_homes = _unique_user_values(target_cohort["users"], target_homes)
        probability = _target_home_probability(unique_target_homes, target_ids)
    else:
        probability = quota_probability(target_home_quota, target_ids)
        target_home_audit = {
            "contract": target_home_quota["schema"],
            "fit_split": target_city + "_day_split_train_only",
            "input": "aggregate_home_counts",
            "generation_received_target_person_ids": False,
            "generation_received_target_checkins": False,
        }
    progress("building exact 25-neighbor H contexts and Hamilton home quotas")
    source_context, source_context_audit = selected_home_context(
        source_raw, source_group, source_centers, np.searchsorted(source_ids, unique_source_homes))
    target_context, target_context_audit = selected_home_context(
        target_raw, target_group, target_centers, np.flatnonzero(probability > 0))
    assignments, home_audits = OrderedDict(), OrderedDict()
    for seed in sampling_seeds:
        homes, audit = context_quota_home_assignment(
            unique_source_homes, source_context, target_context, probability,
            seed, source_ids, target_ids)
        assignments[seed] = _broadcast_user_values(source_cohort["users"], source_users, homes)
        home_audits[seed] = {**audit, "target_city": target_city,
                            "home_quota_source": target_city + "_day_split_train_only"}

    progress("fitting source-train opportunity ranks; no physical distance correction")
    source_weights, source_weight_audit = static_poi_weights(source["grid_csv"], source_ids)
    target_weights, target_weight_audit = static_poi_weights(target_grid_csv, target_ids)
    source_rank = SpatialRankIndex(source_centers, source_weights)
    target_rank = SpatialRankIndex(target_centers, target_weights, cache_rows=rank_cache_rows)
    source_piecewise = fit_train_rank_distance_semantics(
        source_reference, source_projected, source_centers, source_ids,
        weights=source_weights, rank_index=source_rank)
    profiles = fit_personal_profiles(source_reference, source_projected,
        source_centers, source_ids, rank_index=source_rank)
    edges = RANK_PIECEWISE_EDGES
    weights_audit = {"source": source_weight_audit, "target": target_weight_audit,
                     "weighted_rank": True}
    source_roles = fit_train_code_attraction_semantics(
        source_reference, source_projected, source_raw, source_ids,
        residential_grid_table=source_group)
    attraction, attraction_audit = map_marginal_rescaled_attraction(source_roles, target_raw, target_group)
    capacity, capacity_audit = derive_familiar_location_capacity(
        source_reference, source_projected, source_cohort["users"])
    chronology = validate_interleaved_split_chronology(source_cohort)
    parameters = dict(source["arms"]["D1A1_raw"]["parameters"])
    for key in ("distance_tau", "selection_temperature", "attraction_strength"):
        if key not in parameters or not np.isfinite(float(parameters[key])):
            raise RuntimeError("missing frozen source parameter: " + key)
    if any(parameters.get(key) != value for key, value in (
        ("distance_mode", "mixture"), ("attraction_mode", "raw"), ("selection_mode", "sample")
    )):
        raise RuntimeError("P3-B2 D1A1_raw winner semantics changed")
    lanes = []
    for index, name in enumerate(devices):
        device = torch.device(name)
        if index:
            from copy import deepcopy
            lane_model = deepcopy(model).to(device).eval()
        else:
            lane_model = model
        lanes.append({
            "model": lane_model, "device": device,
            "target": F.normalize(target_tensor.detach().to(device), dim=-1),
            "rank": target_rank, "encoded_query_cache": {},
            "encoded_query_cache_stats": {"hits": 0, "misses": 0},
        })
    return {
        "target_centers": target_centers, "target_grid_ids": target_ids,
        "target_static": target_static, "target_raw": target_raw,
        "target_group": target_group, "target_l2": target_tensor.detach().cpu().numpy(),
        "normalization": state["normalization"], "lanes": lanes,
        "selected_beta": closeout["selected_beta"], "base_parameters": parameters,
        "source_context_homes": assignments, "source_familiar_capacity": capacity,
        "transferred_attraction_rescaled": attraction,
        "personal_profiles": profiles, "personal_scale": personal_scale,
        "scale_specs": {"rank": {
            "semantics": source_piecewise, "coordinate": "spatial_rank",
            "coarse_edges": edges[[0, 4, 8, 12, 16]], "piecewise_edges": edges,
            "contract": "source_train_opportunity_rank_v1",
        }},
        "provenance": {
            "sampling_seeds": sampling_seeds,
            "source_binding": source["binding"], "closeout_binding": closeout["binding"],
            "source_generation": source_cohort["generated_binding"],
            "source_home": source_home_audit, "target_home": target_home_audit,
            "source_static_context": source_context_audit, "target_static_context": target_context_audit,
            "H_home_assignment": home_audits, "R_capacity": capacity_audit,
            "R_chronology": chronology, "A_attraction": attraction_audit,
            "target_distance_calibration": False,
            "physical_distance_cost": False,
            "tokenizer_training_contract": TOKENIZER_TRAINING_CONTRACT,
            "opportunity_weights": weights_audit,
            "source_distance_semantics": source_piecewise,
            "source_fitted_grid_rows": len(fitted_source_ids),
            "source_rank_grid_rows": len(source_ids),
            "target_grid_binding": target_binding, "normalization": state["normalization"],
            "projections": {"source": source_projection, "target": target_projection},
            "group_repairs": {source_city: source_repair, target_city: target_repair},
            "target_home_probability": probability,
            "target_rank_matrix_allocated": False,
            "target_rank_cache_rows": target_rank.cache_rows,
            "target_true_codes_or_reconstructed_H_used": False,
            "target_train_statistics": ["population_home_marginal"],
            "target_test_locations_read_by_generation": False,
            "target_train_distance_quantiles_used_by_distance_scale": False,
            "target_aggregate_quota_only": target_home_quota is not None,
        },
    }
