"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn.functional as F
from SHAPE.stage3.group_index import query_to_group_index
from SHAPE.stage3.basic_metrics import jensen_shannon_distribution


PERIODS = (
    ("morning_06_11", 6 * 60, 11 * 60),
    ("afternoon_11_17", 11 * 60, 17 * 60),
    ("evening_17_22", 17 * 60, 22 * 60),
    ("night_22_06", 22 * 60, 6 * 60),
)


SPATIAL_GRANULARITIES = OrderedDict((
    ("od_100x100", (10, 10)),
    ("od_50x50", (5, 10)),
    ("od_25x25", (5, 5)),
))


DEFAULT_OD_MASS_TOLERANCE = 0.05


OD_MASS_UNIT = "within_day_od_transition_count"


def _normalized(values):
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values)


def _entropy(values):
    probabilities = _normalized(values)
    keep = probabilities > 0
    return float(-np.sum(probabilities[keep] * np.log2(probabilities[keep])))


def attraction_distribution(group_table, include_transit=True):
    """Residential-attraction proxy available in all three static tables."""
    table = np.asarray(group_table, dtype=np.float64)
    poi = np.square(table[:, :13])
    poi /= poi.sum(axis=1, keepdims=True).clip(min=1e-12)
    keep = poi > 0
    richness = np.zeros(len(poi), dtype=np.float64)
    richness[:] = -np.sum(np.where(keep, poi * np.log(poi.clip(min=1e-12)), 0.0), axis=1)
    richness /= math.log(13.0)
    density = table[:, 13]
    if include_transit:
        transit = table[:, 14:].mean(axis=1)
        score = 0.50 * density + 0.30 * richness + 0.20 * transit
    else:
        # B-heat and B-gravity are preregistered POI-density/mix controls.
        score = 0.625 * density + 0.375 * richness
    score = np.maximum(score, 0.0) + 1e-6
    return score / score.sum(), score


@torch.inference_mode()
def _encode_landing_queries(model, features, device, batch_size=8192):
    """Encode landing queries once and keep the exact FP32 values on-device."""
    features = np.asarray(features, dtype=np.float32)
    if batch_size < 1:
        raise ValueError("landing query batch size must be positive")
    if len(features) == 0:
        return model.encode_query(torch.from_numpy(features).to(device))
    batches = []
    for batch_start in range(0, len(features), int(batch_size)):
        h = torch.from_numpy(
            features[batch_start:batch_start + int(batch_size)]
        ).to(device)
        batches.append(model.encode_query(h))
    return torch.cat(batches, dim=0)


@torch.inference_mode()
def group_autoregressive_landing(
    model, target_table, source_raw, normalization, features, offsets, homes,
    device, temperature=1.0, continuity_sigma=1.0, revisit_beta=0.5,
    group_index_step=None,
    drop_metro_frequency=False,
    target_grid_ids=None, centers=None,
    candidate_top_k=512, candidate_radius_multiplier=3.0,
    omitted_mass_tolerance=1e-4, omitted_mass_audit_steps=128,
    sequence_batch_size=256, return_audit=False,
    encoded_queries=None, normalized_target_tensor=None,
    prepared_center_tensor=None, neighborhood_cache=None,
    event_batch_size=512,
    code_target_distance_km=None, distance_kernel_tau=1.0,
    history_seed_grid_ids=None, distance_log_epsilon_km=0.05,
    code_distance_bin_probabilities=None,
    physical_distance_bin_edges_km=(0.0, 1.0, 5.0, 20.0, np.inf),
    code_distance_piecewise_probabilities=None,
    physical_distance_piecewise_edges_km=None,
    selection_mode="argmax", selection_seed=None,
    selection_temperature=1.0,
    code_attraction_table=None, event_attraction_code=None,
    attraction_mode=None, attraction_strength=0.0,
    attraction_epsilon=1e-8,
    hard_distance_ring_constraint=False,
    hard_distance_ring_seed=None,
    hard_distance_ring_empty_policy="fallback_unconstrained",
    distance_coordinate="physical_km",
    prepared_spatial_rank_tensor=None,
    event_code_keys=None, same_code_revisit_beta=0.0,
    same_code_key_layers="L0", same_code_ring_bias=0.0,
    personal_ring_multiplier=None, sequence_revisit_beta=None,
    initial_home_count=1.0, initial_anchor_count=1.0,
):
    if not all(np.isfinite(v) and v > 0 for v in (initial_home_count, initial_anchor_count)):
        raise ValueError("initial memory counts must be finite and positive")
    if temperature <= 0 or continuity_sigma <= 0:
        raise ValueError("landing temperature and continuity sigma must be positive")
    if normalized_target_tensor is None:
        target = torch.from_numpy(
            np.asarray(target_table, dtype=np.float32)
        ).to(device)
        target = F.normalize(target, dim=-1)
    else:
        target = normalized_target_tensor.to(device=device, dtype=torch.float32)
        if tuple(target.shape) != tuple(np.asarray(target_table).shape):
            raise ValueError("prepared target tensor shape changed")
    offsets = np.asarray(offsets, dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)
    homes = np.asarray(homes, dtype=np.int64)
    if sequence_revisit_beta is not None:
        sequence_revisit_beta = np.asarray(sequence_revisit_beta, dtype=np.float32)
        if sequence_revisit_beta.shape != (len(offsets)-1,) \
                or not np.isfinite(sequence_revisit_beta).all() \
                or np.any(sequence_revisit_beta < 0):
            raise ValueError("personal beta must be sequence-aligned and nonnegative")
    if personal_ring_multiplier is not None and not hard_distance_ring_constraint:
        raise ValueError("personal ring tilt requires a hard rank ring")
    same_code_revisit_beta = float(same_code_revisit_beta)
    same_code_ring_bias = float(same_code_ring_bias)
    if not np.isfinite(same_code_revisit_beta) \
            or same_code_revisit_beta < 0:
        raise ValueError("same-code revisit beta must be finite and non-negative")
    if not np.isfinite(same_code_ring_bias) or same_code_ring_bias < 0:
        raise ValueError("same-code ring bias must be finite and non-negative")
    same_code_key_layers = str(same_code_key_layers)
    if same_code_key_layers not in ("L0", "L01", "L012"):
        raise ValueError("same-code key layers must be L0, L01, or L012")
    same_code_keys = None
    if event_code_keys is not None:
        raw_code_keys = np.asarray(event_code_keys, dtype=np.int64)
        if raw_code_keys.ndim == 1:
            if raw_code_keys.shape != (len(features),):
                raise ValueError("flattened same-code keys must be event-aligned")
            same_code_keys = raw_code_keys.copy()
        elif raw_code_keys.ndim == 2 \
                and raw_code_keys.shape[0] == len(features) \
                and raw_code_keys.shape[1] >= {
                    "L0": 1, "L01": 2, "L012": 3,
                }[same_code_key_layers]:
            same_code_keys = raw_code_keys[:, 0].copy()
            if same_code_key_layers in ("L01", "L012"):
                same_code_keys = same_code_keys * 64 + raw_code_keys[:, 1]
            if same_code_key_layers == "L012":
                same_code_keys = same_code_keys * 64 + raw_code_keys[:, 2]
        else:
            raise ValueError(
                "same-code keys must be event-aligned flat keys or code rows"
            )
        if np.any(same_code_keys < 0):
            raise ValueError("same-code keys must be non-negative")
    elif same_code_revisit_beta > 0 or same_code_ring_bias > 0:
        raise ValueError("same-code beta/ring bias require event_code_keys")
    if selection_mode not in ("argmax", "sample"):
        raise ValueError("landing selection mode must be argmax or sample")
    if not np.isfinite(selection_temperature) or selection_temperature <= 0:
        raise ValueError("landing selection temperature must be finite and positive")
    if selection_mode == "sample" and selection_seed is None:
        raise ValueError("seeded sampling requires selection_seed")
    selection_uniform_tensor = None
    if selection_mode == "sample":
        selection_uniforms = np.random.default_rng(int(selection_seed)).random(
            len(features)
        )
        selection_uniform_tensor = torch.from_numpy(
            selection_uniforms.astype(np.float32)
        ).to(device)

    distance_coordinate = str(distance_coordinate)
    if distance_coordinate not in ("physical_km", "spatial_rank"):
        raise ValueError(
            "distance coordinate must be physical_km or spatial_rank"
        )
    mixture_distance = code_distance_bin_probabilities is not None
    point_distance = code_target_distance_km is not None
    if mixture_distance and point_distance:
        raise ValueError(
            "point and mixture code-distance semantics are mutually exclusive"
        )
    distance_aware = point_distance or mixture_distance
    if point_distance and distance_coordinate != "physical_km":
        raise ValueError(
            "point code-distance semantics require physical_km coordinates"
        )
    if distance_aware and (
            not np.isfinite(distance_kernel_tau) or distance_kernel_tau <= 0):
        raise ValueError("distance-kernel tau must be finite and positive")
    if point_distance:
        code_target_distance_km = np.asarray(
            code_target_distance_km, dtype=np.float32
        )
        if code_target_distance_km.shape != (len(features),) \
                or np.any(~np.isfinite(code_target_distance_km)) \
                or np.any(code_target_distance_km < 0):
            raise ValueError(
                "code-conditioned target distances must be finite, non-negative, "
                "and event-aligned"
            )
        if not np.isfinite(distance_log_epsilon_km) \
                or distance_log_epsilon_km <= 0:
            raise ValueError("distance log epsilon must be finite and positive")
        code_target_distance_tensor = torch.from_numpy(
            code_target_distance_km
        ).to(device)
    else:
        code_target_distance_tensor = None
    physical_edges = np.asarray(
        physical_distance_bin_edges_km, dtype=np.float64
    )
    if physical_edges.ndim != 1 or len(physical_edges) < 3 \
            or physical_edges[0] != 0.0 \
            or not np.isposinf(physical_edges[-1]) \
            or not np.all(np.isfinite(physical_edges[:-1])) \
            or np.any(np.diff(physical_edges[:-1]) <= 0):
        raise ValueError(
            "physical distance-bin edges must increase from zero to infinity"
        )
    distance_bin_count = len(physical_edges) - 1
    physical_edge_tensor = torch.as_tensor(
        physical_edges[1:-1], dtype=torch.float32, device=device
    )
    distance_probability_tensor = None
    hard_ring_probability_tensor = None
    hard_ring_target_bin_tensor = None
    hard_ring_uniform_tensor = None
    if mixture_distance:
        code_distance_bin_probabilities = np.asarray(
            code_distance_bin_probabilities, dtype=np.float32
        )
        if code_distance_bin_probabilities.shape != (
            len(features), distance_bin_count
        ) or np.any(~np.isfinite(code_distance_bin_probabilities)) \
                or np.any(code_distance_bin_probabilities < 0):
            raise ValueError(
                "code-distance mixtures must be finite, non-negative, "
                "event-aligned probability rows"
            )
        row_sums = code_distance_bin_probabilities.sum(axis=1)
        if np.any(row_sums <= 0) \
                or not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("code-distance mixture rows must sum to one")
        distance_probability_tensor = torch.from_numpy(
            code_distance_bin_probabilities
        ).to(device)
        hard_ring_probability = code_distance_bin_probabilities
        if personal_ring_multiplier is not None:
            from SHAPE.stage3.opportunity_rank import tilt_ring_probabilities
            hard_ring_probability = tilt_ring_probabilities(
                hard_ring_probability, personal_ring_multiplier)
        hard_ring_probability_tensor = torch.from_numpy(hard_ring_probability).to(device)
    hard_distance_ring_constraint = bool(hard_distance_ring_constraint)
    if hard_distance_ring_constraint:
        if not mixture_distance or distance_bin_count != 4:
            raise ValueError(
                "hard distance-ring constraint requires four-bin mixture semantics"
            )
        ring_seed = (
            selection_seed
            if hard_distance_ring_seed is None else hard_distance_ring_seed
        )
        if ring_seed is None:
            raise ValueError(
                "hard distance-ring constraint requires seeded sampling for "
                "target-ring draws"
            )
        if hard_distance_ring_empty_policy not in (
                "fallback_unconstrained", "expand_adjacent"):
            raise ValueError(
                "hard distance-ring empty policy must be fallback_unconstrained "
                "or expand_adjacent"
            )
        ring_rng = np.random.default_rng(np.random.SeedSequence((
            int(ring_seed), 0x50334232,
        )))
        ring_uniforms = ring_rng.random(len(features))
        ring_cdf = np.cumsum(
            hard_ring_probability.astype(np.float64), axis=1
        )
        hard_ring_target_bins = np.sum(
            ring_cdf < ring_uniforms[:, None], axis=1
        ).clip(max=distance_bin_count - 1).astype(np.int64)
        hard_ring_target_bin_tensor = torch.from_numpy(
            hard_ring_target_bins
        ).to(device)
        hard_ring_uniform_tensor = torch.from_numpy(ring_uniforms).to(
            device=device, dtype=torch.float64
        )
        hard_distance_ring_seed = int(ring_seed)
    elif hard_distance_ring_seed is not None \
            or hard_distance_ring_empty_policy != "fallback_unconstrained":
        raise ValueError(
            "hard distance-ring seed/policy require the hard-ring constraint"
        )
    if same_code_ring_bias > 0 and not hard_distance_ring_constraint:
        raise ValueError(
            "same-code ring bias requires the hard distance-ring constraint"
        )
    piecewise_distance = code_distance_piecewise_probabilities is not None
    piecewise_edges = None
    piecewise_edge_tensor = None
    if piecewise_distance:
        if not mixture_distance:
            raise ValueError(
                "piecewise distance density requires the four-bin mixture"
            )
        piecewise_edges = np.asarray(
            physical_distance_piecewise_edges_km, dtype=np.float64
        )
        code_distance_piecewise_probabilities = np.asarray(
            code_distance_piecewise_probabilities, dtype=np.float32
        )
        if piecewise_edges.ndim != 1 or len(piecewise_edges) < 3 \
                or piecewise_edges[0] != 0.0 \
                or not np.isposinf(piecewise_edges[-1]) \
                or not np.all(np.isfinite(piecewise_edges[:-1])) \
                or np.any(np.diff(piecewise_edges[:-1]) <= 0) \
                or code_distance_piecewise_probabilities.shape != (
                    len(features), len(piecewise_edges) - 1
                ) \
                or np.any(~np.isfinite(code_distance_piecewise_probabilities)) \
                or np.any(code_distance_piecewise_probabilities < 0):
            raise ValueError(
                "piecewise distance densities must be finite, non-negative, "
                "event-aligned probability rows with increasing edges"
            )
        piecewise_sums = code_distance_piecewise_probabilities.sum(axis=1)
        if not np.allclose(piecewise_sums, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("piecewise distance probability rows must sum to one")
        piecewise_edge_tensor = torch.as_tensor(
            piecewise_edges[1:-1], dtype=torch.float32, device=device
        )
        distance_probability_tensor = torch.from_numpy(
            code_distance_piecewise_probabilities
        ).to(device)
    attraction_enabled = code_attraction_table is not None
    attraction_rank_table = None
    attraction_value_tensor = None
    attraction_rank_tensor = None
    event_attraction_code_tensor = None
    if attraction_enabled:
        code_attraction_table = np.asarray(code_attraction_table, dtype=np.float32)
        event_attraction_code = np.asarray(event_attraction_code, dtype=np.int64)
        if code_attraction_table.ndim != 2 \
                or code_attraction_table.shape[1] != len(target_table) \
                or event_attraction_code.shape != (len(features),):
            raise ValueError(
                "code attraction must be [code, vocabulary] with event-aligned codes"
            )
        if np.any(~np.isfinite(code_attraction_table)) \
                or np.any(code_attraction_table < 0) \
                or np.any(event_attraction_code < 0) \
                or np.any(event_attraction_code >= len(code_attraction_table)):
            raise ValueError("code attraction contains invalid values or code ids")
        if attraction_mode not in ("raw", "rank"):
            raise ValueError("attraction mode must be raw or rank when enabled")
        if not np.isfinite(attraction_strength) or attraction_strength <= 0:
            raise ValueError("attraction strength must be finite and positive")
        if not np.isfinite(attraction_epsilon) or attraction_epsilon <= 0:
            raise ValueError("attraction epsilon must be finite and positive")
        if attraction_mode == "rank":
            attraction_rank_table = np.empty_like(code_attraction_table)
            grid_order = np.arange(code_attraction_table.shape[1], dtype=np.int64)
            for code_index, values in enumerate(code_attraction_table):
                order = np.lexsort((grid_order, -values.astype(np.float64)))
                ranks = np.empty(len(values), dtype=np.float32)
                ranks[order] = np.arange(1, len(values) + 1, dtype=np.float32)
                attraction_rank_table[code_index] = ranks
        attraction_value_tensor = torch.from_numpy(code_attraction_table).to(device)
        if attraction_rank_table is not None:
            attraction_rank_tensor = torch.from_numpy(attraction_rank_table).to(device)
        event_attraction_code_tensor = torch.from_numpy(
            event_attraction_code
        ).to(device)
    elif attraction_mode is not None or event_attraction_code is not None:
        raise ValueError("attraction mode/event codes require a code attraction table")
    vocabulary = (
        np.arange(len(target_table), dtype=np.int64)
        if target_grid_ids is None else np.asarray(target_grid_ids, dtype=np.int64)
    )
    sequence_count = len(offsets) - 1
    if history_seed_grid_ids is not None:
        history_seed_grid_ids = np.asarray(history_seed_grid_ids, dtype=np.int64)
        if history_seed_grid_ids.ndim != 2 \
                or history_seed_grid_ids.shape[0] != sequence_count:
            raise ValueError(
                "history seed grids must be a padded [sequence, K] array"
            )
    if centers is None:
        fixture_width = max(1, int(np.ceil(np.sqrt(len(vocabulary)))))
        ordinal = np.arange(len(vocabulary), dtype=np.int64)
        centers = np.column_stack(
            (ordinal // fixture_width, ordinal % fixture_width)
        ).astype(np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if prepared_center_tensor is None:
        center_tensor = torch.from_numpy(centers).to(device)
    else:
        center_tensor = prepared_center_tensor.to(
            device=device, dtype=torch.float32
        )
        if tuple(center_tensor.shape) != tuple(centers.shape):
            raise ValueError("prepared center tensor shape changed")
    spatial_rank_tensor = None
    if distance_coordinate == "spatial_rank":
        if not mixture_distance:
            raise ValueError(
                "spatial-rank distance semantics require a code mixture"
            )
        if prepared_spatial_rank_tensor is None:
            spatial_rank_tensor = torch.from_numpy(
                spatial_rank_table(centers)
            ).to(device)
        elif not torch.is_tensor(prepared_spatial_rank_tensor):
            from SHAPE.stage3.opportunity_rank import SpatialRankIndex
            if not isinstance(prepared_spatial_rank_tensor, SpatialRankIndex) \
                    or prepared_spatial_rank_tensor.shape != (len(vocabulary), len(vocabulary)):
                raise ValueError("prepared lazy spatial-rank index shape changed")
            spatial_rank_tensor = prepared_spatial_rank_tensor
        else:
            spatial_rank_tensor = prepared_spatial_rank_tensor.to(
                device=device, dtype=torch.float32
            )
            if tuple(spatial_rank_tensor.shape) != (
                    len(vocabulary), len(vocabulary)):
                raise ValueError("prepared spatial-rank tensor shape changed")
    predicted = np.empty(len(features), dtype=np.int64)
    if candidate_top_k < 1 or candidate_radius_multiplier <= 0 \
            or not 0 <= omitted_mass_tolerance < 1 \
            or omitted_mass_audit_steps < 0 or sequence_batch_size < 1 \
            or event_batch_size < 1:
        raise ValueError("candidate truncation options are out of range")
    # Encode once and retain the exact FP32 values on the accelerator.  G0 can
    # pass the same real-code tensor to L1 and every L2 arm, eliminating five
    # redundant encoder passes and all associated D2H/H2D round trips.
    raw_queries = (
        _encode_landing_queries(model, features, device)
        if encoded_queries is None else encoded_queries
    )
    if len(raw_queries) != len(features):
        raise ValueError("prepared landing queries are not event-aligned")
    if group_index_step is None:
        repaired_queries = raw_queries
    else:
        query_values = (
            raw_queries.detach().cpu().numpy()
            if torch.is_tensor(raw_queries)
            else np.asarray(raw_queries, dtype=np.float32)
        )
        repaired_queries = query_to_group_index(
            query_values, normalization, source_raw, group_index_step,
            drop_metro_frequency=drop_metro_frequency,
        )
    # B-2: a latitude-sorted physical index provides exact 3-sigma
    # neighbourhoods without materialising an O(|V|^2) distance matrix.
    latitude_order = np.argsort(centers[:, 0], kind="stable")
    sorted_latitudes = centers[latitude_order, 0]
    radius_km = float(candidate_radius_multiplier) * float(continuity_sigma)
    latitude_radius = radius_km / 111.32
    neighborhood_cache = (
        {} if neighborhood_cache is None else neighborhood_cache
    )
    used_neighborhoods = set()

    def neighborhood(position):
        position = int(position)
        used_neighborhoods.add(position)
        cached = neighborhood_cache.get(position)
        if cached is not None:
            return cached
        latitude = float(centers[position, 0])
        left = np.searchsorted(sorted_latitudes, latitude - latitude_radius, side="left")
        right = np.searchsorted(sorted_latitudes, latitude + latitude_radius, side="right")
        candidates = latitude_order[left:right]
        distance = _distance_row_km(centers[position], centers[candidates])
        cached = candidates[distance <= radius_km + 1e-9].astype(np.int64)
        neighborhood_cache[position] = cached
        return cached

    audit_masses, candidate_sizes, effective_top_ks = [], [], []
    hard_ring_target_counts = np.zeros(distance_bin_count, dtype=np.int64)
    hard_ring_realized_counts = np.zeros(distance_bin_count, dtype=np.int64)
    hard_ring_empty_fallback_count = 0
    hard_ring_violation_count = 0
    hard_ring_unpermitted_violation_count = 0
    ring_relaxation_count = 0
    ring_relaxation_step_total = 0
    ring_relaxation_radius_counts = np.zeros(
        distance_bin_count, dtype=np.int64
    )
    same_code_trigger_count = 0
    same_code_choice_count = 0
    same_code_ring_prebias_veto_count = 0
    same_code_ring_postbias_veto_count = 0
    same_code_ring_target_changed_count = 0
    sequence_index = 0
    while sequence_index < sequence_count:
        block_sequence_end = min(
            sequence_count, sequence_index + int(sequence_batch_size)
        )
        block_event_start = int(offsets[sequence_index])
        # Bound the full batched attribute product.  G0 can raise this limit
        # on 16-GB accelerators, while the historical 512-event default keeps
        # every other Phase-3 caller's memory and numerical route unchanged.
        while block_sequence_end > sequence_index + 1 \
                and int(offsets[block_sequence_end]) - block_event_start \
                > int(event_batch_size):
            block_sequence_end -= 1
        block_event_end = int(offsets[block_sequence_end])
        query_block = repaired_queries[block_event_start:block_event_end]
        if not torch.is_tensor(query_block):
            query_block = torch.from_numpy(
                np.asarray(query_block, dtype=np.float32)
            ).to(device)
        else:
            query_block = query_block.to(device=device, dtype=torch.float32)
        base_logits = F.normalize(query_block, dim=-1) @ target.T
        base_logits *= model.logit_scale.exp().clamp(max=100.0)
        scaled_logits = base_logits / float(temperature)
        if distance_aware:
            # The code-conditioned score is not monotone in either the
            # attribute logit or distance.  Evaluate the complete city
            # vocabulary so a long-jump ring can never be removed by the old
            # 3-sigma neighbourhood and the omitted-mass audit is exact.
            top_k = len(vocabulary)
            effective_top_ks.append(int(top_k))
            top_indices_np = None
        else:
            full_log_norm = torch.logsumexp(scaled_logits, dim=1)
            top_k = min(int(candidate_top_k), len(vocabulary))
            while True:
                top_scores, top_indices = torch.topk(
                    scaled_logits, k=top_k, dim=1, largest=True, sorted=False
                )
                retained = torch.exp(
                    torch.logsumexp(top_scores, dim=1) - full_log_norm
                )
                attribute_omitted = 1.0 - retained
                if float(attribute_omitted.max().item()) <= omitted_mass_tolerance \
                        or top_k == len(vocabulary):
                    break
                top_k = min(len(vocabulary), top_k * 2)
            effective_top_ks.append(int(top_k))
            top_indices_np = top_indices.cpu().numpy()
        block_sequences = list(range(sequence_index, block_sequence_end))
        currents, memories, code_memories = {}, {}, {}
        for sequence in block_sequences:
            start, end = int(offsets[sequence]), int(offsets[sequence + 1])
            if end <= start:
                continue
            home = int(homes[sequence])
            current = int(np.searchsorted(vocabulary, home))
            if current >= len(vocabulary) or vocabulary[current] != home:
                raise RuntimeError("Home is absent from the target landing vocabulary")
            currents[sequence] = current
            memories[sequence] = {current: float(initial_home_count)}
            if same_code_keys is not None:
                code_memories[sequence] = {
                    int(same_code_keys[start]): {current: 1.0}
                }
            if history_seed_grid_ids is not None:
                for grid_id in history_seed_grid_ids[sequence]:
                    grid_id = int(grid_id)
                    if grid_id < 0:
                        continue
                    position = int(np.searchsorted(vocabulary, grid_id))
                    if position >= len(vocabulary) \
                            or vocabulary[position] != grid_id:
                        raise RuntimeError(
                            "History seed grid is absent from the target landing "
                            "vocabulary"
                        )
                    # Home has its own prior. Duplicate anchors are idempotent.
                    if position != current:
                        memories[sequence].setdefault(position, float(initial_anchor_count))
            predicted[start] = home
        maximum_length = max(
            (int(offsets[sequence + 1] - offsets[sequence])
             for sequence in block_sequences), default=0
        )
        for relative_step in range(1, maximum_length):
            active = [
                sequence for sequence in block_sequences
                if sequence in currents
                and int(offsets[sequence]) + relative_step < int(offsets[sequence + 1])
            ]
            if not active:
                continue
            candidate_rows, event_rows, memory_rows = [], [], []
            same_code_memory_rows = []
            for sequence in active:
                event = int(offsets[sequence]) + relative_step
                event_row = event - block_event_start
                memory = memories[sequence]
                memory_positions = np.fromiter(
                    memory.keys(), dtype=np.int64
                )
                memory_counts = np.fromiter(
                    memory.values(), dtype=np.float32
                )
                if same_code_keys is None:
                    code_positions = np.empty(0, dtype=np.int64)
                    code_counts = np.empty(0, dtype=np.float32)
                else:
                    code_memory = code_memories[sequence].get(
                        int(same_code_keys[event]), {}
                    )
                    code_positions = np.fromiter(
                        code_memory.keys(), dtype=np.int64
                    )
                    code_counts = np.fromiter(
                        code_memory.values(), dtype=np.float32
                    )
                    same_code_trigger_count += int(len(code_positions) > 0)
                if distance_aware:
                    union = np.arange(len(vocabulary), dtype=np.int64)
                else:
                    union = np.unique(np.concatenate((
                        top_indices_np[event_row], neighborhood(currents[sequence]),
                        memory_positions, code_positions,
                    )))
                candidate_rows.append(union)
                event_rows.append(event_row)
                memory_rows.append((memory_positions, memory_counts))
                same_code_memory_rows.append((code_positions, code_counts))
                candidate_sizes.append(int(len(union)))
            width = max(len(row) for row in candidate_rows)
            padded = np.full((len(active), width), -1, dtype=np.int64)
            memory_bonus_values = np.zeros(
                (len(active), width), dtype=np.float32
            )
            same_code_bonus_values = np.zeros(
                (len(active), width), dtype=np.float32
            )
            for row_index, row in enumerate(candidate_rows):
                padded[row_index, :len(row)] = row
                memory_positions, memory_counts = memory_rows[row_index]
                memory_columns = np.searchsorted(row, memory_positions)
                memory_bonus_values[row_index, memory_columns] = memory_counts
                code_positions, code_counts = same_code_memory_rows[row_index]
                code_columns = np.searchsorted(row, code_positions)
                same_code_bonus_values[row_index, code_columns] = code_counts
            candidate_tensor = torch.from_numpy(padded).to(device)
            mask = candidate_tensor >= 0
            safe_candidates = candidate_tensor.clamp(min=0)
            event_tensor = torch.as_tensor(event_rows, dtype=torch.long, device=device)
            candidate_logits = scaled_logits[
                event_tensor.unsqueeze(1), safe_candidates
            ]
            candidate_centers = center_tensor[safe_candidates]
            previous = center_tensor[torch.as_tensor(
                [currents[sequence] for sequence in active],
                dtype=torch.long, device=device,
            )].unsqueeze(1)
            mean_lat = torch.deg2rad(0.5 * (candidate_centers[..., 0] + previous[..., 0]))
            north = (candidate_centers[..., 0] - previous[..., 0]) * 111.32
            east = (candidate_centers[..., 1] - previous[..., 1]) \
                * 111.32 * torch.cos(mean_lat)
            distance = torch.sqrt(north.square() + east.square())
            if distance_coordinate == "spatial_rank":
                origins = [currents[sequence] for sequence in active]
                if torch.is_tensor(spatial_rank_tensor):
                    origin_tensor = torch.as_tensor(
                        origins, dtype=torch.long, device=device,
                    ).unsqueeze(1).expand_as(safe_candidates)
                    distance_coordinate_values = spatial_rank_tensor[origin_tensor, safe_candidates]
                else:
                    distance_coordinate_values = torch.from_numpy(
                        spatial_rank_tensor.gather(origins, np.maximum(padded, 0))
                    ).to(device)
            else:
                distance_coordinate_values = distance
            memory_bonus = torch.from_numpy(memory_bonus_values).to(device)
            active_beta = float(revisit_beta) if sequence_revisit_beta is None else torch.as_tensor(
                sequence_revisit_beta[np.asarray(active)], device=device).unsqueeze(1)
            same_code_bonus = torch.from_numpy(
                same_code_bonus_values
            ).to(device)
            distance_bins = None
            if distance_aware or attraction_enabled:
                distance_bins = torch.bucketize(
                    distance_coordinate_values, physical_edge_tensor, right=True
                )
            if mixture_distance:
                if piecewise_distance:
                    density_bins = torch.bucketize(
                        distance_coordinate_values, piecewise_edge_tensor,
                        right=True,
                    )
                    density_bin_count = len(piecewise_edges) - 1
                else:
                    density_bins = distance_bins
                    density_bin_count = distance_bin_count
                absolute_events = torch.as_tensor(
                    np.asarray(event_rows, dtype=np.int64) + block_event_start,
                    dtype=torch.long, device=device,
                )
                event_probabilities = distance_probability_tensor[absolute_events]
                selected_probability = torch.gather(
                    event_probabilities, 1, density_bins
                )
                bin_cardinality = torch.zeros(
                    (len(active), density_bin_count),
                    dtype=distance.dtype, device=device,
                )
                bin_cardinality.scatter_add_(
                    1, density_bins,
                    torch.ones_like(distance, dtype=distance.dtype) * mask,
                )
                selected_cardinality = torch.gather(
                    bin_cardinality, 1, density_bins
                ).clamp(min=1.0)
                distance_score = (
                    torch.log(selected_probability.clamp(min=1e-12))
                    - torch.log(selected_cardinality)
                ) / float(distance_kernel_tau)
                score = candidate_logits + distance_score \
                    + active_beta * torch.log1p(memory_bonus)
            elif point_distance:
                target_distance = code_target_distance_tensor[torch.as_tensor(
                    np.asarray(event_rows, dtype=np.int64) + block_event_start,
                    dtype=torch.long, device=device,
                )].unsqueeze(1)
                distance_score = -torch.abs(
                    torch.log(distance + float(distance_log_epsilon_km))
                    - torch.log(target_distance + float(distance_log_epsilon_km))
                ) / float(distance_kernel_tau)
                score = candidate_logits + distance_score \
                    + active_beta * torch.log1p(memory_bonus)
            else:
                score = candidate_logits - distance / float(continuity_sigma) \
                    + active_beta * torch.log1p(memory_bonus)
            if same_code_revisit_beta > 0:
                score = score + float(same_code_revisit_beta) * torch.log1p(
                    same_code_bonus
                )
            if attraction_enabled:
                event_codes = event_attraction_code_tensor[torch.as_tensor(
                    np.asarray(event_rows, dtype=np.int64) + block_event_start,
                    dtype=torch.long, device=device,
                )]
                if attraction_mode == "raw":
                    attraction_values = attraction_value_tensor[event_codes]
                    attraction_log = float(attraction_strength) * torch.log(
                        attraction_values.clamp(min=float(attraction_epsilon))
                    )
                else:
                    attraction_ranks = attraction_rank_tensor[event_codes]
                    attraction_log = -float(attraction_strength) * torch.log(
                        attraction_ranks
                    )
                attraction_log = torch.gather(
                    attraction_log, 1, safe_candidates
                ).masked_fill(~mask, float("-inf"))
                # Convert the attraction into a within-ring likelihood ratio.
                # Combining it with the mixture's p(bin)/|bin| term therefore
                # redistributes mass inside a ring without assigning extra
                # probability mass to that distance ring.
                attraction_adjustment = torch.zeros_like(attraction_log)
                for jump_bin in range(distance_bin_count):
                    in_bin = mask & (distance_bins == jump_bin)
                    count = in_bin.sum(dim=1, keepdim=True)
                    masked_attraction = attraction_log.masked_fill(
                        ~in_bin, float("-inf")
                    )
                    log_normalizer = torch.logsumexp(
                        masked_attraction, dim=1, keepdim=True
                    )
                    adjustment = attraction_log - log_normalizer \
                        + torch.log(count.clamp(min=1).to(attraction_log.dtype))
                    attraction_adjustment = torch.where(
                        in_bin, adjustment, attraction_adjustment
                    )
                score = score + attraction_adjustment
            if hard_distance_ring_constraint:
                base_target_bins = hard_ring_target_bin_tensor[
                    absolute_events
                ].unsqueeze(1)
                target_bins = base_target_bins
                same_code_target_mask = same_code_bonus > 0
                same_code_trigger_mask = same_code_target_mask.any(dim=1)
                base_hard_mask = mask & (distance_bins == base_target_bins)
                same_code_ring_prebias_veto_count += int((
                    same_code_trigger_mask
                    & ~(base_hard_mask & same_code_target_mask).any(dim=1)
                ).sum().item())
                if same_code_ring_bias > 0:
                    same_code_bin_counts = torch.zeros(
                        (len(active), distance_bin_count),
                        dtype=same_code_bonus.dtype, device=device,
                    )
                    same_code_bin_counts.scatter_add_(
                        1, distance_bins,
                        same_code_bonus * mask.to(same_code_bonus.dtype),
                    )
                    # Scoring may use the finer piecewise distribution, but
                    # the hard ring is always drawn from the frozen four-bin
                    # coarse distribution.
                    coarse_event_probabilities = hard_ring_probability_tensor[
                        absolute_events
                    ]
                    coarse_probability = coarse_event_probabilities.to(
                        torch.float64
                    )
                    log_tilted = torch.where(
                        coarse_probability > 0,
                        torch.log(coarse_probability.clamp(
                            min=torch.finfo(torch.float64).tiny
                        )) + float(same_code_ring_bias) * torch.log1p(
                            same_code_bin_counts.to(torch.float64)
                        ),
                        torch.full_like(coarse_probability, float("-inf")),
                    )
                    tilted = torch.exp(
                        log_tilted - torch.logsumexp(
                            log_tilted, dim=1, keepdim=True
                        )
                    )
                    tilted_cdf = torch.cumsum(tilted, dim=1)
                    ring_uniform = hard_ring_uniform_tensor[
                        absolute_events
                    ].unsqueeze(1)
                    tilted_bins = torch.sum(
                        tilted_cdf < ring_uniform, dim=1
                    ).clamp(max=distance_bin_count - 1).unsqueeze(1)
                    target_bins = torch.where(
                        same_code_trigger_mask.unsqueeze(1),
                        tilted_bins, base_target_bins,
                    )
                    same_code_ring_target_changed_count += int((
                        same_code_trigger_mask
                        & (target_bins.squeeze(1) != base_target_bins.squeeze(1))
                    ).sum().item())
                hard_mask = mask & (distance_bins == target_bins)
                has_hard_candidate = hard_mask.any(dim=1)
                hard_ring_allowed_radius = torch.zeros(
                    len(active), dtype=torch.long, device=device
                )
                initially_empty = ~has_hard_candidate
                if hard_distance_ring_empty_policy == "expand_adjacent":
                    relaxed_mask = hard_mask
                    unresolved = initially_empty.clone()
                    for radius in range(1, distance_bin_count):
                        expanded = mask & (
                            torch.abs(distance_bins - target_bins) <= radius
                        )
                        newly_resolved = unresolved & expanded.any(dim=1)
                        relaxed_mask = torch.where(
                            newly_resolved.unsqueeze(1), expanded, relaxed_mask
                        )
                        hard_ring_allowed_radius = torch.where(
                            newly_resolved,
                            torch.full_like(hard_ring_allowed_radius, radius),
                            hard_ring_allowed_radius,
                        )
                        unresolved &= ~newly_resolved
                    if bool(unresolved.any().item()):
                        raise RuntimeError(
                            "hard distance-ring adjacent expansion exhausted all "
                            "physical bands without a candidate"
                        )
                    radii = hard_ring_allowed_radius.cpu().numpy()
                    relaxed = radii > 0
                    ring_relaxation_count += int(np.count_nonzero(relaxed))
                    ring_relaxation_step_total += int(radii.sum())
                    ring_relaxation_radius_counts += np.bincount(
                        radii, minlength=distance_bin_count
                    )
                    mask = relaxed_mask
                else:
                    hard_ring_empty_fallback_count += int(
                        initially_empty.sum().item()
                    )
                    mask = torch.where(
                        has_hard_candidate.unsqueeze(1), hard_mask, mask
                    )
                same_code_ring_postbias_veto_count += int((
                    same_code_trigger_mask
                    & ~(mask & same_code_target_mask).any(dim=1)
                ).sum().item())
                hard_ring_target_counts += np.bincount(
                    target_bins.squeeze(1).cpu().numpy(),
                    minlength=distance_bin_count,
                )
            score = score.masked_fill(~mask, float("-inf"))
            if selection_mode == "argmax":
                chosen_columns = score.argmax(dim=1).cpu().numpy()
            else:
                sampling_probability = torch.softmax(
                    score / float(selection_temperature), dim=1
                )
                cdf = torch.cumsum(sampling_probability, dim=1)
                event_indices = np.asarray(event_rows, dtype=np.int64) \
                    + block_event_start
                uniforms = selection_uniform_tensor[torch.as_tensor(
                    event_indices, dtype=torch.long, device=device
                )].unsqueeze(1)
                chosen_columns = torch.sum(cdf < uniforms, dim=1) \
                    .clamp(max=width - 1).cpu().numpy()
                chosen_columns = np.minimum(
                    chosen_columns,
                    np.asarray([len(row) - 1 for row in candidate_rows], dtype=np.int64),
                )
            if hard_distance_ring_constraint:
                chosen_distance_bins = distance_bins[
                    torch.arange(len(active), device=device),
                    torch.as_tensor(chosen_columns, dtype=torch.long, device=device),
                ].cpu().numpy()
                target_distance_bins = target_bins.squeeze(1).cpu().numpy()
                hard_ring_realized_counts += np.bincount(
                    chosen_distance_bins, minlength=distance_bin_count
                )
                differences = np.abs(
                    chosen_distance_bins - target_distance_bins
                )
                hard_ring_violation_count += int(np.count_nonzero(differences))
                permitted_radius = hard_ring_allowed_radius.cpu().numpy()
                hard_ring_unpermitted_violation_count += int(
                    np.count_nonzero(differences > permitted_radius)
                )
            for row_index, sequence in enumerate(active):
                chosen = int(candidate_rows[row_index][chosen_columns[row_index]])
                event = int(offsets[sequence]) + relative_step
                if same_code_keys is not None \
                        and same_code_bonus_values[
                            row_index, chosen_columns[row_index]
                        ] > 0:
                    same_code_choice_count += 1
                if distance_aware and len(audit_masses) \
                        < int(omitted_mass_audit_steps):
                    # Every active grid was scored above, so this is an exact
                    # zero rather than the legacy attribute-only estimate.
                    audit_masses.append(0.0)
                elif len(audit_masses) < int(omitted_mass_audit_steps):
                    previous_position = currents[sequence]
                    full_distance = torch.from_numpy(_distance_row_km(
                        centers[previous_position], centers
                    ).astype(np.float32)).to(device)
                    full_memory_values = np.zeros(
                        len(vocabulary), dtype=np.float32
                    )
                    memory = memories[sequence]
                    memory_positions = np.fromiter(
                        memory.keys(), dtype=np.int64
                    )
                    memory_counts = np.fromiter(
                        memory.values(), dtype=np.float32
                    )
                    full_memory_values[memory_positions] = memory_counts
                    full_memory = torch.from_numpy(full_memory_values).to(device)
                    full_score = scaled_logits[event_rows[row_index]] \
                        - full_distance / float(continuity_sigma) \
                        + (float(revisit_beta) if sequence_revisit_beta is None
                           else float(sequence_revisit_beta[sequence])) * torch.log1p(full_memory)
                    probability = torch.softmax(full_score, dim=0)
                    retained_mass = probability[torch.as_tensor(
                        candidate_rows[row_index], dtype=torch.long, device=device
                    )].sum()
                    audit_masses.append(float((1.0 - retained_mass).clamp(min=0).item()))
                currents[sequence] = chosen
                memories[sequence][chosen] = memories[sequence].get(chosen, 0.0) + 1.0
                if same_code_keys is not None:
                    key = int(same_code_keys[event])
                    code_memory = code_memories[sequence].setdefault(key, {})
                    code_memory[chosen] = code_memory.get(chosen, 0.0) + 1.0
                predicted[event] = vocabulary[chosen]
        sequence_index = block_sequence_end
    omitted_max = max(audit_masses, default=0.0)
    audit = {
        "contract": "attribute_topk_plus_previous_3sigma_plus_revisit_memory_v1",
        "attribute_scoring": "batched_H_at_A_transpose",
        "autoregression": "relative_step_user_batch_with_padding_and_mask",
        "requested_attribute_top_k": int(candidate_top_k),
        "effective_attribute_top_k_max": max(effective_top_ks, default=0),
        "candidate_size_mean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "candidate_size_max": max(candidate_sizes, default=0),
        "physical_neighbor_radius_sigma": float(candidate_radius_multiplier),
        "physical_neighbor_cache_entries": len(used_neighborhoods),
        "physical_radius_candidates_used": bool(used_neighborhoods),
        "pareto_distance_cost": False,
        "omitted_probability_mass_exact_audit_steps": len(audit_masses),
        "omitted_probability_mass_max": float(omitted_max),
        "omitted_probability_mass_mean": float(np.mean(audit_masses)) if audit_masses else 0.0,
        "omitted_probability_mass_target": float(omitted_mass_tolerance),
        "omitted_probability_mass_pass": bool(omitted_max <= omitted_mass_tolerance),
        "selection_mode": selection_mode,
        "selection_seed": (
            None if selection_seed is None else int(selection_seed)
        ),
        "selection_temperature": float(selection_temperature),
        "selection_randomness_contract": (
            "none_argmax" if selection_mode == "argmax" else
            "numpy_PCG64_event_indexed_uniform_then_gpu_cdf_v1"
        ),
    }
    if personal_ring_multiplier is not None or sequence_revisit_beta is not None:
        audit["personal_scale"] = {
            "ring_tilt_enabled": personal_ring_multiplier is not None,
            "personal_revisit_enabled": sequence_revisit_beta is not None,
        }
    if same_code_keys is not None:
        audit["same_code_revisit"] = {
            "enabled": bool(same_code_keys is not None),
            "key_layers": same_code_key_layers,
            "revisit_beta": float(same_code_revisit_beta),
            "ring_bias": float(same_code_ring_bias),
            "trigger_count": int(same_code_trigger_count),
            "choice_count": int(same_code_choice_count),
            "choice_fraction_given_trigger": float(
                same_code_choice_count / max(1, same_code_trigger_count)
            ),
            "ring_prebias_veto_count": int(
                same_code_ring_prebias_veto_count
            ),
            "ring_postbias_veto_count": int(
                same_code_ring_postbias_veto_count
            ),
            "ring_target_changed_count": int(
                same_code_ring_target_changed_count
            ),
            "ring_reweighting": (
                "support_preserving_p_code_bin_times_"
                "one_plus_same_code_count_power_bias"
                if same_code_ring_bias > 0 else "disabled"
            ),
        }
    if distance_aware:
        if mixture_distance:
            score_formula = (
                "candidate_logits + log(p(code_distance_segment)/"
                "segment_cardinality)/tau + revisit"
                if piecewise_distance else
                "candidate_logits + log(p(code_bin)/bin_cardinality)/tau + "
                "revisit"
            )
        else:
            score_formula = (
                "candidate_logits - abs(log(distance+epsilon)-"
                "log(code_target+epsilon))/tau + revisit"
            )
        if attraction_enabled:
            score_formula += " + normalized_attraction"
        if same_code_revisit_beta > 0:
            score_formula += " + same_code_beta*log1p(daily_code_count)"
        audit.update({
            "contract": (
                "full_vocabulary_code_distance_mixture_revisit_v2"
                if mixture_distance else
                "full_vocabulary_code_distance_kernel_revisit_v1"
            ),
            "distance_mode": (
                (
                    "source_train_layer0_code_conditioned_spatial_rank_mixture"
                    if distance_coordinate == "spatial_rank" else
                    "source_train_layer0_code_conditioned_physical_bin_mixture"
                ) if mixture_distance else
                "source_train_layer0_code_conditioned_log_distance_kernel"
            ),
            "distance_coordinate": distance_coordinate,
            "distance_kernel_tau": float(distance_kernel_tau),
            "distance_log_epsilon_km": float(distance_log_epsilon_km),
            "score_formula": score_formula,
            "physical_distance_bin_edges_km": [
                "inf" if np.isposinf(value) else float(value)
                for value in physical_edges
            ],
            "distance_bin_edges": [
                "inf" if np.isposinf(value) else float(value)
                for value in physical_edges
            ],
            "within_bin_distance_density": (
                "source_train_empirical_piecewise_then_uniform_grid_support"
                if piecewise_distance else "uniform_grid_support_within_4_bins"
            ),
            "physical_distance_piecewise_edges_km": (
                None if piecewise_edges is None else [
                    "inf" if np.isposinf(value) else float(value)
                    for value in piecewise_edges
                ]
            ),
            "distance_piecewise_edges": (
                None if piecewise_edges is None else [
                    "inf" if np.isposinf(value) else float(value)
                    for value in piecewise_edges
                ]
            ),
            "attraction_in_score": bool(attraction_enabled),
            "attraction_deferred_to": (
                None if attraction_enabled else "P3-B2"
            ),
            "history_seeded": bool(history_seed_grid_ids is not None),
            "full_vocabulary_scoring": True,
            "hard_distance_ring_constraint": hard_distance_ring_constraint,
            "hard_distance_ring_seed": hard_distance_ring_seed,
            "hard_distance_ring_empty_policy": (
                hard_distance_ring_empty_policy
            ),
            "physical_neighbor_radius_sigma": None,
            "physical_neighbor_cache_entries": 0,
        })
        if hard_distance_ring_constraint:
            target_total = int(hard_ring_target_counts.sum())
            realized_total = int(hard_ring_realized_counts.sum())
            if distance_coordinate == "physical_km":
                ring_contract = (
                    "full_vocabulary_hard_physical_ring_with_adjacent_"
                    "band_expansion_v2"
                    if hard_distance_ring_empty_policy == "expand_adjacent"
                    else "full_vocabulary_sampled_hard_physical_ring_v1"
                )
                sampled_ring = "sampled_physical_bin"
            else:
                ring_contract = (
                    "full_vocabulary_hard_spatial_rank_ring_with_adjacent_"
                    "band_expansion_v2"
                    if hard_distance_ring_empty_policy == "expand_adjacent"
                    else "full_vocabulary_sampled_hard_spatial_rank_ring_v1"
                )
                sampled_ring = "sampled_spatial_rank_bin"
            audit.update({
                "contract": ring_contract,
                "score_formula": (
                    (
                        "adjacent_expand_if_empty(hard_mask("
                        "%s)) + " % sampled_ring
                        if hard_distance_ring_empty_policy == "expand_adjacent"
                        else "hard_mask(%s) + " % sampled_ring
                    ) + score_formula
                ),
                "hard_distance_ring_target_sampling": (
                    "source_train_code_%s_bin_mixture_"
                    % ("spatial_rank" if distance_coordinate == "spatial_rank"
                       else "physical") +
                    "numpy_PCG64_independent_stream_v1"
                ),
                "hard_distance_ring_seed": int(hard_distance_ring_seed),
                "hard_distance_ring_empty_policy": (
                    hard_distance_ring_empty_policy
                ),
                "hard_distance_ring_target_counts": (
                    hard_ring_target_counts.tolist()
                ),
                "hard_distance_ring_target_share": (
                    hard_ring_target_counts / max(1, target_total)
                ).tolist(),
                "hard_distance_ring_realized_counts": (
                    hard_ring_realized_counts.tolist()
                ),
                "hard_distance_ring_realized_share": (
                    hard_ring_realized_counts / max(1, realized_total)
                ).tolist(),
                "hard_distance_ring_transition_count": target_total,
                "hard_distance_ring_empty_fallback_count": int(
                    hard_ring_empty_fallback_count
                ),
                "hard_distance_ring_empty_fallback_fraction": float(
                    hard_ring_empty_fallback_count / max(1, target_total)
                ),
                "ring_relaxation_count": int(ring_relaxation_count),
                "ring_relaxation_event_fraction": float(
                    ring_relaxation_count / max(1, target_total)
                ),
                "ring_relaxation_step_count": int(
                    ring_relaxation_step_total
                ),
                "ring_relaxation_radius_counts": (
                    ring_relaxation_radius_counts.tolist()
                ),
                "ring_relaxation_max_radius": int(
                    max(
                        (index for index, count in enumerate(
                            ring_relaxation_radius_counts
                        ) if count),
                        default=0,
                    )
                ),
                "hard_distance_ring_violation_count": int(
                    hard_ring_violation_count
                ),
                "hard_distance_ring_unpermitted_violation_count": int(
                    hard_ring_unpermitted_violation_count
                ),
                "hard_distance_ring_pass": bool(
                    hard_ring_empty_fallback_count == 0
                    and hard_ring_unpermitted_violation_count == 0
                ),
            })
    if attraction_enabled:
        audit.update({
            "attraction_contract": (
                "code_conditioned_map_only_within_physical_bin_"
                "likelihood_ratio_v1"
                if distance_coordinate == "physical_km" else
                "code_conditioned_map_only_within_spatial_rank_bin_"
                "likelihood_ratio_v1"
            ),
            "attraction_mode": attraction_mode,
            "attraction_strength": float(attraction_strength),
            "attraction_epsilon": float(attraction_epsilon),
            "attraction_normalization": (
                "per_event_per_current_grid_per_%s_bin_unit_mean"
                % ("physical" if distance_coordinate == "physical_km"
                   else "spatial_rank")
            ),
            "attraction_changes_distance_bin_mass_directly": False,
        })
    return (predicted, audit) if return_audit else predicted


def _distance_row_km(center, centers):
    centers = np.asarray(centers, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    mean_lat = np.radians(0.5 * (centers[:, 0] + center[0]))
    north = (centers[:, 0] - center[0]) * 111.32
    east = (centers[:, 1] - center[1]) * 111.32 * np.cos(mean_lat)
    return np.sqrt(north * north + east * east)


def spatial_rank_table(centers, weights=None, *, lazy=False, cache_rows=64):
    """Return Noulas spatial rank: number of destinations strictly closer.

    Rows are origins and columns are destinations. Equal-distance destinations
    receive the same rank, and the origin itself therefore has rank zero. The
    table is small for CIVIC's emitted sparse grids and can be cached on a GPU
    across all landing arms.
    """
    from SHAPE.stage3.opportunity_rank import SpatialRankIndex
    index = SpatialRankIndex(centers, weights=weights, cache_rows=cache_rows)
    return index if lazy else index.dense()


def _period_index(minutes):
    minutes = np.asarray(minutes, dtype=np.int64)
    result = np.full(len(minutes), 3, dtype=np.int8)
    result[(minutes >= 6 * 60) & (minutes < 11 * 60)] = 0
    result[(minutes >= 11 * 60) & (minutes < 17 * 60)] = 1
    result[(minutes >= 17 * 60) & (minutes < 22 * 60)] = 2
    return result


def _pearson(left, right):
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _od_mass_audit(
        predicted, reference, *, mass_tolerance=DEFAULT_OD_MASS_TOLERANCE,
        require_mass_alignment=True):
    """Audit the unnormalised OD mass before scale-sensitive metrics.

    CPC and NRMSE silently become measures of sample size when the two OD
    matrices carry different total transition mass.  Keep this check next to
    the formulas so every caller is fail-closed unless it explicitly requests
    a descriptive-only comparison.
    """
    tolerance = float(mass_tolerance)
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("mass_tolerance must be finite and in [0, 1)")
    pred_od = np.asarray(predicted["od"], dtype=np.float64)
    ref_od = np.asarray(reference["od"], dtype=np.float64)
    if pred_od.shape != ref_od.shape:
        raise ValueError(
            "predicted and reference OD arrays must have identical shapes"
        )
    if np.any(~np.isfinite(pred_od)) or np.any(~np.isfinite(ref_od)) \
            or np.any(pred_od < 0.0) or np.any(ref_od < 0.0):
        raise ValueError("OD arrays must contain finite non-negative values")
    pred_mass = float(pred_od.sum())
    ref_mass = float(ref_od.sum())
    mass_ratio = None if ref_mass <= 0.0 else float(pred_mass / ref_mass)
    if pred_mass == 0.0 and ref_mass == 0.0:
        # One-event strata legitimately contain no OD transitions.  Their CPC
        # and NRMSE remain undefined, but there is no cross-side mass mismatch.
        mass_ratio = 1.0
        folded_ratio = 1.0
        cpc_ceiling = None
        aligned = True
    elif pred_mass <= 0.0 or ref_mass <= 0.0:
        folded_ratio = None
        cpc_ceiling = 0.0
        aligned = False
    else:
        folded_ratio = float(max(mass_ratio, 1.0 / mass_ratio))
        # This symmetric form is valid both below and above ratio 1.  For
        # pred_mass >= ref_mass it reduces to the familiar 2 / (1 + ratio).
        cpc_ceiling = float(2.0 / (1.0 + folded_ratio))
        aligned = bool(abs(mass_ratio - 1.0) <= tolerance)
    audit = {
        "pred_mass": pred_mass,
        "ref_mass": ref_mass,
        "mass_ratio": mass_ratio,
        "mass_ratio_larger_over_smaller": folded_ratio,
        "cpc_ceiling": cpc_ceiling,
        "mass_unit": OD_MASS_UNIT,
        "mass_tolerance": tolerance,
        "mass_alignment_required": bool(require_mass_alignment),
        "mass_alignment_pass": bool(aligned),
        "metric_role": (
            "formal" if require_mass_alignment else "descriptive_only"
        ),
    }
    if require_mass_alignment and not aligned:
        ratio_text = "undefined" if mass_ratio is None else f"{mass_ratio:.6g}"
        ceiling_text = (
            "undefined" if cpc_ceiling is None else f"{cpc_ceiling:.6g}"
        )
        raise ValueError(
            "OD mass alignment required before CPC/NRMSE evaluation: "
            f"pred_mass={pred_mass:.6g}, ref_mass={ref_mass:.6g}, "
            f"pred/ref={ratio_text}, allowed=[{1.0 - tolerance:.6g}, "
            f"{1.0 + tolerance:.6g}], maximum_possible_CPC={ceiling_text}. "
            "Align the person-day shell first, or pass "
            "require_mass_alignment=False only for an explicitly "
            "descriptive read."
        )
    return audit


def _compare_aggregates(
        predicted, reference, poi_table, *,
        mass_tolerance=DEFAULT_OD_MASS_TOLERANCE,
        require_mass_alignment=True, mass_audit=None):
    audit = _od_mass_audit(
        predicted,
        reference,
        mass_tolerance=mass_tolerance,
        require_mass_alignment=require_mass_alignment,
    )
    if mass_audit is not None:
        mass_audit.update(audit)
    pred_od, ref_od = predicted["od"], reference["od"]
    denominator = float(pred_od.sum() + ref_od.sum())
    od_cpc = None if denominator == 0 else float(
        2.0 * np.minimum(pred_od, ref_od).sum() / denominator
    )
    ref_mean = float(ref_od.mean())
    od_nrmse = None if ref_mean <= 0 else float(
        np.sqrt(np.mean(np.square(pred_od - ref_od))) / ref_mean
    )
    period_values = [
        jensen_shannon_distribution(predicted["periods"][index], reference["periods"][index])
        for index in range(4)
    ]
    pred_visits, ref_visits = predicted["visits"], reference["visits"]
    grid_count = len(pred_visits)
    poi = np.asarray(poi_table[:, :13], dtype=np.float64)
    poi = np.square(poi)
    poi /= poi.sum(axis=1, keepdims=True).clip(min=1e-12)
    pred_poi, ref_poi = pred_visits @ poi, ref_visits @ poi
    return {
        "M1_visit_jsd": float(jensen_shannon_distribution(pred_visits, ref_visits)),
        "M2_od_cpc": od_cpc,
        "M2_od_pearson": _pearson(pred_od, ref_od),
        "M2_od_nrmse": od_nrmse,
        "M3_period_space_jsd_mean": float(np.mean(period_values)),
        **{"M3_%s_jsd" % PERIODS[index][0]: float(value)
           for index, value in enumerate(period_values)},
        "M4_rhythm_24h_jsd_descriptive": float(
            jensen_shannon_distribution(predicted["hours"], reference["hours"])
        ),
        "A_visit_entropy_predicted": _entropy(pred_visits) / max(1.0, math.log2(grid_count)),
        "A_visit_entropy_reference": _entropy(ref_visits) / max(1.0, math.log2(grid_count)),
        "A_grid_coverage_predicted": float(np.count_nonzero(pred_visits) / grid_count),
        "A_grid_coverage_reference": float(np.count_nonzero(ref_visits) / grid_count),
        "A_radius_of_gyration_distribution_jsd": float(
            jensen_shannon_distribution(predicted["radius_hist"], reference["radius_hist"])
        ),
        "A_jump_distance_spectrum_jsd": float(
            jensen_shannon_distribution(predicted["jump_hist"], reference["jump_hist"])
        ),
        "A_jump_bin_distribution_jsd": float(
            jensen_shannon_distribution(
                predicted["jump_bin_hist"], reference["jump_bin_hist"]
            )
        ),
        "A_visited_poi_category_share_jsd": float(
            jensen_shannon_distribution(pred_poi, ref_poi)
        ),
    }
