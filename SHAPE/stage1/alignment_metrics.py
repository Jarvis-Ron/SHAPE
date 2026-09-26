"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from SHAPE.stage1.config import MODULE1_CONFIG


POI_JS_THRESH = float(MODULE1_CONFIG["poi_eval_js_threshold"])


TRANSIT_COS_THRESH = 0.55 # min cosine between 6-D transit vectors


SOFT_RECALL_KS = [1, 5, 10]


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def valid_mask(features: np.ndarray) -> np.ndarray:
    return features[:, :14].sum(axis=1) > 0


def dominant_cat(features: np.ndarray) -> np.ndarray:
    return features[:, :13].argmax(axis=1).astype(np.int64)


def poi_js_matrix(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """Return pairwise natural-log Jensen-Shannon divergence for POI profiles."""
    n_a, n_b = len(feat_a), len(feat_b)
    poi_a = feat_a[:, :13]
    poi_b = feat_b[:, :13]
    # Vectorized JS divergence in two blocks because of memory.
    # Blocking keeps the bounded diagnostic matrix within a predictable budget.
    pair_js = np.empty((n_a, n_b), dtype=np.float32)
    block = 256
    eps = 1e-12
    p_a = np.clip(poi_a, 0.0, None)
    s_a = p_a.sum(axis=1, keepdims=True)
    p_a = p_a / np.maximum(s_a, eps)
    for s in range(0, n_b, block):
        e = min(s + block, n_b)
        p_b = np.clip(poi_b[s:e], 0.0, None)
        s_b = p_b.sum(axis=1, keepdims=True)
        p_b = p_b / np.maximum(s_b, eps)
        m = 0.5 * (p_a[:, None, :] + p_b[None, :, :])
        kl_am = (p_a[:, None, :] * np.log((p_a[:, None, :] + eps) / (m + eps))).sum(axis=2)
        kl_bm = (p_b[None, :, :] * np.log((p_b[None, :, :] + eps) / (m + eps))).sum(axis=2)
        pair_js[:, s:e] = 0.5 * (kl_am + kl_bm)
    return pair_js


def poi_sim_matrix(
        feat_a: np.ndarray, feat_b: np.ndarray,
        threshold: float = POI_JS_THRESH) -> np.ndarray:
    """Return non-empty POI-profile matches below a strict JSD cutoff."""
    if not 0.0 < threshold < np.log(2.0):
        raise ValueError("POI JSD threshold must be between 0 and ln(2)")
    poi_a = feat_a[:, :13]
    poi_b = feat_b[:, :13]
    pair_gs = poi_js_matrix(feat_a, feat_b) < float(threshold)
    eps = 1e-12
    valid_a = np.clip(poi_a, 0.0, None).sum(axis=1) > eps
    valid_b = np.clip(poi_b, 0.0, None).sum(axis=1) > eps
    return pair_gs & valid_a[:, None] & valid_b[None, :]


def transit_sim_matrix(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """Return cross-city transit-feature matches using cosine similarity."""
    return _cosine_matrix(
        feat_a[:, 14:20], feat_b[:, 14:20]
    ) >= TRANSIT_COS_THRESH


def soft_sim_matrix(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """Row-wise soft-matching matrix via POI JS AND transit cosine."""
    return poi_sim_matrix(feat_a, feat_b) & transit_sim_matrix(feat_a, feat_b)


def recall_at_k(sim: np.ndarray, gt: np.ndarray, ks: List[int]) -> Dict[int, float]:
    """sim: [Nq, Nt]; gt: [Nq, Nt] boolean. Recall@K = P(hit in top-K)."""
    results = {}
    for requested_k in ks:
        effective_k = min(requested_k, sim.shape[1])
        top = np.argpartition(-sim, effective_k - 1, axis=1)[:, :effective_k]
        row_gt = np.take_along_axis(gt, top, axis=1)
        results[requested_k] = float(np.mean(row_gt.any(axis=1)))
    return results


def mrr(sim: np.ndarray, gt: np.ndarray) -> float:
    """Mean reciprocal rank of the first correct hit."""
    n = sim.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-sim, axis=1)
    sorted_gt = np.take_along_axis(gt, order, axis=1)
    ranks = np.argmax(sorted_gt, axis=1) + 1.0
    ranks[~sorted_gt.any(axis=1)] = np.inf
    return float(np.mean(1.0 / np.maximum(ranks, 1.0)))


def retrieval_summary(
        embs: Mapping[str, np.ndarray],
        feats: Mapping[str, np.ndarray],
        ks: Sequence[int] = SOFT_RECALL_KS) -> Dict[str, float]:
    """Return compact bidirectional retrieval metrics for training logs.

    The full alignment report below intentionally retains its historical
    per-pair layout.  This summary evaluates both directions of every city
    pair and averages them into a small set of values that can be printed and
    stored every few training epochs.
    """
    cities = list(embs)
    if len(cities) < 2:
        raise ValueError("retrieval evaluation requires at least two cities")

    requested_ks = [int(k) for k in ks]
    modes = ("hard", "poi", "transit", "soft")
    recalls = {
        mode: {k: [] for k in requested_ks} for mode in modes
    }
    reciprocal_ranks = {mode: [] for mode in modes}
    poi_positive_densities = []
    poi_query_coverages = []
    for index, city_a in enumerate(cities):
        for city_b in cities[index + 1:]:
            valid_a = valid_mask(feats[city_a])
            valid_b = valid_mask(feats[city_b])
            if not valid_a.any() or not valid_b.any():
                continue
            emb_a, emb_b = embs[city_a][valid_a], embs[city_b][valid_b]
            feat_a, feat_b = feats[city_a][valid_a], feats[city_b][valid_b]
            sim_ab = _cosine_matrix(emb_a, emb_b)
            cat_a, cat_b = dominant_cat(feat_a), dominant_cat(feat_b)
            hard_ab = cat_a[:, None] == cat_b[None, :]
            poi_ab = poi_sim_matrix(feat_a, feat_b)
            transit_ab = transit_sim_matrix(feat_a, feat_b)
            soft_ab = soft_sim_matrix(feat_a, feat_b)
            for sim, ground_truth in (
                    (sim_ab, {
                        "hard": hard_ab,
                        "poi": poi_ab,
                        "transit": transit_ab,
                        "soft": soft_ab,
                    }),
                    (sim_ab.T, {
                        "hard": hard_ab.T,
                        "poi": poi_ab.T,
                        "transit": transit_ab.T,
                        "soft": soft_ab.T,
                    })):
                for mode, gt in ground_truth.items():
                    values = recall_at_k(sim, gt, requested_ks)
                    for k in requested_ks:
                        recalls[mode][k].append(values[k])
                    reciprocal_ranks[mode].append(mrr(sim, gt))
                    if mode == "poi":
                        poi_positive_densities.append(float(gt.mean()))
                        poi_query_coverages.append(
                            float(gt.any(axis=1).mean())
                        )

    if not reciprocal_ranks["hard"]:
        raise ValueError("retrieval evaluation found no valid cross-city pairs")

    summary = {}
    for mode in modes:
        for k in requested_ks:
            summary[f"{mode}_r@{k}"] = float(np.mean(recalls[mode][k]))
        summary[f"{mode}_mrr"] = float(np.mean(reciprocal_ranks[mode]))
    summary["poi_positive_density"] = float(
        np.mean(poi_positive_densities)
    )
    summary["poi_query_coverage"] = float(np.mean(poi_query_coverages))
    return summary


def evaluate_live_retrieval(
        model: torch.nn.Module,
        city_data: Mapping[str, Tuple[np.ndarray, np.ndarray]],
        device: Optional[torch.device] = None,
        max_grids_per_city: Optional[int] = None) -> Dict[str, float]:
    """Encode complete cities with a live model and score retrieval quality."""
    try:
        model_device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive for parameterless fakes
        model_device = torch.device("cpu")
    resolved_device = torch.device(device) if device is not None else model_device
    was_training = model.training
    model.eval()
    embs, feats = {}, {}
    try:
        with torch.inference_mode():
            for city, value in city_data.items():
                features, grid_ids = value[:2]
                positions = None
                if len(value) >= 4:
                    positions = np.column_stack((value[2], value[3])).astype(np.int64)
                    positions = positions % 32 + 4
                if max_grids_per_city is not None and len(features) > max_grids_per_city:
                    keep = np.linspace(
                        0, len(features) - 1, int(max_grids_per_city), dtype=np.int64
                    )
                    features, grid_ids = features[keep], grid_ids[keep]
                    if positions is not None:
                        positions = positions[keep]
                feature_tensor = torch.from_numpy(features).float().to(resolved_device)
                grid_tensor = torch.from_numpy(grid_ids).long().to(resolved_device)
                position_tensor = (
                    torch.from_numpy(positions).long().to(resolved_device)
                    if positions is not None else None
                )
                embeddings = (
                    model.encode_city(
                        feature_tensor, grid_tensor, positions=position_tensor
                    )
                    if position_tensor is not None else
                    model.encode_city(feature_tensor, grid_tensor)
                )
                embs[city] = embeddings.float().cpu().numpy()
                feats[city] = features
    finally:
        model.train(was_training)
    return retrieval_summary(embs, feats)
