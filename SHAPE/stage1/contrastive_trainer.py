"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _city_parts(value):
    """Accept legacy pairs and stride-aware (features, ids, rows, cols)."""
    if len(value) == 2:
        features, grid_ids = value
        positions = None
    elif len(value) == 4:
        features, grid_ids, rows, cols = value
        positions = np.column_stack((rows, cols)).astype(np.int64)
    else:
        raise ValueError("city data needs (features, ids[, rows, cols])")
    return features, grid_ids, positions


class MultiCityContrastiveDataset(Dataset):
    """Multi-city contrastive dataset for cross-city feature alignment.

    pos_rule:
      - "feat_sim" (default): keep the legacy behaviour -- cross-city positives
        are the top-k most feature-similar grids (no semantic constraint).
      - "dominant_cat": cross-city positives additionally require that the two
        grids share the same dominant POI category. If no same-category grid
        remains among the top-k, the single most similar same-category grid is
        kept (so every valid anchor still sees at least one cross-city positive).
      - "poi_profile": rank positives by cosine similarity of the complete
        normalized 13-D POI profile.  The loss then uses the complete in-batch
        POI-similarity distribution as a continuous multi-positive target,
        rather than treating one sampled neighbor as the only positive.

    hard_neg_k (int): if >0, for every anchor collect up to hard_neg_k
        cross-city grids whose dominant category differs from the anchor but
        whose raw feature similarity is the highest. These are returned in
        __getitem__ as an explicit negative feature for hard negative mining.
    """

    def __init__(self, city_data, topk_feat_sim=5, temperature=0.07, seed=42,
                 pos_rule="feat_sim", hard_neg_k=0,
                 dominant_boost_topk=6, min_dom_pos=3, hard_neg_n=1):
        self.city_data = city_data
        self.city_names = list(city_data.keys())
        self.city_indices = {
            name: index for index, name in enumerate(self.city_names)
        }
        self.topk_feat_sim = topk_feat_sim
        self.temperature = temperature
        self.pos_rule = pos_rule
        if self.pos_rule not in ("feat_sim", "dominant_cat", "poi_profile"):
            raise ValueError("unknown Module-1 positive rule: %s" % self.pos_rule)
        self.hard_neg_k = max(0, hard_neg_k)
        if self.pos_rule == "poi_profile":
            # A different dominant category can still have a very similar
            # mixed 13-D profile.  Mining it as an explicit negative directly
            # contradicts the continuous POI target, so rely on the in-batch
            # target distribution for negatives under this contract.
            self.hard_neg_k = 0
        # Number of cross-city hard negatives to return per pair.
        self.hard_neg_n = max(1, int(hard_neg_n))
        # When pos_rule=="dominant_cat", each anchor keeps at least min_dom_pos
        # cross-city positives that share its dominant category: it first scans
        # the feature-sim top-k, then back-fills the most similar same-category
        # grids from the whole target city.
        self.min_dom_pos = max(1, int(min_dom_pos))
        self.dominant_boost_topk = max(1, int(dominant_boost_topk))
        self.rng = random.Random(seed)
        self.dominant_cats = {}
        for name, value in city_data.items():
            feat, _, _ = _city_parts(value)
            # 13-D POI profile; dominant category = argmax over first 13 dims.
            self.dominant_cats[name] = feat[:, :13].argmax(axis=1).astype(np.int64)
        self._build_positive_pairs()
        # Store city feature tensors for mixup
        self._city_name_list = list(city_data.keys())
        self._city_feat_cache = {}
        for name in self._city_name_list:
            feat, _, _ = _city_parts(city_data[name])
            self._city_feat_cache[name] = torch.from_numpy(feat).float()

    def _build_positive_pairs(self):
        self.pairs = []
        self.pos_weights = []
        # hard_neg_feats[idx] is a feature vector for one cross-city hard
        # negative (or None when none exists / mining disabled).
        self.hard_neg_feats = []
        self.hard_neg_grid_ids = []

        if len(self.city_names) == 1:
            city = self.city_names[0]
            feat, _, _ = _city_parts(self.city_data[city])
            norms = np.linalg.norm(feat, axis=1, keepdims=True)
            sim = feat @ feat.T / (norms * norms.T + 1e-8)
            dom = self.dominant_cats[city]
            for idx_a in range(len(feat)):
                topk = min(self.topk_feat_sim + 1, len(feat))
                top_indices = np.argsort(-sim[idx_a])[:topk]
                for idx_b in top_indices:
                    if idx_b != idx_a:
                        self.pairs.append((city, idx_a, city, idx_b))
                        self.pos_weights.append(1.0)
                        neg_feats, neg_ids = self._pick_hard_neg(
                            city, idx_a, city, sim[idx_a], dom
                        )
                        self.hard_neg_feats.append(neg_feats)
                        self.hard_neg_grid_ids.append(neg_ids)
            return

        for i, city_a in enumerate(self.city_names):
            feat_a, _, _ = _city_parts(self.city_data[city_a])
            for city_b in self.city_names[i + 1:]:
                feat_b, _, _ = _city_parts(self.city_data[city_b])
                sim_feat_a, sim_feat_b = feat_a, feat_b
                if self.pos_rule == "poi_profile":
                    sim_feat_a = feat_a[:, :13]
                    sim_feat_b = feat_b[:, :13]
                norm_a = sim_feat_a / (
                    np.linalg.norm(sim_feat_a, axis=1, keepdims=True) + 1e-8
                )
                norm_b = sim_feat_b / (
                    np.linalg.norm(sim_feat_b, axis=1, keepdims=True) + 1e-8
                )
                candidates = (
                    np.where(np.clip(feat_b[:, :13], 0.0, None).sum(axis=1) > 1e-8)[0]
                    if self.pos_rule == "poi_profile" else np.arange(len(feat_b))
                )
                # Never allocate a CA x target dense similarity matrix.
                for start in range(0, len(feat_a), 512):
                    end = min(start + 512, len(feat_a))
                    sim_block = norm_a[start:end] @ norm_b.T
                    for local, idx_a in enumerate(range(start, end)):
                        sim_row = sim_block[local]
                        if (self.pos_rule == "poi_profile" and
                                float(np.clip(feat_a[idx_a, :13], 0.0, None).sum()) <= 1e-8):
                            continue
                        topk = min(self.topk_feat_sim, len(candidates))
                        if topk == 0:
                            continue
                        scores = sim_row[candidates]
                        if topk < len(candidates):
                            selected = np.argpartition(-scores, topk - 1)[:topk]
                            selected = selected[np.argsort(-scores[selected])]
                        else:
                            selected = np.argsort(-scores)
                        order = candidates[selected]
                        if self.pos_rule == "dominant_cat":
                            dom_a = self.dominant_cats[city_a][idx_a]
                            dom_b = self.dominant_cats[city_b]
                            same = dom_b[order] == dom_a
                            if same.any():
                                order = order[same]
                            else:
                                cand = np.where(dom_b == dom_a)[0]
                                order = (np.array([cand[np.argmax(sim_row[cand])]])
                                         if len(cand) else np.array([], dtype=np.int64))
                            if len(order) < self.min_dom_pos:
                                used = set(int(value) for value in order)
                                remain = np.asarray([
                                    value for value in np.where(dom_b == dom_a)[0]
                                    if int(value) not in used
                                ], dtype=np.int64)
                                if len(remain):
                                    remain = remain[np.argsort(-sim_row[remain])]
                                    order = np.concatenate((
                                        order, remain[:self.min_dom_pos - len(order)]
                                    ))
                        for idx_b in order:
                            self.pairs.append((city_a, idx_a, city_b, int(idx_b)))
                            self.pos_weights.append(
                                max(0.05, float(sim_row[idx_b]))
                                if self.pos_rule == "poi_profile" else 1.0
                            )
                            neg_feats, neg_ids = self._pick_hard_neg(
                                city_a, idx_a, city_b, sim_row,
                                self.dominant_cats[city_b]
                            )
                            self.hard_neg_feats.append(neg_feats)
                            self.hard_neg_grid_ids.append(neg_ids)

    def _pick_hard_neg(self, city_a, idx_a, city_b, sim_row, dom_b):
        """Return up to hard_neg_n list entries of hard negative features.

        Hard negatives are the top-N most feature-similar grids in the target
        city whose dominant category differs from the anchor. Returns a list of
        feature vectors, padded with None up to hard_neg_n.
        """
        n = self.hard_neg_n
        if not self.hard_neg_k or n <= 0:
            return [None] * n, [None] * n
        dom_a = self.dominant_cats[city_a][idx_a]
        different = dom_b != dom_a
        if not different.any():
            return [None] * n, [None] * n
        cand_idx = np.where(different)[0]
        k = min(self.hard_neg_k, len(cand_idx))
        top = cand_idx[np.argsort(-sim_row[cand_idx])[:k]]
        feats, grid_ids, _ = _city_parts(self.city_data[city_b])
        out = [feats[t].copy() for t in top[:n]]
        out_ids = [int(grid_ids[t]) for t in top[:n]]
        out += [None] * (n - len(out))
        out_ids += [None] * (n - len(out_ids))
        return out, out_ids

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        city_a, idx_a, city_b, idx_b = self.pairs[idx]
        feat_a, grid_ids_a, positions_a = _city_parts(self.city_data[city_a])
        feat_b, grid_ids_b, positions_b = _city_parts(self.city_data[city_b])
        neg_list = self.hard_neg_feats[idx]
        neg_id_list = self.hard_neg_grid_ids[idx]
        return {
            "anchor_feat": torch.from_numpy(feat_a[idx_a]).float(),
            "pos_feat": torch.from_numpy(feat_b[idx_b]).float(),
            "anchor_dom": int(self.dominant_cats[city_a][idx_a]),
            "pos_dom": int(self.dominant_cats[city_b][idx_b]),
            "pos_weight": float(self.pos_weights[idx]),
            "neg_feats": [
                torch.from_numpy(f).float() if f is not None else None
                for f in neg_list
            ],
            "neg_ids": neg_id_list,
            "anchor_id": torch.tensor(int(grid_ids_a[idx_a]), dtype=torch.long),
            "pos_id": torch.tensor(int(grid_ids_b[idx_b]), dtype=torch.long),
            "anchor_position": (
                torch.from_numpy(positions_a[idx_a] % 32 + 4).long()
                if positions_a is not None else None
            ),
            "pos_position": (
                torch.from_numpy(positions_b[idx_b] % 32 + 4).long()
                if positions_b is not None else None
            ),
            "city_a": city_a,
            "city_b": city_b,
            "anchor_city": self.city_indices[city_a],
            "pos_city": self.city_indices[city_b],
        }


def collate_multi_contrastive(batch, mixup_alpha=0.0, city_feat_cache=None):
    """Collate with optional cross-city mixup augmentation.

    When mixup_alpha > 0, randomly mix POI features (first 14 dims) between
    anchor and a random different city's feature, with lambda ~ Beta(alpha, alpha).
    """
    anchor_feat_list = [b["anchor_feat"] for b in batch]
    pos_feat_list = [b["pos_feat"] for b in batch]

    if mixup_alpha > 0 and city_feat_cache is not None:
        city_names = list(city_feat_cache.keys())
        if len(city_names) >= 2:
            mixed_anchor = []
            mixed_pos = []
            for b in batch:
                other_city = random.choice([c for c in city_names if c != b["city_a"]])
                other_feat = city_feat_cache[other_city]
                other_idx = torch.randint(0, other_feat.size(0), (1,)).item()
                other_feat_single = other_feat[other_idx]

                lambda_ = np.random.beta(mixup_alpha, mixup_alpha)
                anchor_feat = b["anchor_feat"].clone()
                pos_feat = b["pos_feat"].clone()
                anchor_feat[:14] = lambda_ * anchor_feat[:14] + (1 - lambda_) * other_feat_single[:14].to(anchor_feat.dtype)
                pos_feat[:14] = lambda_ * pos_feat[:14] + (1 - lambda_) * other_feat_single[:14].to(pos_feat.dtype)
                mixed_anchor.append(anchor_feat)
                mixed_pos.append(pos_feat)
            anchor_feat_list = mixed_anchor
            pos_feat_list = mixed_pos

    neg_feats = []
    neg_ids = []
    neg_valid = []
    n_neg_max = 0
    for b in batch:
        nfl = b.get("neg_feats") or [None]
        n_neg_max = max(n_neg_max, len(nfl))
    for b in batch:
        nfl = b.get("neg_feats") or [None]
        nil = b.get("neg_ids") or [None] * len(nfl)
        row = []
        row_ids = []
        rowv = []
        for f, grid_id in zip(nfl, nil):
            if f is not None:
                row.append(f)
                row_ids.append(int(grid_id))
                rowv.append(True)
            else:
                row.append(torch.zeros_like(b["anchor_feat"]))
                row_ids.append(0)
                rowv.append(False)
        # pad to the max width of this batch
        while len(row) < n_neg_max:
            row.append(torch.zeros_like(b["anchor_feat"]))
            row_ids.append(0)
            rowv.append(False)
        neg_feats.append(torch.stack(row))
        neg_ids.append(row_ids)
        neg_valid.append(rowv)

    result = {
        "anchor_feat": torch.stack(anchor_feat_list),
        "pos_feat": torch.stack(pos_feat_list),
        "anchor_dom": torch.tensor([b["anchor_dom"] for b in batch], dtype=torch.long),
        "pos_dom": torch.tensor([b["pos_dom"] for b in batch], dtype=torch.long),
        "pos_weight": torch.tensor(
            [b.get("pos_weight", 1.0) for b in batch], dtype=torch.float32
        ),
        "neg_feat": torch.stack(neg_feats),          # [B, N_neg, d_feat]
        "neg_id": torch.tensor(neg_ids, dtype=torch.long),
        "neg_valid": torch.tensor(neg_valid, dtype=torch.bool),  # [B, N_neg]
        "anchor_id": torch.stack([b["anchor_id"] for b in batch]),
        "pos_id": torch.stack([b["pos_id"] for b in batch]),
        "anchor_city": torch.tensor(
            [b["anchor_city"] for b in batch], dtype=torch.long
        ),
        "pos_city": torch.tensor(
            [b["pos_city"] for b in batch], dtype=torch.long
        ),
    }
    if all(b.get("anchor_position") is not None for b in batch):
        result["anchor_position"] = torch.stack([
            b["anchor_position"] for b in batch
        ])
        result["pos_position"] = torch.stack([
            b["pos_position"] for b in batch
        ])
    return result


def poi_profile_target_distribution(
        anchor_poi, candidate_poi, target_temperature=0.10,
        allowed_mask=None):
    """Return row-normalized continuous targets from 13-D POI JSD.

    Every non-empty candidate can receive target mass, with exponentially more
    mass assigned to a closer profile.  Empty rows fall back to their diagonal
    pair so a sparse mini-batch never produces NaNs.
    """
    if target_temperature <= 0:
        raise ValueError("POI target temperature must be positive")
    anchor = anchor_poi.clamp(min=0.0)
    candidate = candidate_poi.clamp(min=0.0)
    anchor_mass = anchor.sum(dim=-1, keepdim=True)
    candidate_mass = candidate.sum(dim=-1, keepdim=True)
    anchor = anchor / anchor_mass.clamp(min=1e-8)
    candidate = candidate / candidate_mass.clamp(min=1e-8)
    midpoint = 0.5 * (anchor[:, None, :] + candidate[None, :, :])
    anchor_expanded = anchor[:, None, :]
    candidate_expanded = candidate[None, :, :]
    anchor_kl = torch.where(
        anchor_expanded > 0,
        anchor_expanded * (
            torch.log(anchor_expanded.clamp(min=1e-12))
            - torch.log(midpoint.clamp(min=1e-12))
        ),
        torch.zeros_like(midpoint),
    ).sum(dim=-1)
    candidate_kl = torch.where(
        candidate_expanded > 0,
        candidate_expanded * (
            torch.log(candidate_expanded.clamp(min=1e-12))
            - torch.log(midpoint.clamp(min=1e-12))
        ),
        torch.zeros_like(midpoint),
    ).sum(dim=-1)
    jsd = 0.5 * (anchor_kl + candidate_kl)
    target_logits = -jsd / float(target_temperature)
    candidate_valid = candidate_mass.squeeze(-1) > 1e-8
    target_logits = target_logits.masked_fill(
        ~candidate_valid.unsqueeze(0), -torch.inf
    )
    if allowed_mask is not None:
        allowed_mask = allowed_mask.to(
            device=target_logits.device, dtype=torch.bool
        )
        if allowed_mask.shape != target_logits.shape:
            raise ValueError("POI target allowed mask has the wrong shape")
        target_logits = target_logits.masked_fill(~allowed_mask, -torch.inf)
    invalid_rows = (
        (anchor_mass.squeeze(-1) <= 1e-8)
        | ~torch.isfinite(target_logits).any(dim=1)
    )
    if invalid_rows.any():
        target_logits = target_logits.clone()
        target_logits[invalid_rows] = -torch.inf
        diagonal = torch.arange(
            target_logits.size(0), device=target_logits.device
        ).clamp(max=target_logits.size(1) - 1)
        target_logits[invalid_rows, diagonal[invalid_rows]] = 0.0
    return torch.softmax(target_logits, dim=-1)


def info_nce_loss_uniform(anchor_emb, pos_emb, temperature=0.07,
                          uniformity_weight=0.1,
                          neg_emb=None, neg_valid=None,
                          anchor_dom=None, pos_dom=None,
                          positive_weight=None,
                          anchor_poi=None, pos_poi=None,
                          cross_city_mask=None,
                          poi_target_temperature=0.10,
                          in_batch_cat_mask=True):
    """InfoNCE with optional explicit cross-city hard negatives.

    neg_emb: [B, N_neg, d] (or [B, d] when N_neg==1) embeddings of hard
        negatives mined by the dataset. Columns are concatenated after the
        in-batch negatives. Rows where neg_valid is False are ignored by
        masking the extra logits to a very negative value.

    anchor_dom / pos_dom: [B] integer dominant POI category for the anchor
        and positive rows. When in_batch_cat_mask is True, the in-batch
        negative logits of row i (Sim(i,j), j != i) are masked out whenever
        anchor_dom[i] == pos_dom[j]: those rows share the same dominant
        category and are semantically candidates for positives, so treating
        them as hard negatives would erase the category structure we want to
        learn. This is the key structural fix that stops infoNCE from pushing
        same-category grids apart across cities.  When anchor_poi and pos_poi
        are supplied, a continuous multi-positive target replaces this coarse
        mask and the loss is made bidirectional.
    """
    batch_size = anchor_emb.size(0)
    anchor_emb = F.normalize(anchor_emb, p=2, dim=-1)
    pos_emb = F.normalize(pos_emb, p=2, dim=-1)
    in_batch_sim = anchor_emb @ pos_emb.T / temperature
    sim = in_batch_sim
    use_poi_targets = anchor_poi is not None and pos_poi is not None
    if (not use_poi_targets and in_batch_cat_mask
            and anchor_dom is not None and pos_dom is not None):
        anchor_dom = anchor_dom.to(anchor_emb.device)
        pos_dom = pos_dom.to(anchor_emb.device)
        # [B, B]: row i (anchor), col j (in-batch pos row). Mask the diagonal
        # anyway (it is the positive itself) plus any same-category offsets.
        cat_equal = (anchor_dom[:, None] == pos_dom[None, :])
        eye = torch.eye(batch_size, dtype=torch.bool, device=anchor_emb.device)
        mask = cat_equal & ~eye
        sim = sim.masked_fill(mask, -1e9)
    hard_neg_dim = 0
    if neg_emb is not None:
        ne = F.normalize(neg_emb, p=2, dim=-1)
        is_single = ne.dim() == 2
        if ne.dim() == 2:
            ne = ne.unsqueeze(1)
        hard_neg_dim = ne.size(1)
        neg_sim = torch.bmm(anchor_emb.unsqueeze(1), ne.transpose(1, 2)).squeeze(1) / temperature
        # bmm -> [B, N_neg]
        if neg_valid is not None:
            neg_valid = neg_valid.to(anchor_emb.device)
            if is_single:
                # neg_valid is [B] bool for the single-column case.
                neg_sim = neg_sim.masked_fill(~neg_valid.unsqueeze(1), -1e9)
            else:
                # neg_valid is [B, N_neg] bool.
                neg_sim = neg_sim.masked_fill(~neg_valid, -1e9)
        sim = torch.cat([sim, neg_sim], dim=1)
    if use_poi_targets:
        anchor_poi = anchor_poi.to(anchor_emb.device)
        pos_poi = pos_poi.to(anchor_emb.device)
        forward_targets = poi_profile_target_distribution(
            anchor_poi, pos_poi, poi_target_temperature,
            allowed_mask=cross_city_mask,
        )
        forward_loss = -(
            forward_targets * F.log_softmax(sim, dim=1)[:, :batch_size]
        ).sum(dim=1)
        reverse_targets = poi_profile_target_distribution(
            pos_poi, anchor_poi, poi_target_temperature,
            allowed_mask=(
                cross_city_mask.T if cross_city_mask is not None else None
            ),
        )
        reverse_loss = -(
            reverse_targets * F.log_softmax(in_batch_sim.T, dim=1)
        ).sum(dim=1)
        row_loss = 0.5 * (forward_loss + reverse_loss)
    else:
        labels = torch.arange(batch_size, device=anchor_emb.device)
        row_loss = F.cross_entropy(sim, labels, reduction="none")
    if positive_weight is not None and not use_poi_targets:
        positive_weight = positive_weight.to(
            device=row_loss.device, dtype=row_loss.dtype
        ).clamp(min=0.0)
        positive_weight = positive_weight / positive_weight.mean().clamp(min=1e-8)
        row_loss = row_loss * positive_weight
    loss_contrastive = row_loss.mean()
    if uniformity_weight > 0:
        all_emb = torch.cat([anchor_emb, pos_emb], dim=0)
        sim_matrix = all_emb @ all_emb.T
        eye = torch.eye(2 * batch_size, dtype=torch.bool, device=all_emb.device)
        off_mask = ~eye
        # Class-aware uniformity: dispersion pushes the embedding shell apart
        # (anti-collapse) while an intra-class cohesion term pulls anchors and
        # positives that share a dominant category closer together. The two
        # terms compete so the shell stays spread out without erasing category
        # structure.
        dispersion = torch.exp(-sim_matrix[off_mask]).mean().log()
        intra_sim = torch.zeros((), device=all_emb.device)
        if use_poi_targets:
            all_poi = torch.cat([anchor_poi, pos_poi], dim=0)
            poi_targets = poi_profile_target_distribution(
                all_poi, all_poi, poi_target_temperature
            )
            poi_targets = poi_targets.masked_fill(eye, 0.0)
            if poi_targets.sum() > 0:
                intra_sim = (
                    poi_targets * sim_matrix
                ).sum() / poi_targets.sum().clamp(min=1e-8)
        elif anchor_dom is not None and pos_dom is not None:
            all_dom = torch.cat([anchor_dom, pos_dom], dim=0).to(all_emb.device)
            intra_mask = off_mask & (all_dom[:, None] == all_dom[None, :])
            if intra_mask.any():
                intra_sim = sim_matrix[intra_mask].mean()
        loss_uniformity = uniformity_weight * (dispersion - intra_sim)
    else:
        loss_uniformity = torch.tensor(0.0, device=anchor_emb.device)
    return loss_contrastive, loss_uniformity, hard_neg_dim


def metric_checkpoint_is_better(
        selection_value, best_selection, monitor_loss, best_selection_loss,
        selection_min_delta=0.0, loss_min_delta=0.0):
    """Rank a retrieval checkpoint, using validation loss to break plateaus.

    Retrieval R@K values are discrete and commonly remain identical across
    several evaluations.  Treat values within ``selection_min_delta`` as a
    tie and keep the lower-loss checkpoint instead of permanently preserving
    the first epoch that reached the plateau.
    """
    if selection_min_delta < 0 or loss_min_delta < 0:
        raise ValueError("checkpoint min_delta values must be non-negative")
    if not math.isfinite(selection_value) or not math.isfinite(monitor_loss):
        return False
    primary_improvement = (
        selection_value > best_selection + selection_min_delta
    )
    tied_metric = (
        selection_value >= best_selection - selection_min_delta
        and selection_value <= best_selection + selection_min_delta
    )
    loss_tiebreak = (
        monitor_loss < best_selection_loss - loss_min_delta
    )
    return primary_improvement or (tied_metric and loss_tiebreak)


def loss_early_stopping_step(
        monitor_loss, best_loss, wait, patience, min_delta=0.0):
    """Advance loss-based patience independently of checkpoint ranking."""
    if patience < 1:
        raise ValueError("early-stopping patience must be positive")
    if min_delta < 0:
        raise ValueError("early-stopping min_delta must be non-negative")
    improved = (
        math.isfinite(monitor_loss)
        and monitor_loss < best_loss - min_delta
    )
    if improved:
        return monitor_loss, 0, False, True
    next_wait = wait + 1
    return best_loss, next_wait, next_wait >= patience, False


def train_contrastive(model, train_loader, val_loader=None, nyc_val_loader=None,
                      device="cuda", epochs=200, lr=3e-4,
                      temperature=0.07, uniformity_weight=0.3,
                      patience=30, save_dir="checkpoints",
                      mixup_alpha=0.0, semantic_head_weight=0.0,
                      eval_fn=None, eval_every=5, dom_align_weight=0.0,
                      initial_best_loss=None, weight_decay=1e-5,
                      checkpoint_metadata=None,
                      poi_target_temperature=0.10,
                      selection_metric=None,
                      selection_min_delta=0.0,
                      early_stopping_min_delta=0.0,
                      use_poi_multi_positive=False):
    """
    Train with cross-city validation and mixup augmentation.

    Args:
        nyc_val_loader: NYC validation loader (used for cross-city early stopping)
        mixup_alpha: Beta distribution alpha for mixup (0 = disabled)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Checkpoint ranking and early stopping intentionally use separate state:
    # retrieval selects the artifact, while validation loss controls patience.
    # A saturated or temporarily flat retrieval metric therefore cannot stop
    # training after the first evaluation interval.
    starting_loss = (
        float(initial_best_loss)
        if initial_best_loss is not None else float("inf")
    )
    best_checkpoint_loss = starting_loss
    best_stop_loss = starting_loss
    best_selection = -float("inf")
    best_selection_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "nyc_val_loss": [], "retrieval_eval": []}

    has_semantic = semantic_head_weight > 0 and hasattr(model, "semantic_head") \
        and model.semantic_head is not None
    if selection_metric is not None and (eval_fn is None or eval_every <= 0):
        raise ValueError(
            "selection_metric requires eval_fn and a positive eval_every"
        )

    def save_checkpoint(epoch, monitor_loss, metric_value=None):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": monitor_loss,
        }
        if selection_metric is not None:
            checkpoint["selection_metric"] = selection_metric
            checkpoint["selection_value"] = metric_value
        if checkpoint_metadata is not None:
            checkpoint["config"] = dict(checkpoint_metadata)
        torch.save(
            checkpoint,
            os.path.join(save_dir, "grid_transformer_best.pth")
        )

    def compute_batch_loss(anchor_feat, pos_feat, neg_feat, neg_valid,
                           neg_id=None,
                           anchor_id=None, pos_id=None,
                           anchor_position=None, pos_position=None,
                           anchor_dom=None, pos_dom=None,
                           anchor_city=None, pos_city=None,
                           positive_weight=None):
        # Use each row's real per-city grid id (0..899) for positional
        # encoding so training positions match inference (which passes the
        # true grid coordinates). Falling back to arange keeps callers that do
        # not provide ids working.
        if anchor_id is not None:
            ids = anchor_id.to(device)
        else:
            ids = torch.arange(anchor_feat.size(0), device=device)
        if has_semantic:
            anchor_emb, anchor_logits = model.forward_with_semantic(
                anchor_feat, ids, positions=(
                    anchor_position.to(device) if anchor_position is not None else None
                )
            )
        else:
            anchor_emb = model(
                anchor_feat, ids, positions=(
                    anchor_position.to(device) if anchor_position is not None else None
                )
            )
            anchor_logits = None
        if pos_id is not None:
            pos_ids = pos_id.to(device)
        else:
            pos_ids = torch.arange(pos_feat.size(0), device=device)
        if has_semantic:
            pos_emb, pos_logits = model.forward_with_semantic(
                pos_feat, pos_ids, positions=(
                    pos_position.to(device) if pos_position is not None else None
                )
            )
        else:
            pos_emb = model(
                pos_feat, pos_ids, positions=(
                    pos_position.to(device) if pos_position is not None else None
                )
            )
            pos_logits = None
        neg_emb = None
        if neg_feat is not None and neg_feat.numel() > 0:
            # neg_feat: [B, N_neg, d_feat]; encode all rows, then reshape
            B, Nneg, _ = neg_feat.shape
            flat = neg_feat.reshape(-1, neg_feat.size(-1))
            if neg_id is not None:
                flat_ids = neg_id.to(device).reshape(-1)
            else:
                flat_ids = torch.arange(flat.size(0), device=device)
            flat_emb = model(flat, flat_ids)
            neg_emb = flat_emb.view(B, Nneg, -1)
        loss_c, loss_u, n_hard = info_nce_loss_uniform(
            anchor_emb, pos_emb, temperature, uniformity_weight,
            neg_emb=neg_emb, neg_valid=neg_valid,
            anchor_dom=anchor_dom, pos_dom=pos_dom,
            positive_weight=positive_weight,
            anchor_poi=(
                anchor_feat[:, :13] if use_poi_multi_positive else None
            ),
            pos_poi=(pos_feat[:, :13] if use_poi_multi_positive else None),
            cross_city_mask=(
                anchor_city.to(device)[:, None]
                != pos_city.to(device)[None, :]
                if (use_poi_multi_positive and anchor_city is not None
                    and pos_city is not None) else None
            ),
            poi_target_temperature=poi_target_temperature,
            in_batch_cat_mask=True)
        loss = loss_c + loss_u
        sem_loss = torch.tensor(0.0, device=device)
        if has_semantic and anchor_logits is not None:
            # Soft 13-D profile CE preserves the real mixed-POI structure
            # instead of forcing a coarse dominant-category one-hot.
            anchor_target = anchor_feat[:, :13].clamp(min=0.0)
            anchor_target = anchor_target / anchor_target.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-8)
            pos_target = pos_feat[:, :13].clamp(min=0.0)
            pos_target = pos_target / pos_target.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-8)
            anchor_ce = -(
                anchor_target * F.log_softmax(anchor_logits, dim=-1)
            ).sum(dim=1)
            pos_ce = -(
                pos_target * F.log_softmax(pos_logits, dim=-1)
            ).sum(dim=1)
            soft_ce = 0.5 * (anchor_ce.mean() + pos_ce.mean())
            sem_loss = soft_ce
            loss = loss + semantic_head_weight * sem_loss
        dom_align = torch.tensor(0.0, device=device)
        if dom_align_weight > 0:
            ae = F.normalize(anchor_emb, dim=-1)
            pe = F.normalize(pos_emb, dim=-1)
            dom_align = (1.0 - (ae * pe).sum(dim=-1)).mean()
            loss = loss + dom_align_weight * dom_align
        return loss, loss_c, loss_u, sem_loss, dom_align, n_hard

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            anchor_feat = batch["anchor_feat"].to(device)
            pos_feat = batch["pos_feat"].to(device)
            neg_feat = batch["neg_feat"].to(device)
            neg_valid = batch["neg_valid"].to(device)
            loss, _, _, _, _, _ = compute_batch_loss(
                anchor_feat, pos_feat, neg_feat, neg_valid,
                neg_id=batch.get("neg_id"),
                anchor_id=batch.get("anchor_id"), pos_id=batch.get("pos_id"),
                anchor_position=batch.get("anchor_position"),
                pos_position=batch.get("pos_position"),
                anchor_dom=batch.get("anchor_dom"), pos_dom=batch.get("pos_dom"),
                anchor_city=batch.get("anchor_city"), pos_city=batch.get("pos_city"),
                positive_weight=batch.get("pos_weight"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= max(1, len(train_loader))
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    anchor_feat = batch["anchor_feat"].to(device)
                    pos_feat = batch["pos_feat"].to(device)
                    neg_feat = batch["neg_feat"].to(device)
                    neg_valid = batch["neg_valid"].to(device)
                    loss, _, _, _, _, _ = compute_batch_loss(
                        anchor_feat, pos_feat, neg_feat, neg_valid,
                        neg_id=batch.get("neg_id"),
                        anchor_id=batch.get("anchor_id"), pos_id=batch.get("pos_id"),
                        anchor_position=batch.get("anchor_position"),
                        pos_position=batch.get("pos_position"),
                        anchor_dom=batch.get("anchor_dom"), pos_dom=batch.get("pos_dom"),
                        anchor_city=batch.get("anchor_city"), pos_city=batch.get("pos_city"),
                        positive_weight=batch.get("pos_weight"))
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))
            history["val_loss"].append(val_loss)

        # Cross-city validation: compute NYC val_loss for early stopping
        if nyc_val_loader is not None:
            model.eval()
            nyc_val_loss = 0.0
            with torch.no_grad():
                for batch in nyc_val_loader:
                    anchor_feat = batch["anchor_feat"].to(device)
                    pos_feat = batch["pos_feat"].to(device)
                    neg_feat = batch["neg_feat"].to(device)
                    neg_valid = batch["neg_valid"].to(device)
                    loss, _, _, _, _, _ = compute_batch_loss(
                        anchor_feat, pos_feat, neg_feat, neg_valid,
                        neg_id=batch.get("neg_id"),
                        anchor_id=batch.get("anchor_id"), pos_id=batch.get("pos_id"),
                        anchor_position=batch.get("anchor_position"),
                        pos_position=batch.get("pos_position"),
                        anchor_dom=batch.get("anchor_dom"), pos_dom=batch.get("pos_dom"),
                        anchor_city=batch.get("anchor_city"), pos_city=batch.get("pos_city"),
                        positive_weight=batch.get("pos_weight"))
                    nyc_val_loss += loss.item()
            if len(nyc_val_loader) > 0:
                nyc_val_loss /= len(nyc_val_loader)
                history["nyc_val_loss"].append(nyc_val_loss)
                monitor_loss = nyc_val_loss  # Use NYC val_loss for early stopping
            else:
                # Empty NYC loader: fall back to the primary validation metric.
                nyc_val_loss = float("nan")
                history["nyc_val_loss"].append(nyc_val_loss)
                monitor_loss = history["val_loss"][-1] if val_loader is not None else history["train_loss"][-1]
        elif val_loader is not None:
            monitor_loss = history["val_loss"][-1]
        else:
            monitor_loss = history["train_loss"][-1]
        if epoch % 10 == 0:
            val_str = ""
            if val_loader is not None:
                val_str += f", val_loss={val_loss:.4f}"
            if nyc_val_loader is not None:
                val_str += f", nyc_val_loss={nyc_val_loss:.4f}"
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}{val_str}")

        eval_metrics = None
        if (eval_fn is not None and eval_every > 0
                and (epoch % eval_every == 0 or epoch == epochs)):
            try:
                eval_metrics = eval_fn(model)
                history["retrieval_eval"].append({"epoch": epoch, **eval_metrics})
                r_str = ", ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items() if isinstance(v, float))
                print(f"    [retrieval] epoch {epoch}: {r_str}")
            except Exception as _e:  # pragma: no cover - eval should not kill training
                if selection_metric is not None:
                    raise RuntimeError(
                        "retrieval checkpoint selection failed"
                    ) from _e
                print(f"    [retrieval] epoch {epoch} failed: {_e}")

        if selection_metric is not None:
            if eval_metrics is not None:
                if selection_metric not in eval_metrics:
                    raise KeyError(
                        "retrieval selection metric %r is missing"
                        % selection_metric
                    )
                selection_value = float(eval_metrics[selection_metric])
                if metric_checkpoint_is_better(
                        selection_value,
                        best_selection,
                        monitor_loss,
                        best_selection_loss,
                        selection_min_delta=selection_min_delta,
                        loss_min_delta=early_stopping_min_delta):
                    best_selection = max(best_selection, selection_value)
                    best_selection_loss = monitor_loss
                    save_checkpoint(
                        epoch, monitor_loss, metric_value=selection_value
                    )
        elif monitor_loss < best_checkpoint_loss - early_stopping_min_delta:
            best_checkpoint_loss = monitor_loss
            save_checkpoint(epoch, monitor_loss)

        best_stop_loss, wait, should_stop, _ = loss_early_stopping_step(
            monitor_loss,
            best_stop_loss,
            wait,
            patience,
            min_delta=early_stopping_min_delta,
        )

        if should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore the best checkpoint seen during this run so downstream latent
    # extraction uses the best alignment, not the final-step parameters.
    best_path = os.path.join(save_dir, "grid_transformer_best.pth")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state_dict"], strict=False)

    return model, history
