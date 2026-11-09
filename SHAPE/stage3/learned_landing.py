"""Source-only learning of rank landing strengths and residential mixtures.

The tokenizer and landing head stay frozen. Training conditions on the observed
source coarse opportunity ring; generation still samples that ring using the
unchanged source statistics. No target trajectory is an input to this trainer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = "shape_learned_rank_landing_v1"
EPS = 1e-6


class LearnedLanding(nn.Module):
    """Three positive strengths, one density mixture and one gate per code."""

    def __init__(self, gates, density_mix=0.625):
        super().__init__()
        gates = torch.as_tensor(gates, dtype=torch.float32)
        if gates.ndim != 1 or not len(gates) or not torch.isfinite(gates).all() \
                or torch.any((gates <= 0) | (gates >= 1)):
            raise ValueError("initial residential gates must be strictly between 0 and 1")
        if not 0 < density_mix < 1:
            raise ValueError("initial density mix must be strictly between 0 and 1")
        self.raw_strengths = nn.Parameter(torch.log(torch.expm1(torch.tensor([1., 1., .8]))))
        self.density_logit = nn.Parameter(torch.tensor(math.log(density_mix / (1 - density_mix))))
        self.gate_logits = nn.Parameter(torch.logit(gates))

    def strengths(self):
        # beta, alpha, opportunity coefficient (inverse old distance_tau).
        return F.softplus(self.raw_strengths)

    def attraction_table(self, poi, density, diversity, categories):
        a = self.density_logit.sigmoid()
        # Preserve the two smoothing additions of the legacy residential path.
        residential = (a * density + (1 - a) * diversity).clamp(min=0) + 2 * EPS
        residential = residential / residential.sum()
        ordinary = categories @ poi.T
        gate = self.gate_logits.sigmoid().unsqueeze(1)
        table = gate * residential.unsqueeze(0) + (1 - gate) * ordinary + EPS
        return table / table.mean(dim=1, keepdim=True)

    def export_values(self):
        beta, alpha, rank = self.strengths().detach().cpu().tolist()
        return dict(revisit_beta=beta, attraction_strength=alpha,
                    opportunity_coefficient=rank, distance_tau=1 / rank,
                    density_mix=float(self.density_logit.sigmoid().detach().cpu()),
                    residential_gates=self.gate_logits.sigmoid().detach().cpu().tolist())


def attraction_inputs(raw, group, categories, device="cpu"):
    """Map-only features with exactly the baseline POI/richness convention."""
    raw = np.maximum(np.asarray(raw, dtype=np.float64), 0)
    group = np.asarray(group, dtype=np.float64)
    categories = np.asarray(categories, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 20 or group.shape != raw.shape \
            or categories.ndim != 2 or categories.shape[1] != 13 \
            or not all(np.isfinite(a).all() for a in (raw, group, categories)):
        raise ValueError("invalid static attraction inputs")
    poi = raw[:, :13] + EPS
    poi /= poi.sum(axis=0, keepdims=True)
    proportions = np.square(group[:, :13])
    proportions /= proportions.sum(axis=1, keepdims=True).clip(min=1e-12)
    diversity = -(proportions * np.log(proportions.clip(min=1e-12))).sum(axis=1) / math.log(13)
    return tuple(torch.as_tensor(a, dtype=torch.float32, device=device)
                 for a in (poi, group[:, 13], diversity, categories))


def conditional_logits(attribute, rank_log_density, memory_counts, attraction,
                       strengths, coarse_bins, gold, temperature=.75):
    """Same within-ring distribution as the existing sampler, without sampling.

    The sampler's attraction log-normalizer is constant inside the conditioned
    ring and cancels from softmax. Dropping that constant avoids undefined
    gradients for empty rings without changing the conditional probabilities.
    """
    beta, alpha, rank = strengths.unbind()
    logits = attribute + rank * rank_log_density + beta * torch.log1p(memory_counts) \
        + alpha * torch.log(attraction.clamp(min=EPS))
    gold_ring = coarse_bins.gather(1, gold[:, None])
    return (logits / temperature).masked_fill(coarse_bins != gold_ring, -torch.inf)


def teacher_forced_records(reference, projected, vocabulary):
    """Train and val transitions with causal, train-only cross-day memory.

    Val/test days never update training memory. For validation, earlier train
    days and the current day's observed prefix are available. The generator's
    same capacity/top-count/recency rule is used; anchors enter with count one.
    """
    from SHAPE.stage3.context_memory import _memory_anchor_ids
    vocabulary = np.asarray(vocabulary, dtype=np.int64)
    projected = np.asarray(projected, dtype=np.int64)
    positions = np.searchsorted(vocabulary, projected)
    if np.any(positions >= len(vocabulary)) or not np.array_equal(vocabulary[positions], projected):
        raise ValueError("projected observations outside source vocabulary")
    offsets = np.asarray(reference["day_offsets"], dtype=np.int64)
    users = np.asarray(reference["day_user_ids"], dtype=str)
    dates = np.asarray(reference["day_dates"], dtype=str)
    splits = np.asarray(reference["day_splits"], dtype=str)
    if len(offsets) != len(users) + 1 or offsets[0] != 0 or offsets[-1] != len(projected):
        raise ValueError("source daily offsets are not aligned")
    locations = defaultdict(set)
    for user, split, start, end in zip(users, splits, offsets[:-1], offsets[1:]):
        if split == "train":
            locations[user].update(projected[start:end].tolist())
    states, records = {}, {"train": [], "val": []}
    ordering = sorted(range(len(users)), key=lambda i: (users[i], dates[i]))
    for i in ordering:
        split, user = splits[i], users[i]
        if split not in records:
            continue
        start, end = int(offsets[i]), int(offsets[i+1])
        if not locations[user]:
            raise ValueError("source validation user lacks training history")
        if end <= start:
            continue
        anchors = _memory_anchor_ids(states.get(user), len(locations[user]))
        memory = {int(np.searchsorted(vocabulary, g)): 1 for g in anchors}
        memory[int(positions[start])] = 1
        for event in range(start + 1, end):
            keys = np.array(sorted(memory), dtype=np.int32)
            counts = np.array([memory[k] for k in keys], dtype=np.float32)
            records[split].append((event, int(positions[event-1]), int(positions[event]), keys, counts))
            target = int(positions[event])
            memory[target] = memory.get(target, 0) + 1
        if split == "train":
            state = states.setdefault(user, dict(counts=Counter(), last={}, clock=0))
            for grid in projected[start:end]:
                grid = int(grid)
                state["clock"] += 1
                state["counts"][grid] += 1
                state["last"][grid] = state["clock"]
            retained = _memory_anchor_ids(state, len(locations[user]))
            state["counts"] = Counter({g: state["counts"][g] for g in retained})
            state["last"] = {g: state["last"][g] for g in retained}
    return records


def _pack_records(records, reference, device):
    if not records:
        raise ValueError("source train and validation must contain transitions")
    width = max(len(r[3]) for r in records)
    positions = np.zeros((len(records), width), dtype=np.int32)
    counts = np.zeros((len(records), width), dtype=np.float32)
    for i, row in enumerate(records):
        positions[i, :len(row[3])] = row[3]
        counts[i, :len(row[4])] = row[4]
    events = np.array([r[0] for r in records], dtype=np.int64)
    arrays = dict(event=events, origin=np.array([r[1] for r in records]),
                  gold=np.array([r[2] for r in records]),
                  code=np.asarray(reference["codes"])[events, 0].astype(np.int64),
                  memory_positions=positions, memory_counts=counts)
    return {k: torch.as_tensor(a, device=device) for k, a in arrays.items()}


def load_learned_landing(path, *, source_codes, landing_checkpoint, source_grid, source_city):
    """Reject stale learned parameters rather than silently mix model versions."""
    from SHAPE.stage3.contracts import sha256_file
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != SCHEMA or state.get("source_city") != source_city:
        raise ValueError("learned landing schema/source city mismatch")
    for key, value in dict(source_codes=source_codes, landing_checkpoint=landing_checkpoint,
                           source_grid=source_grid).items():
        if state["bindings"].get(key) != sha256_file(value):
            raise ValueError("learned landing binding mismatch: " + key)
    model = LearnedLanding(np.full(state["codebook_size"], .5))
    model.load_state_dict(state["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite learned landing parameters")
    values = model.export_values()
    if min(values[k] for k in ("revisit_beta", "attraction_strength", "opportunity_coefficient")) <= 0:
        raise ValueError("invalid learned landing strength")
    return model.eval(), state


def fit(codes, landing_checkpoint, source_grid, output, *, device="cpu", seed=42,
        epochs=20, batch_size=512, lr=.01, patience=5, min_delta=1e-6):
    from SHAPE.gridspec import StaticGrid
    from SHAPE.stage3.contracts import load_npz, require_current_tokenizer_training, sha256_file, write_json
    from SHAPE.stage3.grid_projection import _project_nominal_ids_to_vocabulary
    from SHAPE.stage3.group_index import repair_group_index, load_v4_transport_provenance
    from SHAPE.stage3.landing_head import load_landing_runtime
    from SHAPE.stage3.opportunity_rank import SpatialRankIndex, static_poi_weights
    from SHAPE.stage3.prototype_semantics import fit_train_code_attraction_semantics
    from SHAPE.stage3.rank_semantics import fit_train_rank_distance_semantics, RANK_PIECEWISE_EDGES
    from SHAPE.stage3.static_features import _load_raw_table

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if epochs < 1 or batch_size < 1 or lr <= 0 or patience < 1:
        raise ValueError("invalid source training budget")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    reference = load_npz(codes)
    require_current_tokenizer_training(reference)
    city = str(reference["source_city"].item())
    loaded = load_landing_runtime(landing_checkpoint, None, source_grid, device, vocabulary_policy="all_emitted")
    head, table, _, centers, ids, _, frozen = loaded
    if frozen.get("source_city") != city or frozen.get("landing_mode") != "l2_attributes" \
            or frozen.get("codes_sha256") != sha256_file(codes) \
            or frozen.get("tokenizer_sha256") != str(reference["tokenizer_sha256"].item()):
        raise ValueError("source frozen head/codes/tokenizer mismatch")
    head.requires_grad_(False)
    static = StaticGrid.load(source_grid, city=city)
    projected, projection = _project_nominal_ids_to_vocabulary(reference["grid_id"], ids, static.spec)
    raw, _ = _load_raw_table(source_grid)
    group, _, _ = repair_group_index(raw, city,
        transport_provenance=load_v4_transport_provenance(source_grid, ids))
    weights, weight_audit = static_poi_weights(source_grid, ids)
    ranks = SpatialRankIndex(centers, weights, cache_rows=len(ids))
    print("Fitting source-train rank and attraction statistics", flush=True)
    semantics = fit_train_rank_distance_semantics(reference, projected, centers, ids, rank_index=ranks, weights=weights)
    roles = fit_train_code_attraction_semantics(reference, projected, raw, ids, residential_grid_table=group)
    gates = (roles["home_night_support"] + 1.) / (roles["support"] + 2.)
    scorer = LearnedLanding(gates).to(device)
    initial = scorer.export_values()
    attraction_args = attraction_inputs(raw, group, roles["category_probabilities"], device)
    print("Building past-only source train/val transitions", flush=True)
    records = teacher_forced_records(reference, projected, ids)
    datasets = {split: _pack_records(rows, reference, device) for split, rows in records.items()}
    del records
    # uint8 lookup is bounded to the source map; target generation remains lazy.
    bins = np.stack([np.searchsorted(RANK_PIECEWISE_EDGES[1:-1], ranks.row(i), side="right")
                     for i in range(len(ids))]).astype(np.uint8)
    bin_counts = np.stack([np.bincount(row, minlength=16) for row in bins])
    fine_bins = torch.as_tensor(bins, device=device)
    log_counts = torch.as_tensor(np.log(bin_counts.clip(min=1)), dtype=torch.float32, device=device)
    log_probability = torch.as_tensor(np.log(semantics["piecewise_probabilities"].clip(min=1e-12)), device=device)
    with torch.no_grad():
        h = np.asarray(reference["reconstructed_h"], dtype=np.float32)
        queries = torch.cat([F.normalize(head.encode_query(torch.as_tensor(h[i:i+8192], device=device)), dim=-1)
                             for i in range(0, len(h), 8192)])
        normalized_table = F.normalize(table, dim=-1)
        scale = head.logit_scale.exp().clamp(max=100).detach()
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    def epoch(split, training):
        data = datasets[split]
        n = len(data["gold"])
        order = rng.permutation(n) if training else np.arange(n)
        loss_sum = 0.
        grad_max = {name: 0. for name, _ in scorer.named_parameters()}
        for left in range(0, n, batch_size):
            idx = torch.as_tensor(order[left:left+batch_size], device=device)
            origin, code, gold = (data[k][idx].long() for k in ("origin", "code", "gold"))
            fine = fine_bins[origin].long()
            with torch.set_grad_enabled(training):
                attribute = scale * queries[data["event"][idx]] @ normalized_table.T
                rank_density = log_probability[code].gather(1, fine) - log_counts[origin].gather(1, fine)
                memory = torch.zeros_like(attribute).scatter_add_(1, data["memory_positions"][idx].long(), data["memory_counts"][idx])
                attraction = scorer.attraction_table(*attraction_args)[code]
                logits = conditional_logits(attribute, rank_density, memory, attraction,
                    scorer.strengths(), fine // 4, gold)
                loss = F.cross_entropy(logits, gold)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite source conditional NLL")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for name, parameter in scorer.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("missing/nonfinite learned parameter gradient: " + name)
                        grad_max[name] = max(grad_max[name], float(parameter.grad.detach().abs().max().cpu()))
                    optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(idx)
        return loss_sum / n, grad_max

    best_loss, _ = epoch("val", False)
    best_state, best_epoch, stale = copy.deepcopy(scorer.state_dict()), 0, 0
    history = [dict(epoch=0, val_nll=best_loss, values=initial)]
    print(json.dumps(history[-1]), flush=True)
    for index in range(1, epochs + 1):
        started = time.monotonic()
        train_loss, grads = epoch("train", True)
        validation, _ = epoch("val", False)
        improved = validation < best_loss - min_delta
        if improved:
            best_loss, best_epoch, best_state, stale = validation, index, copy.deepcopy(scorer.state_dict()), 0
        else:
            stale += 1
        row = dict(epoch=index, train_nll=train_loss, val_nll=validation, improved=improved,
                   gradient_max=grads, values=scorer.export_values(), seconds=time.monotonic()-started)
        history.append(row)
        write_json(output / "history.json", history)
        print(json.dumps({k: v for k, v in row.items() if k != "values"}), flush=True)
        if stale >= patience:
            break
    scorer.load_state_dict(best_state)
    bindings = {key: sha256_file(path) for key, path in dict(source_codes=codes,
        landing_checkpoint=landing_checkpoint, source_grid=source_grid).items()}
    state = dict(schema=SCHEMA, source_city=city, codebook_size=len(gates),
                 state_dict={k: v.cpu() for k, v in best_state.items()}, bindings=bindings,
                 best_epoch=best_epoch, val_nll=best_loss, training_seed=seed,
                 tokenizer_sha256=str(reference["tokenizer_sha256"].item()),
                 selection_temperature=.75, landing_temperature=1.,
                 rank_semantics_sha256=semantics["table_sha256"],
                 attraction_semantics_sha256=roles["table_sha256"])
    torch.save(state, output / "learned_landing.pth")
    report = dict(schema=SCHEMA, source_city=city, status="complete", bindings=bindings,
        training_seed=seed, best_epoch=best_epoch, epochs_run=len(history)-1,
        initial_val_nll=history[0]["val_nll"], selected_val_nll=best_loss,
        transition_counts={s: len(d["gold"]) for s, d in datasets.items()},
        initial_values=initial, selected_values=scorer.export_values(),
        trainable_parameter_count=sum(p.numel() for p in scorer.parameters()),
        frozen_head=True, target_trajectories_used=False, projection=projection,
        opportunity_weight_provenance=weight_audit,
        fit_configuration=dict(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, min_delta=min_delta),
        checkpoint_sha256=sha256_file(output / "learned_landing.pth"))
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--landing-checkpoint", required=True)
    parser.add_argument("--source-grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    args = vars(parser.parse_args(argv))
    fit(**args)


if __name__ == "__main__":
    main()
