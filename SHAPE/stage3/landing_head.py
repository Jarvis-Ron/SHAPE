"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from SHAPE.stage1.grid_text_encoder import load_grid_features
from SHAPE.gridspec import SparseLatentTable, StaticGrid, infer_city
from SHAPE.stage3.contracts import CODES_SCHEMA, load_npz, require_scalar, sha256_file, write_json


LANDING_CHECKPOINT_SCHEMA = "civic_phase3_landing_head_checkpoint_v4"


LANDING_EVENTS_SCHEMA = "civic_phase3_landing_events_v2"


LANDING_MODES = ("raw_m1", "l1_whitened_m1", "l2_attributes")


class LandingHead(nn.Module):
    """Map reconstructed H into shared M1 space, then retrieve a city table."""

    def __init__(self, input_dim=64, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.query = nn.Sequential(
            nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(), nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_query(self, h):
        return self.query(h.float())

    def score(self, h, grid_latents, score_mode="cosine"):
        query = self.encode_query(h)
        table = grid_latents.float()
        if score_mode == "cosine":
            query = F.normalize(query, dim=-1)
            table = F.normalize(table, dim=-1)
        elif score_mode != "dot":
            raise ValueError("score_mode must be cosine or dot")
        return self.logit_scale.exp().clamp(max=100.0) * query @ table.T

    def forward(self, h, grid_latents):
        return self.score(h, grid_latents, "cosine")

    def export_config(self):
        return {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim}


def _set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_canonical_grid_table(latents_path, grid_csv):
    """Return an emitted ID-keyed M1 table without dense ID scattering."""
    features, csv_grid_ids, _ = load_grid_features(str(grid_csv))
    order = np.argsort(csv_grid_ids)
    csv_grid_ids = np.asarray(csv_grid_ids, dtype=np.int64)[order]
    if str(latents_path).lower().endswith(".npz"):
        latent_table = SparseLatentTable.load(latents_path)
    else:
        latent_table = SparseLatentTable(csv_grid_ids, np.load(latents_path))
    if latent_table.embeddings.shape[1] != 64:
        raise RuntimeError("M1 grid latents must have shape [N, 64]")
    positions = np.searchsorted(csv_grid_ids, latent_table.grid_ids)
    safe = np.clip(positions, 0, max(0, len(csv_grid_ids) - 1))
    if not np.all((positions < len(csv_grid_ids)) &
                  (csv_grid_ids[safe] == latent_table.grid_ids)):
        raise RuntimeError("M1 latent IDs must be an emitted-grid subset")
    table = latent_table.embeddings
    grid_ids = latent_table.grid_ids
    return table, grid_ids, {
        "latents_path": str(latents_path),
        "latents_sha256": sha256_file(latents_path),
        "grid_csv": str(grid_csv),
        "grid_csv_sha256": sha256_file(grid_csv),
        "canonical_table_sha256": _array_sha256(table),
        "grid_ids_sha256": _array_sha256(grid_ids),
        "landing_vocabulary_rows": int(len(grid_ids)),
        "vocabulary_contract": "sorted_emitted_raw_int64_grid_ids",
    }


def load_canonical_grid_attributes(grid_csv):
    """Return homologous 20-D attributes and centers in sorted emitted order."""
    features, grid_ids, centers = load_grid_features(str(grid_csv))
    order = np.argsort(grid_ids)
    grid_ids = np.asarray(grid_ids, dtype=np.int64)[order]
    table = np.asarray(features, dtype=np.float32)[order]
    center_table = np.asarray(centers, dtype=np.float64)[order]
    if len(np.unique(grid_ids)) != len(grid_ids) or np.any(grid_ids < 0):
        raise RuntimeError("Attribute grid IDs must be unique non-negative int64")
    return table, center_table, grid_ids, {
        "grid_csv": str(grid_csv),
        "grid_csv_sha256": sha256_file(grid_csv),
        "canonical_attribute_sha256": _array_sha256(table),
        "canonical_center_sha256": _array_sha256(center_table),
        "grid_ids_sha256": _array_sha256(grid_ids),
        "landing_vocabulary_rows": int(len(grid_ids)),
    }


def _whiten_per_city(table):
    table = np.asarray(table, dtype=np.float32)
    mean = table.mean(axis=0)
    scale = table.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return ((table - mean) / scale).astype(np.float32), {
        "contract": "per_city_dimension_zscore",
        "mean": mean.tolist(), "scale": scale.tolist(),
    }


def fit_pooled_quantiles(grid_csvs, low=0.05, high=0.95):
    """Fit one robust 20-D normalization from pooled city grid attributes."""
    if not grid_csvs:
        raise RuntimeError("L2 needs at least one pooled-normalization grid CSV")
    pooled = np.concatenate([
        load_canonical_grid_attributes(path)[0] for path in grid_csvs
    ], axis=0)
    lower = np.quantile(pooled, float(low), axis=0).astype(np.float32)
    upper = np.quantile(pooled, float(high), axis=0).astype(np.float32)
    upper = np.where(upper - lower < 1e-6, lower + 1.0, upper)
    return {
        "contract": "pooled_city_q05_q95_clip_v1",
        "quantile_low": float(low), "quantile_high": float(high),
        "lower": lower.tolist(), "upper": upper.tolist(),
        "fit_grid_csv_sha256": [sha256_file(path) for path in grid_csvs],
    }


def apply_pooled_quantiles(table, normalization):
    lower = np.asarray(normalization["lower"], dtype=np.float32)
    upper = np.asarray(normalization["upper"], dtype=np.float32)
    return np.clip((np.asarray(table, dtype=np.float32) - lower) /
                   (upper - lower), 0.0, 1.0).astype(np.float32)


def _representation_table(mode, grid_latents, grid_csv, normalization=None,
                          *, vocabulary_policy="legacy"):
    if vocabulary_policy not in ("legacy", "all_emitted"):
        raise ValueError("unknown landing vocabulary policy")
    attributes, centers, attribute_ids, attribute_binding = \
        load_canonical_grid_attributes(grid_csv)
    if mode == "l2_attributes":
        if normalization is None:
            raise RuntimeError("L2 checkpoint is missing pooled quantile normalization")
        grid_ids = attribute_ids
        try:
            city_key = infer_city(grid_csv)
        except ValueError:
            city_key = None
        if city_key == "CA" and vocabulary_policy == "legacy":
            active_path = Path("dataset/CA_active_grids.csv")
            if not active_path.is_file():
                raise FileNotFoundError(
                    "CA L2 landing requires dataset/CA_active_grids.csv"
                )
            active_grid = StaticGrid.load(
                grid_csv, city="ca", active_ids_path=active_path
            )
            grid_ids = active_grid.anchor_ids()
            indices = np.searchsorted(attribute_ids, grid_ids)
            attributes, centers = attributes[indices], centers[indices]
        table = apply_pooled_quantiles(attributes, normalization)
        binding = {**attribute_binding, "representation": "20d_static_attributes",
                   "canonical_table_sha256": _array_sha256(table),
                   "grid_ids_sha256": _array_sha256(grid_ids),
                   "landing_vocabulary_rows": int(len(grid_ids)),
                   "vocabulary_contract": "sorted_emitted_raw_int64_grid_ids"}
        if vocabulary_policy != "legacy":
            binding["vocabulary_policy"] = vocabulary_policy
            binding["checkin_active_filter_used"] = False
    else:
        if grid_latents is None:
            raise RuntimeError("M1 landing requires --grid-latents")
        table, grid_ids, binding = load_canonical_grid_table(grid_latents, grid_csv)
        indices = np.searchsorted(attribute_ids, grid_ids)
        if not np.all(attribute_ids[indices] == grid_ids):
            raise RuntimeError("Landing latent IDs are absent from attribute table")
        attributes = attributes[indices]
        centers = centers[indices]
        if mode == "l1_whitened_m1":
            table, local_normalization = _whiten_per_city(table)
            binding = {**binding, "normalization": local_normalization,
                       "canonical_table_sha256": _array_sha256(table)}
        binding["representation"] = "64d_m1_latent"
    return table, attributes, centers, grid_ids, binding


def _validate_events(encoded, expected_city=None):
    schema = str(require_scalar(encoded, "schema"))
    if schema not in (CODES_SCHEMA, LANDING_EVENTS_SCHEMA):
        raise RuntimeError("Unsupported landing-event artifact schema: %s" % schema)
    city = str(require_scalar(encoded, "source_city"))
    if expected_city is not None and city != str(expected_city).lower():
        raise RuntimeError("Landing event city mismatch: %s != %s" % (city, expected_city))
    h = np.asarray(encoded["reconstructed_h"], dtype=np.float32)
    grid = np.asarray(encoded["grid_id"], dtype=np.int64)
    split = np.asarray(encoded["event_splits"], dtype=np.str_)
    if h.ndim != 2 or h.shape[1] != 64 or len(h) != len(grid) \
            or len(grid) != len(split):
        raise RuntimeError("Landing-event arrays have inconsistent dimensions")
    offsets = encoded.get("sequence_offsets", encoded.get("day_offsets"))
    if offsets is None:
        offsets = np.asarray([0, len(grid)], dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(grid) \
            or np.any(np.diff(offsets) < 0):
        raise RuntimeError("Landing event archive has invalid sequence offsets")
    return city, h, grid, split, offsets


def _split_arrays(encoded, split_name, expected_city=None):
    city, h, grid, split, offsets = _validate_events(encoded, expected_city)
    keep = (split == str(split_name)) & (grid >= 0)
    if not keep.any():
        raise RuntimeError("No valid %s %s landing events" % (city, split_name))
    previous = np.full(len(grid), -1, dtype=np.int64)
    for start, end in zip(offsets[:-1], offsets[1:]):
        start, end = int(start), int(end)
        if end - start > 1:
            previous[start + 1:end] = grid[start:end - 1]
    return h[keep], grid[keep], previous[keep]


def _map_split_to_vocabulary(features, raw_grid, previous, vocabulary_ids):
    vocabulary_ids = np.asarray(vocabulary_ids, dtype=np.int64)
    positions = np.searchsorted(vocabulary_ids, raw_grid)
    safe = np.clip(positions, 0, max(0, len(vocabulary_ids) - 1))
    present = (positions < len(vocabulary_ids)) & (vocabulary_ids[safe] == raw_grid)
    if not present.any():
        raise RuntimeError("No landing events belong to the emitted vocabulary")
    return features[present], positions[present].astype(np.int64), previous[present]


def _loader(features, targets, previous, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(targets),
                      torch.from_numpy(previous)),
        batch_size=int(batch_size), shuffle=bool(shuffle), generator=generator,
    )


def _epoch(model, table, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = count = correct1 = correct5 = 0
    for h, target, _ in loader:
        h, target = h.to(device), target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(h, table)
            loss = F.cross_entropy(logits, target)
            if training:
                loss.backward()
                optimizer.step()
        top5 = logits.topk(min(5, logits.shape[1]), dim=-1).indices
        loss_sum += float(loss.detach()) * len(target)
        count += len(target)
        correct1 += int((top5[:, 0] == target).sum())
        correct5 += int((top5 == target.unsqueeze(1)).any(dim=1).sum())
    return {"loss": loss_sum / max(1, count),
            "top1": correct1 / max(1, count),
            "top5": correct5 / max(1, count), "events": int(count)}


def train_landing_head(
    codes, grid_latents, grid_csv, output, device=torch.device("cpu"),
    epochs=20, batch_size=4096, hidden_dim=128, lr=1e-3,
    weight_decay=1e-5, patience=5, seed=42, mode="raw_m1",
    normalization_grid_csvs=None,
):
    """Train only on the source city bound into the daily-code archive."""
    _set_seed(seed)
    encoded = load_npz(codes)
    source_city = str(require_scalar(encoded, "source_city")).lower()
    train_h, train_grid, train_previous = _split_arrays(
        encoded, "train", source_city
    )
    val_h, val_grid, val_previous = _split_arrays(
        encoded, "val", source_city
    )
    mode = str(mode)
    if mode not in LANDING_MODES:
        raise ValueError("Unsupported landing mode: %s" % mode)
    normalization = None
    if mode == "l2_attributes":
        normalization = fit_pooled_quantiles(
            normalization_grid_csvs or [grid_csv]
        )
    table_np, _, _, vocabulary_ids, table_binding = _representation_table(
        mode, grid_latents, grid_csv, normalization
    )
    train_h, train_grid, train_previous = _map_split_to_vocabulary(
        train_h, train_grid, train_previous, vocabulary_ids
    )
    val_h, val_grid, val_previous = _map_split_to_vocabulary(
        val_h, val_grid, val_previous, vocabulary_ids
    )
    table = torch.from_numpy(table_np).to(device)
    model = LandingHead(train_h.shape[1], hidden_dim, table_np.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    train_loader = _loader(train_h, train_grid, train_previous, batch_size, True, seed)
    val_loader = _loader(val_h, val_grid, val_previous, batch_size, False, seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _epoch(model, table, train_loader, device, optimizer)
        val_metrics = _epoch(model, table, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print("Landing epoch %3d/%d train=%.4f val=%.4f top5=%.4f" % (
            epoch, epochs, train_metrics["loss"], val_metrics["loss"],
            val_metrics["top5"],
        ))
        if val_metrics["loss"] < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_metrics["loss"], epoch, 0
            state = {
                "schema": LANDING_CHECKPOINT_SCHEMA,
                "model_config": model.export_config(),
                "model_state_dict": model.state_dict(),
                "landing_mode": mode, "normalization": normalization,
                "source_city": source_city,
                "target_city_training_data_used": False,
                "tokenizer_sha256": str(require_scalar(encoded, "tokenizer_sha256")),
                "codes_sha256": sha256_file(codes),
                "source_grid_table": table_binding,
                "source_grid_ids": torch.from_numpy(vocabulary_ids),
                "training_config": {"epochs": int(epochs), "landing_mode": mode,
                                    "batch_size": int(batch_size),
                                    "seed": int(seed), "lr": float(lr),
                                    "weight_decay": float(weight_decay)},
                "best_epoch": int(best_epoch), "best_val_loss": float(best_loss),
            }
            temporary = output.with_name(output.name + ".tmp")
            torch.save(state, temporary)
            os.replace(temporary, output)
        else:
            stale += 1
            if stale >= int(patience):
                break
    payload = {
        "schema": "civic_phase3_landing_training_history_v4",
        "source_city": source_city,
        "checkpoint": str(output),
        "tokenizer_sha256": str(require_scalar(encoded, "tokenizer_sha256")),
        "codes_sha256": sha256_file(codes), "source_grid_table": table_binding,
        "landing_mode": mode, "normalization": normalization,
        "best_epoch": int(best_epoch), "best_val_loss": float(best_loss),
        "history": history,
    }
    write_json(output.with_name(output.stem + "_history.json"), payload)
    return payload


def load_landing_runtime(checkpoint, grid_latents, grid_csv, device, *,
                         vocabulary_policy="legacy"):
    """Load a frozen head and its city-specific representation/attribute tables."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("schema") != LANDING_CHECKPOINT_SCHEMA:
        raise RuntimeError("Landing trajectory evaluation requires a v4 checkpoint")
    mode = state.get("landing_mode", "raw_m1")
    options = {} if vocabulary_policy == "legacy" else {
        "vocabulary_policy": vocabulary_policy,
    }
    table, attributes, centers, vocabulary_ids, binding = _representation_table(
        mode, grid_latents, grid_csv, state.get("normalization"), **options
    )
    model = LandingHead(**state["model_config"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    return (model.to(device).eval(), torch.from_numpy(table).to(device),
            attributes, centers, vocabulary_ids, binding, state)


def _device(value):
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train a source-city landing head")
    parser.add_argument("--codes", required=True)
    parser.add_argument("--grid-latents", default=None)
    parser.add_argument("--grid-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("l2_attributes",), default="l2_attributes")
    parser.add_argument(
        "--normalization-grid-csv", action="append", default=[],
        help="Grid CSVs pooled to fit the L2 20-D q05/q95 normalizer",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 \
            or args.patience < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("landing-head training options are out of range")
    result = train_landing_head(
        args.codes, args.grid_latents, args.grid_csv, args.output,
        _device(args.device), args.epochs, args.batch_size, args.hidden_dim,
        args.lr, args.weight_decay, args.patience, args.seed,
        args.mode, args.normalization_grid_csv,
    )
    print(json.dumps({"best_epoch": result["best_epoch"]}, indent=2))
