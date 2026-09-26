"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
from functools import partial
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from .alignment_metrics import evaluate_live_retrieval
from .config import MODULE1_CONFIG
from .contrastive_trainer import MultiCityContrastiveDataset, collate_multi_contrastive, train_contrastive
from .extract_embeddings import CITY_CSV_PATHS, _build_model
from .grid_text_encoder import load_grid_features
from SHAPE.gridspec import StaticGrid, verify_all_candidate_files


def load_city_training_data(csv_paths=None):
    """Load all cities and preserve each CSV's real local grid identifiers."""
    released_inputs = csv_paths is None
    csv_paths = dict(CITY_CSV_PATHS if released_inputs else csv_paths)
    missing = [path for path in csv_paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Module-1 city feature CSVs not found: %s" % ", ".join(missing)
        )
    city_data = {}
    for city, path in csv_paths.items():
        if released_inputs:
            static_grid = StaticGrid.load(
                path, city=city,
                grid_spec_path=MODULE1_CONFIG["grid_spec_path"],
                release_manifest_path=MODULE1_CONFIG["release_manifest_path"],
                verify_release=True,
            )
            # All emitted static cells: no target check-in activity filter.
            keep = np.arange(len(static_grid), dtype=np.int64)
            features = static_grid.features[keep]
            grid_ids = static_grid.grid_ids[keep]
            grid_rows = np.asarray(static_grid.rows)[keep]
            grid_cols = np.asarray(static_grid.cols)[keep]
        else:
            features, grid_ids, _ = load_grid_features(path)
            fixture_width = max(1, int(np.ceil(np.sqrt(len(grid_ids)))))
            fixture_positions = np.arange(len(grid_ids), dtype=np.int64)
            grid_rows = fixture_positions // fixture_width
            grid_cols = fixture_positions % fixture_width
        if features.ndim != 2 or features.shape[1] != MODULE1_CONFIG["d_in"]:
            raise ValueError(
                "%s has feature shape %s, expected [N, %d]"
                % (city, features.shape, MODULE1_CONFIG["d_in"])
            )
        city_data[city] = (features, grid_ids, grid_rows, grid_cols)
    return city_data


def build_training_loaders(city_data, batch_size, seed, num_workers=0):
    """Build deterministic 90/10 pair loaders for joint three-city training."""
    dataset = MultiCityContrastiveDataset(
        city_data,
        topk_feat_sim=(
            MODULE1_CONFIG["cross_semantic_topk"]
            if MODULE1_CONFIG["pos_rule"] == "poi_profile"
            else MODULE1_CONFIG["contrastive_topk_feat_sim"]
        ),
        temperature=MODULE1_CONFIG["contrastive_temperature"],
        seed=seed,
        pos_rule=MODULE1_CONFIG["pos_rule"],
        hard_neg_k=MODULE1_CONFIG["num_hard_negatives"],
        hard_neg_n=2,
    )
    if len(dataset) < 2:
        raise ValueError(
            "Module-1 contrastive pairing produced fewer than two examples"
        )
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    train_collate = partial(
        collate_multi_contrastive,
        mixup_alpha=MODULE1_CONFIG["contrastive_mixup_alpha"],
        city_feat_cache=dataset._city_feat_cache,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_data, shuffle=True, generator=generator,
        collate_fn=train_collate, **common
    )
    val_loader = DataLoader(
        val_data, shuffle=False, collate_fn=collate_multi_contrastive,
        **common
    )
    return dataset, train_loader, val_loader


def run_training(
        checkpoint_path=None, history_path=None, device=None, csv_paths=None,
        epochs=None, batch_size=None, lr=None, seed=42, num_workers=0,
        force=False):
    """Train in a staging directory, then atomically publish checkpoint/history."""
    checkpoint_path = Path(checkpoint_path or os.path.join(
        MODULE1_CONFIG["save_dir"], MODULE1_CONFIG["model_name"]
    ))
    history_path = Path(history_path or checkpoint_path.with_name(
        "m1_training_history.json"
    ))
    if checkpoint_path.exists() and not force:
        raise FileExistsError(
            "Refusing to overwrite Module-1 checkpoint: %s. Pass --force only "
            "when intentionally retraining the shared embedding space."
            % checkpoint_path
        )
    epochs = int(epochs or MODULE1_CONFIG["contrastive_epochs"])
    batch_size = int(batch_size or MODULE1_CONFIG["contrastive_batch_size"])
    lr = float(lr or MODULE1_CONFIG["contrastive_lr"])
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    city_data = load_city_training_data(csv_paths)
    dataset, train_loader, val_loader = build_training_loaders(
        city_data, batch_size=batch_size, seed=seed, num_workers=num_workers
    )
    model = _build_model()
    all_features = np.concatenate(
        [value[0] for value in city_data.values()], axis=0
    )
    model.update_feat_stats(torch.from_numpy(all_features).float())

    metadata = {
        "training_contract": "three_city_sparse_tiled_v2",
        "cities": list(city_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "feature_stats_scope": "all_training_cities",
        "positive_pair_contract": MODULE1_CONFIG["positive_pair_contract"],
        "cross_semantic_topk": MODULE1_CONFIG["cross_semantic_topk"],
        "explicit_hard_negatives": dataset.hard_neg_k,
        "poi_target_temperature": MODULE1_CONFIG["poi_target_temperature"],
        "poi_multi_positive": MODULE1_CONFIG["poi_multi_positive"],
        "mixup_alpha": MODULE1_CONFIG["contrastive_mixup_alpha"],
        "checkpoint_selection_metric": MODULE1_CONFIG[
            "checkpoint_selection_metric"
        ],
        "checkpoint_selection_min_delta": MODULE1_CONFIG[
            "checkpoint_selection_min_delta"
        ],
        "early_stopping_metric": MODULE1_CONFIG["early_stopping_metric"],
        "early_stopping_min_delta": MODULE1_CONFIG[
            "early_stopping_min_delta"
        ],
        "poi_eval_js_threshold": MODULE1_CONFIG["poi_eval_js_threshold"],
        "retrieval_eval_every": 5,
        "grid_contract": "grid_spec_v4_int64_sparse_ids",
        "tile_size": MODULE1_CONFIG["tile_size"],
        "tile_halo": MODULE1_CONFIG["tile_halo"],
    }
    if csv_paths is None:
        metadata["candidate_release_receipts"] = verify_all_candidate_files(
            Path(MODULE1_CONFIG["grid_spec_path"]).resolve().parent.parent,
            MODULE1_CONFIG["release_manifest_path"],
        )
    retrieval_eval_fn = partial(
        evaluate_live_retrieval,
        city_data=city_data,
        device=resolved_device,
        max_grids_per_city=MODULE1_CONFIG["retrieval_eval_max_grids_per_city"],
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".m1_training_", dir=os.fspath(checkpoint_path.parent)) as temp:
        model, history = train_contrastive(
            model,
            train_loader,
            val_loader=val_loader,
            device=resolved_device,
            epochs=epochs,
            lr=lr,
            temperature=MODULE1_CONFIG["contrastive_temperature"],
            uniformity_weight=MODULE1_CONFIG["uniformity_weight"],
            patience=MODULE1_CONFIG["contrastive_patience"],
            save_dir=temp,
            mixup_alpha=MODULE1_CONFIG["contrastive_mixup_alpha"],
            semantic_head_weight=MODULE1_CONFIG["semantic_head_weight"],
            weight_decay=MODULE1_CONFIG["contrastive_weight_decay"],
            checkpoint_metadata=metadata,
            eval_fn=retrieval_eval_fn,
            eval_every=metadata["retrieval_eval_every"],
            poi_target_temperature=MODULE1_CONFIG["poi_target_temperature"],
            use_poi_multi_positive=MODULE1_CONFIG["poi_multi_positive"],
            selection_metric=metadata["checkpoint_selection_metric"],
            selection_min_delta=metadata[
                "checkpoint_selection_min_delta"
            ],
            early_stopping_min_delta=metadata[
                "early_stopping_min_delta"
            ],
        )
        staged_checkpoint = Path(temp) / "grid_transformer_best.pth"
        if not staged_checkpoint.is_file():
            raise RuntimeError("Module-1 training produced no best checkpoint")
        staged_history = Path(temp) / "m1_training_history.json"
        staged_history.write_text(
            json.dumps({"config": metadata, "history": history}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged_checkpoint, checkpoint_path)
        os.replace(staged_history, history_path)

    return {
        "status": "complete",
        "checkpoint": os.fspath(checkpoint_path),
        "history": os.fspath(history_path),
        "best_val_loss": min(history["val_loss"]),
        "best_poi_r@1": max(
            row["poi_r@1"] for row in history["retrieval_eval"]
        ),
        "best_hard_r@1": max(
            row["hard_r@1"] for row in history["retrieval_eval"]
        ),
        "epochs_completed": len(history["train_loss"]),
        "config": metadata,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the shared CA/NYC/Paris Module-1 grid embedding"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("learning rate must be positive")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    if args.num_workers < 0:
        parser.error("num workers must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    result = run_training(
        checkpoint_path=args.checkpoint,
        history_path=args.history,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
