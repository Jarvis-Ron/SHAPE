"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from .config import MODULE1_CONFIG
from .grid_text_encoder import CityGridEmbedding, load_grid_features
from SHAPE.gridspec import StaticGrid, verify_all_candidate_files


CITY_CSV_PATHS = {
    "ca": MODULE1_CONFIG["ca_csv_path"],
    "nyc": MODULE1_CONFIG["csv_path"],
    "paris": MODULE1_CONFIG["paris_csv_path"],
}


def _build_model():
    return CityGridEmbedding(
        d_feat=MODULE1_CONFIG["d_in"],
        d_model=MODULE1_CONFIG["d_model"],
        nhead=MODULE1_CONFIG["nhead"],
        num_layers=MODULE1_CONFIG["num_layers"],
        dim_feedforward=MODULE1_CONFIG["dim_feedforward"],
        dropout=MODULE1_CONFIG["dropout"],
        d_proj=MODULE1_CONFIG["proj_dim"],
        max_grid_size=MODULE1_CONFIG["max_tile_position"],
        num_poi_categories=MODULE1_CONFIG["num_poi_categories"],
    )


def run_extraction(
        output_dir=MODULE1_CONFIG["output_dir"], device=None, csv_paths=None,
        checkpoint_path=None, force=False):
    """Load one trained M1 checkpoint and write sparse `<city>_latent.npz` files."""
    released_inputs = csv_paths is None
    csv_paths = dict(CITY_CSV_PATHS if csv_paths is None else csv_paths)
    checkpoint_path = checkpoint_path or os.path.join(
        MODULE1_CONFIG["save_dir"], MODULE1_CONFIG["model_name"]
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Module-1 checkpoint not found: %s" % checkpoint_path)
    missing_csv = [path for path in csv_paths.values() if not os.path.isfile(path)]
    if missing_csv:
        raise FileNotFoundError(
            "Module-1 city feature CSVs not found: %s" % ", ".join(missing_csv)
        )

    output_dir = Path(output_dir)
    output_paths = {
        city: output_dir / ("%s_latent.npz" % city) for city in csv_paths
    }
    collisions = [os.fspath(path) for path in output_paths.values() if path.exists()]
    if collisions and not force:
        raise FileExistsError(
            "Refusing to overwrite existing latents: {}. Pass --force only "
            "when replacing all bindings intentionally.".format(", ".join(collisions))
        )

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = state.get("model_state_dict", state)
    model = _build_model()
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Module-1 checkpoint architecture mismatch: missing={}, unexpected={}"
            .format(
                list(incompatible.missing_keys), list(incompatible.unexpected_keys)
            )
        )
    model.to(resolved_device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    release_receipts = None
    if released_inputs:
        release_receipts = verify_all_candidate_files(
            Path(MODULE1_CONFIG["grid_spec_path"]).resolve().parent.parent,
            MODULE1_CONFIG["release_manifest_path"],
        )

    results = {}
    with torch.inference_mode():
        for city, csv_path in csv_paths.items():
            static_grid = None
            if released_inputs:
                static_grid = StaticGrid.load(
                    csv_path, city=city,
                    grid_spec_path=MODULE1_CONFIG["grid_spec_path"],
                    release_manifest_path=MODULE1_CONFIG["release_manifest_path"],
                    verify_release=True,
                )
                features, grid_ids = static_grid.features, static_grid.grid_ids
            else:
                features, grid_ids, _ = load_grid_features(csv_path)
            if features.shape[1] != MODULE1_CONFIG["d_in"]:
                raise ValueError(
                    "%s has %d features, expected %d"
                    % (city, features.shape[1], MODULE1_CONFIG["d_in"])
                )
            if static_grid is None:
                feature_tensor = torch.from_numpy(features).float().to(resolved_device)
                grid_tensor = torch.from_numpy(grid_ids).long().to(resolved_device)
                latent = model.encode_city(feature_tensor, grid_tensor).cpu().numpy()
                latent_ids = np.asarray(grid_ids, dtype=np.int64)
            else:
                anchors = None  # complete static vocabulary; no visit filter
                latent, latent_ids = model.encode_static_grid(
                    static_grid, device=str(resolved_device),
                    tile_size=MODULE1_CONFIG["tile_size"],
                    halo=MODULE1_CONFIG["tile_halo"], anchor_ids=anchors,
                )
            np.savez_compressed(
                output_paths[city], grid_ids=np.asarray(latent_ids, dtype=np.int64),
                embeddings=np.asarray(latent, dtype=np.float32),
                city=np.asarray(city.lower()),
            )
            results[city] = {
                "path": os.fspath(output_paths[city]),
                "shape": list(latent.shape),
                "grid_id_dtype": "int64",
                "release_receipt": (
                    release_receipts.get(city.upper())
                    if release_receipts is not None else None
                ),
            }
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract M1 city latents for zero-shot stage-one M2 data"
    )
    parser.add_argument("--output-dir", default=MODULE1_CONFIG["output_dir"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = run_extraction(
        output_dir=args.output_dir,
        device=args.device,
        checkpoint_path=args.checkpoint,
        force=args.force,
    )
    for city, result in results.items():
        print("[%s] %s -> %s" % (city, result["shape"], result["path"]))
    return results


if __name__ == "__main__":
    main()
