"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import torch
from .config import CITY_DATA_DIRS, MODULE2_CONFIG
from .data import get_dataloaders
from .rqvae_modules import INJECTION_MODES
from .tokenizer import MIXED_QUANTIZER_PRESETS, SUPPORTED_QUANTIZERS, build_tokenizer
from .trainer import train_vqvae


REQUIRED_DATA_FILES = (
    "event_npz_file",
    "event_meta_file",
    "train_ids_file",
    "val_ids_file",
    "test_ids_file",
)


KNOWN_FSQ_LEVELS = {
    27: (3, 3, 3),
    32: (4, 4, 2),
    64: (4, 4, 4),
}


def _parse_fsq_levels(value):
    try:
        levels = tuple(int(item) for item in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "FSQ levels must be comma-separated integers"
        ) from exc
    if not levels or any(level < 2 for level in levels):
        raise argparse.ArgumentTypeError(
            "every FSQ level must be an integer greater than one"
        )
    return levels


def _parse_quantizer_layers(value):
    layers = tuple(
        item.strip().lower() for item in str(value).split(",")
    )
    if not layers or any(item not in ("fsq", "ema") for item in layers):
        raise argparse.ArgumentTypeError(
            "quantizer layers must be comma-separated fsq/ema values"
        )
    return layers


def _capacity_suffix(quantizer_type, codebook_size=None, fsq_levels=None):
    """Return an explicit capacity label without changing legacy names."""
    if codebook_size is None:
        return ""
    suffix = "_cb%d" % int(codebook_size)
    if quantizer_type in ("fsq", "mixed") and fsq_levels:
        suffix += "_l%s" % "x".join(str(level) for level in fsq_levels)
    return suffix


def artifact_names(
        quantizer_type, use_surprisal=False, jump_aux_weight=0.0,
        poi_aux_weight=0.0, time_aux_weight=0.0, codebook_size=None,
        fsq_levels=None, encoder_perturb_mode="none",
        encoder_perturb_sigma=0.0, quantizer_types=None,
        aux_routing="aggregate", injection_mode="layered",
        num_quantizers=None):
    """Return collision-free checkpoint, history, and plot names."""
    if quantizer_type not in SUPPORTED_QUANTIZERS:
        raise ValueError("quantizer_type must be fsq, ema, or mixed")
    if quantizer_type == "fsq":
        suffix = ""
    elif quantizer_type == "ema":
        suffix = "_ema"
    else:
        layers = quantizer_types or MIXED_QUANTIZER_PRESETS["mix1"]
        suffix = "_mixed_%s" % "-".join(layers)
    suffix += _capacity_suffix(
        quantizer_type, codebook_size=codebook_size,
        fsq_levels=fsq_levels,
    )
    if use_surprisal:
        suffix += "_surprisal"
    if float(jump_aux_weight) > 0.0:
        suffix += "_jumpaux"
    if float(poi_aux_weight) > 0.0:
        suffix += "_poiaux"
    if float(time_aux_weight) > 0.0:
        suffix += "_timeaux"
    if str(encoder_perturb_mode) != "none":
        sigma_label = ("%g" % float(encoder_perturb_sigma)).replace(".", "p")
        suffix += "_%s_s%s" % (encoder_perturb_mode, sigma_label)
    if str(aux_routing) == "layer_targeted":
        suffix += "_targetedaux"
    injection_mode = str(injection_mode).lower()
    if injection_mode not in INJECTION_MODES:
        raise ValueError(
            "injection_mode must be layered or first_layer_all"
        )
    if injection_mode == "first_layer_all":
        suffix += "_inj1"
    if num_quantizers is not None:
        suffix += "_rq%d" % int(num_quantizers)
    return {
        "model_name": "rqvae_transformer%s.pth" % suffix,
        "history_name": "training_history%s.json" % suffix,
        "plot_name": "rqvae_training_{city}%s.png" % suffix,
    }


def _validate_event_data(data_dir, config):
    missing = [
        os.path.join(data_dir, config[key])
        for key in REQUIRED_DATA_FILES
        if not os.path.isfile(os.path.join(data_dir, config[key]))
    ]
    if missing:
        raise FileNotFoundError(
            "Stage-one event data are incomplete. Missing: {}. Run "
            "`python -m SHAPE.stage2.prepare_city_event_data --city {}` "
            "after producing the Module-1 latent.".format(
                ", ".join(missing), config["city"]
            )
        )
    if config["city"] == "ca":
        contract_path = os.path.join(data_dir, "data_contract.json")
        if not os.path.isfile(contract_path):
            raise RuntimeError(
                "CA event data predate the timezone repair. Rebuild them with "
                "`python -m SHAPE.cli prepare-data --city ca`."
            )
        with open(contract_path, "r", encoding="utf-8") as handle:
            contract = json.load(handle)
        if (
            contract.get("local_time_contract") != "iana_timezone_v1"
            or contract.get("timezone") != "America/Los_Angeles"
        ):
            raise RuntimeError(
                "CA event data use an incompatible local-time contract: %r. "
                "Rebuild them after the timezone repair." % contract
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the fixed SHAPE source-city Residual FSQ tokenizer")
    parser.add_argument("--city", default="nyc", choices=sorted(CITY_DATA_DIRS))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--quantizer-type", default="fsq", choices=("fsq",))
    parser.add_argument("--injection-mode", default="layered", choices=("layered",))
    parser.add_argument("--num-quantizers", type=int, choices=(3,), default=None)
    parser.add_argument("--codebook-size", type=int, choices=(64,), default=None)
    parser.add_argument("--fsq-levels", type=_parse_fsq_levels, choices=((4,4,4),), default=None)
    for name in ("epochs", "patience", "batch-size", "seed"):
        parser.add_argument("--" + name, type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-attention-cells", type=int, default=8388608,
                        help="Bound padded attention memory without truncating source sequences")
    parser.add_argument("--small-sample", action="store_true")
    parser.add_argument("--small-ratio", type=float, default=0.1)
    parser.add_argument("--jump-aux-weight", type=float, choices=(0.125,), default=0.125)
    parser.add_argument("--poi-aux-weight", type=float, choices=(0.075,), default=0.075)
    parser.add_argument("--time-aux-weight", type=float, choices=(0.0,), default=0.0)
    parser.add_argument("--encoder-perturb-mode", choices=("iid",), default="iid")
    parser.add_argument("--encoder-perturb-sigma", type=float, choices=(0.1,), default=0.1)
    parser.add_argument("--save-dir", default=MODULE2_CONFIG["save_dir"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--history-name", default=None)
    parser.add_argument("--plot-path", default=None)
    args = parser.parse_args(argv)
    for name in ("epochs", "patience", "batch_size", "max_seq_len", "max_attention_cells", "lr"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    if not 0 < args.small_ratio <= 1:
        parser.error("small-ratio must be in (0,1]")
    args.depth_explicit = args.num_quantizers is not None
    args.capacity_explicit = args.codebook_size is not None or args.fsq_levels is not None
    args.num_quantizers, args.codebook_size, args.fsq_levels = 3, 64, (4,4,4)
    args.quantizer_types = ("fsq",) * 3
    args.aux_routing = "aggregate"
    args.use_surprisal = False
    return args


def build_training_config(args):
    config = dict(MODULE2_CONFIG)
    config["city"] = args.city
    config["data_dir"] = args.data_dir or CITY_DATA_DIRS[args.city]
    config["quantizer_type"] = args.quantizer_type
    config["quantizer_types"] = list(args.quantizer_types)
    config["mixed_stack_contract"] = "per_layer_quantizer_v1"
    config["aux_routing"] = args.aux_routing
    config["injection_mode"] = args.injection_mode
    config["num_quantizers"] = int(args.num_quantizers)
    config["codebook_size"] = int(args.codebook_size)
    config["num_embeddings"] = int(args.codebook_size)
    config["fsq_levels"] = list(args.fsq_levels)
    config["capacity_explicit"] = bool(args.capacity_explicit)
    config["save_dir"] = args.save_dir
    config["use_surprisal"] = bool(args.use_surprisal)
    config["encoder_perturb_mode"] = args.encoder_perturb_mode
    config["encoder_perturb_sigma"] = args.encoder_perturb_sigma
    if args.jump_aux_weight is not None:
        config["jump_aux_weight"] = float(args.jump_aux_weight)
    if args.poi_aux_weight is not None:
        config["poi_aux_weight"] = float(args.poi_aux_weight)
    if args.time_aux_weight is not None:
        config["time_aux_weight"] = float(args.time_aux_weight)
    names = artifact_names(
        args.quantizer_type, args.use_surprisal, config["jump_aux_weight"],
        config["poi_aux_weight"], config["time_aux_weight"],
        codebook_size=(
            config["codebook_size"] if args.capacity_explicit else None
        ),
        fsq_levels=(
            config["fsq_levels"] if "fsq" in args.quantizer_types else None
        ),
        encoder_perturb_mode=config["encoder_perturb_mode"],
        encoder_perturb_sigma=config["encoder_perturb_sigma"],
        quantizer_types=config["quantizer_types"],
        aux_routing=config["aux_routing"],
        injection_mode=config["injection_mode"],
        num_quantizers=(
            config["num_quantizers"] if args.depth_explicit else None
        ),
    )
    config["model_name"] = args.model_name or names["model_name"]
    config["history_name"] = args.history_name or names["history_name"]
    for name in ("epochs", "patience", "batch_size", "lr", "seed"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    config["max_seq_len"] = args.max_seq_len
    config["max_attention_cells"] = args.max_attention_cells
    return config


def main(argv=None):
    args = parse_args(argv)
    config = build_training_config(args)
    frozen = dict(quantizer_type='fsq', num_quantizers=3, codebook_size=64,
                  injection_mode='layered', use_surprisal=False,
                  jump_aux_weight=0.125, poi_aux_weight=0.075, time_aux_weight=0.0,
                  encoder_perturb_mode='iid', encoder_perturb_sigma=0.1)
    for key, value in frozen.items():
        if config.get(key) != value:
            raise ValueError('SHAPE training requires %s=%r' % (key, value))
    _validate_event_data(config["data_dir"], config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_loader, val_loader, _ = get_dataloaders(
        config=config,
        small_sample=args.small_sample,
        small_ratio=args.small_ratio,
        batch_size=config["batch_size"],
        city=args.city,
        max_seq_len=args.max_seq_len,
        seed=config["seed"],
        max_attention_cells=args.max_attention_cells,
    )
    model = build_tokenizer(config).to(device)
    names = artifact_names(
        args.quantizer_type, args.use_surprisal, config["jump_aux_weight"],
        config["poi_aux_weight"], config["time_aux_weight"],
        codebook_size=(
            config["codebook_size"] if args.capacity_explicit else None
        ),
        fsq_levels=(
            config["fsq_levels"] if "fsq" in args.quantizer_types else None
        ),
        encoder_perturb_mode=config["encoder_perturb_mode"],
        encoder_perturb_sigma=config["encoder_perturb_sigma"],
        quantizer_types=config["quantizer_types"],
        aux_routing=config["aux_routing"],
        injection_mode=config["injection_mode"],
        num_quantizers=(
            config["num_quantizers"] if args.depth_explicit else None
        ),
    )
    plot_path = args.plot_path or os.path.join(
        "SHAPE", "stage2", "output", names["plot_name"].format(city=args.city)
    )
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    train_vqvae(
        model,
        train_loader,
        val_loader,
        config=config,
        device=device,
        small_sample=args.small_sample,
        plot_path=plot_path,
    )
    result = {
        "status": "complete",
        "city": args.city,
        "quantizer_type": args.quantizer_type,
        "quantizer_types": config["quantizer_types"],
        "codebook_size": config["codebook_size"],
        "fsq_levels": (
            config["fsq_levels"] if "fsq" in args.quantizer_types else None
        ),
        "capacity_explicit": config["capacity_explicit"],
        "checkpoint": os.path.join(config["save_dir"], config["model_name"]),
        "history": os.path.join(config["save_dir"], config["history_name"]),
        "plot": plot_path,
        "encoder_feature_contract": config["encoder_feature_contract"],
        "use_surprisal": config["use_surprisal"],
        "jump_aux_weight": config["jump_aux_weight"],
        "poi_aux_weight": config["poi_aux_weight"],
        "time_aux_weight": config["time_aux_weight"],
        "encoder_perturb_mode": config["encoder_perturb_mode"],
        "encoder_perturb_sigma": config["encoder_perturb_sigma"],
        "aux_routing": config["aux_routing"],
        "injection_mode": config["injection_mode"],
        "num_quantizers": config["num_quantizers"],
        "epochs": config["epochs"],
        "patience": config["patience"],
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
