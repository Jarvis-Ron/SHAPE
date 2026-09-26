"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import re
import torch
from .config import MODULE2_CONFIG
from .model import RQVAETransformerExtractor


LAYER_QUANTIZERS = ("fsq", "ema")


SUPPORTED_QUANTIZERS = ("fsq", "ema", "mixed")


MIXED_QUANTIZER_PRESETS = {
    "mix1": ("fsq", "ema", "ema"),
    "mix2": ("fsq", "fsq", "ema"),
}


ENCODER_FEATURE_CONTRACT = "city_invariant_v3_sparse_grid_flag"


def checkpoint_state_dict(state):
    """Extract a model state dict from a wrapped or bare checkpoint."""
    return state.get("model_state_dict", state)


def normalize_quantizer_types(
        quantizer_type, quantizer_types=None, num_quantizers=3):
    """Return a canonical family label and one concrete type per RQ layer."""
    family = str(quantizer_type).lower()
    if family not in SUPPORTED_QUANTIZERS:
        raise ValueError("quantizer_type must be 'fsq', 'ema', or 'mixed'")
    num_quantizers = int(num_quantizers)
    if num_quantizers < 1:
        raise ValueError("num_quantizers must be positive")
    if quantizer_types is None:
        if family == "mixed":
            if num_quantizers != len(MIXED_QUANTIZER_PRESETS["mix1"]):
                raise ValueError(
                    "the default mixed stack requires exactly three layers"
                )
            layers = MIXED_QUANTIZER_PRESETS["mix1"]
        else:
            layers = (family,) * num_quantizers
    else:
        if isinstance(quantizer_types, str):
            quantizer_types = quantizer_types.split(",")
        layers = tuple(str(value).strip().lower() for value in quantizer_types)
        if len(layers) != num_quantizers:
            raise ValueError(
                "quantizer_types must contain exactly %d entries"
                % num_quantizers
            )
        if any(value not in LAYER_QUANTIZERS for value in layers):
            raise ValueError("quantizer_types entries must be 'fsq' or 'ema'")

    inferred_family = layers[0] if len(set(layers)) == 1 else "mixed"
    if family != inferred_family:
        raise ValueError(
            "quantizer_type %r disagrees with per-layer layout %s"
            % (family, ",".join(layers))
        )
    return inferred_family, layers


def infer_quantizer_types(state_dict, num_quantizers=None):
    """Infer the concrete FSQ/EMA implementation used by every RQ layer."""
    markers = {}
    pattern = re.compile(r"(?:^|\.)rq_layer\.rq_layers\.(\d+)\.(.+)$")
    for key in state_dict:
        match = pattern.search(key)
        if match is None:
            continue
        layer = int(match.group(1))
        suffix = match.group(2)
        kinds = markers.setdefault(layer, set())
        if suffix == "log_quant_scale":
            kinds.add("fsq")
        if suffix == "ema_cluster_size":
            kinds.add("ema")
    if num_quantizers is None:
        if not markers:
            raise RuntimeError(
                "Could not infer checkpoint quantizer layers from FSQ or EMA state"
            )
        num_quantizers = max(markers) + 1
    layers = []
    for layer in range(int(num_quantizers)):
        kinds = markers.get(layer, set())
        if len(kinds) != 1:
            raise RuntimeError(
                "Could not infer exactly one quantizer type for layer %d: %s"
                % (layer, sorted(kinds))
            )
        layers.append(next(iter(kinds)))
    return tuple(layers)


def require_quantizer_compatible(
        state_dict, quantizer_type="fsq", num_quantizers=3,
        quantizer_types=None):
    """Reject partial or mislabeled per-layer quantizer checkpoints."""
    _, layers = normalize_quantizer_types(
        quantizer_type, quantizer_types, num_quantizers
    )
    fsq_parameters = (
        "input_norm.weight",
        "input_norm.bias",
        "bottleneck_norm.running_mean",
        "bottleneck_norm.running_var",
        "bottleneck_norm.num_batches_tracked",
        "down.weight",
        "down.bias",
        "up.weight",
        "up.bias",
        "log_quant_scale",
    )
    ema_parameters = (
        "embedding.0",
        "ema_cluster_size",
        "ema_dw",
        "ema_decay_val",
        "inject_scales",
    )
    required = []
    for layer, layer_type in enumerate(layers):
        parameters = fsq_parameters if layer_type == "fsq" else ema_parameters
        required.extend(
            "rq_layer.rq_layers.{}.{}".format(layer, parameter)
            for parameter in parameters
        )
    missing = [
        marker for marker in required
        if not any(key.endswith(marker) for key in state_dict)
    ]
    if missing:
        raise RuntimeError(
            "Checkpoint does not contain the complete {} stack (missing {}). "
            "Retrain the stage-one tokenizer; legacy or partial checkpoints "
            "are incompatible.".format(
                "/".join(value.upper() for value in layers),
                ", ".join(missing),
            )
        )
    inferred_layers = infer_quantizer_types(
        state_dict, num_quantizers=num_quantizers
    )
    if inferred_layers != layers:
        raise RuntimeError(
            "Checkpoint quantizer layout mismatch: state=%s expected=%s"
            % (",".join(inferred_layers), ",".join(layers))
        )
    return True


def build_tokenizer(config=None):
    """Build M2 from a complete or partial stage-one configuration."""
    resolved = dict(MODULE2_CONFIG)
    if config is not None:
        resolved.update(config)
    if resolved.get("encoder_feature_contract") != ENCODER_FEATURE_CONTRACT:
        raise ValueError(
            "Unsupported encoder feature contract: %r"
            % resolved.get("encoder_feature_contract")
        )
    quantizer_type, quantizer_types = normalize_quantizer_types(
        resolved["quantizer_type"], resolved.get("quantizer_types"),
        resolved["num_quantizers"],
    )
    resolved["quantizer_type"] = quantizer_type
    resolved["quantizer_types"] = list(quantizer_types)
    return RQVAETransformerExtractor(
        d_model=resolved["d_model"],
        nhead=resolved["nhead"],
        num_encoder_layers=resolved["num_encoder_layers"],
        num_decoder_layers=resolved["num_decoder_layers"],
        dim_feedforward=resolved["dim_feedforward"],
        dropout=resolved["dropout"],
        num_quantizers=resolved["num_quantizers"],
        codebook_size=resolved["codebook_size"],
        commit_beta=resolved.get("rq_commit_beta", resolved["commit_beta"]),
        entropy_weight=resolved.get("diversity_entropy_weight", 1.0),
        orth_weight=resolved.get("diversity_orth_weight", 0.1),
        inject_scale_init=resolved.get("inject_scale_init", 1.0),
        lambda_spa=resolved.get("lambda_spa", 0.5),
        lambda_sem=resolved.get("lambda_sem", 0.5),
        gamma=resolved.get("gamma", 2.0),
        temporal_self_loop_w=resolved.get("temporal_self_loop_w", 0.05),
        temporal_ze_sim_w=resolved.get("temporal_ze_sim_w", 0.01),
        temporal_entropy_w=resolved.get("temporal_entropy_w", 0.1),
        entropy_tau=resolved.get("entropy_tau", 0.1),
        quantizer_type=quantizer_type,
        quantizer_types=quantizer_types,
        fsq_levels=tuple(resolved.get("fsq_levels", (4, 4, 4))),
        fsq_scale_init=resolved.get("fsq_scale_init", 1.0),
        fsq_diversity_weight=resolved.get("fsq_diversity_weight", 0.01),
        fsq_diversity_tau=resolved.get("fsq_diversity_tau", 0.2),
        use_surprisal=resolved.get("use_surprisal", False),
        jump_aux_enabled=resolved.get("jump_aux_weight", 0.0) > 0.0,
        poi_aux_enabled=resolved.get("poi_aux_weight", 0.0) > 0.0,
        num_poi_categories=resolved.get("num_poi_categories", 13),
        time_aux_enabled=resolved.get("time_aux_weight", 0.0) > 0.0,
        time_bin_boundaries=tuple(
            resolved.get("time_bin_boundaries", (5, 10, 14, 18, 22))
        ),
        jump_bin_thresholds=tuple(
            resolved.get("jump_bin_thresholds", (1e-8, 0.05, 0.15, 0.30))
        ),
        encoder_perturb_mode=resolved.get("encoder_perturb_mode", "none"),
        encoder_perturb_sigma=resolved.get("encoder_perturb_sigma", 0.0),
        aux_routing=resolved.get("aux_routing", "aggregate"),
        injection_mode=resolved.get("injection_mode", "layered"),
    )


def load_tokenizer(checkpoint_path, device, quantizer_type=None):
    """Load one source-city M2 tokenizer with strict quantizer compatibility."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint_state_dict(state)
    checkpoint_config = state.get("config", {}) if isinstance(state, dict) else {}
    checkpoint_contract = checkpoint_config.get("encoder_feature_contract")
    if checkpoint_contract != ENCODER_FEATURE_CONTRACT:
        raise RuntimeError(
            "Checkpoint encoder feature contract is {!r}, expected {!r}. "
            "Retrain the stage-one tokenizer after the Route-A encoder change."
            .format(checkpoint_contract, ENCODER_FEATURE_CONTRACT)
        )
    num_quantizers = int(checkpoint_config.get(
        "num_quantizers", MODULE2_CONFIG["num_quantizers"]
    ))
    inferred_layers = infer_quantizer_types(
        state_dict, num_quantizers=num_quantizers
    )
    inferred = (
        inferred_layers[0] if len(set(inferred_layers)) == 1 else "mixed"
    )
    configured, configured_layers = normalize_quantizer_types(
        checkpoint_config.get("quantizer_type", inferred),
        checkpoint_config.get("quantizer_types", inferred_layers),
        num_quantizers,
    )
    if configured_layers != inferred_layers:
        raise RuntimeError(
            "Checkpoint config quantizer layout %s disagrees with state %s"
            % (",".join(configured_layers), ",".join(inferred_layers))
        )
    requested = str(quantizer_type or configured).lower()
    if requested not in SUPPORTED_QUANTIZERS:
        raise ValueError("quantizer_type must be 'fsq', 'ema', or 'mixed'")
    if requested != inferred:
        raise RuntimeError(
            "Requested %s but checkpoint state contains %s quantizer parameters"
            % (requested, inferred)
        )

    config = dict(MODULE2_CONFIG)
    config.update(checkpoint_config)
    config["quantizer_type"] = requested
    config["quantizer_types"] = list(configured_layers)
    require_quantizer_compatible(
        state_dict, requested, num_quantizers=config["num_quantizers"],
        quantizer_types=config["quantizer_types"],
    )
    model = build_tokenizer(config)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    encoding_missing = [
        key for key in missing
        if key.startswith("surprisal_encoder")
        or key.startswith("transformer_encoder")
        or key.startswith("rq_layer")
    ]
    required_aux_prefixes = [
        prefix for enabled, prefix in (
            (config.get("jump_aux_weight", 0.0) > 0.0, "jump_head"),
            (config.get("poi_aux_weight", 0.0) > 0.0, "poi_head"),
            (config.get("time_aux_weight", 0.0) > 0.0, "time_head"),
        ) if enabled
    ]
    encoding_missing.extend(
        key for key in missing
        if any(key.startswith(prefix) for prefix in required_aux_prefixes)
    )
    if encoding_missing:
        raise RuntimeError(
            "Checkpoint is missing parameters required for encoding: %s"
            % ", ".join(encoding_missing)
        )
    model.to(device).eval()
    return model, config, {
        "inferred_quantizer_type": inferred,
        "quantizer_types": list(inferred_layers),
        "mixed_stack_contract": checkpoint_config.get(
            "mixed_stack_contract"
        ),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_epoch": state.get("epoch") if isinstance(state, dict) else None,
        "checkpoint_val_loss": (
            state.get("val_loss") if isinstance(state, dict) else None
        ),
        "checkpoint_city": checkpoint_config.get("city"),
        "configured_epochs": checkpoint_config.get("epochs"),
        "configured_patience": checkpoint_config.get("patience"),
        "configured_batch_size": checkpoint_config.get("batch_size"),
        "configured_max_seq_len": checkpoint_config.get("max_seq_len"),
        "codebook_size": int(config["codebook_size"]),
        "fsq_levels": (
            list(config.get("fsq_levels", ()))
            if "fsq" in inferred_layers else None
        ),
        "capacity_explicit": bool(
            checkpoint_config.get("capacity_explicit", False)
        ),
        "encoder_feature_contract": checkpoint_contract,
        "use_surprisal": bool(checkpoint_config.get("use_surprisal", False)),
        "jump_aux_weight": float(
            checkpoint_config.get("jump_aux_weight", 0.0)
        ),
        "poi_aux_weight": float(
            checkpoint_config.get("poi_aux_weight", 0.0)
        ),
        "time_aux_weight": float(
            checkpoint_config.get("time_aux_weight", 0.0)
        ),
        "encoder_perturb_mode": str(
            checkpoint_config.get("encoder_perturb_mode", "none")
        ),
        "encoder_perturb_sigma": float(
            checkpoint_config.get("encoder_perturb_sigma", 0.0)
        ),
        "aux_routing": str(
            checkpoint_config.get("aux_routing", "aggregate")
        ),
        "injection_mode": str(model.rq_layer.injection_mode),
        "replace_unused_every_n_steps": checkpoint_config.get(
            "replace_unused_every_n_steps"
        ),
        "replace_unused_n_replace": checkpoint_config.get(
            "replace_unused_n_replace"
        ),
        "reset_threshold": checkpoint_config.get("reset_threshold"),
        "training_completion": (
            state.get("training_completion")
            if isinstance(state, dict) else None
        ),
    }
