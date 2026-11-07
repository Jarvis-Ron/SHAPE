"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

TOKENIZER_TRAINING_CONTRACT = "residual_fsq_no_layer_semantics_v2"
REMOVED_SEMANTIC_OPTIONS = frozenset({
    "injection_mode", "semantic_loss_weight",
    "semantic_warm_start_steps", "semantic_warm_start_lr",
})


def require_no_semantic_options(config):
    removed = REMOVED_SEMANTIC_OPTIONS.intersection(config)
    if removed:
        raise ValueError("Layer semantic supervision options were removed: "
                         + ", ".join(sorted(removed)))
    if config.get("aux_routing", "aggregate") != "aggregate":
        raise ValueError("Layer-targeted auxiliary options were removed; use aggregate objectives")


CITY_DATA_DIRS = {
    "nyc": "SHAPE/stage2/nyc_event_driven",
    "paris": "SHAPE/stage2/paris_event_driven",
    "ca": "SHAPE/stage2/ca_event_driven",
}


def resolve_data_dir(city="nyc", config=None):
    if config is not None:
        custom_dir = config.get("data_dir")
        if custom_dir and custom_dir != "SHAPE/stage2/data_cache_event_driven":
            return custom_dir
    return CITY_DATA_DIRS.get(city, CITY_DATA_DIRS["nyc"])


MODULE2_CONFIG = {
    "tokenizer_training_contract": TOKENIZER_TRAINING_CONTRACT,
    # v3 adds one explicit missing-static/latent flag at column 135.  The
    # city-invariant encoder selection remains unchanged; the flag is audited.
    "encoder_feature_contract": "city_invariant_v3_sparse_grid_flag",
    # SHAPE does not consume trajectory-derived surprisal channels.
    "use_surprisal": False,
    "data_dir": "SHAPE/stage2/nyc_event_driven",
    "city": "nyc",
    "event_npz_file": "event_driven_data.npz",
    "event_meta_file": "user_meta.json",
    "train_ids_file": "train_user_ids.npy",
    "val_ids_file": "val_user_ids.npy",
    "test_ids_file": "test_user_ids.npy",
    "surprisal_file": "surprisal_tables.npz",
    "d_input": 136,
    "d_target": 64,
    "d_proj": 16,
    "d_hidden_mlp": 64,
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 2,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "num_quantizers": 3,
    "codebook_size": 64,
    "num_embeddings": 64,
    "quantizer_type": "fsq",
    # Optional per-layer layout is persisted only for mixed checkpoints.
    "aux_routing": "aggregate",
    "fsq_levels": [4, 4, 4],
    "fsq_scale_init": 1.0,
    "fsq_diversity_weight": 0.01,
    "fsq_diversity_tau": 0.2,
    # FSQ learns no embedding table, so commitment must remain exactly zero.
    "fsq_commit_weight": 0.0,
    # Module 2 must not hand a collapsed checkpoint to joint training.
    "fsq_min_active_codes": 16,
    "fsq_min_perplexity": 8.0,
    "fsq_require_all_buckets": True,
    "fsq_strict_acceptance": True,
    "commit_beta": 0.25,
    "reset_threshold": 2.0,
    "diversity_entropy_weight": 1.0,
    "diversity_orth_weight": 0.1,
    "inject_scale_init": 1.0,
    "entropy_tau": 0.1,
    "entropy_tau_start": 0.3,
    "entropy_tau_end": 0.15,
    "entropy_tau_anneal_start_epoch": 5,
    "entropy_tau_anneal_end_epoch": 15,
    # Fixed source jump-bin auxiliary objective in the reference checkpoint.
    "jump_aux_weight": 0.125,
    "jump_bin_thresholds": [1e-8, 0.05, 0.15, 0.30],
    # Fixed 13-category POI auxiliary supervision from source events.
    "poi_aux_weight": 0.075,
    "num_poi_categories": 13,
    # Time auxiliary supervision is disabled in the fixed SHAPE checkpoint.
    "time_aux_weight": 0.0,
    "time_bin_boundaries": [5.0, 10.0, 14.0, 18.0, 22.0],
    "rq_commit_beta": 0.25,
    "encoder_noise": 1.0,
    "encoder_perturb": 0.05,
    "vq_proj_dim": 512,
    "vq_loss_weight": 10.0,
    "vq_loss_warmup_epochs": 0,
    "vq_loss_start_weight": 5.0,
    "vq_loss_final_weight": 5.0,
    "vq_commit_weight": 5.0,
    # FSQ diversity is already scaled inside each quantizer.  Keep the outer
    # coefficient at one so the loss is active instead of silently disabled.
    "vq_entropy_weight": 1.0,
    "vq_orth_weight": 4.0,
    # Fixed IID pre-quantization noise used when training SHAPE.
    "encoder_perturb_mode": "iid",
    "encoder_perturb_sigma": 0.1,
    # Legacy EMA schedule retained for old checkpoints only.
    "encoder_perturb_start": 0.0,
    "encoder_perturb_end": 0.0,
    "encoder_perturb_decay_epochs": 5,
    "ema_decay_start": 0.90,
    "ema_decay_end": 0.995,
    "ema_decay_warmup_epochs": 5,
    "replace_unused_every_n_steps": 10,
    "replace_unused_n_replace": 16,
    "lambda_spa": 0.5,
    "lambda_sem": 0.5,
    "gamma": 2.0,
    "entropy_beta": 3.0,
    "temporal_self_loop_w": 0.05,
    "temporal_ze_sim_w": 0.01,
    "temporal_entropy_w": 0.1,
    "small_sample": False,
    "small_ratio": 0.1,
    "temporal_warmup_epochs": 5,
    "epochs": 50,
    "batch_size": 196,
    # Optional cap on B * padded_sequence_length^2.  Source-city runners set
    # this explicitly so long-tail users are isolated without truncation.
    "max_attention_cells": 8388608,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 10,
    "log_interval": 50,
    "save_dir": "SHAPE/stage2/checkpoints",
    "model_name": "rqvae_transformer.pth",
    "seed": 42,
}
