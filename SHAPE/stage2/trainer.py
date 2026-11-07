"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

import os, time, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import MODULE2_CONFIG as C, require_no_semantic_options


@torch.no_grad()
def _fsq_layer_diagnostics(model, codes=None):
    """Snapshot per-dimension FSQ statistics and scalar-bucket occupancy."""
    diagnostics = []
    for layer, quantizer in enumerate(model.rq_layer.rq_layers):
        if not hasattr(quantizer, "quant_scale"):
            continue
        bucket_counts = [
            quantizer.last_bucket_counts[dimension, :level]
            .detach().cpu().tolist()
            for dimension, level in enumerate(quantizer.levels)
        ]
        # Validation codes are already padding-filtered.  Decode them back to
        # mixed-radix scalar buckets so acceptance covers the whole split,
        # instead of only whichever mini-batch happened to run last.
        if codes is not None and codes.numel():
            flat_codes = codes.reshape(-1, codes.shape[-1])[:, layer].long()
            bucket_counts = []
            for level, basis in zip(
                    quantizer.levels, quantizer.basis.detach().cpu()):
                indices = (flat_codes // int(basis)) % level
                bucket_counts.append(
                    torch.bincount(indices, minlength=level).tolist()
                )
        diagnostics.append({
            "layer": layer,
            "scale": float(quantizer.quant_scale.detach()),
            "pre_tanh_std": float(quantizer.last_pre_tanh_std),
            "quantized_std": float(quantizer.last_quantized_std),
            "pre_tanh_mean_per_dim": quantizer.last_pre_tanh_mean_per_dim
                .detach().cpu().tolist(),
            "pre_tanh_std_per_dim": quantizer.last_pre_tanh_std_per_dim
                .detach().cpu().tolist(),
            "quantized_std_per_dim": quantizer.last_quantized_std_per_dim
                .detach().cpu().tolist(),
            "bucket_counts": bucket_counts,
            "buckets_all_nonzero": all(
                count > 0 for dimension in bucket_counts
                for count in dimension
            ),
        })
    return diagnostics


def _fsq_acceptance_failures(layer_metrics, diagnostics,
                             min_active=16, min_perplexity=8.0,
                             require_all_buckets=True):
    """Return precise per-layer reasons a Module-2 FSQ gate did not pass."""
    diagnostic_by_layer = {item["layer"]: item for item in diagnostics}
    failures = []
    for metric in layer_metrics:
        layer = metric["layer"]
        if layer not in diagnostic_by_layer:
            continue
        if metric["active"] < min_active:
            failures.append(
                "L%d active=%d<%d" % (layer, metric["active"], min_active)
            )
        if metric["perplexity"] < min_perplexity:
            failures.append(
                "L%d perplexity=%.2f<%.2f"
                % (layer, metric["perplexity"], min_perplexity)
            )
        diagnostic = diagnostic_by_layer.get(layer)
        if require_all_buckets and (
                diagnostic is None or not diagnostic["buckets_all_nonzero"]):
            failures.append("L%d has an empty scalar bucket" % layer)
    return failures


def train_epoch(model, loader, optimizer, device, grad_clip=1.0, log_interval=50,
                temporal_weight=1.0, vq_commit_weight=5.0, vq_entropy_weight=1.0,
                vq_orth_weight=4.0, replace_every=20, replace_n=12,
                jump_aux_weight=0.0,
                poi_aux_weight=0.0, time_aux_weight=0.0):
    """Train one epoch."""
    model.train()
    total_loss = total_recon = total_vq = total_future = total_temporal = 0.0
    total_jump_aux = 0.0
    total_poi_aux = 0.0
    total_time_aux = 0.0
    total_vq_commit = total_vq_entropy = total_vq_orth = 0.0
    # Weighted VQ contributions for the total loss (tracked separately)
    total_vq_commit_w = total_vq_entropy_w = total_vq_orth_w = 0.0
    total_perp = 0.0
    z_e = None
    all_codes = []
    uses_ema = bool(getattr(model.rq_layer, "uses_ema", True))

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        x = batch["x"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        H_hat, vq_loss, perplexity, z_e, z_q, codes, future_loss, temporal_loss = model(x, attention_mask=attention_mask)
        recon_loss = model.compute_loss(x, H_hat, mask=attention_mask)
        jump_aux_loss = model.compute_jump_aux_loss(
            x, z_q, mask=attention_mask
        )
        poi_aux_loss = model.compute_poi_aux_loss(
            x, z_q, mask=attention_mask
        )
        time_aux_loss = model.compute_time_aux_loss(
            x, z_q, mask=attention_mask
        )

        # Decoupled VQ weighting: commit / entropy / orth each get their own
        # coefficient.  `vq_loss` (sum of the three) is still returned for logs.
        commit_l = model.rq_layer.last_vq_commit
        entropy_l = model.rq_layer.last_vq_entropy
        orth_l = model.rq_layer.last_vq_orth
        weighted_vq = (vq_commit_weight * commit_l
                       + vq_entropy_weight * entropy_l
                       + vq_orth_weight * orth_l)

        loss = (recon_loss + weighted_vq + 0.1 * future_loss
                + temporal_weight * temporal_loss
                + jump_aux_weight * jump_aux_loss
                + poi_aux_weight * poi_aux_loss
                + time_aux_weight * time_aux_loss)



        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if model.rq_layer.quantizer_type == "mixed":
            # EMA statistics were accumulated only by EMA layers during the
            # forward pass. Refresh them only after backward, while FSQ layers
            # continue to learn their scalar projections through gradients.
            model.rq_layer.refresh_ema_codebooks()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_vq += (commit_l.item() + entropy_l.item() + orth_l.item()) * bs
        total_vq_commit += commit_l.item() * bs
        total_vq_entropy += entropy_l.item() * bs
        total_vq_orth += orth_l.item() * bs
        total_vq_commit_w += (vq_commit_weight * commit_l.item()) * bs
        total_vq_entropy_w += (vq_entropy_weight * entropy_l.item()) * bs
        total_vq_orth_w += (vq_orth_weight * orth_l.item()) * bs
        total_future += future_loss.item() * bs
        total_temporal += temporal_loss.item() * bs
        total_jump_aux += jump_aux_loss.item() * bs
        total_poi_aux += poi_aux_loss.item() * bs
        total_time_aux += time_aux_loss.item() * bs
        total_perp += perplexity.item() * bs
        all_codes.append(codes.detach().cpu()[attention_mask.detach().cpu()])

        if (uses_ema and model.training and z_e is not None
                and replace_every > 0 and step % replace_every == 0):
              with torch.no_grad():
                  # Replace under-used codes every replace_every steps
                  # Use the same guided space the quantizer looked up so dead
                  # codes are reset with on-sphere replacements in that domain.
                  n_replaced = model.rq_layer.replace_unused_codes(
                      z_e.detach(), mask=attention_mask, n_replace=replace_n,
                      guided=model.rq_layer.last_guided_list
                  )

    n = len(loader.dataset)
    all_codes = torch.cat(all_codes, dim=0)
    layer_metrics = model.rq_layer.code_usage_metrics(all_codes)
    for metric, layer_type in zip(
            layer_metrics, model.rq_layer.quantizer_types):
        metric["quantizer_type"] = layer_type
    usage_ratio = float(np.mean([m["usage_ratio"] for m in layer_metrics]))
    return (total_loss/n, total_recon/n, total_vq/n, total_future/n, total_temporal/n,
            total_jump_aux/n, total_poi_aux/n,
            total_time_aux/n,
            total_vq_commit/n, total_vq_entropy/n, total_vq_orth/n,
            total_perp/n, usage_ratio, layer_metrics)


@torch.no_grad()
def evaluate(model, loader, device):
    """Validation/test set evaluation."""
    model.eval()
    total_recon = 0.0
    total_perp = 0.0
    total_jump_aux = 0.0
    total_poi_aux = 0.0
    total_time_aux = 0.0
    total_vq = 0.0
    total_vq_commit = total_vq_entropy = total_vq_orth = 0.0
    z_e = None
    all_codes = []
    for batch in tqdm(loader, desc="  Eval", leave=False):
        x = batch["x"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        H_hat, vq_loss, perplexity, z_e, z_q, codes, future_loss, temporal_loss = model(x, attention_mask=attention_mask)
        recon_loss = model.compute_loss(x, H_hat, mask=attention_mask)
        jump_aux_loss = model.compute_jump_aux_loss(
            x, z_q, mask=attention_mask
        )
        poi_aux_loss = model.compute_poi_aux_loss(
            x, z_q, mask=attention_mask
        )
        time_aux_loss = model.compute_time_aux_loss(
            x, z_q, mask=attention_mask
        )

        bs = x.size(0)
        total_recon += recon_loss.item() * bs
        total_vq += vq_loss.item() * bs
        total_vq_commit += model.rq_layer.last_vq_commit.item() * bs
        total_vq_entropy += model.rq_layer.last_vq_entropy.item() * bs
        total_vq_orth += model.rq_layer.last_vq_orth.item() * bs
        total_perp += perplexity.item() * bs
        total_jump_aux += jump_aux_loss.item() * bs
        total_poi_aux += poi_aux_loss.item() * bs
        total_time_aux += time_aux_loss.item() * bs
        all_codes.append(codes.cpu()[attention_mask.cpu()])

    n = len(loader.dataset)
    all_codes = torch.cat(all_codes, dim=0)
    layer_metrics = model.rq_layer.code_usage_metrics(all_codes)
    for metric, layer_type in zip(
            layer_metrics, model.rq_layer.quantizer_types):
        metric["quantizer_type"] = layer_type
    usage_ratio = float(np.mean([m["usage_ratio"] for m in layer_metrics]))
    return (total_recon/n, total_vq/n, total_vq_commit/n, total_vq_entropy/n,
            total_vq_orth/n, total_perp/n, usage_ratio,
            total_jump_aux/n, total_poi_aux/n, total_time_aux/n,
            layer_metrics, all_codes)


def plot_training(history, save_path, num_embeddings=256):
    """Plot and save training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train", marker=".")
    ax.plot(epochs, history["val_loss"], label="Val", marker=".")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[2, 0]
    fsq_scales = history.get("val_fsq_scale", [])
    fsq_layers = [
        index for index, layer_type in enumerate(
            history.get("quantizer_types", [])
        ) if layer_type == "fsq"
    ]
    if fsq_scales and fsq_scales[-1]:
        for offset, layer in enumerate(fsq_layers or range(len(fsq_scales[-1]))):
            ax.plot(
                epochs, [row[offset] for row in fsq_scales],
                marker=".", label="L{} scale".format(layer),
            )
    ax.set_title("FSQ Learned Quantization Scale")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[2, 1]
    fsq_stds = history.get("val_fsq_z_std", [])
    if fsq_stds and fsq_stds[-1]:
        for offset, layer in enumerate(fsq_layers or range(len(fsq_stds[-1]))):
            ax.plot(
                epochs, [row[offset] for row in fsq_stds],
                marker=".", label="L{} tanh std".format(layer),
            )
    ax.axhspan(0.5, 0.7, color="green", alpha=0.1, label="target 0.5-0.7")
    ax.set_title("FSQ Quantized Scalar Spread")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.plot(epochs, history["train_recon"], label="Train", marker=".")
    ax.plot(epochs, history["val_recon"], label="Val", marker=".")
    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(epochs, history["train_perp"], label="Train", marker=".")
    ax.plot(epochs, history["val_perp"], label="Val", marker=".")
    for layer in range(len(history.get("val_layer_perp", [[]])[-1])):
        ax.plot(
            epochs,
            [row[layer] for row in history["val_layer_perp"]],
            linestyle="--", alpha=0.75, label="Val L{}".format(layer),
        )
    ax.set_title("Codebook Perplexity")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.plot(epochs, history["train_usage"], label="Train", marker=".", color="green")
    for layer in range(len(history.get("val_layer_usage", [[]])[-1])):
        ax.plot(
            epochs,
            [row[layer] for row in history["val_layer_usage"]],
            linestyle="--", alpha=0.75, label="Val L{}".format(layer),
        )
    ax.set_title(f"Codebook Usage / {num_embeddings}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved: {save_path}")


def plot_fsq_health(history, save_path, num_embeddings=64,
                    min_active=16, min_perplexity=8.0):
    """Render the strict anti-collapse signals used by the Module-2 gate."""
    epochs = range(1, len(history["val_layer_perp"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fsq_layers = [
        index for index, layer_type in enumerate(
            history.get("quantizer_types", [])
        ) if layer_type == "fsq"
    ]
    if not fsq_layers:
        fsq_layers = list(range(
            len(history.get("val_fsq_pre_std_min", [[]])[-1])
        ))
    for offset, layer in enumerate(fsq_layers):
        axes[0, 0].plot(
            epochs,
            [row[layer] * num_embeddings for row in history["val_layer_usage"]],
            marker=".", label="Layer %d" % layer,
        )
        axes[0, 1].plot(
            epochs, [row[layer] for row in history["val_layer_perp"]],
            marker=".", label="Layer %d" % layer,
        )
        axes[1, 0].plot(
            epochs, [row[offset] for row in history["val_fsq_pre_std_min"]],
            marker=".", label="Layer %d" % layer,
        )
        axes[1, 1].plot(
            epochs, [row[offset] for row in history["val_fsq_bucket_min"]],
            marker=".", label="Layer %d" % layer,
        )
    axes[0, 0].axhline(min_active, color="red", linestyle="--")
    axes[0, 0].set_title("Active FSQ Codes (minimum %d)" % min_active)
    axes[0, 1].axhline(min_perplexity, color="red", linestyle="--")
    axes[0, 1].set_title("FSQ Perplexity (minimum %.1f)" % min_perplexity)
    axes[1, 0].set_title("Minimum Pre-tanh Std Across 3 Dimensions")
    axes[1, 1].axhline(1.0, color="red", linestyle="--")
    axes[1, 1].set_title("Minimum Validation Count Across 3 x 4 Buckets")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  FSQ health plot saved: %s" % save_path)


def set_seed(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_vqvae(model, train_loader, val_loader, config=None, device="cpu",
                small_sample=False, plot_path=None, logger=None):
    """Full VQ-VAE training loop."""
    _log = logger.info if logger else print
    if config is None:
        config = C
    require_no_semantic_options(config)
    _seed = config.get("seed", 42)
    set_seed(_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config["patience"] // 3)
    best_val_loss = float("inf")
    best_epoch = None
    patience_counter = 0
    early_stopping_triggered = False
    history = {"train_loss": [], "train_recon": [], "train_vq": [], "train_vq_commit": [], "train_vq_entropy": [], "train_vq_orth": [],
               "train_future": [], "train_temporal": [],
               "train_jump_aux": [], "val_jump_aux": [],
               "train_poi_aux": [], "val_poi_aux": [],
               "train_time_aux": [], "val_time_aux": [],
               "val_loss": [], "val_recon": [], "val_perp": [], "train_perp": [],
               "train_usage": [], "val_usage": [],
               "cb_active": [], "cb_dead": [], "cb_unused": [], "cb_entropy": [],
               "train_layer_perp": [], "val_layer_perp": [],
               "train_layer_usage": [], "val_layer_usage": [],
               "train_layer_max_share": [], "val_layer_max_share": [],
               "train_fsq_scale": [], "val_fsq_scale": [],
               "train_fsq_z_std": [], "val_fsq_z_std": [],
               "train_fsq_pre_std_min": [], "val_fsq_pre_std_min": [],
               "val_fsq_z_std_min": [], "val_fsq_bucket_min": []}
    history["quantizer_types"] = list(model.rq_layer.quantizer_types)
    history["aux_routing"] = str(getattr(model, "aux_routing", "aggregate"))
    n_epochs = config["epochs"]
    _log("")
    _log("  VQ-VAE Training (device=%s)" % device)
    _log("  small_sample=%s" % small_sample)
    _log("  epochs=%d, lr=%s" % (n_epochs, config['lr']))
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    attention_budget = getattr(
        batch_sampler, "max_attention_cells", None
    )
    if attention_budget is None:
        _log("  batch_size=%s" % train_loader.batch_size)
    else:
        _log(
            "  batch_size=%d (maximum; variable attention-budget batches)"
            % batch_sampler.max_batch_size
        )
        _log("  max_attention_cells=%d (B*T_max^2)" % attention_budget)
    _log("  d_model=%d, codebook=%d" % (config['d_model'], config['num_embeddings']))
    has_ema = bool(getattr(model.rq_layer, "has_ema", True))
    has_fsq = bool(getattr(model.rq_layer, "has_fsq", False))
    uses_ema = has_ema
    accepted_checkpoint = False
    last_fsq_failures = []
    _log("  quantizer_layers=%s" % "/".join(
        value.upper() for value in model.rq_layer.quantizer_types
    ))
    _log("  lambda_spa=%s, lambda_sem=%s, gamma=%s" % (config['lambda_spa'], config['lambda_sem'], config['gamma']))
    _log("")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        temporal_weight = min(1.0, 0.5 + 0.5 * epoch / max(1, config.get("temporal_warmup_epochs", 50)))
        # Entropy softmax temperature schedule: high tau early for exploration,
        # linear anneal to a sharp temperature for late-stage codebook refinement.
        tau_start = config.get("entropy_tau_start", 3.0)
        tau_end = config.get("entropy_tau_end", 0.5)
        anneal_start = config.get("entropy_tau_anneal_start_epoch", 5)
        anneal_end = config.get("entropy_tau_anneal_end_epoch", 15)
        if epoch <= anneal_start:
            entropy_tau = tau_start
        elif epoch >= anneal_end:
            entropy_tau = tau_end
        else:
            frac = (epoch - anneal_start) / max(1, anneal_end - anneal_start)
            entropy_tau = tau_start + (tau_end - tau_start) * frac
        if uses_ema:
            model.rq_layer.set_entropy_tau(entropy_tau)
        
        # Decoupled VQ weights: commit/entropy/orth each keep a fixed scale.
        vq_commit_w = (
            config.get("vq_commit_weight", 5.0) if uses_ema
            else config.get("fsq_commit_weight", 0.0)
        )
        vq_entropy_w = config.get("vq_entropy_weight", 1.0)
        vq_orth_w = config.get("vq_orth_weight", 4.0)
        
        # Encoder perturbation schedule
        perturb_start = config.get("encoder_perturb_start", config.get("encoder_perturb", 0.05))
        perturb_end = config.get("encoder_perturb_end", 0.01)
        decay_epochs = config.get("encoder_perturb_decay_epochs", 6)
        if epoch <= decay_epochs:
            current_perturb = perturb_start + (perturb_end - perturb_start) * (epoch - 1) / max(1, decay_epochs - 1)
        else:
            current_perturb = perturb_end
        if uses_ema and hasattr(model, 'encoder_perturb'):
            model.encoder_perturb = current_perturb
        if model.rq_layer.quantizer_type == "mixed":
            current_perturb = float(
                config.get("encoder_perturb_sigma", 0.0)
            )
        elif not uses_ema:
            current_perturb = 0.0
        
        # EMA decay: low->high (fast adaptation early, stable later)
        ema_start = config.get("ema_decay_start", 0.85)
        ema_end = config.get("ema_decay_end", 0.99)
        ema_warmup = config.get("ema_decay_warmup_epochs", 12)
        if epoch <= ema_warmup:
            current_ema_decay = ema_start + (ema_end - ema_start) * (epoch - 1) / max(1, ema_warmup - 1)
        else:
            current_ema_decay = ema_end
        if uses_ema:
            model.rq_layer.ema_decay_val = torch.tensor(
                current_ema_decay, device=device
            )
        else:
            current_ema_decay = 0.0
        
        # Replace unused schedule
        replace_every = config.get("replace_unused_every_n_steps", 20)
        replace_n = config.get("replace_unused_n_replace", 8)
        # For small sample, use a moderate cap to avoid over-resetting
        if small_sample:
            replace_n = min(16, replace_n)

        (train_loss, train_recon, train_vq, train_future, train_temporal,
         train_jump_aux, train_poi_aux, train_time_aux,
         train_vq_commit, train_vq_entropy, train_vq_orth,
         train_perp, train_usage, train_layer_metrics) = \
           train_epoch(model, train_loader, optimizer, device,
                       grad_clip=config["grad_clip"],
                       log_interval=config["log_interval"],
                       temporal_weight=temporal_weight,
                       vq_commit_weight=vq_commit_w,
                       vq_entropy_weight=vq_entropy_w,
                       vq_orth_weight=vq_orth_w,
                       replace_every=replace_every,
                       replace_n=replace_n,
                       jump_aux_weight=config.get("jump_aux_weight", 0.0),
                       poi_aux_weight=config.get("poi_aux_weight", 0.0),
                       time_aux_weight=config.get("time_aux_weight", 0.0))
        train_fsq_diagnostics = _fsq_layer_diagnostics(model)
        reset_th = config.get("reset_threshold", 3.0)
        if small_sample:
            reset_th = max(1.0, reset_th * 0.5)
        n_reset = 0
        model.eval()
        with torch.no_grad():
            # Conservative reset: only every 5 epochs, give codes time to stabilize
            if uses_ema and epoch % 5 == 0:
                for batch in val_loader:
                    x_batch = batch["x"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    output = model(
                        x_batch, attention_mask=attention_mask
                    )
                    z_e = output[3]
                    n_reset = model.rq_layer.replace_unused_codes(
                        z_e.detach(), mask=attention_mask, n_replace=8,
                        threshold=reset_th,
                        guided=model.rq_layer.last_guided_list,
                    )
                    break
        model.train()
        (val_recon, val_vq, val_vq_commit, val_vq_entropy, val_vq_orth,
         val_perp, val_usage, val_jump_aux, val_poi_aux,
         val_time_aux,
         val_layer_metrics,
         val_codes) = evaluate(
             model, val_loader, device
         )
        val_loss = (
            val_recon
            + config.get("jump_aux_weight", 0.0) * val_jump_aux
            + config.get("poi_aux_weight", 0.0) * val_poi_aux
            + config.get("time_aux_weight", 0.0) * val_time_aux
        )
        val_fsq_diagnostics = _fsq_layer_diagnostics(model, codes=val_codes)

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_vq"].append(train_vq)
        history["train_vq_commit"].append(train_vq_commit)
        history["train_vq_entropy"].append(train_vq_entropy)
        history["train_vq_orth"].append(train_vq_orth)
        history["train_future"].append(train_future)
        history["train_temporal"].append(train_temporal)
        history["train_jump_aux"].append(train_jump_aux)
        history["train_poi_aux"].append(train_poi_aux)
        history["train_time_aux"].append(train_time_aux)
        history["train_perp"].append(train_perp)
        history["train_usage"].append(train_usage)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_jump_aux"].append(val_jump_aux)
        history["val_poi_aux"].append(val_poi_aux)
        history["val_time_aux"].append(val_time_aux)
        history["val_perp"].append(val_perp)
        history["val_usage"].append(val_usage)
        history["train_layer_perp"].append(
            [m["perplexity"] for m in train_layer_metrics]
        )
        history["val_layer_perp"].append(
            [m["perplexity"] for m in val_layer_metrics]
        )
        history["train_layer_usage"].append(
            [m["usage_ratio"] for m in train_layer_metrics]
        )
        history["val_layer_usage"].append(
            [m["usage_ratio"] for m in val_layer_metrics]
        )
        history["train_layer_max_share"].append(
            [m["max_code_share"] for m in train_layer_metrics]
        )
        history["val_layer_max_share"].append(
            [m["max_code_share"] for m in val_layer_metrics]
        )
        history["train_fsq_scale"].append(
            [m["scale"] for m in train_fsq_diagnostics]
        )
        history["val_fsq_scale"].append(
            [m["scale"] for m in val_fsq_diagnostics]
        )
        history["train_fsq_z_std"].append(
            [m["quantized_std"] for m in train_fsq_diagnostics]
        )
        history["val_fsq_z_std"].append(
            [m["quantized_std"] for m in val_fsq_diagnostics]
        )
        history["train_fsq_pre_std_min"].append([
            min(m["pre_tanh_std_per_dim"])
            for m in train_fsq_diagnostics
        ])
        history["val_fsq_pre_std_min"].append([
            min(m["pre_tanh_std_per_dim"])
            for m in val_fsq_diagnostics
        ])
        history["val_fsq_z_std_min"].append([
            min(m["quantized_std_per_dim"])
            for m in val_fsq_diagnostics
        ])
        history["val_fsq_bucket_min"].append([
            min(count for dimension in m["bucket_counts"] for count in dimension)
            for m in val_fsq_diagnostics
        ])
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        # Codebook monitoring
        cb_total = model.rq_layer.num_quantizers * model.rq_layer.codebook_size
        cb_active = sum(m["active"] for m in val_layer_metrics)
        cb_unused = sum(m["dead"] for m in val_layer_metrics)
        # FSQ has no persistent dead-code state by construction.  ``cb_unused``
        # separately reports codes absent from this validation epoch.
        cb_dead = sum(
            metric["dead"] for metric in val_layer_metrics
            if metric.get("quantizer_type") == "ema"
        )
        cb_entropy = float(np.mean([m["entropy"] for m in val_layer_metrics]))
        max_entropy = math.log(model.rq_layer.codebook_size)
        entropy_ratio = cb_entropy / max_entropy if max_entropy > 0 else 0.0
        history["cb_active"].append(cb_active)
        history["cb_dead"].append(cb_dead)
        history["cb_unused"].append(cb_unused)
        history["cb_entropy"].append(entropy_ratio)
        _log("  Epoch %3d/%d | loss=%.4f(val=%.4f) | recon=%.3f future=%.4f temporal=%.4f(w=%.2f) jump=%.4f(w=%.2f) poi=%.4f(w=%.2f) time=%.4f(w=%.2f) | vq=%.4f(commit=%.4f ent=%.4f orth=%.4f, val=%.4f) | perp=%.1f(%.2f%%) | cb=%d/%d dead=%d H=%.2f | w_c=%.2f w_e=%.4f w_o=%.2f ep=%.3f ema=%.3f | reset=%d | lr=%.2e | %ds" % (
            epoch, n_epochs,
            train_loss, val_loss,
            train_recon, train_future, train_temporal, temporal_weight,
            train_jump_aux,
            config.get("jump_aux_weight", 0.0),
            train_poi_aux, config.get("poi_aux_weight", 0.0),
            train_time_aux, config.get("time_aux_weight", 0.0),
            train_vq, train_vq_commit, train_vq_entropy, train_vq_orth, val_vq,
            train_perp, train_usage * 100,
            cb_active, cb_total, cb_dead, entropy_ratio,
            vq_commit_w, vq_entropy_w, vq_orth_w,
            current_perturb, current_ema_decay,
            n_reset,
            lr_now,
            elapsed
        ))
        diagnostic_by_layer = {
            item["layer"]: item for item in val_fsq_diagnostics
        }
        layer_parts = []
        for metric in val_layer_metrics:
            part = "L{layer}:perp={perplexity:.1f},use={active}/{total}".format(
                total=model.rq_layer.codebook_size, **metric
            )
            diagnostic = diagnostic_by_layer.get(metric["layer"])
            if diagnostic is not None:
                pre_std = ",".join(
                    "%.3f" % value
                    for value in diagnostic["pre_tanh_std_per_dim"]
                )
                z_std = ",".join(
                    "%.3f" % value
                    for value in diagnostic["quantized_std_per_dim"]
                )
                buckets = "/".join(
                    "d%d:%s" % (dimension, counts)
                    for dimension, counts in enumerate(
                        diagnostic["bucket_counts"]
                    )
                )
                part += (
                    ",scale={scale:.3f},prestd=[{pre}],zstd=[{z}],bins={bins}"
                    .format(
                        scale=diagnostic["scale"], pre=pre_std,
                        z=z_std, bins=buckets,
                    )
                )
            layer_parts.append(part)
        layer_summary = " ".join(layer_parts)
        _log("    [Quantizer] %s | unused_this_val=%d" % (
            layer_summary, cb_unused
        ))
        if has_fsq:
            last_fsq_failures = _fsq_acceptance_failures(
                val_layer_metrics,
                val_fsq_diagnostics,
                min_active=config.get("fsq_min_active_codes", 16),
                min_perplexity=config.get("fsq_min_perplexity", 8.0),
                require_all_buckets=config.get(
                    "fsq_require_all_buckets", True
                ),
            )
            if last_fsq_failures:
                _log(
                    "    [Quantizer][ACCEPTANCE] FAIL | %s"
                    % "; ".join(last_fsq_failures)
                )
            else:
                _log("    [Quantizer][ACCEPTANCE] PASS")
        acceptance_ok = not last_fsq_failures
        if acceptance_ok and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            accepted_checkpoint = True
            patience_counter = 0
            os.makedirs(config["save_dir"], exist_ok=True)
            ckpt_path = os.path.join(config["save_dir"], config["model_name"])
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, ckpt_path)
            _log("  * Best model saved: %s (val_loss=%.4f)" % (ckpt_path, val_loss))
        elif acceptance_ok:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                early_stopping_triggered = True
                _log("  * Early stop, %d epochs no improvement" % config["patience"])
                break
    if plot_path:
        plot_training(history, plot_path, model.rq_layer.codebook_size)
        if has_fsq:
            root, extension = os.path.splitext(plot_path)
            plot_fsq_health(
                history,
                root + "_fsq_health" + (extension or ".png"),
                num_embeddings=model.rq_layer.codebook_size,
                min_active=config.get("fsq_min_active_codes", 16),
                min_perplexity=config.get("fsq_min_perplexity", 8.0),
            )
    training_completion = {
        "status": "complete" if accepted_checkpoint else "no_checkpoint",
        "configured_epochs": int(n_epochs),
        "configured_patience": int(config["patience"]),
        "epochs_completed": int(len(history["val_loss"])),
        "best_epoch": best_epoch,
        "best_val_loss": (
            float(best_val_loss) if np.isfinite(best_val_loss) else None
        ),
        "early_stopping_triggered": bool(early_stopping_triggered),
        "quantizer_types": list(model.rq_layer.quantizer_types),
        "final_layer_health": [
            {
                "layer": int(metric["layer"]),
                "quantizer_type": str(metric["quantizer_type"]),
                "active": int(metric["active"]),
                "eligible": int(metric["active"]),
                "total": int(model.rq_layer.codebook_size),
                "perplexity": float(metric["perplexity"]),
                "max_code_share": float(metric["max_code_share"]),
            }
            for metric in val_layer_metrics
        ],
        "stop_reason": (
            "early_stopping" if early_stopping_triggered
            else "budget_exhausted"
        ),
    }
    history["training_completion"] = training_completion
    if accepted_checkpoint:
        ckpt_path = os.path.join(config["save_dir"], config["model_name"])
        try:
            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch < 2.0 compatibility
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        checkpoint["training_completion"] = training_completion
        torch.save(checkpoint, ckpt_path)

    # Save training history for codebook evaluation
    import json
    hist_path = os.path.join(
        config["save_dir"], config.get("history_name", "training_history.json")
    )
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    hist_serializable = {}
    for k, v in history.items():
        if isinstance(v, list) and len(v) > 0 and hasattr(v[0], "item"):
            hist_serializable[k] = [x.item() if hasattr(x, "item") else x for x in v]
        else:
            hist_serializable[k] = v
    with open(hist_path, "w") as f:
        json.dump(hist_serializable, f, indent=2)
    _log("  [Save] Training history -> %s" % hist_path)

    if (has_fsq and config.get("fsq_strict_acceptance", True)
            and not accepted_checkpoint):
        raise RuntimeError(
            "FSQ collapse gate failed; no checkpoint met the complete "
            "active/perplexity/3x4-bucket criteria: %s"
            % ("; ".join(last_fsq_failures) or "no validation result")
        )

    _log("")
    _log("  Training complete! Best val_loss=%.4f" % best_val_loss)
    _log("")
    return model, history
