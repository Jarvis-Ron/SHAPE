"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def summarize_code_usage_counts(counts):
    """Return per-layer usage diagnostics from ``[layers, codes]`` counts."""
    if counts.dim() != 2:
        raise ValueError("counts must have shape [layers, codebook_size]")
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    codebook_size = counts.shape[1]
    metrics = []
    for layer in range(counts.shape[0]):
        histogram = counts[layer].detach().float().cpu()
        probs = histogram / histogram.sum().clamp_min(1.0)
        entropy = float(
            -(probs * torch.log(probs.clamp_min(1e-10))).sum().item()
        )
        active = int((histogram > 0).sum().item())
        metrics.append({
            "layer": layer,
            "active": active,
            "dead": int(codebook_size - active),
            "usage_ratio": active / float(codebook_size),
            "entropy": entropy,
            "perplexity": math.exp(entropy),
            "max_code_share": float(probs.max().item()),
            "histogram": histogram,
        })
    return metrics


@torch.no_grad()
def summarize_code_usage(codes, codebook_size, mask=None):
    """Return per-layer usage diagnostics for ``[..., num_quantizers]`` codes.

    The returned dictionaries are JSON-friendly except for ``histogram``, which
    is a CPU tensor intended for plotting.  Padding can be excluded with a mask
    matching ``codes.shape[:-1]``.
    """
    if codes.dim() < 2:
        raise ValueError("codes must end with a quantizer-layer dimension")
    flat = codes.reshape(-1, codes.shape[-1]).long()
    if mask is not None:
        if tuple(mask.shape) != tuple(codes.shape[:-1]):
            raise ValueError("mask must match codes.shape[:-1]")
        flat = flat[mask.reshape(-1).bool()]
    counts_by_layer = []
    for layer in range(codes.shape[-1]):
        values = flat[:, layer] if flat.numel() else flat.new_zeros((0,))
        valid = values[(values >= 0) & (values < codebook_size)]
        counts_by_layer.append(
            torch.bincount(valid, minlength=codebook_size)
        )
    counts = torch.stack(counts_by_layer, dim=0)
    return summarize_code_usage_counts(counts)


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization with a fixed mixed-radix code space.

    ``levels=(4, 4, 4)`` produces exactly 64 discrete codes.  Unlike EMA-VQ,
    FSQ has no learned codebook state, dead codes, reset path, or cross-rank
    accumulator synchronization.  The learnable ``down``/``up`` projections
    keep the public 256-dimensional interface used by the residual stack.
    """

    uses_ema = False

    def __init__(self, embedding_dim=256, levels=(4, 4, 4),
                 commitment_cost=0.0, inject_scale_init=1.0,
                 quant_scale_init=1.0, diversity_weight=0.01,
                 diversity_tau=0.2):
        super().__init__()
        if not levels or any(int(level) < 2 for level in levels):
            raise ValueError("FSQ levels must contain integers >= 2")
        self.levels = tuple(int(level) for level in levels)
        self.codebook_size = math.prod(self.levels)
        self.embedding_dim = embedding_dim
        # FSQ has no learned codebook to pull encoder outputs towards.  A VQ
        # commitment term is therefore both unnecessary and harmful: once a
        # scalar saturates, it reinforces the fixed sign and tanh supplies
        # almost no gradient to escape.  Keep the constructor argument for
        # checkpoint/config API compatibility, but deliberately ignore it.
        self.commitment_cost = 0.0
        self.diversity_weight = float(diversity_weight)
        self.diversity_tau = float(diversity_tau)
        if quant_scale_init <= 0:
            raise ValueError("quant_scale_init must be positive")
        if diversity_tau <= 0:
            raise ValueError("diversity_tau must be positive")
        # Residual magnitudes shrink naturally in later RQ layers.  Normalize
        # each residual before the 3-D bottleneck, then let every
        # layer learn its own bounded input scale instead of collapsing into
        # the two bins around zero.
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, len(self.levels))
        # LayerNorm above normalizes each 256-D token, but cannot prevent any
        # one of the three projected scalar dimensions from taking a fixed
        # sign across the batch.  Normalize each scalar across valid tokens,
        # tracking running statistics for deterministic evaluation.
        self.bottleneck_norm = nn.BatchNorm1d(
            len(self.levels), affine=False, momentum=0.1
        )
        self.up = nn.Linear(len(self.levels), embedding_dim)
        self.log_quant_scale = nn.Parameter(
            torch.tensor(math.log(float(quant_scale_init)))
        )
        self.inject_scales = nn.Parameter(
            torch.ones(1) * inject_scale_init
        )
        level_sizes = torch.tensor(self.levels, dtype=torch.float32)
        basis = []
        stride = 1
        for level in reversed(self.levels):
            basis.append(stride)
            stride *= level
        self.register_buffer("level_sizes", level_sizes)
        self.register_buffer(
            "basis", torch.tensor(list(reversed(basis)), dtype=torch.long)
        )
        self.register_buffer(
            "last_usage", torch.zeros(self.codebook_size), persistent=False
        )
        self.register_buffer(
            "last_pre_tanh_std", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "last_quantized_std", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "last_pre_tanh_mean_per_dim",
            torch.zeros(len(self.levels)), persistent=False,
        )
        self.register_buffer(
            "last_pre_tanh_std_per_dim",
            torch.zeros(len(self.levels)), persistent=False,
        )
        self.register_buffer(
            "last_quantized_std_per_dim",
            torch.zeros(len(self.levels)), persistent=False,
        )
        self.register_buffer(
            "last_bucket_counts",
            torch.zeros(len(self.levels), max(self.levels), dtype=torch.long),
            persistent=False,
        )
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def quant_scale(self):
        """Positive per-layer scale with a safety bound against saturation."""
        return self.log_quant_scale.exp().clamp(min=0.1, max=10.0)

    def _soft_code_probabilities(self, z):
        """Differentiable probabilities over the complete mixed-radix space."""
        probabilities = []
        for dimension, level in enumerate(self.levels):
            values = torch.linspace(
                -1.0, 1.0, steps=level, device=z.device, dtype=z.dtype
            )
            distances = (z[:, dimension:dimension + 1] - values) ** 2
            probabilities.append(
                torch.softmax(-distances / self.diversity_tau, dim=-1)
            )
        joint = probabilities[0]
        for dimension_probs in probabilities[1:]:
            joint = (
                joint.unsqueeze(-1) * dimension_probs.unsqueeze(-2)
            ).reshape(z.shape[0], -1)
        return joint

    def _quantize_scalars(self, z):
        max_index = self.level_sizes.to(z.device, z.dtype) - 1
        indices = torch.round((z + 1.0) * 0.5 * max_index)
        indices = indices.clamp_min(0.0)
        indices = torch.minimum(indices, max_index)
        quantized = indices / max_index * 2.0 - 1.0
        quantized_ste = z + (quantized - z).detach()
        codes = (indices.long() * self.basis.to(indices.device)).sum(dim=-1)
        return quantized_ste, quantized, codes, indices.long()

    def _normalize_bottleneck(self, projected, valid):
        """Normalize each scalar dimension using valid trajectory rows only."""
        flat = projected.reshape(-1, projected.shape[-1])
        normalized = torch.zeros_like(flat)
        if not valid.any():
            return normalized.reshape_as(projected)
        valid_values = flat[valid]
        if self.training and valid_values.shape[0] == 1:
            # BatchNorm cannot estimate variance from one row.  Use the most
            # recent running statistics without mutating them for this batch.
            normalized_values = F.batch_norm(
                valid_values,
                self.bottleneck_norm.running_mean,
                self.bottleneck_norm.running_var,
                weight=None,
                bias=None,
                training=False,
                momentum=0.0,
                eps=self.bottleneck_norm.eps,
            )
        else:
            normalized_values = self.bottleneck_norm(valid_values)
        normalized[valid] = normalized_values
        return normalized.reshape_as(projected)

    def quantize_guided(self, guided, return_all_codes=True, mask=None):
        B, T, _ = guided.shape
        if mask is None:
            valid = torch.ones(B * T, dtype=torch.bool, device=guided.device)
        else:
            if tuple(mask.shape) != (B, T):
                raise ValueError("mask must match guided.shape[:2]")
            valid = mask.reshape(-1).bool()
        projected = self.down(self.input_norm(guided))
        normalized = self._normalize_bottleneck(projected, valid)
        # 0.7 keeps a standard-normal bottleneck away from tanh's flat tails
        # while still reaching all four scalar buckets.
        pre_tanh = self.quant_scale * 0.7 * normalized
        z = torch.tanh(pre_tanh)
        z_ste, z_hard, flat_codes, scalar_indices = \
            self._quantize_scalars(z)
        z_q = self.up(z_ste)
        if mask is not None:
            z_q = z_q * mask.unsqueeze(-1).to(z_q.dtype)
        z_flat = z.reshape(-1, z.shape[-1])
        z_hard_flat = z_hard.reshape(-1, z_hard.shape[-1])
        code_flat = flat_codes.reshape(-1)
        if valid.any():
            commit_loss = self.commitment_cost * F.mse_loss(
                z_flat[valid], z_hard_flat[valid].detach()
            )
            usage = torch.bincount(
                code_flat[valid], minlength=self.codebook_size
            ).to(guided.dtype)
            # The hard FSQ index is non-differentiable.  A soft joint-code
            # distribution supplies the missing utilization pressure: tokens
            # should be individually decisive while the batch should occupy a
            # broad set of codes.  This is the standard entropy-difference
            # objective, applied only to real (non-padding) trajectory rows.
            soft_codes = self._soft_code_probabilities(z_flat[valid].float())
            token_entropy = -(
                soft_codes * torch.log(soft_codes.clamp_min(1e-10))
            ).sum(dim=-1).mean()
            batch_probs = soft_codes.mean(dim=0)
            batch_entropy = -(
                batch_probs * torch.log(batch_probs.clamp_min(1e-10))
            ).sum()
            # ``token_entropy - batch_entropy`` is the usual diversity reward,
            # but it is negative when utilization improves and made the logged
            # VQ "loss" negative.  Adding log(K) turns the same objective into
            # a non-negative deficit: zero is attained by decisive tokens whose
            # aggregate distribution is uniform.  The additive constant leaves
            # every optimization gradient unchanged.
            max_entropy = soft_codes.new_tensor(
                math.log(self.codebook_size)
            )
            entropy_deficit = (
                token_entropy + max_entropy - batch_entropy
            )
            entropy_loss = (
                self.diversity_weight * entropy_deficit.clamp_min(0.0)
            )
        else:
            commit_loss = guided.new_zeros(())
            usage = guided.new_zeros(self.codebook_size)
            entropy_loss = guided.new_zeros(())
        with torch.no_grad():
            self.last_usage.copy_(usage.to(self.last_usage.dtype))
            self.last_bucket_counts.zero_()
            if valid.any():
                valid_pre_tanh = pre_tanh.reshape(
                    -1, pre_tanh.shape[-1]
                )[valid].float()
                valid_z = z_flat[valid].float()
                self.last_pre_tanh_std.copy_(
                    valid_pre_tanh.std(unbiased=False)
                )
                self.last_quantized_std.copy_(
                    valid_z.std(unbiased=False)
                )
                self.last_pre_tanh_mean_per_dim.copy_(
                    valid_pre_tanh.mean(dim=0)
                )
                self.last_pre_tanh_std_per_dim.copy_(
                    valid_pre_tanh.std(dim=0, unbiased=False)
                )
                self.last_quantized_std_per_dim.copy_(
                    valid_z.std(dim=0, unbiased=False)
                )
                valid_indices = scalar_indices.reshape(
                    -1, scalar_indices.shape[-1]
                )[valid]
                for dimension, level in enumerate(self.levels):
                    counts = torch.bincount(
                        valid_indices[:, dimension], minlength=level
                    )
                    self.last_bucket_counts[dimension, :level].copy_(counts)
            else:
                self.last_pre_tanh_std.zero_()
                self.last_quantized_std.zero_()
                self.last_pre_tanh_mean_per_dim.zero_()
                self.last_pre_tanh_std_per_dim.zero_()
                self.last_quantized_std_per_dim.zero_()
        probs = usage / usage.sum().clamp_min(1.0)
        perplexity = torch.exp(
            -(probs * torch.log(probs.clamp_min(1e-10))).sum()
        )
        codes = flat_codes.reshape(B, T, 1)
        if not return_all_codes:
            codes = codes[..., 0]
        per_layer_losses = {
            "commitment": float(commit_loss.detach()),
            "entropy": float(entropy_loss.detach()),
            "orthogonality": 0.0,
        }
        return (z_q, commit_loss, entropy_loss, codes, perplexity,
                per_layer_losses)

    def forward(self, z_e, return_all_codes=True, mask=None):
        z_q, commit, entropy, codes, perplexity, losses = \
            self.quantize_guided(
                z_e, return_all_codes=return_all_codes, mask=mask
            )
        return z_q, commit + entropy, perplexity, codes, losses

    def codebook_vectors(self):
        """Return all deterministic dequantized vectors in code-index order."""
        codes = torch.arange(self.codebook_size, device=self.basis.device)
        digits = []
        for level, basis in zip(self.levels, self.basis):
            digits.append((codes // basis) % level)
        indices = torch.stack(digits, dim=-1).to(self.up.weight.dtype)
        max_index = self.level_sizes.to(indices.device, indices.dtype) - 1
        scalars = indices / max_index * 2.0 - 1.0
        return self.up(scalars)

    @torch.no_grad()
    def replace_unused_codes(self, *args, **kwargs):
        return 0

    @torch.no_grad()
    def get_per_layer_usage(self):
        active = int((self.last_usage > 0).sum().item())
        return [{
            "layer": 0,
            "active": active,
            "total": self.codebook_size,
            "ratio": active / self.codebook_size,
            "quant_scale": float(self.quant_scale.detach()),
            "pre_tanh_std": float(self.last_pre_tanh_std),
            "quantized_std": float(self.last_quantized_std),
            "pre_tanh_mean_per_dim": self.last_pre_tanh_mean_per_dim.tolist(),
            "pre_tanh_std_per_dim": self.last_pre_tanh_std_per_dim.tolist(),
            "quantized_std_per_dim": self.last_quantized_std_per_dim.tolist(),
            "bucket_counts": [
                self.last_bucket_counts[dimension, :level].tolist()
                for dimension, level in enumerate(self.levels)
            ],
            "buckets_all_nonzero": all(
                bool((self.last_bucket_counts[dimension, :level] > 0).all())
                for dimension, level in enumerate(self.levels)
            ),
        }]


class ResidualQuantizer(nn.Module):
    uses_ema = True

    def __init__(self, codebook_size=64, embedding_dim=256,
                 commitment_cost=0.25, epsilon=1e-5, ema_decay=0.9,
                 entropy_weight=1.0, orth_weight=0.5, inject_scale_init=1.0,
                 entropy_tau=0.1):
        super().__init__()
        self.num_quantizers = 1
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.ema_decay = ema_decay
        self.entropy_weight = entropy_weight
        self.orth_weight = orth_weight
        self.entropy_tau = entropy_tau

        self.embedding = nn.ParameterList([
            nn.Parameter(torch.empty(codebook_size, embedding_dim))
        ])
        nn.init.normal_(self.embedding[0], mean=0.0, std=1.0 / math.sqrt(embedding_dim))

        self.register_buffer('ema_cluster_size',
            torch.zeros(1, codebook_size, dtype=torch.float32))
        self.register_buffer('ema_dw',
            torch.zeros(1, codebook_size, embedding_dim, dtype=torch.float32))
        self.register_buffer('ema_decay_val', torch.tensor(ema_decay, dtype=torch.float32))

        self.inject_scales = nn.Parameter(
            torch.ones(1) * inject_scale_init
        )

    def forward(self, z_e, return_all_codes=True):
        B, T, D = z_e.shape
        device = z_e.device

        z_q, commit_loss, entropy_loss, orth_loss, codes, perp, per_layer_losses = \
            self._quantize_to_z_e(z_e, return_all_codes=return_all_codes)

        return z_q, commit_loss + entropy_loss + orth_loss, perp, codes, per_layer_losses

    @torch.no_grad()
    def refresh_codebook_from_ema(self):
        """Rebuild the codebook from the current EMA accumulators."""
        cluster_size = self.ema_cluster_size[0]
        n = cluster_size.sum()
        if n <= 0:
            return
        normalized_size = (cluster_size + self.epsilon) / (
            n + self.epsilon * self.codebook_size
        ) * n
        codebook = self.ema_dw[0] / (
            normalized_size.unsqueeze(1) + self.epsilon
        )
        self.embedding[0].copy_(F.normalize(codebook, dim=-1))

    def _quantize_to_z_e(self, z_e, return_all_codes=True):
        """Core single-layer VQ on a query tensor.

        Returns:
          z_q          : [B, T, D] codebook reconstruction with straight-through
          commit_loss  : scalar
          entropy_loss : scalar
          orth_loss    : scalar (always 0 for a single layer)
          codes        : [B, T, 1] (or [B, T] if return_all_codes=False)
          perplexity   : scalar
          per_layer_losses : dict
        """
        B, T, D = z_e.shape
        device = z_e.device
        flat_z_e = z_e.reshape(-1, D)
        codebook = self.embedding[0]
        # L2 normalize both query and codebook, then use cosine distance.
        # This decouples entropy magnitude from embedding dimension scaling.
        z_e_norm = F.normalize(flat_z_e, dim=-1)
        cb_norm = F.normalize(codebook, dim=-1)
        dist = 2 - 2 * (z_e_norm @ cb_norm.T)  # cosine distance in [0, 4]

        indices = dist.argmin(dim=-1)
        one_hot = F.one_hot(indices, self.codebook_size).to(codebook.dtype)

        z_q = one_hot @ cb_norm
        z_q = z_q.reshape(B, T, D)

        commit_loss = self.commitment_cost * F.mse_loss(
            z_e.detach(), z_q
        )

        # EMA statistics are state, not part of the autograd graph.  The
        # codebook itself must be refreshed only after backward has consumed
        # this forward's graph; changing it here invalidates saved tensors in
        # F.normalize/matmul backward.
        if self.training:
            with torch.no_grad():
                usage = one_hot.sum(dim=0)
                self.ema_cluster_size.data[0] = (
                    self.ema_decay_val * self.ema_cluster_size[0]
                    + (1 - self.ema_decay_val) * usage
                )
                # EMA tracks directions (normalized), not raw vectors
                dw = one_hot.T @ z_e_norm
                self.ema_dw.data[0] = (
                    self.ema_decay_val * self.ema_dw[0]
                    + (1 - self.ema_decay_val) * dw
                )

        z_q = z_q * self.inject_scales[0]
        z_q = z_e + (z_q - z_e).detach()

        codes = indices.reshape(B, T, 1)

        max_entropy = math.log(self.codebook_size)
        probs = one_hot.mean(dim=0)
        probs = probs + self.epsilon
        layer_entropy = -(probs * torch.log(probs)).sum()
        entropy_loss = self.entropy_weight * (max_entropy - layer_entropy)
        orth_loss = torch.tensor(0.0, device=device)
        perplexity = torch.exp(-(probs * torch.log(probs + self.epsilon)).sum())

        per_layer_losses = {
            "commitment": commit_loss.item(),
            "entropy": entropy_loss.item(),
            "orthogonality": orth_loss.item(),
        }

        if not return_all_codes:
            codes = codes[..., 0]

        return z_q, commit_loss, entropy_loss, orth_loss, codes, perplexity, per_layer_losses

    def quantize_guided(self, guided, return_all_codes=True, mask=None):
        """Nearest-code lookup + EMA update on the guided (projected) representation.

        Returns a straight-through codebook vector (guided + (z_q - guided).detach())
        so gradients can flow back to the encoder, plus codes/perplexity/losses.
        NOTE: NOT under no_grad -- soft entropy + commitment must keep gradients
        flowing to `guided` (and through it to encoder/projections). Discrete
        index/EMA bookkeeping is isolated inside no_grad blocks below.
        Used by sequential ResidualRQ for design C.
        """
        B, T, D = guided.shape
        device = guided.device
        flat = guided.reshape(-1, D)
        codebook = self.embedding[0]
        # L2 normalize both query and codebook, then use cosine distance.
        # Keeps soft-entropy logits bounded regardless of embedding dimension.
        guided_norm = F.normalize(flat, dim=-1)
        cb_norm = F.normalize(codebook, dim=-1)
        dist = 2 - 2 * (guided_norm @ cb_norm.T)  # cosine distance in [0, 4]
        with torch.no_grad():
            indices = dist.argmin(dim=-1)
            one_hot = F.one_hot(indices, self.codebook_size).to(codebook.dtype)
            z_q_raw = one_hot @ cb_norm
            z_q_raw = z_q_raw.reshape(B, T, D)

        # Straight-through: keep encoder/projection gradients while using the
        # selected codebook output numerically. Codebook gradients come from
        # the differentiable soft-entropy and orthogonality terms below.
        z_q_ste = guided + (z_q_raw - guided).detach()

        # Commitment pulls guided toward the selected codebook vector. Its
        # gradient flows to the encoder/projection through `guided` only.
        commit_loss = self.commitment_cost * F.mse_loss(guided, z_q_raw.detach())

        # Soft entropy over distances: differentiable, drives code diversity.
        soft_logits = -dist / max(1e-6, self.entropy_tau)
        soft_probs = F.softmax(soft_logits, dim=-1)
        soft_entropy = -(soft_probs * torch.log(soft_probs + self.epsilon)).sum(dim=-1).mean()
        # Do not rewrite the codebook until backward is complete.  This
        # forward contributes differentiable entropy/orthogonality terms that
        # save the codebook for backward, so an in-place EMA refresh here causes
        # autograd's "modified by an inplace operation" failure.
        if self.training:
            with torch.no_grad():
                usage = one_hot.sum(dim=0)
                self.ema_cluster_size.data[0] = (
                    self.ema_decay_val * self.ema_cluster_size[0]
                    + (1 - self.ema_decay_val) * usage
                )
                # EMA tracks directions (normalized), not raw vectors
                dw = one_hot.T @ guided_norm
                self.ema_dw.data[0] = (
                    self.ema_decay_val * self.ema_dw[0]
                    + (1 - self.ema_decay_val) * dw
                )

        max_entropy = math.log(self.codebook_size)
        entropy_loss = self.entropy_weight * (max_entropy - soft_entropy)
        hard_probs = one_hot.mean(dim=0) + self.epsilon
        perplexity = torch.exp(-(hard_probs * torch.log(hard_probs)).sum())

        codes = indices.reshape(B, T, 1)
        if not return_all_codes:
            codes = codes[..., 0]
        per_layer_losses = {
            "commitment": commit_loss.item(),
            "entropy": entropy_loss.item(),
            "orthogonality": 0.0,
        }
        return z_q_ste, commit_loss, entropy_loss, codes, perplexity, per_layer_losses

    @torch.no_grad()
    def replace_unused_codes(self, z_e, mask=None, n_replace=16, threshold=1.0):
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)
        if mask is not None:
            flat = flat[mask.reshape(-1)]

        cluster_size = self.ema_cluster_size[0]
        dead_mask = cluster_size < threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        n_replace_actual = min(n_replace, n_dead, len(flat))
        if n_replace_actual <= 0:
            return 0

        rand_idx = torch.randperm(len(flat))[:n_replace_actual]
        new_vectors = flat[rand_idx]

        dead_indices = torch.where(dead_mask)[0][:n_replace_actual]
        # Codebook lives on the unit sphere; normalize replacement vectors too.
        self.embedding[0].data[dead_indices] = F.normalize(new_vectors, dim=-1)
        # Raise the post-reset EMA floor so a freshly seeded code survives the
        # housekeeping window (decay 0.99) without being immediately rejudged
        # dead while it is still being warmed up by usage.
        self.ema_cluster_size.data[0, dead_indices] = max(4.0, threshold)
        self.ema_dw.data[0, dead_indices] = F.normalize(new_vectors, dim=-1)

        return n_replace_actual

    @torch.no_grad()
    def get_per_layer_usage(self):
        n_active = (self.ema_cluster_size[0] > 0).sum().item()
        return [{
            "layer": 0,
            "active": n_active,
            "total": self.codebook_size,
            "ratio": n_active / self.codebook_size,
        }]


class ResidualRQ(nn.Module):
    """Sequential residual quantization with learned, unassigned projections.

    All layers learn through the shared reconstruction, quantization and
    aggregate auxiliary objectives. No layer receives a feature-path target.
    """

    def __init__(self, num_quantizers=3, codebook_size=64, embedding_dim=256,
                 commitment_cost=0.25, epsilon=1e-5, ema_decay=0.9,
                 entropy_weight=1.0, orth_weight=0.2, inject_scale_init=1.0,
                 entropy_tau=0.1,
                 quantizer_type="ema", quantizer_types=None,
                 fsq_levels=(4, 4, 4),
                 fsq_scale_init=1.0, fsq_diversity_weight=0.01,
                 fsq_diversity_tau=0.2):
        super().__init__()
        self._num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.orth_weight = orth_weight
        self.quantizer_type = str(quantizer_type).lower()
        if quantizer_types is None:
            if self.quantizer_type not in ("ema", "fsq"):
                raise ValueError(
                    "mixed quantizers require an explicit per-layer layout"
                )
            self.quantizer_types = (
                self.quantizer_type,
            ) * int(num_quantizers)
        else:
            self.quantizer_types = tuple(
                str(value).strip().lower() for value in quantizer_types
            )
            if len(self.quantizer_types) != int(num_quantizers):
                raise ValueError("quantizer_types must match num_quantizers")
            if any(value not in ("ema", "fsq")
                   for value in self.quantizer_types):
                raise ValueError(
                    "quantizer_types entries must be 'ema' or 'fsq'"
                )
            inferred = (
                self.quantizer_types[0]
                if len(set(self.quantizer_types)) == 1 else "mixed"
            )
            if inferred != self.quantizer_type:
                raise ValueError(
                    "quantizer_type disagrees with per-layer layout"
                )
        if "fsq" in self.quantizer_types:
            if math.prod(tuple(fsq_levels)) != codebook_size:
                raise ValueError(
                    "FSQ levels product must equal codebook_size: {} != {}"
                    .format(math.prod(tuple(fsq_levels)), codebook_size)
                )
        layers = []
        for layer_type in self.quantizer_types:
            if layer_type == "fsq":
                layers.append(FSQQuantizer(
                    embedding_dim=embedding_dim,
                    levels=fsq_levels,
                    commitment_cost=0.0,
                    inject_scale_init=inject_scale_init,
                    quant_scale_init=fsq_scale_init,
                    diversity_weight=fsq_diversity_weight,
                    diversity_tau=fsq_diversity_tau,
                ))
            else:
                layers.append(ResidualQuantizer(
                    codebook_size=codebook_size,
                    embedding_dim=embedding_dim,
                    commitment_cost=commitment_cost,
                    epsilon=epsilon,
                    ema_decay=ema_decay,
                    entropy_weight=entropy_weight,
                    orth_weight=orth_weight,
                    inject_scale_init=inject_scale_init,
                    entropy_tau=entropy_tau,
                ))
        self.rq_layers = nn.ModuleList(layers)
        # Learned residual projections without prescribed feature targets.
        self.residual_projections = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(num_quantizers)
        ])
        # Cached per-forward VQ loss breakdown (scalars), updated in forward().
        # Read by the trainer for logging without changing the return signature.
        self.last_vq_commit = None
        self.last_vq_entropy = None
        self.last_vq_orth = None
        self.last_guided_list = None
        self.last_quantizer_inputs = None
        self.last_quantized_layers = None
        self.last_reset_by_layer = None

    def forward(self, z_e, mask=None, return_all_codes=True,
                quantizer_perturb_mode="none", quantizer_perturb_sigma=0.0):
        """Sequential residual quantization through learned projections.

        Args:
          z_e: [B, T, D] shared-transformer mixed representation.
          mask: [B, T] optional; used to zero out the decoder-side contribution and
                restrict EMA updates to real steps when provided by caller via .detach().
        Returns:
          z_q, vq_loss, perplexity, codes, per_layer_losses, guided_list
        """
        B, T, D = z_e.shape
        device = z_e.device
        z_q = z_e.new_zeros(B, T, D)
        residual = z_e
        codes_list = []
        perp_sum = 0.0
        vq_loss = z_e.new_zeros(())
        commit_sum = z_e.new_zeros(())
        entropy_sum = z_e.new_zeros(())
        guided_list = []
        quantizer_inputs = []
        quantized_layers = []
        perturb_mode = str(quantizer_perturb_mode).lower()
        perturb_sigma = float(quantizer_perturb_sigma)
        if perturb_mode not in ("none", "iid", "batch_shift"):
            raise ValueError(
                "quantizer_perturb_mode must be none, iid, or batch_shift"
            )
        if perturb_sigma < 0.0:
            raise ValueError("quantizer_perturb_sigma must be non-negative")

        for l in range(self._num_quantizers):
            guided = self.residual_projections[l](residual)
            guided_list.append(guided)
            quantizer_input = guided
            # Mixed-stack M5 noise is a direct FSQ-input intervention.  EMA
            # layers keep a clean input, so commitment and EMA updates never
            # consume the perturbation intended for FSQ discretization.
            if (self.training and self.quantizer_types[l] == "fsq"
                    and perturb_mode != "none" and perturb_sigma > 0.0):
                if perturb_mode == "iid":
                    noise = torch.randn_like(guided)
                else:
                    noise = torch.randn(
                        1, 1, D, dtype=guided.dtype, device=guided.device
                    )
                candidate = guided + perturb_sigma * noise
                quantizer_input = (
                    torch.where(
                        mask.unsqueeze(-1).bool(), candidate, guided
                    ) if mask is not None else candidate
                )
            quantizer_inputs.append(quantizer_input)
            z_q_raw, commit_l, ent_l, codes_l, perp_l, _ = \
                self.rq_layers[l].quantize_guided(
                    quantizer_input, return_all_codes=True, mask=mask
                )
            z_q_l = z_q_raw * self.rq_layers[l].inject_scales[0]
            quantized_layers.append(z_q_l)

            z_q = z_q + z_q_l
            residual = residual - z_q_l.detach()
            codes_list.append(codes_l)
            vq_loss = vq_loss + commit_l + ent_l
            commit_sum = commit_sum + commit_l
            entropy_sum = entropy_sum + ent_l
            perp_sum = perp_sum + perp_l

        codes = torch.cat(codes_list, dim=-1)  # [B, T, L]
        perplexity = perp_sum / max(1, self._num_quantizers)

        # Cross-layer codebook orthogonality (kept from design B)
        if self.orth_weight > 0 and self._num_quantizers > 1:
            orth_loss = 0.0
            n_pairs = 0
            for i in range(self._num_quantizers):
                for j in range(i + 1, self._num_quantizers):
                    cb_i = self.codebook_vectors(i)
                    cb_j = self.codebook_vectors(j)
                    cb_i_norm = F.normalize(cb_i, dim=-1)
                    cb_j_norm = F.normalize(cb_j, dim=-1)
                    sim = cb_i_norm @ cb_j_norm.T
                    orth_loss = orth_loss + sim.pow(2).mean()
                    n_pairs += 1
            orth_loss = self.orth_weight * orth_loss / max(1, n_pairs)
        else:
            orth_loss = z_e.new_zeros(())
        vq_loss = vq_loss + orth_loss

        # Cache breakdown for the trainer's per-step/per-epoch logging.
        # The trainer applies separate coefficients to these terms.  Preserve
        # their graph here; logging calls ``item()`` only after backward.
        self.last_vq_commit = commit_sum
        self.last_vq_entropy = entropy_sum
        self.last_vq_orth = orth_loss
        # Cache the per-layer guided projections so the trainer can reset dead
        # codes in the same space the quantizer actually looked up (design domain).
        self.last_guided_list = [g.detach() for g in guided_list]
        self.last_quantizer_inputs = [
            value.detach() for value in quantizer_inputs
        ]
        # Auxiliary routing needs the differentiable quantized contribution.
        self.last_quantized_layers = quantized_layers

        if not return_all_codes:
            codes = codes[..., 0]
        return z_q, vq_loss, perplexity, codes, None, guided_list

    @torch.no_grad()
    def replace_unused_codes(self, z_e, mask=None, n_replace=16, threshold=1.0,
                             guided=None):
        if not self.has_ema:
            self.last_reset_by_layer = [0] * self._num_quantizers
            return 0
        total_replaced = 0
        per_layer = []
        for l in range(self._num_quantizers):
            if not getattr(self.rq_layers[l], "uses_ema", False):
                per_layer.append(0)
                continue
            if guided is not None:
                if guided[l].shape[:-1] == z_e.shape[:-1]:
                    rep_vec = guided[l]
                else:
                    rep_vec = z_e
            else:
                rep_vec = z_e
            n = self.rq_layers[l].replace_unused_codes(
                rep_vec, mask=mask, n_replace=n_replace, threshold=threshold)
            total_replaced += n
            per_layer.append(n)
        self.last_reset_by_layer = per_layer
        return total_replaced

    def set_entropy_tau(self, tau):
        """Apply a per-epoch entropy softmax temperature to every quantizer layer."""
        for l in range(self._num_quantizers):
            if hasattr(self.rq_layers[l], "entropy_tau"):
                self.rq_layers[l].entropy_tau = float(tau)

    @torch.no_grad()
    def refresh_ema_codebooks(self):
        """Refresh only EMA layers after backward has released the graph."""
        for layer in self.rq_layers:
            if getattr(layer, "uses_ema", False):
                layer.refresh_codebook_from_ema()

    @property
    def uses_ema(self):
        """Compatibility signal: true when any EMA layer needs housekeeping."""
        return self.has_ema

    @property
    def has_ema(self):
        return any(getattr(layer, "uses_ema", False) for layer in self.rq_layers)

    @property
    def has_fsq(self):
        return any(not getattr(layer, "uses_ema", False)
                   for layer in self.rq_layers)

    def codebook_vectors(self, layer):
        quantizer = self.rq_layers[layer]
        if hasattr(quantizer, "codebook_vectors"):
            return quantizer.codebook_vectors()
        return quantizer.embedding[0]

    def code_usage_metrics(self, codes, mask=None):
        return summarize_code_usage(codes, self.codebook_size, mask=mask)

    @torch.no_grad()
    def get_per_layer_usage(self):
        usage = []
        for l in range(self._num_quantizers):
            u = self.rq_layers[l].get_per_layer_usage()[0]
            u["layer"] = l
            u["quantizer_type"] = self.quantizer_types[l]
            usage.append(u)
        return usage

    @property
    def inject_scales(self):
        return torch.cat([l.inject_scales for l in self.rq_layers])

    @property
    def ema_cluster_size(self):
        ema_layers = [layer for layer in self.rq_layers
                      if getattr(layer, "uses_ema", False)]
        if not ema_layers:
            return torch.stack([layer.last_usage for layer in self.rq_layers])
        return torch.cat([layer.ema_cluster_size for layer in ema_layers])

    @property
    def embedding(self):
        ema_layers = [layer for layer in self.rq_layers
                      if getattr(layer, "uses_ema", False)]
        if not ema_layers:
            raise AttributeError("FSQ has no learned embedding codebook")
        return nn.ParameterList([layer.embedding[0] for layer in ema_layers])

    @property
    def ema_dw(self):
        ema_layers = [layer for layer in self.rq_layers
                      if getattr(layer, "uses_ema", False)]
        if not ema_layers:
            raise AttributeError("FSQ has no EMA accumulator")
        return torch.cat([layer.ema_dw for layer in ema_layers])

    @property
    def ema_decay_val(self):
        ema_layers = [layer for layer in self.rq_layers
                      if getattr(layer, "uses_ema", False)]
        if not ema_layers:
            raise AttributeError("FSQ has no EMA decay")
        return ema_layers[0].ema_decay_val

    @ema_decay_val.setter
    def ema_decay_val(self, val):
        for layer in self.rq_layers:
            if getattr(layer, "uses_ema", False):
                layer.ema_decay_val.data.copy_(val)

    @property
    def num_quantizers(self):
        return self._num_quantizers

    @num_quantizers.setter
    def num_quantizers(self, val):
        self._num_quantizers = val
