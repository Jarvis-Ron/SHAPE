"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rqvae_modules import ResidualRQ, SEMANTIC_TARGET_KEYS


class SurprisalEncoder(nn.Module):
    """Multi-path encoder over the city-invariant event-feature contract.

    The v2 sparse-grid event archive is 136-dimensional; column 135 is an
    explicit missing-static/latent flag and is excluded from the encoder.
    but the encoder consumes only:

    - Time2Vec + delta-time (0:17), plus periodic weekday/hour derived from
      the raw values in columns 17 and 18;
    - the shared M1 grid latent (19:83);
    - a learned categorical POI embedding derived from raw category column
      124 (the archived 83:91 periodic encoding is intentionally ignored);
    - optional spatial/semantic surprisal (125:127); and
    - relative jump distance (127).

    Venue Word2Vec (91:123), labels/grid ids (123:125), and city/user summary
    statistics (128:135) never enter the encoder.  This is Route A's pure
    encoder-side change: existing event archives do not need to be rebuilt.

    Returns both the main output and the feature paths for RQ layer injection.
    """

    def __init__(self, d_model=256, d_hidden=64, d_proj=16,
                 use_surprisal=False):
        super().__init__()
        self.use_surprisal = bool(use_surprisal)

        # 16-D Time2Vec + delta-time + weekday/hour sin/cos = 21 dims.
        self.time_proj = nn.Linear(21, d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_scale = nn.Parameter(torch.ones(1) * 2.0)

        # Category ids are nominal rather than periodic: numeric neighbors do
        # not imply semantic neighbors.  Consume the already archived raw id
        # at column 124 through a learned 13x8 embedding, avoiding any data
        # rebuild while removing the false geometry of sin/cos(category id).
        self.spatial_proj = nn.Linear(64, d_model)
        self.poi_embedding = nn.Embedding(13, 8)
        self.poi_proj = nn.Linear(8, d_model)

        # Surprisal paths
        self.spa_mlp = nn.Sequential(
            nn.Linear(1, d_hidden), nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True), nn.Linear(d_hidden, d_proj),
        )
        self.sem_mlp = nn.Sequential(
            nn.Linear(1, d_hidden), nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True), nn.Linear(d_hidden, d_proj),
        )
        self.spa_residual = nn.Linear(d_proj, d_model)
        self.sem_residual = nn.Linear(d_proj, d_model)

        # City-invariant motion path: delta-time and relative jump distance.
        self.behavior = nn.Sequential(
            nn.Linear(2, d_hidden), nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True), nn.Linear(d_hidden, d_proj),
        )
        self.behavior_residual = nn.Linear(d_proj, d_model)

    @staticmethod
    def _cyclic_time_features(x):
        """Return weekday/hour sin/cos without using city-specific scaling."""
        weekday_angle = 2.0 * math.pi * x[..., 17:18]
        hour_angle = 2.0 * math.pi * x[..., 18:19] / 24.0
        return torch.cat([
            torch.sin(weekday_angle), torch.cos(weekday_angle),
            torch.sin(hour_angle), torch.cos(hour_angle),
        ], dim=-1)

    @staticmethod
    def _poi_category_ids(x):
        """Decode the archived raw POI category into safe ids 0..12."""
        raw = torch.nan_to_num(x[..., 124], nan=0.0, posinf=12.0, neginf=0.0)
        return raw.round().long().clamp_(0, 12)

    def forward(self, x):
        if x.shape[-1] < 128:
            raise ValueError(
                "Route-A encoder expects at least 128 event features, got %d"
                % x.shape[-1]
            )

        # Time path: Time2Vec(16), delta-time(1), weekday/hour sin/cos(4).
        time_feat = torch.cat([
            x[..., :17], self._cyclic_time_features(x)
        ], dim=-1)
        time_out = self.time_proj(time_feat)
        time_out = self.time_norm(time_out) * self.time_scale

        # Shared grid semantics and learned city-independent POI category.
        spatial_out = self.spatial_proj(x[..., 19:83])
        poi_out = self.poi_proj(
            self.poi_embedding(self._poi_category_ids(x))
        )

        # Surprisal is useful for offline discovery but unavailable to a
        # generator before the next event is known.  Keep it opt-in.
        if self.use_surprisal:
            spa_emb = self.spa_mlp(x[..., 125:126])
            sem_emb = self.sem_mlp(x[..., 126:127])
            spa_out = self.spa_residual(spa_emb)
            sem_out = self.sem_residual(sem_emb)
        else:
            spa_out = torch.zeros_like(time_out)
            sem_out = torch.zeros_like(time_out)

        behavior_feat = torch.cat([x[..., 16:17], x[..., 127:128]], dim=-1)
        behavior_out = self.behavior_residual(self.behavior(behavior_feat))

        out = (
            time_out + spatial_out + poi_out + spa_out + sem_out
            + behavior_out
        )

        # Layer 2 uses POI as a stable target and adds surprisal only when the
        # switch is enabled.  Thus all three RQ semantic paths remain active.
        feature_paths = {
            "time": time_out,
            "spatial": spatial_out,
            "poi": poi_out,
            "spa_surprisal": spa_out,
            "sem_surprisal": sem_out,
            "behavioral": behavior_out,
        }
        return out, feature_paths


class RQVAETransformerExtractor(nn.Module):
    """RQVAE Transformer with 3-layer Residual Quantization.

    Output 9-tuple: (H_hat, vq_loss, perplexity, z_e, z_q, codes, future_loss,
                     temporal_loss, semantic_loss)
    where codes: [B, T, 3] (3-layer RQ indices)
    """

    def __init__(self, d_model=256, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1,
                 num_quantizers=3, codebook_size=512, commit_beta=0.25,
                 entropy_weight=1.0, orth_weight=0.1, inject_scale_init=1.0,
                 lambda_spa=0.5, lambda_sem=0.5, gamma=2.0,
                 temporal_self_loop_w=0.05, temporal_ze_sim_w=0.01,
                 temporal_entropy_w=0.1, entropy_tau=0.1,
                 quantizer_type="fsq", quantizer_types=None,
                 fsq_levels=(4, 4, 4),
                 fsq_scale_init=1.0, fsq_diversity_weight=0.01,
                 fsq_diversity_tau=0.2, use_surprisal=False,
                 jump_aux_enabled=False,
                 poi_aux_enabled=False, num_poi_categories=13,
                 jump_bin_thresholds=(1e-8, 0.05, 0.15, 0.30),
                 time_aux_enabled=False,
                 time_bin_boundaries=(5.0, 10.0, 14.0, 18.0, 22.0),
                 encoder_perturb_mode="none", encoder_perturb_sigma=0.0,
                 aux_routing="aggregate", injection_mode="layered"):
        super().__init__()

        self.d_model = d_model
        self.lambda_spa = lambda_spa
        self.lambda_sem = lambda_sem
        self.gamma = gamma
        self.temporal_self_loop_w = temporal_self_loop_w
        self.temporal_ze_sim_w = temporal_ze_sim_w
        self.temporal_entropy_w = temporal_entropy_w
        self.entropy_tau = entropy_tau
        self.use_surprisal = bool(use_surprisal)
        self.jump_aux_enabled = bool(jump_aux_enabled)
        self.poi_aux_enabled = bool(poi_aux_enabled)
        self.time_aux_enabled = bool(time_aux_enabled)
        self.aux_routing = str(aux_routing).lower()
        if self.aux_routing not in ("aggregate", "layer_targeted"):
            raise ValueError(
                "aux_routing must be aggregate or layer_targeted"
            )
        self.encoder_perturb_mode = str(encoder_perturb_mode).lower()
        self.encoder_perturb_sigma = float(encoder_perturb_sigma)
        if self.encoder_perturb_mode not in ("none", "iid", "batch_shift"):
            raise ValueError(
                "encoder_perturb_mode must be none, iid, or batch_shift"
            )
        if self.encoder_perturb_sigma < 0.0:
            raise ValueError("encoder_perturb_sigma must be non-negative")
        if self.encoder_perturb_mode == "none" \
                and self.encoder_perturb_sigma != 0.0:
            raise ValueError(
                "encoder_perturb_sigma must be zero when perturbation is disabled"
            )
        self._gradient_checkpointing = False

        # Step 1: SurprisalEncoder
        self.surprisal_encoder = SurprisalEncoder(
            d_model, use_surprisal=self.use_surprisal
        )

        # Step 2: TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
        )

        # Step 3: semantic-guided sequential residual quantizer (design C)
        self.rq_layer = ResidualRQ(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            embedding_dim=d_model,
            commitment_cost=commit_beta,
            entropy_weight=entropy_weight,
            orth_weight=orth_weight,
            inject_scale_init=inject_scale_init,
            semantic_keys=SEMANTIC_TARGET_KEYS,
            entropy_tau=entropy_tau,
            quantizer_type=quantizer_type,
            quantizer_types=quantizer_types,
            fsq_levels=fsq_levels,
            fsq_scale_init=fsq_scale_init,
            fsq_diversity_weight=fsq_diversity_weight,
            fsq_diversity_tau=fsq_diversity_tau,
            injection_mode=injection_mode,
        )

        # Step 4: TransformerDecoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_decoder_layers,
        )

        # Step 5: Output projection -> 64-dim H
        self.output_proj = nn.Linear(d_model, 64)

        # Future prediction heads
        self.delta_head = nn.Linear(d_model, 1)
        self.dist_head = nn.Linear(d_model, 1)
        if self.jump_aux_enabled:
            thresholds = tuple(float(value) for value in jump_bin_thresholds)
            if len(thresholds) != 4 or tuple(sorted(thresholds)) != thresholds:
                raise ValueError(
                    "jump_bin_thresholds must contain four increasing values"
                )
            self.register_buffer(
                "jump_bin_thresholds", torch.tensor(thresholds)
            )
            self.jump_head = nn.Linear(d_model, len(thresholds) + 1)
        else:
            self.jump_head = None
        self.poi_head = (
            nn.Linear(d_model, int(num_poi_categories))
            if self.poi_aux_enabled else None
        )
        if self.time_aux_enabled:
            boundaries = tuple(float(value) for value in time_bin_boundaries)
            if (len(boundaries) != 5
                    or any(left >= right for left, right in zip(
                        boundaries, boundaries[1:]
                    ))):
                raise ValueError(
                    "time_bin_boundaries must contain five increasing values"
                )
            self.register_buffer(
                "time_bin_boundaries", torch.tensor(boundaries)
            )
            self.time_head = nn.Linear(d_model, len(boundaries) + 1)
        else:
            self.time_head = None

    def encode(self, x, mask=None):
        emb, feature_paths = self.surprisal_encoder(x)
        if mask is not None:
            # Replace virtual positions with zeros
            emb = emb * mask.unsqueeze(-1).float()
        z_e = self.transformer_encoder(emb)
        return z_e

    def decode(self, z_q, mask=None):
        if mask is not None:
            z_d = self.transformer_decoder(z_q * mask.unsqueeze(-1).float())
        else:
            z_d = self.transformer_decoder(z_q)
        H_hat = self.output_proj(z_d)
        return H_hat

    def perturb_encoder_latents(self, z_e, mask=None):
        """Apply the registered M5 shift immediately before quantization."""
        if (not self.training or self.encoder_perturb_mode == "none"
                or self.encoder_perturb_sigma == 0.0):
            return z_e
        if self.encoder_perturb_mode == "iid":
            noise = torch.randn_like(z_e)
        else:
            noise = torch.randn(
                1, 1, z_e.shape[-1], dtype=z_e.dtype, device=z_e.device
            )
        perturbed = z_e + self.encoder_perturb_sigma * noise
        if mask is not None:
            perturbed = torch.where(mask.unsqueeze(-1).bool(), perturbed, z_e)
        return perturbed

    @staticmethod
    def calibrate_encoder_latents(z_e, encoder_affine=None):
        """Apply evaluation-only per-city affine standardization for M5 R-B."""
        if encoder_affine is None:
            return z_e
        mean, scale = encoder_affine
        mean = torch.as_tensor(mean, dtype=z_e.dtype, device=z_e.device)
        scale = torch.as_tensor(scale, dtype=z_e.dtype, device=z_e.device)
        if mean.shape != (z_e.shape[-1],) or scale.shape != (z_e.shape[-1],):
            raise ValueError("encoder affine statistics must have shape [d_model]")
        return (z_e - mean) / scale.clamp_min(1e-6)

    def forward(self, x, attention_mask=None, encoder_affine=None):
        B, T, _ = x.shape
        device = x.device

        # 1. Encode
        emb, feature_paths = self.surprisal_encoder(x)

        # 2. SHARED transformer encodes the full mixed representation (design A)
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1).float()
        z_e = self.transformer_encoder(emb)

        # 3. Semantic-guided sequential residual quantization (design C)
        if self.rq_layer.quantizer_type == "mixed":
            z_e_quant = self.calibrate_encoder_latents(z_e, encoder_affine)
            rq_perturb_mode = self.encoder_perturb_mode
            rq_perturb_sigma = self.encoder_perturb_sigma
        else:
            # Preserve the established homogeneous FSQ/EMA behavior exactly.
            z_e_quant = self.perturb_encoder_latents(z_e, attention_mask)
            z_e_quant = self.calibrate_encoder_latents(
                z_e_quant, encoder_affine
            )
            rq_perturb_mode = "none"
            rq_perturb_sigma = 0.0
        z_q, vq_loss, perplexity, codes, _, semantic_loss, guided_list = \
            self.rq_layer(
                z_e_quant, feature_paths=feature_paths, mask=attention_mask,
                quantizer_perturb_mode=rq_perturb_mode,
                quantizer_perturb_sigma=rq_perturb_sigma,
            )

        # 4. Decode
        if attention_mask is not None:
            z_d = self.transformer_decoder(z_q * attention_mask.unsqueeze(-1).float())
        else:
            z_d = self.transformer_decoder(z_q)
        H_hat = self.output_proj(z_d)

        # 5. Future loss on the full mixed z_e
        future_loss = self._compute_future_loss(x, z_e, attention_mask)

        # 6. Temporal loss on the full mixed z_e
        temporal_loss = self._compute_temporal_loss(codes, z_e, x, attention_mask)

        return H_hat, vq_loss, perplexity, z_e, z_q, codes, future_loss, temporal_loss, semantic_loss

    def compute_loss(self, x, H_hat, mask=None):
        """Surprisal-weighted MSE reconstruction loss."""
        H_target = x[..., 19:83]  # [B, T, 64]
        rho_spa = x[..., 125:126]
        rho_sem = x[..., 126:127]
        weight = (
            1.0
            + self.lambda_spa * (1.0 - rho_spa) ** self.gamma
            + self.lambda_sem * (1.0 - rho_sem) ** self.gamma
        )
        squared_error = (H_hat - H_target).pow(2).sum(dim=-1, keepdim=True)
        recon_loss = (weight * squared_error).mean()
        return recon_loss

    def compute_recon_loss(self, x, H_hat, mask=None):
        """Alias for compute_loss, used by joint model."""
        return self.compute_loss(x, H_hat, mask=mask)

    def compute_jump_aux_loss(self, x, z_q, mask=None):
        """Predict the fixed atlas jump bin from quantized event latents."""
        if self.jump_head is None:
            return z_q.sum() * 0.0
        jump = x[..., 127]
        valid = torch.isfinite(jump) & (jump >= 0.0) & (jump <= 1.0 + 1e-5)
        if mask is not None:
            valid &= mask.bool()
        if not valid.any():
            return z_q.sum() * 0.0
        targets = torch.bucketize(
            jump[valid], self.jump_bin_thresholds.to(jump.dtype), right=False
        )
        logits = self.jump_head(self._auxiliary_latent("jump", z_q)[valid])
        return F.cross_entropy(logits, targets)

    def compute_poi_aux_loss(self, x, z_q, mask=None):
        """Predict the shared 13-way POI category from quantized latents."""
        if self.poi_head is None:
            return z_q.sum() * 0.0
        raw = x[..., 124]
        rounded = raw.round()
        valid = torch.isfinite(raw) & (raw >= 0.0) & (raw <= 12.0) \
            & ((raw - rounded).abs() <= 1e-4)
        if mask is not None:
            valid &= mask.bool()
        if not valid.any():
            return z_q.sum() * 0.0
        targets = rounded[valid].long()
        logits = self.poi_head(self._auxiliary_latent("poi", z_q)[valid])
        return F.cross_entropy(logits, targets)

    def _auxiliary_latent(self, axis, aggregate):
        """Select aggregate or explicitly targeted mixed-stack contribution."""
        if self.aux_routing != "layer_targeted":
            return aggregate
        layers = self.rq_layer.last_quantized_layers
        if not layers:
            return aggregate
        if axis == "jump":
            return layers[0]
        if axis == "poi":
            ema_layers = [
                value for value, layer_type in zip(
                    layers, self.rq_layer.quantizer_types
                ) if layer_type == "ema"
            ]
            if ema_layers:
                return torch.stack(ema_layers, dim=0).sum(dim=0)
        return aggregate

    def time_bin_targets(self, hours):
        """Map local hour-of-day to six fixed, provenance-stable periods."""
        if not hasattr(self, "time_bin_boundaries"):
            raise RuntimeError("time auxiliary head is disabled")
        return torch.bucketize(
            hours.contiguous(), self.time_bin_boundaries.to(hours.dtype), right=True
        )

    def compute_time_aux_loss(self, x, z_q, mask=None):
        """Predict the fixed six-way local-time period from quantized events."""
        if self.time_head is None:
            return z_q.sum() * 0.0
        hours = x[..., 18]
        valid = torch.isfinite(hours) & (hours >= 0.0) & (hours < 24.0)
        if mask is not None:
            valid &= mask.bool()
        if not valid.any():
            return z_q.sum() * 0.0
        logits = self.time_head(z_q[valid])
        return F.cross_entropy(logits, self.time_bin_targets(hours[valid]))

    def _compute_future_loss(self, x, z_e, mask):
        B, T = x.shape[:2]
        device = x.device

        delta_target = torch.zeros(B, 1, device=device)
        dist_target = torch.zeros(B, 1, device=device)
        has_next = torch.zeros(B, 1, dtype=torch.bool, device=device)
        cur_indices = torch.full((B,), T - 1, dtype=torch.long, device=device)

        if mask is not None:
            for b in range(B):
                real_steps = torch.where(mask[b])[0]
                if len(real_steps) >= 2:
                    cur_t = real_steps[-2].item()
                    next_t = real_steps[-1].item()
                    delta_target[b, 0] = float(next_t - cur_t)
                    # Column 127 is the GridSpec-derived physical jump in km,
                    # normalized by the city diagonal during preparation.
                    dist_target[b, 0] = x[b, next_t, 127].clamp(0.0, 1.0)
                    has_next[b, 0] = True
                    cur_indices[b] = cur_t
                elif len(real_steps) >= 1:
                    cur_indices[b] = real_steps[0].item()
        else:
            for b in range(B):
                if T >= 2:
                    cur_t = T - 2
                    next_t = T - 1
                    delta_target[b, 0] = 1.0
                    dist_target[b, 0] = x[b, next_t, 127].clamp(0.0, 1.0)
                    has_next[b, 0] = True
                    cur_indices[b] = cur_t

        z_cur = z_e[torch.arange(B), cur_indices]
        delta_pred = self.delta_head(z_cur)
        dist_pred = self.dist_head(z_cur)

        if has_next.any():
            return (F.mse_loss(delta_pred[has_next], delta_target[has_next])
                    + F.mse_loss(dist_pred[has_next], dist_target[has_next]))
        return torch.tensor(0.0, device=device)

    def _compute_temporal_loss(self, codes, z_e, x, mask):
        """Temporal regularization: penalize self-loops, encourage z_e diversity."""
        B, T, L = codes.shape  # [B, T, 3] for RQ
        device = codes.device
        is_virtual_mask = mask if mask is not None else torch.ones(B, T, dtype=torch.bool, device=device)

        # Use the first layer codes for temporal loss (most significant)
        codes_1 = codes[..., 0]

        # 1. Hard self-loop penalty
        hard_self_loop_count = 0.0
        n_valid = 0
        for b in range(B):
            real_steps = torch.where(is_virtual_mask[b])[0]
            if len(real_steps) >= 2:
                for i in range(len(real_steps) - 1):
                    if codes_1[b, real_steps[i]].item() == codes_1[b, real_steps[i + 1]].item():
                        hard_self_loop_count += 1.0
                    n_valid += 1.0
        hard_self_loop_rate = hard_self_loop_count / max(1.0, n_valid)

        # 2. z_e cosine similarity penalty
        z_e_norm = F.normalize(z_e, dim=-1)
        z_e_cos_sim = (z_e_norm[:, :-1] * z_e_norm[:, 1:]).sum(dim=-1)
        soft_mask = is_virtual_mask[:, :-1] & is_virtual_mask[:, 1:]
        z_e_cos_sim_mean = z_e_cos_sim[soft_mask].mean() if soft_mask.any() else z_e_cos_sim.mean()
        z_e_cos_sim_mean = torch.clamp(z_e_cos_sim_mean, min=0.0, max=1.0)

        # 3. Transition entropy: maximize diversity of code transitions
        trans_entropy_total = 0.0
        n_trans_entropy = 0
        V = self.rq_layer.codebook_size
        for b in range(B):
            real_steps = torch.where(is_virtual_mask[b])[0]
            if len(real_steps) >= 2:
                code_seq = codes_1[b, real_steps]
                from_codes = code_seq[:-1]
                to_codes = code_seq[1:]
                for fc_val in from_codes.unique():
                    mask_row = (from_codes == fc_val)
                    if mask_row.sum() < 1:
                        continue
                    targets = to_codes[mask_row]
                    trans_probs = torch.zeros(V, device=device)
                    for tc_val in targets.unique():
                        trans_probs[tc_val.item()] = (targets == tc_val).float().mean()
                    non_zero = trans_probs > 0
                    if non_zero.any():
                        row_entropy = -(trans_probs[non_zero] * torch.log(trans_probs[non_zero] + 1e-10)).sum()
                        trans_entropy_total += row_entropy
                        n_trans_entropy += 1.0

        if n_trans_entropy > 0:
            avg_trans_entropy = trans_entropy_total / n_trans_entropy
            trans_entropy_loss = -avg_trans_entropy
        else:
            trans_entropy_loss = torch.tensor(0.0, device=device)

        temporal_loss = (self.temporal_self_loop_w * hard_self_loop_rate
                         + self.temporal_ze_sim_w * z_e_cos_sim_mean
                         + self.temporal_entropy_w * trans_entropy_loss)
        return temporal_loss
