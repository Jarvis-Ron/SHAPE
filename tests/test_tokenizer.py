import math

import pytest
import torch
from torch.utils.data import DataLoader

from SHAPE.stage2.config import MODULE2_CONFIG
from SHAPE.stage2.data import AttentionBudgetBatchSampler
from SHAPE.stage2.model import SurprisalEncoder
from SHAPE.stage2.rqvae_modules import FSQQuantizer, ResidualRQ
from SHAPE.stage2.tokenizer import (
    build_tokenizer,
    load_tokenizer,
    require_quantizer_compatible,
)
from SHAPE.stage2.trainer import (
    _fsq_acceptance_failures,
    evaluate,
    train_epoch,
)




def test_attention_budget_batch_sampler_retains_complete_sequences():
    lengths = [100, 18, 20, 90, 10, 11]
    sampler = AttentionBudgetBatchSampler(
        lengths, max_batch_size=3, max_attention_cells=3000,
        shuffle=True, seed=7,
    )
    twin = AttentionBudgetBatchSampler(
        lengths, max_batch_size=3, max_attention_cells=3000,
        shuffle=True, seed=7,
    )
    batches = list(sampler)

    assert batches == list(twin)
    assert sorted(index for batch in batches for index in batch) == \
        list(range(len(lengths)))
    assert all(len(batch) <= 3 for batch in batches)
    for batch in batches:
        cells = AttentionBudgetBatchSampler.padded_attention_cells(
            [lengths[index] for index in batch]
        )
        assert cells <= 3000 or len(batch) == 1
    assert sampler.oversized_singletons == 2


def test_jump_auxiliary_head_is_opt_in_and_uses_fixed_bins():
    common = {
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 16,
        "num_quantizers": 1,
        "codebook_size": 8,
        "quantizer_type": "fsq",
        "fsq_levels": [2, 2, 2],
    }
    baseline = build_tokenizer({**common, "jump_aux_weight": 0.0})
    enabled = build_tokenizer({**common, "jump_aux_weight": 0.1})
    assert baseline.jump_head is None
    assert enabled.jump_head.out_features == 5

    sequence = torch.zeros(1, 5, 135)
    sequence[0, :, 127] = torch.tensor([0.0, 0.03, 0.10, 0.20, 0.40])
    z_q = torch.randn(1, 5, 8, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.bool)
    loss = enabled.compute_jump_aux_loss(sequence, z_q, mask)
    loss.backward()

    assert loss.item() > 0.0
    assert z_q.grad is not None
    assert enabled.jump_head.weight.grad is not None


def test_jump_auxiliary_loss_runs_through_train_and_eval_loops():
    config = {
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "num_quantizers": 1,
        "codebook_size": 8,
        "quantizer_type": "fsq",
        "fsq_levels": [2, 2, 2],
        "jump_aux_weight": 0.1,
        "poi_aux_weight": 0.05,
        "time_aux_weight": 0.05,
    }
    model = build_tokenizer(config)
    samples = []
    for offset in (0.0, 0.02):
        sequence = torch.randn(4, 135)
        sequence[:, 127] = torch.tensor([0.0, 0.04, 0.12, 0.35]) + offset
        sequence[:, 124] = torch.tensor([0.0, 3.0, 7.0, 12.0])
        sequence[:, 18] = torch.tensor([1.0, 7.0, 16.0, 23.0])
        samples.append({
            "x": sequence,
            "attention_mask": torch.ones(4, dtype=torch.bool),
        })
    loader = DataLoader(samples, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_metrics = train_epoch(
        model, loader, optimizer, torch.device("cpu"),
        replace_every=0, jump_aux_weight=0.1, poi_aux_weight=0.05,
        time_aux_weight=0.05,
    )
    eval_metrics = evaluate(model, loader, torch.device("cpu"))

    assert len(train_metrics) == 15
    assert train_metrics[6] > 0.0
    assert train_metrics[7] > 0.0
    assert train_metrics[8] > 0.0
    assert len(eval_metrics) == 13
    assert eval_metrics[8] > 0.0
    assert eval_metrics[9] > 0.0
    assert eval_metrics[10] > 0.0


def test_route_a_encoder_ignores_city_specific_and_target_only_columns():
    torch.manual_seed(7)
    encoder = SurprisalEncoder(d_model=16, use_surprisal=False).eval()
    source = torch.randn(2, 4, 135)
    shifted = source.clone()
    shifted[..., 91:124] = torch.randn_like(shifted[..., 91:124]) * 100
    shifted[..., 125:127] = torch.randn_like(shifted[..., 125:127]) * 100
    shifted[..., 128:135] = torch.randn_like(shifted[..., 128:135]) * 100

    source_out, source_paths = encoder(source)
    shifted_out, shifted_paths = encoder(shifted)

    assert torch.equal(source_out, shifted_out)
    for key in source_paths:
        assert torch.equal(source_paths[key], shifted_paths[key])


def test_route_a_poi_embedding_uses_category_not_legacy_periodic_columns():
    torch.manual_seed(19)
    encoder = SurprisalEncoder(d_model=16, use_surprisal=False).eval()
    source = torch.zeros(1, 3, 135)
    legacy_shift = source.clone()
    legacy_shift[..., 83:91] = 100.0
    category_shift = source.clone()
    category_shift[..., 124] = 7.0

    source_out, source_paths = encoder(source)
    legacy_out, legacy_paths = encoder(legacy_shift)
    category_out, category_paths = encoder(category_shift)

    assert torch.equal(source_out, legacy_out)
    assert torch.equal(source_paths["poi"], legacy_paths["poi"])
    assert not torch.allclose(source_out, category_out)
    assert not torch.allclose(source_paths["poi"], category_paths["poi"])


def test_poi_auxiliary_head_is_opt_in_and_backpropagates():
    common = {
        "d_model": 8, "nhead": 2, "num_encoder_layers": 1,
        "num_decoder_layers": 1, "dim_feedforward": 16,
        "num_quantizers": 1, "codebook_size": 8,
        "quantizer_type": "fsq", "fsq_levels": [2, 2, 2],
    }
    baseline = build_tokenizer({**common, "poi_aux_weight": 0.0})
    enabled = build_tokenizer({**common, "poi_aux_weight": 0.05})
    assert baseline.poi_head is None
    assert enabled.poi_head.out_features == 13

    sequence = torch.zeros(1, 4, 135)
    sequence[0, :, 124] = torch.tensor([0.0, 3.0, 7.0, 12.0])
    z_q = torch.randn(1, 4, 8, requires_grad=True)
    loss = enabled.compute_poi_aux_loss(
        sequence, z_q, torch.ones(1, 4, dtype=torch.bool)
    )
    loss.backward()

    assert loss.item() > 0.0
    assert z_q.grad is not None
    assert enabled.poi_head.weight.grad is not None


def test_time_auxiliary_head_uses_six_fixed_local_time_periods():
    common = {
        "d_model": 8, "nhead": 2, "num_encoder_layers": 1,
        "num_decoder_layers": 1, "dim_feedforward": 16, "dropout": 0.0,
        "num_quantizers": 1, "codebook_size": 8, "quantizer_type": "fsq",
        "fsq_levels": [2, 2, 2],
    }
    disabled = build_tokenizer(common)
    enabled = build_tokenizer({**common, "time_aux_weight": 0.05})
    assert disabled.time_head is None
    assert enabled.time_head.out_features == 6
    sequence = torch.zeros(1, 6, 135)
    sequence[0, :, 18] = torch.tensor([0.0, 5.0, 10.0, 14.0, 18.0, 22.0])
    targets = enabled.time_bin_targets(sequence[0, :, 18])
    assert targets.tolist() == [0, 1, 2, 3, 4, 5]
    z_q = torch.randn(1, 6, 8, requires_grad=True)
    loss = enabled.compute_time_aux_loss(
        sequence, z_q, torch.ones(1, 6, dtype=torch.bool)
    )
    loss.backward()
    assert enabled.time_head.weight.grad is not None


def test_time_aux_checkpoint_contract_cannot_claim_a_missing_head(tmp_path):
    config = {
        "d_model": 8, "nhead": 2, "num_encoder_layers": 1,
        "num_decoder_layers": 1, "dim_feedforward": 16, "dropout": 0.0,
        "num_quantizers": 1, "codebook_size": 8, "quantizer_type": "fsq",
        "fsq_levels": [2, 2, 2], "time_aux_weight": 0.05,
    }
    model = build_tokenizer(config)
    complete = tmp_path / "complete.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": {
        **MODULE2_CONFIG, **config,
    }}, complete)
    loaded, _, report = load_tokenizer(complete, torch.device("cpu"), "fsq")
    assert loaded.time_head is not None
    assert report["time_aux_weight"] == pytest.approx(0.05)

    missing = tmp_path / "missing.pth"
    state = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith("time_head")
    }
    torch.save({"model_state_dict": state, "config": {
        **MODULE2_CONFIG, **config,
    }}, missing)
    with pytest.raises(RuntimeError, match="required for encoding"):
        load_tokenizer(missing, torch.device("cpu"), "fsq")


def test_route_a_surprisal_is_explicit_opt_in_and_keeps_poi_fallback():
    torch.manual_seed(13)
    source = torch.randn(2, 4, 135)
    shifted = source.clone()
    shifted[..., 125:127] += 5.0
    disabled = SurprisalEncoder(d_model=16, use_surprisal=False).eval()
    enabled = SurprisalEncoder(d_model=16, use_surprisal=True).eval()

    disabled_out, disabled_paths = disabled(source)
    disabled_shifted, _ = disabled(shifted)
    enabled_out, enabled_paths = enabled(source)
    enabled_shifted, enabled_shifted_paths = enabled(shifted)

    assert torch.equal(disabled_out, disabled_shifted)
    assert torch.count_nonzero(disabled_paths["poi"]) > 0
    assert torch.count_nonzero(disabled_paths["spa_surprisal"]) == 0
    assert not torch.allclose(enabled_out, enabled_shifted)
    assert not torch.allclose(
        enabled_paths["spa_surprisal"],
        enabled_shifted_paths["spa_surprisal"],
    )


def test_route_a_time_cycles_and_three_semantic_targets_without_surprisal():
    times = torch.zeros(1, 2, 135)
    times[0, 1, 17] = 1.0
    times[0, 1, 18] = 24.0
    periodic = SurprisalEncoder._cyclic_time_features(times)
    assert torch.allclose(periodic[:, 0], periodic[:, 1], atol=1e-6)

    encoder = SurprisalEncoder(d_model=8, use_surprisal=False).eval()
    rq = ResidualRQ(
        num_quantizers=3,
        codebook_size=8,
        embedding_dim=8,
        quantizer_type="fsq",
        fsq_levels=(2, 2, 2),
    )
    _, paths = encoder(torch.randn(2, 3, 135))
    targets = rq.compute_semantic_target(paths)
    assert len(targets) == 3
    assert all(target.shape == (2, 3, 8) for target in targets)
    assert all(torch.count_nonzero(target) > 0 for target in targets)
    assert torch.equal(targets[2], paths["poi"])






def test_injection_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="injection_mode"):
        ResidualRQ(injection_mode="unknown")




def test_fsq_has_exact_64_code_interface_and_gradients():
    quantizer = FSQQuantizer(embedding_dim=8, levels=(4, 4, 4))
    guided = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([
        [False, True, True, True, True],
        [False, False, True, True, True],
    ])

    z_q, commitment, entropy, codes, perplexity, _ = \
        quantizer.quantize_guided(guided, mask=mask)

    assert z_q.shape == guided.shape
    assert codes.shape == (2, 5, 1)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 64
    assert quantizer.codebook_vectors().shape == (64, 8)
    assert commitment.item() == 0.0
    assert entropy.item() >= 0.0
    assert 1.0 <= perplexity.item() <= 64.0
    (z_q.square().mean() + entropy).backward()
    assert guided.grad is not None
    assert quantizer.down.weight.grad is not None
    assert quantizer.up.weight.grad is not None
    assert quantizer.log_quant_scale.grad is not None


def test_fsq_entropy_deficit_is_a_nonnegative_constant_shift():
    torch.manual_seed(11)
    quantizer = FSQQuantizer(
        embedding_dim=8, levels=(4, 4, 4), diversity_weight=0.03
    ).eval()
    guided = torch.randn(3, 5, 8, requires_grad=True)
    mask = torch.tensor([
        [True, True, True, True, False],
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])

    entropy = quantizer.quantize_guided(guided, mask=mask)[2]
    valid = mask.reshape(-1)
    projected = quantizer.down(quantizer.input_norm(guided))
    normalized = quantizer._normalize_bottleneck(projected, valid)
    z = torch.tanh(quantizer.quant_scale * 0.7 * normalized)
    soft_codes = quantizer._soft_code_probabilities(
        z.reshape(-1, z.shape[-1])[valid].float()
    )
    token_entropy = -(
        soft_codes * torch.log(soft_codes.clamp_min(1e-10))
    ).sum(dim=-1).mean()
    batch_probs = soft_codes.mean(dim=0)
    batch_entropy = -(
        batch_probs * torch.log(batch_probs.clamp_min(1e-10))
    ).sum()
    prior_reward = quantizer.diversity_weight * (
        token_entropy - batch_entropy
    )
    expected = prior_reward + (
        quantizer.diversity_weight * math.log(quantizer.codebook_size)
    )

    assert entropy.item() >= 0.0
    assert torch.allclose(entropy, expected, atol=1e-7)
    actual_grad = torch.autograd.grad(entropy, guided, retain_graph=True)[0]
    prior_grad = torch.autograd.grad(prior_reward, guided)[0]
    assert torch.allclose(actual_grad, prior_grad, atol=1e-7)


def test_fsq_statistics_ignore_padding_rows():
    torch.manual_seed(23)
    left = FSQQuantizer(embedding_dim=8, levels=(4, 4, 4))
    right = FSQQuantizer(embedding_dim=8, levels=(4, 4, 4))
    right.load_state_dict(left.state_dict())
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, False],
    ])
    valid_values = torch.randn(2, 5, 8)
    left_input = valid_values.clone()
    right_input = valid_values.clone()
    left_input[~mask] = torch.randn_like(left_input[~mask]) * 1e5
    right_input[~mask] = torch.randn_like(right_input[~mask]) * -1e5

    left_output = left.quantize_guided(left_input, mask=mask)[0]
    right_output = right.quantize_guided(right_input, mask=mask)[0]

    assert torch.allclose(left_output[mask], right_output[mask], atol=1e-6)
    assert torch.count_nonzero(left_output[~mask]) == 0
    assert torch.count_nonzero(right_output[~mask]) == 0
    assert torch.allclose(
        left.bottleneck_norm.running_mean,
        right.bottleneck_norm.running_mean,
    )
    assert torch.allclose(
        left.bottleneck_norm.running_var,
        right.bottleneck_norm.running_var,
    )
    assert torch.equal(left.last_bucket_counts, right.last_bucket_counts)


def test_residual_rq_fsq_keeps_three_layer_signature_without_ema():
    rq = ResidualRQ(
        num_quantizers=3,
        codebook_size=64,
        embedding_dim=8,
        quantizer_type="fsq",
        fsq_levels=(4, 4, 4),
    )
    z_e = torch.randn(2, 4, 8)
    mask = torch.tensor([
        [False, True, True, True],
        [False, False, True, True],
    ])

    z_q, loss, perplexity, codes, _, semantic, guided = rq(z_e, mask=mask)

    assert z_q.shape == z_e.shape
    assert codes.shape == (2, 4, 3)
    assert len(guided) == 3
    assert semantic is None
    assert loss.item() >= 0.0
    assert rq.last_vq_entropy.item() >= 0.0
    assert perplexity.ndim == 0
    assert rq.uses_ema is False
    assert rq.replace_unused_codes(z_e, mask=mask) == 0
    assert len(rq.code_usage_metrics(codes, mask)) == 3




def test_m5_fsq_iid_and_batch_shift_reach_prequantization_latents():
    common = {
        "d_model": 8, "nhead": 2, "num_encoder_layers": 1,
        "num_decoder_layers": 1, "dim_feedforward": 16,
        "num_quantizers": 1, "codebook_size": 8,
        "quantizer_type": "fsq", "fsq_levels": [2, 2, 2],
    }
    z_e = torch.zeros(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    iid = build_tokenizer({
        **common, "encoder_perturb_mode": "iid",
        "encoder_perturb_sigma": 0.1,
    }).train()
    torch.manual_seed(5)
    iid_shifted = iid.perturb_encoder_latents(z_e, mask)
    assert not torch.equal(iid_shifted[0, 0], iid_shifted[0, 1])
    assert torch.count_nonzero(iid_shifted[~mask]) == 0

    batch = build_tokenizer({
        **common, "encoder_perturb_mode": "batch_shift",
        "encoder_perturb_sigma": 0.1,
    }).train()
    torch.manual_seed(5)
    batch_shifted = batch.perturb_encoder_latents(z_e, mask)
    assert torch.equal(batch_shifted[0, 0], batch_shifted[0, 1])
    assert torch.equal(batch_shifted[0, 0], batch_shifted[1, 2])
    assert torch.count_nonzero(batch_shifted[~mask]) == 0
    batch.eval()
    assert torch.equal(batch.perturb_encoder_latents(z_e, mask), z_e)


def test_checkpoint_compatibility_rejects_legacy_and_partial_fsq():
    with pytest.raises(RuntimeError, match="complete FSQ"):
        require_quantizer_compatible({"rq_layer.embedding": torch.ones(1)})
    with pytest.raises(RuntimeError, match="input_norm.weight"):
        require_quantizer_compatible({
            "rq_layer.rq_layers.0.down.weight": torch.ones(1)
        })

    complete = {}
    for layer in range(3):
        for parameter in (
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
        ):
            complete[
                "rq_layer.rq_layers.{}.{}".format(layer, parameter)
            ] = torch.ones(1)
    assert require_quantizer_compatible(complete)


def test_checkpoint_loading_rejects_pre_route_a_encoder(tmp_path):
    model = build_tokenizer({
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 16,
        "num_quantizers": 1,
        "codebook_size": 8,
        "quantizer_type": "fsq",
        "fsq_levels": [2, 2, 2],
    })
    checkpoint = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": {}}, checkpoint)

    with pytest.raises(RuntimeError, match="Route-A encoder change"):
        load_tokenizer(checkpoint, torch.device("cpu"))


def test_fsq_acceptance_requires_usage_perplexity_and_all_buckets():
    layer_metrics = [
        {"layer": layer, "active": 24, "perplexity": 12.0}
        for layer in range(3)
    ]
    diagnostics = [
        {"layer": layer, "buckets_all_nonzero": True}
        for layer in range(3)
    ]
    assert _fsq_acceptance_failures(layer_metrics, diagnostics) == []

    layer_metrics[1]["active"] = 15
    diagnostics[2]["buckets_all_nonzero"] = False
    failures = _fsq_acceptance_failures(layer_metrics, diagnostics)
    assert "L1 active=15<16" in failures
    assert "L2 has an empty scalar bucket" in failures
