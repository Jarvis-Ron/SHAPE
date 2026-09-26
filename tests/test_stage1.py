import csv
import json

import numpy as np
import pytest
import torch

from SHAPE.stage1.contrastive_trainer import (
    MultiCityContrastiveDataset,
    info_nce_loss_uniform,
    loss_early_stopping_step,
    metric_checkpoint_is_better,
    poi_profile_target_distribution,
)
from SHAPE.stage1.alignment_metrics import (
    POI_JS_THRESH,
    evaluate_live_retrieval,
    poi_sim_matrix,
)
from SHAPE.stage1.config import MODULE1_CONFIG
from SHAPE.stage1.extract_embeddings import _build_model, run_extraction
from SHAPE.stage1.grid_text_encoder import load_grid_features
from SHAPE.stage1.train_grid_embeddings import run_training


def _write_city_csv(path, n_grids=4):
    fieldnames = (
        ["grid_id"]
        + ["poi_prop_%d" % index for index in range(13)]
        + [
            "poi_density_level",
            "total_stations",
            "metro_stations",
            "bus_stations",
            "total_frequency",
            "metro_frequency",
            "bus_frequency",
            "grid_lat_max",
            "grid_lat_min",
            "grid_lng_max",
            "grid_lng_min",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for grid_id in range(n_grids):
            row = {name: 0 for name in fieldnames}
            row.update({
                "grid_id": grid_id,
                "poi_prop_%d" % (grid_id % 13): 1.0,
                "poi_density_level": grid_id / n_grids,
                "grid_lat_max": 1.0,
                "grid_lat_min": 0.0,
                "grid_lng_max": 1.0,
                "grid_lng_min": 0.0,
            })
            writer.writerow(row)


def test_m1_checkpoint_extraction_writes_stage_one_latent(tmp_path):
    csv_path = tmp_path / "ca.csv"
    checkpoint = tmp_path / "m1.pth"
    output_dir = tmp_path / "latents"
    _write_city_csv(csv_path)
    torch.save({"model_state_dict": _build_model().state_dict()}, checkpoint)

    result = run_extraction(
        output_dir=output_dir,
        device="cpu",
        csv_paths={"ca": str(csv_path)},
        checkpoint_path=str(checkpoint),
    )

    with np.load(output_dir / "ca_latent.npz", allow_pickle=False) as latent:
        assert latent["embeddings"].shape == (4, 64)
        assert latent["grid_ids"].dtype == np.int64
        assert np.array_equal(latent["grid_ids"], np.arange(4))
    assert result["ca"]["shape"] == [4, 64]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_extraction(
            output_dir=output_dir,
            device="cpu",
            csv_paths={"ca": str(csv_path)},
            checkpoint_path=str(checkpoint),
        )




def test_m1_pairs_preserve_real_grid_ids():
    features = np.zeros((2, 20), dtype=np.float32)
    features[:, 0] = 1.0
    dataset = MultiCityContrastiveDataset({
        "ca": (features, np.array([4, 9])),
        "nyc": (features.copy(), np.array([3, 8])),
    }, topk_feat_sim=1, pos_rule="dominant_cat")

    item = dataset[0]

    assert item["anchor_id"].item() in (4, 9)
    assert item["pos_id"].item() in (3, 8)
    assert item["anchor_city"] != item["pos_city"]


def test_m1_soft_positive_uses_complete_poi_profile_not_dominant_only():
    anchor = np.zeros((1, 20), dtype=np.float32)
    anchor[0, :2] = [0.6, 0.4]
    targets = np.zeros((2, 20), dtype=np.float32)
    targets[0, :2] = [1.0, 0.0]
    targets[1, :2] = [0.6, 0.4]
    # A non-POI feature must not override the 13-D semantic ranking.
    targets[0, 14:] = 100.0
    dataset = MultiCityContrastiveDataset({
        "ca": (anchor, np.array([5])),
        "nyc": (targets, np.array([10, 11])),
    }, topk_feat_sim=1, pos_rule="poi_profile")

    item = dataset[0]

    assert item["pos_id"].item() == 11
    assert item["pos_weight"] == pytest.approx(1.0)
    assert dataset.hard_neg_k == 0


def test_m1_poi_loss_uses_continuous_multi_positive_targets():
    anchor_poi = torch.zeros(3, 13)
    pos_poi = torch.zeros(3, 13)
    anchor_poi[0, :2] = torch.tensor([0.6, 0.4])
    anchor_poi[1, :2] = torch.tensor([0.55, 0.45])
    anchor_poi[2, 2] = 1.0
    pos_poi.copy_(anchor_poi)
    targets = poi_profile_target_distribution(
        anchor_poi, pos_poi, target_temperature=0.10
    )

    assert torch.allclose(targets.sum(dim=1), torch.ones(3))
    assert targets[0, 1] > targets[0, 2]
    assert targets[0, 1] > 0.1
    cross_city_only = poi_profile_target_distribution(
        anchor_poi, pos_poi, target_temperature=0.10,
        allowed_mask=torch.tensor([
            [True, False, True],
            [True, True, True],
            [True, True, True],
        ]),
    )
    assert cross_city_only[0, 1] == 0.0

    anchor_emb = torch.randn(3, 8, requires_grad=True)
    pos_emb = torch.randn(3, 8, requires_grad=True)
    loss, _, hard_neg_dim = info_nce_loss_uniform(
        anchor_emb, pos_emb, uniformity_weight=0.0,
        anchor_poi=anchor_poi, pos_poi=pos_poi,
        poi_target_temperature=0.10,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert hard_neg_dim == 0
    assert anchor_emb.grad is not None
    assert pos_emb.grad is not None


def test_poi_retrieval_does_not_treat_empty_profiles_as_matches():
    populated = np.zeros((1, 20), dtype=np.float32)
    populated[0, 0] = 1.0
    empty = np.zeros((1, 20), dtype=np.float32)

    assert not poi_sim_matrix(populated, empty).any()
    assert not poi_sim_matrix(empty, populated).any()


def test_poi_retrieval_threshold_rejects_moderately_different_profiles():
    source = np.zeros((1, 20), dtype=np.float32)
    source[0, 0] = 1.0
    targets = np.zeros((2, 20), dtype=np.float32)
    targets[0, :2] = [0.9, 0.1]
    targets[1, :2] = [0.5, 0.5]

    assert POI_JS_THRESH == pytest.approx(0.15)
    assert MODULE1_CONFIG["poi_eval_js_threshold"] == pytest.approx(0.15)
    assert poi_sim_matrix(source, targets).tolist() == [[True, False]]
    # The former 0.45-nat threshold classified both rows as positives.
    assert poi_sim_matrix(source, targets, threshold=0.45).all()


def test_retrieval_checkpoint_plateau_uses_loss_tiebreak_but_not_patience():
    assert metric_checkpoint_is_better(
        1.0, -float("inf"), 0.8, float("inf"),
        selection_min_delta=1e-4, loss_min_delta=1e-4,
    )
    assert metric_checkpoint_is_better(
        1.0, 1.0, 0.7, 0.8,
        selection_min_delta=1e-4, loss_min_delta=1e-4,
    )
    assert not metric_checkpoint_is_better(
        0.99, 1.0, 0.6, 0.7,
        selection_min_delta=1e-4, loss_min_delta=1e-4,
    )

    # Reproduce the former failure mode: retrieval is already 1.0 at epoch 5,
    # but validation loss continues improving.  The selected checkpoint must
    # continue advancing and the metric plateau must not consume patience.
    best_selection = -float("inf")
    best_selection_loss = float("inf")
    best_stop_loss = float("inf")
    wait = 0
    saved_epochs = []
    for epoch in range(1, 41):
        loss = 1.0 / (epoch + 1)
        if epoch % 5 == 0 and metric_checkpoint_is_better(
                1.0, best_selection, loss, best_selection_loss,
                selection_min_delta=1e-4, loss_min_delta=1e-4):
            best_selection = max(best_selection, 1.0)
            best_selection_loss = loss
            saved_epochs.append(epoch)
        best_stop_loss, wait, should_stop, _ = loss_early_stopping_step(
            loss, best_stop_loss, wait, patience=30, min_delta=1e-4
        )
        assert not should_stop
    assert saved_epochs == list(range(5, 41, 5))

    best_loss, wait = float("inf"), 0
    for loss in (0.8, 0.7):
        best_loss, wait, should_stop, improved = loss_early_stopping_step(
            loss, best_loss, wait, patience=2, min_delta=1e-4
        )
        assert improved
        assert not should_stop
    best_loss, wait, should_stop, _ = loss_early_stopping_step(
        0.7, best_loss, wait, patience=2, min_delta=1e-4
    )
    assert not should_stop
    best_loss, wait, should_stop, _ = loss_early_stopping_step(
        0.71, best_loss, wait, patience=2, min_delta=1e-4
    )
    assert should_stop


def test_m1_training_stages_checkpoint_and_records_joint_stats(tmp_path):
    csv_paths = {}
    for city in ("ca", "nyc", "paris"):
        path = tmp_path / (city + ".csv")
        _write_city_csv(path)
        csv_paths[city] = str(path)
    checkpoint = tmp_path / "checkpoints" / "m1.pth"

    result = run_training(
        checkpoint_path=checkpoint,
        device="cpu",
        csv_paths=csv_paths,
        epochs=5,
        batch_size=16,
    )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_features = []
    for path in csv_paths.values():
        expected_features.append(load_grid_features(path)[0])
    expected_mean = torch.from_numpy(np.concatenate(expected_features).mean(0))
    assert torch.allclose(state["model_state_dict"]["feat_mean"], expected_mean)
    assert state["config"]["training_contract"] == "three_city_sparse_tiled_v2"
    assert state["config"]["grid_contract"] == "grid_spec_v4_int64_sparse_ids"
    assert state["config"]["retrieval_eval_every"] == 5
    assert state["config"]["positive_pair_contract"] == \
        "poi_profile_multi_positive_v2"
    assert state["config"]["cross_semantic_topk"] == 20
    assert state["config"]["explicit_hard_negatives"] == 0
    assert state["config"]["poi_multi_positive"] is True
    assert state["config"]["mixup_alpha"] == 0.0
    assert state["config"]["checkpoint_selection_metric"] == "hard_r@1"
    assert state["config"]["checkpoint_selection_min_delta"] == 1e-4
    assert state["config"]["early_stopping_metric"] == "val_loss"
    assert state["config"]["early_stopping_min_delta"] == 1e-4
    assert state["config"]["poi_eval_js_threshold"] == pytest.approx(0.15)
    assert state["selection_metric"] == "hard_r@1"
    assert state["selection_value"] == result["best_hard_r@1"]
    assert np.isfinite(result["best_val_loss"])
    history_path = checkpoint.parent / "m1_training_history.json"
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert history_payload["history"]["retrieval_eval"][0]["epoch"] == 5
    assert set(history_payload["history"]["retrieval_eval"][0]) >= {
        "hard_r@1", "hard_r@5", "hard_mrr",
        "poi_r@1", "poi_r@5", "poi_mrr",
        "transit_r@1", "transit_r@5", "transit_mrr",
        "soft_r@1", "soft_r@5", "soft_mrr",
    }

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_training(
            checkpoint_path=checkpoint,
            device="cpu",
            csv_paths=csv_paths,
            epochs=1,
            batch_size=16,
        )


def test_live_m1_retrieval_is_bidirectional_and_restores_training_mode():
    class IdentityCityEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def encode_city(self, features, grid_ids):
            del grid_ids
            return torch.nn.functional.normalize(features[:, :2], dim=-1)

    features = np.zeros((2, 20), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    features[:, 14] = 1.0
    city_data = {
        city: (features.copy(), np.array([0, 1], dtype=np.int64))
        for city in ("ca", "nyc", "paris")
    }
    model = IdentityCityEncoder().train()

    metrics = evaluate_live_retrieval(model, city_data, device=torch.device("cpu"))

    assert model.training
    assert metrics == {
        "hard_r@1": 1.0,
        "hard_r@5": 1.0,
        "hard_r@10": 1.0,
        "hard_mrr": 1.0,
        "poi_r@1": 1.0,
        "poi_r@5": 1.0,
        "poi_r@10": 1.0,
        "poi_mrr": 1.0,
        "transit_r@1": 1.0,
        "transit_r@5": 1.0,
        "transit_r@10": 1.0,
        "transit_mrr": 1.0,
        "soft_r@1": 1.0,
        "soft_r@5": 1.0,
        "soft_r@10": 1.0,
        "soft_mrr": 1.0,
        "poi_positive_density": 0.5,
        "poi_query_coverage": 1.0,
    }
