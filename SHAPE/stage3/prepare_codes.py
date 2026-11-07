"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
from SHAPE.evaluation.prototype_atlas import categorize_jump_distances, event_revisit_flags, extract_event_attributes
from SHAPE.stage2.data import EventDrivenDataset, collate_event_seqs
from SHAPE.stage2.tokenizer import SUPPORTED_QUANTIZERS, load_tokenizer
from SHAPE.stage3.config import GEN_CONFIG
from SHAPE.stage3.contracts import CODES_SCHEMA, json_scalar, save_npz, sha256_file
from SHAPE.stage3.city_contract import SOURCE_CITIES, canonical_source_city, city_paths, load_raw_calendar
from SHAPE.stage2.prepare_city_event_data import load_city_paths


DEFAULT_RAW_FILES = (
    "dataset/Gowalla-CA/gowalla_train.csv",
    "dataset/Gowalla-CA/gowalla_val.csv",
    "dataset/Gowalla-CA/gowalla_test.csv",
)


SPLIT_FILES = {
    "train": "train_user_ids.npy",
    "val": "val_user_ids.npy",
    "test": "test_user_ids.npy",
}


ANALYSIS_DEPTH_CODES_SCHEMA = "civic_codebook_depth_archive_v1"


class _TaggedDataset(Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["source_split"] = self.split
        return item


def _collate(batch):
    output = collate_event_seqs(batch)
    output["source_splits"] = [item["source_split"] for item in batch]
    return output


def assign_day_splits(user_id, dates, seed=42):
    """Create within-user held-out days while guaranteeing a train prefix set."""
    unique_dates = np.asarray(sorted(set(str(value) for value in dates)), dtype=np.str_)
    count = len(unique_dates)
    if count == 0:
        return {}
    stable = sum((index + 1) * ord(char) for index, char in enumerate(str(user_id)))
    rng = np.random.default_rng(int(seed) + stable)
    order = unique_dates[rng.permutation(count)]
    n_test = max(1, int(round(0.10 * count))) if count >= 3 else 0
    n_val = max(1, int(round(0.10 * count))) if count >= 5 else 0
    if n_test + n_val >= count:
        n_val = max(0, count - n_test - 1)
    n_train = count - n_val - n_test
    mapping = {str(value): "train" for value in order[:n_train]}
    for value in order[n_train:n_train + n_val]:
        mapping[str(value)] = "val"
    for value in order[n_train + n_val:]:
        mapping[str(value)] = "test"
    return mapping


def _build_loader(data_dir, batch_size, num_workers, max_users=None):
    datasets = []
    for split, filename in SPLIT_FILES.items():
        dataset = EventDrivenDataset(
            str(Path(data_dir) / "event_driven_data.npz"),
            str(Path(data_dir) / "user_meta.json"),
            str(Path(data_dir) / filename),
        )
        dataset = _TaggedDataset(dataset, split)
        datasets.append(dataset)
    combined = ConcatDataset(datasets)
    if max_users is not None:
        combined = Subset(combined, range(min(int(max_users), len(combined))))
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )


def validate_aux_contract(
    report,
    expected_jump_aux_weight=None,
    expected_poi_aux_weight=None,
    expected_time_aux_weight=None,
    require_no_time_aux=False,
    subject="Tokenizer",
):
    """Validate and normalize the frozen tokenizer's semantic-head contract."""
    aux_contract = {
        name: float(report.get("%s_aux_weight" % name, 0.0))
        for name in ("jump", "poi", "time")
    }
    expected = {"jump": expected_jump_aux_weight, "poi": expected_poi_aux_weight,
                "time": expected_time_aux_weight}
    for name, value in expected.items():
        if value is not None and not math.isclose(
            aux_contract[name], float(value), rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError(
                "%s %s auxiliary weight mismatch: actual=%g expected=%g"
                % (subject, name, aux_contract[name], float(value))
            )
    if require_no_time_aux and aux_contract["time"] != 0.0:
        raise RuntimeError(
            "%s violates the Phase-2 v3 no-timeaux contract" % subject
        )
    return aux_contract


@torch.inference_mode()
def prepare_codes(
    checkpoint,
    data_dir,
    raw_checkins,
    output,
    device,
    batch_size=32,
    num_workers=0,
    seed=42,
    max_users=None,
    expected_jump_aux_weight=None,
    expected_poi_aux_weight=None,
    expected_time_aux_weight=None,
    require_no_time_aux=False,
    quantizer_type="fsq",
    source_city="ca",
    grid_csv=None,
    evaluation_only_cross_city=False,
    analysis_only_variable_depth=False,
    pad_analysis_codes_to_three=False,
):
    source_city = canonical_source_city(source_city)
    checkpoint_sha = sha256_file(checkpoint)
    model, model_config, checkpoint_report = load_tokenizer(
        checkpoint, device, quantizer_type=quantizer_type
    )
    checkpoint_city = str(model_config.get("city", source_city)).lower()
    if checkpoint_city != source_city and not evaluation_only_cross_city:
        raise RuntimeError(
            "Tokenizer source city mismatch: checkpoint=%s requested=%s"
            % (checkpoint_city, source_city)
        )
    if evaluation_only_cross_city and checkpoint_city == source_city:
        raise RuntimeError(
            "--evaluation-only-cross-city requires different tokenizer and "
            "evaluation cities"
        )
    native_num_layers = int(model_config.get("num_quantizers", 0))
    native_codebook_size = int(model_config.get("codebook_size", 0))
    if not analysis_only_variable_depth and native_num_layers != 3:
        raise RuntimeError("Phase 2 requires exactly three residual code layers")
    if not analysis_only_variable_depth and native_codebook_size != 64:
        raise RuntimeError("Phase 2 requires a frozen 64-code tokenizer")
    if analysis_only_variable_depth and native_num_layers < 1:
        raise RuntimeError("Analysis checkpoints need at least one code layer")
    if pad_analysis_codes_to_three and not analysis_only_variable_depth:
        raise RuntimeError(
            "--pad-analysis-codes-to-three requires "
            "--analysis-only-variable-depth"
        )
    if pad_analysis_codes_to_three and native_num_layers > 3:
        raise RuntimeError("Only one- or two-layer analysis arms can be padded")
    if float(checkpoint_report.get("jump_aux_weight", 0.0)) <= 0.0:
        raise RuntimeError(
            "Phase 2 requires the selected jumpaux tokenizer checkpoint"
        )
    aux_contract = validate_aux_contract(
        checkpoint_report,
        expected_jump_aux_weight=expected_jump_aux_weight,
        expected_poi_aux_weight=expected_poi_aux_weight,
        expected_time_aux_weight=expected_time_aux_weight,
        require_no_time_aux=require_no_time_aux,
    )
    raw_calendar = load_raw_calendar(
        source_city,
        raw_checkins=raw_checkins,
        grid_csv=grid_csv or city_paths(source_city)["grid_csv"],
    )
    loader = _build_loader(
        data_dir, batch_size=batch_size, num_workers=num_workers,
        max_users=max_users,
    )

    day_records = []
    for batch in tqdm(loader, desc="Prepare %s daily codes" % source_city.upper()):
        sequence = batch["x"].to(device)
        mask = batch["attention_mask"].to(device)
        embedded, _ = model.surprisal_encoder(sequence)
        embedded = embedded * mask.unsqueeze(-1).to(embedded.dtype)
        z_e = model.transformer_encoder(embedded)
        z_q, _, _, codes, _, _ = model.rq_layer(z_e, mask=mask)
        reconstructed_h = model.decode(z_q, mask=mask)

        x_np = sequence.cpu().numpy()
        mask_np = mask.cpu().numpy().astype(bool)
        codes_np = codes.cpu().numpy()
        h_np = reconstructed_h.cpu().numpy()
        for row, user_id in enumerate(batch["user_ids"]):
            user_id = str(user_id)
            keep = mask_np[row]
            valid_x = x_np[row, keep]
            code_dtype = (
                np.int16 if native_codebook_size <= np.iinfo(np.int16).max
                else np.int32
            )
            valid_codes = codes_np[row, keep].astype(code_dtype, copy=False)
            if pad_analysis_codes_to_three and valid_codes.shape[1] < 3:
                valid_codes = np.pad(
                    valid_codes,
                    ((0, 0), (0, 3 - valid_codes.shape[1])),
                    mode="constant",
                    constant_values=0,
                )
            valid_h = h_np[row, keep].astype(np.float32, copy=False)
            valid_grid_ids = batch["grid_ids"][row, keep].cpu().numpy().astype(
                np.int64, copy=False
            )
            calendar = raw_calendar.get(user_id)
            if calendar is None or len(calendar) != len(valid_x):
                raise RuntimeError(
                    "Raw/calendar alignment failed for user %s: raw=%s archive=%d"
                    % (user_id, None if calendar is None else len(calendar), len(valid_x))
                )
            dates = np.asarray([value.strftime("%Y-%m-%d") for value in calendar])
            minutes = np.asarray(
                [value.hour * 60 + value.minute for value in calendar], dtype=np.int16
            )
            weekdays = np.asarray([value.weekday() for value in calendar], dtype=np.int8)
            archive_hours = np.rint(valid_x[:, 18]).astype(np.int16)
            archive_weekdays = np.rint(valid_x[:, 17] * 7.0).astype(np.int16)
            if not np.array_equal(archive_hours, minutes // 60) or not np.array_equal(
                archive_weekdays, weekdays
            ):
                raise RuntimeError(
                    "Raw/calendar order disagrees with the prepared archive for user %s"
                    % user_id
                )
            attrs = extract_event_attributes(valid_x)
            attrs["grid_id"] = valid_grid_ids
            attrs["jump_bin"] = categorize_jump_distances(attrs["jump_distance"])
            attrs["revisit"] = event_revisit_flags(attrs["grid_id"])
            day_splits = assign_day_splits(user_id, dates, seed=seed)
            for date in sorted(set(dates.tolist())):
                indices = np.flatnonzero(dates == date)
                day_records.append({
                    "user_id": user_id,
                    "date": date,
                    "split": day_splits[date],
                    "source_split": str(batch["source_splits"][row]),
                    "codes": valid_codes[indices],
                    "minute": minutes[indices],
                    "weekday": weekdays[indices],
                    "poi_category": attrs["poi_category"][indices],
                    "jump_distance": attrs["jump_distance"][indices],
                    "jump_bin": attrs["jump_bin"][indices],
                    "grid_id": attrs["grid_id"][indices],
                    "revisit": attrs["revisit"][indices],
                    "reconstructed_h": valid_h[indices],
                })

    if not day_records:
        raise RuntimeError("No active %s days were encoded" % source_city.upper())
    day_records.sort(key=lambda item: (item["user_id"], item["date"]))
    event_keys = (
        "codes", "minute", "weekday", "poi_category", "jump_distance",
        "jump_bin", "grid_id", "revisit", "reconstructed_h",
    )
    offsets = [0]
    for day in day_records:
        offsets.append(offsets[-1] + len(day["codes"]))
    output_schema = (
        ANALYSIS_DEPTH_CODES_SCHEMA
        if analysis_only_variable_depth else CODES_SCHEMA
    )
    output_num_layers = (
        3 if pad_analysis_codes_to_three else native_num_layers
    )
    metadata = {
        "schema": output_schema,
        "source_city": source_city,
        "tokenizer_source_city": checkpoint_city,
        "evaluation_only_cross_city": bool(evaluation_only_cross_city),
        "zero_shot_contract": GEN_CONFIG["zero_shot_contract"],
        "daily_split_contract": GEN_CONFIG["daily_split_contract"],
        "tokenizer_sha256": checkpoint_sha,
        "tokenizer_checkpoint": str(checkpoint),
        "encoder_feature_contract": checkpoint_report["encoder_feature_contract"],
        "jump_aux_weight": aux_contract["jump"],
        "poi_aux_weight": aux_contract["poi"],
        "time_aux_weight": aux_contract["time"],
        "quantizer_type": str(checkpoint_report["inferred_quantizer_type"]),
        "quantizer_types": list(checkpoint_report["quantizer_types"]),
        "aux_routing": str(checkpoint_report.get("aux_routing", "aggregate")),
        "tokenizer_training_contract": checkpoint_report["tokenizer_training_contract"],
        "num_code_layers": output_num_layers,
        "native_num_code_layers": native_num_layers,
        "analysis_padding": (
            "right_zero_to_three" if pad_analysis_codes_to_three else "none"
        ),
        "analysis_only_variable_depth": bool(analysis_only_variable_depth),
        "codebook_size": native_codebook_size,
        "seed": int(seed),
        "event_count": int(offsets[-1]),
        "day_count": len(day_records),
        "user_count": len(set(item["user_id"] for item in day_records)),
    }
    arrays = {
        "schema": np.asarray(output_schema),
        "metadata_json": json_scalar(metadata),
        "tokenizer_sha256": np.asarray(checkpoint_sha),
        "source_city": np.asarray(source_city),
        "tokenizer_source_city": np.asarray(checkpoint_city),
        "evaluation_only_cross_city": np.asarray(
            bool(evaluation_only_cross_city)
        ),
        "quantizer_type": np.asarray(
            str(checkpoint_report["inferred_quantizer_type"])
        ),
        "quantizer_types": np.asarray(checkpoint_report["quantizer_types"]),
        "tokenizer_training_contract": np.asarray(checkpoint_report["tokenizer_training_contract"]),
        "codebook_size": np.asarray(native_codebook_size, dtype=np.int32),
        "num_code_layers": np.asarray(output_num_layers, dtype=np.int8),
        "native_num_code_layers": np.asarray(native_num_layers, dtype=np.int8),
        "analysis_padding": np.asarray(metadata["analysis_padding"]),
        "analysis_only_variable_depth": np.asarray(
            bool(analysis_only_variable_depth)
        ),
        "day_offsets": np.asarray(offsets, dtype=np.int64),
        "day_user_ids": np.asarray([item["user_id"] for item in day_records]),
        "day_dates": np.asarray([item["date"] for item in day_records]),
        "day_splits": np.asarray([item["split"] for item in day_records]),
        "source_user_splits": np.asarray(
            [item["source_split"] for item in day_records]
        ),
    }
    for key in event_keys:
        arrays[key] = np.concatenate([item[key] for item in day_records], axis=0)
    event_lengths = np.diff(arrays["day_offsets"])
    arrays["event_user_ids"] = np.repeat(arrays["day_user_ids"], event_lengths)
    arrays["event_dates"] = np.repeat(arrays["day_dates"], event_lengths)
    arrays["event_splits"] = np.repeat(arrays["day_splits"], event_lengths)
    save_npz(output, **arrays)
    manifest_path = str(Path(output).with_suffix(".manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-encode a source city into date-aware tokenizer triples"
    )
    parser.add_argument(
        "--source-city", choices=SOURCE_CITIES, default="ca",
        help=(
            "City whose event archive is encoded; normally the tokenizer "
            "source city (default: ca)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="SHAPE/stage2/checkpoints/rqvae_transformer_jumpaux_poiaux.pth",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--city-config", help="City-path JSON used by prepare-events")
    parser.add_argument(
        "--grid-csv", default=None,
        help="Optional registered grid-table override for this city",
    )
    parser.add_argument(
        "--raw-checkins", nargs="+", default=None,
        help="Optional CA raw CSV override; NYC/Paris use registered raw paths",
    )
    parser.add_argument(
        "--output", default=None
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--expected-jump-aux-weight", type=float, default=None)
    parser.add_argument("--expected-poi-aux-weight", type=float, default=None)
    parser.add_argument("--expected-time-aux-weight", type=float, default=None)
    parser.add_argument("--require-no-time-aux", action="store_true")
    parser.add_argument(
        "--quantizer-type", choices=SUPPORTED_QUANTIZERS, default="fsq",
        help="Strict tokenizer family expected in the checkpoint",
    )
    parser.add_argument(
        "--evaluation-only-cross-city", action="store_true",
        help=(
            "Explicitly encode an evaluation city with a tokenizer trained in "
            "another city and record an evaluation-only provenance receipt."
        ),
    )
    parser.add_argument(
        "--analysis-only-variable-depth", action="store_true",
        help=(
            "Write the isolated depth-ablation schema instead of a Phase-2 "
            "consumer artifact; permits non-3-layer and non-64-code models"
        ),
    )
    parser.add_argument(
        "--pad-analysis-codes-to-three", action="store_true",
        help=(
            "Right-pad one/two-layer analysis codes with constant zeros for "
            "explicit fixed-shape downstream controls"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.city_config:
        load_city_paths(args.city_config)
    paths = city_paths(args.source_city)
    data_dir = args.data_dir or paths["data_dir"]
    output = args.output or (
        "SHAPE/stage3/artifacts/%s_daily_codes.npz" % args.source_city
    )
    raw_checkins = args.raw_checkins
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    metadata = prepare_codes(
        checkpoint=args.checkpoint,
        data_dir=data_dir,
        raw_checkins=raw_checkins,
        output=output,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_users=args.max_users,
        expected_jump_aux_weight=args.expected_jump_aux_weight,
        expected_poi_aux_weight=args.expected_poi_aux_weight,
        expected_time_aux_weight=args.expected_time_aux_weight,
        require_no_time_aux=args.require_no_time_aux,
        quantizer_type=args.quantizer_type,
        source_city=args.source_city,
        grid_csv=args.grid_csv or paths["grid_csv"],
        evaluation_only_cross_city=args.evaluation_only_cross_city,
        analysis_only_variable_depth=args.analysis_only_variable_depth,
        pad_analysis_codes_to_three=args.pad_analysis_codes_to_three,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
