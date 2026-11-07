"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from SHAPE.evaluation.prototype_atlas import extract_event_attributes
from SHAPE.stage2.data import EventDrivenDataset, collate_event_seqs
from SHAPE.stage2.tokenizer import SUPPORTED_QUANTIZERS, load_tokenizer
from SHAPE.stage3.contracts import json_scalar, save_npz, sha256_file
from SHAPE.stage3.landing_head import LANDING_EVENTS_SCHEMA
from SHAPE.stage3.prepare_codes import validate_aux_contract


SPLIT_FILES = {
    "train": "train_user_ids.npy",
    "val": "val_user_ids.npy",
    "test": "test_user_ids.npy",
}


@torch.inference_mode()
def prepare_landing_events(
    checkpoint, data_dir, city, output, device=torch.device("cpu"),
    split="test", batch_size=32, num_workers=0, max_users=None,
    expected_jump_aux_weight=None, expected_poi_aux_weight=None,
    expected_time_aux_weight=None, require_no_time_aux=False,
    quantizer_type="fsq",
):
    """Use target events for evaluation only; no parameter is trained or selected."""
    city = str(city).lower()
    if city not in ("ca", "nyc", "paris"):
        raise ValueError("city must be ca, nyc, or paris")
    data_dir = Path(data_dir)
    dataset = EventDrivenDataset(
        str(data_dir / "event_driven_data.npz"),
        str(data_dir / "user_meta.json"),
        str(data_dir / SPLIT_FILES[split]),
    )
    if max_users is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(int(max_users), len(dataset))))
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=int(num_workers), collate_fn=collate_event_seqs,
    )
    checkpoint_sha = sha256_file(checkpoint)
    model, config, report = load_tokenizer(
        checkpoint, device, quantizer_type=quantizer_type
    )
    tokenizer_source_city = str(config.get("city", "ca")).lower()
    if tokenizer_source_city not in ("ca", "nyc", "paris"):
        raise RuntimeError("Landing tokenizer has no valid source-city contract")
    if int(config.get("num_quantizers", 0)) != 3 \
            or int(config.get("codebook_size", 0)) != 64:
        raise RuntimeError("Landing evaluation requires a frozen 3x64 tokenizer")
    aux_contract = validate_aux_contract(
        report, expected_jump_aux_weight, expected_poi_aux_weight,
        expected_time_aux_weight, require_no_time_aux,
        subject="Landing tokenizer",
    )
    all_h, all_grid, all_minute, all_weekday, all_users, lengths = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Encode %s landing events" % city.upper()):
        sequence = batch["x"].to(device)
        mask = batch["attention_mask"].to(device)
        embedded, _ = model.surprisal_encoder(sequence)
        embedded = embedded * mask.unsqueeze(-1).to(embedded.dtype)
        z_e = model.transformer_encoder(embedded)
        z_q, _, _, _, _, _ = model.rq_layer(z_e, mask=mask)
        reconstructed = model.decode(z_q, mask=mask).cpu().numpy()
        x = sequence.cpu().numpy()
        valid = mask.cpu().numpy().astype(bool)
        for row, user in enumerate(batch["user_ids"]):
            keep = valid[row]
            h = reconstructed[row, keep].astype(np.float32, copy=False)
            attributes = extract_event_attributes(x[row, keep])
            grid = batch["grid_ids"][row, keep].cpu().numpy().astype(
                np.int64, copy=False
            )
            # The event archive stores hour resolution.  This is sufficient
            # for the preregistered four-period and 24-hour diagnostics.
            minute = (attributes["hour"].astype(np.int16) * 60).astype(np.int16)
            weekday = attributes["weekday"].astype(np.int8, copy=False)
            all_h.append(h)
            all_grid.append(grid)
            all_minute.append(minute)
            all_weekday.append(weekday)
            all_users.append(str(user))
            lengths.append(len(h))
    if not all_h:
        raise RuntimeError("No %s %s events were encoded" % (city, split))
    offsets = np.concatenate((
        np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)
    ))
    metadata = {
        "schema": LANDING_EVENTS_SCHEMA, "source_city": city,
        "evaluation_city": city,
        "tokenizer_source_city": tokenizer_source_city,
        "split": split, "tokenizer_sha256": checkpoint_sha,
        "tokenizer_checkpoint": str(checkpoint),
        "encoder_feature_contract": report.get("encoder_feature_contract"),
        "quantizer_type": str(report["inferred_quantizer_type"]),
        "quantizer_types": list(report["quantizer_types"]),
        "tokenizer_training_contract": report["tokenizer_training_contract"],
        "aux_contract": aux_contract,
        "target_city_training_data_used": False,
        "target_city_evaluation_data_used": bool(city != tokenizer_source_city),
        "sequence_count": len(all_users), "event_count": int(offsets[-1]),
    }
    save_npz(
        output, schema=np.asarray(LANDING_EVENTS_SCHEMA),
        metadata_json=json_scalar(metadata),
        tokenizer_sha256=np.asarray(checkpoint_sha),
        quantizer_type=np.asarray(str(report["inferred_quantizer_type"])),
        quantizer_types=np.asarray(report["quantizer_types"]),
        tokenizer_training_contract=np.asarray(report["tokenizer_training_contract"]),
        source_city=np.asarray(city), split=np.asarray(split),
        evaluation_city=np.asarray(city),
        sequence_offsets=offsets, sequence_user_ids=np.asarray(all_users),
        event_user_ids=np.repeat(np.asarray(all_users), np.asarray(lengths)),
        event_splits=np.asarray([split] * int(offsets[-1])),
        reconstructed_h=np.concatenate(all_h), grid_id=np.concatenate(all_grid),
        minute=np.concatenate(all_minute), weekday=np.concatenate(all_weekday),
    )
    manifest = Path(output).with_suffix(".manifest.json")
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Encode target-city events for evaluation-only landing"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--city", required=True, choices=("ca", "nyc", "paris"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--expected-jump-aux-weight", type=float, default=None)
    parser.add_argument("--expected-poi-aux-weight", type=float, default=None)
    parser.add_argument("--expected-time-aux-weight", type=float, default=None)
    parser.add_argument("--require-no-time-aux", action="store_true")
    parser.add_argument(
        "--quantizer-type", choices=SUPPORTED_QUANTIZERS, default="fsq",
        help="Strict tokenizer family expected in the checkpoint",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 \
            or (args.max_users is not None and args.max_users < 1):
        parser.error("landing-event encoding options are out of range")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    report = prepare_landing_events(
        args.checkpoint, args.data_dir, args.city, args.output, device,
        args.split, args.batch_size, args.num_workers, args.max_users,
        args.expected_jump_aux_weight, args.expected_poi_aux_weight,
        args.expected_time_aux_weight, args.require_no_time_aux,
        args.quantizer_type,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
