"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

import os, json
import functools
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from .config import MODULE2_CONFIG as C
from .config import resolve_data_dir
from SHAPE.npz_utils import ProcessLocalNpzArchive


H_START, H_END = 19, 83  # H embedding at 19:83 (64-dim)


class AttentionBudgetBatchSampler:
    """Group similarly sized users under a padded-attention memory budget.

    A Transformer batch pads every sequence to its longest member, so its
    dominant attention allocation grows with ``B * T_max ** 2``.  A fixed
    user-count batch is therefore unsafe when refreshed city data contain a
    long-tail user.  This sampler preserves every complete sequence while
    treating ``max_batch_size`` as a user-count cap and
    ``max_attention_cells`` as a cap on that quadratic padded footprint.

    A sequence whose own ``T ** 2`` exceeds the budget is emitted alone.  It
    cannot be split without changing the complete-user training contract, but
    it no longer inflates every other sequence in the batch.
    """

    def __init__(self, sequence_lengths, max_batch_size,
                 max_attention_cells, shuffle=False, seed=42):
        self.sequence_lengths = [int(length) for length in sequence_lengths]
        self.max_batch_size = int(max_batch_size)
        self.max_attention_cells = int(max_attention_cells)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.max_attention_cells < 1:
            raise ValueError("max_attention_cells must be positive")
        if any(length < 1 for length in self.sequence_lengths):
            raise ValueError("sequence lengths must be positive")
        self._batches = self._build_batches()
        self.oversized_singletons = sum(
            1 for batch in self._batches
            if len(batch) == 1
            and self.sequence_lengths[batch[0]] ** 2
            > self.max_attention_cells
        )

    @staticmethod
    def padded_attention_cells(batch_lengths):
        """Return ``B * T_max ** 2`` for one padded batch."""
        if not batch_lengths:
            return 0
        return len(batch_lengths) * max(batch_lengths) ** 2

    def _build_batches(self):
        ordered = sorted(
            range(len(self.sequence_lengths)),
            key=lambda index: (self.sequence_lengths[index], index),
        )
        batches = []
        current = []
        current_max = 0
        for index in ordered:
            length = self.sequence_lengths[index]
            next_max = max(current_max, length)
            next_size = len(current) + 1
            exceeds_count = next_size > self.max_batch_size
            exceeds_attention = (
                next_size * next_max ** 2 > self.max_attention_cells
            )
            if current and (exceeds_count or exceeds_attention):
                batches.append(current)
                current = []
                current_max = 0
            current.append(index)
            current_max = max(current_max, length)
        if current:
            batches.append(current)
        return batches

    def __iter__(self):
        order = list(range(len(self._batches)))
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        if self.shuffle:
            rng.shuffle(order)
        for position in order:
            batch = list(self._batches[position])
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self._batches)


def _sequence_lengths(dataset, max_seq_len=None):
    """Read sequence lengths without opening the event arrays."""
    if isinstance(dataset, Subset):
        base_lengths = _sequence_lengths(dataset.dataset, max_seq_len)
        return [base_lengths[int(index)] for index in dataset.indices]
    if not hasattr(dataset, "meta"):
        raise TypeError(
            "attention-budget batching requires a metadata-backed dataset"
        )
    lengths = [int(item["length"]) for item in dataset.meta]
    if max_seq_len is not None:
        lengths = [min(length, int(max_seq_len)) for length in lengths]
    return lengths


class EventDrivenDataset(Dataset):
    """
    Load event-driven sequence data.

    Data format:
        - npz: one array per user [L, 136]
        - meta: JSON list, containing user_id, length, seq_idx, split
        - split_ids: list of user indices for the corresponding split

    __getitem__ returns: {
        "seq": np.array [L, 136],
        "user_id": str,
        "length": int,
    }
    """
    def __init__(self, npz_path, meta_path, split_ids_path):
        self._npz_archive = ProcessLocalNpzArchive(
            npz_path, mmap_mode="r"
        )
        grid_ids_path = os.path.join(
            os.path.dirname(os.path.abspath(npz_path)), "grid_ids_int64.npz"
        )
        self._grid_id_archive = (
            ProcessLocalNpzArchive(grid_ids_path, mmap_mode="r")
            if os.path.isfile(grid_ids_path) else None
        )
        with open(meta_path, "r", encoding="utf-8") as f:
            all_meta = json.load(f)
        self.split_ids = np.load(split_ids_path)

        # Filter meta by split_ids
        id_set = set(int(x) for x in self.split_ids)
        self.meta = [m for m in all_meta if m["seq_idx"] in id_set]
        self.meta.sort(key=lambda m: m["seq_idx"])

        # Build index mapping
        self.user_indices = [m["seq_idx"] for m in self.meta]

        # Stats
        self.total_seqs = len(self.meta)
        self.total_events = sum(m["length"] for m in self.meta)
        print(f"  [EventDrivenDataset] {self.total_seqs} users, "
            f"{self.total_events} events, dim={self._get_dim()}")


    def _get_dim(self):
        key = f"arr_{self.user_indices[0]}"
        return self.npz[key].shape[-1]

    def __len__(self):
        return self.total_seqs

    @property
    def npz(self):
        """Return the archive handle owned by the current process."""
        return self._npz_archive.get()

    def __getitem__(self, idx):
        meta = self.meta[idx]
        key = f"arr_{meta['seq_idx']}"
        seq = self.npz[key].astype(np.float32)
        grid_ids = (
            self._grid_id_archive.get()[key].astype(np.int64)
            if self._grid_id_archive is not None else
            np.rint(seq[:, 123]).astype(np.int64)
        )
        return {
            "seq": seq,
            "grid_ids": grid_ids,
            "user_id": meta["user_id"],
            "length": meta["length"],
        }


def collate_event_seqs(batch, max_seq_len=None):
    """
    Pad variable-length sequences in a batch to the same length.
    Supports max_seq_len truncation to prevent OOM on long sequences.

    Args:
        batch: list of dict, each containing seq [L, D], user_id, length
        max_seq_len: optional, truncate to this length (tail truncation)
    Returns:
        x: [B, max_len, D]
        attention_mask: [B, max_len], bool, True = valid positions
        target_H: [B, max_len, 64]
        lengths: list of int
        user_ids: list of str
    """
    max_len = max(item["length"] for item in batch)
    if max_seq_len is not None:
        max_len = min(max_len, max_seq_len)
    B = len(batch)
    D = batch[0]["seq"].shape[-1]

    padded_seqs = torch.zeros(B, max_len, D, dtype=torch.float32)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    grid_ids = torch.full((B, max_len), -1, dtype=torch.int64)
    lengths = []
    user_ids = []

    for i, item in enumerate(batch):
        L = item["length"]
        if max_seq_len is not None:
            L = min(L, max_seq_len)
        seq_t = torch.from_numpy(item["seq"])
        if max_seq_len is not None and item["length"] > max_seq_len:
            seq_t = seq_t[:max_seq_len]
        padded_seqs[i, :L] = seq_t
        grid_ids[i, :L] = torch.from_numpy(item["grid_ids"][:L])
        attention_mask[i, :L] = True
        lengths.append(L)
        user_ids.append(item["user_id"])

    # Target: H embedding (19:83)
    target_H = padded_seqs[..., H_START:H_END].clone()

    return {
        "x": padded_seqs,                    # [B, max_len, D]
        "attention_mask": attention_mask,     # [B, max_len]
        "target_H": target_H,                 # [B, max_len, 64]
        "lengths": lengths,                   # list of int
        "user_ids": user_ids,                 # list of str
        "grid_ids": grid_ids,                 # raw nominal int64 IDs
    }


def get_dataloaders(config=None, small_sample=False, small_ratio=0.1,
                    batch_size=None, city=None, max_seq_len=None, seed=42,
                    max_attention_cells=None):
    """
    Create train/val/test DataLoaders.

    Args:
        config:       config dict (default MODULE2_CONFIG)
        small_sample: True -> take only first small_ratio of users
        small_ratio:  small sample ratio (default 0.1)
        batch_size:   override config batch_size
        city:         city name nyc/paris/ca, default from config[city]
        max_seq_len:  optional, truncate long sequences to this length
        max_attention_cells: optional ``B * T_max ** 2`` memory budget;
            preserves complete sequences and treats batch_size as a cap
    Returns:
        train_loader, val_loader, test_loader
    """
    if config is None:
        config = C
    if batch_size is None:
        batch_size = config["batch_size"]
    if max_attention_cells is None:
        max_attention_cells = config.get("max_attention_cells")

    # Resolve data directory by city
    if city is None:
        city = config.get("city", "nyc")
    data_dir = resolve_data_dir(city, config)
    npz_path = os.path.join(data_dir, config.get("event_npz_file", "event_driven_data.npz"))
    meta_path = os.path.join(data_dir, config.get("event_meta_file", "user_meta.json"))

    train_ds = EventDrivenDataset(
        npz_path, meta_path,
        os.path.join(data_dir, config.get("train_ids_file", "train_user_ids.npy")))
    val_ds = EventDrivenDataset(
        npz_path, meta_path,
        os.path.join(data_dir, config.get("val_ids_file", "val_user_ids.npy")))
    test_ds = EventDrivenDataset(
        npz_path, meta_path,
        os.path.join(data_dir, config.get("test_ids_file", "test_user_ids.npy")))

    if small_sample:
        n_train = max(1, int(len(train_ds) * small_ratio))
        n_val   = max(1, int(len(val_ds) * small_ratio))
        n_test  = max(1, int(len(test_ds) * small_ratio))
        train_ds = Subset(train_ds, range(n_train))
        val_ds   = Subset(val_ds, range(n_val))
        print(f"  [Data] Small sample mode: train={n_train}, val={n_val}, test={n_test}")
        print(f"  [Data] Small sample mode: train={n_train}, val={n_val}, test={n_test}")
        print(f"  [Data] Full mode: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        print(f"  [Data] Full mode: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    collate_fn = functools.partial(collate_event_seqs, max_seq_len=max_seq_len)

    if max_attention_cells is None:
        train_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            generator=train_generator,
            num_workers=0, pin_memory=True,
            collate_fn=collate_fn)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
            collate_fn=collate_fn)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
            collate_fn=collate_fn)
    else:
        def budget_loader(dataset, shuffle, loader_seed):
            sampler = AttentionBudgetBatchSampler(
                _sequence_lengths(dataset, max_seq_len=max_seq_len),
                max_batch_size=batch_size,
                max_attention_cells=max_attention_cells,
                shuffle=shuffle,
                seed=loader_seed,
            )
            return DataLoader(
                dataset, batch_sampler=sampler, num_workers=0,
                pin_memory=True, collate_fn=collate_fn,
            )

        train_loader = budget_loader(train_ds, True, seed)
        val_loader = budget_loader(val_ds, False, seed + 1)
        test_loader = budget_loader(test_ds, False, seed + 2)
        oversized = sum(
            loader.batch_sampler.oversized_singletons
            for loader in (train_loader, val_loader, test_loader)
        )
        print(
            "  [Data] Attention-budget batches: max_users=%d, "
            "max_cells=%d, train/val/test_batches=%d/%d/%d, "
            "irreducible_singletons=%d"
            % (batch_size, max_attention_cells, len(train_loader),
               len(val_loader), len(test_loader), oversized)
        )

    return train_loader, val_loader, test_loader
