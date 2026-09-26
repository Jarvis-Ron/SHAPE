"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np


CODES_SCHEMA = "civic_phase2_daily_codes_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(()).item()
    return value


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def require_scalar(payload, key, expected=None):
    if key not in payload:
        raise RuntimeError("Artifact is missing required field %r" % key)
    value = scalar(payload[key])
    if expected is not None and value != expected:
        raise RuntimeError(
            "Artifact %s mismatch: got %r, expected %r"
            % (key, value, expected)
        )
    return value


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def json_scalar(payload):
    return np.asarray(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), dtype=np.str_
    )
