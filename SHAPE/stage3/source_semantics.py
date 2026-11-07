"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from SHAPE.stage3.contracts import CODES_SCHEMA, load_npz, require_scalar


def _load_codes_artifact(reference_codes):
    artifact = (
        load_npz(reference_codes)
        if isinstance(reference_codes, (str, os.PathLike, Path))
        else reference_codes
    )
    require_scalar(artifact, "schema", CODES_SCHEMA)
    return artifact




def derive_projected_train_homes(
    reference_codes, projected_grid, sequence_users
):
    """Resolve modal train homes after vocabulary projection with stable ties."""
    artifact = _load_codes_artifact(reference_codes)
    event_users = np.asarray(artifact["event_user_ids"], dtype=np.str_)
    event_splits = np.asarray(artifact["event_splits"], dtype=np.str_)
    projected_grid = np.asarray(projected_grid, dtype=np.int64)
    if len(projected_grid) != len(event_users):
        raise RuntimeError("P3-B1 projected home arrays are not event-aligned")
    counts = defaultdict(Counter)
    for user, split, grid in zip(event_users, event_splits, projected_grid):
        if str(split) == "train":
            counts[str(user)][int(grid)] += 1
    homes = []
    mapping = {}
    for user in np.asarray(sequence_users, dtype=np.str_):
        ranking = sorted(
            counts[str(user)].items(), key=lambda item: (-item[1], item[0])
        )
        if not ranking:
            raise RuntimeError(
                "P3-B1 selected user has no projected train home: %s" % user
            )
        home = int(ranking[0][0])
        homes.append(home)
        mapping[str(user)] = home
    homes = np.asarray(homes, dtype=np.int64)
    return homes, {
        "contract": "projected_train_mode_count_desc_grid_id_asc_v1",
        "fit_split": "train_only",
        "sequence_count": int(len(homes)),
        "unique_user_count": int(len(mapping)),
        "user_home_sha256": hashlib.sha256(json.dumps(
            mapping, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }
