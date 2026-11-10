"""Exact runtime-array checks tolerate only discarded intermediate precision."""
from copy import deepcopy
import numpy as np
from SHAPE.stage3.prototype_semantics import attraction_effective_sha256


def test_digest_ignores_intermediate_hash_but_detects_every_effective_array_change():
    stats = dict(attraction_table=np.array([[1.,2.]],dtype=np.float32),
                 support=np.array([3],dtype=np.int64),
                 category_probabilities=np.array([[.5,.5]],dtype=np.float32),
                 role=['general'], table_sha256='float64-intermediate-a')
    digest = attraction_effective_sha256(stats)
    other = dict(stats, table_sha256='float64-intermediate-b')
    assert attraction_effective_sha256(other) == digest
    for key in ('attraction_table','support','category_probabilities'):
        changed = deepcopy(stats)
        changed[key].flat[0] += 1
        assert attraction_effective_sha256(changed) != digest
    assert attraction_effective_sha256(dict(stats,role=['residential'])) != digest
