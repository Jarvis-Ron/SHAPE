"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import numpy as np


def jensen_shannon_distribution(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("JSD arrays must share a shape")
    left = left / max(float(left.sum()), 1.0)
    right = right / max(float(right.sum()), 1.0)
    middle = 0.5 * (left + right)

    def divergence(values):
        keep = values > 0
        return float(np.sum(values[keep] * np.log2(values[keep] / middle[keep])))

    return 0.5 * (divergence(left) + divergence(right))
