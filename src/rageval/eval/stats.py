"""Paired statistics. Every English/Arabic comparison uses the same questions, so tests are paired:
question difficulty cancels out of the difference instead of widening its interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Mean of a - b with a percentile bootstrap CI, resampling questions (pairs stay together)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty and the same length")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_resamples, diff.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high)}


def mcnemar_exact(a_hit: Sequence[bool], b_hit: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test on paired binary outcomes.

    Only discordant questions carry information: `a_only` hit under A but not B, `b_only` the
    reverse. Under the null both are equally likely, so min(a_only, b_only) follows a
    Binomial(a_only + b_only, 0.5) lower tail; the two-sided p-value doubles it.
    """
    if len(a_hit) != len(b_hit):
        raise ValueError("a_hit and b_hit must be the same length")
    a_only = sum(bool(x) and not bool(y) for x, y in zip(a_hit, b_hit))
    b_only = sum(bool(y) and not bool(x) for x, y in zip(a_hit, b_hit))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return {"a_only": a_only, "b_only": b_only, "p_value": min(1.0, 2 * tail)}
