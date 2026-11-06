"""Cluster bootstrap for statistics over positions grouped into documents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lo: float
    hi: float
    boot: np.ndarray  # bootstrap distribution of the statistic

    def p_value_le(self, threshold: float) -> float:
        """One-sided bootstrap p-value for H0: statistic <= threshold."""
        return float((np.sum(self.boot <= threshold) + 1) / (len(self.boot) + 1))


def cluster_bootstrap_mean(
    clusters: Sequence[np.ndarray],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap the pooled mean, resampling whole clusters (documents) with replacement.

    Positions inside one document are correlated, so resampling individual positions would
    understate uncertainty. Each cluster contributes in proportion to its size.
    """
    sums = np.array([float(c.sum()) for c in clusters])
    sizes = np.array([len(c) for c in clusters], dtype=float)
    keep = sizes > 0
    sums, sizes = sums[keep], sizes[keep]
    if len(sizes) == 0:
        raise ValueError("no data")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sizes), size=(n_boot, len(sizes)))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=float(sums.sum() / sizes.sum()),
        lo=float(np.quantile(boot, alpha)),
        hi=float(np.quantile(boot, 1.0 - alpha)),
        boot=boot,
    )
