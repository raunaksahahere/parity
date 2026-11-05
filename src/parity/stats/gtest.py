"""Goodness-of-fit G-test (log-likelihood ratio) for categorical samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class GTestResult:
    g: float  # Williams-corrected G statistic
    df: int
    p_value: float
    n: int
    n_cells: int  # cells after pooling
    tv: float  # plug-in total variation distance between empirical and expected


def pool_small_cells(
    observed: np.ndarray, probs: np.ndarray, n: int, min_expected: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge cells with expected count below ``min_expected`` into one pooled cell.

    The chi-square approximation needs reasonably large expected counts; pooling the
    smallest cells keeps the test valid without discarding any observations.
    """
    expected = probs * n
    small = expected < min_expected
    if small.sum() < 2:
        return observed, probs
    keep = ~small
    pooled_obs = observed[small].sum()
    pooled_p = probs[small].sum()
    return (
        np.concatenate([observed[keep], [pooled_obs]]),
        np.concatenate([probs[keep], [pooled_p]]),
    )


def g_test(observed: np.ndarray, probs: np.ndarray, *, min_expected: float = 5.0) -> GTestResult:
    """Test H0: ``observed`` counts come from the categorical distribution ``probs``.

    ``probs`` must sum to 1 (up to float error) and be non-negative. Cells with zero
    probability must be handled by the caller (they are deterministic violations,
    not part of a G-test).
    """
    observed = np.asarray(observed, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if observed.shape != probs.shape:
        raise ValueError("observed and probs must have the same shape")
    if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError("probs must be a probability vector")
    n = int(observed.sum())
    if n == 0:
        raise ValueError("no observations")
    probs = probs / probs.sum()
    live = probs > 0
    if np.any(observed[~live] > 0):
        raise ValueError("observations in cells with zero expected probability")
    observed, probs = observed[live], probs[live]
    observed, probs = pool_small_cells(observed, probs, n, min_expected)
    k = len(probs)
    emp = observed / n
    tv = 0.5 * float(np.abs(emp - probs).sum())
    if k < 2:
        return GTestResult(g=0.0, df=0, p_value=1.0, n=n, n_cells=k, tv=tv)
    mask = observed > 0
    g = 2.0 * float(np.sum(observed[mask] * np.log(observed[mask] / (probs[mask] * n))))
    williams = 1.0 + (k + 1.0) / (6.0 * n)
    g_adj = g / williams
    df = k - 1
    return GTestResult(
        g=g_adj, df=df, p_value=float(stats.chi2.sf(g_adj, df)), n=n, n_cells=k, tv=tv
    )


def expected_null_tv(
    probs: np.ndarray, n: int, *, sims: int = 100, seed: int = 0, min_expected: float = 5.0
) -> float:
    """Mean plug-in TV distance when the samples really do come from ``probs``.

    The plug-in TV estimator is biased upward by roughly sqrt(cells / n); subtracting this
    null expectation gives an effect size that is ~0 for a conforming sampler.
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return float(
        np.mean(
            [
                g_test(rng.multinomial(n, probs), probs, min_expected=min_expected).tv
                for _ in range(sims)
            ]
        )
    )
