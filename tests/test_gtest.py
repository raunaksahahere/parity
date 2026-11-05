from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from parity.stats.gtest import g_test, pool_small_cells


def zipf(k: int, s: float = 1.1) -> np.ndarray:
    w = 1.0 / np.arange(1, k + 1) ** s
    return w / w.sum()


def test_matches_scipy_log_likelihood_ratio_without_correction() -> None:
    probs = np.array([0.5, 0.3, 0.2])
    obs = np.array([480.0, 330.0, 190.0])
    ours = g_test(obs, probs)
    ref = stats.power_divergence(obs, probs * obs.sum(), lambda_="log-likelihood")
    williams = 1 + (3 + 1) / (6 * obs.sum())
    assert ours.g == pytest.approx(ref.statistic / williams)
    assert ours.df == 2


def test_false_positive_rate_matches_nominal_alpha() -> None:
    rng = np.random.default_rng(1)
    probs = zipf(40)
    alpha, sims, n = 0.05, 3000, 500
    pvals = np.array([g_test(rng.multinomial(n, probs), probs).p_value for _ in range(sims)])
    fpr = float((pvals < alpha).mean())
    # binomial standard error at 3000 sims is ~0.004; allow 4 sigma
    assert abs(fpr - alpha) < 4 * np.sqrt(alpha * (1 - alpha) / sims)
    # p-values are uniform under H0
    assert stats.kstest(pvals, "uniform").pvalue > 0.001


def test_power_increases_with_sample_size_and_effect() -> None:
    rng = np.random.default_rng(2)
    true = zipf(30, 1.1)
    # alternative: the sampler flattens the distribution (as if temperature were ignored)
    flat = true**0.8
    flat /= flat.sum()

    def power(n: int, alt: np.ndarray, sims: int = 300) -> float:
        hits = sum(g_test(rng.multinomial(n, alt), true).p_value < 0.05 for _ in range(sims))
        return hits / sims

    p_small, p_mid, p_big = power(100, flat), power(500, flat), power(2000, flat)
    assert p_small < p_mid < p_big
    assert p_big > 0.95
    # a tiny effect is hard to see at small n (power near alpha), an extreme one is easy
    near = true**0.98
    near /= near.sum()
    assert power(200, near) < 0.2
    very_flat = true**0.5
    very_flat /= very_flat.sum()
    assert power(200, very_flat) > 0.9


def test_total_variation_reports_the_effect() -> None:
    probs = np.array([0.5, 0.5])
    res = g_test(np.array([700.0, 300.0]), probs)
    assert res.tv == pytest.approx(0.2)
    assert res.p_value < 1e-10


def test_small_cells_are_pooled_and_counts_preserved() -> None:
    probs = np.array([0.6, 0.38, 0.01, 0.005, 0.005])
    obs = np.array([60.0, 38.0, 1.0, 0.0, 1.0])
    o, p = pool_small_cells(obs, probs, 100, 5.0)
    assert o.sum() == obs.sum() and p.sum() == pytest.approx(1.0)
    assert len(o) == 3


def test_zero_probability_cells_with_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero expected probability"):
        g_test(np.array([5.0, 1.0]), np.array([1.0, 0.0]))


def test_degenerate_single_cell_is_a_trivial_pass() -> None:
    res = g_test(np.array([10.0, 0.0]), np.array([1.0, 0.0]))
    assert res.p_value == 1.0 and res.df == 0


def test_rejects_non_probability_vectors() -> None:
    with pytest.raises(ValueError, match="probability vector"):
        g_test(np.array([1.0, 1.0]), np.array([0.7, 0.7]))
