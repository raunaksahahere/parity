from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from parity.stats.bootstrap import cluster_bootstrap_mean
from parity.stats.kl import truncated_kl


def dist(seed: int, v: int = 500, sharp: float = 1.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = -sharp * np.log(np.arange(1, v + 1)) + rng.normal(0, 0.3, v)
    return logits - np.log(np.exp(logits).sum())


def topk(lp: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-lp)[:k]
    return idx, lp[idx]


def true_kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.exp(p) * (p - q)))


def test_identical_distributions_have_zero_kl() -> None:
    p = dist(1)
    ids, lp = topk(p, 50)
    res = truncated_kl(ids, lp, ids, lp)
    assert res.kl == pytest.approx(0.0, abs=1e-9) and res.tv == pytest.approx(0.0, abs=1e-9)
    assert res.top1_match


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(3, 200), noise=st.floats(0.01, 1.0))
def test_truncated_kl_is_a_lower_bound_on_the_true_kl(seed: int, k: int, noise: float) -> None:
    p = dist(seed)
    rng = np.random.default_rng(seed + 1)
    q_logits = p + rng.normal(0, noise, len(p))
    q = q_logits - np.log(np.exp(q_logits).sum())
    est = truncated_kl(*topk(p, k), *topk(q, k))
    assert est.kl <= true_kl(p, q) + 1e-9
    assert est.kl >= 0.0


def test_bound_tightens_as_k_grows() -> None:
    p = dist(7)
    rng = np.random.default_rng(8)
    ql = p + rng.normal(0, 0.4, len(p))
    q = ql - np.log(np.exp(ql).sum())
    ests = [truncated_kl(*topk(p, k), *topk(q, k)).kl for k in (5, 20, 100, 500)]
    assert ests[0] <= ests[1] <= ests[2] <= ests[3] + 1e-12
    assert ests[3] == pytest.approx(true_kl(p, q), rel=1e-6)


def test_disagreeing_reported_sets_lose_divergence_rather_than_invent_it() -> None:
    p = dist(3)
    q = np.roll(p, 1)  # same shape, tokens shifted by one: very different distributions
    full = true_kl(p, q)
    est = truncated_kl(*topk(p, 10), *topk(q, 10))
    assert est.kl < full


def test_top1_mismatch_is_reported() -> None:
    p = np.log(np.array([0.6, 0.3, 0.1]))
    q = np.log(np.array([0.3, 0.6, 0.1]))
    ids = np.arange(3)
    assert not truncated_kl(ids, p, ids, q).top1_match


def test_cluster_bootstrap_ci_covers_the_truth_at_roughly_nominal_rate() -> None:
    rng = np.random.default_rng(0)
    covered, trials = 0, 200
    for t in range(trials):
        # 25 documents with strong within-document correlation and different sizes
        clusters = []
        for _ in range(25):
            base = rng.normal(0.0, 1.0)
            clusters.append(base + rng.normal(0, 0.3, rng.integers(20, 60)))
        res = cluster_bootstrap_mean(clusters, n_boot=500, seed=t)
        covered += res.lo <= 0.0 <= res.hi
    assert 0.88 <= covered / trials <= 0.99


def test_cluster_bootstrap_is_wider_than_naive_for_correlated_data() -> None:
    rng = np.random.default_rng(1)
    clusters = [rng.normal(rng.normal(0, 1), 0.1, 40) for _ in range(20)]
    res = cluster_bootstrap_mean(clusters, n_boot=1000)
    flat = np.concatenate(clusters)
    naive_halfwidth = 1.96 * flat.std(ddof=1) / np.sqrt(len(flat))
    assert (res.hi - res.lo) / 2 > naive_halfwidth


def test_bootstrap_p_value_direction() -> None:
    clusters = [np.full(30, 0.5) + np.random.default_rng(i).normal(0, 0.05, 30) for i in range(15)]
    res = cluster_bootstrap_mean(clusters)
    assert res.p_value_le(0.1) < 0.01
    assert res.p_value_le(1.0) > 0.99
