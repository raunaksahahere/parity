from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from parity.stats.multitest import holm_bonferroni
from parity.stats.sprt import BinomialSPRT, Decision, GaussianSPRT, fixed_sample_size_binomial


def test_holm_known_example() -> None:
    res = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, alpha=0.05)
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert res.adjusted == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})
    assert res.reject == {"d": True, "a": True, "c": False, "b": False}


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(0.0, 1.0), min_size=1, max_size=12))
def test_holm_dominates_bonferroni_and_is_bounded(ps: list[float]) -> None:
    pv = {f"p{i}": p for i, p in enumerate(ps)}
    holm = holm_bonferroni(pv, 0.05)
    for k, p in pv.items():
        bonf = min(1.0, p * len(pv))
        assert holm.adjusted[k] <= bonf + 1e-12
        assert p - 1e-12 <= holm.adjusted[k] <= 1.0
        if bonf < 0.05:
            assert holm.reject[k]


def test_holm_controls_familywise_error_under_the_global_null() -> None:
    rng = np.random.default_rng(0)
    sims, m = 4000, 8
    false_any = sum(
        any(
            holm_bonferroni(
                {str(i): float(p) for i, p in enumerate(rng.uniform(size=m))}
            ).reject.values()
        )
        for _ in range(sims)
    )
    fwer = false_any / sims
    assert fwer <= 0.05 + 4 * np.sqrt(0.05 * 0.95 / sims)


def test_holm_rejects_alpha_validation() -> None:
    with pytest.raises(ValueError):
        holm_bonferroni({"a": 0.5}, alpha=1.5)


def run_binomial(
    rng: np.random.Generator, p_true: float, p0: float, p1: float, cap: int = 5000
) -> tuple[Decision, int]:
    t = BinomialSPRT(p0, p1, alpha=0.05, beta=0.1)
    for _ in range(cap):
        d = t.update(bool(rng.random() < p_true))
        if d is not Decision.CONTINUE:
            return d, t.n
    return Decision.CONTINUE, cap


def test_binomial_sprt_operating_characteristics() -> None:
    rng = np.random.default_rng(3)
    p0, p1, trials = 0.02, 0.10, 1500
    h0 = [run_binomial(rng, p0, p0, p1) for _ in range(trials)]
    h1 = [run_binomial(rng, p1, p0, p1) for _ in range(trials)]
    type1 = np.mean([d is Decision.ACCEPT_H1 for d, _ in h0])
    type2 = np.mean([d is Decision.ACCEPT_H0 for d, _ in h1])
    assert type1 <= 0.05 + 0.015
    assert type2 <= 0.10 + 0.02
    # the point of sequential testing: fewer samples than the fixed-n test on average
    fixed = fixed_sample_size_binomial(p0, p1, 0.05, 0.10)
    assert np.mean([n for _, n in h0]) < fixed
    assert np.mean([n for _, n in h1]) < fixed


def test_binomial_sprt_stops_much_sooner_for_gross_deviations() -> None:
    rng = np.random.default_rng(4)
    ns = [run_binomial(rng, 0.6, 0.02, 0.10)[1] for _ in range(200)]
    assert np.mean(ns) < 30


def test_binomial_sprt_validates_parameters() -> None:
    with pytest.raises(ValueError):
        BinomialSPRT(0.5, 0.2)


def test_gaussian_sprt_operating_characteristics() -> None:
    rng = np.random.default_rng(5)
    mu0, mu1, sigma, trials = 0.0, 0.5, 1.0, 1000

    def run(mu: float) -> tuple[Decision, int]:
        t = GaussianSPRT(mu0, mu1, sigma, alpha=0.05, beta=0.1)
        for _ in range(2000):
            d = t.update(float(rng.normal(mu, sigma)))
            if d is not Decision.CONTINUE:
                return d, t.n
        return Decision.CONTINUE, 2000

    h0 = [run(mu0) for _ in range(trials)]
    h1 = [run(mu1) for _ in range(trials)]
    assert np.mean([d is Decision.ACCEPT_H1 for d, _ in h0]) <= 0.065
    assert np.mean([d is Decision.ACCEPT_H0 for d, _ in h1]) <= 0.12
