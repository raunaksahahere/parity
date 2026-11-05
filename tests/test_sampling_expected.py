from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from parity.stats.sampling import (
    SamplerSettings,
    UnsupportedSamplerCaseError,
    expected_distribution,
)

IDS = np.arange(100, 108)
LOGITS = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0])
LP = LOGITS - np.log(np.exp(LOGITS).sum())  # a normalized distribution over 8 tokens


def brute_force(settings_: SamplerSettings) -> dict[int, float]:
    """Independent reference: HF-style processing on a dict, one token at a time."""
    probs = {int(i): float(np.exp(lp)) for i, lp in zip(IDS, LP, strict=True)}
    ranked = sorted(probs, key=lambda t: -probs[t])
    if settings_.top_k:
        ranked = ranked[: settings_.top_k]
    if settings_.top_p < 1.0:
        kept, cum = [], 0.0
        for t in ranked:
            kept.append(t)
            cum += probs[t]
            if cum >= settings_.top_p:
                break
        ranked = kept
    w = {t: probs[t] ** (1.0 / settings_.temperature) for t in ranked}
    z = sum(w.values())
    return {t: v / z for t, v in w.items()}


@pytest.mark.parametrize(
    "s",
    [
        SamplerSettings(),
        SamplerSettings(temperature=0.7),
        SamplerSettings(temperature=1.4),
        SamplerSettings(top_k=3),
        SamplerSettings(top_k=1),
        SamplerSettings(temperature=0.6, top_k=5),
        SamplerSettings(top_p=0.8),
        SamplerSettings(top_p=0.5, top_k=4),
    ],
)
def test_matches_brute_force(s: SamplerSettings) -> None:
    exp = expected_distribution(IDS, LP, s)
    ref = brute_force(s)
    assert set(exp.ids.tolist()) == set(ref)
    for t, p in zip(exp.ids.tolist(), exp.probs.tolist(), strict=True):
        assert p == pytest.approx(ref[t], rel=1e-9)
    assert exp.probs.sum() == pytest.approx(1.0)


def test_min_p_keeps_tokens_relative_to_the_top_token() -> None:
    exp = expected_distribution(IDS, LP, SamplerSettings(min_p=0.3))
    p1 = np.exp(LP)
    assert set(exp.ids.tolist()) == {
        int(i) for i, p in zip(IDS, p1, strict=True) if p >= 0.3 * p1.max()
    }


def test_input_order_does_not_matter() -> None:
    perm = np.random.default_rng(0).permutation(len(IDS))
    a = expected_distribution(IDS, LP, SamplerSettings(top_k=4))
    b = expected_distribution(IDS[perm], LP[perm], SamplerSettings(top_k=4))
    assert a.ids.tolist() == b.ids.tolist()
    np.testing.assert_allclose(a.probs, b.probs)


def test_boundary_gap_and_tail_mass_reported() -> None:
    exp = expected_distribution(IDS, LP, SamplerSettings(top_k=3))
    assert exp.boundary_gap == pytest.approx(LP[2] - LP[3])
    assert exp.tail_mass == pytest.approx(float(np.exp(LP[3:]).sum()))
    assert expected_distribution(IDS, LP, SamplerSettings()).boundary_gap == float("inf")


def test_unreported_mass_when_topk_is_partial() -> None:
    exp = expected_distribution(IDS[:4], LP[:4], SamplerSettings())
    assert exp.unreported_mass == pytest.approx(float(np.exp(LP[4:]).sum()))


def test_order_dependent_combinations_are_refused() -> None:
    with pytest.raises(UnsupportedSamplerCaseError):
        expected_distribution(IDS, LP, SamplerSettings(temperature=0.7, top_p=0.9))


@settings(max_examples=100, deadline=None)
@given(
    t=st.floats(0.2, 3.0),
    k=st.integers(0, 8),
)
def test_expected_is_always_a_valid_distribution_over_the_top_tokens(t: float, k: int) -> None:
    exp = expected_distribution(IDS, LP, SamplerSettings(temperature=t, top_k=k))
    assert exp.probs.sum() == pytest.approx(1.0)
    assert (exp.probs > 0).all()
    assert len(exp.ids) == (k if k else len(IDS))
    assert np.all(np.diff(exp.probs) <= 1e-12)  # ranked descending
