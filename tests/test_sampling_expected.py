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


def test_top_p_cutoff_margin_measures_distance_to_the_cumulative_mass() -> None:
    lp = np.log(np.array([0.5, 0.098, 0.2, 0.202]))  # sorted: .5 .202 .2 .098
    ids = np.arange(4)
    exp = expected_distribution(ids, lp, SamplerSettings(top_p=0.7))
    # cumulative: .5, .702 -> keeps 2 tokens; .702 is only .002 above the cutoff
    assert len(exp.ids) == 2
    assert exp.cutoff_margin == pytest.approx(0.002, abs=1e-9)
    safe = expected_distribution(ids, lp, SamplerSettings(top_p=0.6))
    assert safe.cutoff_margin == pytest.approx(0.1, abs=1e-9)  # .6 vs .5 before / .702 after
    assert expected_distribution(ids, lp, SamplerSettings(top_k=2)).cutoff_margin == float("inf")


def test_min_p_cutoff_margin_is_in_log_space() -> None:
    p = np.array([0.5, 0.26, 0.24])
    exp = expected_distribution(np.arange(3), np.log(p), SamplerSettings(min_p=0.5))
    assert [int(i) for i in exp.ids] == [0, 1]
    assert exp.cutoff_margin == pytest.approx(min(np.log(0.26 / 0.25), np.log(0.25 / 0.24)))


def test_stable_top_p_sits_mid_interval_near_the_nominal_value() -> None:
    from parity.stats.sampling import stable_top_p

    p = np.array([0.30, 0.20, 0.1005, 0.1, 0.1, 0.0995, 0.1])
    chosen = stable_top_p(np.log(p), 0.6)
    assert chosen is not None
    c = np.cumsum(np.sort(p)[::-1])
    assert min(np.abs(c - chosen)) >= 0.02  # never close to a cumulative boundary
    exp = expected_distribution(np.arange(len(p)), np.log(p), SamplerSettings(top_p=chosen))
    assert exp.cutoff_margin >= 0.02
    # nothing qualifies when no token is large enough to give a margin
    assert stable_top_p(np.log(np.full(100, 0.01)), 0.6) is None
