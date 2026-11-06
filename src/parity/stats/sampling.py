"""Expected next-token distributions under sampler settings.

Inputs are the target's own reported log-probabilities at temperature 1 with no
truncation, so any difference between the expected and the observed samples is
attributable to the sampler rather than to model weights.

Ordering matters for ``top_p`` combined with ``temperature != 1``: HF/vLLM apply
temperature first, llama.cpp truncates first. Only combinations whose result does
not depend on the order are supported (``top_k`` with any temperature, ``top_p``
and ``min_p`` at temperature 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class UnsupportedSamplerCaseError(ValueError):
    pass


@dataclass(frozen=True)
class SamplerSettings:
    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0
    min_p: float = 0.0

    def validate(self) -> None:
        if self.temperature <= 0:
            raise UnsupportedSamplerCaseError("temperature must be positive")
        if self.temperature != 1.0 and (self.top_p < 1.0 or self.min_p > 0.0):
            raise UnsupportedSamplerCaseError(
                "top_p/min_p with temperature != 1 depend on sampler order; not supported"
            )

    def api_params(self) -> dict[str, float | int]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
        }


@dataclass(frozen=True)
class Expected:
    ids: np.ndarray  # token ids in the expected support, descending by probability
    probs: np.ndarray  # conditional probabilities on the support, summing to 1
    tail_mass: float  # reported-set mass lost to truncation (not in support)
    unreported_mass: float  # mass outside the provider's reported top-K at T=1
    boundary_gap: float  # log-prob gap at the truncation boundary (inf if no truncation)
    # Distance between the top_p/min_p cutoff and the quantity it is compared with (probability
    # units for top_p, nats for min_p); small values mean numerical jitter can move the boundary.
    cutoff_margin: float = float("inf")


def expected_distribution(
    ids: np.ndarray, logprobs: np.ndarray, settings: SamplerSettings
) -> Expected:
    """Conditional expected distribution on the support kept by ``settings``.

    ``ids``/``logprobs`` are the provider's reported top-K at temperature 1 (any order).
    Samples that land outside the reported set cannot be tested conditionally and are
    handled by the caller.
    """
    settings.validate()
    order = np.argsort(-logprobs, kind="stable")
    ids, lp = ids[order], logprobs[order]
    p1 = np.exp(lp)
    unreported = max(0.0, 1.0 - float(p1.sum()))
    gap = float("inf")
    margin = float("inf")

    keep = len(ids)
    if settings.top_k > 0 and settings.top_k < keep:
        gap = min(gap, float(lp[settings.top_k - 1] - lp[settings.top_k]))
        keep = settings.top_k
    if settings.top_p < 1.0:
        csum = np.cumsum(p1)
        # smallest prefix whose cumulative probability reaches top_p
        n_p = int(np.searchsorted(csum, settings.top_p - 1e-12) + 1)
        before = float(csum[n_p - 2]) if n_p >= 2 else 0.0
        reached = float(csum[min(n_p, len(csum)) - 1])
        margin = min(margin, settings.top_p - before, reached - settings.top_p)
        if n_p < keep:
            if n_p < len(lp):
                gap = min(gap, float(lp[n_p - 1] - lp[n_p]))
            keep = n_p
    if settings.min_p > 0.0:
        thresh = settings.min_p * p1[0]
        n_m = int(np.sum(p1 >= thresh))
        log_t = float(np.log(thresh))
        if n_m < len(lp):
            margin = min(margin, float(lp[n_m - 1]) - log_t, log_t - float(lp[n_m]))
        if n_m < keep:
            keep = n_m
            if keep < len(lp):
                gap = min(gap, float(lp[keep - 1] - lp[keep]))

    sel_ids, sel_lp = ids[:keep], lp[:keep]
    scaled = sel_lp / settings.temperature
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return Expected(
        ids=sel_ids,
        probs=probs,
        tail_mass=float(p1[keep:].sum()),
        unreported_mass=unreported,
        boundary_gap=gap,
        cutoff_margin=margin,
    )


def stable_top_p(
    logprobs: np.ndarray, nominal: float, *, min_margin: float = 0.02, min_tail: float = 0.05
) -> float | None:
    """A top_p value near ``nominal`` that no small numerical jitter can move across a boundary.

    With cumulative masses c_1 < c_2 < ... (tokens sorted by probability), nucleus sampling at
    p keeps n tokens exactly when c_{n-1} < p <= c_n. Choosing p at the midpoint of such an
    interval puts it at least half the n-th token's probability away from either boundary.
    Returns the midpoint closest to ``nominal`` whose margin is >= ``min_margin`` and that
    leaves at least ``min_tail`` of the mass excluded (so an ignored top_p is detectable),
    or None if no interval qualifies.
    """
    c = np.cumsum(np.sort(np.exp(logprobs))[::-1])
    best: float | None = None
    for n in range(2, len(c) + 1):
        lo, hi = float(c[n - 2]), float(c[n - 1])
        if hi - lo < 2 * min_margin or 1.0 - hi < min_tail:
            continue
        mid = (lo + hi) / 2
        if best is None or abs(mid - nominal) < abs(best - nominal):
            best = mid
    return best
