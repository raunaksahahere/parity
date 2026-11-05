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

    keep = len(ids)
    if settings.top_k > 0 and settings.top_k < keep:
        gap = min(gap, float(lp[settings.top_k - 1] - lp[settings.top_k]))
        keep = settings.top_k
    if settings.top_p < 1.0:
        csum = np.cumsum(p1)
        # smallest prefix whose cumulative probability reaches top_p
        n_p = int(np.searchsorted(csum, settings.top_p - 1e-12) + 1)
        if n_p < keep:
            if n_p < len(lp):
                gap = min(gap, float(lp[n_p - 1] - lp[n_p]))
            keep = n_p
    if settings.min_p > 0.0:
        thresh = settings.min_p * p1[0]
        n_m = int(np.sum(p1 >= thresh))
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
    )
