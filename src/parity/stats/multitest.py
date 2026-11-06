"""Multiple-testing correction across probes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HolmResult:
    adjusted: dict[str, float]  # Holm-adjusted p-values (monotone, capped at 1)
    reject: dict[str, bool]  # adjusted p < alpha


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> HolmResult:
    """Holm's step-down procedure: controls the familywise error rate at ``alpha``.

    Uniformly more powerful than plain Bonferroni and valid under arbitrary dependence
    between the tests.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    m = len(pvalues)
    order = sorted(pvalues, key=lambda k: pvalues[k])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(order):
        p = min(1.0, (m - rank) * pvalues[key])
        running = max(running, p)  # enforce monotonicity
        adjusted[key] = running
    return HolmResult(adjusted=adjusted, reject={k: v < alpha for k, v in adjusted.items()})
