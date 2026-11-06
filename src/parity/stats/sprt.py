"""Wald sequential probability ratio tests for early stopping under a request budget."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    CONTINUE = "continue"
    ACCEPT_H0 = "accept_h0"
    ACCEPT_H1 = "accept_h1"


@dataclass
class BinomialSPRT:
    """Test a Bernoulli rate: H0 ``p = p0`` against H1 ``p = p1`` (``p1 > p0``).

    ``alpha`` bounds P(accept H1 | H0) and ``beta`` bounds P(accept H0 | H1) (Wald bounds,
    approximate because the log-likelihood ratio can overshoot a boundary).
    """

    p0: float
    p1: float
    alpha: float = 0.05
    beta: float = 0.2
    llr: float = 0.0
    n: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.p0 < self.p1 < 1:
            raise ValueError("need 0 < p0 < p1 < 1")
        self.upper = math.log((1 - self.beta) / self.alpha)
        self.lower = math.log(self.beta / (1 - self.alpha))

    def update(self, success: bool) -> Decision:
        self.n += 1
        if success:
            self.llr += math.log(self.p1 / self.p0)
        else:
            self.llr += math.log((1 - self.p1) / (1 - self.p0))
        return self.decision

    @property
    def decision(self) -> Decision:
        if self.llr >= self.upper:
            return Decision.ACCEPT_H1
        if self.llr <= self.lower:
            return Decision.ACCEPT_H0
        return Decision.CONTINUE


@dataclass
class GaussianSPRT:
    """Test a mean with known variance: H0 ``mu = mu0`` against H1 ``mu = mu1`` (``mu1 > mu0``)."""

    mu0: float
    mu1: float
    sigma: float
    alpha: float = 0.05
    beta: float = 0.2
    llr: float = 0.0
    n: int = 0

    def __post_init__(self) -> None:
        if not self.mu1 > self.mu0 or self.sigma <= 0:
            raise ValueError("need mu1 > mu0 and sigma > 0")
        self.upper = math.log((1 - self.beta) / self.alpha)
        self.lower = math.log(self.beta / (1 - self.alpha))

    def update(self, x: float) -> Decision:
        self.n += 1
        self.llr += (self.mu1 - self.mu0) * (x - (self.mu0 + self.mu1) / 2.0) / self.sigma**2
        if self.llr >= self.upper:
            return Decision.ACCEPT_H1
        if self.llr <= self.lower:
            return Decision.ACCEPT_H0
        return Decision.CONTINUE


def fixed_sample_size_binomial(p0: float, p1: float, alpha: float, beta: float) -> int:
    """Normal-approximation sample size of the fixed-n one-sided test, for comparison."""
    from scipy.stats import norm

    za, zb = float(norm.ppf(1 - alpha)), float(norm.ppf(1 - beta))
    num = za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))
    return math.ceil((num / (p1 - p0)) ** 2)
