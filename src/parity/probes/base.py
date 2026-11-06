"""Probe framework: context, result types and the probe base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from parity.adapters import Adapter
from parity.client import EndpointClient
from parity.tokenization import TokenizerKit


class Status(StrEnum):
    PASS = "PASS"
    DEVIATION = "DEVIATION"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"


class Finding(BaseModel):
    severity: Literal["info", "warn", "deviation"]
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ProbeResult(BaseModel):
    probe: str
    family: str
    version: str
    status: Status
    # H0: the endpoint conforms. None for deterministic checks (see ``hard_evidence``).
    p_value: float | None = None
    # True when the deviation is deterministic evidence (e.g. a token-count mismatch)
    # rather than the outcome of a statistical test.
    hard_evidence: bool = False
    effect: float | None = None
    effect_unit: str = ""
    ci: tuple[float, float] | None = None
    threshold: float | None = None
    n_requests: int = 0
    power: Literal["normal", "lower"] = "normal"
    metrics: dict[str, float] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReferenceLike(Protocol):
    """What probes need from a reference implementation."""

    def teacher_force(self, ids: list[int], top_k: int) -> dict[str, Any]: ...

    def next_token_logprobs(self, prefix_ids: list[int]) -> Any: ...

    def greedy(self, prefix_ids: list[int], n_tokens: int) -> list[int]: ...


@dataclass
class ProbeContext:
    client: EndpointClient
    adapter: Adapter
    model: str
    kit: TokenizerKit
    reference: ReferenceLike | None = None
    alpha: float = 0.05
    seed: int = 0
    budget: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def can_afford(self, n_more: int) -> bool:
        """False if sending ``n_more`` further requests would exceed the request budget."""
        return self.budget is None or self.client.requests_sent + n_more <= self.budget


class Probe(ABC):
    id: str
    family: str
    version: str = "1"

    @abstractmethod
    async def run(self, ctx: ProbeContext) -> ProbeResult: ...

    def result(self, status: Status, **kwargs: Any) -> ProbeResult:
        return ProbeResult(
            probe=self.id, family=self.family, version=self.version, status=status, **kwargs
        )
