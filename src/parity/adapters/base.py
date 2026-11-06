"""Adapter interface: one place where provider quirks live."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from parity.client import EndpointClient
from parity.schema import Completion, PositionLogprobs


@dataclass(frozen=True)
class Capabilities:
    """What a provider can do; drives fallback to lower-power behavioural probes."""

    token_id_prompts: bool  # accepts prompts as lists of token ids (exact alignment)
    prompt_logprobs: bool  # returns log-probs of prompt tokens in one request
    top_logprobs: bool  # returns top-k alternatives at each position
    max_top_k: int = 20


class Adapter(ABC):
    name: str
    capabilities: Capabilities

    @abstractmethod
    async def teacher_force(
        self, client: EndpointClient, model: str, ids: list[int], *, top_k: int
    ) -> list[PositionLogprobs]:
        """One normalized position per predicted token ``ids[1:]``.

        ``logprob`` is the forced token's log-prob, or ``None`` if the provider did
        not report it (e.g. it fell outside the returned top-k).
        """

    @abstractmethod
    async def next_token(
        self,
        client: EndpointClient,
        model: str,
        prefix_ids: list[int],
        *,
        top_k: int,
        sampling: dict[str, Any],
    ) -> Completion:
        """Generate one token after ``prefix_ids`` under explicit sampling parameters."""

    async def generate(
        self,
        client: EndpointClient,
        model: str,
        prefix_ids: list[int],
        *,
        n_tokens: int,
        sampling: dict[str, Any],
    ) -> Completion:
        """Generate up to ``n_tokens`` tokens; ``positions[i].token_id`` is the i-th token id."""
        raise NotImplementedError(f"{self.name} adapter cannot generate from token-id prompts")
