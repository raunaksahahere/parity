"""Normalized log-probability schema shared by all provider adapters.

Every adapter converts a provider-specific response into these types so probes
never see provider quirks. Token ids are ``None`` when a provider reports only
token strings; alignment is then done on byte strings (see ``adapters``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopEntry(BaseModel):
    token_id: int | None = None
    token: str
    logprob: float
    token_bytes: list[int] | None = None


class PositionLogprobs(BaseModel):
    """Log-probabilities at one position: the chosen/forced token plus top-k."""

    token_id: int | None = None
    token: str
    logprob: float | None = None  # None when the chosen token's logprob is unavailable
    top: list[TopEntry] = Field(default_factory=list)


class Completion(BaseModel):
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    positions: list[PositionLogprobs] = Field(default_factory=list)
