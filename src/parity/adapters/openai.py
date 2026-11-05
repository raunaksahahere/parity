"""Generic OpenAI-compatible adapter (text prompts, legacy ``echo`` log-probs).

Token ids cannot be sent, so alignment with the reference is by byte offsets and
positions where the two tokenizations disagree are dropped (see ``alignment``).
Many hosted chat-only endpoints expose no prompt log-probs at all; then the
distributional probe is unavailable and behavioural probes are marked lower power.
"""

from __future__ import annotations

from typing import Any

from parity.adapters.base import Adapter, Capabilities
from parity.adapters.parsing import parse_logprobs
from parity.client import EndpointClient
from parity.schema import Completion, PositionLogprobs


class OpenAIAdapter(Adapter):
    name = "openai"
    capabilities = Capabilities(
        token_id_prompts=False, prompt_logprobs=True, top_logprobs=True, max_top_k=5
    )

    async def teacher_force(
        self, client: EndpointClient, model: str, ids: list[int], *, top_k: int
    ) -> list[PositionLogprobs]:
        raise NotImplementedError(
            "OpenAI-style echo needs the text of the prompt; use teacher_force_text"
        )

    async def teacher_force_text(
        self, client: EndpointClient, model: str, text: str, *, top_k: int
    ) -> list[PositionLogprobs]:
        data = await client.completions(
            model, text, max_tokens=0, echo=True, logprobs=min(top_k, self.capabilities.max_top_k)
        )
        positions = parse_logprobs(data["choices"][0].get("logprobs") or {})
        return positions[1:]  # first prompt token has no preceding context

    async def next_token(
        self,
        client: EndpointClient,
        model: str,
        prefix_ids: list[int],
        *,
        top_k: int,
        sampling: dict[str, Any],
    ) -> Completion:
        raise NotImplementedError("the generic adapter cannot send token-id prompts")
