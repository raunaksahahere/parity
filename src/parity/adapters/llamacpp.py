"""llama.cpp server adapter.

``echo``/prompt log-probs are not supported, so teacher forcing is done one
position at a time with token-id prompts (the server's prompt cache keeps the
incremental cost to one token per request). The forced token's log-prob is only
known when it is inside the returned top-k; otherwise it is reported as ``None``.
"""

from __future__ import annotations

from typing import Any

from parity.adapters.base import Adapter, Capabilities
from parity.adapters.parsing import parse_logprobs
from parity.client import EndpointClient
from parity.schema import Completion, PositionLogprobs

# Sent explicitly because the server applies its own defaults (top_k=40, top_p=0.95, min_p=0.05).
NEUTRAL_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
}


class LlamaCppAdapter(Adapter):
    name = "llamacpp"
    capabilities = Capabilities(
        token_id_prompts=True, prompt_logprobs=False, top_logprobs=True, max_top_k=1000
    )

    async def next_token(
        self,
        client: EndpointClient,
        model: str,
        prefix_ids: list[int],
        *,
        top_k: int,
        sampling: dict[str, Any],
    ) -> Completion:
        data = await client.completions(
            model,
            prefix_ids,
            max_tokens=1,
            logprobs=top_k,
            **{**NEUTRAL_SAMPLING, **sampling},
        )
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return Completion(
            text=choice.get("text", ""),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            positions=parse_logprobs(choice.get("logprobs") or {}),
        )

    async def generate(
        self,
        client: EndpointClient,
        model: str,
        prefix_ids: list[int],
        *,
        n_tokens: int,
        sampling: dict[str, Any],
    ) -> Completion:
        data = await client.completions(
            model,
            prefix_ids,
            max_tokens=n_tokens,
            logprobs=1,
            **{**NEUTRAL_SAMPLING, **sampling},
        )
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return Completion(
            text=choice.get("text", ""),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            positions=parse_logprobs(choice.get("logprobs") or {}),
        )

    async def teacher_force(
        self, client: EndpointClient, model: str, ids: list[int], *, top_k: int
    ) -> list[PositionLogprobs]:
        out: list[PositionLogprobs] = []
        for i in range(1, len(ids)):
            comp = await self.next_token(
                client,
                model,
                ids[:i],
                top_k=top_k,
                sampling={"temperature": 1.0, "seed": 0},
            )
            pos = comp.positions[0] if comp.positions else PositionLogprobs(token="")
            forced = next((t for t in pos.top if t.token_id == ids[i]), None)
            out.append(
                PositionLogprobs(
                    token_id=ids[i],
                    token=forced.token if forced else "",
                    logprob=forced.logprob if forced else None,
                    top=pos.top,
                )
            )
        return out
