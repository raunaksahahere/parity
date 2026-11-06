"""vLLM adapter: native ``prompt_logprobs`` gives teacher forcing in a single request.

Implemented against vLLM's documented response format and covered by recorded
fixtures; it has not been validated against a live vLLM server in this repository.
"""

from __future__ import annotations

from typing import Any

from parity.adapters.base import Adapter, Capabilities
from parity.adapters.parsing import parse_logprobs, parse_vllm_prompt_logprobs
from parity.client import EndpointClient
from parity.schema import Completion, PositionLogprobs


class VLLMAdapter(Adapter):
    name = "vllm"
    capabilities = Capabilities(
        token_id_prompts=True, prompt_logprobs=True, top_logprobs=True, max_top_k=20
    )

    async def teacher_force(
        self, client: EndpointClient, model: str, ids: list[int], *, top_k: int
    ) -> list[PositionLogprobs]:
        data = await client.completions(
            model, ids, max_tokens=1, temperature=0.0, prompt_logprobs=top_k
        )
        choice = data["choices"][0]
        return parse_vllm_prompt_logprobs(choice.get("prompt_logprobs") or [], ids)

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
            model, prefix_ids, max_tokens=n_tokens, logprobs=1, return_token_ids=True, **sampling
        )
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        positions = parse_logprobs(choice.get("logprobs") or {})
        ids = choice.get("token_ids")
        if ids and len(ids) == len(positions):
            for pos, tid in zip(positions, ids, strict=True):
                pos.token_id = int(tid)
        return Completion(
            text=choice.get("text", ""),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            positions=positions,
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
            return_tokens_as_token_ids=True,
            **sampling,
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
