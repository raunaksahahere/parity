"""Ollama adapter (OpenAI-compatible ``/v1`` API of an Ollama server).

Verified against Ollama 0.33.2 (docs/data/v0.2.0-ollama-capabilities-*.json, decision D17):

* ``/v1/completions`` rejects token-id prompts (HTTP 400) and returns no prompt log-probs, so
  the distribution, greedy and sampler probes are skipped and reports are marked lower power.
* ``/v1/chat/completions`` reports ``usage.prompt_tokens`` (template and long-context probes
  run) and log-probs of generated tokens.
* ``chat_template_kwargs`` is ignored. Thinking models are switched off with
  ``reasoning_effort: "none"``, which this adapter sends whenever a probe asks for
  ``enable_thinking: false``.
* ``tool_choice`` ("none" and "required") is ignored; models whose library template has no tool
  format reject ``tools`` with HTTP 400 ("does not support tools").
"""

from __future__ import annotations

from typing import Any

from parity.adapters.base import Capabilities
from parity.adapters.openai import OpenAIAdapter


class OllamaAdapter(OpenAIAdapter):
    name = "ollama"
    capabilities = Capabilities(
        token_id_prompts=False,
        prompt_logprobs=False,
        top_logprobs=False,
        max_top_k=20,
        chat_logprobs=True,
    )

    def chat_params(self, params: dict[str, Any]) -> dict[str, Any]:
        out = dict(params)
        kwargs = dict(out.pop("chat_template_kwargs", None) or {})
        if kwargs.get("enable_thinking") is False:
            out.setdefault("reasoning_effort", "none")
        return out
