"""Tokenizer and chat-template rendering, independent of any model weights."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


class TokenizerKit:
    """Wraps a Hugging Face tokenizer: encode/decode and official chat-template rendering."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str | Path) -> TokenizerKit:
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(str(path)))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(ids), skip_special_tokens=False))

    def render_chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        **template_kwargs: Any,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **template_kwargs,
        )
        return str(text)

    def chat_token_ids(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> list[int]:
        return self.encode(self.render_chat(messages, **kwargs))

    def stop_strings(self) -> list[str]:
        """Special-token strings that end an assistant turn (EOS and friends)."""
        tok = self.tokenizer
        out = [str(t) for t in (getattr(tok, "eos_token", None),) if t]
        out += [str(t) for t in getattr(tok, "all_special_tokens", []) if str(t) not in out]
        return out
