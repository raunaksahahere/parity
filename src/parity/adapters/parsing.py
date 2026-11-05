"""Parsers that turn provider log-prob payloads into normalized positions."""

from __future__ import annotations

from typing import Any

from parity.schema import PositionLogprobs, TopEntry


def _entry(d: dict[str, Any]) -> TopEntry:
    return TopEntry(
        token_id=d.get("id"),
        token=str(d.get("token", "")),
        logprob=float(d["logprob"]),
        token_bytes=d.get("bytes"),
    )


def parse_content_style(logprobs: dict[str, Any]) -> list[PositionLogprobs]:
    """``{"content": [{"token", "logprob", "id"?, "top_logprobs": [...]}, ...]}``.

    This is the chat-completions shape; llama.cpp also uses it on ``/v1/completions``.
    """
    out: list[PositionLogprobs] = []
    for item in logprobs.get("content") or []:
        top = [_entry(t) for t in item.get("top_logprobs") or []]
        out.append(
            PositionLogprobs(
                token_id=item.get("id"),
                token=str(item.get("token", "")),
                logprob=float(item["logprob"]) if item.get("logprob") is not None else None,
                top=top,
            )
        )
    return out


def parse_legacy_style(logprobs: dict[str, Any]) -> list[PositionLogprobs]:
    """Classic OpenAI completions shape: parallel ``tokens``/``token_logprobs`` arrays.

    With ``echo=true`` the prompt tokens are included; the first prompt token has a
    ``None`` log-prob because nothing precedes it.
    """
    tokens: list[str] = logprobs.get("tokens") or []
    lps: list[float | None] = logprobs.get("token_logprobs") or []
    tops: list[dict[str, float] | None] = logprobs.get("top_logprobs") or [None] * len(tokens)
    out: list[PositionLogprobs] = []
    for tok, lp, top in zip(tokens, lps, tops, strict=False):
        entries = [TopEntry(token=k, logprob=float(v)) for k, v in (top or {}).items()]
        entries.sort(key=lambda e: e.logprob, reverse=True)
        out.append(
            PositionLogprobs(token=tok, logprob=None if lp is None else float(lp), top=entries)
        )
    return out


def parse_logprobs(logprobs: dict[str, Any]) -> list[PositionLogprobs]:
    if "content" in logprobs:
        return parse_content_style(logprobs)
    if "tokens" in logprobs:
        return parse_legacy_style(logprobs)
    return []


def parse_vllm_prompt_logprobs(
    prompt_logprobs: list[dict[str, Any] | None], ids: list[int]
) -> list[PositionLogprobs]:
    """vLLM ``prompt_logprobs``.

    Entry ``i`` maps token-id strings to ``{logprob, rank, decoded_token}`` for the
    distribution that predicted ``ids[i]``; entry 0 is ``None``.

    Returns one normalized position per predicted token ``ids[1:]``.
    """
    out: list[PositionLogprobs] = []
    for i in range(1, len(ids)):
        entry = prompt_logprobs[i] if i < len(prompt_logprobs) else None
        if not entry:
            out.append(PositionLogprobs(token_id=ids[i], token="", logprob=None))
            continue
        top = [
            TopEntry(
                token_id=int(tid),
                token=str(v.get("decoded_token", "")),
                logprob=float(v["logprob"]),
            )
            for tid, v in entry.items()
        ]
        top.sort(key=lambda e: e.logprob, reverse=True)
        forced = entry.get(str(ids[i]))
        out.append(
            PositionLogprobs(
                token_id=ids[i],
                token=str(forced.get("decoded_token", "")) if forced else "",
                logprob=float(forced["logprob"]) if forced else None,
                top=top,
            )
        )
    return out
