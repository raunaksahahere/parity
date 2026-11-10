"""Preflight check: what can this endpoint be tested for, before any budget is spent?

`parity doctor` sends a handful of small requests and reports, for every probe, whether it can
run on the endpoint and in which mode. It needs no reference model. The result is a prediction
from observed behaviour; the run itself remains the authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from parity.adapters import get_adapter
from parity.client import EndpointClient, EndpointError
from parity.detect import detect_adapter

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Diagnosis:
    adapter: str
    checks: list[Check] = field(default_factory=list)
    probes: dict[str, str] = field(default_factory=dict)  # probe id -> "full" | mode | reason
    requests: int = 0

    @property
    def identity_checkable(self) -> bool:
        return any(
            not self.probes[p].startswith("cannot run")
            for p in ("distribution.teacher_forced", "greedy.agreement")
        )

    def ok(self, name: str) -> bool:
        return any(c.ok for c in self.checks if c.name == name)


async def diagnose(
    base_url: str, model: str, *, adapter: str = "auto", api_key_env: str | None = None,
    transport: Any = None,
) -> Diagnosis:  # fmt: skip
    async with EndpointClient(
        base_url, api_key_env=api_key_env, max_retries=0, transport=transport
    ) as client:
        name = await detect_adapter(client) if adapter == "auto" else adapter
        ad = get_adapter(name)
        d = Diagnosis(adapter=name)
        msgs = [{"role": "user", "content": "Reply with the single word: ready"}]

        async def attempt(check: str, call: Any, good: Any) -> Any:
            try:
                data = await call()
            except EndpointError as exc:
                d.checks.append(Check(check, False, str(exc)[:140]))
                return None
            except Exception as exc:  # malformed response shapes are a finding, not a crash
                d.checks.append(Check(check, False, f"{type(exc).__name__}: {exc}"[:140]))
                return None
            try:
                ok, detail = good(data)
            except Exception as exc:
                ok, detail = False, f"unexpected response shape: {type(exc).__name__}"
            d.checks.append(Check(check, bool(ok), detail))
            return data

        def chat(**params: Any) -> Any:
            base = {"max_tokens": 8, "temperature": 0.0}
            return lambda: client.chat(model, msgs, **ad.chat_params({**base, **params}))

        await attempt(
            "chat",
            chat(),
            lambda r: (bool(r["choices"][0]["message"]), "chat completions answer"),
        )
        await attempt(
            "usage.prompt_tokens",
            chat(),
            lambda r: (
                isinstance((r.get("usage") or {}).get("prompt_tokens"), int),
                f"prompt_tokens={(r.get('usage') or {}).get('prompt_tokens')}",
            ),
        )
        await attempt(
            "tools",
            chat(tools=[TOOL], max_tokens=48),
            lambda r: (True, "tool definitions accepted"),
        )
        await attempt(
            "response_format json_schema",
            chat(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "p", "schema": SCHEMA, "strict": True},
                },
                max_tokens=24,
            ),
            lambda r: (True, "structured output accepted"),
        )
        await attempt(
            "token-id prompts",
            lambda: client.completions(model, [1, 2, 3], max_tokens=1, temperature=0.0),
            lambda r: (
                (r.get("usage") or {}).get("prompt_tokens") in (3, 4),
                f"prompt_tokens={(r.get('usage') or {}).get('prompt_tokens')} for 3 ids",
            ),
        )
        await attempt(
            "completion log-probs",
            lambda: client.completions(model, "The", max_tokens=1, temperature=0.0, logprobs=5),
            lambda r: (
                bool(r["choices"][0].get("logprobs")),
                "top-k log-probs returned" if r["choices"][0].get("logprobs") else "none returned",
            ),
        )
        await attempt(
            "chat log-probs",
            chat(logprobs=True, top_logprobs=5),
            lambda r: (
                bool((r["choices"][0].get("logprobs") or {}).get("content")),
                "log-probs of generated tokens returned",
            ),
        )
        d.requests = client.requests_sent

    ids, lp = d.ok("token-id prompts"), d.ok("completion log-probs")
    chat_ok, usage = d.ok("chat"), d.ok("usage.prompt_tokens")
    no_chat = "cannot run: chat completions failed"
    d.probes = {
        "template.token_counts": "full"
        if usage
        else "cannot run: no usage.prompt_tokens"
        if chat_ok
        else no_chat,
        "longctx.needle": "full"
        if usage
        else "retrieval only: no usage.prompt_tokens"
        if chat_ok
        else no_chat,
        "tools.structured": (
            "full"
            if d.ok("tools") and d.ok("response_format json_schema")
            else "structured output only: tools rejected"
            if d.ok("response_format json_schema")
            else "tools only: response_format rejected"
            if d.ok("tools")
            else "cannot run: tools and response_format rejected"
        ),
        "greedy.agreement": "full (token-id prompts)"
        if ids
        else "chat mode (text comparison, includes the chat template)"
        if chat_ok
        else no_chat,
        "distribution.teacher_forced": "full"
        if ids and lp
        else "cannot run: needs token-id prompts and top-k log-probs",
        "sampler.conformance": "full"
        if ids and lp
        else "cannot run: needs token-id prompts and top-k log-probs",
    }
    return d
