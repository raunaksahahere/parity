from __future__ import annotations

import json
from typing import Any

import httpx

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.probes.base import ProbeContext, Status
from parity.probes.template import Observation, TemplateProbe, diagnose
from parity.tokenization import TokenizerKit


def make_server(kit: TokenizerKit, mutate: Any = None) -> httpx.MockTransport:
    """A fake server that renders with the official template, optionally corrupted."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        msgs, tools = body["messages"], body.get("tools")
        if mutate:
            msgs, tools = mutate(msgs, tools)
        n = len(kit.chat_token_ids(msgs, tools=tools))
        return httpx.Response(
            200,
            json={
                "usage": {
                    "prompt_tokens": n
                    + (mutate.extra if mutate and hasattr(mutate, "extra") else 0)
                }
            },
        )

    return httpx.MockTransport(handler)


async def run_probe(kit: TokenizerKit, transport: httpx.MockTransport, **options: Any):  # type: ignore[no-untyped-def]
    async with EndpointClient("http://x", transport=transport) as c:
        ctx = ProbeContext(
            client=c, adapter=get_adapter("llamacpp"), model="m", kit=kit, options=options
        )
        return await TemplateProbe().run(ctx)


async def test_identical_template_passes(toy_kit: TokenizerKit) -> None:
    res = await run_probe(toy_kit, make_server(toy_kit))
    assert res.status is Status.PASS
    assert res.metrics["mismatches"] == 0
    assert res.n_requests == 7


async def test_dropped_system_prompt_is_detected_and_named(toy_kit: TokenizerKit) -> None:
    def drop_system(msgs: list[dict[str, Any]], tools: Any) -> tuple[list[dict[str, Any]], Any]:
        return [m for m in msgs if m["role"] != "system"], tools

    res = await run_probe(toy_kit, make_server(toy_kit, drop_system))
    assert res.status is Status.DEVIATION and res.hard_evidence
    assert "system prompt is dropped" in res.findings[-1].message


async def test_duplicated_bos_is_a_constant_plus_one(toy_kit: TokenizerKit) -> None:
    def ident(msgs: Any, tools: Any) -> tuple[Any, Any]:
        return msgs, tools

    ident.extra = 1  # type: ignore[attr-defined]
    res = await run_probe(toy_kit, make_server(toy_kit, ident))
    assert res.status is Status.DEVIATION
    assert "duplicated BOS" in res.findings[-1].message


async def test_constant_offset_can_be_tolerated_by_configuration(toy_kit: TokenizerKit) -> None:
    def ident(msgs: Any, tools: Any) -> tuple[Any, Any]:
        return msgs, tools

    ident.extra = 3  # type: ignore[attr-defined]
    res = await run_probe(toy_kit, make_server(toy_kit, ident), tolerate_constant_offset=True)
    assert res.status is Status.PASS


async def test_tools_formatted_differently_is_named(toy_kit: TokenizerKit) -> None:
    def drop_tools(msgs: Any, tools: Any) -> tuple[Any, Any]:
        return msgs, None

    res = await run_probe(toy_kit, make_server(toy_kit, drop_tools))
    assert res.status is Status.DEVIATION
    assert "tool definitions are not rendered" in res.findings[-1].message


async def test_missing_usage_is_inconclusive_not_a_pass(toy_kit: TokenizerKit) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    res = await run_probe(toy_kit, transport)
    assert res.status is Status.INCONCLUSIVE


async def test_server_errors_are_inconclusive(toy_kit: TokenizerKit) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(400, text="nope"))
    res = await run_probe(toy_kit, transport)
    assert res.status is Status.INCONCLUSIVE


def test_diagnose_is_empty_when_all_match() -> None:
    obs = [Observation("a", "plain", 10, 10, {}), Observation("b", "tools", 30, 30, {})]
    assert diagnose(obs) == []


async def test_system_merged_into_user_turn_is_named(toy_kit: TokenizerKit) -> None:
    def merge(msgs: list[dict[str, Any]], tools: Any) -> tuple[list[dict[str, Any]], Any]:
        if msgs[0]["role"] == "system" and len(msgs) > 1 and msgs[1]["role"] == "user":
            merged = {"role": "user", "content": f"{msgs[0]['content']}\n\n{msgs[1]['content']}"}
            return [merged, *msgs[2:]], tools
        return msgs, tools

    res = await run_probe(toy_kit, make_server(toy_kit, merge))
    assert res.status is Status.DEVIATION
    assert "merged into the first user turn" in res.findings[-1].message
