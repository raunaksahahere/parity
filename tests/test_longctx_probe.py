from __future__ import annotations

import json
import re
from typing import Any

import httpx

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.probes.base import ProbeContext, ProbeResult, Status
from parity.probes.longctx import LongContextProbe, build_case
from parity.tokenization import TokenizerKit

KW = {"enable_thinking": False}


def chat_server(kit: TokenizerKit, ctx_size: int, mode: str) -> httpx.MockTransport:
    """mode: 'honest' (no limit), 'truncate' (silent left truncation), 'reject' (HTTP 400)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        ids = kit.chat_token_ids(body["messages"])
        if len(ids) > ctx_size:
            if mode == "reject":
                return httpx.Response(400, text="prompt is too long for the context window")
            if mode == "truncate":
                ids = ids[-ctx_size:]
        text = kit.decode(ids)
        m = re.search(r"vault passphrase is (\S+?)\.", text)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": m.group(1) if m else "unknown"}}],
                "usage": {"prompt_tokens": len(ids)},
            },
        )

    return httpx.MockTransport(handler)


async def run(kit: TokenizerKit, transport: httpx.MockTransport, **opts: Any) -> ProbeResult:
    async with EndpointClient("http://x", transport=transport) as c:
        ctx = ProbeContext(
            client=c,
            adapter=get_adapter("llamacpp"),
            model="m",
            kit=kit,
            options={"template_kwargs": {}, **opts},
        )
        return await LongContextProbe().run(ctx)


async def test_honest_server_passes(toy_kit: TokenizerKit) -> None:
    res = await run(
        toy_kit,
        chat_server(toy_kit, 10**6, "honest"),
        claimed_context=2048,
        longctx_lengths=[256, 1024, 2048],
    )
    assert res.status is Status.PASS
    assert res.metrics["retrieval_rate"] == 1.0


async def test_silent_truncation_is_hard_evidence(toy_kit: TokenizerKit) -> None:
    res = await run(
        toy_kit,
        chat_server(toy_kit, 600, "truncate"),
        claimed_context=2048,
        longctx_lengths=[256, 1024, 2048],
    )
    assert res.status is Status.DEVIATION and res.hard_evidence
    assert "silent truncation" in res.findings[-1].message


async def test_wrong_claimed_window_rejects_long_prompts(toy_kit: TokenizerKit) -> None:
    res = await run(
        toy_kit,
        chat_server(toy_kit, 600, "reject"),
        claimed_context=2048,
        longctx_lengths=[256, 1024],
    )
    assert res.status is Status.DEVIATION
    assert "claimed window" in res.findings[-1].message


def test_built_prompt_length_is_tight_and_never_exceeds_target(toy_kit: TokenizerKit) -> None:
    from parity.client import EndpointClient as C

    ctx = ProbeContext(
        client=C("http://x"), adapter=get_adapter("llamacpp"), model="m", kit=toy_kit
    )
    for target in (300, 1000):
        case = build_case(ctx, target, "middle", 0.5, 0, {})
        assert target - 60 <= case.local_tokens <= target
        assert case.answer in case.messages[0]["content"]


def test_constant_accounting_offset_is_diagnosed_as_template_not_truncation() -> None:
    from parity.probes.longctx import diagnose_accounting

    msg = diagnose_accounting({"512/start": 4, "512/end": 4, "2048/start": 4})
    assert len(msg) == 1 and "template" in msg[0] and "truncation" in msg[0].split("not ")[-1]
    trunc = diagnose_accounting({"512/start": 0, "2048/start": 600, "4032/start": 2500})
    assert len(trunc) == 2 and all("silent truncation" in m for m in trunc)
    assert diagnose_accounting({"512/start": 0}) == []
