from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.faults.proxy import (
    Mutation,
    break_argument_schema,
    drop_tools,
    ignore_tool_choice,
    malform_json_content,
    malform_tool_arguments,
    rename_tool_calls,
)
from parity.probes.base import ProbeContext, ProbeResult, Status
from parity.probes.tools import (
    ToolsProbe,
    call_valid,
    min_detectable,
    parse_reference_calls,
    schema_ok,
    sign_test,
)
from parity.suites import DEFAULT_SUITE, load_json
from parity.tokenization import TokenizerKit

SPEC = load_json(DEFAULT_SUITE, "tools.json")
TRIGGERS = set(SPEC["trigger_prompts"])
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in SPEC["tools"]}
PERSON = '{"name": "Ann", "age": 30, "city": "Rome"}'
CALL = '{"name": "get_weather", "arguments": {"city": "Rome"}}'
BAD_CALL = '{"name": "set_timer", "arguments": {"minutes": "ten"}}'


class ScriptedReference:
    """Reference whose 'model' is a lookup on the decoded prompt."""

    def __init__(self, kit: TokenizerKit, invalid_calls: bool = False) -> None:
        self.kit = kit
        self.call = BAD_CALL if invalid_calls else CALL

    def teacher_force(self, ids: list[int], top_k: int) -> dict[str, Any]:
        raise NotImplementedError

    def next_token_logprobs(self, prefix_ids: list[int]) -> Any:
        raise NotImplementedError

    def greedy(self, prefix_ids: list[int], n_tokens: int) -> list[int]:
        text = self.kit.decode(prefix_ids)
        if "Extract the person" in text:
            reply = PERSON
        elif any(t in text for t in TRIGGERS):
            reply = f"<tool_call>\n{self.call}\n</tool_call>"
        else:
            reply = "Sure."
        return self.kit.encode(reply)


def llamacpp_like(mode: str = "honest", mutation: Mutation | None = None) -> httpx.MockTransport:
    """Behaves like llama.cpp b12128 serving Qwen3: tools parsed under auto, tool_choice
    "none" leaves the model's markup in the content, "required" is not enforced."""

    def answer(body: dict[str, Any]) -> dict[str, Any]:
        prompt = body["messages"][-1]["content"]
        if "response_format" in body:
            if mode == "ignore_format":
                return {
                    "message": {"content": "The person is Ann, 30, Rome."},
                    "finish_reason": "stop",
                }
            if mode == "truncated_json":
                return {"message": {"content": PERSON[:12]}, "finish_reason": "length"}
            return {"message": {"content": PERSON}, "finish_reason": "stop"}
        if body.get("tools") and prompt in TRIGGERS:
            call = json.loads(BAD_CALL if mode == "model_invalid" else CALL)
            markup = f"<tool_call>\n{json.dumps(call)}\n</tool_call>"
            if body.get("tool_choice") == "none" or mode == "raw_markup":
                return {"message": {"content": markup}, "finish_reason": "stop"}
            return {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": call["name"],
                                      "arguments": json.dumps(call["arguments"])}}
                    ],
                },
                "finish_reason": "tool_calls",
            }  # fmt: skip
        return {"message": {"content": "Sure."}, "finish_reason": "stop"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if mutation is not None:
            body = mutation(request.url.path, body)
        resp = {"choices": [answer(body)], "usage": {}}
        if mutation is not None and mutation.response is not None:
            resp = mutation.response(request.url.path, body, resp)
        return httpx.Response(200, json=resp)

    return httpx.MockTransport(handler)


async def run(
    kit: TokenizerKit,
    mode: str = "honest",
    mutation: Mutation | None = None,
    reference: ScriptedReference | bool | None = True,
) -> ProbeResult:
    ref = ScriptedReference(kit) if reference is True else reference or None
    async with EndpointClient("http://x", transport=llamacpp_like(mode, mutation)) as c:
        ctx = ProbeContext(
            client=c,
            adapter=get_adapter("llamacpp"),
            model="m",
            kit=kit,
            reference=ref,
            alpha=0.05 / 4,  # the probe's share in the core suite
            options={"template_kwargs": {}},
        )
        return await ToolsProbe().run(ctx)


def messages(res: ProbeResult) -> list[str]:
    return [f.message for f in res.findings]


async def test_honest_llamacpp_like_target_passes(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit)
    assert res.status is Status.PASS, messages(res)
    assert res.metrics["trigger_f1"] == 1.0 and res.metrics["json_schema_adherence"] == 1.0
    assert res.metrics["tool_choice_none_violations"] == 0
    # not enforcing tool_choice="required" is reported, never held against the deployment
    assert res.metrics["tool_choice_required_honored"] == 0.0
    assert any("capability, not a deviation" in n for n in res.notes)


@pytest.mark.parametrize(
    ("mutation", "needle", "hard"),
    [
        (drop_tools(), "tool-trigger decisions differ", False),
        (ignore_tool_choice(), 'tool_choice="none" returned tool calls', True),
        (rename_tool_calls(), "tool calls are invalid", False),
        (malform_tool_arguments(), "tool calls are invalid", False),
        (break_argument_schema(), "tool calls are invalid", False),
        (malform_json_content(), "response_format is not enforced", True),
    ],
    ids=lambda v: v.label if isinstance(v, Mutation) else None,
)
async def test_each_simulated_tool_fault_is_detected(
    toy_kit: TokenizerKit, mutation: Mutation, needle: str, hard: bool
) -> None:
    res = await run(toy_kit, mutation=mutation)
    assert res.status is Status.DEVIATION, messages(res)
    assert any(needle in m for m in messages(res)), messages(res)
    assert res.hard_evidence is hard
    if not hard:
        assert res.p_value is not None and res.p_value < 0.05 / 4  # survives Holm in the suite


async def test_unparsed_tool_markup_is_deterministic_evidence(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, "raw_markup")
    assert res.status is Status.DEVIATION and res.hard_evidence
    assert any("raw <tool_call> markup" in m for m in messages(res))


async def test_ignored_response_format_is_detected(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, "ignore_format")
    assert res.status is Status.DEVIATION and res.hard_evidence
    assert res.metrics["json_schema_adherence"] == 0.0


async def test_json_cut_off_by_max_tokens_is_not_deterministic_evidence(
    toy_kit: TokenizerKit,
) -> None:
    res = await run(toy_kit, "truncated_json")
    assert res.metrics["json_invalid_with_stop"] == 0
    assert not res.hard_evidence  # the paired test still reports it, with a p-value


async def test_the_models_own_invalid_calls_are_not_blamed_on_the_server(
    toy_kit: TokenizerKit,
) -> None:
    """If the reference model makes the same invalid call, the pair is concordant."""
    res = await run(
        toy_kit, "model_invalid", reference=ScriptedReference(toy_kit, invalid_calls=True)
    )
    assert res.status is Status.PASS, messages(res)
    assert res.metrics["invalid_tool_calls"] == len(TRIGGERS)
    assert res.metrics["invalid_where_reference_valid"] == 0


async def test_without_reference_only_deterministic_checks_and_lower_power(
    toy_kit: TokenizerKit,
) -> None:
    res = await run(toy_kit, "ignore_format", reference=None)
    assert res.power == "lower" and res.status is Status.DEVIATION
    ok = await run(toy_kit, "honest", reference=None)
    assert ok.status is Status.PASS and ok.power == "lower"
    bad = await run(toy_kit, mutation=rename_tool_calls(), reference=None)
    assert bad.status is Status.PASS  # cannot be told from the model's own mistakes
    assert any("cannot be told apart" in m for m in messages(bad))


def test_prompt_set_is_balanced_and_versioned() -> None:
    assert SPEC["version"] == DEFAULT_SUITE
    assert len(SPEC["trigger_prompts"]) == len(SPEC["plain_prompts"]) == 48
    assert set(SPEC["tool_choice_none_prompts"]) <= TRIGGERS
    assert not set(SPEC["tool_choice_required_prompts"]) & TRIGGERS
    assert len(SPEC["json_texts"]) == 24


def test_schema_validator() -> None:
    schema = SPEC["json_schema"]
    assert schema_ok({"name": "a", "age": 3, "city": "b"}, schema)
    assert not schema_ok({"name": "a", "age": "3", "city": "b"}, schema)
    assert not schema_ok({"name": "a", "age": True, "city": "b"}, schema)  # bool is not an integer
    assert not schema_ok({"name": "a", "age": 3}, schema)
    assert not schema_ok({"name": "a", "age": 3, "city": "b", "x": 1}, schema)


def test_call_validity() -> None:
    assert call_valid("get_weather", '{"city": "Rome"}', SCHEMAS)
    assert call_valid("get_weather", {"city": "Rome"}, SCHEMAS)
    assert not call_valid("get_weather_v2", '{"city": "Rome"}', SCHEMAS)
    assert not call_valid("get_weather", '{"city": "Rome"', SCHEMAS)
    assert not call_valid("get_weather", '{"city_value": "Rome"}', SCHEMAS)


def test_reference_markup_parser_keeps_malformed_calls_as_invalid() -> None:
    text = f"<tool_call>\n{CALL}\n</tool_call> and <tool_call>{{not json}}</tool_call>"
    calls = parse_reference_calls(text)
    assert calls[0] is not None and calls[0]["name"] == "get_weather" and calls[1] is None


def test_sign_test_and_min_detectable() -> None:
    assert sign_test(0, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(0.5**10)
    assert min_detectable(24, 0.1, 0.01) >= min_detectable(24, 0.1, 0.05)


async def test_requests_are_concurrent_but_bounded_by_the_client(toy_kit: TokenizerKit) -> None:
    """140 independent requests must not run one at a time (they dominate study run time), and
    must never exceed the client's concurrency limit."""
    import asyncio

    inner = llamacpp_like()
    state = {"now": 0, "peak": 0}

    class Slow(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.002)
            state["now"] -= 1
            resp = inner.handle_request(request)
            await resp.aread()
            return httpx.Response(resp.status_code, content=resp.content, headers=resp.headers)

    async with EndpointClient("http://x", transport=Slow(), max_concurrency=4) as c:
        ctx = ProbeContext(
            client=c,
            adapter=get_adapter("llamacpp"),
            model="m",
            kit=toy_kit,
            reference=ScriptedReference(toy_kit),
            options={"template_kwargs": {}},
        )
        res = await ToolsProbe().run(ctx)
    assert res.status is Status.PASS
    assert state["peak"] == 4
