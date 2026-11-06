from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.probes.base import ProbeContext, ProbeResult, Status
from parity.probes.tools import ToolsProbe, min_detectable, parse_reference_calls, schema_ok
from parity.suites import load_json
from parity.tokenization import TokenizerKit

SPEC = load_json("core-v1", "tools.json")
TRIGGERS = set(SPEC["trigger_prompts"])
PERSON = '{"name": "Ann", "age": 30, "city": "Rome"}'
CALL = '{"name": "get_weather", "arguments": {"city": "Rome"}}'


class ScriptedReference:
    """Reference whose 'model' is a lookup on the decoded prompt."""

    def __init__(self, kit: TokenizerKit) -> None:
        self.kit = kit

    def teacher_force(self, ids: list[int], top_k: int) -> dict[str, Any]:
        raise NotImplementedError

    def next_token_logprobs(self, prefix_ids: list[int]) -> Any:
        raise NotImplementedError

    def greedy(self, prefix_ids: list[int], n_tokens: int) -> list[int]:
        text = self.kit.decode(prefix_ids)
        if any(t in text for t in TRIGGERS):
            reply = f"<tool_call>\n{CALL}\n</tool_call>"
        elif "Extract the person" in text:
            reply = PERSON
        else:
            reply = "Sure."
        return self.kit.encode(reply)


def server(mode: str = "honest") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        msg: dict[str, Any] = {"content": "Sure."}
        if "response_format" in body:
            msg = {"content": PERSON if mode != "ignore_format" else "The person is Ann, 30, Rome."}
        elif "tools" in body and prompt in TRIGGERS and mode != "no_tools":
            name = "teleport" if mode == "bad_name" else "get_weather"
            if mode == "raw_markup":
                msg = {"content": f"<tool_call>\n{CALL}\n</tool_call>"}
            else:
                msg = {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": name, "arguments": json.dumps({"city": "Rome"})}}
                    ],
                }
        return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {}})

    return httpx.MockTransport(handler)


async def run(kit: TokenizerKit, mode: str, reference: bool = True) -> ProbeResult:
    async with EndpointClient("http://x", transport=server(mode)) as c:
        ctx = ProbeContext(
            client=c,
            adapter=get_adapter("llamacpp"),
            model="m",
            kit=kit,
            reference=ScriptedReference(kit) if reference else None,
            options={"template_kwargs": {}},
        )
        return await ToolsProbe().run(ctx)


async def test_honest_target_passes(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, "honest")
    assert res.status is Status.PASS, [f.message for f in res.findings]
    assert res.metrics["trigger_f1"] == 1.0 and res.metrics["json_schema_adherence"] == 1.0


@pytest.mark.parametrize(
    ("mode", "needle"),
    [
        ("no_tools", "tool-trigger decisions differ"),
        ("bad_name", "not provided"),
        ("raw_markup", "raw <tool_call> markup"),
    ],
)
async def test_tool_faults_are_detected(toy_kit: TokenizerKit, mode: str, needle: str) -> None:
    res = await run(toy_kit, mode)
    assert res.status is Status.DEVIATION
    assert any(needle in f.message for f in res.findings), [f.message for f in res.findings]


async def test_ignored_response_format_is_detected(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, "ignore_format")
    assert res.status is Status.DEVIATION
    assert res.metrics["json_schema_adherence"] == 0.0
    assert any("structured output" in f.message for f in res.findings)


async def test_without_reference_only_absolute_checks_and_lower_power(
    toy_kit: TokenizerKit,
) -> None:
    res = await run(toy_kit, "ignore_format", reference=False)
    assert res.power == "lower" and res.status is Status.DEVIATION
    ok = await run(toy_kit, "honest", reference=False)
    assert ok.status is Status.PASS and ok.power == "lower"


def test_schema_validator() -> None:
    schema = SPEC["json_schema"]
    assert schema_ok({"name": "a", "age": 3, "city": "b"}, schema)
    assert not schema_ok({"name": "a", "age": "3", "city": "b"}, schema)
    assert not schema_ok({"name": "a", "age": True, "city": "b"}, schema)  # bool is not an integer
    assert not schema_ok({"name": "a", "age": 3}, schema)
    assert not schema_ok({"name": "a", "age": 3, "city": "b", "x": 1}, schema)


def test_reference_markup_parser_ignores_malformed_calls() -> None:
    text = f"<tool_call>\n{CALL}\n</tool_call> and <tool_call>{{not json}}</tool_call>"
    assert [c["name"] for c in parse_reference_calls(text)] == ["get_weather"]


def test_min_detectable_is_monotone_in_alpha() -> None:
    assert min_detectable(24, 0.1, 0.01) >= min_detectable(24, 0.1, 0.05)
