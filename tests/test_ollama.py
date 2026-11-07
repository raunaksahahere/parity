"""v0.2.0: Ollama adapter behaviour and probes on endpoints or models without tool support."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx

from parity import Reference, Suite, Target
from parity.adapters import get_adapter
from parity.probes.base import Status
from parity.report_model import ReferenceInfo
from parity.tokenization import TokenizerKit
from tests.conftest import CHATML
from tests.fake_endpoint import PERSON, FakeEndpoint, SuiteReference

FAST: dict[str, Any] = {
    "sampler_looks": [60, 150],
    "claimed_context": 1024,
    "longctx_lengths": [256, 1024],
    "template_kwargs": {},
}
INFO = ReferenceInfo(kind="hf:local", model_id="toy", revision="abc123", dtype="float32")
NO_TOOLS_TEMPLATE = (
    "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)


def test_ollama_translates_thinking_off_into_reasoning_effort() -> None:
    a = get_adapter("ollama")
    out = a.chat_params({"max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}})
    assert out == {"max_tokens": 4, "reasoning_effort": "none"}
    assert a.chat_params({"chat_template_kwargs": {}}) == {}
    assert not a.capabilities.token_id_prompts
    llama = get_adapter("llamacpp")
    params = {"chat_template_kwargs": {"enable_thinking": False}}
    assert llama.chat_params(params) == params  # servers that read the kwargs get them as-is


class OllamaLike(FakeEndpoint):
    """FakeEndpoint answering like Ollama: /api/version, no token-id prompts, tools rejected for
    a model whose library template has no tool format, reasoning_effort recorded."""

    def __init__(self, kit: TokenizerKit, **kw: Any) -> None:
        super().__init__(kit, **kw)
        self.bodies: list[dict[str, Any]] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if request.url.path == "/props":
            return httpx.Response(404)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.33.2"})
        body = json.loads(request.content)
        self.bodies.append(body)
        if request.url.path == "/v1/completions" and isinstance(body.get("prompt"), list):
            return httpx.Response(400, json={"error": {"message": "cannot unmarshal array"}})
        if body.get("tools"):
            return httpx.Response(
                400, json={"error": {"message": "model 'm' does not support tools"}}
            )
        if "response_format" in body:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": PERSON}, "finish_reason": "stop"}]},
            )
        return self._chat(body)


def no_tool_kit(kit: TokenizerKit) -> TokenizerKit:
    tok = copy.deepcopy(kit.tokenizer)
    tok.chat_template = NO_TOOLS_TEMPLATE
    return TokenizerKit(tok)


async def test_ollama_endpoint_runs_the_chat_probes_and_reports_coverage(
    toy_kit: TokenizerKit,
) -> None:
    kit = no_tool_kit(toy_kit)
    ep = OllamaLike(kit, seed=1)
    suite = Suite(options={**FAST, "template_kwargs": {"enable_thinking": False}})
    target = Target("http://h:11434/v1", "m", transport=ep.transport())
    rep = await suite.arun(target, Reference(kit, SuiteReference(kit), INFO))
    assert rep.target.adapter == "ollama"  # detected from /api/version
    st = {o.result.probe: o.final_status for o in rep.probes}
    assert st["distribution.teacher_forced"] is Status.SKIPPED
    assert st["template.token_counts"] is Status.PASS
    assert rep.coverage is not None and set(rep.coverage.skipped) == {
        "distribution.teacher_forced",
        "greedy.agreement",
        "sampler.conformance",
    }
    assert rep.verdict is Status.PASS and rep.lower_power
    tools = next(o.result for o in rep.probes if o.result.probe == "tools.structured")
    assert any("no tool format" in n for n in tools.notes)  # skipped, JSON still tested
    assert tools.metrics["json_schema_adherence"] == 1.0
    # every chat that asked for thinking off carried Ollama's own switch, not the kwargs
    asked = [b for b in ep.bodies if "messages" in b and "reasoning_effort" in b]
    assert asked and not any("chat_template_kwargs" in b for b in ep.bodies if "messages" in b)


async def test_an_endpoint_that_rejects_tools_skips_the_tool_checks(
    toy_kit: TokenizerKit,
) -> None:
    """The model's template has a tool format but the endpoint refuses tools (HTTP 400)."""
    ep = OllamaLike(toy_kit, seed=2)
    suite = Suite(options=FAST, probes=["template.token_counts", "tools.structured"])
    target = Target("http://h:1/v1", "m", adapter="ollama", transport=ep.transport())
    rep = await suite.arun(target, Reference(toy_kit, SuiteReference(toy_kit), INFO))
    tools = next(o.result for o in rep.probes if o.result.probe == "tools.structured")
    assert tools.status is Status.PASS and tools.power == "lower"
    assert any("rejects tool definitions" in n for n in tools.notes)
    template = next(o.result for o in rep.probes if o.result.probe == "template.token_counts")
    assert template.status is Status.PASS
    assert any("rejects tools" in n for n in template.notes)


async def test_llamacpp_generic_tool_prompt_is_not_blamed_on_a_model_without_tool_format(
    toy_kit: TokenizerKit,
) -> None:
    """llama.cpp injects its own tool instructions for templates without a tool format; the
    official template defines none, so the template probe must not compare them."""
    kit = no_tool_kit(toy_kit)

    class InjectsToolPrompt(FakeEndpoint):
        def _chat(self, body: dict[str, Any]) -> httpx.Response:
            resp = super()._chat(body)
            if body.get("tools"):
                data = resp.json()
                data["usage"]["prompt_tokens"] += 40  # a generic tool system prompt
                return httpx.Response(200, json=data)
            return resp

    ep = InjectsToolPrompt(kit, seed=3)
    suite = Suite(options=FAST, probes=["template.token_counts"])
    target = Target("http://h:1/v1", "m", transport=ep.transport())
    rep = await suite.arun(target, Reference(kit, SuiteReference(kit), INFO))
    res = rep.probes[0].result
    assert res.status is Status.PASS, [f.message for f in res.findings]
    assert any("no tool format" in n for n in res.notes)
    assert CHATML != NO_TOOLS_TEMPLATE  # the default toy template does have a tool format
