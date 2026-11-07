"""Family 6: tool-call and structured-output conformance (version 2).

Paired design: the same greedy prompts are sent to the target and run through the reference.

Statistical evidence (Bonferroni across the three tests):

* **Trigger agreement** (``tool_choice`` auto): does the target call a tool where the
  reference does? Disagreements are tested against the calibrated benign disagreement rate
  (exact binomial test). Detects, for example, dropped tool definitions.
* **Call validity**: on prompts where both sides call a tool, a call is valid when its name was
  provided and its arguments are JSON satisfying the declared schema. A one-sided sign test on
  the discordant pairs (target invalid where the reference is valid vs the reverse) detects
  corrupted names or arguments without blaming the model for its own invalid calls.
* **Structured output**: the target is asked for ``response_format=json_schema`` (strict); the
  reference generates unconstrained. Sign test on discordant pairs.

Deterministic evidence (a conforming server cannot produce these whatever the model does):

* structured ``tool_calls`` returned for a request with ``tool_choice="none"``;
* a strict ``json_schema`` reply that finished normally (``finish_reason="stop"``) but is not
  valid JSON for the schema;
* raw ``<tool_call>`` markup in the content of an auto request (tool calls not parsed).

``tool_choice="required"`` is reported as a capability only: llama.cpp b12128 accepts it but
does not enforce it for Qwen3 (docs/decisions.md D15), so a missing forced call is not
evidence of a faulty deployment.

Reference tool calls are parsed from the Hermes/Qwen ``<tool_call>`` markup. Without a
reference only the deterministic checks run and the result is marked lower power.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from scipy import stats

from parity.client import BudgetExhaustedError, EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.suites import DEFAULT_SUITE, load_json

DEFAULT_DISAGREEMENT = 0.10
N_TESTS = 3  # trigger agreement, call validity, structured output
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def schema_ok(value: Any, schema: dict[str, Any]) -> bool:
    """Minimal JSON-schema check: object type, required keys, property types, enums, extras."""
    if schema.get("type") != "object" or not isinstance(value, dict):
        return False
    props = schema.get("properties", {})
    if any(k not in value for k in schema.get("required", [])):
        return False
    if schema.get("additionalProperties") is False and any(k not in props for k in value):
        return False
    for k, v in value.items():
        spec = props.get(k)
        if spec is None:
            continue
        check = TYPE_CHECKS.get(spec.get("type", ""))
        if check is not None and not check(v):
            return False
        if "enum" in spec and v not in spec["enum"]:
            return False
    return True


def parse_reference_calls(text: str) -> list[dict[str, Any] | None]:
    """Calls in Hermes/Qwen markup; ``None`` for a ``<tool_call>`` block that is not JSON."""
    calls: list[dict[str, Any] | None] = []
    for m in TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            calls.append(None)
            continue
        calls.append(obj if isinstance(obj, dict) and "name" in obj else None)
    return calls


def call_valid(name: Any, arguments: Any, schemas: dict[str, dict[str, Any]]) -> bool:
    """A call is valid if it names a provided tool and its arguments satisfy that tool's schema.
    ``arguments`` may be a JSON string (OpenAI wire format) or an already decoded object."""
    if name not in schemas:
        return False
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False
    return schema_ok(arguments, schemas[name])


def parse_json_text(text: str) -> Any | None:
    s = text.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def min_detectable(n: int, p0: float, alpha: float) -> int:
    """Smallest number of disagreements out of ``n`` that the test would call significant."""
    for k in range(n + 1):
        if stats.binom.sf(k - 1, n, p0) < alpha:
            return k
    return n + 1


def sign_test(b: int, c: int) -> float:
    """One-sided exact sign test: P(X >= b) for X ~ Bin(b + c, 1/2)."""
    return float(stats.binom.sf(b - 1, b + c, 0.5)) if b else 1.0


def _message(data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    choice = (data.get("choices") or [{}])[0]
    return choice.get("message") or {}, choice.get("finish_reason")


class ToolsProbe(Probe):
    id = "tools.structured"
    family = "tools"
    version = "2"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        opts = ctx.options
        spec = load_json(opts.get("suite", DEFAULT_SUITE), "tools.json")
        tools = spec["tools"]
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        tkw = dict(opts.get("template_kwargs", {"enable_thinking": False}))
        d0 = float(opts.get("thresholds", {}).get("tool_disagreement", DEFAULT_DISAGREEMENT))
        max_tokens = int(spec.get("max_tokens", 160))
        alpha_test = ctx.alpha / N_TESTS
        has_ref = ctx.reference is not None
        n_requests = 0
        findings: list[Finding] = []
        notes: list[str] = []
        hard: list[str] = []
        metrics: dict[str, float] = {}

        async def chat(msgs: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
            nonlocal n_requests
            try:
                out = await ctx.client.chat(
                    ctx.model,
                    msgs,
                    max_tokens=params.pop("max_tokens", max_tokens),
                    temperature=0.0,
                    chat_template_kwargs=tkw,
                    **params,
                )
            except BudgetExhaustedError:
                raise  # refused by the budget cap: never sent, not counted
            except EndpointError:
                n_requests += 1
                raise
            n_requests += 1
            return out

        async def batch(
            calls: list[tuple[list[dict[str, Any]], dict[str, Any]]],
        ) -> list[dict[str, Any] | EndpointError]:
            """Send independent requests concurrently (the client's semaphore bounds how many
            are in flight); responses come back in request order."""

            async def one(msgs: list[dict[str, Any]], params: dict[str, Any]) -> Any:
                try:
                    return await chat(msgs, **params)
                except EndpointError as exc:
                    return exc

            return list(await asyncio.gather(*(one(m, dict(p)) for m, p in calls)))

        def failed(exc: EndpointError, what: str) -> ProbeResult:
            return self.result(
                Status.INCONCLUSIVE,
                n_requests=n_requests,
                findings=[Finding(severity="warn", message=f"{what} request failed: {exc}")],
            )

        # -- 1. tool_choice auto: trigger agreement and call validity ------------------------
        prompts = list(spec["trigger_prompts"]) + list(spec["plain_prompts"])
        tgt_trigger: list[bool] = []
        ref_trigger: list[bool] = []
        tgt_valid: list[bool | None] = []  # None: no call
        ref_valid: list[bool | None] = []
        calls_total = bad_calls = raw_markup = 0
        auto = await batch([([{"role": "user", "content": t}], {"tools": tools}) for t in prompts])
        for text, data in zip(prompts, auto, strict=True):
            msgs = [{"role": "user", "content": text}]
            if isinstance(data, EndpointError):
                return failed(data, "tool")
            msg, _ = _message(data)
            tcs = msg.get("tool_calls") or []
            if "<tool_call>" in (msg.get("content") or ""):
                raw_markup += 1
            tgt_trigger.append(bool(tcs))
            valid = [
                call_valid((tc.get("function") or {}).get("name"),
                           (tc.get("function") or {}).get("arguments") or "", schemas)
                for tc in tcs
            ]  # fmt: skip
            calls_total += len(valid)
            bad_calls += valid.count(False)
            tgt_valid.append(all(valid) if valid else None)
            if ctx.reference is not None:
                ids = ctx.kit.chat_token_ids(msgs, tools=tools, **tkw)
                ref_text = ctx.kit.decode(ctx.reference.greedy(ids, max_tokens))
                ref_calls = parse_reference_calls(ref_text)
                ref_trigger.append(bool(ref_calls))
                rv = [c is not None and call_valid(c["name"], c.get("arguments"), schemas)
                      for c in ref_calls]  # fmt: skip
                ref_valid.append(all(rv) if rv else None)

        metrics.update(
            tool_calls=float(calls_total),
            invalid_tool_calls=float(bad_calls),
            raw_tool_markup_in_content=float(raw_markup),
            target_trigger_rate=sum(tgt_trigger) / len(tgt_trigger),
        )
        p_values: list[float] = []
        if raw_markup:
            hard.append(
                f"{raw_markup} responses to auto tool requests contain raw <tool_call> markup "
                "instead of structured tool_calls (tool parsing is not enabled or the template "
                "differs)"
            )
        effect = 0.0
        if has_ref:
            n = len(prompts)
            pairs = list(zip(tgt_trigger, ref_trigger, strict=True))
            disagree = sum(a != b for a, b in pairs)
            tp = sum(a and b for a, b in pairs)
            fp = sum(a and not b for a, b in pairs)
            fn_ = sum(b and not a for a, b in pairs)
            f1 = 2 * tp / (2 * tp + fp + fn_) if (2 * tp + fp + fn_) else 1.0
            p_trig = float(stats.binom.sf(disagree - 1, n, d0)) if disagree else 1.0
            p_values.append(p_trig)
            effect = disagree / n
            both = [(t, r) for t, r in zip(tgt_valid, ref_valid, strict=True)
                    if t is not None and r is not None]  # fmt: skip
            b = sum((not t) and r for t, r in both)
            c = sum(t and not r for t, r in both)
            p_valid = sign_test(b, c)
            p_values.append(p_valid)
            metrics.update(
                trigger_f1=f1,
                trigger_disagreements=float(disagree),
                trigger_prompts=float(n),
                reference_trigger_rate=sum(ref_trigger) / n,
                min_detectable_disagreements=float(min_detectable(n, d0, alpha_test)),
                paired_calls=float(len(both)),
                invalid_where_reference_valid=float(b),
                valid_where_reference_invalid=float(c),
            )
            if p_trig < alpha_test:
                findings.append(
                    Finding(
                        severity="deviation",
                        message=f"tool-trigger decisions differ from the reference on "
                        f"{disagree}/{n} prompts (F1 {f1:.2f}; target calls a tool on "
                        f"{sum(tgt_trigger)}, reference on {sum(ref_trigger)}; benign rate "
                        f"{d0:g}; p={p_trig:.3g})",
                    )
                )
            if p_valid < alpha_test:
                findings.append(
                    Finding(
                        severity="deviation",
                        message=f"tool calls are invalid (unknown tool name, malformed or "
                        f"schema-violating arguments) where the reference's are valid on {b} "
                        f"prompts vs {c} the other way (p={p_valid:.3g})",
                    )
                )
        else:
            notes.append("no reference: only the deterministic checks ran (lower power)")
            if bad_calls:
                findings.append(
                    Finding(
                        severity="warn",
                        message=f"{bad_calls} of {calls_total} tool calls are invalid (without a "
                        "reference this cannot be told apart from the model's own mistakes)",
                    )
                )

        # -- 2. tool_choice: "none" must never produce tool calls -----------------------------
        none_violations = 0
        none_prompts = list(spec.get("tool_choice_none_prompts", []))
        for data in await batch(
            [([{"role": "user", "content": t}], {"tools": tools, "tool_choice": "none"})
             for t in none_prompts]
        ):  # fmt: skip
            if isinstance(data, EndpointError):
                findings.append(Finding(severity="warn", message=f"tool_choice=none: {data}"))
                continue
            msg, _ = _message(data)
            none_violations += bool(msg.get("tool_calls"))
        if none_prompts:
            metrics["tool_choice_none_violations"] = float(none_violations)
        if none_violations:
            hard.append(
                f'{none_violations} of {len(none_prompts)} requests with tool_choice="none" '
                "returned tool calls (tool_choice is ignored)"
            )

        # -- 3. tool_choice: "required" (capability, not evidence) ----------------------------
        req_prompts = list(spec.get("tool_choice_required_prompts", []))
        honored = 0
        for data in await batch(
            [([{"role": "user", "content": t}], {"tools": tools, "tool_choice": "required"})
             for t in req_prompts]
        ):  # fmt: skip
            if isinstance(data, EndpointError):
                continue
            msg, _ = _message(data)
            honored += bool(msg.get("tool_calls"))
        if req_prompts:
            metrics["tool_choice_required_honored"] = honored / len(req_prompts)
            if honored < len(req_prompts):
                notes.append(
                    f'tool_choice="required" produced a tool call in {honored}/'
                    f"{len(req_prompts)} requests; reported as a capability, not a deviation "
                    "(some correct servers, e.g. llama.cpp b12128, do not enforce it)"
                )

        # -- 4. structured output -------------------------------------------------------------
        schema = spec["json_schema"]
        ok_t: list[bool] = []
        ok_r: list[bool] = []
        broken_stop = 0
        fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": schema,
                                                       "strict": True}}  # fmt: skip
        json_msgs = [
            [{"role": "user", "content": spec["json_instruction"] + t}] for t in spec["json_texts"]
        ]
        replies = await batch([(m, {"max_tokens": 120, "response_format": fmt}) for m in json_msgs])
        for msgs, data in zip(json_msgs, replies, strict=True):
            if isinstance(data, EndpointError):
                findings.append(Finding(severity="warn", message=f"json request failed: {data}"))
                continue
            msg, finish = _message(data)
            val = parse_json_text(msg.get("content") or "")
            ok = val is not None and schema_ok(val, schema)
            ok_t.append(ok)
            broken_stop += (not ok) and finish == "stop"
            if ctx.reference is not None:
                ids = ctx.kit.chat_token_ids(msgs, **tkw)
                ref_out = ctx.kit.decode(ctx.reference.greedy(ids, 120))
                ref_json = parse_json_text(re.sub(r"<\|im_end\|>.*$", "", ref_out, flags=re.DOTALL))
                ok_r.append(ref_json is not None and schema_ok(ref_json, schema))
        if ok_t:
            metrics["json_schema_adherence"] = sum(ok_t) / len(ok_t)
            metrics["json_invalid_with_stop"] = float(broken_stop)
        if broken_stop:
            hard.append(
                f"{broken_stop} of {len(ok_t)} strict json_schema replies finished normally but "
                "are not valid JSON for the schema (response_format is not enforced)"
            )
        if ok_t and ok_r:
            b = sum(r and not t for t, r in zip(ok_t, ok_r, strict=True))
            c = sum(t and not r for t, r in zip(ok_t, ok_r, strict=True))
            metrics["json_reference_adherence"] = sum(ok_r) / len(ok_r)
            p_json = sign_test(b, c)
            p_values.append(p_json)
            if p_json < alpha_test:
                findings.append(
                    Finding(
                        severity="deviation",
                        message=f"structured output worse than the reference: target invalid "
                        f"where reference valid in {b} prompts vs {c} the other way "
                        f"(p={p_json:.3g})",
                    )
                )

        if hard:
            findings.append(Finding(severity="deviation", message="; ".join(hard)))
        deviating = any(f.severity == "deviation" for f in findings)
        p_final = min(1.0, min(p_values) * N_TESTS) if p_values else None
        status = Status.DEVIATION if deviating else Status.PASS
        if not findings:
            findings.append(
                Finding(severity="info", message="tool and structured-output behaviour matches")
            )
        return self.result(
            status,
            p_value=p_final,
            hard_evidence=bool(hard),
            effect=effect,
            effect_unit="tool-trigger disagreement rate",
            n_requests=n_requests,
            power="normal" if has_ref else "lower",
            metrics=metrics,
            findings=findings,
            notes=[
                *notes,
                f"{len(prompts)} auto prompts, {len(none_prompts)} tool_choice=none prompts, "
                f"{len(spec['json_texts'])} JSON prompts; see min_detectable_disagreements",
            ],
        )
