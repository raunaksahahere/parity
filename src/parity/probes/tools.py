"""Family 6: tool-call and structured-output conformance.

Paired design: the same greedy prompts are sent to the target and run through the reference,
so tests are exact binomial tests on the pairs.

* Tool trigger agreement: whether the model calls a tool (24 prompts, half should trigger).
  Disagreements with the reference are tested against a benign disagreement rate.
* Tool-call validity: called name is one of the provided tools, arguments are valid JSON and
  satisfy the declared parameter schema.
* Structured output: the target is sent ``response_format=json_schema``; the reference
  generates unconstrained. Target outputs that are invalid where the reference's are valid
  are tested with a one-sided sign test on discordant pairs.

Reference tool calls are parsed from the Hermes/Qwen ``<tool_call>`` markup (v0.1 supports
that format only). Without a reference, only the absolute validity checks run and the result
is marked lower power.
"""

from __future__ import annotations

import json
import re
from typing import Any

from scipy import stats

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.suites import load_json

DEFAULT_DISAGREEMENT = 0.10
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
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


def parse_reference_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for m in TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)
    return calls


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


class ToolsProbe(Probe):
    id = "tools.structured"
    family = "tools"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        opts = ctx.options
        spec = load_json(opts.get("suite", "core-v1"), "tools.json")
        tools = spec["tools"]
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        tkw = dict(opts.get("template_kwargs", {"enable_thinking": False}))
        d0 = float(opts.get("thresholds", {}).get("tool_disagreement", DEFAULT_DISAGREEMENT))
        prompts = [(p, True) for p in spec["trigger_prompts"]] + [
            (p, False) for p in spec["plain_prompts"]
        ]
        n_requests = 0
        findings: list[Finding] = []
        notes: list[str] = []

        tgt_trigger: list[bool] = []
        ref_trigger: list[bool] = []
        calls_total = bad_name = bad_args = raw_markup = 0
        for text, _ in prompts:
            msgs = [{"role": "user", "content": text}]
            try:
                n_requests += 1
                data = await ctx.client.chat(
                    ctx.model,
                    msgs,
                    tools=tools,
                    max_tokens=160,
                    temperature=0.0,
                    chat_template_kwargs=tkw,
                )
            except EndpointError as exc:
                return self.result(
                    Status.INCONCLUSIVE,
                    n_requests=n_requests,
                    findings=[Finding(severity="warn", message=f"tool request failed: {exc}")],
                )
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            tcs = msg.get("tool_calls") or []
            if "<tool_call>" in (msg.get("content") or ""):
                raw_markup += 1
            tgt_trigger.append(bool(tcs))
            for tc in tcs:
                calls_total += 1
                fn = tc.get("function", {})
                if fn.get("name") not in schemas:
                    bad_name += 1
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "")
                except json.JSONDecodeError:
                    bad_args += 1
                    continue
                if not schema_ok(args, schemas[fn["name"]]):
                    bad_args += 1
            if ctx.reference is not None:
                ids = ctx.kit.chat_token_ids(msgs, tools=tools, **tkw)
                ref_text = ctx.kit.decode(ctx.reference.greedy(ids, 160))
                ref_trigger.append(bool(parse_reference_calls(ref_text)))

        metrics: dict[str, float] = {
            "tool_calls": float(calls_total),
            "invalid_tool_name": float(bad_name),
            "invalid_arguments": float(bad_args),
            "raw_tool_markup_in_content": float(raw_markup),
            "target_trigger_rate": sum(tgt_trigger) / len(tgt_trigger),
        }
        p_values: list[float] = []
        hard: list[str] = []
        if raw_markup:
            hard.append(
                f"{raw_markup} responses contain raw <tool_call> markup instead of structured "
                "tool_calls (tool parsing is not enabled or the template differs)"
            )
        if bad_name:
            hard.append(f"{bad_name} of {calls_total} tool calls name a tool that was not provided")
        if bad_args:
            hard.append(
                f"{bad_args} of {calls_total} tool calls have invalid or non-conforming arguments"
            )

        effect = 0.0
        if ctx.reference is not None:
            n = len(prompts)
            disagree = sum(a != b for a, b in zip(tgt_trigger, ref_trigger, strict=True))
            tp = sum(a and b for a, b in zip(tgt_trigger, ref_trigger, strict=True))
            fp = sum(a and not b for a, b in zip(tgt_trigger, ref_trigger, strict=True))
            fn_ = sum(b and not a for a, b in zip(tgt_trigger, ref_trigger, strict=True))
            f1 = 2 * tp / (2 * tp + fp + fn_) if (2 * tp + fp + fn_) else 1.0
            p_trig = float(stats.binom.sf(disagree - 1, n, d0)) if disagree else 1.0
            p_values.append(p_trig)
            effect = disagree / n
            metrics.update(
                trigger_f1=f1,
                trigger_disagreements=float(disagree),
                reference_trigger_rate=sum(ref_trigger) / n,
                min_detectable_disagreements=float(min_detectable(n, d0, ctx.alpha / 2)),
            )
            if p_trig < ctx.alpha / 2:
                findings.append(
                    Finding(
                        severity="deviation",
                        message=f"tool-trigger decisions differ from the reference on "
                        f"{disagree}/{n} "
                        f"prompts (F1 {f1:.2f}; benign rate {d0:g}; p={p_trig:.3g})",
                    )
                )
        else:
            notes.append("no reference: only absolute validity checks ran (lower power)")

        # structured output: target constrained vs reference unconstrained, paired
        schema = spec["json_schema"]
        ok_t: list[bool] = []
        ok_r: list[bool] = []
        for text in spec["json_texts"]:
            content = spec["json_instruction"] + text
            msgs = [{"role": "user", "content": content}]
            try:
                n_requests += 1
                data = await ctx.client.chat(
                    ctx.model,
                    msgs,
                    max_tokens=120,
                    temperature=0.0,
                    chat_template_kwargs=tkw,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "person", "schema": schema, "strict": True},
                    },
                )
            except EndpointError as exc:
                findings.append(Finding(severity="warn", message=f"json request failed: {exc}"))
                continue
            out = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            val = parse_json_text(out)
            ok_t.append(val is not None and schema_ok(val, schema))
            if ctx.reference is not None:
                ids = ctx.kit.chat_token_ids(msgs, **tkw)
                ref_out = ctx.kit.decode(ctx.reference.greedy(ids, 120))
                rv = parse_json_text(re.sub(r"<\|im_end\|>.*$", "", ref_out, flags=re.DOTALL))
                ok_r.append(rv is not None and schema_ok(rv, schema))
        if ok_t:
            metrics["json_schema_adherence"] = sum(ok_t) / len(ok_t)
        if ok_t and ok_r:
            b = sum(r and not t for t, r in zip(ok_t, ok_r, strict=True))
            c = sum(t and not r for t, r in zip(ok_t, ok_r, strict=True))
            metrics["json_reference_adherence"] = sum(ok_r) / len(ok_r)
            p_json = float(stats.binom.sf(b - 1, b + c, 0.5)) if b + c and b else 1.0
            p_values.append(p_json)
            if p_json < ctx.alpha / 2:
                findings.append(
                    Finding(
                        severity="deviation",
                        message=f"structured output worse than the reference: target invalid where "
                        f"reference valid in {b} prompts vs {c} the other way (p={p_json:.3g})",
                    )
                )
        elif ok_t and not ok_r and sum(ok_t) < len(ok_t):
            hard.append(
                f"{len(ok_t) - sum(ok_t)} of {len(ok_t)} structured outputs violate the requested "
                "schema (response_format may be ignored)"
            )

        if hard:
            findings.append(Finding(severity="deviation", message="; ".join(hard)))
        deviating = any(f.severity == "deviation" for f in findings)
        p_final = min(1.0, min(p_values) * 2) if p_values else None
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
            power="normal" if ctx.reference is not None else "lower",
            metrics=metrics,
            findings=findings,
            notes=[
                *notes,
                "24 trigger prompts and 8 JSON prompts give limited power; "
                "see min_detectable_disagreements",
            ],
        )
