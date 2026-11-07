"""Family 1: chat-template and tokenizer conformance.

Each crafted conversation is rendered locally with the official chat template and
tokenized; the count is compared with the ``usage.prompt_tokens`` the server
reports for the same conversation. Token counts are deterministic, so any mismatch
is hard evidence that the server renders a different prompt. The mismatch pattern
is then classified (duplicated BOS, dropped system prompt, tool formatting, ...).

Limit: equal counts do not prove identical prompts (a template could differ in
content but not length). The first-token distribution check in the distributional
family covers that case when log-probs are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.suites import DEFAULT_SUITE, load_json


@dataclass(frozen=True)
class Observation:
    id: str
    kind: str
    local: int
    server: int | None
    # Count change each hypothesized corruption would cause, computed with the local
    # template: {"drop_system": -9, "drop_tools": -250, ...}.
    hypotheses: dict[str, int]

    @property
    def delta(self) -> int | None:
        return None if self.server is None else self.server - self.local


HYPOTHESIS_TEXT = {
    "drop_system": "the system prompt is dropped",
    "drop_tools": "tool definitions are not rendered",
    "drop_history": "earlier turns are dropped (only the last user message is kept)",
    "merge_system": "the system text is merged into the first user turn (no system role)",
}


def diagnose(obs: list[Observation]) -> list[str]:
    """Explain a pattern of count mismatches. Pure function of the observations."""
    seen = [o for o in obs if o.delta is not None]
    bad = [o for o in seen if o.delta != 0]
    if not bad:
        return []
    deltas = {o.delta for o in seen}
    if len(bad) == len(seen) and len(deltas) == 1:
        c = next(iter(deltas))
        assert c is not None
        hint = " (consistent with a duplicated BOS/special token)" if c == 1 else ""
        return [f"every conversation is off by the same {c:+d} tokens{hint}"]
    out: list[str] = []
    explained: set[str] = set()
    for o in bad:
        for name, expected in o.hypotheses.items():
            if o.delta == expected:
                out.append(f"{o.id}: count matches the case where {HYPOTHESIS_TEXT[name]}")
                explained.add(o.id)
    for o in bad:
        if o.id not in explained:
            out.append(f"{o.id}: off by {o.delta:+d} tokens with no matching known pattern")
    if all(o.kind == "tools" for o in bad):
        out.append(
            "only tool-bearing conversations differ: tool definitions are formatted differently"
        )
    return out


def tools_change_render(kit: Any, msgs: list[dict[str, Any]], tools: Any, **kw: Any) -> bool:
    """False when the official chat template ignores tool definitions (no tool format)."""
    try:
        return bool(kit.render_chat(msgs, tools=tools, **kw) != kit.render_chat(msgs, **kw))
    except Exception:
        return False


def rejects_tools(exc: EndpointError) -> bool:
    """The endpoint refused a request because of its tool definitions (unsupported feature)."""
    return "tool" in str(exc).lower() and (exc.status or 0) >= 400


class TemplateProbe(Probe):
    id = "template.token_counts"
    family = "template"
    version = "2"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        suite = ctx.options.get("suite", DEFAULT_SUITE)
        spec = load_json(suite, "template.json")
        tools = spec["tools"]
        tolerate_constant = bool(ctx.options.get("tolerate_constant_offset", False))
        template_kwargs: dict[str, Any] = ctx.options.get("template_kwargs", {})
        observations: list[Observation] = []
        findings: list[Finding] = []
        notes: list[str] = []
        n_requests = 0

        for conv in spec["conversations"]:
            msgs = conv["messages"]
            use_tools = bool(conv.get("use_tools"))
            if use_tools and not tools_change_render(ctx.kit, msgs, tools, **template_kwargs):
                notes.append(
                    f"{conv['id']}: skipped, the official chat template has no tool format, so "
                    "there is no reference rendering of tool definitions to compare"
                )
                continue
            tools_arg = tools if use_tools else None
            local = len(ctx.kit.chat_token_ids(msgs, tools=tools_arg, **template_kwargs))
            hypotheses: dict[str, int] = {}

            def count(
                m: list[dict[str, Any]], t: Any, _kw: dict[str, Any] = template_kwargs
            ) -> int:
                return len(ctx.kit.chat_token_ids(m, tools=t, **_kw))

            if any(m["role"] == "system" for m in msgs):
                hypotheses["drop_system"] = (
                    count([m for m in msgs if m["role"] != "system"], tools_arg) - local
                )
            if msgs[0]["role"] == "system" and len(msgs) > 1 and msgs[1]["role"] == "user":
                merged = [
                    {"role": "user", "content": f"{msgs[0]['content']}\n\n{msgs[1]['content']}"},
                    *msgs[2:],
                ]
                hypotheses["merge_system"] = count(merged, tools_arg) - local
            if use_tools:
                hypotheses["drop_tools"] = count(msgs, None) - local
            if conv["kind"] == "multi":
                keep = [m for m in msgs if m["role"] == "system"] + [msgs[-1]]
                hypotheses["drop_history"] = count(keep, tools_arg) - local
            params: dict[str, Any] = {"max_tokens": 1, "temperature": 0.0}
            if use_tools:
                params["tools"] = tools
            server: int | None = None
            try:
                n_requests += 1
                data = await ctx.chat(msgs, **params)
                server = (data.get("usage") or {}).get("prompt_tokens")
            except EndpointError as exc:
                if use_tools and rejects_tools(exc):
                    notes.append(f"{conv['id']}: skipped, the endpoint rejects tools: {exc}")
                    continue
                findings.append(
                    Finding(severity="warn", message=f"{conv['id']}: request failed: {exc}")
                )
            if server is None and not any(f.message.startswith(conv["id"]) for f in findings):
                findings.append(
                    Finding(
                        severity="warn",
                        message=f"{conv['id']}: response has no usage.prompt_tokens",
                    )
                )
            observations.append(Observation(conv["id"], conv["kind"], local, server, hypotheses))

        table = {
            o.id: {"local": o.local, "server": o.server, "delta": o.delta} for o in observations
        }
        measured = [o for o in observations if o.delta is not None]
        metrics = {
            "conversations": float(len(observations)),
            "measured": float(len(measured)),
            "mismatches": float(sum(1 for o in measured if o.delta != 0)),
        }
        if not measured:
            return self.result(
                Status.INCONCLUSIVE,
                n_requests=n_requests,
                metrics=metrics,
                notes=notes,
                findings=[
                    *findings,
                    Finding(
                        severity="warn", message="no usable usage.prompt_tokens from the target"
                    ),
                ],
            )

        deltas = {o.delta for o in measured}
        if tolerate_constant and len(deltas) == 1 and next(iter(deltas)) != 0:
            findings.append(
                Finding(
                    severity="info",
                    message=(
                        f"constant offset of {next(iter(deltas)):+d} tokens "
                        "tolerated by configuration"
                    ),
                    detail={"table": table},
                )
            )
            return self.result(
                Status.PASS,
                hard_evidence=False,
                n_requests=n_requests,
                metrics=metrics,
                notes=notes,
                findings=findings,
            )

        reasons = diagnose(observations)
        if reasons:
            findings.append(
                Finding(
                    severity="deviation",
                    message="; ".join(reasons),
                    detail={"table": table},
                )
            )
            return self.result(
                Status.DEVIATION,
                hard_evidence=True,
                effect=float(max(abs(o.delta or 0) for o in measured)),
                effect_unit="max |token count difference|",
                n_requests=n_requests,
                metrics=metrics,
                notes=notes,
                findings=findings,
            )
        findings.append(
            Finding(
                severity="info",
                message="all token counts match the local render",
                detail={"table": table},
            )
        )
        return self.result(
            Status.PASS,
            effect=0.0,
            effect_unit="max |token count difference|",
            n_requests=n_requests,
            metrics=metrics,
            notes=notes,
            findings=findings,
        )
