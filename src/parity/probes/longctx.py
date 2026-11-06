"""Family 5: long-context integrity.

A unique "needle" sentence is buried at three depths of a filler haystack of increasing
length, followed by a question. Two layers of evidence:

1. Token accounting (deterministic): the locally rendered prompt has an exactly known token
   count. A server that reports fewer ``usage.prompt_tokens``, or rejects a prompt that is
   within the claimed context window, is silently truncating or advertising a wrong window.
2. Retrieval (behavioural): the model must repeat the needle. A needle at the *start* that
   is lost while the *end* needle survives, at a length where the start needle was found
   before, suggests left truncation that accounting did not reveal.

Retrieval alone is weak evidence for a 0.6B model, so a failed retrieval without an
accounting problem never yields DEVIATION on its own; it yields INCONCLUSIVE.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status

DEPTHS = (("start", 0.03), ("middle", 0.5), ("end", 0.97))
WORDS = ["amber", "cobalt", "juniper", "saffron", "indigo", "tundra", "marlin", "quartz"]
SUBJECTS = ["the maintenance crew", "the night shift", "an inspector", "the logistics team"]
ACTIONS = ["checked valve", "logged pump", "reviewed gauge", "recorded sensor", "inspected relay"]
OUTCOMES = ["and found nothing unusual", "and noted normal readings", "with no issues reported"]


def filler_line(i: int, rng: random.Random) -> str:
    return (
        f"Log entry {i:05d}: {rng.choice(SUBJECTS)} {rng.choice(ACTIONS)} "
        f"{rng.randint(1, 99)} {rng.choice(OUTCOMES)}."
    )


@dataclass(frozen=True)
class Case:
    length: int
    depth: str
    messages: list[dict[str, Any]]
    local_tokens: int
    answer: str


def build_case(
    ctx: ProbeContext, length: int, depth: str, frac: float, seed: int, kwargs: dict[str, Any]
) -> Case:
    """Build a prompt whose rendered length is the largest not exceeding ``length`` tokens."""
    rng = random.Random(f"{seed}-{length}-{depth}")
    code = f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
    needle = f"IMPORTANT: The vault passphrase is {code}."
    question = "\n\nWhat is the vault passphrase? Reply with the passphrase only."
    pool = [filler_line(i, rng) for i in range(length // 8 + 8)]  # >= 8 tokens per line

    def render(n_lines: int) -> list[dict[str, Any]]:
        lines = pool[:n_lines]
        lines.insert(int(frac * len(lines)), needle)
        return [{"role": "user", "content": "\n".join(lines) + question}]

    def count(n_lines: int) -> int:
        return len(ctx.kit.chat_token_ids(render(n_lines), **kwargs))

    lo, hi = 1, len(pool)  # largest n with count(n) <= length
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= length:
            lo = mid
        else:
            hi = mid - 1
    return Case(length, depth, render(lo), count(lo), code)


def diagnose_accounting(diffs: dict[str, int]) -> list[str]:
    """Explain token-accounting mismatches (``diffs`` maps "length/depth" to local - server).

    The same difference at every length, including the shortest, is a template or
    tokenization difference rather than truncation; a shortfall that appears or grows with
    length is truncation.
    """
    bad = {k: d for k, d in diffs.items() if d != 0}
    if not bad:
        return []
    values = set(diffs.values())
    if len(values) == 1:
        d = next(iter(values))
        return [
            f"every prompt, at every length, differs from the local render by {d:+d} tokens "
            "(local minus server): a chat-template or tokenization difference, not truncation"
        ]
    out = []
    for key, d in bad.items():
        if d > 0:
            out.append(
                f"{key}: server counted {d} fewer prompt tokens than the prompt has "
                f"(silent truncation of {d})"
            )
        else:
            out.append(f"{key}: server counted {-d} more prompt tokens than the prompt has")
    return out


class LongContextProbe(Probe):
    id = "longctx.needle"
    family = "longctx"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        opts = ctx.options
        claimed = int(opts.get("claimed_context", 4096))
        template_kwargs = dict(opts.get("template_kwargs", {"enable_thinking": False}))
        headroom = int(opts.get("longctx_headroom", 64))  # room for the reply and template slack
        limit = claimed - headroom
        lengths = [
            n for n in opts.get("longctx_lengths", [512, 2048, 4096, 8192, 16384]) if n <= limit
        ]
        if limit > max(lengths, default=0):
            lengths.append(limit)  # always test close to the advertised limit
        findings: list[Finding] = []
        n_requests = 0
        hard: list[str] = []
        table: dict[str, dict[str, Any]] = {}
        diffs: dict[str, int] = {}  # key -> local count minus server count

        for length in lengths:
            for depth, frac in DEPTHS:
                case = build_case(ctx, length, depth, frac, ctx.seed, template_kwargs)
                key = f"{length}/{depth}"
                n_requests += 1
                try:
                    data = await ctx.client.chat(
                        ctx.model,
                        case.messages,
                        max_tokens=24,
                        temperature=0.0,
                        chat_template_kwargs=template_kwargs,
                    )
                except EndpointError as exc:
                    table[key] = {"error": str(exc)[:160], "local": case.local_tokens}
                    hard.append(
                        f"{key}: request of {case.local_tokens} tokens rejected although the "
                        f"claimed window is {claimed}: {str(exc)[:100]}"
                    )
                    continue
                server_tokens = (data.get("usage") or {}).get("prompt_tokens")
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
                    "content"
                ) or ""
                found = case.answer in content
                table[key] = {"local": case.local_tokens, "server": server_tokens, "found": found}
                if server_tokens is not None:
                    diffs[key] = case.local_tokens - server_tokens

        hard.extend(diagnose_accounting(diffs))

        evaluated = [v for v in table.values() if "found" in v]
        found_rate = sum(v["found"] for v in evaluated) / max(1, len(evaluated))
        metrics = {
            "requests": float(n_requests),
            "retrieval_rate": found_rate,
            "max_length_tested": float(max(lengths, default=0)),
            "accounting_problems": float(len(hard)),
        }
        # behavioural hint: start-needle lost where end-needle is found, after earlier success
        hints: list[str] = []
        for length in lengths:
            s, e = table.get(f"{length}/start", {}), table.get(f"{length}/end", {})
            if s.get("found") is False and e.get("found") is True:
                earlier = any(
                    table.get(f"{m}/start", {}).get("found") for m in lengths if m < length
                )
                if earlier:
                    hints.append(f"{length}: start needle lost while end needle found")
        details: dict[str, Any] = {"table": table, "claimed_context": claimed}
        if hard:
            findings.append(Finding(severity="deviation", message="; ".join(hard), detail=details))
            return self.result(
                Status.DEVIATION,
                hard_evidence=True,
                effect=float(len(hard)),
                effect_unit="lengths/depths with accounting problems",
                n_requests=n_requests,
                metrics=metrics,
                findings=findings,
            )
        if hints:
            findings.append(
                Finding(
                    severity="warn",
                    message="possible silent truncation: " + "; ".join(hints),
                    detail=details,
                )
            )
            return self.result(
                Status.INCONCLUSIVE, n_requests=n_requests, metrics=metrics, findings=findings
            )
        findings.append(
            Finding(
                severity="info",
                message=f"token accounting consistent up to {max(lengths, default=0)} tokens; "
                f"needle retrieval rate {found_rate:.2f}",
                detail=details,
            )
        )
        return self.result(
            Status.PASS,
            effect=0.0,
            effect_unit="lengths/depths with accounting problems",
            n_requests=n_requests,
            metrics=metrics,
            findings=findings,
            notes=[
                "retrieval ability of the model itself is not compared with the reference; "
                "only accounting and start-vs-end consistency are used as evidence"
            ],
        )
