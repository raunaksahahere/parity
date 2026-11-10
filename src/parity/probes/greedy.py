"""Family 3: greedy (temperature-0) agreement with the reference.

Both sides continue the same token-id prompts greedily. Run-to-run numerical noise makes
exact agreement impossible even for a correct deployment (near-tied tokens flip), so the
statistic is a *divergence hazard*: every compared position is a Bernoulli trial for "the
continuations first diverge here", and a prompt stops contributing at its first divergence
(afterwards the contexts differ and later tokens are not comparable). The hazard is tested
sequentially with a Bernoulli SPRT:

    H0: hazard = h0 (benign, from calibration)   vs   H1: hazard = h1 (a real deviation)

This uses the full divergence-position distribution rather than only "identical or not".

**Chat mode.** Endpoints that cannot take token-id prompts (Ollama, most hosted APIs) would
otherwise get no model-identity evidence at all. For them the probe sends plain chat prompts at
temperature 0, renders the same chat with the official template for the reference, and compares
the two replies after tokenizing both texts with the reference tokenizer. The same hazard test
applies, with its own threshold (`greedy_chat_hazard`), because the server's chat template is
now part of what is compared: a different template shows up here as well as in the template
probe.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.stats.sprt import BinomialSPRT, Decision
from parity.suites import DEFAULT_SUITE, load_corpus, load_json

DEFAULT_H0 = 0.005  # per-token benign divergence hazard; replaced by calibration
DEFAULT_H1 = 0.025
DEFAULT_CHAT_H0 = 0.01  # chat mode, uncalibrated: lenient, the template path is included


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first position where ``a`` and ``b`` differ, ``None`` if one is a prefix
    of the other with both ended naturally at the same length."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


class GreedyProbe(Probe):
    id = "greedy.agreement"
    family = "greedy"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        caps = ctx.adapter.capabilities
        if ctx.reference is None:
            return self.result(Status.SKIPPED, notes=["no reference implementation configured"])
        opts = ctx.options
        th = opts.get("thresholds", {})
        chat_mode = not caps.token_id_prompts
        n_tokens = int(opts.get("greedy_tokens", 48))
        tkw = dict(opts.get("template_kwargs", {"enable_thinking": False}))
        prompts: list[Any]
        if chat_mode:
            h0 = float(th.get("greedy_chat_hazard", DEFAULT_CHAT_H0))
            spec = load_json(opts.get("suite", DEFAULT_SUITE), "tools.json")
            prompts = [[{"role": "user", "content": t}] for t in spec["plain_prompts"]]
        else:
            h0 = float(th.get("greedy_hazard", DEFAULT_H0))
            prompt_len = int(opts.get("greedy_prompt_tokens", 16))
            prompts = [
                ids
                for d in load_corpus(opts.get("suite", DEFAULT_SUITE))
                if len(ids := ctx.kit.encode(d.text)[:prompt_len]) >= 4
            ]
        h1 = float(opts.get("greedy_h1", max(DEFAULT_H1, 5 * h0)))

        async def target_and_reference(p: Any) -> tuple[list[int] | None, list[int], bool]:
            """(target ids or None, reference ids, comparable to the end)."""
            assert ctx.reference is not None
            if not chat_mode:
                comp = await ctx.adapter.generate(
                    ctx.client, ctx.model, p, n_tokens=n_tokens, sampling={"temperature": 0.0}
                )
                ids = [pos.token_id for pos in comp.positions]
                tgt = (
                    None if any(t is None for t in ids) else [int(t) for t in ids if t is not None]
                )
                return tgt, ctx.reference.greedy(p, n_tokens), True
            data = await ctx.chat(p, max_tokens=n_tokens, temperature=0.0, chat_template_kwargs=tkw)
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            ref_ids = ctx.reference.greedy(ctx.kit.chat_token_ids(p, **tkw), n_tokens)
            ref_text = ctx.kit.decode(ref_ids)
            for stop in ctx.kit.stop_strings():
                ref_text = ref_text.split(stop)[0]
            ref_done = len(ref_ids) < n_tokens or ref_text != ctx.kit.decode(ref_ids)
            tgt_done = choice.get("finish_reason") == "stop"
            if not text.strip():
                return None, [], False
            # both replies ended on their own: a different length is a divergence too
            return (
                ctx.kit.encode(text.strip()),
                ctx.kit.encode(ref_text.strip()),
                ref_done and tgt_done,
            )

        sprt = BinomialSPRT(h0, h1, alpha=ctx.alpha, beta=float(opts.get("beta", 0.1)))
        n_requests = 0
        trials = events = 0
        div_positions: list[int] = []
        n_prompts = empty = 0
        decision = Decision.CONTINUE
        chunk = int(opts.get("greedy_chunk", 4))
        calibrating = bool(opts.get("calibrating"))
        for start in range(0, len(prompts), chunk):
            batch = prompts[start : start + chunk]
            if not ctx.can_afford(len(batch)):
                break
            try:
                comps = await asyncio.gather(*(target_and_reference(p) for p in batch))
            except EndpointError as exc:
                return self.result(
                    Status.INCONCLUSIVE,
                    n_requests=n_requests,
                    findings=[Finding(severity="warn", message=f"target request failed: {exc}")],
                )
            n_requests += len(batch)
            for tgt, ref, to_end in comps:
                n_prompts += 1
                if tgt is None:
                    if chat_mode:
                        empty += 1
                        continue
                    return self.result(
                        Status.SKIPPED,
                        n_requests=n_requests,
                        notes=["target did not report token ids"],
                    )
                if to_end:
                    d = first_divergence(tgt, ref)
                    n_pos = len(ref) if d is None else d + 1
                else:  # a reply was cut off by max_tokens: compare the common prefix only
                    common = min(len(tgt), len(ref))
                    d = first_divergence(tgt[:common], ref[:common])
                    n_pos = common if d is None else d + 1
                for i in range(n_pos):
                    diverged = d is not None and i == d
                    trials += 1
                    events += diverged
                    decision = sprt.update(diverged)
                    if decision is not Decision.CONTINUE and not calibrating:
                        break
                if d is not None:
                    div_positions.append(d)
                if decision is not Decision.CONTINUE and not calibrating:
                    break
            if decision is not Decision.CONTINUE and not calibrating:
                break

        if chat_mode and not trials:
            return self.result(
                Status.SKIPPED,
                n_requests=n_requests,
                notes=[
                    f"the endpoint returned no reply text for {empty} of {n_prompts} chat prompts "
                    "(a thinking model whose reasoning could not be switched off?)"
                ],
            )
        hazard = events / trials if trials else float("nan")
        metrics = {
            "chat_mode": float(chat_mode),
            "empty_replies": float(empty),
            "hazard": hazard,
            "trials": float(trials),
            "divergences": float(events),
            "prompts_diverged_fraction": len(div_positions)
            / max(1, start // chunk * chunk + chunk),
            "median_divergence_position": float(sorted(div_positions)[len(div_positions) // 2])
            if div_positions
            else float("nan"),
            "h0": h0,
            "h1": h1,
        }
        msg = (
            f"per-token divergence hazard {hazard:.4f} ({events} divergences in {trials} compared "
            f"positions); benign h0={h0:g}, deviation h1={h1:g}"
        )
        status = {
            Decision.ACCEPT_H0: Status.PASS,
            Decision.ACCEPT_H1: Status.DEVIATION,
            Decision.CONTINUE: Status.INCONCLUSIVE,
        }[decision]
        sev = {"PASS": "info", "DEVIATION": "deviation", "INCONCLUSIVE": "warn"}[status.value]
        return self.result(
            status,
            p_value=min(1.0, math.exp(-sprt.llr)),
            effect=hazard,
            effect_unit="per-token divergence hazard",
            threshold=h0,
            n_requests=n_requests,
            metrics=metrics,
            findings=[Finding(severity=sev, message=msg)],
            notes=[
                "p-value is exp(-LLR), a valid sequential p-value (Ville's inequality); the "
                f"test stops for H1 at p <= {ctx.alpha:.3g} and for H0 at the Wald boundary "
                f"(beta={sprt.beta})",
                *(
                    [
                        "chat mode: the endpoint takes no token-id prompts, so replies to chat "
                        "prompts were compared as text re-tokenized with the reference tokenizer; "
                        "the server's chat template is part of what is compared"
                    ]
                    if chat_mode
                    else []
                ),
            ],
        )
