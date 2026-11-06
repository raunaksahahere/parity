"""Family 3: greedy (temperature-0) agreement with the reference.

Both sides continue the same token-id prompts greedily. Run-to-run numerical noise makes
exact agreement impossible even for a correct deployment (near-tied tokens flip), so the
statistic is a *divergence hazard*: every compared position is a Bernoulli trial for "the
continuations first diverge here", and a prompt stops contributing at its first divergence
(afterwards the contexts differ and later tokens are not comparable). The hazard is tested
sequentially with a Bernoulli SPRT:

    H0: hazard = h0 (benign, from calibration)   vs   H1: hazard = h1 (a real deviation)

This uses the full divergence-position distribution rather than only "identical or not".
"""

from __future__ import annotations

import asyncio

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.stats.sprt import BinomialSPRT, Decision
from parity.suites import load_corpus

DEFAULT_H0 = 0.004  # per-token benign divergence hazard; replaced by calibration
DEFAULT_H1 = 0.02


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
        if not caps.token_id_prompts:
            return self.result(
                Status.SKIPPED,
                notes=["target cannot take token-id prompts; continuations cannot be aligned"],
            )
        opts = ctx.options
        th = opts.get("thresholds", {})
        h0 = float(th.get("greedy_hazard", DEFAULT_H0))
        h1 = float(opts.get("greedy_h1", max(DEFAULT_H1, 5 * h0)))
        n_tokens = int(opts.get("greedy_tokens", 48))
        prompt_len = int(opts.get("greedy_prompt_tokens", 16))
        prompts = [
            ids
            for d in load_corpus(opts.get("suite", "core-v1"))
            if len(ids := ctx.kit.encode(d.text)[:prompt_len]) >= 4
        ]

        sprt = BinomialSPRT(h0, h1, alpha=ctx.alpha, beta=float(opts.get("beta", 0.1)))
        n_requests = 0
        trials = events = 0
        div_positions: list[int] = []
        n_prompts = 0
        decision = Decision.CONTINUE
        chunk = int(opts.get("greedy_chunk", 4))
        for start in range(0, len(prompts), chunk):
            batch = prompts[start : start + chunk]
            try:
                comps = await asyncio.gather(
                    *(
                        ctx.adapter.generate(
                            ctx.client,
                            ctx.model,
                            p,
                            n_tokens=n_tokens,
                            sampling={"temperature": 0.0},
                        )
                        for p in batch
                    )
                )
            except EndpointError as exc:
                return self.result(
                    Status.INCONCLUSIVE,
                    n_requests=n_requests,
                    findings=[Finding(severity="warn", message=f"target request failed: {exc}")],
                )
            n_requests += len(batch)
            for p, comp in zip(batch, comps, strict=True):
                n_prompts += 1
                tgt = [pos.token_id for pos in comp.positions]
                if any(t is None for t in tgt):
                    return self.result(
                        Status.SKIPPED,
                        n_requests=n_requests,
                        notes=["target did not report token ids"],
                    )
                ref = ctx.reference.greedy(p, n_tokens)
                d = first_divergence([int(t) for t in tgt if t is not None], ref)
                n_pos = len(ref) if d is None else d + 1
                for i in range(n_pos):
                    diverged = d is not None and i == d
                    trials += 1
                    events += diverged
                    decision = sprt.update(diverged)
                    if decision is not Decision.CONTINUE:
                        break
                if d is not None:
                    div_positions.append(d)
                if decision is not Decision.CONTINUE:
                    break
            if decision is not Decision.CONTINUE:
                break

        hazard = events / trials if trials else float("nan")
        metrics = {
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
            effect=hazard,
            effect_unit="per-token divergence hazard",
            threshold=h0,
            n_requests=n_requests,
            metrics=metrics,
            findings=[Finding(severity=sev, message=msg)],
            notes=[
                "SPRT p-values are not defined; the verdict comes from the Wald boundaries "
                f"(alpha={ctx.alpha}, beta={sprt.beta})"
            ],
        )
