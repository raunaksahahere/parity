"""Family 4: sampler conformance.

For contexts whose next-token distribution is known (taken from the target's own
temperature-1 log-probs, so model differences cannot masquerade as sampler bugs),
draw samples under explicit temperature/top_k/top_p and test them against the
expected truncated, renormalized distribution.

Two tests per case:
* violations: samples outside the expected support (e.g. a token ranked beyond
  ``top_k``). Under H0 the rate is ~0; a one-sided binomial test against a small
  numerical tolerance gives the p-value.
* G-test on the in-support samples against the expected conditional distribution
  (catches ignored or misapplied temperature, and wrong truncation shapes).

Sampling is sequential: after each look the tests are evaluated at a Bonferroni-
spent alpha, so a clear deviation stops early and the overall error rate stays
below ``alpha`` (see docs/methodology.md).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.stats.gtest import g_test
from parity.stats.sampling import Expected, SamplerSettings, expected_distribution
from parity.suites import load_json

VIOLATION_TOLERANCE = 1e-3  # numerical slack for the out-of-support rate under H0
MIN_VIOLATION_RATE = 0.01  # practical threshold for calling a violation rate a deviation
MIN_TV = 0.03  # practical threshold on the in-support total variation distance
MIN_COVERAGE = 0.9  # reported top-K must cover this much probability mass
MIN_BOUNDARY_GAP = 0.05  # skip cases whose truncation boundary sits on near-tied tokens


@dataclass
class PromptState:
    text: str
    ids: list[int]
    token_ids: np.ndarray
    logprobs: np.ndarray
    coverage: float
    entropy: float


@dataclass
class CaseState:
    case_id: str
    settings: SamplerSettings
    per_prompt: list[tuple[PromptState, Expected]]
    counts: list[dict[int, int]] = field(default_factory=list)
    n: int = 0
    done: bool = False
    verdict: str = "pass"  # pass | deviation
    p_violation: float = 1.0
    p_gtest: float = 1.0
    violation_rate: float = 0.0
    tv: float = 0.0
    stopped_at: int = 0
    detect_rate: float = 0.0  # violation rate the case would show if truncation were ignored


def _entropy(lp: np.ndarray) -> float:
    p = np.exp(lp)
    return float(-(p * lp).sum())


def evaluate_case(case: CaseState) -> tuple[float, float, float, float]:
    """Returns (p_violation, p_gtest, violation_rate, tv) from the current counts."""
    violations = 0
    total = 0
    g_total, df_total, tv_num, tv_den = 0.0, 0, 0.0, 0
    for (_, exp), counts in zip(case.per_prompt, case.counts, strict=True):
        support = {int(t): i for i, t in enumerate(exp.ids)}
        obs = np.zeros(len(exp.ids))
        in_r = 0
        for tok, c in counts.items():
            total += c
            if tok in support:
                obs[support[tok]] += c
                in_r += c
            else:
                violations += c
        if in_r > 0 and len(exp.ids) > 1:
            res = g_test(obs, exp.probs)
            g_total += res.g
            df_total += res.df
            tv_num += res.tv * in_r
            tv_den += in_r
    p_v = float(stats.binom.sf(violations - 1, total, VIOLATION_TOLERANCE)) if total else 1.0
    p_g = float(stats.chi2.sf(g_total, df_total)) if df_total > 0 else 1.0
    return p_v, p_g, (violations / total if total else 0.0), (tv_num / tv_den if tv_den else 0.0)


def describe(case: CaseState) -> str:
    s = case.settings
    if case.violation_rate >= MIN_VIOLATION_RATE:
        what = []
        if s.top_k > 0:
            what.append(f"top_k={s.top_k}")
        if s.top_p < 1.0:
            what.append(f"top_p={s.top_p}")
        names = " / ".join(what) or "the expected support"
        return (
            f"{case.violation_rate:.1%} of samples fell outside the tokens allowed by {names} "
            f"(expected ~0%): the parameter is ignored or applied differently"
        )
    if s.temperature != 1.0:
        return (
            f"samples do not follow temperature={s.temperature} (TV distance {case.tv:.3f}); "
            "temperature may be ignored or applied at a different stage"
        )
    return f"samples deviate from the expected distribution (TV distance {case.tv:.3f})"


class SamplerProbe(Probe):
    id = "sampler.conformance"
    family = "sampler"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        caps = ctx.adapter.capabilities
        if not (caps.token_id_prompts and caps.top_logprobs):
            return self.result(
                Status.SKIPPED,
                notes=[
                    "target cannot take token-id prompts and return log-probs; "
                    "the sampler probe needs both to know the true next-token distribution"
                ],
            )
        spec = load_json(ctx.options.get("suite", "core-v1"), "sampler.json")
        looks: list[int] = list(ctx.options.get("sampler_looks", [100, 250, 500]))
        n_prompts_used = int(ctx.options.get("sampler_prompts", 2))
        top_k_report = min(caps.max_top_k, 1000)
        n_requests = 0
        notes: list[str] = []

        # 1. known next-token distributions from the target's own neutral log-probs
        prompts: list[PromptState] = []
        for text in spec["prompts"]:
            ids = ctx.kit.encode(text)
            try:
                comp = await ctx.adapter.next_token(
                    ctx.client, ctx.model, ids, top_k=top_k_report, sampling={}
                )
            except EndpointError as exc:
                notes.append(f"could not fetch log-probs for a context: {exc}")
                continue
            n_requests += 1
            if not comp.positions or not comp.positions[0].top:
                continue
            top = comp.positions[0].top
            if any(t.token_id is None for t in top):
                continue
            tok_ids = np.array([t.token_id for t in top], dtype=np.int64)
            lp = np.array([t.logprob for t in top])
            cov = float(np.exp(lp).sum())
            if cov < MIN_COVERAGE:
                notes.append(f"context {text!r} skipped: reported log-probs cover only {cov:.2f}")
                continue
            prompts.append(PromptState(text, ids, tok_ids, lp, cov, _entropy(lp)))
        if not prompts:
            return self.result(
                Status.INCONCLUSIVE,
                n_requests=n_requests,
                notes=[*notes, "no context with usable log-probs"],
                power="lower",
            )
        prompts = sorted(prompts, key=lambda p: -p.entropy)[:n_prompts_used]

        # 2. build cases
        cases: list[CaseState] = []
        for c in spec["cases"]:
            settings = SamplerSettings(
                c["temperature"], c["top_k"], c["top_p"], c.get("min_p", 0.0)
            )
            per_prompt: list[tuple[PromptState, Expected]] = []
            for p in prompts:
                exp = expected_distribution(p.token_ids, p.logprobs, settings)
                if exp.boundary_gap < MIN_BOUNDARY_GAP:
                    notes.append(
                        f"{c['id']}: context {p.text!r} skipped (near-tied truncation boundary)"
                    )
                    continue
                per_prompt.append((p, exp))
            if not per_prompt:
                continue
            case = CaseState(c["id"], settings, per_prompt, counts=[{} for _ in per_prompt])
            truncating = settings.top_k > 0 or settings.top_p < 1.0
            case.detect_rate = (
                float(np.mean([e.tail_mass for _, e in per_prompt])) if truncating else 0.0
            )
            cases.append(case)

        n_tests = max(1, 2 * len(cases) * len(looks))
        alpha_each = ctx.alpha / n_tests

        # 3. sequential sampling
        drawn = 0
        for look_i, target_n in enumerate(looks):
            active = [c for c in cases if not c.done]
            if not active:
                break
            batch = target_n - drawn
            jobs = [
                (
                    case,
                    pi,
                    ctx.adapter.next_token(
                        ctx.client, ctx.model, ps.ids, top_k=1, sampling=case.settings.api_params()
                    ),
                )
                for case in active
                for pi, (ps, _) in enumerate(case.per_prompt)
                for _ in range(batch)
            ]
            results = await asyncio.gather(*(j[2] for j in jobs), return_exceptions=True)
            n_requests += len(jobs)
            for (case, pi, _), res in zip(jobs, results, strict=True):
                if (
                    isinstance(res, BaseException)
                    or not res.positions
                    or res.positions[0].token_id is None
                ):
                    continue
                tok = res.positions[0].token_id
                case.counts[pi][tok] = case.counts[pi].get(tok, 0) + 1
            drawn = target_n
            last = look_i == len(looks) - 1
            for case in active:
                case.n = drawn
                p_v, p_g, vr, tv = evaluate_case(case)
                case.p_violation, case.p_gtest, case.violation_rate, case.tv = p_v, p_g, vr, tv
                significant = (p_v < alpha_each and vr >= MIN_VIOLATION_RATE) or (
                    p_g < alpha_each and tv >= MIN_TV
                )
                if significant:
                    case.verdict, case.done, case.stopped_at = "deviation", True, drawn
                elif last:
                    case.done, case.stopped_at = True, drawn

        # 4. assemble result
        findings: list[Finding] = []
        metrics: dict[str, float] = {"cases": float(len(cases))}
        worst_p = 1.0
        worst_effect = 0.0
        for case in cases:
            metrics[f"{case.case_id}.tv"] = case.tv
            metrics[f"{case.case_id}.violation_rate"] = case.violation_rate
            metrics[f"{case.case_id}.samples"] = float(case.n)
            p_case = min(1.0, min(case.p_violation, case.p_gtest) * n_tests)
            worst_p = min(worst_p, p_case)
            if case.verdict == "deviation":
                worst_effect = max(worst_effect, case.violation_rate, case.tv)
                findings.append(
                    Finding(
                        severity="deviation",
                        message=f"{case.case_id}: {describe(case)}",
                        detail={
                            "n_per_prompt": case.n,
                            "p_violation": case.p_violation,
                            "p_gtest": case.p_gtest,
                        },
                    )
                )
            elif case.detect_rate < 0.02 and (case.settings.top_k > 0 or case.settings.top_p < 1.0):
                notes.append(
                    f"{case.case_id}: only {case.detect_rate:.1%} of mass lies beyond the "
                    "truncation, so an ignored parameter would be hard to see here"
                )
        status = Status.DEVIATION if findings else Status.PASS
        if not findings:
            findings.append(
                Finding(
                    severity="info",
                    message=f"{len(cases)} sampler cases match the expected distributions",
                )
            )
        return self.result(
            status,
            p_value=worst_p,
            effect=worst_effect,
            effect_unit="max(violation rate, TV distance)",
            n_requests=n_requests,
            metrics=metrics,
            findings=findings,
            notes=notes,
        )
