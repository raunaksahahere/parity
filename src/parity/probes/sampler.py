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
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import stats

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.stats.gtest import expected_null_tv, g_test
from parity.stats.sampling import (
    Expected,
    SamplerSettings,
    expected_distribution,
    stable_top_p,
)
from parity.suites import load_json

VIOLATION_TOLERANCE = 1e-3  # numerical slack for the out-of-support rate under H0
MIN_VIOLATION_RATE = 0.01  # practical threshold for calling a violation rate a deviation
MIN_TV = 0.03  # practical threshold on the in-support total variation distance
MIN_COVERAGE = 0.9  # reported top-K must cover this much probability mass
# Preferred minimum distance between a placed top_p and the neighbouring cumulative masses.
MIN_CUTOFF_MARGIN = 0.02


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
    # each context gets its own settings: top_p is placed per context (see stable_top_p)
    per_prompt: list[tuple[PromptState, Expected, SamplerSettings]]
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
    # v3: each sample is checked against the distribution reported in its own response
    violations: int = 0
    checked: int = 0  # samples whose own reported distribution determined the support
    # G-test on samples that landed in their own response's support, per context: observed
    # counts and the sum of each such sample's own expected distribution (a mixture)
    g_obs: list[dict[int, int]] = field(default_factory=list)
    g_exp: list[dict[int, float]] = field(default_factory=list)


def _entropy(lp: np.ndarray) -> float:
    p = np.exp(lp)
    return float(-(p * lp).sum())


def evaluate_case(case: CaseState) -> tuple[float, float, float, float]:
    """Returns (p_violation, p_gtest, violation_rate, excess_tv) from the samples so far.

    Every sample is judged against the distribution the server reported *in the same
    response*. Violations (truncating cases only) use ``own_support_violation``. The G-test
    compares the tokens of samples that landed in their own response's support with the
    mixture of those responses' expected distributions; conditional on landing in that support,
    each sample follows its own expected distribution exactly, whatever the jitter between
    requests. ``excess_tv`` is the plug-in TV distance minus its expectation under H0.
    """
    s = case.settings
    truncating = s.top_k > 0 or s.top_p < 1.0 or s.min_p > 0.0
    g_total, df_total, tv_num, tv_den = 0.0, 0, 0.0, 0
    for obs_d, exp_d in zip(case.g_obs, case.g_exp, strict=True):
        n = sum(obs_d.values())
        if n == 0 or len(exp_d) < 2:
            continue
        ids = sorted(exp_d)
        probs = np.array([exp_d[i] for i in ids]) / n
        obs = np.array([obs_d.get(i, 0) for i in ids], dtype=float)
        res = g_test(obs, probs / probs.sum())
        g_total += res.g
        df_total += res.df
        tv_num += (res.tv - expected_null_tv(probs / probs.sum(), n)) * n
        tv_den += n
    if truncating and case.checked:
        p_v = float(stats.binom.sf(case.violations - 1, case.checked, VIOLATION_TOLERANCE))
        rate = case.violations / case.checked
    else:
        p_v, rate = 1.0, 0.0
    p_g = float(stats.chi2.sf(g_total, df_total)) if df_total > 0 else 1.0
    return p_v, p_g, rate, (tv_num / tv_den if tv_den else 0.0)


def record_sample(
    case: CaseState, pi: int, sampled: int, own_ids: np.ndarray, own_lp: np.ndarray
) -> None:
    """Account one sample against the distribution reported in its own response."""
    st = case.per_prompt[pi][2]
    case.counts[pi][sampled] = case.counts[pi].get(sampled, 0) + 1
    if len(own_ids) == 0:
        return
    truncating = st.top_k > 0 or st.top_p < 1.0 or st.min_p > 0.0
    verdict = own_support_violation(sampled, own_ids, own_lp, st) if truncating else None
    if truncating and verdict is not None:
        case.checked += 1
        case.violations += verdict
    if truncating and verdict is None:
        return  # this response does not determine the support: leave the sample out
    exp = expected_distribution(own_ids, own_lp, st)
    support = [int(t) for t in exp.ids]
    if sampled not in support:
        return  # outside its own support: a violation or, untruncated, beyond the report
    case.g_obs[pi][sampled] = case.g_obs[pi].get(sampled, 0) + 1
    for tid, pr in zip(support, exp.probs.tolist(), strict=True):
        case.g_exp[pi][tid] = case.g_exp[pi].get(tid, 0.0) + pr


def own_support_violation(
    sampled: int, own_ids: np.ndarray, own_lp: np.ndarray, settings: SamplerSettings
) -> bool | None:
    """Is ``sampled`` outside the support allowed by ``settings``, judged on the distribution the
    server reported *in the same response*? None when that response does not determine it.

    The server's log-probs jitter between requests (on low-bit quantizations by more than the
    gap between neighbouring tokens), so only the same forward pass is a safe reference.
    """
    if len(own_ids) == 0:
        return None
    exp = expected_distribution(own_ids, own_lp, settings)
    if sampled in {int(t) for t in exp.ids}:
        return False
    reported = {int(t) for t in own_ids}
    if settings.top_k > 0:
        # outside the top k of a list holding at least k tokens: ranked below k for sure
        return True if len(own_ids) >= settings.top_k else None
    if settings.top_p < 1.0 and float(np.exp(own_lp).sum()) >= settings.top_p:
        return True  # the nucleus closed inside the reported list and excludes the sample
    if settings.min_p > 0.0 and sampled in reported:
        return True
    return None


def describe(case: CaseState) -> str:
    s = case.settings
    if case.violation_rate >= MIN_VIOLATION_RATE:
        what = []
        if s.top_k > 0:
            what.append(f"top_k={s.top_k}")
        if s.top_p < 1.0:
            what.append(f"top_p (about {s.top_p}, placed per context)")
        names = " / ".join(what) or "the expected support"
        return (
            f"{case.violation_rate:.1%} of samples fell outside the tokens allowed by {names} "
            f"(expected ~0%): the parameter is ignored or applied differently"
        )
    if s.temperature != 1.0:
        return (
            f"samples do not follow temperature={s.temperature} "
            f"(excess TV distance {case.tv:.3f}); "
            "temperature may be ignored or applied at a different stage"
        )
    return f"samples deviate from the expected distribution (excess TV distance {case.tv:.3f})"


class SamplerProbe(Probe):
    id = "sampler.conformance"
    family = "sampler"
    # v2: top_p placed per context away from cumulative-mass boundaries
    # v3: support violations judged on each response's own reported distribution
    version = "3"

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
            per_prompt: list[tuple[PromptState, Expected, SamplerSettings]] = []
            for p in prompts:
                ps = settings
                if settings.top_p < 1.0:
                    # Preference only: v3 judges each sample on its own response, so any top_p is
                    # valid; a value clear of cumulative-mass boundaries keeps the nucleus stable.
                    chosen = stable_top_p(p.logprobs, settings.top_p, min_margin=MIN_CUTOFF_MARGIN)
                    if chosen is not None:
                        ps = replace(settings, top_p=round(chosen, 4))
                    notes.append(f"{c['id']}: context {p.text!r} uses top_p={ps.top_p}")
                # Near-tied truncation boundaries need no special handling: support and expected
                # distribution come from each sample's own response, not from this fetch.
                exp = expected_distribution(p.token_ids, p.logprobs, ps)
                per_prompt.append((p, exp, ps))
            if not per_prompt:
                continue
            case = CaseState(
                c["id"],
                settings,
                per_prompt,
                counts=[{} for _ in per_prompt],
                g_obs=[{} for _ in per_prompt],
                g_exp=[{} for _ in per_prompt],
            )
            truncating = settings.top_k > 0 or settings.top_p < 1.0
            case.detect_rate = (
                float(np.mean([e.tail_mass for _, e, _s in per_prompt])) if truncating else 0.0
            )
            cases.append(case)

        # every sampling request also returns its own distribution (see own_support_violation)
        report_k = min(caps.max_top_k, int(ctx.options.get("sampler_report_k", 50)))
        n_tests = max(1, 2 * len(cases) * len(looks))
        alpha_each = ctx.alpha / n_tests

        # 3. sequential sampling
        drawn = 0
        budget_hit = False
        for look_i, target_n in enumerate(looks):
            active = [c for c in cases if not c.done]
            if not active:
                break
            batch = target_n - drawn
            n_jobs = sum(len(c.per_prompt) for c in active) * batch
            if not ctx.can_afford(n_jobs):
                budget_hit = True
                notes.append(f"request budget reached before look {look_i + 1}/{len(looks)}")
                break
            jobs = [
                (
                    case,
                    pi,
                    ctx.adapter.next_token(
                        ctx.client, ctx.model, ps.ids, top_k=report_k, sampling=st.api_params()
                    ),
                )
                for case in active
                for pi, (ps, _, st) in enumerate(case.per_prompt)
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
                pos = res.positions[0]
                if pos.token_id is None:
                    continue
                own = [t for t in pos.top if t.token_id is not None]
                record_sample(
                    case,
                    pi,
                    pos.token_id,
                    np.array([t.token_id for t in own], dtype=np.int64),
                    np.array([t.logprob for t in own]),
                )
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
            metrics[f"{case.case_id}.excess_tv"] = case.tv
            metrics[f"{case.case_id}.violation_rate"] = case.violation_rate
            metrics[f"{case.case_id}.p_gtest"] = case.p_gtest
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
        if budget_hit and not findings:
            status = Status.INCONCLUSIVE
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
