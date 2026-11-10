"""Combine probe results into a verdict. Never an accusation without evidence."""

from __future__ import annotations

from parity.probes.base import ProbeResult, Status
from parity.report_model import Coverage, ProbeOutcome
from parity.stats.multitest import holm_bonferroni

EXIT_CODES = {Status.PASS: 0, Status.DEVIATION: 1, Status.INCONCLUSIVE: 2}
CORE_STATISTICAL = ("distribution.teacher_forced", "sampler.conformance")
# Probes whose result depends on which weights are served. Template, long-context, sampler and
# tool checks describe the serving stack; they pass for any model behind a correct stack.
IDENTITY_PROBES = ("distribution.teacher_forced", "greedy.agreement")
NO_IDENTITY = (
    "No model-identity probe (distribution, greedy) reached a decision, so nothing here shows "
    "which weights are served and a PASS is withheld."
)


def decide(results: list[ProbeResult], alpha: float) -> tuple[list[ProbeOutcome], Status, bool]:
    """Returns (per-probe outcomes, overall verdict, lower_power flag)."""
    pvals = {
        r.probe: r.p_value
        for r in results
        if r.p_value is not None and not r.hard_evidence and r.status is not Status.SKIPPED
    }
    holm = holm_bonferroni(pvals, alpha) if pvals else None

    outcomes: list[ProbeOutcome] = []
    for r in results:
        adj = holm.adjusted.get(r.probe) if holm else None
        final, reason = r.status, ""
        if (
            r.status is Status.DEVIATION
            and not r.hard_evidence
            and adj is not None
            and adj >= alpha
        ):
            final = Status.INCONCLUSIVE
            reason = (
                f"significant on its own but Holm-adjusted p={adj:.3g} >= alpha={alpha}; "
                "not enough evidence after correcting for the number of probes"
            )
        elif r.status is Status.DEVIATION and r.hard_evidence:
            reason = "deterministic evidence (not subject to multiplicity correction)"
        elif r.status is Status.SKIPPED:
            reason = "; ".join(r.notes) or "skipped"
        outcomes.append(ProbeOutcome(result=r, adjusted_p=adj, final_status=final, reason=reason))

    finals = [o.final_status for o in outcomes]
    evaluated = [s for s in finals if s is not Status.SKIPPED]
    skipped_core = any(
        o.final_status is Status.SKIPPED
        and o.result.probe in CORE_STATISTICAL
        and not any(n.startswith("not run: --fail-fast") for n in o.result.notes)
        for o in outcomes
    )
    lower_power = skipped_core or any(o.result.power == "lower" for o in outcomes)

    if Status.DEVIATION in finals:
        overall = Status.DEVIATION
    elif not evaluated or Status.PASS not in evaluated or Status.INCONCLUSIVE in finals:
        overall = Status.INCONCLUSIVE
    else:
        overall = Status.PASS
    return outcomes, overall, lower_power


def summarize(
    outcomes: list[ProbeOutcome],
    overall: Status,
    lower_power: bool,
    unmet_required: list[str] | None = None,
    no_identity: bool = False,
) -> str:
    dev = [o.result.probe for o in outcomes if o.final_status is Status.DEVIATION]
    inc = [o.result.probe for o in outcomes if o.final_status is Status.INCONCLUSIVE]
    skip = [o.result.probe for o in outcomes if o.final_status is Status.SKIPPED]
    if overall is Status.DEVIATION:
        text = "Behaviour differs from the reference in: " + ", ".join(dev) + "."
    elif overall is Status.PASS:
        text = "No deviation from the reference was detected by the evaluated probes."
    else:
        text = "The evidence is inconclusive" + (f" ({', '.join(inc)})." if inc else ".")
    if skip:
        text += " Skipped: " + ", ".join(skip) + "."
    if lower_power:
        text += " Some core probes could not run, so this result has lower power."
    if unmet_required:
        text += " Required probes did not reach a decision: " + ", ".join(unmet_required) + "."
    if no_identity:
        text += " " + NO_IDENTITY
    return text


def coverage(outcomes: list[ProbeOutcome], required: list[str] | None = None) -> Coverage:
    req = list(required or [])
    decided = [
        o.result.probe for o in outcomes if o.final_status in (Status.PASS, Status.DEVIATION)
    ]
    return Coverage(
        total=len(outcomes),
        evaluated=[o.result.probe for o in outcomes if o.final_status is not Status.SKIPPED],
        decided=decided,
        skipped={
            o.result.probe: (o.reason or "skipped")
            for o in outcomes
            if o.final_status is Status.SKIPPED
        },
        required=req,
        unmet_required=[p for p in req if p not in decided],
    )


def apply_requirements(overall: Status, cov: Coverage) -> Status:
    """A PASS needs every required probe to have reached a decision; a DEVIATION stands on its
    own evidence whatever else was skipped."""
    if cov.unmet_required and overall is Status.PASS:
        return Status.INCONCLUSIVE
    return overall


def coverage_line(cov: Coverage) -> str:
    text = f"{len(cov.evaluated)}/{cov.total} probes evaluated"
    if cov.skipped:
        text += "; skipped: " + ", ".join(cov.skipped)
    if cov.unmet_required:
        text += "; REQUIRED BUT NOT EVALUATED: " + ", ".join(cov.unmet_required)
    return text


def identity_unverified(outcomes: list[ProbeOutcome]) -> bool:
    """True when identity probes were part of the run but none of them reached a decision."""
    selected = [o for o in outcomes if o.result.probe in IDENTITY_PROBES]
    return bool(selected) and not any(
        o.final_status in (Status.PASS, Status.DEVIATION) for o in selected
    )


def apply_identity_rule(overall: Status, outcomes: list[ProbeOutcome], allow: bool) -> Status:
    """A PASS needs evidence about the weights. Without it the run is INCONCLUSIVE, unless the
    caller explicitly accepts a PASS that only describes the serving stack."""
    if overall is Status.PASS and not allow and identity_unverified(outcomes):
        return Status.INCONCLUSIVE
    return overall
