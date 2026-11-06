"""Combine probe results into a verdict. Never an accusation without evidence."""

from __future__ import annotations

from parity.probes.base import ProbeResult, Status
from parity.report_model import ProbeOutcome
from parity.stats.multitest import holm_bonferroni

EXIT_CODES = {Status.PASS: 0, Status.DEVIATION: 1, Status.INCONCLUSIVE: 2}
CORE_STATISTICAL = ("distribution.teacher_forced", "sampler.conformance")


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
        o.final_status is Status.SKIPPED and o.result.probe in CORE_STATISTICAL for o in outcomes
    )
    lower_power = skipped_core or any(o.result.power == "lower" for o in outcomes)

    if Status.DEVIATION in finals:
        overall = Status.DEVIATION
    elif not evaluated or Status.PASS not in evaluated or Status.INCONCLUSIVE in finals:
        overall = Status.INCONCLUSIVE
    else:
        overall = Status.PASS
    return outcomes, overall, lower_power


def summarize(outcomes: list[ProbeOutcome], overall: Status, lower_power: bool) -> str:
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
    return text
