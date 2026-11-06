"""Turn raw study results into the tables published in docs/validation.md."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from scipy import stats

PROBES = [
    "template.token_counts",
    "distribution.teacher_forced",
    "greedy.agreement",
    "sampler.conformance",
    "longctx.needle",
    "tools.structured",
]
SHORT = {
    "template.token_counts": "template",
    "distribution.teacher_forced": "distribution",
    "greedy.agreement": "greedy",
    "sampler.conformance": "sampler",
    "longctx.needle": "longctx",
    "tools.structured": "tools",
}


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval."""
    a = (1 - confidence) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def _pct(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def summarize(rows: list[dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    by_sc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sc[r["scenario"]].append(r)
    controls: dict[str, dict[str, Any]] = {}
    faults: dict[str, dict[str, Any]] = {}
    for name, rs in by_sc.items():
        n = len(rs)
        probe_flags = {p: sum(r["probe_status"].get(p) == "DEVIATION" for r in rs) for p in PROBES}
        entry = {
            "n": n,
            "kind": rs[0]["kind"],
            "deviation": sum(r["verdict"] == "DEVIATION" for r in rs),
            "inconclusive": sum(r["verdict"] == "INCONCLUSIVE" for r in rs),
            "pass": sum(r["verdict"] == "PASS" for r in rs),
            "probe_flags": probe_flags,
            "median_requests": statistics.median(r["requests"] for r in rs),
            "median_seconds": statistics.median(r["seconds"] for r in rs),
            "min_requests": min(r["requests"] for r in rs),
            "max_requests": max(r["requests"] for r in rs),
        }
        (controls if entry["kind"] == "control" else faults)[name] = entry
    ctrl_n = sum(c["n"] for c in controls.values())
    ctrl_fp = sum(c["deviation"] for c in controls.values())
    return {"controls": controls, "faults": faults, "control_runs": ctrl_n,
            "control_false_positives": ctrl_fp, "alpha": alpha}  # fmt: skip


def to_markdown(summary: dict[str, Any]) -> str:
    out: list[str] = []
    n, fp = summary["control_runs"], summary["control_false_positives"]
    out += ["### False positives on correct deployments (controls)", ""]
    if n:
        lo, hi = clopper_pearson(fp, n)
        out += [
            f"Nominal familywise alpha: {summary['alpha']}. Observed false-positive rate over all "
            f"control runs: **{fp}/{n}** (95% CI {100 * lo:.1f}-{100 * hi:.1f}%).",
            "",
            "| Control | Runs | PASS | INCONCLUSIVE | DEVIATION | Probe false flags "
            "| Median requests | Median seconds |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for name, c in summary["controls"].items():
            flags = ", ".join(f"{SHORT[p]} {k}" for p, k in c["probe_flags"].items() if k) or "none"
            out.append(
                f"| {name} | {c['n']} | {c['pass']} | {c['inconclusive']} | {c['deviation']} | "
                f"{flags} | {c['median_requests']:.0f} | {c['median_seconds']:.0f} |"
            )
    header = " | ".join(SHORT[p] for p in PROBES)
    out += [
        "",
        "### Detection of injected faults",
        "",
        "Each probe cell: repetitions in which the probe returned DEVIATION (after Holm).",
        "",
        f"| Fault | Reps | Detected (verdict DEVIATION) | {header} "
        "| Median requests | Median seconds |",
        "|---|---:|---|" + "---:|" * len(PROBES) + "---:|---:|",
    ]
    for name, f in summary["faults"].items():
        cells = " | ".join(f"{f['probe_flags'][p]}/{f['n']}" for p in PROBES)
        out.append(
            f"| {name} | {f['n']} | {_pct(f['deviation'], f['n'])} | {cells} | "
            f"{f['median_requests']:.0f} | {f['median_seconds']:.0f} |"
        )
    return "\n".join(out) + "\n"


def to_readme_table(summary: dict[str, Any]) -> str:
    """Compact results table for the README."""
    n, fp = summary["control_runs"], summary["control_false_positives"]
    lines = []
    if n:
        lo, hi = clopper_pearson(fp, n)
        lines.append(
            f"False positives on correct deployments (held-out controls): **{fp}/{n}** runs "
            f"(95% CI {100 * lo:.1f}-{100 * hi:.1f}%), nominal alpha {summary['alpha']}.\n"
        )
    lines += [
        "| Injected fault | Detected | Flagged by | Median requests | Median time |",
        "|---|---|---|---:|---:|",
    ]
    for name, f in summary["faults"].items():
        by = (
            ", ".join(
                SHORT[p] if k == f["n"] else f"{SHORT[p]} ({k}/{f['n']})"
                for p, k in f["probe_flags"].items()
                if k
            )
            or "nothing"
        )
        secs = f["median_seconds"]
        t = f"{secs:.0f} s" if secs < 120 else f"{secs / 60:.1f} min"
        lines.append(
            f"| {name.removeprefix('fault-')} | {f['deviation']}/{f['n']} | {by} | "
            f"{f['median_requests']:.0f} | {t} |"
        )
    return "\n".join(lines) + "\n"


def splice(text: str, marker: str, body: str) -> str:
    """Replace the content between ``<!-- marker:BEGIN -->`` and ``<!-- marker:END -->``."""
    begin, end = f"<!-- {marker}:BEGIN -->", f"<!-- {marker}:END -->"
    i, j = text.index(begin) + len(begin), text.index(end)
    return text[:i] + "\n" + body + text[j:]
