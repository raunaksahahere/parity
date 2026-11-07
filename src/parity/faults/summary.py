"""Turn raw study results into the tables published in docs/validation.md."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from scipy import stats

# suite execution order (cheapest first); fail_fast_requests depends on it
PROBES = [
    "template.token_counts",
    "longctx.needle",
    "tools.structured",
    "greedy.agreement",
    "distribution.teacher_forced",
    "sampler.conformance",
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


def fail_fast_requests(row: dict[str, Any]) -> int:
    """Requests a ``--fail-fast`` run would have used: the cumulative per-probe count, in suite
    order, through the first deviating probe. Derived from full runs, which is valid because
    probes do not influence each other."""
    total = 0
    for p in PROBES:
        total += int(row.get("probe_requests", {}).get(p, 0))
        if row["probe_status"].get(p) == "DEVIATION":
            break
    return total


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
            "median_fail_fast_requests": statistics.median(fail_fast_requests(r) for r in rs),
            "median_seconds": statistics.median(r["seconds"] for r in rs),
            "min_requests": min(r["requests"] for r in rs),
            "max_requests": max(r["requests"] for r in rs),
        }
        (controls if entry["kind"] == "control" else faults)[name] = entry
    ctrl_n = sum(c["n"] for c in controls.values())
    ctrl_fp = sum(c["deviation"] for c in controls.values())
    return {"controls": controls, "faults": faults, "control_runs": ctrl_n,
            "control_false_positives": ctrl_fp, "alpha": alpha}  # fmt: skip


def to_markdown(summary: dict[str, Any], heading: str = "###") -> str:
    out: list[str] = []
    n, fp = summary["control_runs"], summary["control_false_positives"]
    out += [f"{heading} False positives on correct deployments (controls)", ""]
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
        f"{heading} Detection of injected faults",
        "",
        "Each probe cell: repetitions in which the probe returned DEVIATION (after Holm).",
        "",
        f"| Fault | Reps | Detected (verdict DEVIATION) | {header} "
        "| Median requests (full suite) | Median requests to detection (--fail-fast) "
        "| Median seconds (full suite) |",
        "|---|---:|---|" + "---:|" * len(PROBES) + "---:|---:|---:|",
    ]
    for name, f in summary["faults"].items():
        cells = " | ".join(f"{f['probe_flags'][p]}/{f['n']}" for p in PROBES)
        out.append(
            f"| {name} | {f['n']} | {_pct(f['deviation'], f['n'])} | {cells} | "
            f"{f['median_requests']:.0f} | {f['median_fail_fast_requests']:.0f} | "
            f"{f['median_seconds']:.0f} |"
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
        "| Injected fault | Detected | Flagged by | Requests to detection (fail-fast) "
        "| Full-suite requests | Full-suite time |",
        "|---|---|---|---:|---:|---:|",
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
            f"{f['median_fail_fast_requests']:.0f} | {f['median_requests']:.0f} | {t} |"
        )
    return "\n".join(lines) + "\n"


def splice(text: str, marker: str, body: str) -> str:
    """Replace the content between ``<!-- marker:BEGIN -->`` and ``<!-- marker:END -->``."""
    begin, end = f"<!-- {marker}:BEGIN -->", f"<!-- {marker}:END -->"
    i, j = text.index(begin) + len(begin), text.index(end)
    return text[:i] + "\n" + body + text[j:]


SAMPLER_FAULTS = frozenset(
    {
        "fault-topk-ignored",
        "fault-topp-ignored",
        "fault-temperature-ignored",
        "fault-temperature-x1.5",
    }
)


def to_sampler_table(rows: list[dict[str, Any]]) -> str:
    """Sampler-only re-validation: does the sampler probe flag exactly the sampler faults?"""
    by_sc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sc[r["scenario"]].append(r)
    out = [
        "| Scenario | Sampler fault? | Should report | Reps | Sampler DEVIATION | "
        "Median sampler requests | Median seconds |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, rs in by_sc.items():
        is_fault = name in SAMPLER_FAULTS
        flagged = sum(r["probe_status"].get("sampler.conformance") == "DEVIATION" for r in rs)
        req = statistics.median(r["probe_requests"].get("sampler.conformance", 0) for r in rs)
        out.append(
            f"| {name} | {'yes' if is_fault else 'no'} | {'DEVIATION' if is_fault else 'PASS'} | "
            f"{len(rs)} | {flagged}/{len(rs)} | {req:.0f} | "
            f"{statistics.median(r['seconds'] for r in rs):.0f} |"
        )
    return "\n".join(out) + "\n"


# -- v0.1.1 study tables --------------------------------------------------------------------------

ALL_PROBES = PROBES


def _med(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _fmt_s(secs: float | None) -> str:
    if secs is None:
        return "n/a"
    return f"{secs:.0f} s" if secs < 120 else f"{secs / 60:.1f} min"


def _fmt_mb(mb: float | None) -> str:
    return "n/a" if mb is None else f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def _ci(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{100 * lo:.0f}-{100 * hi:.0f}%"


def base_fault(name: str) -> str:
    return name.removesuffix("-cpu")


def study_summary(rows: list[dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    """Per-scenario aggregates of a v0.1.1 study (rows from results.jsonl)."""
    by_sc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sc[r["scenario"]].append(r)
    scen: dict[str, dict[str, Any]] = {}
    for name, rs in by_sc.items():
        scen[name] = {
            "name": name,
            "kind": rs[0]["kind"],
            "backend": rs[0].get("backend", "gpu"),
            "simulation": rs[0].get("simulation", "real"),
            "fresh": bool(rs[0].get("fresh")),
            "n": len(rs),
            "seeds": sorted(r.get("seed", r["rep"]) for r in rs),
            "pass": sum(r["verdict"] == "PASS" for r in rs),
            "inconclusive": sum(r["verdict"] == "INCONCLUSIVE" for r in rs),
            "deviation": sum(r["verdict"] == "DEVIATION" for r in rs),
            "probe_dev": {p: sum(r["probe_status"].get(p) == "DEVIATION" for r in rs)
                          for p in ALL_PROBES},
            "probe_inc": {p: sum(r["probe_status"].get(p) == "INCONCLUSIVE" for r in rs)
                          for p in ALL_PROBES},
            "median_requests": _med([r["requests"] for r in rs]),
            "median_fail_fast_inferred": _med([fail_fast_requests(r) for r in rs]),
            "median_seconds": _med([r["seconds"] for r in rs]),
            "median_parity_rss_mb": _med([r.get("parity_peak_rss_mb") for r in rs]),
            "max_parity_rss_mb": max((r.get("parity_peak_rss_mb") or 0 for r in rs), default=0)
            or None,
            "median_server_rss_mb": _med([r.get("server_peak_rss_mb") for r in rs]),
        }  # fmt: skip
    controls = [s for s in scen.values() if s["kind"] == "control"]
    faults = [s for s in scen.values() if s["kind"] == "fault"]
    return {
        "alpha": alpha,
        "scenarios": scen,
        "controls": controls,
        "faults": faults,
        "control_runs": sum(c["n"] for c in controls),
        "control_fp": sum(c["deviation"] for c in controls),
        "control_inconclusive": sum(c["inconclusive"] for c in controls),
        "control_deployments": len(controls),
        "control_deployments_with_fp": sum(c["deviation"] > 0 for c in controls),
    }


def study_control_table(s: dict[str, Any]) -> str:
    n, fp, inc = s["control_runs"], s["control_fp"], s["control_inconclusive"]
    k, kfp = s["control_deployments"], s["control_deployments_with_fp"]
    fresh = [c for c in s["controls"] if c["fresh"]]
    fresh_n = sum(c["n"] for c in fresh)
    fresh_fp = sum(c["deviation"] for c in fresh)
    out = [
        f"**False positives: {fp}/{n} runs** (95% CI {_ci(fp, n)}; nominal familywise alpha "
        f"{s['alpha']}), from **{k} independent correct deployment configurations** "
        f"({kfp}/{k} with any false positive, 95% CI {_ci(kfp, k)}). INCONCLUSIVE: {inc}/{n} "
        f"runs. Fresh held-out configurations alone: {fresh_fp}/{fresh_n} runs over "
        f"{len(fresh)} configurations.",
        "",
        "Runs of one configuration repeat the suite with different seeds against the same "
        "server process; they are not independent deployments.",
        "",
        "| Control | Backend | Fresh | Seeds | PASS | INCONCLUSIVE | DEVIATION | Probe flags "
        "| Probe INCONCLUSIVE | Median requests | Median time | Peak RSS parity / server |",
        "|---|---|:-:|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for c in sorted(s["controls"], key=lambda c: (c["backend"], not c["fresh"], c["name"])):
        flags = ", ".join(f"{SHORT[p]} {v}" for p, v in c["probe_dev"].items() if v) or "none"
        incs = ", ".join(f"{SHORT[p]} {v}" for p, v in c["probe_inc"].items() if v) or "none"
        out.append(
            f"| {c['name']} | {c['backend'].upper()} | {'yes' if c['fresh'] else ''} | {c['n']} "
            f"| {c['pass']} | {c['inconclusive']} | {c['deviation']} | {flags} | {incs} "
            f"| {c['median_requests']:.0f} | {_fmt_s(c['median_seconds'])} "
            f"| {_fmt_mb(c['max_parity_rss_mb'])} / {_fmt_mb(c['median_server_rss_mb'])} |"
        )
    return "\n".join(out) + "\n"


def study_fault_table(s: dict[str, Any], measured_ff: dict[str, float] | None = None) -> str:
    """One row per fault: GPU and CPU deployments side by side, per-probe attribution."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in s["faults"]:
        groups[base_fault(f["name"])].append(f)
    header = " | ".join(SHORT[p] for p in ALL_PROBES)
    out = [
        "Detected = runs with verdict DEVIATION. Probe cells: runs in which that probe returned "
        "DEVIATION (after Holm), GPU and CPU runs together. Inconclusive runs are counted, not "
        "hidden.",
        "",
        f"| Fault | Simulation | GPU detected | CPU detected | INCONCLUSIVE | {header} "
        "| Median requests (full) | Requests to detection (--fail-fast) | Median time (full) |",
        "|---|---|---|---|---:|" + "---:|" * len(ALL_PROBES) + "---:|---:|---:|",
    ]
    for name, fs in groups.items():
        gpu = next((f for f in fs if f["backend"] == "gpu"), None)
        cpu = next((f for f in fs if f["backend"] == "cpu"), None)

        def det(f: dict[str, Any] | None) -> str:
            if f is None:
                return "not run"
            return f"{f['deviation']}/{f['n']} ({_ci(f['deviation'], f['n'])})"

        n_all = sum(f["n"] for f in fs)
        cells = " | ".join(f"{sum(f['probe_dev'][p] for f in fs)}/{n_all}" for p in ALL_PROBES)
        ff_inferred = _med([f["median_fail_fast_inferred"] for f in fs if f["backend"] == "gpu"])
        ff = (
            f"{measured_ff[name]:.0f} measured"
            if measured_ff and name in measured_ff
            else f"{ff_inferred:.0f} inferred"
            if ff_inferred is not None
            else "n/a"
        )
        main = gpu or cpu
        assert main is not None
        out.append(
            f"| {name.removeprefix('fault-')} | {main['simulation']} | {det(gpu)} | {det(cpu)} "
            f"| {sum(f['inconclusive'] for f in fs)} | {cells} "
            f"| {main['median_requests']:.0f} | {ff} | {_fmt_s(main['median_seconds'])} |"
        )
    return "\n".join(out) + "\n"


def fail_fast_table(ff_rows: list[dict[str, Any]], full: dict[str, Any]) -> str:
    """Real --fail-fast runs next to the request count inferred from the full-suite runs."""
    by_sc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ff_rows:
        by_sc[r["scenario"]].append(r)
    out = [
        "| Scenario | Runs | Verdicts | First deviating probe | Measured requests (median, "
        "range) | Inferred from full runs | Measured time |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for name, rs in by_sc.items():
        verdicts = ", ".join(
            f"{v} {sum(r['verdict'] == v for r in rs)}"
            for v in ("DEVIATION", "INCONCLUSIVE", "PASS")
            if any(r["verdict"] == v for r in rs)
        )
        first = {
            next((SHORT[p] for p in PROBES if r["probe_status"].get(p) == "DEVIATION"), "none")
            for r in rs
        }
        req = [r["requests"] for r in rs]
        inferred = full["scenarios"].get(name, {}).get("median_fail_fast_inferred")
        out.append(
            f"| {name} | {len(rs)} | {verdicts} | {', '.join(sorted(first))} "
            f"| {statistics.median(req):.0f} ({min(req)}-{max(req)}) "
            f"| {'n/a' if inferred is None else f'{inferred:.0f}'} "
            f"| {_fmt_s(statistics.median(r['seconds'] for r in rs))} |"
        )
    return "\n".join(out) + "\n"


def study_readme_table(s: dict[str, Any], measured_ff: dict[str, float] | None = None) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in s["faults"]:
        groups[base_fault(f["name"])].append(f)
    n, fp = s["control_runs"], s["control_fp"]
    lines = [
        f"False positives on correct deployments: **{fp}/{n} runs** (95% CI {_ci(fp, n)}) over "
        f"{s['control_deployments']} independent configurations, {s['control_inconclusive']} "
        "inconclusive.\n",
        "| Injected fault | Simulation | Detected (GPU · CPU) | Flagged by | Requests to "
        "detection (fail-fast) | Full-suite requests | Full-suite time |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for name, fs in groups.items():
        n_all = sum(f["n"] for f in fs)
        flagged = {p: sum(f["probe_dev"][p] for f in fs) for p in ALL_PROBES}
        by = (
            ", ".join(
                SHORT[p] if k == n_all else f"{SHORT[p]} ({k}/{n_all})"
                for p, k in flagged.items()
                if k
            )
            or "nothing"
        )
        det = " · ".join(
            f"{f['deviation']}/{f['n']}" for f in sorted(fs, key=lambda f: f["backend"] != "gpu")
        )
        main = next((f for f in fs if f["backend"] == "gpu"), fs[0])
        ff = (
            f"{measured_ff[name]:.0f}"
            if measured_ff and name in measured_ff
            else f"{main['median_fail_fast_inferred']:.0f}*"
        )
        lines.append(
            f"| {name.removeprefix('fault-')} | {main['simulation']} | {det} | {by} | {ff} "
            f"| {main['median_requests']:.0f} | {_fmt_s(main['median_seconds'])} |"
        )
    return "\n".join(lines) + "\n"
