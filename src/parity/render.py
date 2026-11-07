"""Render a RunReport as HTML, Markdown or JSON. Reports never contain secrets: the report
model has no credential fields and target URLs are sanitized when the report is built."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from parity.report_model import RunReport

_env = Environment(
    loader=PackageLoader("parity", "templates"),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=False,
)


def to_html(report: RunReport) -> str:
    return _env.get_template("report.html.j2").render(r=report)


def _fmt(v: float | None, spec: str = ".4g") -> str:
    return "n/a" if v is None else format(v, spec)


def _cell(text: str) -> str:
    """Escape a Markdown table cell (a bare | would start a new column)."""
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(report: RunReport) -> str:
    r = report
    lines = [
        f"# Parity conformance report: {r.target.model}",
        "",
        f"**Verdict: {r.verdict.value}**. {r.summary}",
        "",
    ]
    if r.coverage is not None:
        c = r.coverage
        lines.append(f"**Coverage: {len(c.evaluated)}/{c.total} probes evaluated.**")
        if c.skipped:
            lines += ["", "| Skipped probe | Why |", "|---|---|"]
            lines += [f"| `{p}` | {_cell(why)} |" for p, why in c.skipped.items()]
        if c.required:
            unmet = ", ".join(f"`{p}`" for p in c.unmet_required) or "none"
            lines += ["", f"Required probes: {', '.join(c.required)}; not decided: {unmet}."]
        lines.append("")
    lines += [
        f"- Target: `{r.target.base_url}` (model `{r.target.model}`, adapter `{r.target.adapter}`)",
        f"- Reference: {r.reference.kind} `{r.reference.model_id}` @ `{r.reference.revision}`",
        f"- Run `{r.run_id}`, suite `{r.suite}`, alpha {r.alpha}, {r.requests_total} requests, "
        f"{r.duration_s} s, parity {r.parity_version}",
    ]
    if r.requests_consumed is not None and (r.resumed or r.requests_consumed != r.requests_total):
        lines.append(
            f"- Requests consumed (retries, interrupted probes and earlier sessions included): "
            f"{r.requests_consumed}" + (" (resumed run)" if r.resumed else "")
        )
    if r.calibration:
        th = ", ".join(f"{k}={v:.3g}" for k, v in r.calibration.thresholds.items())
        lines.append(f"- Calibration ({r.calibration.created}): {th}")
    else:
        lines.append("- Calibration: none (uncalibrated defaults)")
    if r.lower_power:
        lines.append("- Power: lower, some core probes could not run on this endpoint")
    lines += [
        "",
        "## Probes",
        "",
        "| Probe | Verdict | Effect | Threshold | p / Holm p | Requests |",
        "|---|---|---|---|---|---|",
    ]
    for o in r.probes:
        p = o.result
        was = f" (was {p.status.value})" if o.final_status != p.status else ""
        pv = f"{_fmt(p.p_value, '.3g')} / {_fmt(o.adjusted_p, '.3g')}"
        effect = f"{_fmt(p.effect)} {_cell(p.effect_unit)}"
        lines.append(
            f"| `{p.probe}` | {o.final_status.value}{was} | {effect} | "
            f"{_fmt(p.threshold, '.3g')} | {pv} | {p.n_requests} |"
        )
    lines.append("")
    for o in r.probes:
        p = o.result
        lines.append(f"### `{p.probe}`")
        if o.reason:
            lines.append(f"_{o.reason}_")
        lines += [f"- **{f.severity}**: {f.message}" for f in p.findings]
        lines += [f"- note: {n}" for n in p.notes]
        lines.append("")
    lines += ["## What this does not prove", ""]
    lines += [f"- {item}" for item in r.limits]
    return "\n".join(lines) + "\n"


def write_reports(report: RunReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "report.json",
        "md": out_dir / "report.md",
        "html": out_dir / "report.html",
    }
    paths["json"].write_text(report.model_dump_json(indent=1))
    paths["md"].write_text(to_markdown(report))
    paths["html"].write_text(to_html(report))
    return paths
