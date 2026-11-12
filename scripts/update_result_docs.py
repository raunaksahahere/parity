"""Insert generated validation tables into README.md and docs/validation.md.

Every number comes from raw result files committed under docs/data; nothing is typed by hand.

  uv run python scripts/update_result_docs.py            # regenerate from docs/data
  uv run python scripts/update_result_docs.py --import-study runs/v011-study \\
      --import-fail-fast runs/v011-fail-fast            # copy raw results in first

v0.1.0 (historical) tables come from docs/data/validation-results.jsonl (second full run),
validation-v1-results.jsonl (first run) and validation-sampler-v3-results.jsonl. v0.1.1 tables
come from docs/data/v0.1.1-*.
"""

import argparse
import json
import shutil
import statistics
from pathlib import Path

from parity.faults.study import load_results
from parity.faults.summary import (
    fail_fast_table,
    splice,
    study_control_table,
    study_fault_table,
    study_readme_table,
    study_summary,
    summarize,
    to_markdown,
    to_sampler_table,
)

DATA = Path("docs/data")
STUDY = DATA / "v0.1.1-study-results.jsonl"
FAIL_FAST = DATA / "v0.1.1-fail-fast-results.jsonl"
DEV_TOOLS = DATA / "v0.1.1-dev-tools-probe.jsonl"
CONFIRM = DATA / "v0.3.0-confirmation-results.jsonl"
OLLAMA = DATA / "v0.3.0-ollama-results.jsonl"
SMOLLM2 = DATA / "v0.3.0-smollm2-study-results.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--import-study", type=Path, help="study directory to copy raw results from")
ap.add_argument("--import-fail-fast", type=Path, help="fail-fast study directory")
ns = ap.parse_args()

if ns.import_study:
    shutil.copy(ns.import_study / "results.jsonl", STUDY)
    shutil.copy(ns.import_study / "calibration.json", DATA / "v0.1.1-calibration.json")
    shutil.copy(ns.import_study / "study-manifest.json", DATA / "v0.1.1-study-manifest.json")
if ns.import_fail_fast:
    shutil.copy(ns.import_fail_fast / "results.jsonl", FAIL_FAST)
    shutil.copy(
        ns.import_fail_fast / "study-manifest.json", DATA / "v0.1.1-fail-fast-manifest.json"
    )

val = Path("docs/validation.md")
text = val.read_text()

# -- v0.1.0 (historical, unchanged data) --------------------------------------------------------
text = splice(
    text, "TABLES", to_markdown(summarize(load_results(DATA / "validation-results.jsonl")), "####")
)
v1 = summarize(load_results(DATA / "validation-v1-results.jsonl"))
text = splice(text, "TABLES_V1", to_markdown(v1, "####"))
srows = load_results(DATA / "validation-sampler-v3-results.jsonl")
text = splice(text, "TABLES_SAMPLER_V3", to_sampler_table(srows))

# -- v0.1.1 ----------------------------------------------------------------------------------
readme_body = None
if STUDY.exists():
    rows = load_results(STUDY)
    s = study_summary(rows)
    ff_rows = load_results(FAIL_FAST)
    measured = {
        name: statistics.median(r["requests"] for r in ff_rows if r["scenario"] == name)
        for name in {r["scenario"] for r in ff_rows}
    }
    text = splice(text, "STUDY_CONTROLS", study_control_table(s))
    text = splice(text, "STUDY_FAULTS", study_fault_table(s, measured))
    if ff_rows:
        text = splice(text, "STUDY_FAIL_FAST", fail_fast_table(ff_rows, s))
    manifest = json.loads((DATA / "v0.1.1-study-manifest.json").read_text())
    git = manifest.get("git") or {}
    readme_body = (
        "## Measured results\n\n"
        f"v0.1.1 study: Qwen3-0.6B served by llama.cpp b12128 on a laptop (RX 6500M GPU and "
        f"CPU), {s['control_runs']} runs of {s['control_deployments']} correct deployment "
        f"configurations and {sum(f['n'] for f in s['faults'])} runs of "
        f"{len({f['name'].removesuffix('-cpu') for f in s['faults']})} injected faults, code "
        "frozen at "
        f"`{git.get('commit', 'unknown')[:12]}`. Method, controls and caveats: "
        "[docs/validation.md](docs/validation.md).\n\n"
        + study_readme_table(s, measured)
        + "\n<sub>Requests to detection are medians of real `--fail-fast` runs; * marks a value "
        "inferred from full runs. Simulation: *real* is a real server setting, *request* a proxy "
        "rewriting requests, *response* a proxy corrupting responses.</sub>\n"
    )
if DEV_TOOLS.exists():
    dev = load_results(DEV_TOOLS)
    lines = [
        "| Development run | Tools probe | Deterministic | Trigger disagreements | Invalid where "
        "reference valid | tool_choice=none violations | JSON invalid (finish stop) | Requests |",
        "|---|---|:-:|---:|---:|---:|---:|---:|",
    ]
    for r in dev:
        m = r["metrics"]
        lines.append(
            f"| {r['fault']} | {r['status']} | {'yes' if r['hard_evidence'] else ''} "
            f"| {m.get('trigger_disagreements', 0):.0f}/{m.get('trigger_prompts', 0):.0f} "
            f"| {m.get('invalid_where_reference_valid', 0):.0f} "
            f"| {m.get('tool_choice_none_violations', 0):.0f} "
            f"| {m.get('json_invalid_with_stop', 0):.0f} | {r['requests']} |"
        )
    text = splice(text, "DEV_TOOLS", "\n".join(lines) + "\n")
if CONFIRM.exists():
    c = study_summary(load_results(CONFIRM))
    text = splice(text, "CONFIRM_V030", study_control_table(c) + "\n" + study_fault_table(c))
if OLLAMA.exists():
    orows = load_results(OLLAMA)
    lines = [
        "| Target | Kind | Expected | Runs | Verdicts | Evaluated probes | Flagged by |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for name in dict.fromkeys(r["scenario"] for r in orows):
        rs = [r for r in orows if r["scenario"] == name]
        verdicts = ", ".join(
            f"{v} {sum(r['verdict'] == v for r in rs)}"
            for v in ("PASS", "DEVIATION", "INCONCLUSIVE")
            if any(r["verdict"] == v for r in rs)
        )
        flagged = sorted(
            {p for r in rs for p, st in r["probe_status"].items() if st == "DEVIATION"}
        )
        evaluated = rs[0]["coverage"]["evaluated"] if rs[0].get("coverage") else []
        lines.append(
            f"| {name} | {rs[0]['kind']} | {rs[0]['expects']} | {len(rs)} | {verdicts} "
            f"| {len(evaluated)}/6 | {', '.join(flagged) or 'nothing'} |"
        )
    text = splice(text, "OLLAMA", "\n".join(lines) + "\n")
if SMOLLM2.exists():
    sm = study_summary(load_results(SMOLLM2))
    text = splice(text, "SMOLLM2", study_control_table(sm) + "\n" + study_fault_table(sm))
val.write_text(text)

if readme_body is not None:
    readme = Path("README.md")
    readme.write_text(splice(readme.read_text(), "RESULTS", readme_body))
print("updated docs/validation.md" + (" and README.md" if readme_body else ""))
