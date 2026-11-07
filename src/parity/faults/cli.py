"""``parity inject-faults``: run the validation study and summarize its results."""

from __future__ import annotations

import argparse
from pathlib import Path

from parity.faults.scenarios import SCENARIOS
from parity.faults.study import StudyConfig, load_results, run_study
from parity.faults.summary import summarize, to_markdown


def main(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="parity inject-faults", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list scenarios")
    run = sub.add_parser("run", help="run (and resume) the study")
    run.add_argument("--out", type=Path, default=Path("runs/validation"))
    run.add_argument("--scenarios", nargs="*", help="only these scenarios")
    run.add_argument("--reps", type=int, help="override repetitions per scenario")
    run.add_argument("--probes", nargs="*", help="run only these probes (default: core suite)")
    run.add_argument("--cooldown", type=float, default=0.0, help="idle seconds after each rep")
    run.add_argument("--concurrency", type=int, default=4, help="concurrent requests per run")
    run.add_argument("--hf-model", type=Path, default=Path("models/hf/Qwen3-0.6B"))
    run.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    summ = sub.add_parser("summarize", help="print result tables as Markdown")
    summ.add_argument("--out", type=Path, default=Path("runs/validation"))
    ns = ap.parse_args(args)

    if ns.cmd == "list":
        for s in SCENARIOS:
            print(f"{s.name:28s} {s.kind:8s} reps={s.reps:<3d} {s.description}")
        return 0
    if ns.cmd == "run":
        run_study(
            StudyConfig(
                out=ns.out,
                hf_model=ns.hf_model,
                revision=ns.revision,
                scenarios=ns.scenarios,
                reps=ns.reps,
                probes=ns.probes,
                cooldown_s=ns.cooldown,
                concurrency=ns.concurrency,
            )
        )
        return 0
    rows = load_results(ns.out / "results.jsonl")
    if not rows:
        print("no results yet")
        return 2
    print(to_markdown(summarize(rows)))
    return 0
