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
    run.add_argument(
        "--cool-to", type=float, help="before each rep, wait until sensors are this cool (deg C)"
    )
    run.add_argument(
        "--seed-base", type=int, default=0, help="rep r uses seed SEED_BASE + r (fresh data)"
    )
    run.add_argument("--fail-fast", action="store_true", help="real --fail-fast runs")
    run.add_argument("--hf-model", type=Path, default=Path("models/hf/Qwen3-0.6B"))
    run.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    run.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="model name in reports")
    summ = sub.add_parser("summarize", help="print result tables as Markdown")
    summ.add_argument("--out", type=Path, default=Path("runs/validation"))
    ns = ap.parse_args(args)

    if ns.cmd == "list":
        for s in SCENARIOS:
            print(
                f"{s.name:32s} {s.kind:8s} {s.backend} {s.simulation:8s} reps={s.reps:<3d} "
                f"{s.description}"
            )
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
                cool_to_c=ns.cool_to,
                concurrency=ns.concurrency,
                seed_base=ns.seed_base,
                fail_fast=ns.fail_fast,
                model_id=ns.model_id,
            )
        )
        return 0
    rows = load_results(ns.out / "results.jsonl")
    if not rows:
        print("no results yet")
        return 2
    print(to_markdown(summarize(rows)))
    return 0
