"""Run the validation study: calibrate on benign deployments, then run every scenario.

Results are appended to ``results.jsonl`` as each repetition finishes, so an interrupted
study resumes where it stopped. Nothing is simulated: every number comes from a run of the
real suite against a real llama.cpp server (plus the fault proxy for simulated bugs).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parity.faults.proxy import FaultProxy
from parity.faults.scenarios import CLAIMED_CONTEXT, Paths, Scenario, by_name
from parity.local.llamacpp import LlamaServer
from parity.probes.base import Status
from parity.report_model import ReferenceInfo, RunReport
from parity.stats.calibration import FLOORS, Calibration, benign_values, calibrate_threshold
from parity.suite import STATISTICAL_PROBES, SUITES, Reference, Suite, Target

SERVED_MODEL = "qwen3"
CALIBRATION_PROBES = ["distribution.teacher_forced", "greedy.agreement", "tools.structured"]


@dataclass
class StudyConfig:
    out: Path
    hf_model: Path = Path("models/hf/Qwen3-0.6B")
    revision: str = "c1899de289a04d12100db370d81485cdf75e47ca"
    paths: Paths = field(default_factory=Paths)
    scenarios: list[str] | None = None
    reps: int | None = None  # override every scenario's repetition count
    probes: list[str] | None = None  # run only these probes (default: the whole core suite)
    cooldown_s: float = 0.0  # idle time after each repetition (keeps a laptop from overheating)
    safety_factor: float = 2.0
    log: Callable[[str], None] = print

    @property
    def results_file(self) -> Path:
        return self.out / "results.jsonl"

    @property
    def calibration_file(self) -> Path:
        return self.out / "calibration.json"


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reference(cfg: StudyConfig) -> Reference:
    from parity.reference import build_hf_reference

    hf = build_hf_reference(str(cfg.hf_model), revision=cfg.revision)
    return Reference(
        hf.kit,
        hf,
        ReferenceInfo(
            kind="hf:local", model_id="Qwen/Qwen3-0.6B", revision=cfg.revision, dtype="float32"
        ),
    )


class _Deployment:
    """A scenario's server (shared between scenarios with the same spec) plus optional proxy."""

    def __init__(self, cfg: StudyConfig) -> None:
        self.cfg = cfg
        self.server: LlamaServer | None = None
        self.key: str | None = None

    def url_for(self, sc: Scenario) -> tuple[str, FaultProxy | None]:
        if self.key != sc.server.key:
            self.close()
            log = self.cfg.out / "logs" / f"{sc.name}.log"
            self.cfg.log(f"  starting server for {sc.name}: {sc.server}")
            self.server = LlamaServer(sc.server.config(self.cfg.paths, log)).start()
            self.key = sc.server.key
        assert self.server is not None
        if sc.mutation is None:
            return self.server.base_url, None
        proxy = FaultProxy(self.server.base_url, sc.mutation).start()
        return proxy.url, proxy

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.server, self.key = None, None


def calibrate(cfg: StudyConfig, reference: Reference) -> Calibration:
    values: dict[str, dict[str, float]] = {}
    dep = _Deployment(cfg)
    try:
        for sc in [s for s in by_name(None) if s.kind == "benign"]:
            url, _ = dep.url_for(sc)
            cfg.log(f"calibrating on {sc.name}")
            rep = Suite(
                probes=CALIBRATION_PROBES,
                options={"calibrating": True, "claimed_context": CLAIMED_CONTEXT},
            ).run(Target(url, SERVED_MODEL), reference)
            for o in rep.probes:
                for metric, v in benign_values(o.result).items():
                    values.setdefault(metric, {})[sc.name] = v
    finally:
        dep.close()
    cal = Calibration(model_id="Qwen/Qwen3-0.6B", revision=cfg.revision, suite="core-v1")
    for metric, per in values.items():
        cal.metrics[metric] = calibrate_threshold(
            per, safety_factor=cfg.safety_factor, floor=FLOORS.get(metric, 0.0)
        )
    cal.save(cfg.calibration_file)
    return cal


def run_study(cfg: StudyConfig) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    reference = _reference(cfg)
    cal = (
        Calibration.load(cfg.calibration_file)
        if cfg.calibration_file.exists()
        else calibrate(cfg, reference)
    )
    cfg.log(f"thresholds: {cal.thresholds()}")
    done = {(r["scenario"], r["rep"]) for r in load_results(cfg.results_file)}
    dep = _Deployment(cfg)
    try:
        scenarios = [s for s in by_name(cfg.scenarios) if s.kind != "benign"]
        for sc in sorted(scenarios, key=lambda s: s.server.key):  # group by server to reuse it
            for rep in range(cfg.reps if cfg.reps is not None else sc.reps):
                if (sc.name, rep) in done:
                    continue
                url, proxy = dep.url_for(sc)
                try:
                    suite = Suite(
                        probes=cfg.probes,
                        alpha=subset_alpha(cfg.probes),
                        seed=rep,
                        options={"claimed_context": CLAIMED_CONTEXT},
                        calibration=cal,
                        calibration_path=str(cfg.calibration_file),
                    )
                    report = suite.run(
                        Target(url, SERVED_MODEL, concurrency=4),
                        reference,
                        out_dir=cfg.out / sc.name / f"rep{rep}",
                    )
                finally:
                    if proxy is not None:
                        proxy.stop()
                row = summarize_run(sc, rep, report)
                with cfg.results_file.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if cfg.cooldown_s > 0:
                    time.sleep(cfg.cooldown_s)  # outside the timed run: no effect on results
                cfg.log(
                    f"{sc.name} rep{rep}: {row['verdict']} "
                    f"({row['requests']} requests, {row['seconds']:.0f}s) {row['probe_status']}"
                )
    finally:
        dep.close()


def subset_alpha(probes: list[str] | None, alpha: float = 0.05) -> float:
    """Familywise alpha for a probe subset that gives each statistical probe the same share
    (alpha / m_core) it has in the full core suite, so a subset run behaves like the full run."""
    if not probes:
        return alpha
    m_core = sum(p in STATISTICAL_PROBES for p in SUITES["core"])
    m_sub = sum(p in STATISTICAL_PROBES for p in probes)
    return alpha * max(1, m_sub) / m_core


def summarize_run(sc: Scenario, rep: int, report: RunReport) -> dict[str, Any]:
    return {
        "scenario": sc.name,
        "kind": sc.kind,
        "expects": sc.expects,
        "rep": rep,
        "run_id": report.run_id,
        "verdict": report.verdict.value,
        "requests": report.requests_total,
        "seconds": report.duration_s,
        "probe_status": {o.result.probe: o.final_status.value for o in report.probes},
        "probe_requests": {o.result.probe: o.result.n_requests for o in report.probes},
        "probe_effect": {o.result.probe: o.result.effect for o in report.probes},
    }


__all__ = ["Status", "StudyConfig", "calibrate", "load_results", "run_study", "summarize_run"]
