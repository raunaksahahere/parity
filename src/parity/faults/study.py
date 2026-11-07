"""Run the validation study: calibrate on benign deployments, then run every scenario.

Results are appended to ``results.jsonl`` as each repetition finishes, so an interrupted
study resumes where it stopped. Nothing is simulated: every number comes from a run of the
real suite against a real llama.cpp server (plus the fault proxy for simulated bugs).

Every row carries the fingerprint of the study configuration (code, reference, calibration,
probes, seeds, llama.cpp build) and of its scenario (server settings, GGUF file, mutation).
A study directory whose saved rows were produced under another configuration is refused, so
stale rows can never be mixed into the published tables.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parity.faults.proxy import FaultProxy
from parity.faults.resources import peak_rss_mb, reset_peak_rss, wait_until_cool
from parity.faults.scenarios import CLAIMED_CONTEXT, FAMILIES, Paths, Scenario, by_name
from parity.fingerprint import code_fingerprint, digest, file_sha256, git_state
from parity.local.llamacpp import LlamaServer
from parity.probes.base import Status
from parity.report_model import RunReport
from parity.stats.calibration import (
    FLOORS,
    Calibration,
    benign_values,
    calibrate_threshold,
)
from parity.suite import (
    STATISTICAL_PROBES,
    SUITES,
    Reference,
    Suite,
    Target,
    calibration_context,
    hf_reference,
)

SERVED_MODEL = "qwen3"
CALIBRATION_PROBES = ["distribution.teacher_forced", "greedy.agreement", "tools.structured"]


class StudyMismatchError(RuntimeError):
    """Saved study rows were produced under a different configuration."""


@dataclass
class StudyConfig:
    out: Path
    hf_model: Path = Path("models/hf/Qwen3-0.6B")
    model_id: str = "Qwen/Qwen3-0.6B"
    revision: str = "c1899de289a04d12100db370d81485cdf75e47ca"
    paths: Paths = field(default_factory=Paths)
    family: str = "qwen3-0.6b"  # scenario catalogue (faults.scenarios.FAMILIES)
    scenarios: list[str] | None = None
    reps: int | None = None  # override every scenario's repetition count
    probes: list[str] | None = None  # run only these probes (default: the whole core suite)
    cooldown_s: float = 0.0  # idle time after each repetition (keeps a laptop from overheating)
    cool_to_c: float | None = None  # before each repetition, wait until sensors are this cool
    concurrency: int = 4  # concurrent requests to the target
    seed_base: int = 0  # repetition r runs with seed seed_base + r (fresh data: a new base)
    fail_fast: bool = False  # real --fail-fast runs
    safety_factor: float = 2.0
    cache_dir: Path = Path(".parity-cache")
    log: Callable[[str], None] = print

    @property
    def results_file(self) -> Path:
        return self.out / "results.jsonl"

    @property
    def calibration_file(self) -> Path:
        return self.out / "calibration.json"

    @property
    def manifest_file(self) -> Path:
        return self.out / "study-manifest.json"


def config_for_family(out: Path, family: str, **kw: Any) -> StudyConfig:
    """A StudyConfig whose reference, revision and GGUF files come from a model family."""
    fam = FAMILIES[family]
    return StudyConfig(
        out=out,
        family=family,
        hf_model=fam.hf_dir,
        model_id=fam.model_id,
        revision=fam.revision,
        paths=fam.paths(),
        **kw,
    )


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reference(cfg: StudyConfig) -> Reference:
    return hf_reference(
        str(cfg.hf_model), revision=cfg.revision, cache_dir=cfg.cache_dir, model_id=cfg.model_id
    )


def scenario_fingerprint(sc: Scenario, cfg: StudyConfig) -> str:
    """Server settings, mutation, GGUF content and template file of one scenario."""
    paths = cfg.paths
    template = (
        file_sha256(paths.templates_dir / sc.server.template_file)
        if sc.server.template_file
        else None
    )
    gguf = file_sha256(paths.model(sc.server.quant), cfg.cache_dir)
    return digest("scenario", sc.key, gguf, template)


def study_identity(cfg: StudyConfig, reference: Reference, cal: Calibration) -> dict[str, Any]:
    builds = {
        b: file_sha256(cfg.paths.binary(b == "gpu"), cfg.cache_dir)
        for b in ("gpu", "cpu")
        if cfg.paths.binary(b == "gpu").exists()
    }
    return {
        "family": cfg.family,
        "parity_code": code_fingerprint(),
        "reference": reference.info.model_dump(mode="json"),
        "calibration": cal.fingerprint(),
        "probes": cfg.probes,
        "alpha": subset_alpha(cfg.probes),
        "seed_base": cfg.seed_base,
        "fail_fast": cfg.fail_fast,
        "claimed_context": CLAIMED_CONTEXT,
        "llama_server_sha256": builds,
    }


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

    @property
    def pid(self) -> int | None:
        return self.server.pid if self.server is not None else None

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.server, self.key = None, None


def _calibration_suite() -> Suite:
    return Suite(
        probes=CALIBRATION_PROBES,
        options={"calibrating": True, "claimed_context": CLAIMED_CONTEXT},
    )


def calibrate(cfg: StudyConfig, reference: Reference) -> Calibration:
    values: dict[str, dict[str, float]] = {}
    benign = [s for s in by_name(None, cfg.family) if s.kind == "benign"]
    dep = _Deployment(cfg)
    try:
        for sc in benign:
            url, _ = dep.url_for(sc)
            cfg.log(f"calibrating on {sc.name}")
            rep = _calibration_suite().run(Target(url, SERVED_MODEL), reference)
            for o in rep.probes:
                for metric, v in benign_values(o.result).items():
                    values.setdefault(metric, {})[sc.name] = v
    finally:
        dep.close()
    suite = _calibration_suite()
    cal = Calibration(
        model_id=cfg.model_id,
        revision=cfg.revision,
        suite=str(suite.effective_options()["suite"]),
        context=calibration_context(reference, suite.effective_options(), CALIBRATION_PROBES),
        benign_deployments={s.name: s.description for s in benign},
        safety_factor=cfg.safety_factor,
    )
    for metric, per in values.items():
        cal.metrics[metric] = calibrate_threshold(
            per, safety_factor=cfg.safety_factor, floor=FLOORS.get(metric, 0.0)
        )
    cal.save(cfg.calibration_file)
    return cal


def _check_saved_rows(
    rows: list[dict[str, Any]], study_fp: str, scenario_fps: dict[str, str]
) -> None:
    problems: list[str] = []
    for name in dict.fromkeys(r["scenario"] for r in rows):
        mine = [r for r in rows if r["scenario"] == name]
        if any(r.get("study_fingerprint") != study_fp for r in mine):
            problems.append(
                f"{name}: rows were produced under another study configuration "
                f"(code, reference, calibration, probes, seeds or llama.cpp build)"
            )
        elif name in scenario_fps and any(
            r.get("scenario_fingerprint") != scenario_fps[name] for r in mine
        ):
            problems.append(f"{name}: rows were produced with different scenario settings")
    if problems:
        raise StudyMismatchError(
            "the study directory holds results that do not match this configuration:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nUse a new --out directory for a new configuration."
        )


def run_study(cfg: StudyConfig) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    reference = _reference(cfg)
    if cfg.calibration_file.exists():
        cal = Calibration.load(cfg.calibration_file)
        run_probes = cfg.probes or SUITES["core"]
        cal.check(
            calibration_context(reference, Suite().effective_options(), run_probes),
            run_probes,
            str(cfg.calibration_file),
        )
    else:
        cal = calibrate(cfg, reference)
    cfg.log(f"thresholds: {cal.thresholds()}")
    names = cfg.scenarios
    scenarios = [
        s for s in by_name(names) if s.kind in ("control", "fault") or (names and s.kind == "dev")
    ]
    identity = study_identity(cfg, reference, cal)
    study_fp = digest("study", identity)
    scenario_fps = {s.name: scenario_fingerprint(s, cfg) for s in scenarios}
    rows = load_results(cfg.results_file)
    _check_saved_rows(rows, study_fp, scenario_fps)
    saved_manifest: dict[str, Any] = (
        json.loads(cfg.manifest_file.read_text()) if cfg.manifest_file.exists() else {}
    )
    manifest = {
        "study_fingerprint": study_fp,
        "identity": identity,
        "git": git_state(),
        "scenarios": {
            **saved_manifest.get("scenarios", {}),
            **{
                s.name: {"fingerprint": scenario_fps[s.name], "key": s.key, "reps": s.reps}
                for s in scenarios
            },
        },
    }
    cfg.manifest_file.write_text(json.dumps(manifest, indent=1))
    done = {(r["scenario"], r["rep"]) for r in rows}
    dep = _Deployment(cfg)
    try:
        for sc in sorted(scenarios, key=lambda s: s.server.key):  # group by server to reuse it
            for rep in range(cfg.reps if cfg.reps is not None else sc.reps):
                if (sc.name, rep) in done:
                    continue
                url, proxy = dep.url_for(sc)
                if cfg.cool_to_c is not None:
                    wait_until_cool(cfg.cool_to_c, cfg.log)
                reset_peak_rss()
                if dep.pid is not None:
                    reset_peak_rss(dep.pid)
                try:
                    suite = Suite(
                        probes=cfg.probes,
                        alpha=subset_alpha(cfg.probes),
                        seed=cfg.seed_base + rep,
                        options={"claimed_context": CLAIMED_CONTEXT},
                        calibration=cal,
                        calibration_path=str(cfg.calibration_file),
                        fail_fast=cfg.fail_fast,
                    )
                    report = suite.run(
                        Target(url, SERVED_MODEL, concurrency=cfg.concurrency),
                        reference,
                        out_dir=cfg.out / sc.name / f"rep{rep}",
                    )
                finally:
                    if proxy is not None:
                        proxy.stop()
                row = summarize_run(sc, rep, report)
                row.update(
                    study_fingerprint=study_fp,
                    scenario_fingerprint=scenario_fps[sc.name],
                    seed=cfg.seed_base + rep,
                    fail_fast=cfg.fail_fast,
                    concurrency=cfg.concurrency,
                    parity_peak_rss_mb=peak_rss_mb(),
                    server_peak_rss_mb=peak_rss_mb(dep.pid) if dep.pid else None,
                )
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
        "backend": sc.backend,
        "simulation": sc.simulation,
        "fresh": sc.fresh,
        "rep": rep,
        "run_id": report.run_id,
        "run_fingerprint": report.run_fingerprint,
        "verdict": report.verdict.value,
        "requests": report.requests_total,
        "requests_consumed": report.requests_consumed,
        "seconds": report.duration_s,
        "probe_status": {o.result.probe: o.final_status.value for o in report.probes},
        "probe_requests": {o.result.probe: o.result.n_requests for o in report.probes},
        "probe_effect": {o.result.probe: o.result.effect for o in report.probes},
        "probe_seconds": {o.result.probe: o.result.duration_s for o in report.probes},
        "probe_p": {o.result.probe: o.adjusted_p for o in report.probes},
    }


__all__ = [
    "Status",
    "StudyConfig",
    "StudyMismatchError",
    "calibrate",
    "load_results",
    "run_study",
    "summarize_run",
]
