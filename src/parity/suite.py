"""Suite runner: executes probes against a target, applies the verdict logic, checkpoints."""

from __future__ import annotations

import hashlib
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from parity import __version__
from parity.adapters import get_adapter
from parity.client import BudgetExhaustedError, EndpointClient
from parity.detect import detect_adapter
from parity.fingerprint import (
    code_fingerprint,
    environment,
    git_state,
    model_fingerprint,
    suite_fingerprint,
    tokenizer_fingerprint,
)
from parity.manifest import (
    ProbeCheckpoint,
    ResumeError,
    RunManifest,
    canonical,
    file_digest,
    verify_resume,
)
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, ReferenceLike, Status
from parity.probes.distribution import DistributionProbe
from parity.probes.greedy import GreedyProbe
from parity.probes.longctx import LongContextProbe
from parity.probes.sampler import SamplerProbe
from parity.probes.template import TemplateProbe
from parity.probes.tools import ToolsProbe
from parity.report_model import CalibrationInfo, ReferenceInfo, RunReport, TargetInfo
from parity.stats.calibration import CALIBRATION_SETTINGS, Calibration, CalibrationContext
from parity.suites import DEFAULT_SUITE
from parity.tokenization import TokenizerKit
from parity.verdict import (
    EXIT_CODES,
    apply_identity_rule,
    apply_requirements,
    coverage,
    decide,
    identity_unverified,
    summarize,
)

PROBE_CLASSES: dict[str, type[Probe]] = {
    c.id: c
    for c in (
        # cheapest first, so --fail-fast reaches a verdict with as few requests as possible
        TemplateProbe,
        LongContextProbe,
        ToolsProbe,
        GreedyProbe,
        DistributionProbe,
        SamplerProbe,
    )
}
STATISTICAL_PROBES = frozenset(
    {"distribution.teacher_forced", "greedy.agreement", "sampler.conformance", "tools.structured"}
)
FAIL_FAST_NOTE = "not run: --fail-fast stopped the suite after a deviation"
BUDGET_NOTE = "not run: the request budget was used up"
SUITES = {"core": list(PROBE_CLASSES), "quick": ["template.token_counts", "tools.structured"]}


@dataclass(frozen=True)
class Target:
    base_url: str
    model: str
    adapter: str = "auto"
    api_key_env: str | None = None
    concurrency: int = 4
    rate_per_s: float | None = None
    transport: Any = field(default=None, repr=False, compare=False)  # tests only


@dataclass
class Reference:
    kit: TokenizerKit
    impl: ReferenceLike | None
    info: ReferenceInfo


def hf_reference(
    model: str,
    *,
    revision: str | None = None,
    dtype: str = "float32",
    cache_dir: Path = Path(".parity-cache"),
    model_id: str | None = None,
) -> Reference:
    """Build the local Hugging Face reference for ``model`` (a hub id or a local directory).

    Hub ids are downloaded at ``revision`` (default: the current commit, which is recorded in
    the report); reference results are cached on disk under that revision. The weights and the
    tokenizer are fingerprinted so calibrations and saved runs can be checked against them.
    ``model_id`` overrides the name shown in reports (e.g. the hub id of a local copy).
    """
    from parity.reference import build_hf_reference

    hf = build_hf_reference(model, revision=revision, dtype=dtype, cache_dir=cache_dir)
    info = ReferenceInfo(
        kind="hf:local",
        model_id=model_id or model,
        revision=hf.revision,
        dtype=dtype,
        model_fingerprint=model_fingerprint(Path(hf.model_path), cache_dir),
        tokenizer_fingerprint=tokenizer_fingerprint(hf.kit.tokenizer),
    )
    return Reference(hf.kit, hf, info)


def calibration_context(
    reference: Reference, options: dict[str, Any], probe_ids: list[str]
) -> CalibrationContext:
    """The context a calibration made (or used) under these conditions must carry."""
    info = reference.info
    if info.tokenizer_fingerprint is None:
        info.tokenizer_fingerprint = tokenizer_fingerprint(reference.kit.tokenizer)
    suite_data = str(options.get("suite", DEFAULT_SUITE))
    return CalibrationContext(
        model_id=info.model_id,
        revision=info.revision,
        model_fingerprint=info.model_fingerprint,
        tokenizer_fingerprint=info.tokenizer_fingerprint,
        reference_kind=info.kind,
        reference_dtype=info.dtype,
        suite_data=suite_data,
        suite_fingerprint=suite_fingerprint(suite_data),
        probe_versions={pid: PROBE_CLASSES[pid].version for pid in probe_ids},
        settings={k: canonical(options.get(k)) for k in CALIBRATION_SETTINGS},
        parity_version=__version__,
    )


def resolve_required(required: list[str] | None, probe_ids: list[str]) -> list[str]:
    """Expand ``all``, probe ids and family names (``sampler``) into probe ids."""
    families = {cls.family: pid for pid, cls in PROBE_CLASSES.items()}
    out: list[str] = []
    for item in required or []:
        ids = (
            list(probe_ids)
            if item == "all"
            else [item]
            if item in PROBE_CLASSES
            else [families[item]]
            if item in families
            else None
        )
        if ids is None:
            raise ValueError(
                f"unknown required probe {item!r}; use 'all', a probe id {sorted(PROBE_CLASSES)} "
                f"or a family {sorted(families)}"
            )
        out += [i for i in ids if i not in out]
    missing = [p for p in out if p not in probe_ids]
    if missing:
        raise ValueError(f"required probes {missing} are not part of this run's probe selection")
    return out


class _Ledger:
    """Keeps the manifest's request count current and saves it at most once a second, so an
    interrupted run still knows how many requests it consumed."""

    def __init__(self) -> None:
        self.manifest: RunManifest | None = None
        self.path: Path | None = None
        self.offset = 0
        self._last = 0.0

    def __call__(self, sent_this_session: int) -> None:
        if self.manifest is None:
            return
        self.manifest.requests_consumed = self.offset + sent_this_session
        now = time.monotonic()
        if self.path is not None and now - self._last >= 1.0:
            self.manifest.save(self.path)
            self._last = now

    def flush(self, sent_this_session: int) -> None:
        if self.manifest is not None:
            self.manifest.requests_consumed = self.offset + sent_this_session
            if self.path is not None:
                self.manifest.save(self.path)


def sanitize_url(url: str) -> str:
    """Drop credentials, query and fragment so reports never carry secrets."""
    p = urlsplit(url)
    host = p.hostname or ""
    if p.port:
        host = f"{host}:{p.port}"
    return urlunsplit((p.scheme, host, p.path, "", ""))


@dataclass
class Suite:
    name: str = "core"
    alpha: float = 0.05
    budget: int | None = None
    seed: int = 0
    probes: list[str] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    calibration: Calibration | None = None
    calibration_path: str = ""
    fail_fast: bool = False  # stop after the first DEVIATION (later probes are not run)
    # probes (ids, family names or "all") that must reach PASS or DEVIATION; otherwise a PASS
    # becomes INCONCLUSIVE
    require: list[str] | None = None
    # accept a PASS although no model-identity probe (distribution, greedy) reached a decision
    allow_unverified_pass: bool = False

    def probe_ids(self) -> list[str]:
        ids = self.probes or SUITES[self.name]
        unknown = [i for i in ids if i not in PROBE_CLASSES]
        if unknown:
            raise ValueError(f"unknown probes {unknown}; available: {sorted(PROBE_CLASSES)}")
        return ids

    def run(self, target: Target, reference: Reference, **kwargs: Any) -> RunReport:
        import asyncio

        return asyncio.run(self.arun(target, reference, **kwargs))

    def effective_options(self) -> dict[str, Any]:
        return {
            "suite": DEFAULT_SUITE,
            "thresholds": self.calibration.thresholds() if self.calibration else {},
            **self.options,
        }

    def check_calibration(self, reference: Reference) -> CalibrationContext:
        """Raise CalibrationError unless the calibration fits this run (no request is sent)."""
        ctx = calibration_context(reference, self.effective_options(), self.probe_ids())
        if self.calibration is not None:
            self.calibration.check(ctx, self.probe_ids(), self.calibration_path)
        return ctx

    async def arun(
        self,
        target: Target,
        reference: Reference,
        *,
        out_dir: Path | None = None,
        run_id: str | None = None,
        resume: bool = False,
        on_probe: Callable[[ProbeResult], None] | None = None,
    ) -> RunReport:
        started = datetime.now(UTC)
        t0 = time.monotonic()
        run_id = run_id or self._new_run_id(target, started)
        run_dir = (out_dir / run_id) if out_dir else None
        if resume and run_dir is None:
            raise ResumeError("resuming a run needs the output directory that holds it")
        probe_ids = self.probe_ids()
        required = resolve_required(self.require, probe_ids)
        options = self.effective_options()
        ref_ctx = self.check_calibration(reference)  # before any request is sent
        probe_versions = {pid: PROBE_CLASSES[pid].version for pid in probe_ids}
        ledger = _Ledger()

        async with EndpointClient(
            target.base_url,
            api_key_env=target.api_key_env,
            max_concurrency=target.concurrency,
            rate_per_s=target.rate_per_s,
            transport=target.transport,
            on_request=ledger,
        ) as client:
            adapter_name = target.adapter
            if adapter_name == "auto":
                adapter_name = await detect_adapter(client)
            adapter = get_adapter(adapter_name)
            identity = {
                "parity": {"version": __version__, "code": code_fingerprint()},
                "environment": environment(),
                "target": {
                    "base_url": sanitize_url(target.base_url),
                    "model": target.model,
                    "adapter": adapter_name,
                },
                "reference": {
                    "kind": ref_ctx.reference_kind,
                    "model_id": ref_ctx.model_id,
                    "revision": ref_ctx.revision,
                    "dtype": ref_ctx.reference_dtype,
                    "model_fingerprint": ref_ctx.model_fingerprint,
                    "tokenizer_fingerprint": ref_ctx.tokenizer_fingerprint,
                },
                "suite": {
                    "name": self.name,
                    "probes": probe_ids,
                    "probe_versions": probe_versions,
                    "data": ref_ctx.suite_data,
                    "data_fingerprint": ref_ctx.suite_fingerprint,
                    "required": required,
                },
                "statistics": {
                    "alpha": self.alpha,
                    "budget": self.budget,
                    "seed": self.seed,
                    "fail_fast": self.fail_fast,
                    "allow_unverified_pass": self.allow_unverified_pass,
                },
                "options": canonical({k: v for k, v in options.items() if k != "thresholds"}),
                "calibration": self.calibration.fingerprint() if self.calibration else None,
            }
            operational = {
                "concurrency": target.concurrency,
                "rate_per_s": target.rate_per_s,
                "platform": platform.platform(),
                "calibration_path": self.calibration_path,
                "git": git_state(),
            }
            manifest_path = run_dir / "manifest.json" if run_dir else None
            if resume:
                assert run_dir is not None
                manifest = verify_resume(run_dir, identity, probe_versions)
                manifest.sessions += 1
                manifest.operational = operational
            else:
                if run_dir is not None and (
                    (run_dir / "manifest.json").exists() or any(run_dir.glob("probes/*.json"))
                ):
                    raise ResumeError(
                        f"{run_dir} already holds results; resume it (--resume {run_id}) or "
                        "start a new run"
                    )
                manifest = RunManifest(run_id=run_id, identity=identity, operational=operational)
            if run_dir is not None:
                (run_dir / "probes").mkdir(parents=True, exist_ok=True)
            ledger.manifest, ledger.path = manifest, manifest_path
            ledger.offset = manifest.requests_consumed
            remaining = None if self.budget is None else max(0, self.budget - ledger.offset)
            client.request_limit = remaining
            ledger.flush(0)

            ctx = ProbeContext(
                client=client,
                adapter=adapter,
                model=target.model,
                kit=reference.kit,
                reference=reference.impl,
                # Bonferroni share per statistical probe, so a probe-level DEVIATION always
                # survives the Holm step (adjusted p <= m * p <= alpha).
                alpha=self.alpha / max(1, sum(i in STATISTICAL_PROBES for i in probe_ids)),
                seed=self.seed,
                budget=remaining,  # what this session may still send
                options=options,
            )
            results: list[ProbeResult] = []
            try:
                for pid in probe_ids:
                    result = await self._run_probe(
                        pid, ctx, client, results, run_dir, manifest, ledger, resume, remaining
                    )
                    results.append(result)
                    if on_probe is not None:
                        on_probe(result)
            finally:
                ledger.flush(client.requests_sent)

        outcomes, overall, lower_power = decide(results, self.alpha)
        cov = coverage(outcomes, required)
        overall = apply_requirements(overall, cov)
        no_identity = identity_unverified(outcomes)
        overall = apply_identity_rule(overall, outcomes, self.allow_unverified_pass)
        caps = adapter.capabilities
        report = RunReport(
            parity_version=__version__,
            run_id=run_id,
            started_at=started.isoformat(timespec="seconds"),
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
            duration_s=round(time.monotonic() - t0, 2),
            suite=self.name,
            alpha=self.alpha,
            budget=self.budget,
            requests_total=sum(r.n_requests for r in results),
            requests_consumed=manifest.requests_consumed,
            resumed=resume,
            run_fingerprint=manifest.fingerprint(),
            coverage=cov,
            target=TargetInfo(
                base_url=sanitize_url(target.base_url),
                model=target.model,
                adapter=adapter_name,
                capabilities={
                    "token_id_prompts": caps.token_id_prompts,
                    "prompt_logprobs": caps.prompt_logprobs,
                    "top_logprobs": caps.top_logprobs,
                    "max_top_k": caps.max_top_k,
                },
            ),
            reference=reference.info,
            calibration=CalibrationInfo(
                path=self.calibration_path,
                created=self.calibration.created,
                thresholds=self.calibration.thresholds(),
                fingerprint=self.calibration.fingerprint(),
            )
            if self.calibration
            else None,
            probes=outcomes,
            verdict=overall,
            exit_code=EXIT_CODES[overall],
            lower_power=lower_power,
            summary=summarize(
                outcomes,
                overall,
                lower_power,
                cov.unmet_required,
                no_identity and overall is not Status.DEVIATION,
            ),
        )
        if run_dir:
            (run_dir / "report.json").write_text(report.model_dump_json(indent=1))
        return report

    async def _run_probe(
        self,
        pid: str,
        ctx: ProbeContext,
        client: EndpointClient,
        done: list[ProbeResult],
        run_dir: Path | None,
        manifest: RunManifest,
        ledger: _Ledger,
        resume: bool,
        remaining: int | None,
    ) -> ProbeResult:
        cls = PROBE_CLASSES[pid]
        if self.fail_fast and any(r.status is Status.DEVIATION for r in done):
            return ProbeResult(
                probe=pid,
                family=cls.family,
                version=cls.version,
                status=Status.SKIPPED,
                notes=[FAIL_FAST_NOTE],
            )
        saved = run_dir / "probes" / f"{pid}.json" if run_dir else None
        if resume and saved is not None and pid in manifest.checkpoints:
            # verified against the manifest by verify_resume
            return ProbeResult.model_validate_json(saved.read_text())
        if remaining is not None and client.requests_sent >= remaining:
            return ProbeResult(
                probe=pid,
                family=cls.family,
                version=cls.version,
                status=Status.INCONCLUSIVE,
                notes=[BUDGET_NOTE],
                findings=[
                    Finding(
                        severity="warn",
                        message=f"not run: the request budget of {self.budget} was used up "
                        f"({ledger.offset + client.requests_sent} requests consumed, including "
                        "earlier sessions of this run)",
                    )
                ],
            )
        before, t = client.requests_sent, time.monotonic()
        crashed = False
        try:
            result = await cls().run(ctx)
        except BudgetExhaustedError as exc:
            crashed = True
            result = ProbeResult(
                probe=pid,
                family=cls.family,
                version=cls.version,
                status=Status.INCONCLUSIVE,
                n_requests=client.requests_sent - before,
                findings=[Finding(severity="warn", message=f"stopped: {exc}")],
            )
        except Exception as exc:  # one broken probe must not sink the run
            crashed = True
            result = ProbeResult(
                probe=pid,
                family=cls.family,
                version=cls.version,
                status=Status.INCONCLUSIVE,
                n_requests=client.requests_sent - before,
                findings=[
                    Finding(
                        severity="warn",
                        message=f"probe crashed: {type(exc).__name__}: {exc}",
                    )
                ],
            )
        result.duration_s = round(time.monotonic() - t, 2)
        if saved is not None and not crashed:
            # The manifest records the file's checksum before the file appears, so a crash in
            # between leaves a checkpoint without a file (the probe re-runs on resume), never
            # a file the manifest does not vouch for.
            text = result.model_dump_json(indent=1)
            tmp = saved.with_suffix(".tmp")
            tmp.write_text(text)
            manifest.checkpoints[pid] = ProbeCheckpoint(
                version=cls.version,
                status=result.status.value,
                sha256=file_digest(text),
                requests_consumed=client.requests_sent - before,
            )
            ledger.flush(client.requests_sent)
            tmp.replace(saved)
        return result

    @staticmethod
    def _new_run_id(target: Target, started: datetime) -> str:
        h = hashlib.sha256(f"{sanitize_url(target.base_url)}{target.model}{started}".encode())
        return f"{started:%Y%m%dT%H%M%S}-{h.hexdigest()[:6]}"
