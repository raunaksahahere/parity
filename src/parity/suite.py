"""Suite runner: executes probes against a target, applies the verdict logic, checkpoints."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from parity import __version__
from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.detect import detect_adapter
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, ReferenceLike, Status
from parity.probes.distribution import DistributionProbe
from parity.probes.greedy import GreedyProbe
from parity.probes.longctx import LongContextProbe
from parity.probes.sampler import SamplerProbe
from parity.probes.template import TemplateProbe
from parity.probes.tools import ToolsProbe
from parity.report_model import CalibrationInfo, ReferenceInfo, RunReport, TargetInfo
from parity.stats.calibration import Calibration
from parity.tokenization import TokenizerKit
from parity.verdict import EXIT_CODES, decide, summarize

PROBE_CLASSES: dict[str, type[Probe]] = {
    c.id: c
    for c in (
        TemplateProbe,
        DistributionProbe,
        GreedyProbe,
        SamplerProbe,
        LongContextProbe,
        ToolsProbe,
    )
}
STATISTICAL_PROBES = frozenset(
    {"distribution.teacher_forced", "greedy.agreement", "sampler.conformance", "tools.structured"}
)
FAIL_FAST_NOTE = "not run: --fail-fast stopped the suite after a deviation"
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
) -> Reference:
    """Build the local Hugging Face reference for ``model`` (a hub id or a local directory).

    Hub ids are downloaded at ``revision`` (default: the current commit, which is recorded in
    the report); reference results are cached on disk under that revision.
    """
    from parity.reference import build_hf_reference

    hf = build_hf_reference(model, revision=revision, dtype=dtype, cache_dir=cache_dir)
    info = ReferenceInfo(kind="hf:local", model_id=model, revision=hf.revision, dtype=dtype)
    return Reference(hf.kit, hf, info)


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

    def probe_ids(self) -> list[str]:
        ids = self.probes or SUITES[self.name]
        unknown = [i for i in ids if i not in PROBE_CLASSES]
        if unknown:
            raise ValueError(f"unknown probes {unknown}; available: {sorted(PROBE_CLASSES)}")
        return ids

    def run(self, target: Target, reference: Reference, **kwargs: Any) -> RunReport:
        import asyncio

        return asyncio.run(self.arun(target, reference, **kwargs))

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
        if run_dir:
            (run_dir / "probes").mkdir(parents=True, exist_ok=True)

        async with EndpointClient(
            target.base_url,
            api_key_env=target.api_key_env,
            max_concurrency=target.concurrency,
            rate_per_s=target.rate_per_s,
            transport=target.transport,
        ) as client:
            adapter_name = target.adapter
            if adapter_name == "auto":
                adapter_name = await detect_adapter(client)
            adapter = get_adapter(adapter_name)
            options: dict[str, Any] = {
                "suite": "core-v1",
                "thresholds": self.calibration.thresholds() if self.calibration else {},
                **self.options,
            }
            ctx = ProbeContext(
                client=client,
                adapter=adapter,
                model=target.model,
                kit=reference.kit,
                reference=reference.impl,
                # Bonferroni share per statistical probe, so a probe-level DEVIATION always
                # survives the Holm step (adjusted p <= m * p <= alpha).
                alpha=self.alpha / max(1, sum(i in STATISTICAL_PROBES for i in self.probe_ids())),
                seed=self.seed,
                budget=self.budget,
                options=options,
            )
            results: list[ProbeResult] = []
            for pid in self.probe_ids():
                if self.fail_fast and any(r.status is Status.DEVIATION for r in results):
                    results.append(
                        ProbeResult(
                            probe=pid,
                            family=PROBE_CLASSES[pid].family,
                            version=PROBE_CLASSES[pid].version,
                            status=Status.SKIPPED,
                            notes=[FAIL_FAST_NOTE],
                        )
                    )
                    continue
                saved = run_dir / "probes" / f"{pid}.json" if run_dir else None
                if resume and saved is not None and saved.exists():
                    results.append(ProbeResult.model_validate_json(saved.read_text()))
                    if on_probe is not None:
                        on_probe(results[-1])
                    continue
                before = client.requests_sent
                try:
                    result = await PROBE_CLASSES[pid]().run(ctx)
                except Exception as exc:  # one broken probe must not sink the run
                    result = ProbeResult(
                        probe=pid,
                        family=PROBE_CLASSES[pid].family,
                        version=PROBE_CLASSES[pid].version,
                        status=Status.INCONCLUSIVE,
                        n_requests=client.requests_sent - before,
                        findings=[
                            Finding(
                                severity="warn",
                                message=f"probe crashed: {type(exc).__name__}: {exc}",
                            )
                        ],
                    )
                results.append(result)
                if on_probe is not None:
                    on_probe(result)
                if saved is not None:
                    saved.write_text(result.model_dump_json(indent=1))

        outcomes, overall, lower_power = decide(results, self.alpha)
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
            )
            if self.calibration
            else None,
            probes=outcomes,
            verdict=overall,
            exit_code=EXIT_CODES[overall],
            lower_power=lower_power,
            summary=summarize(outcomes, overall, lower_power),
        )
        if run_dir:
            (run_dir / "report.json").write_text(report.model_dump_json(indent=1))
        return report

    @staticmethod
    def _new_run_id(target: Target, started: datetime) -> str:
        h = hashlib.sha256(f"{sanitize_url(target.base_url)}{target.model}{started}".encode())
        return f"{started:%Y%m%dT%H%M%S}-{h.hexdigest()[:6]}"
