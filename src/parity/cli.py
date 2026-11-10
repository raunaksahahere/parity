"""Command line interface. Exit codes: 0 pass, 1 deviation detected, 2 inconclusive or error."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer

from parity import __version__
from parity.history import History
from parity.manifest import ResumeError
from parity.probes.base import ProbeResult
from parity.render import to_html, to_markdown, write_reports
from parity.report_model import ReferenceInfo, RunReport
from parity.stats.calibration import (
    FLOORS,
    Calibration,
    CalibrationError,
    benign_values,
    calibrate_threshold,
)
from parity.suite import (
    PROBE_CLASSES,
    SUITES,
    Reference,
    Suite,
    Target,
    calibration_context,
    hf_reference,
    sanitize_url,
)
from parity.suites import DEFAULT_SUITE
from parity.verdict import coverage_line

CALIBRATION_PROBES = ["distribution.teacher_forced", "greedy.agreement", "tools.structured"]

app = typer.Typer(
    help="Statistical conformance testing for LLM inference endpoints.", no_args_is_help=True
)

ERROR_EXIT = 2


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(ERROR_EXIT)


def _build_reference(
    model: str, reference: str, revision: str | None, dtype: str, cache_dir: Path,
    reference_model: str | None, api_key_env: str | None,
) -> Reference:  # fmt: skip
    from parity.adapters import get_adapter
    from parity.client import EndpointClient
    from parity.detect import detect_adapter
    from parity.endpoint_reference import EndpointReference
    from parity.tokenization import TokenizerKit

    if reference == "hf:local":
        return hf_reference(model, revision=revision, dtype=dtype, cache_dir=cache_dir)
    if reference.startswith(("http://", "https://")):
        kit = TokenizerKit.from_pretrained(model)

        async def pick() -> str:
            async with EndpointClient(reference, api_key_env=api_key_env) as c:
                return await detect_adapter(c)

        name = asyncio.run(pick())
        ref_impl = EndpointReference(
            lambda: EndpointClient(reference, api_key_env=api_key_env),
            get_adapter(name),
            reference_model or model,
        )
        info = ReferenceInfo(
            kind="url", model_id=reference_model or model, revision=revision or "n/a"
        )
        return Reference(kit, ref_impl, info)
    raise _fail(f"--reference must be 'hf:local' or an http(s) URL, got {reference!r}")


def _to_json(r: RunReport) -> str:
    return r.model_dump_json(indent=1)


def _print_result(r: ProbeResult) -> None:
    effect = "" if r.effect is None else f" effect={r.effect:.4g}"
    typer.echo(f"  {r.probe:30s} {r.status.value:12s} requests={r.n_requests}{effect}")


@app.command()
def run(
    model: Annotated[
        str, typer.Option(help="HF model id or local directory (tokenizer/reference)")
    ],
    target: Annotated[str, typer.Option(help="Base URL of the endpoint under test")],
    reference: Annotated[
        str, typer.Option(help="'hf:local' or the URL of a reference endpoint")
    ] = "hf:local",
    suite: Annotated[str, typer.Option(help=f"One of {sorted(SUITES)}")] = "core",
    budget: Annotated[
        int | None, typer.Option(help="Maximum number of requests to the target")
    ] = None,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("parity-runs"),
    served_model: Annotated[str | None, typer.Option(help="Model name sent to the target")] = None,
    revision: Annotated[
        str | None, typer.Option(help="Pinned reference revision (commit hash)")
    ] = None,
    adapter: Annotated[str, typer.Option(help="auto, llamacpp, vllm or openai")] = "auto",
    api_key_env: Annotated[
        str | None, typer.Option(help="Env var holding the target API key")
    ] = None,
    reference_model: Annotated[
        str | None, typer.Option(help="Model name for a URL reference")
    ] = None,
    calibration: Annotated[
        Path | None, typer.Option(help="calibration.json from `parity calibrate`")
    ] = None,
    alpha: Annotated[float, typer.Option(help="Familywise error rate")] = 0.05,
    probes: Annotated[list[str] | None, typer.Option(help="Run only these probes")] = None,
    concurrency: Annotated[int, typer.Option(help="Concurrent requests to the target")] = 4,
    rate_per_s: Annotated[float | None, typer.Option(help="Requests per second limit")] = None,
    claimed_context: Annotated[int, typer.Option(help="Context window the endpoint claims")] = 4096,
    dtype: Annotated[str, typer.Option(help="Reference dtype")] = "float32",
    resume: Annotated[str | None, typer.Option(help="Resume the run with this id")] = None,
    history: Annotated[Path | None, typer.Option(help="DuckDB history file")] = None,
    cache_dir: Annotated[Path, typer.Option(help="Reference cache directory")] = Path(
        ".parity-cache"
    ),
    seed: int = 0,
    fail_fast: Annotated[bool, typer.Option(help="Stop at the first deviation")] = False,
    require_probe: Annotated[
        list[str] | None,
        typer.Option(
            help="Probe that must reach PASS or DEVIATION (id, family name or 'all'; "
            "repeatable). If one is skipped or inconclusive, a PASS becomes INCONCLUSIVE."
        ),
    ] = None,
    allow_unverified_pass: Annotated[
        bool,
        typer.Option(
            help="Accept a PASS even if no model-identity probe (distribution, greedy) could "
            "reach a decision. Without this such a run is INCONCLUSIVE."
        ),
    ] = False,
) -> None:
    """Compare an endpoint with a reference implementation of the same model."""
    try:
        ref = _build_reference(
            model, reference, revision, dtype, cache_dir, reference_model, api_key_env
        )
        cal = Calibration.load(calibration) if calibration else None
        s = Suite(
            name=suite, alpha=alpha, budget=budget, seed=seed, probes=probes,
            options={"claimed_context": claimed_context}, calibration=cal,
            calibration_path=str(calibration or ""), fail_fast=fail_fast, require=require_probe,
            allow_unverified_pass=allow_unverified_pass,
        )  # fmt: skip
        tgt = Target(target, served_model or model, adapter, api_key_env, concurrency, rate_per_s)
        s.check_calibration(ref)  # refuse an incompatible calibration before any request
        typer.echo(f"Running suite '{suite}' against {target} (reference {ref.info.kind})")
        report = s.run(
            tgt, ref, out_dir=out, run_id=resume, resume=resume is not None, on_probe=_print_result
        )
    except typer.Exit:
        raise
    except (CalibrationError, ResumeError) as exc:
        raise _fail(str(exc)) from exc
    except Exception as exc:
        raise _fail(f"{type(exc).__name__}: {exc}") from exc

    paths = write_reports(report, out / report.run_id)
    db = History(history or out / "history.duckdb")
    db.record(report)
    db.close()
    if report.coverage is not None:
        typer.echo(f"\nCoverage: {coverage_line(report.coverage)}")
    typer.echo(f"\nVerdict: {report.verdict.value}. {report.summary}")
    typer.echo(f"Reports: {paths['html']}")
    raise typer.Exit(report.exit_code)


@app.command()
def report(
    source: Annotated[Path, typer.Argument(help="report.json or a run directory")],
    fmt: Annotated[str, typer.Option("--format", help="md, html or json")] = "md",
    out: Annotated[Path | None, typer.Option(help="Write here instead of stdout")] = None,
) -> None:
    """Re-render a stored report."""
    path = source / "report.json" if source.is_dir() else source
    try:
        rep = RunReport.model_validate_json(path.read_text())
    except Exception as exc:
        raise _fail(f"cannot read report {path}: {exc}") from exc
    renderers = {
        "md": to_markdown,
        "html": to_html,
        "json": _to_json,
    }
    text = renderers.get(fmt)
    if text is None:
        raise _fail("--format must be md, html or json")
    rendered = text(rep)
    if out:
        out.write_text(rendered)
        typer.echo(f"wrote {out}")
    else:
        sys.stdout.write(rendered)
    raise typer.Exit(rep.exit_code)


@app.command()
def calibrate(
    model: Annotated[str, typer.Option(help="HF model id or local directory")],
    benign: Annotated[
        list[str], typer.Option(help="LABEL=URL of a known-good deployment (repeatable)")
    ],
    out: Annotated[Path, typer.Option(help="Where to write calibration.json")] = Path(
        "calibration.json"
    ),
    reference: Annotated[str, typer.Option()] = "hf:local",
    revision: Annotated[str | None, typer.Option()] = None,
    served_model: Annotated[str | None, typer.Option()] = None,
    safety_factor: Annotated[
        float, typer.Option(help="Threshold = worst benign value x this")
    ] = 2.0,
    dtype: str = "float32",
    cache_dir: Path = Path(".parity-cache"),
) -> None:
    """Set thresholds from known-good deployments that differ in benign ways.

    Give several `--benign` deployments (different backend, concurrency, hardware): same-config
    repeats alone are often exactly zero and would make the tool over-flag honest deployments.
    """
    try:
        ref = _build_reference(model, reference, revision, dtype, cache_dir, None, None)
        values: dict[str, dict[str, float]] = {}
        deployments: dict[str, str] = {}
        s = Suite(probes=CALIBRATION_PROBES, options={"calibrating": True})
        for item in benign:
            label, _, url = item.partition("=")
            if not url:
                raise _fail(f"--benign expects LABEL=URL, got {item!r}")
            deployments[label] = sanitize_url(url)
            typer.echo(f"Calibrating on {label} ({url})")
            rep = s.run(Target(url, served_model or model), ref, on_probe=_print_result)
            for o in rep.probes:
                for metric, v in benign_values(o.result).items():
                    values.setdefault(metric, {})[label] = v
    except typer.Exit:
        raise
    except Exception as exc:
        raise _fail(f"{type(exc).__name__}: {exc}") from exc

    cal = Calibration(
        model_id=ref.info.model_id,
        revision=ref.info.revision,
        suite=DEFAULT_SUITE,
        context=calibration_context(ref, s.effective_options(), CALIBRATION_PROBES),
        benign_deployments=deployments,
        safety_factor=safety_factor,
    )
    for metric, per_label in values.items():
        cal.metrics[metric] = calibrate_threshold(
            per_label, safety_factor=safety_factor, floor=FLOORS.get(metric, 0.0)
        )
    cal.save(out)
    for metric, mc in cal.metrics.items():
        typer.echo(f"  {metric}: threshold {mc.threshold:.4g} from {mc.benign_values}")
    typer.echo(f"wrote {out}")


@app.command()
def doctor(
    target: Annotated[str, typer.Option(help="Base URL of the endpoint to examine")],
    served_model: Annotated[str, typer.Option(help="Model name sent to the endpoint")],
    adapter: Annotated[str, typer.Option(help="auto, llamacpp, vllm, ollama or openai")] = "auto",
    api_key_env: Annotated[
        str | None, typer.Option(help="Env var holding the endpoint API key")
    ] = None,
) -> None:
    """Check what an endpoint can be tested for, before running the suite (about 7 requests).

    Exit code 0 if a model-identity probe can run, 2 if none can (a run could not PASS).
    """
    from parity.doctor import diagnose

    try:
        d = asyncio.run(diagnose(target, served_model, adapter=adapter, api_key_env=api_key_env))
    except Exception as exc:
        raise _fail(f"{type(exc).__name__}: {exc}") from exc
    typer.echo(f"Endpoint {sanitize_url(target)} looks like: {d.adapter} ({d.requests} requests)")
    typer.echo("\nCapabilities")
    for c in d.checks:
        typer.echo(f"  {'yes' if c.ok else 'no ':3s}  {c.name:28s} {c.detail}")
    typer.echo("\nProbe coverage to expect")
    for pid, mode in d.probes.items():
        typer.echo(f"  {pid:30s} {mode}")
    if d.identity_checkable:
        typer.echo("\nA model-identity probe can run on this endpoint.")
        raise typer.Exit(0)
    typer.echo(
        "\nNo model-identity probe can run on this endpoint: `parity run` could describe the "
        "serving stack but would not return PASS."
    )
    raise typer.Exit(ERROR_EXIT)


@app.command("list-probes")
def list_probes() -> None:
    """List available probes."""
    needs = {
        "template.token_counts": "usage.prompt_tokens",
        "distribution.teacher_forced": "token-id prompts + top-k log-probs",
        "greedy.agreement": "token-id prompts",
        "sampler.conformance": "token-id prompts + top-k log-probs",
        "longctx.needle": "chat endpoint",
        "tools.structured": "chat endpoint with tools",
    }
    for pid, cls in PROBE_CLASSES.items():
        typer.echo(f"{pid:30s} family={cls.family:13s} v{cls.version}  needs: {needs[pid]}")
    typer.echo("\nsuites: " + ", ".join(f"{k} ({len(v)} probes)" for k, v in SUITES.items()))


@app.command()
def history(
    db: Annotated[Path, typer.Option(help="DuckDB history file")] = Path(
        "parity-runs/history.duckdb"
    ),
    model: Annotated[str | None, typer.Option()] = None,
    probe: Annotated[str | None, typer.Option(help="With --metric: print a trend")] = None,
    metric: Annotated[str | None, typer.Option()] = None,
    limit: int = 20,
    export_parquet: Annotated[
        Path | None, typer.Option(help="Export tables as Parquet here")
    ] = None,
) -> None:
    """Show past runs and metric trends."""
    if not db.exists():
        raise _fail(f"no history at {db}")
    h = History(db)
    if export_parquet:
        for p in h.export_parquet(export_parquet):
            typer.echo(f"wrote {p}")
    elif probe and metric:
        for ts, v in h.trend(probe, metric, model=model):
            typer.echo(f"{ts}  {v:.6g}")
    else:
        for r in h.runs(model=model, limit=limit):
            typer.echo(
                f"{r['started_at']}  {r['run_id']}  {r['verdict']:12s} "
                f"{r['target_model']} @ {r['target_url']}  ({r['requests_total']} requests)"
            )
    h.close()


@app.command()
def canary(
    model: Annotated[str, typer.Option(help="HF model id or local directory")],
    target: Annotated[str, typer.Option(help="Base URL of the endpoint under test")],
    interval: Annotated[float, typer.Option(help="Seconds between runs")] = 3600.0,
    port: Annotated[int, typer.Option(help="Port for the Prometheus /metrics endpoint")] = 9108,
    suite: str = "core",
    reference: str = "hf:local",
    revision: str | None = None,
    served_model: str | None = None,
    adapter: str = "auto",
    api_key_env: str | None = None,
    calibration: Path | None = None,
    budget: int | None = None,
    claimed_context: int = 4096,
    out: Path = Path("parity-runs"),
    max_runs: Annotated[
        int | None, typer.Option(help="Stop after N runs (default: forever)")
    ] = None,
    dtype: str = "float32",
    cache_dir: Path = Path(".parity-cache"),
    require_probe: Annotated[
        list[str] | None, typer.Option(help="Probe that must reach PASS or DEVIATION")
    ] = None,
) -> None:
    """Run the suite on a schedule and expose the latest result to Prometheus."""
    import time

    from parity.exporter import CanaryState, serve

    try:
        ref = _build_reference(model, reference, revision, dtype, cache_dir, None, api_key_env)
        cal = Calibration.load(calibration) if calibration else None
        Suite(
            name=suite,
            options={"claimed_context": claimed_context},
            calibration=cal,
            calibration_path=str(calibration or ""),
        ).check_calibration(ref)
    except CalibrationError as exc:
        raise _fail(str(exc)) from exc
    except Exception as exc:
        raise _fail(f"{type(exc).__name__}: {exc}") from exc
    state = CanaryState()
    serve(state, port)
    typer.echo(f"canary: metrics on :{port}/metrics, running every {interval:g}s")
    n = 0
    while max_runs is None or n < max_runs:
        started = time.monotonic()
        try:
            canary_suite = Suite(
                name=suite,
                budget=budget,
                options={"claimed_context": claimed_context},
                calibration=cal,
                calibration_path=str(calibration or ""),
                require=require_probe,
            )
            tgt = Target(target, served_model or model, adapter, api_key_env)
            rep = canary_suite.run(tgt, ref, out_dir=out)
            write_reports(rep, out / rep.run_id)
            db = History(out / "history.duckdb")
            db.record(rep)
            db.close()
            state.update(rep)
            typer.echo(f"canary: {rep.verdict.value} ({rep.requests_total} requests)")
        except Exception as exc:
            state.update(None, error=True)
            typer.echo(f"canary: run failed: {type(exc).__name__}: {exc}", err=True)
        n += 1
        if max_runs is None or n < max_runs:
            time.sleep(max(0.0, interval - (time.monotonic() - started)))


@app.command(
    "inject-faults",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def inject_faults(ctx: typer.Context) -> None:
    """Run the fault-injection validation study (`list`, `run`, `summarize`)."""
    try:
        from parity.faults.cli import main as faults_main
    except ImportError as exc:
        raise _fail("the fault-injection harness is not available in this install") from exc
    raise typer.Exit(faults_main(list(ctx.args)))


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)
