from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from parity import Reference, Suite, Target
from parity.history import History
from parity.probes.base import Status
from parity.render import to_html, to_markdown, write_reports
from parity.report_model import SCHEMA_VERSION, ReferenceInfo, RunReport
from parity.stats.calibration import Calibration, calibrate_threshold
from parity.suite import sanitize_url
from parity.tokenization import TokenizerKit
from tests.fake_endpoint import FakeEndpoint, SuiteReference
from tests.fake_lm import FakeLM

FAST: dict[str, Any] = {
    "sampler_looks": [60, 150],
    "claimed_context": 1024,
    "longctx_lengths": [256, 1024],
    "template_kwargs": {},
}
INFO = ReferenceInfo(kind="hf:local", model_id="toy", revision="abc123", dtype="float32")


def make(kit: TokenizerKit, ep: FakeEndpoint, **suite_kw: Any) -> tuple[Suite, Target, Reference]:
    suite = Suite(options={**FAST, **suite_kw.pop("options", {})}, **suite_kw)
    target = Target("http://user:secret@host:8080/v1?token=abc", "m", transport=ep.transport())
    return suite, target, Reference(kit, SuiteReference(kit), INFO)


def by_probe(report: RunReport) -> dict[str, Status]:
    return {o.result.probe: o.final_status for o in report.probes}


async def test_identical_deployment_passes_everything(toy_kit: TokenizerKit) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, seed=1))
    rep = await suite.arun(target, ref)
    assert rep.verdict is Status.PASS and rep.exit_code == 0, rep.summary
    assert set(by_probe(rep).values()) == {Status.PASS}
    assert rep.target.adapter == "llamacpp"  # auto-detected from /props
    assert rep.requests_total == sum(o.result.n_requests for o in rep.probes) > 0


async def test_template_fault_is_a_deviation_with_exit_code_1(toy_kit: TokenizerKit) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, extra_prompt_tokens=1, seed=2))
    rep = await suite.arun(target, ref)
    assert rep.verdict is Status.DEVIATION and rep.exit_code == 1
    assert by_probe(rep)["template.token_counts"] is Status.DEVIATION


async def test_sampler_fault_is_a_deviation(toy_kit: TokenizerKit) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, ignore_top_k=True, seed=3))
    rep = await suite.arun(target, ref)
    assert by_probe(rep)["sampler.conformance"] is Status.DEVIATION
    assert rep.exit_code == 1


async def test_different_model_is_detected_by_distribution_and_greedy(
    toy_kit: TokenizerKit,
) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, lm=FakeLM(noise=0.6), seed=4))
    rep = await suite.arun(target, ref)
    st = by_probe(rep)
    assert st["distribution.teacher_forced"] is Status.DEVIATION
    assert st["greedy.agreement"] is Status.DEVIATION
    assert rep.verdict is Status.DEVIATION


async def test_report_never_contains_credentials(toy_kit: TokenizerKit, tmp_path: Path) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, seed=5))
    rep = await suite.arun(target, ref, out_dir=tmp_path)
    blob = rep.model_dump_json() + to_html(rep) + to_markdown(rep)
    assert "secret" not in blob and "token=abc" not in blob and "user:" not in blob
    assert rep.target.base_url == "http://host:8080/v1"


async def test_resume_skips_completed_probes(toy_kit: TokenizerKit, tmp_path: Path) -> None:
    ep = FakeEndpoint(toy_kit, seed=6)
    suite, target, ref = make(toy_kit, ep, probes=["template.token_counts", "tools.structured"])
    first = await suite.arun(target, ref, out_dir=tmp_path, run_id="r1")
    calls_after_first = ep.calls
    again = await suite.arun(target, ref, out_dir=tmp_path, run_id="r1", resume=True)
    assert ep.calls - calls_after_first <= 1  # only the /props detection call
    assert [o.result.model_dump() for o in again.probes] == [
        o.result.model_dump() for o in first.probes
    ]


async def test_budget_exhaustion_is_inconclusive_not_pass(toy_kit: TokenizerKit) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, seed=7), budget=40)
    rep = await suite.arun(target, ref)
    assert rep.verdict is not Status.PASS
    assert rep.requests_total <= 60


async def test_probe_crash_is_isolated(
    toy_kit: TokenizerKit, monkeypatch: pytest.MonkeyPatch
) -> None:
    from parity.probes.tools import ToolsProbe

    async def boom(self: Any, ctx: Any) -> Any:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(ToolsProbe, "run", boom)
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, seed=8))
    rep = await suite.arun(target, ref)
    assert by_probe(rep)["tools.structured"] is Status.INCONCLUSIVE
    assert any("kaboom" in f.message for o in rep.probes for f in o.result.findings)
    assert rep.verdict is Status.INCONCLUSIVE and rep.exit_code == 2


async def test_calibration_thresholds_reach_probes(toy_kit: TokenizerKit) -> None:
    cal = Calibration(model_id="toy", revision="abc123", suite="core-v1")
    cal.metrics["kl_mean"] = calibrate_threshold({"benign": 1e-4})
    suite, target, ref = make(
        toy_kit, FakeEndpoint(toy_kit, seed=9), calibration=cal, calibration_path="cal.json"
    )
    rep = await suite.arun(target, ref)
    assert rep.calibration is not None and rep.calibration.thresholds["kl_mean"] == pytest.approx(
        2e-4
    )
    dist = next(o.result for o in rep.probes if o.result.probe == "distribution.teacher_forced")
    assert dist.threshold == pytest.approx(2e-4)


async def test_history_records_and_reports_trends(toy_kit: TokenizerKit, tmp_path: Path) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, seed=10))
    h = History(tmp_path / "h.duckdb")
    for rid in ("a", "b"):
        h.record(await suite.arun(target, ref, run_id=rid))
    h.record(await suite.arun(target, ref, run_id="b"))  # re-recording replaces, no duplicates
    assert len(h.runs()) == 2
    series = h.trend("distribution.teacher_forced", "kl_lower_bound_mean", model="m")
    assert len(series) == 2 and all(v >= 0 for _, v in series)
    files = h.export_parquet(tmp_path / "pq")
    assert all(p.exists() and p.stat().st_size > 0 for p in files)
    import duckdb

    count = duckdb.connect().execute(f"SELECT count(*) FROM '{tmp_path / 'pq' / 'runs.parquet'}'")
    assert count.fetchall() == [(2,)]
    h.close()


async def test_reports_render_in_all_formats(toy_kit: TokenizerKit, tmp_path: Path) -> None:
    suite, target, ref = make(toy_kit, FakeEndpoint(toy_kit, extra_prompt_tokens=1, seed=11))
    rep = await suite.arun(target, ref)
    paths = write_reports(rep, tmp_path)
    html, md = paths["html"].read_text(), paths["md"].read_text()
    for text in (html, md):
        assert "What this does not prove" in text and "DEVIATION" in text
    assert RunReport.model_validate_json(paths["json"].read_text()).run_id == rep.run_id
    assert "<script" not in html  # static, no external resources
    assert "http://" not in html.replace("http://host:8080/v1", "")


def test_json_schema_is_stable() -> None:
    """Changing the report schema must be deliberate: update docs/report-schema.json and bump
    SCHEMA_VERSION if the change is not backwards compatible."""
    committed = Path(__file__).parent.parent / "docs" / "report-schema.json"
    current = RunReport.model_json_schema()
    assert json.loads(committed.read_text()) == current
    assert SCHEMA_VERSION == 1


def test_sanitize_url() -> None:
    assert (
        sanitize_url("https://k:v@api.example.com:9/x/y?key=1#f") == "https://api.example.com:9/x/y"
    )


def test_hf_reference_resolves_a_local_directory_without_loading_weights(
    toy_kit: TokenizerKit, tmp_path: Path
) -> None:
    from parity import hf_reference

    toy_kit.tokenizer.save_pretrained(tmp_path)
    ref = hf_reference(str(tmp_path), revision="deadbeef", cache_dir=tmp_path / "cache")
    assert ref.info.revision == "deadbeef" and ref.info.kind == "hf:local"
    assert ref.kit.decode(ref.kit.encode("hello world")) == "hello world"
