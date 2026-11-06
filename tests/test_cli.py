from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from parity import Reference, Suite, Target
from parity.cli import app
from parity.history import History
from parity.render import write_reports
from parity.report_model import ReferenceInfo
from parity.tokenization import TokenizerKit
from tests.fake_endpoint import FakeEndpoint, SuiteReference

runner = CliRunner()
INFO = ReferenceInfo(kind="hf:local", model_id="toy", revision="abc")
FAST = {
    "sampler_looks": [60, 150],
    "claimed_context": 1024,
    "longctx_lengths": [256, 1024],
    "template_kwargs": {},
}


def test_list_probes_shows_all_six_families() -> None:
    res = runner.invoke(app, ["list-probes"])
    assert res.exit_code == 0
    for fam in ("template", "distribution", "greedy", "sampler", "longctx", "tools"):
        assert f"family={fam}" in res.stdout


def test_no_arguments_prints_help() -> None:
    res = runner.invoke(app, [])
    assert "run" in res.output and "calibrate" in res.output and "inject-faults" in res.output


async def test_report_command_re_renders_and_propagates_exit_code(
    toy_kit: TokenizerKit, tmp_path: Path
) -> None:
    ep = FakeEndpoint(toy_kit, extra_prompt_tokens=1, seed=1)
    rep = await Suite(options=FAST).arun(
        Target("http://h", "m", transport=ep.transport()),
        Reference(toy_kit, SuiteReference(toy_kit), INFO),
    )
    write_reports(rep, tmp_path / rep.run_id)
    md = runner.invoke(app, ["report", str(tmp_path / rep.run_id), "--format", "md"])
    assert md.exit_code == 1 and "What this does not prove" in md.stdout  # deviation -> 1
    html = runner.invoke(
        app, ["report", str(tmp_path / rep.run_id / "report.json"), "--format", "html"]
    )
    assert "<!doctype html>" in html.stdout
    bad = runner.invoke(app, ["report", str(tmp_path), "--format", "md"])
    assert bad.exit_code == 2


def test_unknown_format_and_missing_file_are_exit_2(tmp_path: Path) -> None:
    assert runner.invoke(app, ["report", str(tmp_path / "nope.json")]).exit_code == 2


def test_run_rejects_a_bad_reference_with_exit_2(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        ["run", "--model", "x", "--target", "http://127.0.0.1:1", "--reference", "ftp://nope",
         "--out", str(tmp_path)],
    )  # fmt: skip
    assert res.exit_code == 2 and "--reference" in res.output


def test_calibrate_requires_label_equals_url(tmp_path: Path) -> None:
    res = runner.invoke(
        app, ["calibrate", "--model", str(tmp_path), "--benign", "justaurl", "--revision", "r"]
    )
    assert res.exit_code == 2


async def test_history_command_lists_runs_and_trends(toy_kit: TokenizerKit, tmp_path: Path) -> None:
    ep = FakeEndpoint(toy_kit, seed=2)
    rep = await Suite(probes=["template.token_counts"], options=FAST).arun(
        Target("http://h", "m", transport=ep.transport()),
        Reference(toy_kit, SuiteReference(toy_kit), INFO),
    )
    db = tmp_path / "h.duckdb"
    h = History(db)
    h.record(rep)
    h.close()
    res = runner.invoke(app, ["history", "--db", str(db)])
    assert res.exit_code == 0 and rep.run_id in res.stdout
    trend = runner.invoke(
        app,
        ["history", "--db", str(db), "--probe", "template.token_counts", "--metric", "mismatches"],
    )
    assert trend.exit_code == 0 and trend.stdout.strip().endswith("0")
    assert runner.invoke(app, ["history", "--db", str(tmp_path / "none.duckdb")]).exit_code == 2


def test_canary_runs_once_records_history_and_reports(
    toy_kit: TokenizerKit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import parity.cli as cli
    import parity.exporter as exporter

    ep = FakeEndpoint(toy_kit, extra_prompt_tokens=1, seed=3)
    monkeypatch.setattr(
        cli, "_build_reference", lambda *a, **k: Reference(toy_kit, SuiteReference(toy_kit), INFO)
    )
    monkeypatch.setattr(
        cli, "Target", lambda url, model, *a, **k: Target(url, model, transport=ep.transport())
    )
    served: dict[str, object] = {}
    monkeypatch.setattr(exporter, "serve", lambda state, port: served.setdefault("state", state))
    res = runner.invoke(
        app,
        ["canary", "--model", "x", "--target", "http://h", "--suite", "quick", "--max-runs", "1",
         "--interval", "0", "--out", str(tmp_path)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "canary: DEVIATION" in res.output
    state = served["state"]
    assert "parity_verdict" in state.text()  # type: ignore[attr-defined]
    assert (tmp_path / "history.duckdb").exists()
