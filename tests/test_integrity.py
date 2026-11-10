"""Regression tests for v0.1.1: incompatible calibrations and stale saved results must never
silently influence a verdict, and a resumed run must respect its total request budget."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from parity import Reference, Suite, Target
from parity.client import BudgetExhaustedError, EndpointClient
from parity.faults.study import StudyMismatchError, _check_saved_rows
from parity.manifest import ResumeError, RunManifest
from parity.probes.base import Status
from parity.report_model import ReferenceInfo
from parity.stats.calibration import Calibration, CalibrationError, calibrate_threshold
from parity.suite import BUDGET_NOTE, calibration_context, resolve_required
from parity.tokenization import TokenizerKit
from tests.fake_endpoint import FakeEndpoint, SuiteReference

FAST: dict[str, Any] = {
    "sampler_looks": [60, 150],
    "claimed_context": 1024,
    "longctx_lengths": [256, 1024],
    "template_kwargs": {},
}
CHEAP = ["template.token_counts", "longctx.needle", "tools.structured"]


def info(**kw: Any) -> ReferenceInfo:
    base: dict[str, Any] = {
        "kind": "hf:local",
        "model_id": "toy",
        "revision": "abc123",
        "dtype": "float32",
    }
    return ReferenceInfo(**{**base, **kw})


def setup(
    kit: TokenizerKit, ep: FakeEndpoint, url: str = "http://h:1/v1", **suite_kw: Any
) -> tuple[Suite, Target, Reference]:
    suite = Suite(options={**FAST, **suite_kw.pop("options", {})}, **suite_kw)
    return (
        suite,
        Target(url, "m", transport=ep.transport()),
        Reference(kit, SuiteReference(kit), info()),
    )


def calibrated(suite: Suite, ref: Reference) -> Calibration:
    probes = ["distribution.teacher_forced", "greedy.agreement", "tools.structured"]
    cal = Calibration(
        model_id=ref.info.model_id,
        revision=ref.info.revision,
        suite="core-v1",
        context=calibration_context(ref, suite.effective_options(), probes),
    )
    cal.metrics["kl_mean"] = calibrate_threshold({"benign": 1e-4})
    return cal


# -- calibration compatibility ------------------------------------------------------------------


async def test_compatible_calibration_is_used(toy_kit: TokenizerKit) -> None:
    ep = FakeEndpoint(toy_kit, seed=1)
    suite, target, ref = setup(toy_kit, ep)
    suite.calibration = calibrated(suite, ref)
    rep = await suite.arun(target, ref)
    assert rep.calibration is not None and rep.calibration.fingerprint
    dist = next(o.result for o in rep.probes if o.result.probe == "distribution.teacher_forced")
    assert dist.threshold == pytest.approx(2e-4)


MISMATCHES: dict[str, tuple[str, Any, str]] = {
    "revision": ("revision", "def456", "revision"),
    "model": ("model_id", "other/model", "model"),
    "tokenizer": ("tokenizer_fingerprint", "0" * 64, "tokenizer or chat template"),
    "suite data": ("suite_fingerprint", "1" * 64, "suite data"),
    "probe version": ("probe_versions", {"distribution.teacher_forced": "0"}, "probe"),
    "reference dtype": ("reference_dtype", "bfloat16", "reference dtype"),
    "reference kind": ("reference_kind", "url", "reference"),
    "settings": ("settings", {"distribution_top_k": 20}, "setting distribution_top_k"),
}


@pytest.mark.parametrize("case", sorted(MISMATCHES))
async def test_incompatible_calibration_is_refused_before_any_request(
    toy_kit: TokenizerKit, case: str
) -> None:
    field, value, expect = MISMATCHES[case]
    ep = FakeEndpoint(toy_kit, seed=2)
    suite, target, ref = setup(toy_kit, ep)
    cal = calibrated(suite, ref)
    assert cal.context is not None
    if isinstance(value, dict):
        value = {**getattr(cal.context, field), **value}
    cal.context = cal.context.model_copy(update={field: value})
    suite.calibration, suite.calibration_path = cal, "cal.json"
    with pytest.raises(CalibrationError) as err:
        await suite.arun(target, ref)
    assert expect in str(err.value) and "parity calibrate" in str(err.value)
    assert ep.calls == 0  # nothing was sent to the endpoint


def test_parity_0_1_0_calibration_files_are_refused(toy_kit: TokenizerKit) -> None:
    legacy = Calibration.load(
        Path(__file__).parent.parent / "docs/data/validation-calibration.json"
    )
    assert legacy.schema_version == 1
    suite, _, ref = setup(toy_kit, FakeEndpoint(toy_kit))
    problems = legacy.mismatches(calibration_context(ref, suite.effective_options(), []), [])
    assert problems and "schema 1" in problems[0]


def test_weights_fingerprint_decides_model_identity(toy_kit: TokenizerKit) -> None:
    suite, _, _ = setup(toy_kit, FakeEndpoint(toy_kit))
    a = Reference(toy_kit, None, info(model_fingerprint="a" * 64, model_id="Qwen/Qwen3-0.6B"))
    b = Reference(toy_kit, None, info(model_fingerprint="a" * 64, model_id="models/hf/Qwen3"))
    c = Reference(toy_kit, None, info(model_fingerprint="c" * 64, model_id="Qwen/Qwen3-0.6B"))
    opts = suite.effective_options()
    cal = Calibration(
        model_id="x", revision="abc123", suite="core-v1", context=calibration_context(a, opts, [])
    )
    assert cal.mismatches(calibration_context(b, opts, []), []) == []  # same weights, other path
    assert "model weights" in cal.mismatches(calibration_context(c, opts, []), [])[0]


def test_a_changed_chat_template_changes_the_tokenizer_fingerprint(toy_kit: TokenizerKit) -> None:
    import copy

    from parity.fingerprint import tokenizer_fingerprint

    other = copy.deepcopy(toy_kit.tokenizer)
    other.chat_template = toy_kit.tokenizer.chat_template.replace("<|im_end|>", "</s>")
    assert tokenizer_fingerprint(other) != tokenizer_fingerprint(toy_kit.tokenizer)
    assert tokenizer_fingerprint(toy_kit.tokenizer) == tokenizer_fingerprint(toy_kit.tokenizer)


def test_cli_run_refuses_an_incompatible_calibration(
    toy_kit: TokenizerKit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import parity.cli as cli

    ep = FakeEndpoint(toy_kit, seed=3)
    suite, _, ref = setup(toy_kit, ep)
    cal = calibrated(suite, ref)
    assert cal.context is not None
    cal.context.revision = "an-older-revision"
    cal.save(tmp_path / "cal.json")
    monkeypatch.setattr(cli, "_build_reference", lambda *a, **k: ref)
    res = CliRunner().invoke(
        cli.app,
        ["run", "--model", "toy", "--target", "http://h:1", "--calibration",
         str(tmp_path / "cal.json"), "--out", str(tmp_path / "runs")],
    )  # fmt: skip
    assert res.exit_code == 2
    assert "does not match this run" in res.output and "revision" in res.output
    assert ep.calls == 0


# -- resuming -----------------------------------------------------------------------------------


async def test_resume_reuses_only_verified_results(toy_kit: TokenizerKit, tmp_path: Path) -> None:
    ep = FakeEndpoint(toy_kit, seed=4)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP)
    first = await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    manifest = RunManifest.load(tmp_path / "r" / "manifest.json")
    assert set(manifest.checkpoints) == set(CHEAP)
    assert manifest.requests_consumed == first.requests_consumed == first.requests_total
    calls = ep.calls
    again = await suite.arun(target, ref, out_dir=tmp_path, run_id="r", resume=True)
    assert ep.calls - calls <= 1  # adapter detection only
    assert again.resumed and again.verdict is first.verdict
    assert RunManifest.load(tmp_path / "r" / "manifest.json").sessions == 2


CHANGES: dict[str, dict[str, Any]] = {
    "seed": {"seed": 7},
    "alpha": {"alpha": 0.01},
    "budget": {"budget": 500},
    "options": {"options": {"claimed_context": 2048}},
    "probes": {"probes": CHEAP[:2]},
    "fail_fast": {"fail_fast": True},
}


@pytest.mark.parametrize("case", sorted(CHANGES))
async def test_resume_with_a_different_configuration_is_refused(
    toy_kit: TokenizerKit, tmp_path: Path, case: str
) -> None:
    ep = FakeEndpoint(toy_kit, seed=5)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP)
    await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    kw = {"probes": CHEAP, **CHANGES[case]}
    changed, _, _ = setup(toy_kit, ep, **kw)
    calls = ep.calls
    with pytest.raises(ResumeError) as err:
        await changed.arun(target, ref, out_dir=tmp_path, run_id="r", resume=True)
    assert "do not match this run" in str(err.value)
    assert ep.calls - calls <= 1  # no probe request was sent


async def test_resume_against_another_target_or_calibration_is_refused(
    toy_kit: TokenizerKit, tmp_path: Path
) -> None:
    ep = FakeEndpoint(toy_kit, seed=6)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP)
    await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    other = Target("http://elsewhere:2/v1", "m", transport=ep.transport())
    with pytest.raises(ResumeError, match=r"target\.base_url"):
        await suite.arun(other, ref, out_dir=tmp_path, run_id="r", resume=True)
    suite.calibration = calibrated(suite, ref)
    with pytest.raises(ResumeError, match="calibration"):
        await suite.arun(target, ref, out_dir=tmp_path, run_id="r", resume=True)


async def test_an_edited_result_cannot_change_the_verdict(
    toy_kit: TokenizerKit, tmp_path: Path
) -> None:
    ep = FakeEndpoint(toy_kit, seed=7)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP)
    await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    saved = tmp_path / "r" / "probes" / "template.token_counts.json"
    data = json.loads(saved.read_text())
    data["status"] = "DEVIATION"
    saved.write_text(json.dumps(data))
    with pytest.raises(ResumeError, match="modified after it was saved"):
        await suite.arun(target, ref, out_dir=tmp_path, run_id="r", resume=True)


async def test_a_result_copied_from_another_run_is_rejected(
    toy_kit: TokenizerKit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deviating result from a faulty deployment must not leak into an honest run."""
    bad_ep = FakeEndpoint(toy_kit, extra_prompt_tokens=1, seed=8)
    bad, bad_target, ref = setup(toy_kit, bad_ep, probes=CHEAP)
    bad_rep = await bad.arun(bad_target, ref, out_dir=tmp_path / "a", run_id="r")
    assert bad_rep.verdict is Status.DEVIATION

    # an honest run interrupted during its first probe: nothing of it is saved yet
    from parity.probes.template import TemplateProbe

    original = TemplateProbe.run

    async def interrupted(self: Any, ctx: Any) -> Any:
        raise KeyboardInterrupt

    ep = FakeEndpoint(toy_kit, seed=9)
    honest, target, _ = setup(toy_kit, ep, probes=CHEAP)
    monkeypatch.setattr(TemplateProbe, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        await honest.arun(target, ref, out_dir=tmp_path / "b", run_id="r")
    monkeypatch.setattr(TemplateProbe, "run", original)
    shutil.copy(
        tmp_path / "a/r/probes/template.token_counts.json",
        tmp_path / "b/r/probes/template.token_counts.json",
    )
    with pytest.raises(ResumeError, match="not recorded in the manifest"):
        await honest.arun(target, ref, out_dir=tmp_path / "b", run_id="r", resume=True)


async def test_results_of_an_older_probe_version_are_rejected(
    toy_kit: TokenizerKit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from parity.probes.template import TemplateProbe

    ep = FakeEndpoint(toy_kit, seed=10)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP)
    await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    saved_version = TemplateProbe.version
    monkeypatch.setattr(TemplateProbe, "version", "99")
    with pytest.raises(ResumeError) as err:
        await suite.arun(target, ref, out_dir=tmp_path, run_id="r", resume=True)
    assert "probe_versions.template.token_counts" in str(err.value)
    assert f"version {saved_version} of the probe" in str(err.value)


async def test_runs_without_a_manifest_cannot_be_resumed(
    toy_kit: TokenizerKit, tmp_path: Path
) -> None:
    ep = FakeEndpoint(toy_kit, seed=11)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP)
    await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    (tmp_path / "r" / "manifest.json").unlink()  # what a parity 0.1.0 run directory looks like
    with pytest.raises(ResumeError, match=r"no manifest\.json"):
        await suite.arun(target, ref, out_dir=tmp_path, run_id="r", resume=True)
    with pytest.raises(ResumeError, match="nothing to resume"):
        await suite.arun(target, ref, out_dir=tmp_path, run_id="missing", resume=True)


async def test_a_new_run_never_writes_into_existing_results(
    toy_kit: TokenizerKit, tmp_path: Path
) -> None:
    ep = FakeEndpoint(toy_kit, seed=12)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP)
    await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    with pytest.raises(ResumeError, match="already holds results"):
        await suite.arun(target, ref, out_dir=tmp_path, run_id="r")


# -- budgets across sessions ----------------------------------------------------------------------


async def test_a_resumed_run_charges_requests_already_consumed(
    toy_kit: TokenizerKit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session 1 is interrupted inside the tools probe after spending requests. Session 2 may
    only spend what is left of the budget, and the report counts both sessions."""
    from parity.probes.tools import ToolsProbe

    budget = 40
    original = ToolsProbe.run

    async def partial(self: Any, ctx: Any) -> Any:
        for _ in range(10):
            await ctx.client.chat("m", [{"role": "user", "content": "hi"}], max_tokens=1)
        raise KeyboardInterrupt

    ep = FakeEndpoint(toy_kit, seed=13)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP, budget=budget)
    monkeypatch.setattr(ToolsProbe, "run", partial)
    with pytest.raises(KeyboardInterrupt):
        await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    spent_first = RunManifest.load(tmp_path / "r" / "manifest.json").requests_consumed
    assert spent_first >= 10 + 7  # template probe, longctx probe and the interrupted tools work

    monkeypatch.setattr(ToolsProbe, "run", original)
    calls = ep.calls
    rep = await suite.arun(target, ref, out_dir=tmp_path, run_id="r", resume=True)
    sent_second = ep.calls - calls - 1  # minus adapter detection
    assert sent_second <= budget - spent_first
    assert rep.requests_consumed == spent_first + sent_second <= budget
    assert rep.verdict is not Status.PASS  # the tools probe could not finish within budget


async def test_an_exhausted_budget_stops_further_probes(
    toy_kit: TokenizerKit, tmp_path: Path
) -> None:
    ep = FakeEndpoint(toy_kit, seed=14)
    suite, target, ref = setup(toy_kit, ep, probes=CHEAP, budget=7)
    rep = await suite.arun(target, ref, out_dir=tmp_path, run_id="r")
    status = {o.result.probe: o.result for o in rep.probes}
    assert status["template.token_counts"].status is Status.PASS
    assert BUDGET_NOTE in status["tools.structured"].notes
    assert rep.requests_consumed is not None and rep.requests_consumed <= 7
    assert rep.verdict is Status.INCONCLUSIVE


async def test_the_client_never_exceeds_its_request_limit(toy_kit: TokenizerKit) -> None:
    ep = FakeEndpoint(toy_kit)
    async with EndpointClient("http://h:1", transport=ep.transport(), request_limit=2) as c:
        for _ in range(2):
            await c.chat("m", [{"role": "user", "content": "x"}])
        with pytest.raises(BudgetExhaustedError):
            await c.chat("m", [{"role": "user", "content": "x"}])
    assert ep.calls == 2


# -- coverage and required probes ----------------------------------------------------------------


async def test_required_probes_that_are_skipped_turn_a_pass_into_inconclusive(
    toy_kit: TokenizerKit, text_only_adapter: str
) -> None:
    ep = FakeEndpoint(toy_kit, seed=15)
    suite, _, ref = setup(toy_kit, ep, require=["template", "distribution"])
    target = Target("http://h:1/v1", "m", adapter=text_only_adapter, transport=ep.transport())
    rep = await suite.arun(target, ref)
    assert rep.coverage is not None
    assert "distribution.teacher_forced" in rep.coverage.skipped
    assert rep.coverage.unmet_required == ["distribution.teacher_forced"]
    assert rep.verdict is Status.INCONCLUSIVE and rep.exit_code == 2
    assert "Required probes did not reach a decision" in rep.summary

    plain, _, _ = setup(toy_kit, ep)
    rep2 = await plain.arun(target, ref)
    assert rep2.verdict is Status.PASS and rep2.lower_power
    assert rep2.coverage is not None and len(rep2.coverage.evaluated) == 4


def test_required_probe_names_are_validated() -> None:
    probes = ["template.token_counts", "sampler.conformance"]
    assert resolve_required(["all"], probes) == probes
    assert resolve_required(["sampler"], probes) == ["sampler.conformance"]
    with pytest.raises(ValueError, match="unknown required probe"):
        resolve_required(["nope"], probes)
    with pytest.raises(ValueError, match="not part of this run"):
        resolve_required(["greedy"], probes)


async def test_markdown_reports_show_coverage_and_escape_cells(
    toy_kit: TokenizerKit, text_only_adapter: str
) -> None:
    from parity.render import to_html, to_markdown

    ep = FakeEndpoint(toy_kit, seed=16)
    suite, _, ref = setup(toy_kit, ep)
    target = Target("http://h:1/v1", "m", adapter=text_only_adapter, transport=ep.transport())
    rep = await suite.arun(target, ref)
    md = to_markdown(rep)
    assert "**Coverage: 4/6 probes evaluated.**" in md and "| Skipped probe | Why |" in md
    assert "\\|token count difference\\|" in md  # a bare | would break the table
    assert "Coverage: 4/6 probes evaluated" in to_html(rep)


# -- study-level -----------------------------------------------------------------------------------


def test_study_rows_from_another_configuration_are_refused() -> None:
    row = {"scenario": "control-a", "study_fingerprint": "s1", "scenario_fingerprint": "c1"}
    _check_saved_rows([row], "s1", {"control-a": "c1"})  # matching: fine
    with pytest.raises(StudyMismatchError, match="another study configuration"):
        _check_saved_rows([row], "s2", {"control-a": "c1"})
    with pytest.raises(StudyMismatchError, match="different scenario settings"):
        _check_saved_rows([row], "s1", {"control-a": "c2"})
    legacy = {"scenario": "control-a", "verdict": "PASS"}  # a parity 0.1.0 row
    with pytest.raises(StudyMismatchError):
        _check_saved_rows([legacy], "s1", {"control-a": "c1"})


async def test_no_pass_without_model_identity_evidence(toy_kit: TokenizerKit) -> None:
    """Without a reference the identity probes cannot run. A correct serving stack is then not
    enough for a PASS, unless the caller explicitly accepts that."""
    ep = FakeEndpoint(toy_kit, seed=17)
    target = Target("http://h:1/v1", "m", transport=ep.transport())
    no_ref = Reference(toy_kit, None, info())
    rep = await Suite(options=FAST).arun(target, no_ref)
    assert rep.verdict is Status.INCONCLUSIVE and rep.exit_code == 2
    assert "PASS is withheld" in rep.summary
    allowed = await Suite(options=FAST, allow_unverified_pass=True).arun(target, no_ref)
    assert allowed.verdict is Status.PASS and "PASS is withheld" in allowed.summary
    # a run that never asked for identity probes is judged on what it asked for
    quick = await Suite(options=FAST, probes=["template.token_counts"]).arun(target, no_ref)
    assert quick.verdict is Status.PASS
