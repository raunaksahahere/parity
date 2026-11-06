from __future__ import annotations

from parity.probes.base import Finding, ProbeResult, Status
from parity.verdict import decide, summarize


def res(
    probe: str,
    status: Status,
    p: float | None = None,
    hard: bool = False,
    power: str = "normal",
    note: str = "",
) -> ProbeResult:
    return ProbeResult(
        probe=probe,
        family=probe.split(".")[0],
        version="1",
        status=status,
        p_value=p,
        hard_evidence=hard,
        power=power,  # type: ignore[arg-type]
        notes=[note] if note else [],
        findings=[Finding(severity="info", message="x")],
    )


def test_all_pass_is_pass() -> None:
    _, overall, lower = decide(
        [
            res("template.token_counts", Status.PASS),
            res("distribution.teacher_forced", Status.PASS, 0.9),
        ],
        0.05,
    )
    assert overall is Status.PASS and not lower


def test_hard_evidence_deviation_survives_multiplicity() -> None:
    results = [res("template.token_counts", Status.DEVIATION, hard=True)] + [
        res(f"p{i}.x", Status.PASS, 0.5) for i in range(5)
    ]
    out, overall, _ = decide(results, 0.05)
    assert overall is Status.DEVIATION
    assert out[0].final_status is Status.DEVIATION and "deterministic" in out[0].reason


def test_marginal_statistical_deviation_is_downgraded_by_holm() -> None:
    # p=0.02 is significant alone, but with 4 tests Holm adjusts it to 0.08 >= alpha
    results = [res("sampler.conformance", Status.DEVIATION, 0.02)] + [
        res(f"p{i}.x", Status.PASS, 0.6) for i in range(3)
    ]
    out, overall, _ = decide(results, 0.05)
    assert out[0].final_status is Status.INCONCLUSIVE
    assert out[0].adjusted_p is not None and out[0].adjusted_p >= 0.05
    assert overall is Status.INCONCLUSIVE


def test_strong_statistical_deviation_stays_a_deviation() -> None:
    results = [res("sampler.conformance", Status.DEVIATION, 1e-9)] + [
        res(f"p{i}.x", Status.PASS, 0.6) for i in range(3)
    ]
    _, overall, _ = decide(results, 0.05)
    assert overall is Status.DEVIATION


def test_skipped_core_probes_flag_lower_power_but_can_still_pass() -> None:
    results = [
        res("template.token_counts", Status.PASS),
        res("distribution.teacher_forced", Status.SKIPPED, note="no logprobs"),
        res("sampler.conformance", Status.SKIPPED),
        res("tools.structured", Status.PASS, 0.7, power="lower"),
    ]
    out, overall, lower = decide(results, 0.05)
    assert overall is Status.PASS and lower
    assert "no logprobs" in out[1].reason
    assert "lower power" in summarize(out, overall, lower)


def test_nothing_evaluated_is_inconclusive_never_pass() -> None:
    _, overall, _ = decide([res("a.x", Status.SKIPPED), res("b.x", Status.SKIPPED)], 0.05)
    assert overall is Status.INCONCLUSIVE


def test_inconclusive_probe_blocks_a_clean_pass() -> None:
    _, overall, _ = decide([res("a.x", Status.PASS, 0.9), res("b.x", Status.INCONCLUSIVE)], 0.05)
    assert overall is Status.INCONCLUSIVE
