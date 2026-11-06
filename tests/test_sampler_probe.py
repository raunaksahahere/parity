from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.probes.base import ProbeContext, ProbeResult, Status
from parity.probes.sampler import SamplerProbe
from parity.suites import load_json
from parity.tokenization import TokenizerKit
from tests.fake_sampler import FakeSamplerServer, make_logprobs


def make_server(kit: TokenizerKit, **kw: Any) -> FakeSamplerServer:
    spec = load_json("core-v1", "sampler.json")
    prompts = {
        tuple(kit.encode(t)): make_logprobs(i, 0.9 + 0.1 * i) for i, t in enumerate(spec["prompts"])
    }
    return FakeSamplerServer(prompts, **kw)


async def run(
    kit: TokenizerKit, server: FakeSamplerServer, adapter: str = "llamacpp", **opts: Any
) -> ProbeResult:
    async with EndpointClient("http://x", transport=server.transport(), max_concurrency=32) as c:
        ctx = ProbeContext(client=c, adapter=get_adapter(adapter), model="m", kit=kit, options=opts)
        return await SamplerProbe().run(ctx)


async def test_honest_sampler_passes(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, make_server(toy_kit, seed=1))
    assert res.status is Status.PASS, res.findings
    assert res.metrics["cases"] == 7


@pytest.mark.parametrize(
    ("flag", "needle"),
    [
        ({"honor_top_k": False}, "top_k"),
        ({"honor_top_p": False}, "top_p"),
        ({"honor_temperature": False}, "temperature"),
    ],
)
async def test_ignored_parameter_is_detected_and_named(
    toy_kit: TokenizerKit, flag: dict[str, bool], needle: str
) -> None:
    res = await run(toy_kit, make_server(toy_kit, seed=2, **flag))
    assert res.status is Status.DEVIATION
    assert any(needle in f.message for f in res.findings), [f.message for f in res.findings]


async def test_silent_default_truncation_is_detected(toy_kit: TokenizerKit) -> None:
    # a server that always applies top_k=20 even when the client asks for no truncation
    res = await run(toy_kit, make_server(toy_kit, seed=3, always_truncate_to=20))
    assert res.status is Status.DEVIATION
    assert any(f.message.startswith("neutral:") for f in res.findings)


async def test_deviation_stops_early_and_saves_requests(toy_kit: TokenizerKit) -> None:
    honest = await run(toy_kit, make_server(toy_kit, seed=4))
    buggy = await run(
        toy_kit, make_server(toy_kit, seed=4, honor_top_k=False, honor_temperature=False)
    )
    assert buggy.status is Status.DEVIATION
    assert buggy.n_requests < honest.n_requests


async def test_target_without_token_id_prompts_is_skipped_not_passed(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, make_server(toy_kit), adapter="openai")
    assert res.status is Status.SKIPPED


async def test_empirical_false_positive_rate_on_honest_servers(toy_kit: TokenizerKit) -> None:
    runs = 25
    deviations = 0
    for seed in range(100, 100 + runs):
        res = await run(
            toy_kit, make_server(toy_kit, seed=seed), sampler_looks=[60, 150], sampler_prompts=2
        )
        deviations += res.status is Status.DEVIATION
    # nominal alpha is 0.05 per probe run; the Bonferroni-spent tests are conservative
    assert deviations <= 2, f"{deviations}/{runs} honest runs flagged"


async def test_probability_moving_past_the_reported_topk_at_high_temperature_is_not_a_violation(
    toy_kit: TokenizerKit,
) -> None:
    """Regression: at temperature > 1 mass legitimately flows into the unreported tail.

    The live run on a real server flagged this honest behaviour as a 16% 'violation'.
    """
    spec = load_json("core-v1", "sampler.json")
    prompts = {
        tuple(toy_kit.encode(t)): make_logprobs(i, 1.5) for i, t in enumerate(spec["prompts"])
    }
    server = FakeSamplerServer(prompts, report_cap=60, seed=11)
    res = await run(toy_kit, server)
    assert res.status is Status.PASS, [f.message for f in res.findings]
    assert res.metrics["temp-1.4.violation_rate"] == 0.0


async def test_top_p_case_with_a_jittery_cutoff_is_skipped_not_flagged(
    toy_kit: TokenizerKit,
) -> None:
    """Regression from the validation study: on a Q2_K server the cumulative mass sat within a
    hair of top_p=0.6 and request-to-request numerical jitter produced a 2.4% 'violation'."""
    spec = load_json("core-v1", "sampler.json")
    prompts = {}
    for i, t in enumerate(spec["prompts"]):
        lp = make_logprobs(i, 1.5)
        order = np.argsort(-lp)
        p = np.exp(lp[order])
        # move mass so that the cumulative sum hits 0.6 almost exactly at the 3rd token
        target = np.array([0.30, 0.20, 0.1005])
        rest = p[3:] / p[3:].sum() * (1 - target.sum())
        newp = np.concatenate([target, rest])
        lp2 = np.empty_like(lp)
        lp2[order] = np.log(newp)
        prompts[tuple(toy_kit.encode(t))] = lp2
    res = await run(toy_kit, FakeSamplerServer(prompts, seed=21))
    assert res.status is Status.PASS, [f.message for f in res.findings]
    chosen = [n for n in res.notes if n.startswith("top-p-0.6") and "uses top_p=" in n]
    assert chosen, res.notes
    for note in chosen:  # the placed value keeps clear of the 0.6005 cumulative mass
        value = float(note.rsplit("=", 1)[1])
        assert abs(value - 0.6005) >= 0.02
    # and an ignored top_p is still caught with the per-context placement
    buggy = await run(toy_kit, FakeSamplerServer(prompts, seed=22, honor_top_p=False))
    assert any("top_p" in f.message for f in buggy.findings if f.severity == "deviation")
