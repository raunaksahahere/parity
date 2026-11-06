from __future__ import annotations

from typing import Any

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.probes.base import ProbeContext, ProbeResult, Status
from parity.probes.greedy import GreedyProbe, first_divergence
from parity.tokenization import TokenizerKit
from tests.fake_lm import FakeLM, FakeReference, lm_server


async def run(kit: TokenizerKit, target: FakeLM, **opts: Any) -> ProbeResult:
    async with EndpointClient("http://x", transport=lm_server(target), max_concurrency=8) as c:
        ctx = ProbeContext(
            client=c,
            adapter=get_adapter("llamacpp"),
            model="m",
            kit=kit,
            reference=FakeReference(FakeLM()),
            options=opts,
        )
        return await GreedyProbe().run(ctx)


def test_first_divergence() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2  # target stopped early
    assert first_divergence([], [5]) == 0


async def test_identical_model_is_accepted(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, FakeLM(), thresholds={"greedy_hazard": 0.004})
    assert res.status is Status.PASS
    assert res.metrics["divergences"] == 0


async def test_different_model_is_a_deviation_and_stops_quickly(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, FakeLM(noise=2.0), thresholds={"greedy_hazard": 0.004})
    assert res.status is Status.DEVIATION
    assert res.metrics["hazard"] > 0.1
    assert res.n_requests < 25  # decided before using the whole prompt set


async def test_mild_perturbation_has_hazard_between_the_two(toy_kit: TokenizerKit) -> None:
    mild = await run(toy_kit, FakeLM(noise=0.3), thresholds={"greedy_hazard": 1e-3}, greedy_h1=0.9)
    strong = await run(
        toy_kit, FakeLM(noise=2.0), thresholds={"greedy_hazard": 1e-3}, greedy_h1=0.9
    )
    assert mild.metrics["hazard"] <= strong.metrics["hazard"]
