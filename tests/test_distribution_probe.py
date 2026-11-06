from __future__ import annotations

from typing import Any

import numpy as np

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.probes.base import ProbeContext, ProbeResult, Status
from parity.probes.distribution import DistributionProbe, position_stats
from parity.schema import PositionLogprobs, TopEntry
from parity.tokenization import TokenizerKit
from tests.fake_lm import FakeLM, FakeReference, lm_server


async def run(kit: TokenizerKit, target: FakeLM, **opts: Any) -> ProbeResult:
    ref = FakeReference(FakeLM())
    async with EndpointClient("http://x", transport=lm_server(target), max_concurrency=8) as c:
        ctx = ProbeContext(
            client=c,
            adapter=get_adapter("llamacpp"),
            model="m",
            kit=kit,
            reference=ref,
            options=opts,
        )
        return await DistributionProbe().run(ctx)


async def test_identical_model_passes_by_equivalence_and_stops_early(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, FakeLM(), thresholds={"kl_mean": 1e-3})
    assert res.status is Status.PASS
    assert res.effect is not None and res.effect < 1e-9
    assert res.metrics["documents"] < 25  # stopped at an early look


async def test_perturbed_model_is_a_deviation(toy_kit: TokenizerKit) -> None:
    res = await run(toy_kit, FakeLM(noise=0.5), thresholds={"kl_mean": 1e-3})
    assert res.status is Status.DEVIATION
    assert res.ci is not None and res.ci[0] > 1e-3
    assert res.metrics["top1_agreement"] < 1.0


async def test_small_perturbation_straddling_the_threshold_is_inconclusive(
    toy_kit: TokenizerKit,
) -> None:
    probe_noise = FakeLM(noise=0.15)
    sweep = await run(toy_kit, probe_noise, thresholds={"kl_mean": 1e-9})
    kl = sweep.effect
    assert kl is not None and kl > 0
    res = await run(toy_kit, probe_noise, thresholds={"kl_mean": kl})
    assert res.status is Status.INCONCLUSIVE


async def test_effect_grows_monotonically_with_perturbation(toy_kit: TokenizerKit) -> None:
    effects = []
    for noise in (0.05, 0.15, 0.4):
        r = await run(toy_kit, FakeLM(noise=noise), thresholds={"kl_mean": 1e9})
        assert r.effect is not None
        effects.append(r.effect)
    assert effects == sorted(effects) and effects[0] < effects[-1]


async def test_no_reference_is_skipped(toy_kit: TokenizerKit) -> None:
    async with EndpointClient("http://x", transport=lm_server(FakeLM())) as c:
        ctx = ProbeContext(client=c, adapter=get_adapter("llamacpp"), model="m", kit=toy_kit)
        assert (await DistributionProbe().run(ctx)).status is Status.SKIPPED


async def test_text_only_target_is_skipped_with_an_alignment_note(toy_kit: TokenizerKit) -> None:
    async with EndpointClient("http://x", transport=lm_server(FakeLM())) as c:
        ctx = ProbeContext(
            client=c,
            adapter=get_adapter("openai"),
            model="m",
            kit=toy_kit,
            reference=FakeReference(FakeLM()),
        )
        res = await DistributionProbe().run(ctx)
    assert res.status is Status.SKIPPED and "alignment" in res.notes[0]


def test_delta_is_nan_when_forced_token_not_reported() -> None:
    ref = {
        "lp": np.array([-1.0]),
        "top_ids": np.array([[1, 2]]),
        "top_lp": np.array([[-0.5, -1.5]]),
    }
    pos = PositionLogprobs(
        token_id=3,
        token="",
        logprob=None,
        top=[
            TopEntry(token_id=1, token="a", logprob=-0.5),
            TopEntry(token_id=2, token="b", logprob=-1.5),
        ],
    )
    kl, delta, top1, _ = position_stats(ref, [pos])
    assert np.isnan(delta[0]) and kl[0] == 0.0 and top1[0] == 1.0
