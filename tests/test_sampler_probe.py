from __future__ import annotations

from typing import Any

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
