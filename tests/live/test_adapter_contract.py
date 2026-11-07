"""Live adapter contract tests: run against a real server to validate an adapter.

Skipped unless PARITY_LIVE_URL is set (and the `live` marker is selected):

  PARITY_LIVE_URL=http://localhost:8000 PARITY_LIVE_ADAPTER=vllm \\
  PARITY_LIVE_MODEL=Qwen/Qwen3-0.6B PARITY_LIVE_SERVED=Qwen/Qwen3-0.6B \\
  uv run pytest -m live tests/live -v

They check what the probes rely on, against the local Hugging Face reference: token-id prompts
are accepted and counted exactly, teacher-forced log-probs line up with the reference, greedy
generation returns token ids, and sampling parameters are honoured at the response level.
scripts/vllm_live.sh starts a pinned vLLM container and runs these tests plus a full suite.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.reference import build_hf_reference

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("PARITY_LIVE_URL"), reason="PARITY_LIVE_URL not set"),
]
URL = os.environ.get("PARITY_LIVE_URL", "")
ADAPTER = os.environ.get("PARITY_LIVE_ADAPTER", "vllm")
MODEL = os.environ.get("PARITY_LIVE_MODEL", "Qwen/Qwen3-0.6B")
SERVED = os.environ.get("PARITY_LIVE_SERVED", MODEL)
REVISION = os.environ.get("PARITY_LIVE_REVISION", "c1899de289a04d12100db370d81485cdf75e47ca")
TEXT = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"


@pytest.fixture(scope="module")
def ref():  # type: ignore[no-untyped-def]
    return build_hf_reference(MODEL, revision=REVISION)


async def test_token_id_prompts_are_counted_exactly(ref) -> None:  # type: ignore[no-untyped-def]
    adapter = get_adapter(ADAPTER)
    if not adapter.capabilities.token_id_prompts:
        pytest.skip(f"{ADAPTER} does not accept token-id prompts")
    ids = ref.encode(TEXT)
    async with EndpointClient(URL) as c:
        out = await adapter.next_token(
            c, SERVED, ids, top_k=5, sampling={"temperature": 0.0, "top_k": 0, "top_p": 1.0}
        )
    assert out.prompt_tokens == len(ids)


async def test_teacher_forcing_lines_up_with_the_reference(ref) -> None:  # type: ignore[no-untyped-def]
    adapter = get_adapter(ADAPTER)
    if not adapter.capabilities.token_id_prompts:
        pytest.skip(f"{ADAPTER} does not accept token-id prompts")
    ids = ref.encode(TEXT)
    async with EndpointClient(URL) as c:
        positions = await adapter.teacher_force(c, SERVED, ids, top_k=5)
    assert len(positions) == len(ids) - 1
    expected = ref.teacher_force(ids, 5)["lp"]
    got = np.array([p.logprob if p.logprob is not None else np.nan for p in positions])
    ok = ~np.isnan(got)
    assert ok.mean() > 0.8  # forced token inside the reported top-k at most positions
    assert np.abs(got[ok] - expected[ok]).mean() < 0.05  # nats; same weights, other kernels


async def test_greedy_generation_returns_token_ids(ref) -> None:  # type: ignore[no-untyped-def]
    adapter = get_adapter(ADAPTER)
    if not adapter.capabilities.token_id_prompts:
        pytest.skip(f"{ADAPTER} does not accept token-id prompts")
    ids = ref.encode(TEXT)
    async with EndpointClient(URL) as c:
        out = await adapter.generate(
            c, SERVED, ids, n_tokens=8, sampling={"temperature": 0.0, "top_k": 0, "top_p": 1.0}
        )
    got = [p.token_id for p in out.positions]
    assert None not in got and len(got) == 8
    assert got[:2] == ref.greedy(ids, 8)[:2]


async def test_top_k_one_behaves_like_greedy(ref) -> None:  # type: ignore[no-untyped-def]
    adapter = get_adapter(ADAPTER)
    if not adapter.capabilities.token_id_prompts:
        pytest.skip(f"{ADAPTER} does not accept token-id prompts")
    ids = ref.encode(TEXT)
    async with EndpointClient(URL) as c:
        outs = [
            await adapter.next_token(
                c, SERVED, ids, top_k=5, sampling={"temperature": 1.0, "top_k": 1, "top_p": 1.0}
            )
            for _ in range(5)
        ]
    top1 = int(np.argmax(ref.next_token_logprobs(ids)))
    assert {o.positions[0].token_id for o in outs} == {top1}
