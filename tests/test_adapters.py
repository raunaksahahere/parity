import json
from pathlib import Path

import httpx
import pytest

from parity.adapters import get_adapter
from parity.adapters.parsing import parse_legacy_style, parse_logprobs, parse_vllm_prompt_logprobs
from parity.client import EndpointClient

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((FIX / name).read_text())


async def test_llamacpp_teacher_force_replays_recorded_server() -> None:
    rec = load("llamacpp_teacher_force.json")
    ids = rec["ids"]
    steps = iter(rec["steps"])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # the adapter must send token ids, every sampler parameter explicit, and logprobs
        assert body["prompt"] == ids[: len(body["prompt"])]
        assert body["top_k"] == 0 and body["top_p"] == 1.0 and body["min_p"] == 0.0
        assert body["logprobs"] == 5 and body["max_tokens"] == 1
        return httpx.Response(200, json=next(steps))

    async with EndpointClient("http://x", transport=httpx.MockTransport(handler)) as c:
        out = await get_adapter("llamacpp").teacher_force(c, "m", ids, top_k=5)
    assert len(out) == len(ids) - 1
    # forced token 12095 (" Paris") follows "The capital of France is": it is in the top-5
    paris = out[ids.index(12095) - 1]
    assert paris.token_id == 12095
    assert paris.logprob is not None and -0.6 < paris.logprob < -0.2
    assert all(len(p.top) == 5 for p in out)
    assert all(p.top[0].logprob >= p.top[-1].logprob for p in out)


def test_vllm_prompt_logprobs_alignment_and_missing_forced_token() -> None:
    rec = load("vllm_prompt_logprobs.json")
    out = parse_vllm_prompt_logprobs(rec["choices"][0]["prompt_logprobs"], rec["ids"])
    assert [p.token_id for p in out] == [20, 30]
    assert out[0].logprob == -0.5
    assert out[1].logprob == -2.0  # forced token 30 is rank 2
    assert [t.token_id for t in out[1].top] == [31, 30]  # sorted by logprob


def test_vllm_forced_token_outside_topk_is_none_not_guessed() -> None:
    out = parse_vllm_prompt_logprobs(
        [None, {"99": {"logprob": -0.1, "rank": 1, "decoded_token": "z"}}], [1, 2]
    )
    assert out[0].logprob is None


def test_legacy_style_parsing_sorts_top_and_keeps_none() -> None:
    lp = load("openai_echo_logprobs.json")["choices"][0]["logprobs"]
    out = parse_legacy_style(lp)
    assert out[0].logprob is None
    assert out[1].logprob == -1.25
    assert [t.token for t in out[1].top] == [" world", " there"]
    assert parse_logprobs(lp) == out


def test_unknown_adapter_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="choose from"):
        get_adapter("nope")


@pytest.mark.parametrize(
    ("routes", "expected"),
    [
        ({"/props": {"default_generation_settings": {}}}, "llamacpp"),
        ({"/api/version": {"version": "0.33.2"}}, "ollama"),
        ({"/version": {"version": "0.11.0"}}, "vllm"),
        ({}, "openai"),
    ],
)
async def test_server_detection(routes: dict, expected: str) -> None:  # type: ignore[type-arg]
    import httpx

    from parity.client import EndpointClient
    from parity.detect import detect_adapter

    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        return httpx.Response(200, json=body) if body is not None else httpx.Response(404)

    async with EndpointClient("http://h:1", transport=httpx.MockTransport(handler)) as c:
        assert await detect_adapter(c) == expected
