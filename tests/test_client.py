import httpx
import pytest

from parity.client import EndpointClient, EndpointError, TokenBucket, backoff_delay


async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": True})

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("parity.client.asyncio.sleep", no_sleep)
    async with EndpointClient("http://x", transport=httpx.MockTransport(handler)) as c:
        assert await c.post_json("/v1/x", {}) == {"ok": True}
    assert calls["n"] == 3


async def test_non_retryable_status_raises_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    async with EndpointClient("http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(EndpointError) as info:
            await c.post_json("/v1/x", {})
    assert info.value.status == 400
    assert calls["n"] == 1


async def test_api_key_only_from_env_and_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={})

    monkeypatch.setenv("TEST_KEY", "sekret")
    async with EndpointClient(
        "http://x", api_key_env="TEST_KEY", transport=httpx.MockTransport(handler)
    ) as c:
        await c.post_json("/p", {})
        assert "sekret" not in repr(c)
    assert seen["auth"] == "Bearer sekret"


async def test_missing_api_key_env_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    async with EndpointClient("http://x", api_key_env="NOPE_KEY") as c:
        with pytest.raises(EndpointError, match="NOPE_KEY"):
            await c.post_json("/p", {})


async def test_token_bucket_throttles() -> None:
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)
        now[0] += d

    bucket = TokenBucket(rate=10.0, capacity=1.0, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(5):
        await bucket.acquire()
    assert now[0] == pytest.approx(0.4)
    assert len(slept) == 4


def test_backoff_is_bounded() -> None:
    for attempt in range(12):
        assert 0.0 <= backoff_delay(attempt, base=0.5, cap=20.0) <= 20.0


@pytest.mark.parametrize("base", ["http://h:1", "http://h:1/", "http://h:1/v1", "http://h:1/v1/"])
async def test_base_url_forms_resolve_to_the_same_api_paths(base: str) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    async with EndpointClient(base, transport=httpx.MockTransport(handler)) as c:
        await c.chat("m", [])
        await c.completions("m", "x")
        await c.get_json("/props")
    assert seen == ["/v1/chat/completions", "/v1/completions", "/props"]
