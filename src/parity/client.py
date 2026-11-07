"""Async HTTP client for OpenAI-compatible endpoints.

Features: bounded concurrency, token-bucket rate limiting, jittered exponential
retries on transient failures. API keys are read from an environment variable
and are never stored on the object's repr or written to reports.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class EndpointError(RuntimeError):
    """A request failed permanently (non-retryable status or retries exhausted)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class BudgetExhaustedError(EndpointError):
    """The request budget is used up; no further request is sent."""


class TokenBucket:
    """Token bucket: `rate` tokens/second with burst capacity `capacity`."""

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Reserve ``tokens`` (the balance may go negative) and sleep out the deficit.

        Reserving before sleeping keeps concurrent callers correctly spaced and,
        unlike a refill-and-retry loop, cannot spin on float rounding.
        """
        async with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            self._tokens -= tokens
            wait = -self._tokens / self.rate if self._tokens < 0 else 0.0
        if wait > 0:
            await self._sleep(wait)


def backoff_delay(
    attempt: int, base: float = 0.5, cap: float = 20.0, rng: random.Random | None = None
) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**attempt))."""
    r = rng or random
    return float(r.uniform(0.0, min(cap, base * (2**attempt))))


class EndpointClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str | None = None,
        max_concurrency: int = 4,
        rate_per_s: float | None = None,
        max_retries: int = 4,
        timeout_s: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
        request_limit: int | None = None,
        on_request: Callable[[int], None] | None = None,
    ) -> None:
        parts = urlsplit(base_url)
        path = parts.path.rstrip("/") or "/v1"  # bare host:port means the usual /v1 API root
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self.base_url = f"{self._origin}{path}"
        self._api_key_env = api_key_env
        self._sem = asyncio.Semaphore(max_concurrency)
        self._bucket = TokenBucket(rate_per_s) if rate_per_s else None
        self._max_retries = max_retries
        self.requests_sent = 0
        # hard cap on requests_sent (the run's remaining budget); retries count
        self.request_limit = request_limit
        self._on_request = on_request
        self._client = httpx.AsyncClient(
            base_url=self.base_url + "/", timeout=timeout_s, transport=transport
        )

    def __repr__(self) -> str:
        return f"EndpointClient(base_url={self.base_url!r})"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if not key:
                raise EndpointError(f"environment variable {self._api_key_env} is not set")
            headers["authorization"] = f"Bearer {key}"
        return headers

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: str = "unknown error"
        for attempt in range(self._max_retries + 1):
            async with self._sem:
                if self._bucket is not None:
                    await self._bucket.acquire()
                if self.request_limit is not None and self.requests_sent >= self.request_limit:
                    raise BudgetExhaustedError(
                        f"request budget exhausted ({self.request_limit} requests in this session)"
                    )
                self.requests_sent += 1
                if self._on_request is not None:
                    self._on_request(self.requests_sent)
                try:
                    resp = await self._client.post(path, json=payload, headers=self._headers())
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if resp.status_code < 400:
                        data = resp.json()
                        if not isinstance(data, dict):
                            raise EndpointError("response body is not a JSON object")
                        return data
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    if resp.status_code not in RETRYABLE_STATUS:
                        raise EndpointError(last_error, status=resp.status_code)
            if attempt < self._max_retries:
                await asyncio.sleep(backoff_delay(attempt))
        raise EndpointError(f"retries exhausted: {last_error}")

    async def get_json(self, path: str) -> dict[str, Any]:
        """GET a server-root path such as ``/props`` (resolved against the origin, not /v1)."""
        resp = await self._client.get(self._origin + path, headers=self._headers())
        if resp.status_code >= 400:
            raise EndpointError(
                f"HTTP {resp.status_code}: {resp.text[:300]}", status=resp.status_code
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise EndpointError("response body is not a JSON object")
        return data

    async def chat(
        self, model: str, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        return await self.post_json(
            "chat/completions", {"model": model, "messages": messages, **params}
        )

    async def completions(
        self, model: str, prompt: str | list[int], **params: Any
    ) -> dict[str, Any]:
        return await self.post_json("completions", {"model": model, "prompt": prompt, **params})

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
