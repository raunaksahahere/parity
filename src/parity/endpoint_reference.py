"""A reference backed by another OpenAI-compatible endpoint (``--reference <url>``).

Probes call the reference synchronously, while endpoint calls are async. The endpoint
client therefore lives on a private event loop in a background thread.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar

import numpy as np

from parity.adapters import Adapter
from parity.client import EndpointClient

T = TypeVar("T")


class EndpointReference:
    def __init__(self, client_factory: Any, adapter: Adapter, model: str) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.adapter = adapter
        self.model = model
        self.client: EndpointClient = self._run(self._make(client_factory))

    @staticmethod
    async def _make(factory: Any) -> EndpointClient:
        client: EndpointClient = factory()
        return client

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def teacher_force(self, ids: Sequence[int], top_k: int) -> dict[str, Any]:
        positions = self._run(
            self.adapter.teacher_force(self.client, self.model, list(ids), top_k=top_k)
        )
        t = len(positions)
        top_ids = np.full((t, top_k), -1, dtype=np.int64)
        top_lp = np.full((t, top_k), -np.inf)
        lp = np.full(t, np.nan)
        for j, pos in enumerate(positions):
            for k, e in enumerate(pos.top[:top_k]):
                top_ids[j, k] = -1 if e.token_id is None else e.token_id
                top_lp[j, k] = e.logprob
            if pos.logprob is not None:
                lp[j] = pos.logprob
        return {"lp": lp, "top_ids": top_ids, "top_lp": top_lp}

    def next_token_logprobs(self, prefix_ids: Sequence[int]) -> Any:
        raise NotImplementedError("an endpoint reference reports only top-k log-probs")

    def greedy(self, prefix_ids: Sequence[int], n_tokens: int) -> list[int]:
        comp = self._run(
            self.adapter.generate(
                self.client,
                self.model,
                list(prefix_ids),
                n_tokens=n_tokens,
                sampling={"temperature": 0.0},
            )
        )
        return [int(p.token_id) for p in comp.positions if p.token_id is not None]

    def close(self) -> None:
        self._run(self.client.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
