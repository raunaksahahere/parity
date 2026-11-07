"""A fake completions server with a controllable (possibly buggy) sampler."""

from __future__ import annotations

import json

import httpx
import numpy as np

VOCAB = 300


class FakeSamplerServer:
    def __init__(
        self,
        prompts: dict[tuple[int, ...], np.ndarray],
        *,
        honor_top_k: bool = True,
        honor_top_p: bool = True,
        honor_temperature: bool = True,
        always_truncate_to: int = 0,
        report_cap: int = 0,
        jitter: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.logits = prompts
        self.honor_top_k = honor_top_k
        self.honor_top_p = honor_top_p
        self.honor_temperature = honor_temperature
        self.always_truncate_to = always_truncate_to
        self.report_cap = report_cap
        self.jitter = jitter  # per-request logit noise, used for both the report and the sample
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def _sample(self, lp: np.ndarray, body: dict) -> int:  # type: ignore[type-arg]
        t = body.get("temperature", 1.0) if self.honor_temperature else 1.0
        k = body.get("top_k", 0) if self.honor_top_k else 0
        top_p = body.get("top_p", 1.0) if self.honor_top_p else 1.0
        if self.always_truncate_to:
            k = self.always_truncate_to if not k else min(k, self.always_truncate_to)
        order = np.argsort(-lp)
        p1 = np.exp(lp[order])
        keep = len(order)
        if k:
            keep = min(keep, k)
        if top_p < 1.0:
            keep = min(keep, int(np.searchsorted(np.cumsum(p1), top_p - 1e-12) + 1))
        sel = order[:keep]
        w = np.exp(lp[sel] / t)
        w /= w.sum()
        return int(self.rng.choice(sel, p=w))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        lp = self.logits[tuple(body["prompt"])]
        if self.jitter:
            noisy = lp + self.rng.normal(0.0, self.jitter, len(lp))
            lp = noisy - np.log(np.exp(noisy).sum())
        k = int(body.get("logprobs", 1))
        tok = self._sample(lp, body)
        order = np.argsort(-lp)[: max(min(k, self.report_cap) if self.report_cap else k, 1)]
        top = [{"id": int(i), "token": f"t{i}", "logprob": float(lp[i])} for i in order]
        item = {"id": tok, "token": f"t{tok}", "logprob": float(lp[tok]), "top_logprobs": top}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"text": f"t{tok}", "finish_reason": "length", "logprobs": {"content": [item]}}
                ],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 1},
            },
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def make_logprobs(seed: int, sharpness: float = 1.6) -> np.ndarray:
    """A normalized log-distribution with a heavy tail (so truncation is detectable)."""
    rng = np.random.default_rng(seed)
    logits = -sharpness * np.log(np.arange(1, VOCAB + 1)) + rng.normal(0, 0.15, VOCAB)
    return logits - np.log(np.exp(logits).sum())
