"""A synthetic language model used as both reference and (perturbed) target."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import numpy as np

VOCAB = 400


def _rng(prefix: Sequence[int], salt: int) -> np.random.Generator:
    h = hash((tuple(prefix), salt)) & 0xFFFFFFFF
    return np.random.default_rng(h)


class FakeLM:
    """Deterministic next-token log-probs per prefix; ``noise`` perturbs the logits."""

    def __init__(self, noise: float = 0.0, salt: int = 0, noise_salt: int = 99) -> None:
        self.noise = noise
        self.salt = salt
        self.noise_salt = noise_salt

    def logprobs(self, prefix: Sequence[int]) -> np.ndarray:
        # shared structure: zipf-like with a prefix-dependent permutation of which ids are likely
        rng = _rng(prefix, self.salt)
        order = rng.permutation(VOCAB)
        logits = np.empty(VOCAB)
        logits[order] = -1.4 * np.log(np.arange(1, VOCAB + 1))
        if self.noise:
            logits = logits + _rng(prefix, self.noise_salt).normal(0, self.noise, VOCAB)
        return logits - np.log(np.exp(logits).sum())


class FakeReference:
    def __init__(self, lm: FakeLM) -> None:
        self.lm = lm

    def teacher_force(self, ids: Sequence[int], top_k: int) -> dict[str, np.ndarray]:
        lps = [self.lm.logprobs(ids[: i + 1]) for i in range(len(ids) - 1)]
        top_ids = np.stack([np.argsort(-lp)[:top_k] for lp in lps])
        top_lp = np.stack([lp[i] for lp, i in zip(lps, top_ids, strict=True)])
        lp = np.array([lps[i][ids[i + 1]] for i in range(len(lps))])
        return {"lp": lp, "top_ids": top_ids.astype(np.int64), "top_lp": top_lp}

    def next_token_logprobs(self, prefix_ids: Sequence[int]) -> np.ndarray:
        return self.lm.logprobs(prefix_ids)

    def greedy(self, prefix_ids: Sequence[int], n_tokens: int) -> list[int]:
        ids = list(prefix_ids)
        for _ in range(n_tokens):
            ids.append(int(np.argmax(self.lm.logprobs(ids))))
        return ids[len(prefix_ids) :]


def lm_server(lm: FakeLM) -> httpx.MockTransport:
    """A llama.cpp-shaped /v1/completions endpoint serving ``lm`` (greedy for n>1 tokens)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        k = int(body.get("logprobs", 1))
        n = int(body.get("max_tokens", 1))
        ctx = list(body["prompt"])
        items = []
        for _ in range(n):
            lp = lm.logprobs(ctx)
            order = np.argsort(-lp)[:k]
            top = [{"id": int(i), "token": f"t{i}", "logprob": float(lp[i])} for i in order]
            tok = int(order[0])
            items.append(
                {"id": tok, "token": f"t{tok}", "logprob": float(lp[tok]), "top_logprobs": top}
            )
            ctx.append(tok)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "".join(i["token"] for i in items),
                        "finish_reason": "length",
                        "logprobs": {"content": items},
                    }
                ],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": n},
            },
        )

    return httpx.MockTransport(handler)
