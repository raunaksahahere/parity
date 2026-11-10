"""A complete fake OpenAI-compatible endpoint (llama.cpp-shaped) with switchable faults."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import numpy as np

from parity.suites import DEFAULT_SUITE, load_json
from parity.tokenization import TokenizerKit
from tests.fake_lm import FakeLM

TOOLS = load_json(DEFAULT_SUITE, "tools.json")
TRIGGERS = set(TOOLS["trigger_prompts"])
PERSON = '{"name": "Ann", "age": 30, "city": "Rome"}'
CALL = '{"name": "get_weather", "arguments": {"city": "Rome"}}'


class SuiteReference:
    """Reference for the whole suite: FakeLM for log-probs/greedy, scripted for tools."""

    def __init__(self, kit: TokenizerKit, lm: FakeLM | None = None) -> None:
        self.kit = kit
        self.lm = lm or FakeLM()

    def teacher_force(self, ids: list[int], top_k: int) -> dict[str, Any]:
        from tests.fake_lm import FakeReference

        return FakeReference(self.lm).teacher_force(ids, top_k)

    def next_token_logprobs(self, prefix_ids: list[int]) -> Any:
        return self.lm.logprobs(prefix_ids)

    def greedy(self, prefix_ids: list[int], n_tokens: int) -> list[int]:
        text = self.kit.decode(prefix_ids)
        if any(t in text for t in TRIGGERS):
            return self.kit.encode(f"<tool_call>\n{CALL}\n</tool_call>")
        if "Extract the person" in text:
            return self.kit.encode(PERSON)
        ids = list(prefix_ids)
        for _ in range(n_tokens):
            ids.append(int(np.argmax(self.lm.logprobs(ids))))
        return ids[len(prefix_ids) :]


class FakeEndpoint:
    def __init__(
        self,
        kit: TokenizerKit,
        *,
        lm: FakeLM | None = None,
        extra_prompt_tokens: int = 0,
        ignore_top_k: bool = False,
        seed: int = 0,
    ) -> None:
        self.kit = kit
        self.lm = lm or FakeLM()
        self.extra = extra_prompt_tokens
        self.ignore_top_k = ignore_top_k
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {}})
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            return self._chat(body)
        return self._completions(body)

    def _sample(self, lp: np.ndarray, body: dict[str, Any]) -> int:
        t = float(body.get("temperature", 1.0))
        if t <= 0:
            return int(np.argmax(lp))
        k = 0 if self.ignore_top_k else int(body.get("top_k", 0))
        top_p = float(body.get("top_p", 1.0))
        order = np.argsort(-lp)
        keep = len(order)
        if k:
            keep = min(keep, k)
        if top_p < 1.0:
            keep = min(keep, int(np.searchsorted(np.cumsum(np.exp(lp[order])), top_p - 1e-12) + 1))
        sel = order[:keep]
        w = np.exp(lp[sel] / t)
        return int(self.rng.choice(sel, p=w / w.sum()))

    def _completions(self, body: dict[str, Any]) -> httpx.Response:
        k = int(body.get("logprobs", 1))
        n = int(body.get("max_tokens", 1))
        ctx = list(body["prompt"])
        items = []
        for _ in range(n):
            lp = self.lm.logprobs(ctx)
            tok = self._sample(lp, body)
            top = [
                {"id": int(i), "token": f"t{i}", "logprob": float(lp[i])}
                for i in np.argsort(-lp)[: max(k, 1)]
            ]
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

    def _chat(self, body: dict[str, Any]) -> httpx.Response:
        msgs, tools = body["messages"], body.get("tools")
        ids = self.kit.chat_token_ids(msgs, tools=tools)
        usage = {"prompt_tokens": len(ids) + self.extra}
        prompt = msgs[-1]["content"]
        msg: dict[str, Any] = {"content": "Sure."}
        if "response_format" in body:
            msg = {"content": PERSON}
        elif tools and prompt in TRIGGERS and body.get("tool_choice") != "none":
            msg = {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}
                ],
            }
        elif "vault passphrase" in prompt:
            m = re.search(r"vault passphrase is (\S+?)\.", self.kit.decode(ids))
            msg = {"content": m.group(1) if m else "unknown"}
        elif not tools and int(body.get("max_tokens", 1)) > 1:
            # a plain chat: the model's own greedy continuation, cut at a stop token
            stops = {self.kit.tokenizer.convert_tokens_to_ids(t) for t in self.kit.stop_strings()}
            out, items = list(ids), []
            k = int(body.get("top_logprobs") or 0)
            finish = "length"
            for _ in range(int(body["max_tokens"])):
                lp = self.lm.logprobs(out)
                tok = int(np.argmax(lp))
                if tok in stops:
                    finish = "stop"
                    break

                def entry(i: int, lp: np.ndarray = lp) -> dict[str, Any]:
                    text = self.kit.decode([i])
                    return {"token": text, "logprob": float(lp[i]), "bytes": list(text.encode())}

                items.append({**entry(tok), "top_logprobs": [entry(int(i))
                              for i in np.argsort(-lp)[:k]]})  # fmt: skip
                out.append(tok)
            msg = {"content": self.kit.decode(out[len(ids) :])}
            choice: dict[str, Any] = {"message": msg, "finish_reason": finish}
            if body.get("logprobs"):
                choice["logprobs"] = {"content": items}
            return httpx.Response(200, json={"choices": [choice], "usage": usage})
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        return httpx.Response(
            200, json={"choices": [{"message": msg, "finish_reason": finish}], "usage": usage}
        )
