"""Reference implementation backed by Hugging Face transformers (local, CPU).

Reference results are cached on disk under the pinned model revision hash so a
given (model revision, dtype, input) is only ever computed once.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from parity.cache import ArrayCache, stable_hash
from parity.tokenization import TokenizerKit


class ReferenceUnavailableError(RuntimeError):
    pass


class HFReference:
    def __init__(
        self,
        model_path: str | Path,
        *,
        revision: str,
        dtype: str = "float32",
        cache_dir: Path = Path(".parity-cache"),
    ) -> None:
        self.model_path = str(model_path)
        self.revision = revision
        self.dtype = dtype
        self._model: Any = None
        self._tok: Any = None
        ns = f"ref-{Path(self.model_path).name}@{revision[:12]}-{dtype}"
        self.cache = ArrayCache(cache_dir, ns)

    # -- loading ---------------------------------------------------------
    @property
    def tokenizer(self) -> Any:
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_path)
        return self._tok

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ReferenceUnavailableError(
                    "the local HF reference needs torch: pip install 'llm-parity[reference]'"
                ) from exc
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=getattr(torch, self.dtype)
            )
            model.train(False)
            self._model = model
        return self._model

    # -- tokenization / templates (delegated to TokenizerKit) -------------
    @property
    def kit(self) -> TokenizerKit:
        return TokenizerKit(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.kit.encode(text)

    def render_chat(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        return self.kit.render_chat(messages, **kwargs)

    def chat_token_ids(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> list[int]:
        return self.kit.chat_token_ids(messages, **kwargs)

    # -- model calls -----------------------------------------------------
    def teacher_force(self, ids: Sequence[int], top_k: int) -> dict[str, np.ndarray]:
        """Log-probs for tokens ``ids[1:]`` given their prefixes.

        Returns ``lp`` (T-1,) log-prob of the actual next token, and
        ``top_ids``/``top_lp`` (T-1, top_k) the reference top-k at each position.
        """

        def compute() -> dict[str, np.ndarray]:
            import torch

            model = self._load_model()
            with torch.no_grad():
                logits = model(torch.tensor([list(ids)])).logits[0, :-1].float()
                logp = torch.log_softmax(logits, dim=-1)
                target = torch.tensor(list(ids)[1:])
                lp = logp.gather(1, target[:, None])[:, 0]
                top_lp, top_ids = torch.topk(logp, top_k, dim=-1)
            return {
                "lp": lp.numpy(),
                "top_ids": top_ids.numpy().astype(np.int64),
                "top_lp": top_lp.numpy(),
            }

        key = stable_hash("tf", list(ids), top_k)
        return self.cache.get_or_compute(key, compute)

    def next_token_logprobs(self, prefix_ids: Sequence[int]) -> np.ndarray:
        """Full-vocabulary log-softmax for the token following ``prefix_ids``."""

        def compute() -> dict[str, np.ndarray]:
            import torch

            model = self._load_model()
            with torch.no_grad():
                logits = model(torch.tensor([list(prefix_ids)])).logits[0, -1].float()
                return {"logp": torch.log_softmax(logits, dim=-1).numpy()}

        return self.cache.get_or_compute(stable_hash("next", list(prefix_ids)), compute)["logp"]

    def greedy(self, prefix_ids: Sequence[int], n_tokens: int) -> list[int]:
        def compute() -> dict[str, np.ndarray]:
            import torch

            model = self._load_model()
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([list(prefix_ids)]),
                    max_new_tokens=n_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            return {"ids": out[0, len(prefix_ids) :].numpy().astype(np.int64)}

        key = stable_hash("greedy", list(prefix_ids), n_tokens)
        return [int(t) for t in self.cache.get_or_compute(key, compute)["ids"]]


def build_hf_reference(
    model: str,
    *,
    revision: str | None = None,
    dtype: str = "float32",
    cache_dir: Path = Path(".parity-cache"),
) -> HFReference:
    """Resolve ``model`` (a local directory or a Hugging Face id) to a pinned reference.

    For hub ids the weights are downloaded at ``revision`` (the resolved commit hash is
    recorded so the reference cache is keyed by it). A local directory needs an explicit
    ``revision`` label, since there is no hub commit to read.
    """
    path = Path(model)
    if path.is_dir():
        return HFReference(path, revision=revision or "local", dtype=dtype, cache_dir=cache_dir)
    from huggingface_hub import HfApi, snapshot_download

    sha = revision or HfApi().model_info(model).sha
    if sha is None:
        raise ReferenceUnavailableError(f"could not resolve a revision for {model}")
    local = snapshot_download(model, revision=sha)
    return HFReference(local, revision=sha, dtype=dtype, cache_dir=cache_dir)
