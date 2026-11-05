from __future__ import annotations

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from parity.tokenization import TokenizerKit

CHATML = (
    "{%- if tools %}{{ '<|im_start|>system\\n' }}"
    "{%- if messages[0].role == 'system' %}{{ messages[0].content + '\\n\\n' }}{%- endif %}"
    "{{ 'TOOLS: ' }}{%- for t in tools %}{{ t | tojson }}{%- endfor %}{{ '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for m in messages %}"
    "{%- if not (loop.first and m.role == 'system' and tools) %}"
    "{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)

TRAIN_TEXT = [
    "You are a helpful assistant. Answer in one short sentence. Hello there, what is my name?",
    "The quick brown fox jumps over the lazy dog. Weather in Paris and Cairo today, warm or cold.",
    "Translate to English: hello world. Use tools when they help. Nice to meet you, Priya.",
] * 20


@pytest.fixture(scope="session")
def toy_kit() -> TokenizerKit:
    """A tiny byte-level BPE tokenizer with a ChatML-style template, trained in-process."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(TRAIN_TEXT, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, chat_template=CHATML)
    return TokenizerKit(fast)
