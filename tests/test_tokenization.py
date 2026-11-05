from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from parity.tokenization import TokenizerKit

# Surrogates cannot be encoded as UTF-8 and never appear in real request bodies.
text_st = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200)
SETTINGS = settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])


@SETTINGS
@given(text=text_st)
def test_encode_decode_round_trip(toy_kit: TokenizerKit, text: str) -> None:
    assert toy_kit.decode(toy_kit.encode(text)) == text


@SETTINGS
@given(a=text_st, b=text_st)
def test_encoding_never_loses_content_when_concatenated(
    toy_kit: TokenizerKit, a: str, b: str
) -> None:
    # Token boundaries may shift across a concatenation, but the decoded text may not change.
    assert toy_kit.decode(toy_kit.encode(a + b)) == a + b


@SETTINGS
@given(content=text_st, extra=text_st)
def test_longer_conversation_never_has_fewer_tokens_than_its_prefix(
    toy_kit: TokenizerKit, content: str, extra: str
) -> None:
    base = [{"role": "user", "content": content}]
    longer = [*base, {"role": "assistant", "content": "ok"}, {"role": "user", "content": extra}]
    assert len(toy_kit.chat_token_ids(longer)) >= len(toy_kit.chat_token_ids(base))


@SETTINGS
@given(content=text_st)
def test_chat_rendering_is_deterministic_and_ends_with_generation_prompt(
    toy_kit: TokenizerKit, content: str
) -> None:
    msgs = [{"role": "user", "content": content}]
    first = toy_kit.render_chat(msgs)
    assert first == toy_kit.render_chat(msgs)
    assert first.endswith("<|im_start|>assistant\n")
    assert not toy_kit.render_chat(msgs, add_generation_prompt=False).endswith("assistant\n")
