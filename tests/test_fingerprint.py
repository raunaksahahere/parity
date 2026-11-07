from __future__ import annotations

import hashlib
from pathlib import Path

from parity.faults.resources import peak_rss_mb, reset_peak_rss, wait_until_cool
from parity.fingerprint import (
    code_fingerprint,
    environment,
    file_sha256,
    model_fingerprint,
    suite_fingerprint,
)


def test_model_fingerprint_follows_weight_content(tmp_path: Path) -> None:
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text('{"layers": 2}')
    (d / "model.safetensors").write_bytes(b"weights-v1")
    a = model_fingerprint(d, tmp_path / "cache")
    assert a is not None and a == model_fingerprint(d, tmp_path / "cache")
    (d / "model.safetensors").write_bytes(b"weights-v2")
    assert model_fingerprint(d, tmp_path / "cache") != a
    empty = tmp_path / "tok-only"
    empty.mkdir()
    (empty / "tokenizer.json").write_text("{}")
    assert model_fingerprint(empty) is None


def test_hf_cache_blob_names_are_reused_instead_of_rehashing(tmp_path: Path) -> None:
    data = b"large weights"
    sha = hashlib.sha256(data).hexdigest()
    blob = tmp_path / "blobs" / sha
    blob.parent.mkdir()
    blob.write_bytes(data)
    link = tmp_path / "model.safetensors"
    link.symlink_to(blob)
    assert file_sha256(link) == sha


def test_suite_code_and_environment_fingerprints_are_stable() -> None:
    assert suite_fingerprint("core-v1") == suite_fingerprint("core-v1")
    assert len(code_fingerprint()) == 64
    env = environment()
    assert {"python", "parity", "transformers", "torch"} <= set(env)


def test_wait_until_cool_polls_until_below_the_limit() -> None:
    temps = iter([92.0, 80.0, 64.0])
    slept: list[float] = []
    waited = wait_until_cool(
        65, log=lambda _: None, poll_s=5, read=lambda: next(temps), sleep=slept.append
    )
    assert waited == 10 and slept == [5, 5]
    assert wait_until_cool(65, read=lambda: None, sleep=slept.append) == 0  # no sensors


def test_peak_rss_reads_and_resets_on_linux() -> None:
    first = peak_rss_mb()
    assert first is None or first > 0
    reset_peak_rss()  # must not raise even where unsupported
