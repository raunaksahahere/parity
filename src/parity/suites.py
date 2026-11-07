"""Versioned suite data shipped with the package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

# Suite data used by default. core-v1 is kept unchanged so parity 0.1.0 results can be
# reproduced with options={"suite": "core-v1"}.
DEFAULT_SUITE = "core-v2"


@dataclass(frozen=True)
class CorpusDoc:
    id: str
    text: str


def _read(suite: str, name: str) -> Any:
    return json.loads((resources.files("parity") / "suites" / suite / name).read_text("utf-8"))


def load_corpus(suite: str = DEFAULT_SUITE) -> list[CorpusDoc]:
    data = _read(suite, "corpus.json")
    return [CorpusDoc(id=d["id"], text=d["text"]) for d in data["documents"]]


def load_json(suite: str, name: str) -> Any:
    return _read(suite, name)
