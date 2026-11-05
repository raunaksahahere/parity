"""Content-addressed on-disk cache for reference computations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def stable_hash(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class ArrayCache:
    """Caches dicts of numpy arrays as ``.npz`` files under ``root/namespace``."""

    def __init__(self, root: Path, namespace: str) -> None:
        self.dir = root / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self, key: str, compute: Callable[[], dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        path = self.dir / f"{key}.npz"
        if path.exists():
            with np.load(path) as f:
                return {k: f[k] for k in f.files}
        value = compute()
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **value)  # type: ignore[arg-type]
        tmp.replace(path)
        return value
