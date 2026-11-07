"""Fingerprints of everything a result depends on: weights, tokenizer, suite data and code.

They let a calibration or a saved partial run prove that it was produced under the same
conditions as the run that wants to use it. All fingerprints are hex SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import parity

HEX64 = re.compile(r"[0-9a-f]{64}")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth")
MODEL_META_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "chat_template.jinja",
    "chat_template.json",
)
CODE_SUFFIXES = (".py", ".json", ".j2", ".jinja")
LIBRARIES = ("numpy", "scipy", "torch", "transformers", "tokenizers", "jinja2", "httpx")
# Fixed inputs whose tokenization and template rendering characterise a tokenizer's behaviour.
PROBE_TEXT = "Hello, world! Ünïcödé 数学 🙂 <|im_start|> 12345 \n\t  def f(x): return x**2"
PROBE_CHAT = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}]
PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def digest(*parts: Any) -> str:
    """SHA-256 of a canonical JSON encoding of ``parts``."""
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def short(fp: str | None) -> str:
    return "none" if not fp else fp[:12]


class _HashCache:
    """sha256 of large files, remembered on disk by (path, size, mtime) so that hashing a
    multi-gigabyte weight file happens once per machine."""

    def __init__(self, cache_dir: Path | None) -> None:
        self.path = cache_dir / "file-sha256.json" if cache_dir else None
        self.data: dict[str, str] = {}
        if self.path is not None and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def sha256(self, path: Path) -> str:
        real = path.resolve()
        # The Hugging Face cache stores LFS files as blobs named by their sha256.
        if path.is_symlink() and HEX64.fullmatch(real.name):
            return real.name
        st = real.stat()
        key = f"{real}|{st.st_size}|{st.st_mtime_ns}"
        if key in self.data:
            return self.data[key]
        h = hashlib.sha256()
        with real.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        self.data[key] = h.hexdigest()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            tmp.replace(self.path)
        return self.data[key]


def file_sha256(path: Path, cache_dir: Path | None = None) -> str:
    return _HashCache(cache_dir).sha256(path)


def model_fingerprint(model_dir: Path, cache_dir: Path | None = None) -> str | None:
    """Digest of the model config and every weight file in ``model_dir`` (None if no weights)."""
    if not model_dir.is_dir():
        return None
    cache = _HashCache(cache_dir)
    weights = {
        p.name: cache.sha256(p)
        for p in sorted(model_dir.iterdir())
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES
    }
    if not weights:
        return None
    meta = {n: cache.sha256(model_dir / n) for n in MODEL_META_FILES if (model_dir / n).is_file()}
    return digest("model", meta, weights)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Digest of the tokenizer files (when it was loaded from a directory) and of its observed
    behaviour: token ids of a fixed text and renders of a fixed chat with and without tools.
    Behaviour is included because the same files can tokenize or render differently under
    another library version, and that would change what the probes send."""
    files: dict[str, str] = {}
    src = Path(str(getattr(tokenizer, "name_or_path", "") or ""))
    if str(src) and src.is_dir():
        files = {
            n: hashlib.sha256((src / n).read_bytes()).hexdigest()
            for n in TOKENIZER_FILES
            if (src / n).is_file()
        }

    def attempt(fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # a template that rejects tools is itself a behaviour
            return f"error: {type(exc).__name__}"

    def render(**kw: Any) -> Any:
        return tokenizer.apply_chat_template(
            PROBE_CHAT, add_generation_prompt=True, tokenize=False, **kw
        )

    behaviour = {
        "class": type(tokenizer).__name__,
        "size": attempt(lambda: len(tokenizer)),
        "ids": attempt(lambda: list(tokenizer.encode(PROBE_TEXT, add_special_tokens=False))),
        "ids_special": attempt(lambda: list(tokenizer.encode(PROBE_TEXT))),
        "special": attempt(lambda: {k: str(v) for k, v in tokenizer.special_tokens_map.items()}),
        "chat": attempt(render),
        "chat_tools": attempt(lambda: render(tools=PROBE_TOOLS)),
    }
    return digest("tokenizer", files, behaviour)


def suite_fingerprint(suite: str) -> str:
    """Digest of every data file of a shipped suite (prompts, corpus, schemas)."""
    root = resources.files("parity") / "suites" / suite
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.iterdir(), key=lambda p: p.name)
        if p.is_file()
    }
    if not files:
        raise ValueError(f"suite data {suite!r} not found")
    return digest("suite", suite, files)


def code_fingerprint() -> str:
    """Digest of the installed parity package's source, data and templates."""
    root = Path(parity.__file__).parent
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in CODE_SUFFIXES and "__pycache__" not in p.parts
    }
    return digest("code", files)


def environment() -> dict[str, str]:
    """Interpreter and library versions that can change numerics or template rendering."""
    env = {"python": platform.python_version(), "parity": parity.__version__}
    for lib in LIBRARIES:
        try:
            env[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            env[lib] = "absent"
    return env


def git_state(root: Path | None = None) -> dict[str, Any] | None:
    """Commit and dirty flag of the checkout the code runs from (None outside a checkout)."""
    root = root or Path(parity.__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "src_dirty": bool(dirty)}
