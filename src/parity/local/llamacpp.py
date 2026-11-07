"""Lifecycle management for a local llama.cpp ``llama-server`` process.

Used by the validation harness and the calibration tooling. The server must be
launched with its own directory as the working directory, otherwise ggml may
silently skip the Vulkan backend and fall back to CPU; ``start`` therefore
verifies from the server log that offload really happened when a GPU was
requested.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx


class ServerStartError(RuntimeError):
    pass


DEVICE_LINE = re.compile(r"^\s*([A-Za-z]+\d+):\s*(.+?)\s*$")
DEVICE_ID = re.compile(r"[A-Za-z]+\d+")


def parse_devices(list_devices_output: str) -> dict[str, str]:
    """Parse ``llama-server --list-devices`` into {"Vulkan1": "AMD Radeon RX 6500 (...)"}."""
    out = {}
    for line in list_devices_output.splitlines():
        m = DEVICE_LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def pick_device(devices: dict[str, str], wanted: str) -> str:
    """``wanted`` is a backend id ("Vulkan1") or a case-insensitive part of the device name
    ("RX 6500"). Device indices depend on enumeration order, which can change across driver
    updates or reboots, so selecting by name avoids silently landing on another GPU."""
    if DEVICE_ID.fullmatch(wanted):
        if wanted not in devices:
            raise ServerStartError(f"device {wanted} not found; available: {devices}")
        return wanted
    hits = [d for d, desc in devices.items() if wanted.lower() in desc.lower()]
    if len(hits) != 1:
        raise ServerStartError(
            f"expected exactly one device matching {wanted!r}, found {hits}; available: {devices}"
        )
    return hits[0]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass(frozen=True)
class LlamaServerConfig:
    binary: Path
    model: Path
    ctx_size: int = 4096
    # backend id ("Vulkan1") or part of the device name ("RX 6500"); None means CPU only
    device: str | None = None
    n_gpu_layers: int = 99
    threads: int | None = None
    parallel: int = 1
    batch_size: int | None = None
    ubatch_size: int | None = None
    chat_template_file: Path | None = None
    # Host-RAM prompt cache (llama.cpp --cache-ram, default 8192 MiB). It grows with every
    # distinct prompt; on small machines it pushes the system into swap. None keeps the default.
    cache_ram_mib: int | None = None
    extra_args: tuple[str, ...] = ()
    port: int = 0
    log_path: Path | None = None

    @property
    def uses_gpu(self) -> bool:
        return self.device is not None


@dataclass
class LlamaServer:
    config: LlamaServerConfig
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    port: int = field(default=0, init=False)
    log_path: Path | None = field(default=None, init=False)
    device_id: str | None = field(default=None, init=False)  # resolved backend id, e.g. Vulkan1

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _command(self) -> list[str]:
        c = self.config
        cmd = [
            str(c.binary.resolve()),
            "-m",
            str(c.model.resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(c.ctx_size),
            "--parallel",
            str(c.parallel),
            "--jinja",
            "-lv",
            "4",
        ]
        if c.uses_gpu:
            cmd += ["--device", str(self.device_id or c.device), "-ngl", str(c.n_gpu_layers)]
        else:
            cmd += ["--device", "none", "-ngl", "0"]
        if c.threads:
            cmd += ["-t", str(c.threads)]
        if c.batch_size:
            cmd += ["-b", str(c.batch_size)]
        if c.ubatch_size:
            cmd += ["-ub", str(c.ubatch_size)]
        if c.cache_ram_mib is not None:
            cmd += ["--cache-ram", str(c.cache_ram_mib)]
        if c.chat_template_file:
            cmd += ["--chat-template-file", str(c.chat_template_file.resolve())]
        cmd += list(c.extra_args)
        return cmd

    def start(self, timeout_s: float = 120.0) -> Self:
        c = self.config
        self.port = c.port or free_port()
        self.log_path = c.log_path or Path(f"/tmp/parity-llama-{self.port}.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("wb")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{c.binary.resolve().parent}:{env.get('LD_LIBRARY_PATH', '')}"
        if c.uses_gpu:
            listing = subprocess.run(
                [str(c.binary.resolve()), "--list-devices"],
                cwd=c.binary.resolve().parent,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.device_id = pick_device(
                parse_devices(listing.stdout + listing.stderr), str(c.device)
            )
        self._proc = subprocess.Popen(
            self._command(),
            cwd=c.binary.resolve().parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerStartError(f"llama-server exited early:\n{self.tail_log()}")
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.25)
        else:
            self.stop()
            raise ServerStartError(
                f"llama-server not healthy after {timeout_s}s:\n{self.tail_log()}"
            )
        if c.uses_gpu:
            self._assert_offloaded()
        return self

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path else ""

    def tail_log(self, lines: int = 25) -> str:
        return "\n".join(self.log_text().splitlines()[-lines:])

    def offloaded_layers(self) -> tuple[int, int] | None:
        m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", self.log_text())
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _assert_offloaded(self) -> None:
        got = self.offloaded_layers()
        if got is None or got[0] == 0:
            self.stop()
            raise ServerStartError(
                "GPU requested but llama-server did not offload any layers "
                "(Vulkan backend not loaded?)"
            )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
