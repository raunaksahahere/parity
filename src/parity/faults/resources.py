"""Linux process memory and temperature readings for the validation harness.

Peak RSS comes from ``VmHWM`` in ``/proc/<pid>/status``; writing ``5`` to
``/proc/<pid>/clear_refs`` resets it, so each repetition gets its own peak. Temperatures come
from hwmon (CPU ``k10temp``/``coretemp``, GPU ``amdgpu``, ACPI zones). Everything degrades to
``None`` on systems without these files.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

HWMON = Path("/sys/class/hwmon")
THERMAL_SENSORS = ("k10temp", "coretemp", "amdgpu", "acpitz")


def reset_peak_rss(pid: int | str = "self") -> bool:
    try:
        Path(f"/proc/{pid}/clear_refs").write_text("5")
    except OSError:
        return False
    return True


def peak_rss_mb(pid: int | str = "self") -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def max_temperature(root: Path = HWMON) -> float | None:
    """Hottest reading (deg C) over the CPU, GPU and ACPI sensors, or None if unavailable."""
    temps: list[float] = []
    for dev in root.glob("hwmon*") if root.exists() else []:
        try:
            name = (dev / "name").read_text().strip()
        except OSError:
            continue
        if name not in THERMAL_SENSORS:
            continue
        for f in dev.glob("temp*_input"):
            try:
                value = int(f.read_text().strip()) / 1000
            except (OSError, ValueError):
                continue
            if 0 < value < 150:
                temps.append(value)
    return max(temps) if temps else None


def wait_until_cool(
    limit_c: float,
    log: Callable[[str], None] = print,
    poll_s: float = 5.0,
    max_wait_s: float = 1800.0,
    read: Callable[[], float | None] = max_temperature,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Block until the hottest sensor is at or below ``limit_c``; returns seconds waited."""
    waited = 0.0
    temp = read()
    if temp is not None and temp > limit_c:
        log(f"  cooling down: {temp:.0f} C > {limit_c:.0f} C")
    while temp is not None and temp > limit_c and waited < max_wait_s:
        sleep(poll_s)
        waited += poll_s
        temp = read()
    return waited
