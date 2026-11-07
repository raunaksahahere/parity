"""Power of the tools probe's trigger-agreement test as a function of the prompt count.

The calibrated benign disagreement rate is d0 = max(2 x Wilson upper bound of k/n, 0.05) for k
disagreements observed on the benign deployments (parity calibrate). The test rejects when the
observed disagreements reach the smallest count significant at the probe's per-test alpha
(alpha / 4 probes / 3 tests in the core suite). Power is P(Bin(n, delta) >= that count).

  uv run python scripts/tools_power.py
"""

from scipy import stats

from parity.probes.tools import N_TESTS, min_detectable
from parity.stats.calibration import FLOORS, wilson_upper

ALPHA = 0.05 / 4 / N_TESTS
TARGET_DELTA, TARGET_POWER = 0.25, 0.9


def power(n: int, k_benign: int, delta: float) -> tuple[float, int, float]:
    d0 = max(2 * wilson_upper(k_benign, n), FLOORS["tool_disagreement"])
    k_crit = min_detectable(n, d0, ALPHA)
    return d0, k_crit, float(stats.binom.sf(k_crit - 1, n, delta))


print(f"per-test alpha {ALPHA:.4g}; target: power >= {TARGET_POWER} at disagreement {TARGET_DELTA}")
print(
    "| prompts | benign disagreements | calibrated d0 | critical count "
    "| power at 0.25 | power at 0.5 |"
)
print("|---:|---:|---:|---:|---:|---:|")
for n in (24, 48, 72, 96, 120):
    for k in (0, 1, 2):
        d0, kc, p25 = power(n, k, TARGET_DELTA)
        _, _, p50 = power(n, k, 0.5)
        print(f"| {n} | {k} | {d0:.3f} | {kc} | {p25:.3f} | {p50:.3f} |")
