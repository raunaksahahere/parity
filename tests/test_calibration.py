from __future__ import annotations

from pathlib import Path

import pytest

from parity.stats.calibration import Calibration, calibrate_threshold


def test_threshold_sits_above_every_benign_value() -> None:
    cal = calibrate_threshold({"gpu": 1e-5, "cpu": 4e-4, "slots": 2e-5}, safety_factor=2.0)
    assert cal.threshold == pytest.approx(8e-4)
    assert all(cal.threshold > v for v in cal.benign_values.values())


def test_floor_applies_when_benign_noise_is_zero() -> None:
    assert calibrate_threshold({"gpu": 0.0}, floor=1e-4).threshold == 1e-4


def test_validation() -> None:
    with pytest.raises(ValueError):
        calibrate_threshold({})
    with pytest.raises(ValueError):
        calibrate_threshold({"a": 1.0}, safety_factor=0.5)


def test_round_trip(tmp_path: Path) -> None:
    cal = Calibration(model_id="m", revision="abc", suite="core-v1")
    cal.metrics["kl_mean"] = calibrate_threshold({"a": 1e-4})
    cal.save(tmp_path / "c.json")
    again = Calibration.load(tmp_path / "c.json")
    assert again.thresholds() == cal.thresholds() and again.revision == "abc"
