import numpy as np
import pytest

from scanner.calibration import Calibration, apply_map, fit_isotonic
from scanner.pipeline import run_pipeline

ODDS = open(__import__("pathlib").Path(__file__).parent / "sample_odds.txt", encoding="utf-8").read()


def test_isotonic_is_monotone_and_recovers_rate():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 8000)
    y = (rng.uniform(size=8000) < p * 0.8).astype(float)        # true rate = 0.8 * predicted
    k = fit_isotonic(p, y)
    assert all(b >= a for a, b in zip(k["y"], k["y"][1:]))
    assert apply_map(k, 0.5) == pytest.approx(0.40, abs=0.04)


def test_apply_clamps_and_identity_without_map():
    assert apply_map(None, 0.7) == 0.7
    assert apply_map({"x": [0.2, 0.6], "y": [0.1, 0.5]}, 0.9) == 0.5


def test_save_load_roundtrip(tmp_path):
    c = Calibration({"u25": {"x": [0.3, 0.7], "y": [0.25, 0.55]}}, note="n")
    c.save(tmp_path / "c.json")
    d = Calibration.load(tmp_path / "c.json")
    assert d.apply("u25", 0.5) == pytest.approx(0.4) and bool(d) and not Calibration.load(tmp_path / "missing.json")


def test_pipeline_uses_calibrated_legs_and_default_line():
    shrink = Calibration({k: {"x": [0.0, 1.0], "y": [0.0, 0.8]} for k in ("1x", "h15", "u25", "all3")})
    base = run_pipeline(ODDS, min_threshold=0.0, max_xg=10, profile="classic")
    cal = run_pipeline(ODDS, min_threshold=0.0, max_xg=10, calibration_odds=shrink, profile="classic")
    assert base["threshold"] == 0.695 and cal["threshold"] == 0.55 and cal["calibrated"]
    b = {x["match"]: float(x["min_leg"].rstrip("%")) for x in base["ranked_fixtures"]}
    c = {x["match"]: float(x["min_leg"].rstrip("%")) for x in cal["ranked_fixtures"]}
    assert all(c[k] == pytest.approx(b[k] * 0.8, abs=0.1) for k in b)
    assert not any("not been validated" in w for w in cal["warnings"])
