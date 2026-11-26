import pathlib

import pytest

import picklog
from scanner.calibration import Calibration
from scanner.formmodel import FormModel, FormPredictor, load_history
from scanner.selection import run_selection

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fd_2025-26_2026-27.csv"
TXT = "X: Y\n15:00 Even - Match  -  [ 2.00 | 3.40 | 3.80 ]\n16:00 Big - Small  -  [ 1.20 | 6.00 | 12.00 ]"


def test_prices_come_from_the_pasted_odds_and_window_filters():
    r = run_selection(TXT)
    by = {x["match"]: x for x in r["ranked_fixtures"]}
    even = by["Even vs Match"]
    prices = {c["key"]: c["price"] for c in even["candidates"]}
    assert prices["1x"] == pytest.approx(1 / (1 / 2.0 + 1 / 3.4), abs=0.005) and prices["12"] == pytest.approx(1 / (1 / 2.0 + 1 / 3.8), abs=0.005)
    assert even["status"] == "★ PICK" and even["pick"]["key"] == "12" and 1.30 <= even["pick"]["price"] <= 1.35
    assert even["pick"]["ev"] < 0                                              # honest: no edge against the bookmaker price
    big = by["Big vs Small"]
    assert big["status"].startswith("◆") and big["pick"] is None               # every result market priced below 1.30
    assert big["best_by_stats"]["p"] > 0.9
    assert r["ranked_fixtures"][0]["match"] == "Even vs Match"
    assert any("does not create value" in w for w in r["warnings"])


def test_window_is_adjustable_and_picks_the_most_probable_selection_in_it():
    r = run_selection(TXT, price_window=(1.20, 1.30))
    even = {x["match"]: x for x in r["ranked_fixtures"]}["Even vs Match"]
    assert even["pick"]["key"] == "1x"                                         # 1X priced ~1.26 is now the only one inside
    wide = run_selection(TXT, price_window=(1.0, 20.0))
    assert all(x["pick"] for x in wide["ranked_fixtures"])
    best = {x["match"]: x for x in wide["ranked_fixtures"]}["Even vs Match"]
    assert best["pick"]["p"] == max(c["p"] for c in best["candidates"])


def test_calibration_changes_the_probability_used():
    shrink = Calibration({"sel_12": {"x": [0.0, 1.0], "y": [0.0, 0.5]}})
    r = run_selection(TXT, calibration=shrink)
    even = {x["match"]: x for x in r["ranked_fixtures"]}["Even vs Match"]
    c12 = next(c for c in even["candidates"] if c["key"] == "12")
    base = next(c for c in {x["match"]: x for x in run_selection(TXT)["ranked_fixtures"]}["Even vs Match"]["candidates"] if c["key"] == "12")
    assert c12["p"] == pytest.approx(base["p"] * 0.5)


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_no_odds_uses_the_form_model_and_asks_for_a_price():
    fp = FormPredictor(FormModel.load(ROOT / "form_model.json"), load_history([DATA]))
    r = run_selection("ENGLAND: Premier League\n15:00 Man City - Liverpool  - ", form=fp)
    row = r["ranked_fixtures"][0]
    assert row["source"] == "form" and r["summary"]["from_form"] == 1
    assert row["status"] in ("◇ CHECK PRICE", "◆ NO SELECTION IN WINDOW")
    if row["pick"]:
        assert row["pick"]["price"] is None and "needs price" in r["copyable_block"]
    assert run_selection("X: Y\n15:00 Nobody FC - Nowhere  - ", form=fp)["summary"]["skipped_no_data"] == 1


def test_pick_log_for_single_selections_measures_return(tmp_path):
    p = tmp_path / "picks.csv"
    rows = run_selection(TXT)["ranked_fixtures"]
    assert picklog.append_picks(rows, p, picklog.date(2026, 11, 16)) == 1       # only the fixture that had a pick
    assert picklog.settle(p, [{"home": "Even", "away": "Match", "hg": 2, "ag": 0, "date": None}]) == 1
    out = picklog.report(p)
    assert "single selection" in out and "return on stake" in out and "+" in out.split("return on stake")[1][:6]   # won at ~1.31
