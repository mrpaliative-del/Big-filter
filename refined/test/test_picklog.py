import csv
from datetime import date

import picklog
from scanner.calibration import Calibration
from scanner.pipeline import run_pipeline

TXT = open(__import__("pathlib").Path(__file__).parent / "sample_odds.txt", encoding="utf-8").read()


def rows():
    return run_pipeline(TXT, min_threshold=0.0, max_xg=10)["ranked_fixtures"]


def test_append_dedupes_by_date_and_match(tmp_path):
    p = tmp_path / "picks.csv"
    n = picklog.append_picks(rows(), p, date(2026, 11, 16))
    assert n == 7 and picklog.append_picks(rows(), p, date(2026, 11, 16)) == 0
    assert picklog.append_picks(rows(), p, date(2026, 11, 17)) == 7


def test_settle_by_name_window_and_hits(tmp_path):
    p = tmp_path / "picks.csv"
    picklog.append_picks(rows(), p, date(2026, 11, 16))
    res = [{"home": "germany", "away": "NETHERLANDS", "hg": 1, "ag": 1, "date": date(2026, 11, 16)},
           {"home": "Panama", "away": "Curacao", "hg": 0, "ag": 3, "date": date(2026, 12, 25)}]  # too late: ignored
    assert picklog.settle(p, res) == 1
    r = next(x for x in csv.DictReader(open(p, encoding="utf-8")) if x["home"] == "Germany")
    assert (r["hit_1x"], r["hit_h15"], r["hit_u25"], r["hit_all3"]) == ("1", "1", "1", "1")  # 1-1: draw, 2 goals


def test_report_and_wilson(tmp_path):
    p = tmp_path / "picks.csv"
    picklog.append_picks(rows(), p, date(2026, 11, 16))
    picklog.settle(p, [{"home": "Germany", "away": "Netherlands", "hg": 3, "ag": 0, "date": None}])
    out = picklog.report(p)
    assert "1 settled picks" in out and "do not tune on this yet" in out
    lo, hi = picklog.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert picklog.report(tmp_path / "none.csv") == "No settled picks yet."
