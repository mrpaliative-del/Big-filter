import csv
import pathlib

D = pathlib.Path(__file__).resolve().parent.parent / "data"


def test_football_data_extract_is_sane():
    rows = list(csv.DictReader(open(D / "fd_2025-26_2026-27.csv", encoding="utf-8")))
    assert len(rows) > 9000
    with_odds = [r for r in rows if r["AvgH"]]
    assert len(with_odds) > 9000
    for r in with_odds[::25]:
        book = sum(1 / float(r[k]) for k in ("AvgH", "AvgD", "AvgA"))
        assert 1.0 < book < 1.2, (r["Home"], r["Away"], book)
    assert 2.4 < sum(int(r["FTHG"]) + int(r["FTAG"]) for r in rows) / len(rows) < 3.2


def test_argentina_extract():
    rows = list(csv.DictReader(open(D / "arg_2024_2026.csv", encoding="utf-8")))
    assert 1000 < len(rows) < 3000 and all(r["Date"] >= "2024-01-01" for r in rows)
