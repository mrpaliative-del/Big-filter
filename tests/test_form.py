import pathlib

import numpy as np
import pandas as pd
import pytest

from scanner.features import ALL_COLS, build_features
from scanner.formmodel import FormModel, FormPredictor, load_history
from scanner.model import evaluate_profile, leg_outcomes
from scanner.pipeline import run_pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fd_2025-26_2026-27.csv"


def tiny():
    rows = []
    for i, (h, a, hg, ag) in enumerate([("A", "B", 1, 0), ("C", "D", 2, 2), ("B", "C", 0, 3), ("D", "A", 1, 1), ("A", "C", 2, 1), ("B", "D", 0, 0)]):
        rows.append(dict(Season="2026/27", Div="E0", Date=pd.Timestamp("2026-08-01") + pd.Timedelta(days=7 * i), Home=h, Away=a,
                         FTHG=hg, FTAG=ag, HS=10, AS=8, HST=4, AST=3))
    return pd.DataFrame(rows)


def test_features_do_not_leak_the_games_own_result():
    a, b = tiny(), tiny()
    b.loc[5, ["FTHG", "FTAG"]] = [4, 4]                       # change only the LAST game's score
    _, Fa, _ = build_features(a)
    _, Fb, _ = build_features(b)
    assert (Fa.iloc[5] == Fb.iloc[5]).all()                   # its own features unchanged
    assert not (Fa.iloc[:5].values != Fb.iloc[:5].values).any()


def test_features_cover_all_columns_and_react_to_history():
    _, F, eng = build_features(tiny())
    assert set(F.columns) == set(ALL_COLS)
    assert F.iloc[0]["mp_h"] == 0 and F.iloc[4]["mp_h"] == 2           # A had played twice before game 5
    assert F.iloc[4]["unb_h"] >= 1 and F.iloc[4]["h2h_n"] == 0
    assert eng.loc["A"][1] == "E0"


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_predictor_scores_same_league_fixtures_only():
    model = FormModel.load(ROOT / "form_model.json")
    fp = FormPredictor(model, load_history([DATA]))
    p = fp.predict_fixture("Arsenal", "Chelsea")
    assert p and all(0 < p[k] < 1 for k in ("1x", "12", "o15", "all3")) and p["all3"] <= min(p["1x"], p["12"], p["o15"]) + 1e-9
    assert fp.predict_fixture("Arsenal", "Barcelona") is None            # different leagues
    assert fp.predict_fixture("Arsenal", "No Such Team") is None
    assert fp.predict_fixture("  arsenal ", "CHELSEA") is not None        # name matching is forgiving


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_pipeline_uses_form_model_when_no_odds_and_it_can_qualify():
    model = FormModel.load(ROOT / "form_model.json")
    fp = FormPredictor(model, load_history([DATA]))
    txt = "ENGLAND: Premier League\n15:00 Man City - Liverpool  - \n16:00 Nobody FC - Nowhere United  - "
    r = run_pipeline(txt, min_threshold=0.0, form=fp, qualify_by="all3")
    row = r["ranked_fixtures"][0]
    assert row["source"] == "form" and row["xg"] is None and r["summary"]["from_form"] == 1 and r["summary"]["skipped_no_data"] == 1
    assert any("form model" in w for w in r["warnings"])
    assert row["status"] == ("★ QUALIFIED" if row["score"] >= 0.5 else "◆ MODERATE")


def test_regular_profile_all3_is_home_win_and_over_1_5():
    m = evaluate_profile(1.7, 0.9, "regular")
    assert set(m["legs"]) == {"1x", "12", "o15"} and m["all3"] <= m["min_leg"]
    assert leg_outcomes("regular", 2, 0)["all3"] and not leg_outcomes("regular", 1, 1)["all3"] and not leg_outcomes("regular", 1, 0)["all3"]


def test_regular_profile_default_ranks_by_all3_at_50():
    txt = "X: Y\n15:00 Big - Small  -  [ 1.20 | 6.00 | 12.00 ]\n16:00 Even - Match  -  [ 2.80 | 3.20 | 2.60 ]"
    r = run_pipeline(txt, min_threshold=0.0, qualify_by="all3")
    assert r["profile"] == "regular" and r["rank_by"] == "all3" and r["threshold"] == 0.50
    by = {x["match"]: x for x in r["ranked_fixtures"]}
    assert by["Big vs Small"]["status"].startswith("★") and by["Even vs Match"]["status"].startswith("◆")
    assert all("1X:" in l and "12:" in l and "Over 1.5:" in l for l in r["copyable_block"].splitlines())
