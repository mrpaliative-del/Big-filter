import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

import picklog
from scanner import draw as D
from scanner.combined import run_both
from scanner.feeds import fetch_league_history, parse_football_data_csv


def league(n_teams=10, n_games=300, seed=3, div="I2"):
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    att, dfn = rng.normal(0, .25, n_teams), rng.normal(0, .2, n_teams)
    rows, d0 = [], pd.Timestamp("2026-01-01")
    for k in range(n_games):
        h, a = rng.choice(n_teams, 2, replace=False)
        hg, ag = rng.poisson(np.exp(att[h] + dfn[a] + .25)), rng.poisson(np.exp(att[a] + dfn[h]))
        rows.append(dict(Div=div, Date=d0 + pd.Timedelta(days=k), Home=teams[h], Away=teams[a], FTHG=hg, FTAG=ag, HST=rng.poisson(4.5), AST=rng.poisson(3.8),
                         HC=rng.poisson(5.5), AC=rng.poisson(4.5), HF=rng.poisson(11), AF=rng.poisson(11),
                         AvgH=round(rng.uniform(1.6, 3.5), 2), AvgD=round(rng.uniform(3.0, 3.8), 2), AvgA=round(rng.uniform(2.0, 5), 2)))
    return pd.DataFrame(rows)


def test_vectorised_tau_matches_the_scalar_original():
    hg, ag = np.array([0, 0, 1, 1, 2, 3]), np.array([0, 1, 0, 1, 2, 0])
    hx, ax = np.full(6, 1.4), np.full(6, 1.1)
    vec = D._tau_vec(hg, ag, hx, ax, -0.12)
    ref = [D.dixon_coles_tau(h, a, 1.4, 1.1, -0.12) for h, a in zip(hg, ag)]
    assert np.allclose(vec, ref)


def test_predict_dc_prob_matches_a_plain_reference():
    model = {"teams": {"A", "B"}, "attack": {"A": .2, "B": -.1}, "defense": {"A": -.05, "B": .1}, "home_adv": .3, "rho": -.1}
    hx, ax = np.exp(.2 + .1 + .3), np.exp(-.1 - .05)
    m = np.array([[poisson.pmf(i, hx) * poisson.pmf(j, ax) * D.dixon_coles_tau(i, j, hx, ax, -.1) for j in range(7)] for i in range(7)])
    assert D.predict_dc_prob(model, "A", "B") == pytest.approx(np.trace(m / m.sum()), abs=1e-12)
    assert D.predict_dc_prob(model, "A", "Nobody") is None


def test_kelly_and_pick_rule():
    assert D.calculate_kelly_stake(0.25, 3.3, 1000) == 0.0                       # no edge -> no stake
    assert D.calculate_kelly_stake(0.60, 3.3, 1000) == 30.0                      # capped at 3% of the bankroll
    assert D.calculate_kelly_stake(0.32, 3.4, 1000) == pytest.approx(1000 * (((2.4 * .32) - .68) / 2.4) * 0.25, abs=0.01)
    ok = D.evaluate_pick(0.36, 3.2)
    assert ok["qualified"] and ok["edge"] == pytest.approx(0.36 - 1 / 3.2) and ok["stake"] > 0 and ok["reason"] == ""
    assert not D.evaluate_pick(0.30, 3.2)["qualified"] and "edge below" in D.evaluate_pick(0.30, 3.2)["reason"]
    hi = D.evaluate_pick(0.40, 4.5)
    assert not hi["qualified"] and "above 4.10" in hi["reason"]


def test_dixon_coles_fit_recovers_home_advantage():
    df = D.to_script_schema(league(n_games=400))
    m = D.fit_dixon_coles_weighted(df, df["Date"].max())
    assert 0.1 < m["home_adv"] < 0.6 and -0.5 <= m["rho"] <= 0.5 and abs(np.mean(list(m["attack"].values()))) < 1e-6


def test_features_use_only_earlier_games():
    df = D.to_script_schema(league(n_games=120))
    a, _ = D.build_features_and_latest_state(df.copy())
    b = df.copy()
    b.loc[b.index[-1], ["FTHG", "FTAG", "HST", "AST"]] = [9, 9, 20, 20]               # change only the last game
    f, _ = D.build_features_and_latest_state(b)
    assert np.allclose(a[D.FEATURE_COLS].iloc[-1].values.astype(float), f[D.FEATURE_COLS].iloc[-1].values.astype(float), equal_nan=True)
    assert a[D.FEATURE_COLS].iloc[0].isna().all()                                       # needs a 5-game window first


def test_train_league_predictor_reasons_and_persistence(tmp_path):
    hist = league(n_games=300)
    lm = D.train_league("I2", D.to_script_schema(hist))
    assert lm is not None and 0 <= lm.dc_w <= 1 and abs(lm.dc_w + lm.xgb_w - 1) < 1e-9
    dp = D.DrawPredictor({"I2": lm}, hist)
    r = dp.predict_fixture("team1", "TEAM2", (2.2, 3.3, 3.2))
    assert r["league"] == "Italian Serie B" and 0 < r["prob"] < 1 and abs(r["prob"] - (lm.dc_w * r["dc_prob"] + lm.xgb_w * r["xgb_prob"])) < 1e-9
    assert r["edge"] == pytest.approx(r["prob"] - 1 / 3.3) and r["fair_odds"] == round(1 / r["prob"], 2)
    assert dp.predict_fixture("Team1", "Team2", None)["reason"].startswith("no 1X2 odds")
    assert "above 4.10" in dp.predict_fixture("Team1", "Team2", (2.2, 5.0, 3.2))["reason"]
    assert dp.predict_fixture("Team1", "Nobody FC", (2.2, 3.3, 3.2)) is None             # not covered
    p = tmp_path / "m.pkl"
    dp.save_models(p)
    assert set(D.DrawPredictor.load_models(p)) == {"I2"} and D.DrawPredictor.load_models(tmp_path / "none.pkl") == {}


def test_weekly_scan_offline_flow():
    hist = league(n_games=300)
    fx = pd.DataFrame([dict(Div="I2", Date="21/11/2026", Time="15:00", HomeTeam="Team1", AwayTeam="Team2", AvgH=2.2, AvgD=3.3, AvgA=3.2),
                       dict(Div="E0", Date="21/11/2026", Time="15:00", HomeTeam="X", AwayTeam="Y", AvgH=2.2, AvgD=3.3, AvgA=3.2)])
    out = D.weekly_scan(fx, {"I2": D.to_script_schema(hist)}, min_edge=-1.0, log=lambda *_: None)
    assert list(out.columns) == D.PICK_COLUMNS and len(out) == 1 and out.iloc[0].League == "Italian Serie B"
    none = D.weekly_scan(fx, {"I2": D.to_script_schema(hist)}, min_edge=0.9, log=lambda *_: None)
    assert none.empty and list(none.columns) == D.PICK_COLUMNS


class StubDraw:
    """Stands in for DrawPredictor so integration tests are instant."""
    def predict_fixture(self, home, away, odds):
        if home != "Pisa":
            return None
        if not odds:
            return {"div": "I2", "league": "Italian Serie B", "qualified": False, "reason": "no 1X2 odds for this fixture"}
        ev = D.evaluate_pick(0.36, odds[1])
        return {"div": "I2", "league": "Italian Serie B", "dc_prob": 0.35, "xgb_prob": 0.37, "prob": 0.36, "odds": odds[1], "dc_weight": 0.5, "booster": "stub", **ev}


TXT = "ITALY: Serie B\n17:30 Pisa - Sudtirol  -  [ 1.80 | 3.30 | 4.50 ]\nENGLAND: Premier League\n12:00 Arsenal - Spurs  -  [ 1.50 | 4.20 | 6.00 ]\nITALY: Serie B\n09:00 Pisa - Palermo  - "


def test_third_scanner_panel_in_kickoff_order_and_only_where_covered():
    r = run_both(TXT, min_threshold=0.0, selection_source="odds_first", draw=StubDraw())
    assert r["mode"] == "all" and [m["time"] for m in r["ranked_fixtures"]] == ["09:00", "12:00", "17:30"]
    by = {m["match"]: m for m in r["ranked_fixtures"]}
    assert by["Arsenal vs Spurs"]["draw"] is None                                        # league not covered -> "DRAW –"
    assert by["Pisa vs Sudtirol"]["draw"]["qualified"] and by["Pisa vs Sudtirol"]["draw"]["status"] == "★ DRAW PICK" and r["summary"]["qualified_draw"] == 1
    assert by["Pisa vs Palermo"]["draw"]["prob"] is None and by["Pisa vs Palermo"]["draw"]["status"].startswith("◇")
    lines = {l.split(" | ")[2]: l for l in r["copyable_block"].splitlines()}
    assert "DRAW rating:36.0%" in lines["Pisa vs Sudtirol"] and "edge +" in lines["Pisa vs Sudtirol"] and "stake" in lines["Pisa vs Sudtirol"]
    assert "DRAW –" in lines["Arsenal vs Spurs"] and "DRAW n/a" in lines["Pisa vs Palermo"]
    assert any(w.startswith("Draw rating:") and "40% line" in w for w in r["warnings"])


def test_draw_only_mode_and_show_only_qualified():
    r = run_both(TXT, min_threshold=0.0, draw=StubDraw(), scanners=("draw",), selection_source="odds_first")
    assert r["mode"] == "draw" and all("CLASSIC" not in l and "REGULAR" not in l for l in r["copyable_block"].splitlines())
    only = run_both(TXT, min_threshold=0.0, draw=StubDraw(), show_only_qualified=True, selection_source="odds_first")
    assert [m["match"] for m in only["ranked_fixtures"]] == ["Pisa vs Sudtirol"]
    off = run_both(TXT, min_threshold=0.0, selection_source="odds_first")                 # no predictor -> two scanners as before
    assert off["mode"] == "both" and "DRAW" not in off["copyable_block"]


def test_pick_log_tracks_draw_return_on_stake(tmp_path):
    p = tmp_path / "picks.csv"
    r = run_both(TXT, min_threshold=0.0, draw=StubDraw(), selection_source="odds_first")
    n = picklog.append_picks(r["panels"], p, picklog.date(2026, 11, 21))
    draws = [x for x in r["panels"] if x["profile"] == "draw"]
    assert len(draws) == 1 and n == len(r["panels"])                                       # only the scored draw is logged
    assert picklog.settle(p, [{"home": "Pisa", "away": "Sudtirol", "hg": 1, "ag": 1, "date": None}]) >= 1
    out = picklog.report(p)
    assert "draw scanner" in out and "return on stake" in out and "+" in out.split("return on stake")[1][:6]    # a draw at 3.30 won


def test_feeds_keep_corners_fouls_and_download_seasons():
    csv = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HST,AST,HC,AC,HF,AF,AvgH,AvgD,AvgA\n"
           "I2,05/10/25,15:00,Pisa,Bari,1,1,5,3,6,4,12,10,2.1,3.3,3.5\n")
    df = parse_football_data_csv(csv)
    assert df.HC.iloc[0] == 6 and df.AF.iloc[0] == 10
    def fake(url):
        if "2425" in url:
            raise RuntimeError("404")
        return csv
    out = fetch_league_history("I2", ["2425", "2526"], fetch=fake)
    assert len(out) == 1 and out.Season.iloc[0] == "2025/26" and {"HC", "AC", "HF", "AF"} <= set(out.columns)
