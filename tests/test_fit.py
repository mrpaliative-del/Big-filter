import numpy as np
import pytest

from scanner.fit import fit_ratings, read_results
from scanner.model import (AWAY_CONCEDED, AWAY_SCORED, BASE_GOALS,
                           HOME_CONCEDED, HOME_SCORED)


def simulate(n_teams=20, n_games=2500, seed=1):
    rng = np.random.default_rng(seed)
    A, D = np.exp(rng.normal(0, 0.25, n_teams)), np.exp(rng.normal(0, 0.25, n_teams))
    c = 0.5 * BASE_GOALS
    rows = []
    for _ in range(n_games):
        h, a = rng.choice(n_teams, 2, replace=False)
        lh = c * (HOME_SCORED * A[h] + AWAY_CONCEDED * D[a])
        la = c * (AWAY_SCORED * A[a] + HOME_CONCEDED * D[h])
        rows.append({"home": f"T{h}", "away": f"T{a}", "hg": int(rng.poisson(lh)), "ag": int(rng.poisson(la)), "date": None})
    return A, D, rows


def test_fit_recovers_strength_ordering():
    A, D, rows = simulate()
    fit = fit_ratings(rows)
    est_a = np.array([fit[f"T{i}"]["attack"] for i in range(len(A))])
    est_d = np.array([fit[f"T{i}"]["defence"] for i in range(len(D))])
    assert np.corrcoef(est_a, A)[0, 1] > 0.7
    assert np.corrcoef(est_d, D)[0, 1] > 0.6


def test_fit_beats_average_on_held_out_games():
    A, D, rows = simulate(seed=2)
    train, test = rows[:2000], rows[2000:]
    fit = fit_ratings(train)
    c = 0.5 * BASE_GOALS

    def ll(get):
        total = 0.0
        for m in test:
            (ah, dh), (aa, da) = get(m["home"]), get(m["away"])
            lh = c * (HOME_SCORED * ah + AWAY_CONCEDED * da)
            la = c * (AWAY_SCORED * aa + HOME_CONCEDED * dh)
            total += m["hg"] * np.log(lh) - lh + m["ag"] * np.log(la) - la
        return total

    fitted = ll(lambda t: (fit[t]["attack"], fit[t]["defence"]) if t in fit else (1.0, 1.0))
    flat = ll(lambda t: (1.0, 1.0))
    assert fitted > flat


def test_few_game_teams_omitted_and_values_in_range():
    rows = [{"home": "A", "away": "B", "hg": 1, "ag": 0, "date": None}] * 6 + [
        {"home": "C", "away": "A", "hg": 9, "ag": 0, "date": None}]
    fit = fit_ratings(rows, min_games=5)
    assert "C" not in fit and set(fit) == {"A", "B"}
    assert all(0.2 <= v <= 4.0 for r in fit.values() for v in r.values())


def test_read_results_flexible_headers(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("\ufeffDate,HomeTeam,AwayTeam,FTHG,FTAG\n13/11/2026,Arsenal,Chelsea,2,1\n14/11/2026,Chelsea,Arsenal,,\n", encoding="utf-8")
    rows = read_results(p)
    assert len(rows) == 1 and rows[0]["hg"] == 2 and rows[0]["date"].year == 2026


def test_read_results_missing_columns(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        read_results(p)


def test_combined_score_column_and_since_filter(tmp_path):
    from datetime import date
    p = tmp_path / "w.csv"
    p.write_text("Round,Date,Team 1,FT,Team 2\n1,Sat Sep 19 2020,Fiorentina,1-0,Torino\n2,Sun Oct 02 2022,Roma,2-2,Napoli\n3,Mon Oct 03 2022,Roma,,Napoli\n")
    assert len(read_results(p)) == 2
    rows = read_results(p, since=date(2022, 1, 1))
    assert len(rows) == 1 and (rows[0]["hg"], rows[0]["ag"]) == (2, 2)


def test_engsoccerdata_headers(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("Date,Season,home,visitor,FT,hgoal,vgoal\n2024-08-15,2024,Athletic Bilbao,Getafe CF,1-1,1,1\nNA,2024,A,B,NA,NA,NA\n")
    rows = read_results(p)
    assert len(rows) == 1 and rows[0]["away"] == "Getafe CF"
