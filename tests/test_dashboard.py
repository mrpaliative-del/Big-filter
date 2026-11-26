import pathlib
from html.parser import HTMLParser

import pytest

import ui
from scanner import profile as PR
from scanner.combined import run_both
from scanner.drawmodel import DrawScanner
from scanner.formmodel import FormModel, FormPredictor, load_history

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fd_2025-26_2026-27.csv"
VOID = {"meta", "br", "input", "img", "hr", "link"}


def balanced(html: str) -> bool:
    class P(HTMLParser):
        def __init__(s):
            super().__init__()
            s.stack, s.bad = [], False

        def handle_starttag(s, t, a):
            if t not in VOID:
                s.stack.append(t)

        def handle_endtag(s, t):
            if t in VOID:
                return
            if not s.stack or s.stack[-1] != t:
                s.bad = True
            if t in s.stack:
                while s.stack and s.stack.pop() != t:
                    pass
    p = P()
    p.feed(html)
    return not p.bad and not p.stack


def test_grades_follow_the_league_ranking():
    assert PR._grade(0.95) == "Excellent" and PR._grade(0.7) == "Good" and PR._grade(0.5) == "Average" and PR._grade(0.3) == "Poor" and PR._grade(0.05) == "Bad"
    assert PR._grade(0.1, invert=True) == "Lesser" and PR._grade(0.5, invert=True) == "Average" and PR._grade(0.9, invert=True) == "Plenty"
    assert PR._grade(None) == "No data" and PR._pct(3, [1, 2, 3, 4, 5]) == pytest.approx(0.5) and PR._pct(3, [1, 2]) is None


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_profile_has_the_nine_indicators_in_the_recommended_format():
    fp = FormPredictor(FormModel.load(ROOT / "form_model.json"), load_history([DATA]))
    pr = PR.fixture_profile(fp, "Arsenal", "Chelsea")
    assert [r[0] for r in pr["home"]["rows"]] == PR.ROWS == [r[0] for r in pr["away"]["rows"]]
    allowed = {"Excellent", "Good", "Average", "Poor", "Bad", "Lesser", "Plenty", "No data"}
    assert all(r[1] in allowed for t in ("home", "away") for r in pr[t]["rows"])
    assert all(pr[t]["rows"][i][1] in {"Lesser", "Average", "Plenty", "No data"} for t in ("home", "away") for i in (5, 6))     # fouls, cards
    txt = PR.profile_text(pr).splitlines()
    assert txt[0] == "√HOME TEAM (Arsenal)" and txt[1].startswith("- Goal record: ") and "√AWAY TEAM (Chelsea)" in txt
    assert pr["alignment"]["of"] == 9 and 0 <= pr["alignment"]["home_strong"] <= 9
    assert PR.fixture_profile(fp, "Arsenal", "Barcelona") is None


class StubDriver:
    def __init__(self, p):
        self.p = p

    def predict(self, feats, odds):
        return self.p

    def explain(self, feats, odds):
        return [{"driver": "Evenly matched (market)", "value": 0.05, "state": "yes"}]


class StubForm:
    def features_for(self, home, away, min_games=3):
        return ({}, "E1") if home != "Nobody" else None


def test_draw_scanner_line_is_strict_with_a_watchlist_below_it():
    odds = (2.5, 3.2, 3.0)
    assert DrawScanner(StubDriver(0.45), StubForm()).predict_fixture("A", "B", odds)["qualified"]
    exactly = DrawScanner(StubDriver(0.40), StubForm()).predict_fixture("A", "B", odds)
    assert not exactly["qualified"] and exactly["watch"]                      # "40% above" is strictly above
    w = DrawScanner(StubDriver(0.33), StubForm()).predict_fixture("A", "B", odds)
    assert not w["qualified"] and w["watch"] and "watchlist" in w["reason"]
    low = DrawScanner(StubDriver(0.20), StubForm()).predict_fixture("A", "B", odds)
    assert not low["qualified"] and not low["watch"]
    assert DrawScanner(StubDriver(0.33), StubForm()).predict_fixture("A", "B", odds, draw_min=0.30)["qualified"]   # adjustable line
    assert DrawScanner(StubDriver(0.45), StubForm()).predict_fixture("Nobody", "B", odds) is None
    assert "no 1X2 odds" in DrawScanner(StubDriver(0.45), StubForm()).predict_fixture("A", "B", None)["reason"]


TXT = ('ENGLAND: Championship\n17:30 Even - Match  -  [ 2.60 | 3.20 | 2.80 ]\n12:00 <img src=x onerror=alert(1)> - Evil  -  [ 1.20 | 6.00 | 12.00 ]\n'
       "09:15 Plain - Home  -  [ 3.00 | 3.30 | 2.30 ]")


class WatchDraw:
    def predict_fixture(self, home, away, odds, draw_min=None):
        if not odds:
            return None
        return {"div": "E1", "league": "English Championship", "prob": 0.33, "odds": odds[1], "implied": 1 / odds[1], "edge": 0.33 - 1 / odds[1],
                "fair_odds": round(1 / 0.33, 2), "qualified": False, "watch": True, "reason": "watchlist", "stake": 0.0,
                "drivers": [{"driver": "Evenly matched (market)", "value": 0.05, "state": "yes"}, {"driver": "Low combined scoring", "value": 3.0, "state": "against"}]}


def a_rows(res):
    return [p["legs"] for p in res["panels"] if p.get("legs")]


def test_dashboard_structure_order_escaping_and_filters():
    r = run_both(TXT, min_threshold=0.0, selection_source="odds_first", draw=WatchDraw(), show_only_qualified=False)
    html = ui.render_results(r, 0.2, 0.0)
    assert balanced(html) and "Fixtures by kickoff time" in html and "Scanner 1" in html and "Scanner 2" in html and "Scanner 3" in html
    seg = html[html.index("Fixtures by kickoff time"):]
    assert seg.index("09:15") < seg.index("12:00") < seg.index("17:30")                          # kickoff order on screen
    assert "&lt;img src=x onerror=alert(1)&gt;" in html and "<img src=x" not in html            # user text is escaped
    assert 'class="price p-low"' in html and 'class="ring"' in html and "HIGHLY EXPECTED 50–100%" in html and "WATCHLIST" in html
    assert "✓ Evenly matched" in html and "✗ Low combined scoring" in html
    q = run_both(TXT, min_threshold=0.0, selection_source="odds_first", draw=WatchDraw(), show_only_qualified=True)
    assert [m["match"] for m in q["ranked_fixtures"]] == ["Even vs Match"] and q["summary"]["qualified_regular"] == 1       # 27.6% All3, legs 1.43 / 1.35 / 1.29
    nothing = ("ENGLAND: Championship\n12:00 Big - Small  -  [ 1.20 | 6.00 | 12.00 ]\n09:15 Plain - Home  -  [ 3.00 | 3.30 | 2.30 ]")
    filtered = run_both(nothing, min_threshold=0.0, selection_source="odds_first", draw=WatchDraw(), show_only_qualified=True)
    page = ui.render_results(filtered, 0.2, 0.0)
    assert balanced(page) and "No fixture meets the scanner rules" in page and "Closest misses" in page
    assert "Draw watchlist" in page and filtered["summary"]["watch_draw"] == 2 and filtered["near_misses"]
    assert all(ls["price"] >= 1.01 for m in a_rows(filtered) for ls in m)


def test_report_text_and_summary_counts():
    r = run_both(TXT, min_threshold=0.0, selection_source="odds_first", draw=WatchDraw(), show_only_qualified=False)
    rep = r["report_text"]
    assert "SCANNER 1 · 1X · Home +1.5 · Under 2.5" in rep and "SCANNER 2 · 1X · 12 · Over 1.5" in rep and "SCANNER 3 · Draw: rating 33.0%" in rep
    s = r["summary"]
    assert s["scored_count"] == 3 and s["ranked_count"] == 3 and s["watch_draw"] == 3 and s["highly_expected"] == 0


def test_form_has_the_settings_the_app_reads():
    html = ui.render_form(0.20, 0.20, 6.0)
    assert balanced(html)
    for name in ("raw_text", "mode", "selection_source", "threshold", "price_floor", "draw_min", "show", "use_feed", "log_picks", "min_threshold", "max_xg", "qualify_by"):
        assert f'name="{name}"' in html
    assert 'value="1.25"' in html and 'value="0.40"' in html and 'value="qualified"' in html
    assert balanced(ui.render_error("<b>x</b>")) and "&lt;b&gt;x&lt;/b&gt;" in ui.render_error("<b>x</b>")
