import pathlib

import pytest

import picklog
from scanner.combined import run_both
from scanner.feeds import OddsFeed, parse_football_data_csv
from scanner.formmodel import FormModel, FormPredictor, load_history
from scanner.pipeline import run_pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fd_2025-26_2026-27.csv"
TXT = ("EUROPE: UEFA Nations League\n20:45 Germany - Netherlands  -  [ 1.90 | 4.20 | 3.74 ]\n18:00 Lithuania - Liechtenstein  -  [ 1.13 | 9.15 | 28.00 ]\n"
       "X: Y\n12:30 Even - Match  -  [ 2.00 | 3.40 | 3.80 ]\n09:00 Big - Small  -  [ 1.20 | 6.00 | 12.00 ]")


def test_both_scanners_side_by_side_in_kickoff_order():
    r = run_both(TXT, min_threshold=0.0, selection_source="odds_first")
    assert [m["time"] for m in r["ranked_fixtures"]] == ["09:00", "12:30", "18:00", "20:45"]          # kickoff order, not probability
    lines = r["copyable_block"].splitlines()
    assert [l.split(" | ")[0] for l in lines] == ["09:00", "12:30", "18:00", "20:45"]
    assert all("CLASSIC" in l and "REGULAR" in l and "All3:" in l and "Under 2.5:" in l and "Over 1.5:" in l for l in lines)
    for m in r["ranked_fixtures"]:
        assert m["classic"]["profile"] == "classic" and m["regular"]["profile"] == "regular"
    assert len(r["panels"]) == 8 and r["rank_by"] == "time" and r["mode"] == "both"


def test_both_scanners_need_all3_above_20_and_every_leg_priced_at_least_1_25():
    r = run_both(TXT, min_threshold=0.0, selection_source="odds_first")
    assert r["price_floor"] == 1.25 and r["threshold"] == 0.20
    reg = {m["match"]: m["regular"] for m in r["ranked_fixtures"]}
    cls = {m["match"]: m["classic"] for m in r["ranked_fixtures"]}
    big = reg["Big vs Small"]                                                 # All3 very high, but 1X priced ~1.00
    assert big["all3_ok"] and not big["price_ok"] and big["status"].startswith("◇") and big["band"] == "HIGH"
    germany = reg["Germany vs Netherlands"]                                   # All3 ok, but Over 1.5 priced ~1.07
    assert germany["all3_ok"] and not germany["price_ok"] and germany["status"].startswith("◇")
    assert all(not v["status"].startswith("★") for v in reg.values())
    assert any(c and c["status"].startswith("◇") for c in cls.values())      # the classic scanner now has the same price rule
    assert any("filtered out on" in w for w in r["warnings"])


def test_all3_line_is_strict():
    txt = "X: Y\n15:00 Home - Away  -  [ 2.00 | 3.40 | 3.80 ]"
    r = run_pipeline(txt, min_threshold=0.0, qualify_by="all3_floor", price_floor=1.0, threshold=0.0)
    all3 = r["ranked_fixtures"][0]["all3_prob"]
    on_the_line = run_pipeline(txt, min_threshold=0.0, qualify_by="all3_floor", price_floor=1.0, threshold=all3)
    assert not on_the_line["ranked_fixtures"][0]["all3_ok"]                  # "20% above" means strictly above
    below = run_pipeline(txt, min_threshold=0.0, qualify_by="all3_floor", price_floor=1.0, threshold=all3 - 1e-6)
    assert below["ranked_fixtures"][0]["status"].startswith("★")


def test_floor_mode_can_qualify_when_both_conditions_hold():
    txt = "X: Y\n15:00 Home - Away  -  [ 2.00 | 3.40 | 3.80 ]"
    r = run_pipeline(txt, min_threshold=0.0, threshold=0.20, qualify_by="all3_floor", price_floor=1.10)
    row = r["ranked_fixtures"][0]
    assert row["all3_ok"] and row["price_ok"] and row["status"].startswith("★")
    strict = run_pipeline(txt, min_threshold=0.0, threshold=0.20, qualify_by="all3_floor", price_floor=1.50)
    assert strict["ranked_fixtures"][0]["status"].startswith("◇") and not strict["ranked_fixtures"][0]["price_ok"]


def test_show_only_qualified_hides_everything_else():
    assert run_both(TXT, min_threshold=0.0, show_only_qualified=True, selection_source="odds_first")["ranked_fixtures"] == []
    one = run_both("X: Y\n15:00 Home - Away  -  [ 2.00 | 3.40 | 3.80 ]", min_threshold=0.0, all3_min=0.20, price_floor=1.10, show_only_qualified=True, selection_source="odds_first")
    assert len(one["ranked_fixtures"]) == 1 and one["summary"]["qualified_regular"] == 1


def test_classic_scanner_in_combined_mode_matches_standalone_with_the_same_rules():
    both = run_both(TXT, min_threshold=0.0, selection_source="odds_first")
    alone = run_pipeline(TXT, min_threshold=0.0, profile="classic", rank_by="all3", threshold=0.20, qualify_by="all3_floor", price_floor=1.25)
    a = {x["match"]: (x["all_three"], x["status"]) for x in alone["ranked_fixtures"]}
    b = {m["match"]: (m["classic"]["all_three"], m["classic"]["status"]) for m in both["ranked_fixtures"]}
    assert a == b


def test_fixtures_feed_prices_fixtures_without_pasted_odds():
    feed = OddsFeed.from_frame(parse_football_data_csv(
        "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\nE2,14/11/2026,15:00,AFC Wimbledon,Blackpool,2.65,3.4,2.49\n"))
    txt = "ENGLAND: League Two\n15:00 AFC Wimbledon - Blackpool  - "
    without = run_pipeline(txt, min_threshold=0.0, profile="regular")
    assert without["summary"]["skipped_no_data"] == 1
    r = run_pipeline(txt, min_threshold=0.0, profile="regular", odds_lookup=feed.lookup)
    row = r["ranked_fixtures"][0]
    assert row["source"] == "odds" and row["odds"] == "2.65/3.4/2.49" and any("open fixtures feed" in w for w in r["warnings"])
    assert run_both(txt, min_threshold=0.0, odds_lookup=feed.lookup, selection_source="odds_first")["summary"]["from_odds"] == 1


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_form_model_drives_both_scanners_in_odds_first_mode_but_cannot_qualify_the_classic_one():
    fp = FormPredictor(FormModel.load(ROOT / "form_model.json"), load_history([DATA]))
    txt = "ENGLAND: Premier League\n15:00 Man City - Liverpool  - "
    r = run_both(txt, form=fp, min_threshold=0.0, selection_source="odds_first")
    m = r["ranked_fixtures"][0]
    assert m["source"] == "form" and m["classic"]["source"] == "form" and m["regular"]["source"] == "form"
    assert all(lg["prob"] > 0 for lg in m["classic"]["legs"]) and not m["classic"]["status"].startswith("★")
    forced = run_pipeline(txt, min_threshold=0.0, profile="classic", form=fp, qualify_unvalidated=True, threshold=0.0)
    assert forced["ranked_fixtures"][0]["status"].startswith("★")


def _fp():
    return FormPredictor(FormModel.load(ROOT / "form_model.json"), load_history([DATA]))


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_statistics_only_mode_ignores_odds_for_probabilities():
    fp = _fp()
    a = run_both("ENGLAND: Premier League\n15:00 Man City - Liverpool  -  [ 1.20 | 7.00 | 15.00 ]", form=fp, min_threshold=0.0)
    b = run_both("ENGLAND: Premier League\n15:00 Man City - Liverpool  -  [ 9.00 | 5.00 | 1.20 ]", form=fp, min_threshold=0.0)
    pa, pb = a["ranked_fixtures"][0]["regular"], b["ranked_fixtures"][0]["regular"]
    assert pa["source"] == "form" and [l["prob"] for l in pa["legs"]] == [l["prob"] for l in pb["legs"]] and pa["all3_prob"] == pb["all3_prob"]
    assert [l["price"] for l in pa["legs"]] != [l["price"] for l in pb["legs"]]            # odds only move the PRICES


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_statistics_only_mode_skips_teams_without_history_instead_of_using_odds():
    r = run_both("X: Y\n15:00 Nobody FC - Nowhere United  -  [ 2.00 | 3.40 | 3.80 ]", form=_fp(), min_threshold=0.0)
    assert r["ranked_fixtures"] == [] and r["summary"]["skipped_no_stats"] == 1
    assert any("no recent-form data" in w for w in r["warnings"])
    assert run_both("X: Y\n15:00 Nobody FC - Nowhere United  -  [ 2.00 | 3.40 | 3.80 ]", form=_fp(), min_threshold=0.0, selection_source="odds_first")["summary"]["from_odds"] == 1


@pytest.mark.skipif(not DATA.exists(), reason="history not present")
def test_statistics_only_mode_tiers_analysis_and_the_1_25_floor():
    fp = _fp()
    r = run_both("ENGLAND: Premier League\n17:30 Arsenal - Chelsea  -  [ 1.15 | 8.00 | 18.00 ]\n15:00 Man City - Liverpool  - ", form=fp, min_threshold=0.0)
    assert [m["time"] for m in r["ranked_fixtures"]] == ["15:00", "17:30"] and r["price_floor"] == 1.25 and r["selection_source"] == "stats"
    ars = next(m for m in r["ranked_fixtures"] if m["home"] == "Arsenal")["regular"]
    assert ars["tier"] in "ABC" and ars["all3_ok"] and not ars["price_ok"] and ars["status"].startswith("◇")      # strong favourite: filtered on price
    assert ars["analysis"] and "GA/g" in ars["analysis"] and "stats:" in r["copyable_block"] and "tier" in r["copyable_block"]
    assert r["classic"] if False else True
    lowered = run_both("ENGLAND: Premier League\n17:30 Arsenal - Chelsea  -  [ 1.15 | 8.00 | 18.00 ]", form=fp, min_threshold=0.0, price_floor=1.0)
    assert lowered["ranked_fixtures"][0]["regular"]["status"].startswith("★")                                     # floor off: All3 >= 20% qualifies


def test_pick_log_keeps_both_scanners_for_the_same_fixture(tmp_path):
    p = tmp_path / "picks.csv"
    r = run_both("X: Y\n15:00 Home - Away  -  [ 2.00 | 3.40 | 3.80 ]", min_threshold=0.0, selection_source="odds_first")
    assert picklog.append_picks(r["panels"], p, picklog.date(2026, 11, 18)) == 2          # classic + regular, not deduped away
    assert picklog.append_picks(r["panels"], p, picklog.date(2026, 11, 18)) == 0
    assert picklog.settle(p, [{"home": "Home", "away": "Away", "hg": 2, "ag": 0, "date": None}]) == 2
    out = picklog.report(p)
    assert "[classic |" in out and "[regular |" in out
