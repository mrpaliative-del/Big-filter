import pytest
from scanner.model import Rating, Ratings
from scanner.pipeline import run_pipeline

TXT = "ENGLAND: Premier League\n15:00 Liverpool - Chelsea\n17:30 Arsenal - Man City"


def test_all_fixtures_survive_and_unrated_warning():
    r = run_pipeline(TXT, include_unrated=True)
    assert r["summary"]["total_fixtures"] == 2 and r["summary"]["ranked_count"] == 2
    assert r["warnings"] and r["summary"]["rated_fixtures"] == 0 and r["ranked_fixtures"][0]["source"] == "default"


def test_ratings_change_ranking_and_clear_warning():
    ratings = Ratings({"Arsenal": Rating(0.6, 0.7), "Man City": Rating(0.7, 0.7),
                       "Liverpool": Rating(1.1, 1.0), "Chelsea": Rating(1.1, 1.0)})
    r = run_pipeline(TXT, ratings=ratings, profile="classic")
    assert all("never tagged QUALIFIED" in w for w in r["warnings"])   # only the ratings notice
    assert r["ranked_fixtures"][0]["match"] == "Arsenal vs Man City"
    assert r["ranked_fixtures"][0]["score"] > r["ranked_fixtures"][1]["score"]


def test_max_xg_filters():
    assert run_pipeline(TXT, max_xg=1.0, include_unrated=True)["summary"]["ranked_count"] == 0


def test_qualified_tag_uses_threshold():
    assert run_pipeline(TXT, threshold=0.3, include_unrated=True, qualify_unvalidated=True)["ranked_fixtures"][0]["status"] == "★ QUALIFIED"


def test_unrated_skipped_by_default():
    r = run_pipeline(TXT)
    assert r["summary"]["ranked_count"] == 0 and r["summary"]["skipped_no_data"] == 2 and r["warnings"]


import pathlib

ODDS = (pathlib.Path(__file__).parent / "sample_odds.txt").read_text(encoding="utf-8")


def test_odds_page_gives_distinct_informed_scores():
    r = run_pipeline(ODDS, min_threshold=0.0, max_xg=10, profile="classic")
    rows = r["ranked_fixtures"]
    assert len(rows) == 7 and all(x["source"] == "odds" for x in rows)
    assert len({x["score"] for x in rows}) == len(rows)          # no longer identical
    assert r["summary"]["skipped_no_data"] == 1                   # Algeria U23 - Eswatini U23 (no odds)
    assert r["summary"]["skipped_postponed"] == 1 and not r["warnings"][1:]
    by = {x["match"]: x for x in rows}
    # Heavy favourite at home -> large expected goals, so Under 2.5 is the limiting market
    assert by["Lithuania vs Liechtenstein"]["xg"] > by["Ario Eslamshahr vs Saipa"]["xg"] - 0.5
    assert float(by["Namibia vs Cameroon"]["prob_1x"].rstrip("%")) < 40  # Cameroon 1.39 away favourite


def test_default_is_common_line_on_all_three_legs():
    r = run_pipeline(ODDS, min_threshold=0.0, max_xg=10, threshold=0.55, profile="classic")
    assert r["rank_by"] == "min" and r["threshold"] == 0.55
    for x in r["ranked_fixtures"]:
        legs = [float(x[k].rstrip("%")) for k in ("prob_1x", "prob_h_plus_1_5", "prob_u2_5")]
        assert (x["status"].startswith("★")) == (min(legs) / 100 >= 0.55 - 6e-4)   # every leg at/above the line
    scores = [x["score"] for x in r["ranked_fixtures"]]
    assert scores == sorted(scores, reverse=True)
    assert all("MinLeg:" in l and "All3:" in l for l in r["copyable_block"].splitlines())


def test_all3_mode_available_with_own_default_line():
    r = run_pipeline(ODDS, min_threshold=0.0, max_xg=10, rank_by="all3")
    assert r["threshold"] == 0.50 and r["summary"]["ranked_count"] == 7


def test_default_threshold_is_69_5():
    assert run_pipeline(ODDS, min_threshold=0.0, max_xg=10, profile="classic")["threshold"] == 0.695


def test_bad_rank_by_rejected():
    import pytest
    with pytest.raises(ValueError):
        run_pipeline(ODDS, rank_by="nope")


def test_only_odds_rows_can_qualify():
    from scanner.model import Rating, Ratings
    txt = "ENGLAND: Premier League\n15:00 Arsenal - Chelsea  -  [ 1.20 | 6.00 | 12.00 ]\n17:00 Foo - Bar  - "
    ratings = Ratings({"Foo": Rating(0.6, 0.6), "Bar": Rating(0.6, 0.6)})
    r = run_pipeline(txt, threshold=0.0, min_threshold=0.0, max_xg=10, ratings=ratings, qualify_by="all3")
    by = {x["match"]: x for x in r["ranked_fixtures"]}
    assert by["Arsenal vs Chelsea"]["status"].startswith("★") and by["Foo vs Bar"]["status"].startswith("◆")
    assert any("never tagged QUALIFIED" in w for w in r["warnings"])
    r2 = run_pipeline(txt, threshold=0.0, min_threshold=0.0, max_xg=10, ratings=ratings, qualify_unvalidated=True, qualify_by="all3")
    assert all(x["status"].startswith("★") for x in r2["ranked_fixtures"])


def test_leg_prices_follow_the_pasted_1x2_prices():
    from scanner.pipeline import leg_prices
    est = leg_prices("regular", {"1x": 0.7, "12": 0.75, "o15": 0.8}, (2.0, 3.4, 3.8), margin=0.06)
    assert est["1x"] == pytest.approx(1 / (1 / 2.0 + 1 / 3.4)) and est["12"] == pytest.approx(1 / (1 / 2.0 + 1 / 3.8))
    assert est["o15"] == pytest.approx((1 / 0.8) / 1.06)
    none = leg_prices("regular", {"1x": 0.7, "12": 0.75, "o15": 0.8}, None, margin=0.06)
    assert none["1x"] == pytest.approx((1 / 0.7) / 1.06)


def test_default_qualifies_by_the_1_20_to_1_50_price_band():
    txt = ("X: Y\n15:00 Even - Match  -  [ 2.00 | 3.40 | 3.80 ]\n16:00 Big - Small  -  [ 1.20 | 6.00 | 12.00 ]\n"
           "17:00 Long - Shot  -  [ 6.00 | 4.50 | 1.45 ]")
    r = run_pipeline(txt, min_threshold=0.0)
    by = {x["match"]: x for x in r["ranked_fixtures"]}
    assert r["qualify_by"] == "band" and r["price_band"] == [1.2, 1.5]
    assert by["Even vs Match"]["in_band"] and by["Even vs Match"]["status"].startswith("★")      # 1.26 / 1.31 / ~1.2
    assert not by["Big vs Small"]["in_band"] and by["Big vs Small"]["status"].startswith("◆")    # 1X priced ~1.00
    assert not by["Long vs Shot"]["in_band"]
    assert r["ranked_fixtures"][0]["in_band"]                                                      # in-band rows rank first
    assert any("Price-band mode" in w for w in r["warnings"])
    assert all("@" in l and "band" in l for l in r["copyable_block"].splitlines())
    wide = run_pipeline(txt, min_threshold=0.0, price_band=(1.0, 20.0))
    assert all(x["in_band"] for x in wide["ranked_fixtures"])


def test_all3_mode_restores_the_50_percent_rule():
    txt = "X: Y\n15:00 Big - Small  -  [ 1.20 | 6.00 | 12.00 ]"
    r = run_pipeline(txt, min_threshold=0.0, qualify_by="all3")
    assert r["ranked_fixtures"][0]["status"].startswith("★") and not r["ranked_fixtures"][0]["in_band"]


def test_bad_qualify_by_rejected():
    with pytest.raises(ValueError):
        run_pipeline(TXT, qualify_by="nope")
