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
    r = run_pipeline(TXT, ratings=ratings)
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
    r = run_pipeline(ODDS, min_threshold=0.0, max_xg=10)
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
    r = run_pipeline(ODDS, min_threshold=0.0, max_xg=10, threshold=0.55)
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
    assert run_pipeline(ODDS, min_threshold=0.0, max_xg=10)["threshold"] == 0.695


def test_bad_rank_by_rejected():
    import pytest
    with pytest.raises(ValueError):
        run_pipeline(ODDS, rank_by="nope")


def test_only_odds_rows_can_qualify():
    from scanner.model import Rating, Ratings
    txt = "ENGLAND: Premier League\n15:00 Arsenal - Chelsea  -  [ 1.20 | 6.00 | 12.00 ]\n17:00 Foo - Bar  - "
    ratings = Ratings({"Foo": Rating(0.6, 0.6), "Bar": Rating(0.6, 0.6)})
    r = run_pipeline(txt, threshold=0.0, min_threshold=0.0, max_xg=10, ratings=ratings)
    by = {x["match"]: x for x in r["ranked_fixtures"]}
    assert by["Arsenal vs Chelsea"]["status"].startswith("★") and by["Foo vs Bar"]["status"].startswith("◆")
    assert any("never tagged QUALIFIED" in w for w in r["warnings"])
    r2 = run_pipeline(txt, threshold=0.0, min_threshold=0.0, max_xg=10, ratings=ratings, qualify_unvalidated=True)
    assert all(x["status"].startswith("★") for x in r2["ranked_fixtures"])
