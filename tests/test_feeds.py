from datetime import date

import pandas as pd
import pytest

from scanner.feeds import OddsFeed, merge_history, parse_football_data_csv, season_code, update_history

FIXTURES = ("\ufeffDiv,Date,Time,HomeTeam,AwayTeam,Referee,B365H,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
            "E2,14/11/2026,15:00,AFC Wimbledon,Blackpool,O Mackey,2.5,2.65,3.4,2.49,2.65,3.4,2.49,1.75,1.98\n"
            "SP2,15/11/2026,13:00,Tenerife,Ceuta,,,,,,,,,2,1.85\n"
            "SC1,14/11/2026,15:00,Ayr,Raith Rvs,C Stirling,3.3,3.4,3.25,2.15,3.32,3.13,2.08,2.18,1.61\n")
RESULTS = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
           "E0,08/11/26,16:30,Crystal Palace,Liverpool,1,3,A,10,16,3,8,5.15,4.27,1.57,1.6,2.3\n"
           "E0,15/11/2026,14:00,Arsenal,Spurs,,,,,,,,1.5,4.3,6,1.7,2.1\n"
           "SC0,07/11/26,15:00,Aberdeen,Hibernian,0,4,A,9,11,2,7,2.43,3.44,2.66,1.9,1.9\n")


def test_parse_handles_bom_two_date_formats_and_missing_odds():
    fx = parse_football_data_csv(FIXTURES)
    assert list(fx.Home) == ["AFC Wimbledon", "Tenerife", "Ayr"] and fx.Date.iloc[0] == pd.Timestamp("2026-11-14")
    assert fx.AvgH.iloc[0] == 2.65 and pd.isna(fx.AvgH.iloc[1]) and fx.AvgU25.iloc[0] == 1.98
    rs = parse_football_data_csv(RESULTS)
    assert rs.Date.iloc[0] == pd.Timestamp("2026-11-08") and rs.FTHG.iloc[0] == 1 and pd.isna(rs.FTHG.iloc[1])     # 2-digit year ok, unplayed -> NaN


def test_parse_rejects_other_files():
    with pytest.raises(ValueError):
        parse_football_data_csv("a,b\n1,2\n")


def test_season_code():
    assert season_code(date(2026, 11, 17)) == "2627" and season_code(date(2027, 3, 1)) == "2627" and season_code(date(2026, 6, 30)) == "2526"


def test_merge_adds_only_new_played_games_and_dedupes():
    new = parse_football_data_csv(RESULTS)
    merged, added = merge_history(None, new)
    assert added == 2 and set(merged.Season) == {"2026/27"}                  # the unplayed Arsenal game is not history
    again, added2 = merge_history(merged, new)
    assert added2 == 0 and len(again) == 2
    more = parse_football_data_csv("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,08/11/26,crystal palace,LIVERPOOL,1,3\nE0,09/11/26,Man City,Fulham,2,0\n")
    out, added3 = merge_history(merged, more)
    assert added3 == 1 and len(out) == 3                                    # first row is the same game under another spelling


def test_update_history_with_a_fake_fetch(tmp_path):
    p = tmp_path / "h.csv"
    assert update_history(p, fetch=lambda url: RESULTS, today=date(2026, 11, 17)) == 2
    assert update_history(p, fetch=lambda url: RESULTS, today=date(2026, 11, 17)) == 0
    assert len(pd.read_csv(p)) == 2


def test_odds_feed_lookup_by_forgiving_names_and_aliases():
    feed = OddsFeed.from_frame(parse_football_data_csv(FIXTURES), aliases={"Wimbledon": "AFC Wimbledon"})
    assert len(feed) == 2                                                   # Tenerife v Ceuta has no 1X2 odds
    assert feed.lookup("afc wimbledon", "BLACKPOOL") == (2.65, 3.4, 2.49)
    assert feed.lookup("Wimbledon", "Blackpool") == (2.65, 3.4, 2.49) and feed.lookup("Tenerife", "Ceuta") is None
