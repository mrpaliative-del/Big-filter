from scanner.parser import parse_fixtures


def names(txt):
    return [(f["home"], f["away"]) for f in parse_fixtures(txt)["fixtures"]]


def test_menu_words_do_not_eat_team_names():
    txt = "ENGLAND: Premier League\n15:00 Liverpool - Chelsea\n16:00 Tennis Borussia Berlin - Hertha\n18:00 Olivier FC - Toptani"
    assert len(names(txt)) == 3


def test_digit_prefixed_teams_survive():
    assert names("GERMANY: Bundesliga\n15:30 1. FC Köln - 1860 Munich") == [("1. FC Köln", "1860 Munich")]


def test_trailing_two_digit_name_kept():
    assert names("GERMANY: Bundesliga\n15:30 Hertha - Schalke 04") == [("Hertha", "Schalke 04")]


def test_live_minute_and_score_stripped_when_included():
    r = parse_fixtures("ENGLAND: Premier League\n45' Arsenal - Chelsea 1-0", include_in_play=True)
    assert [(f["home"], f["away"]) for f in r["fixtures"]] == [("Arsenal", "Chelsea")]


def test_live_skipped_by_default():
    r = parse_fixtures("ENGLAND: Premier League\n45' Arsenal - Chelsea 1-0")
    assert r["fixtures_count"] == 0 and r["skipped_in_play"] == 1


def test_glued_time_and_dedupe():
    r = parse_fixtures("NIGERIA: NPFL\n16:00Enyimba - Rivers\n16:00Enyimba - Rivers")
    assert r["fixtures_count"] == 1 and r["fixtures"][0]["home"] == "Enyimba"


def test_menu_lines_dropped_headers_kept():
    r = parse_fixtures("Flashscore\nLive\nAdvertisement\nSPAIN: LaLiga Standings\n20:00 Betis - Sevilla")
    assert r["fixtures"][0]["competition"] == "SPAIN: LALIGA"
    assert r["leagues_count"] == 2  # default + LaLiga


def test_empty_input():
    assert parse_fixtures("  \n")["fixtures_count"] == 0


# ---- real Flashscore mobile paste (excerpt) -------------------------------
import pathlib

SAMPLE = (pathlib.Path(__file__).parent / "sample_flashscore.txt").read_text(encoding="utf-8")


def test_real_paste_fixtures_exact():
    r = parse_fixtures(SAMPLE)
    got = [(f["home"], f["away"]) for f in r["fixtures"]]
    assert got == [
        ("Enyimba", "Ikorodu City"), ("Inter Lagos", "Abia Warriors"),
        ("Zambia U23", "Burkina Faso U23"),
        ("Deportivo Moron", "Colegiales"), ("Envigado", "U. Magdalena"),
        ("Racing de Veracruz 2", "Celaya 2"), ("1908 SZAC KSE", "Komaromi VSE"),
        ("1.FC Nurnberg W", "Koln W"), ("3K FC", "Gor Mahia"),
        ("Danubio", "Liverpool M."), ("FK Minija", "FA Siauliai"),
        ("Stoke City LFC W", "Liverpool Feds W"), ("Panama", "Curacao"),
    ]
    assert r["skipped_in_play"] == 2 and r["skipped_postponed"] == 2


def test_real_paste_headers_with_dashes_are_not_fixtures():
    r = parse_fixtures(SAMPLE)
    for f in r["fixtures"]:
        assert ":" not in f["home"] and "provides" not in f["home"]
    comps = {f["competition"] for f in r["fixtures"]}
    assert "AFRICA: AFRICA CUP OF NATIONS U23 - QUALIFICATION - QUALIFICATION" in comps
    assert "NORTH & CENTRAL AMERICA: CONCACAF NATIONS LEAGUE - LEAGUE A - PLAY OFFS" in comps
    # fixture after a dashed header is attributed to THAT header, not the previous league
    assert next(f for f in r["fixtures"] if f["home"] == "Zambia U23")["competition"].startswith("AFRICA:")


ODDS_TXT = (pathlib.Path(__file__).parent / "sample_odds.txt").read_text(encoding="utf-8")


def test_odds_extracted_and_team_names_clean():
    r = parse_fixtures(ODDS_TXT)
    by = {(f["home"], f["away"]): f for f in r["fixtures"]}
    assert by[("Namibia", "Cameroon")]["odds"] == (8.6, 4.1, 1.39)
    assert by[("Algeria U23", "Eswatini U23")]["odds"] is None
    assert all("[" not in f["away"] and "|" not in f["away"] for f in r["fixtures"])
    assert r["skipped_postponed"] == 1


def test_odds_page_chrome_not_parsed():
    r = parse_fixtures(ODDS_TXT)
    assert len(r["fixtures"]) == 8
    assert not any("1xBet" in f["home"] or "provides" in f["home"] for f in r["fixtures"])
    assert not any("1xBet" in c for c in r["cleaned_text"].splitlines())  # not turned into a league
