"""Parser for raw Flashscore mobile copy/paste text.

Turns a pasted page into a list of structured fixtures plus a cleaned,
human-readable schedule. Pure Python, no web-framework dependency.
"""
from __future__ import annotations

import re

DEFAULT_COMPETITION = "GLOBAL LEAGUES"

# Menu / chrome text that appears in a Flashscore copy. These are matched as
# WHOLE WORDS and only on lines that are not fixtures. (The original used a
# plain substring test on every line, which silently dropped any fixture
# containing e.g. "Liverpool" ("live"), "Tennis Borussia" or "Toptani".)
_NOISE_PHRASES = (
    "bet on football", "refresh now", "flashscore", "livescore",
    "basketball", "tennis", "ice hockey", "field hockey", "handball",
    "volleyball", "baseball", "am. football", "rugby union", "cricket",
    "more sports", "top", "m.flashscore", "provided live", "mobile version",
    "advertisement", "all games", "live", "finished", "odds", "draws",
    "football |", "1xbet", "gamble responsibly", "gambling therapy", "copyright",
    "today | yesterday | tomorrow", "football » tomorrow",
)


def _phrase_pattern(phrase: str) -> str:
    start = r"(?<!\w)" if phrase[0].isalnum() or phrase[0] == "_" else ""
    end = r"(?!\w)" if phrase[-1].isalnum() or phrase[-1] == "_" else ""
    return f"{start}{re.escape(phrase)}{end}"


_NOISE_RE = re.compile(
    "|".join(_phrase_pattern(p) for p in sorted(_NOISE_PHRASES, key=len, reverse=True)),
    re.IGNORECASE,
)

_TIME_LEADING_RE = re.compile(r"^(\d{2}:\d{2})")  # exactly 5 chars: "16:001.FC Nurnberg" -> 16:00 + "1.FC ..."
_MINUTE_RE = re.compile(r"^\d{1,3}(?:\+\d{1,2})?['’]\s*")  # live minute, apostrophe REQUIRED
_STATUS_RE = re.compile(r"^(Postponed|Cancell?ed|Abandoned|Awarded)\s*", re.IGNORECASE)  # glued: "PostponedCartagena"
_SCORE_RE = re.compile(r"\d+-\d+")
# Odds page: "14:00Namibia - Cameroon  -  [ 8.60 | 4.10 | 1.39 ]"  (home | draw | away)
_ODDS_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\]")
_TRAILING_SCORE_RE = re.compile(r"\s+\d+-\d+\b.*$")
_HEADER_NOISE_RE = re.compile(r"\b(Standings|Table)\b", re.IGNORECASE)


def _is_country_header(line: str) -> bool:
    """'NIGERIA: NPFL Standings', 'AFRICA: Cup - Qualification - Qualification'.

    Real Flashscore headers start with an UPPERCASE country/region and a colon and
    often contain ' - ' themselves, so they must be recognised before fixtures.
    """
    head, sep, _ = line.partition(":")
    return bool(sep) and head.strip().isupper()


def _clean_header(line: str) -> str:
    return re.sub(r"\s+", " ", _HEADER_NOISE_RE.sub("", line)).strip(" -")


def parse_fixtures(raw_text: str, include_in_play: bool = False) -> dict:
    """Parse raw Flashscore text.

    Returns ``cleaned_text``, ``leagues_count``, ``fixtures_count``, ``fixtures``
    (list of ``{time, competition, home, away, odds}``; odds is ``(home, draw, away)`` or None) and the counts of lines skipped
    because they were already live/finished or postponed/cancelled.
    """
    empty = {
        "cleaned_text": "", "leagues_count": 0, "fixtures_count": 0, "fixtures": [],
        "skipped_in_play": 0, "skipped_postponed": 0,
    }
    if not raw_text or not raw_text.strip():
        return empty

    current_comp = DEFAULT_COMPETITION
    leagues = {current_comp}
    cleaned: list[str] = []
    fixtures: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    skipped_in_play = skipped_postponed = 0

    def set_header(line: str) -> None:
        nonlocal current_comp
        comp = _clean_header(line)
        if comp:
            current_comp = comp
            leagues.add(comp)
            cleaned.append(f"\n--- {comp.upper()} ---")

    for line in raw_text.splitlines():
        line = line.strip().replace(" – ", " - ")  # normalise en dash
        if not line:
            continue

        time_m = _TIME_LEADING_RE.match(line)
        minute_m = _MINUTE_RE.match(line)

        if not (time_m or minute_m):
            # Header, page chrome, or junk - never a fixture.
            if _is_country_header(line):
                set_header(line)
            elif _NOISE_RE.search(line):
                continue
            elif " - " not in line and (":" in line or (len(line) > 3 and not _SCORE_RE.search(line))):
                set_header(line)
            continue

        in_play = bool(minute_m)
        if time_m:
            time_str, rest = time_m.group(1), line[5:].strip()
        else:
            time_str, rest = "00:00", line
        rest = _MINUTE_RE.sub("", rest, count=1).strip()

        postponed = False
        status_m = _STATUS_RE.match(rest)
        if status_m:
            postponed = True
            rest = rest[status_m.end():].strip()

        if " - " not in rest:
            continue
        home, away_raw = rest.split(" - ", 1)
        home = home.strip()
        odds = None
        om = _ODDS_RE.search(away_raw)
        if om:
            vals = tuple(float(g) for g in om.groups())
            odds = vals if all(v > 1.0 for v in vals) else None
            away_raw = _ODDS_RE.sub("", away_raw)
        away = re.sub(r"\s*-\s*$", "", away_raw).strip()
        if _TRAILING_SCORE_RE.search(away):  # "Team 1-0": a score means it is live/finished
            away = _TRAILING_SCORE_RE.sub("", away).strip()
            in_play = True
        # NOTE: no trailing-digit stripping: "Celaya 2", "Minsk 2", "Schalke 04" are real names.

        if not home or not away or home.casefold() == away.casefold():
            continue
        if postponed:
            skipped_postponed += 1
            continue
        if in_play and not include_in_play:
            skipped_in_play += 1
            continue

        key = (time_str, current_comp.casefold(), home.casefold(), away.casefold())
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(f"{time_str} | {home} vs {away}")
        fixtures.append(
            {"time": time_str, "competition": current_comp.upper(), "home": home, "away": away, "odds": odds}
        )

    return {
        "cleaned_text": "\n".join(cleaned).strip(),
        "leagues_count": len(leagues),
        "fixtures_count": len(fixtures),
        "fixtures": fixtures,
        "skipped_in_play": skipped_in_play,
        "skipped_postponed": skipped_postponed,
    }
