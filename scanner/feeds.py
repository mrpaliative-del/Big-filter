"""Open public data feeds (football-data.co.uk): fixtures with odds, and latest results with match statistics.

* FIXTURES_URL: upcoming matches with 1X2 / over-under / Asian-handicap prices from many bookmakers.
  Odds are collected Fridays (weekend games) and Tuesdays (midweek games), so they are a snapshot, not live.
* LATEST_URL: the most recent matchday of results with shots, shots on target, etc. for the current season.

``update_history`` merges new results into the history CSV the form model reads; ``OddsFeed`` prices pasted
fixtures that have no odds. Network calls use ``requests`` with timeouts; parsing is pure and unit tested.
Data is free for personal use; please credit football-data.co.uk.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import season_of
from .model import _norm

FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
LATEST_URL = "https://football-data.co.uk/mmz4281/{code}/Latest_Results.csv"
HISTORY_COLUMNS = ["Div", "Date", "Home", "Away", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "AvgH", "AvgD", "AvgA", "AvgO25", "AvgU25", "Season"]
LEAGUE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"


def season_code(d: date | pd.Timestamp) -> str:
    """'2627' for any date in the 2026/27 season (July start)."""
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def parse_football_data_csv(text: str) -> pd.DataFrame:
    """football-data.co.uk CSV (results or fixtures) -> normalised frame; unparsable rows dropped."""
    df = pd.read_csv(io.StringIO(text.lstrip("\ufeff")), dtype=str, keep_default_na=False, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    need = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if not need <= set(df.columns):
        raise ValueError(f"not a football-data CSV (missing {sorted(need - set(df.columns))})")
    out = pd.DataFrame({"Div": df["Div"].str.strip(), "Home": df["HomeTeam"].str.strip(), "Away": df["AwayTeam"].str.strip()})
    d = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    d = d.fillna(pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce"))
    out["Date"] = d
    out["Time"] = df["Time"].str.strip() if "Time" in df else ""
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
    for src, dst in (("FTHG", "FTHG"), ("FTAG", "FTAG"), ("HS", "HS"), ("AS", "AS"), ("HST", "HST"), ("AST", "AST"),
                     ("HC", "HC"), ("AC", "AC"), ("HF", "HF"), ("AF", "AF"),
                     ("AvgH", "AvgH"), ("AvgD", "AvgD"), ("AvgA", "AvgA"), ("Avg>2.5", "AvgO25"), ("Avg<2.5", "AvgU25")):
        out[dst] = num(src)
    out = out[out["Date"].notna() & (out["Home"] != "") & (out["Away"] != "")]
    return out.reset_index(drop=True)


def merge_history(existing: pd.DataFrame | None, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add played games from ``new`` to ``existing`` (dedupe on date + teams). Returns (merged, number added)."""
    played = new.dropna(subset=["FTHG", "FTAG"]).copy()
    played["Season"] = played["Date"].map(season_of)
    played[["FTHG", "FTAG"]] = played[["FTHG", "FTAG"]].astype(int)
    played = played[[c for c in HISTORY_COLUMNS if c in played.columns]]
    if existing is None or existing.empty:
        return played.sort_values("Date", kind="stable").reset_index(drop=True), len(played)
    key = lambda df: set(zip(df["Date"].dt.strftime("%Y-%m-%d"), df["Home"].map(_norm), df["Away"].map(_norm)))
    have = key(existing)
    fresh = played[[k not in have for k in zip(played["Date"].dt.strftime("%Y-%m-%d"), played["Home"].map(_norm), played["Away"].map(_norm))]]
    merged = pd.concat([existing, fresh], ignore_index=True).sort_values("Date", kind="stable").reset_index(drop=True)
    return merged, len(fresh)


def fetch_text(url: str, timeout: float = 20.0) -> str:
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (football scanner; personal use)"})
    r.raise_for_status()
    return r.content.decode("utf-8-sig", errors="replace")


def update_history(path: str | Path, fetch=fetch_text, today: date | None = None) -> int:
    """Download the latest results and merge them into the history CSV. Returns the number of new games."""
    path = Path(path)
    existing = pd.read_csv(path, parse_dates=["Date"]) if path.is_file() else None
    new = parse_football_data_csv(fetch(LATEST_URL.format(code=season_code(today or date.today()))))
    merged, added = merge_history(existing, new)
    if added:
        merged.to_csv(path, index=False, date_format="%Y-%m-%d")
    return added


@dataclass
class OddsFeed:
    """Upcoming fixtures with average 1X2 prices, looked up by team name."""
    table: dict = field(default_factory=dict)       # (norm home, norm away) -> {"odds", "date", "time", "div"}
    aliases: dict = field(default_factory=dict)
    fetched: str = ""

    def __len__(self):
        return len(self.table)

    @classmethod
    def from_frame(cls, fx: pd.DataFrame, aliases: dict | None = None, fetched: str = "") -> "OddsFeed":
        t = {}
        for r in fx.itertuples():
            if np.isnan(r.AvgH) or np.isnan(r.AvgD) or np.isnan(r.AvgA) or min(r.AvgH, r.AvgD, r.AvgA) <= 1.0:
                continue
            t[(_norm(r.Home), _norm(r.Away))] = {"odds": (float(r.AvgH), float(r.AvgD), float(r.AvgA)), "date": r.Date.date(), "time": r.Time, "div": r.Div}
        return cls(t, {_norm(k): v for k, v in (aliases or {}).items()}, fetched)

    @classmethod
    def fetch(cls, fetch=fetch_text, aliases: dict | None = None) -> "OddsFeed":
        return cls.from_frame(parse_football_data_csv(fetch(FIXTURES_URL)), aliases, datetime.now().isoformat(timespec="minutes"))

    def lookup(self, home: str, away: str):
        h, a = _norm(home), _norm(away)
        h, a = _norm(self.aliases.get(h, h)), _norm(self.aliases.get(a, a))
        hit = self.table.get((h, a))
        return hit["odds"] if hit else None


def fetch_league_history(div: str, seasons, fetch=fetch_text) -> pd.DataFrame:
    """Season CSVs of one division (e.g. seasons ['2425', '2526', '2627']) in the history schema; seasons that fail are skipped."""
    frames = []
    for sc in seasons:
        try:
            df = parse_football_data_csv(fetch(LEAGUE_URL.format(season=sc, div=div)))
        except Exception:
            continue
        frames.append(df.dropna(subset=["FTHG", "FTAG"]))
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["Season"] = out["Date"].map(season_of)
    return out[[c for c in HISTORY_COLUMNS if c in out.columns]].sort_values("Date", kind="stable").reset_index(drop=True)
