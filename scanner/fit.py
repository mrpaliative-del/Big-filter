"""Fit team attack/defence ratings from historical results.

Fits the *same* expected-goals formula the scanner uses (see
``model.expected_goals``) by maximum-likelihood Poisson regression with a
shrinkage prior toward 1.0, so teams with few games stay near average.
numpy only.
"""
from __future__ import annotations

import csv
import math
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .model import (AWAY_CONCEDED, AWAY_SCORED, BASE_GOALS, HOME_CONCEDED,
                    HOME_SCORED)

_ALIASES = {
    "home": ("home", "home_team", "hometeam", "team1", "team 1"),
    "away": ("away", "away_team", "awayteam", "visitor", "team2", "team 2"),
    "hg": ("home_goals", "hg", "fthg", "hgoal", "home_score", "score_home"),
    "ag": ("away_goals", "ag", "ftag", "vgoal", "away_score", "score_away"),
    "ft": ("ft", "score", "result"),  # combined "2-1", used when separate goal columns are absent
    "date": ("date", "match_date", "kickoff"),
}
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d.%m.%Y", "%Y/%m/%d", "%a %b %d %Y")
_FT_RE = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)")


def _parse_date(text: str) -> date | None:
    text = text.strip()
    for candidate in (text, text[:10]):
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    return None


def read_results(path: str | Path, since: date | None = None) -> list[dict]:
    """Read a results CSV: home/away team plus either home/away goal columns or a
    combined "2-1" score column (footballcsv, engsoccerdata and football-data
    header styles all work). A date column is optional unless ``since`` is given;
    ``since`` drops older rows and rows with no usable date."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        lookup = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        col = {}
        for key, names in _ALIASES.items():
            col[key] = next((lookup[n] for n in names if n in lookup), None)
        has_goals = col["hg"] is not None and col["ag"] is not None
        missing = [k for k in ("home", "away") if col[k] is None]
        if not has_goals and col["ft"] is None:
            missing += ["hg", "ag"]
        if missing:
            raise ValueError(
                f"Missing columns for {missing}. Found: {list(lookup)}. "
                "Need home team, away team, and home/away goals (or a combined score)."
            )
        rows = []
        for r in reader:
            try:
                if has_goals:
                    hg, ag = int(float(r[col["hg"]])), int(float(r[col["ag"]]))
                else:
                    m = _FT_RE.match(r[col["ft"]] or "")
                    if not m:
                        continue
                    hg, ag = int(m.group(1)), int(m.group(2))
            except (TypeError, ValueError):
                continue  # unplayed / blank / "NA" score
            if hg < 0 or ag < 0:
                continue
            home, away = (r[col["home"]] or "").strip(), (r[col["away"]] or "").strip()
            if not home or not away or home.casefold() == away.casefold():
                continue
            d = _parse_date(r[col["date"]]) if col["date"] and r.get(col["date"]) else None
            if since is not None and (d is None or d < since):
                continue
            rows.append({"home": home, "away": away, "hg": hg, "ag": ag, "date": d})
    return rows


def fit_ratings(
    matches: list[dict],
    reg: float = 2.0,
    min_games: int = 5,
    half_life_days: float | None = None,
    iters: int = 3000,
    lr: float = 0.05,
    today: date | None = None,
) -> dict[str, dict[str, float]]:
    """Return ``{team: {"attack": a, "defence": d}}`` (1.0 = league average).

    ``reg``: strength of the pull toward 1.0 (bigger = more shrinkage).
    ``half_life_days``: if set and matches have dates, older games count less.
    Teams with fewer than ``min_games`` *effective* (decay-weighted) games are omitted.
    """
    if not matches:
        return {}
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = np.array([idx[m["home"]] for m in matches])
    a = np.array([idx[m["away"]] for m in matches])
    gh = np.array([m["hg"] for m in matches], dtype=float)
    ga = np.array([m["ag"] for m in matches], dtype=float)

    w = np.ones(len(matches))
    if half_life_days:
        ref = today or max((m["date"] for m in matches if m["date"]), default=None)
        if ref:
            for i, m in enumerate(matches):
                if m["date"]:
                    w[i] = 0.5 ** (max((ref - m["date"]).days, 0) / half_life_days)

    # effective games: with recency weighting, old games count for less toward min_games
    games = np.bincount(h, weights=w, minlength=n) + np.bincount(a, weights=w, minlength=n)
    c = 0.5 * BASE_GOALS
    th_a, th_d = np.zeros(n), np.zeros(n)
    m_a, v_a, m_d, v_d = (np.zeros(n) for _ in range(4))
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, iters + 1):
        A, D = np.exp(th_a), np.exp(th_d)
        lam_h = c * (HOME_SCORED * A[h] + AWAY_CONCEDED * D[a])
        lam_a = c * (AWAY_SCORED * A[a] + HOME_CONCEDED * D[h])
        rh, ra = w * (gh / lam_h - 1.0), w * (ga / lam_a - 1.0)
        g_a, g_d = np.zeros(n), np.zeros(n)
        np.add.at(g_a, h, rh * c * HOME_SCORED)
        np.add.at(g_d, a, rh * c * AWAY_CONCEDED)
        np.add.at(g_a, a, ra * c * AWAY_SCORED)
        np.add.at(g_d, h, ra * c * HOME_CONCEDED)
        g_a = g_a * A - reg * th_a
        g_d = g_d * D - reg * th_d
        step_max = 0.0
        for th, g, m_, v_ in ((th_a, g_a, m_a, v_a), (th_d, g_d, m_d, v_d)):
            m_[:] = b1 * m_ + (1 - b1) * g
            v_[:] = b2 * v_ + (1 - b2) * g * g
            step = lr * (m_ / (1 - b1**t)) / (np.sqrt(v_ / (1 - b2**t)) + eps)
            th += step
            step_max = max(step_max, float(np.abs(step).max()))
        if t > 200 and step_max < 1e-6:
            break

    out = {}
    for team, i in idx.items():
        if games[i] < min_games:
            continue
        att = float(np.clip(math.exp(th_a[i]), 0.2, 4.0))
        dfn = float(np.clip(math.exp(th_d[i]), 0.2, 4.0))
        out[team] = {"attack": round(att, 3), "defence": round(dfn, 3)}
    return out
