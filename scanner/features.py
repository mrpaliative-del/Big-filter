"""Pre-match 'technical analysis' features (SofaScore-style), built from earlier games only.

One engine serves both training and live scanning: ``features()`` describes a game that has NOT happened yet,
``update()`` then records its result. Feeding history in date order and calling ``features()`` before each
``update()`` therefore reproduces exactly what was knowable at the time (no leakage).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

GROUPS = {
    "standings": ["ppg_h", "ppg_a", "gdpg_h", "gdpg_a", "rank_h", "rank_a", "mp_h", "mp_a"],
    "form": ["f_ppg_h", "f_ppg_a", "f_gf_h", "f_gf_a", "f_ga_h", "f_ga_a"],
    "streaks": ["unb_h", "unb_a", "winless_h", "winless_a", "cs_h", "cs_a", "scor_h", "scor_a", "ov_h", "ov_a"],
    "head_to_head": ["h2h_n", "h2h_win", "h2h_goals", "h2h_last_ov"],
    "match_stats": ["sh_h", "sh_a", "sot_h", "sot_a", "sota_h", "sota_a"],
    "psychology": ["bounce_h", "bounce_a", "bigwin_h", "bigwin_a", "stakes_h", "stakes_a", "dead_h", "dead_a"],
}
BASE_COLS = sum(GROUPS.values(), [])
# Added for the statistics-only scanner: goals against, venue splits, longer-window and previous-season ("historic") form
EXTRA_GROUPS = {
    "goals_against": ["ga_pg_h", "ga_pg_a", "gf_pg_h", "gf_pg_a", "l10_ga_h", "l10_ga_a", "vga_h", "vga_a", "vgf_h", "vgf_a", "cs10_h", "cs10_a", "fts10_h", "fts10_a"],
    "historic_form": ["l10_ppg_h", "l10_ppg_a", "l10_gf_h", "l10_gf_a", "prev_ppg_h", "prev_ppg_a", "prev_gdpg_h", "prev_gdpg_a", "prev_known_h", "prev_known_a"],
}
EXTRA_COLS = sum(EXTRA_GROUPS.values(), [])
# Match-statistics determinants: corners, fouls, cards, chances created (shots on target) over the last 5 games and in the previous match
STAT_GROUPS = {
    "match_stats2": ["crn_f_h", "crn_f_a", "crn_a_h", "crn_a_a", "foul_h", "foul_a", "card_h", "card_a",
                     "sotpg_h", "sotpg_a", "sotag_h", "sotag_a", "lastsot_h", "lastsot_a", "shotpg_h", "shotpg_a"],
}
STAT_COLS = sum(STAT_GROUPS.values(), [])
# Draw drivers: draw-prone teams, draw runs, head-to-head draws, low-scoring parity
DRAW_COLS = ["drw10_h", "drw10_a", "drwrun_h", "drwrun_a", "h2h_draw"]
ALL_COLS = BASE_COLS + EXTRA_COLS + STAT_COLS + DRAW_COLS
FORM_COLS = BASE_COLS + STAT_COLS      # the form model's inputs: 42 form/standing/h2h features + corners, fouls, cards, chances created


def prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}/{str(y + 1)[2:]}"


def season_of(date: pd.Timestamp) -> str:
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y}/{str(y + 1)[2:]}"


class FeatureEngine:
    def __init__(self, nteams: dict | None = None):
        self.nteams = nteams or {}
        self.table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))   # (season, div) -> team -> pts, gf, ga, mp
        self.hist = defaultdict(lambda: deque(maxlen=10))                      # team -> (gf, ga, shots, sot, sot_against)
        self.h2h = defaultdict(list)
        self.venue = defaultdict(lambda: {"h": deque(maxlen=5), "a": deque(maxlen=5)})   # team -> last 5 home / away (gf, ga)
        self.stats = defaultdict(lambda: deque(maxlen=10))                               # team -> (corners for/against, fouls, cards, sot for/against, shots for)
        self.season_tot = defaultdict(dict)                                              # team -> season -> [pts, gf, ga, mp]
        self.loc: dict[str, tuple[str, str, pd.Timestamp]] = {}               # team -> latest (season, div, date)

    # ------------------------------------------------------------------ features
    def features(self, home: str, away: str, season: str, div: str) -> dict:
        T = self.table[(season, div)]
        order = sorted(T, key=lambda t: (-T[t][0], -(T[t][1] - T[t][2]), -T[t][1]))
        n = max(self.nteams.get((season, div), 20), 2)
        frac = min(max(T[home][3], T[away][3]) / (2 * (n - 1)), 1.0)
        f = {}
        for side, t in (("h", home), ("a", away)):
            pts, gf, ga, mp = T[t]
            f[f"ppg_{side}"], f[f"gdpg_{side}"], f[f"mp_{side}"] = pts / (mp + 3), (gf - ga) / (mp + 3), mp
            f[f"rank_{side}"] = (order.index(t) / max(len(order) - 1, 1)) if t in order and len(order) > 1 else 0.5
            hs = list(self.hist[t]); l5 = hs[-5:]
            f[f"f_ppg_{side}"] = sum(3 if g > a_ else 1 if g == a_ else 0 for g, a_, *_ in l5) / (len(l5) + 2) if l5 else 0.0
            f[f"f_gf_{side}"] = sum(g for g, *_ in l5) / (len(l5) + 2) if l5 else 0.0
            f[f"f_ga_{side}"] = sum(a_ for _, a_, *_ in l5) / (len(l5) + 2) if l5 else 0.0

            def run(cond):
                k = 0
                for g, a_, *_ in reversed(hs):
                    if cond(g, a_):
                        k += 1
                    else:
                        break
                return min(k, 6)

            f[f"unb_{side}"], f[f"winless_{side}"] = run(lambda g, a_: g >= a_), run(lambda g, a_: g <= a_)
            f[f"cs_{side}"], f[f"scor_{side}"] = run(lambda g, a_: a_ == 0), run(lambda g, a_: g >= 1)
            f[f"ov_{side}"] = sum(g + a_ >= 3 for g, a_, *_ in l5) / (len(l5) + 2) if l5 else 0.0
            sv = [s for *_, s, so, sa in l5 if not np.isnan(s)]
            f[f"sh_{side}"] = np.mean(sv) if sv else 0.0
            f[f"sot_{side}"] = np.mean([so for *_, s, so, sa in l5 if not np.isnan(so)]) if sv else 0.0
            f[f"sota_{side}"] = np.mean([sa for *_, s, so, sa in l5 if not np.isnan(sa)]) if sv else 0.0
            last = hs[-1] if hs else None
            f[f"bounce_{side}"] = float(bool(last) and last[1] - last[0] >= 3)
            f[f"bigwin_{side}"] = float(bool(last) and last[0] - last[1] >= 3)
            rk = f[f"rank_{side}"]
            f[f"stakes_{side}"] = float(frac >= 0.7 and (rk <= 0.15 or rk >= 0.85))
            f[f"dead_{side}"] = float(frac >= 0.7 and 0.3 < rk < 0.7)
        prev = self.h2h[frozenset((home, away))][-4:]
        f["h2h_n"] = len(prev)
        if prev:
            w = [1.0 if (hg > ag if h_ == home else ag > hg) else 0.5 if hg == ag else 0.0 for h_, hg, ag in prev]
            f["h2h_win"], f["h2h_goals"] = (sum(w) + 0.5) / (len(w) + 1), float(np.mean([hg + ag for _, hg, ag in prev]))
            f["h2h_last_ov"] = float(prev[-1][1] + prev[-1][2] >= 3)
        else:
            f["h2h_win"], f["h2h_goals"], f["h2h_last_ov"] = 0.5, 2.7, 0.0
        for side, t, venue in (("h", home, "h"), ("a", away, "a")):
            pts, gf, ga, mp = T[t]
            f[f"ga_pg_{side}"], f[f"gf_pg_{side}"] = ga / (mp + 3), gf / (mp + 3)
            l10 = list(self.hist[t])
            f[f"l10_ga_{side}"] = sum(a_ for _, a_, *_ in l10) / (len(l10) + 3) if l10 else 0.0
            f[f"l10_gf_{side}"] = sum(g for g, *_ in l10) / (len(l10) + 3) if l10 else 0.0
            f[f"l10_ppg_{side}"] = sum(3 if g > a_ else 1 if g == a_ else 0 for g, a_, *_ in l10) / (len(l10) + 3) if l10 else 0.0
            f[f"cs10_{side}"] = sum(a_ == 0 for _, a_, *_ in l10) / (len(l10) + 3) if l10 else 0.0
            f[f"fts10_{side}"] = sum(g == 0 for g, *_ in l10) / (len(l10) + 3) if l10 else 0.0
            vv = list(self.venue[t][venue])
            f[f"vgf_{side}"] = sum(g for g, _ in vv) / (len(vv) + 2) if vv else 0.0
            f[f"vga_{side}"] = sum(a_ for _, a_ in vv) / (len(vv) + 2) if vv else 0.0
            pv = self.season_tot[t].get(prev_season(season))
            if pv and pv[3] >= 10:
                f[f"prev_ppg_{side}"], f[f"prev_gdpg_{side}"], f[f"prev_known_{side}"] = pv[0] / (pv[3] + 3), (pv[1] - pv[2]) / (pv[3] + 3), 1.0
            else:
                f[f"prev_ppg_{side}"], f[f"prev_gdpg_{side}"], f[f"prev_known_{side}"] = 0.0, 0.0, 0.0
        for side, t in (("h", home), ("a", away)):
            l10 = list(self.hist[t])
            f[f"drw10_{side}"] = sum(g == c for g, c, *_ in l10) / (len(l10) + 4) if l10 else 0.0
            run = 0
            for g, c, *_ in reversed(l10):
                if g == c:
                    run += 1
                else:
                    break
            f[f"drwrun_{side}"] = min(run, 3)
        pv = self.h2h[frozenset((home, away))][-4:]
        f["h2h_draw"] = (sum(hg == ag for _, hg, ag in pv) + 1.0) / (len(pv) + 4) if pv else 0.25
        for side, t in (("h", home), ("a", away)):
            l5 = list(self.stats[t])[-5:]
            col = lambda i: [x[i] for x in l5 if not np.isnan(x[i])]
            avg = lambda i: float(np.mean(col(i))) if col(i) else 0.0
            f[f"crn_f_{side}"], f[f"crn_a_{side}"], f[f"foul_{side}"], f[f"card_{side}"] = avg(0), avg(1), avg(2), avg(3)
            f[f"sotpg_{side}"], f[f"sotag_{side}"], f[f"shotpg_{side}"] = avg(4), avg(5), avg(6)
            f[f"lastsot_{side}"] = (l5[-1][4] if l5 and not np.isnan(l5[-1][4]) else 0.0)
        return f

    # ------------------------------------------------------------------ record a result
    def update(self, home, away, hg, ag, season, div, date, hs=np.nan, hst=np.nan, as_=np.nan, ast=np.nan,
               hc=np.nan, ac=np.nan, hf=np.nan, af=np.nan, hy=np.nan, ay=np.nan, hr=np.nan, ar=np.nan):
        T = self.table[(season, div)]
        T[home][0] += 3 if hg > ag else 1 if hg == ag else 0
        T[away][0] += 3 if ag > hg else 1 if hg == ag else 0
        T[home][1] += hg; T[home][2] += ag; T[away][1] += ag; T[away][2] += hg
        T[home][3] += 1; T[away][3] += 1
        self.hist[home].append((hg, ag, hs, hst, ast)); self.hist[away].append((ag, hg, as_, ast, hst))
        self.h2h[frozenset((home, away))].append((home, hg, ag))
        self.venue[home]["h"].append((hg, ag)); self.venue[away]["a"].append((ag, hg))
        for t, gf_, ga_ in ((home, hg, ag), (away, ag, hg)):
            tot = self.season_tot[t].setdefault(season, [0, 0, 0, 0])
            tot[0] += 3 if gf_ > ga_ else 1 if gf_ == ga_ else 0; tot[1] += gf_; tot[2] += ga_; tot[3] += 1
        cards = lambda y, r: (y if not np.isnan(y) else np.nan) + (2 * r if not np.isnan(r) else 0.0)
        self.stats[home].append((hc, ac, hf, cards(hy, hr), hst, ast, hs))
        self.stats[away].append((ac, hc, af, cards(ay, ar), ast, hst, as_))
        self.loc[home] = self.loc[away] = (season, div, date)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, FeatureEngine]:
    """Walk through games in date order; returns (sorted games, feature rows, engine primed with all history)."""
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    nteams = df.groupby(["Season", "Div"]).apply(lambda g: len(set(g.Home) | set(g.Away))).to_dict()
    eng = FeatureEngine(nteams)
    rows = []
    for r in df.itertuples():
        rows.append(eng.features(r.Home, r.Away, r.Season, r.Div))
        g = lambda c: getattr(r, c, np.nan)
        eng.update(r.Home, r.Away, r.FTHG, r.FTAG, r.Season, r.Div, r.Date, r.HS, r.HST, r.AS, r.AST,
                   g("HC"), g("AC"), g("HF"), g("AF"), g("HY"), g("AY"), g("HR"), g("AR"))
    return df, pd.DataFrame(rows), eng
