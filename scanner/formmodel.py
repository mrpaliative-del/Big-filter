"""Trained 'technical analysis' model: predicts the next match from previous games, with NO odds needed.

Logistic regression on the SofaScore-style features (standings, form, streaks, head-to-head, past match
statistics, psychology proxies) for the legs of the "regular" profile (1X, 12, Over 1.5) and for All3.
Runtime needs only numpy; sklearn is used for training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import BASE_COLS, FeatureEngine, build_features, prev_season, season_of
from .model import _norm

TARGETS = ("1x", "12", "o15", "all3", "home", "away", "x2", "h15", "u25", "all3c")


def outcomes(hg: int, ag: int) -> dict:
    o = {"1x": hg >= ag, "12": hg != ag, "o15": hg + ag >= 2, "home": hg > ag, "away": ag > hg, "x2": ag >= hg}
    o["all3"] = o["1x"] and o["12"] and o["o15"]          # = home win and over 1.5
    o["h15"], o["u25"] = ag - hg <= 1, hg + ag <= 2
    o["all3c"] = o["1x"] and o["h15"] and o["u25"]        # classic All3 = 1X and Under 2.5 (Home +1.5 is implied by 1X)
    return o


class FormModel:
    def __init__(self, cols, mean, scale, coefs, note=""):
        self.cols, self.mean, self.scale, self.coefs, self.note = list(cols), np.array(mean), np.array(scale), coefs, note

    def predict(self, feats: dict) -> dict:
        x = (np.array([feats[c] for c in self.cols], float) - self.mean) / self.scale
        return {k: float(1 / (1 + np.exp(-(x @ np.array(w) + b)))) for k, (w, b) in self.coefs.items()}

    def save(self, path):
        Path(path).write_text(json.dumps({"note": self.note, "cols": self.cols, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                                          "coefs": {k: [list(w), b] for k, (w, b) in self.coefs.items()}}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        p = Path(path)
        if not p.is_file():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(d["cols"], d["mean"], d["scale"], {k: (v[0], v[1]) for k, v in d["coefs"].items()}, d.get("note", ""))


def train(games: pd.DataFrame, F: pd.DataFrame, mask=None, C: float = 0.05, note: str = "") -> FormModel:
    from sklearn.linear_model import LogisticRegression
    mask = np.ones(len(games), bool) if mask is None else mask
    Z = F[BASE_COLS].values[mask]
    mean, scale = Z.mean(0), Z.std(0)
    scale[scale == 0] = 1.0
    X = (Z - mean) / scale
    Y = pd.DataFrame([outcomes(int(r.FTHG), int(r.FTAG)) for r in games[mask].itertuples()])
    coefs = {}
    for k in TARGETS:
        m = LogisticRegression(C=C, max_iter=1000).fit(X, Y[k].astype(int))
        coefs[k] = (m.coef_[0].tolist(), float(m.intercept_[0]))
    return FormModel(BASE_COLS, mean, scale, coefs, note)


# ---------------------------------------------------------------------------- runtime
def load_history(paths) -> pd.DataFrame:
    """Results CSVs (Date, Home, Away, FTHG, FTAG; Div/Season/HS/HST/AS/AST optional) -> one sorted frame."""
    frames = []
    for p in paths:
        d = pd.read_csv(p, parse_dates=["Date"])
        for c in ("HS", "HST", "AS", "AST"):
            if c not in d:
                d[c] = np.nan
        if "Season" not in d:
            d["Season"] = d["Date"].map(season_of)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).sort_values("Date", kind="stable")
    if "Div" not in df or df["Div"].isna().any():           # extra results without a division: reuse the team's known one
        known = {}
        for r in df.dropna(subset=["Div"]).itertuples() if "Div" in df else []:
            known[r.Home] = known[r.Away] = r.Div
        df["Div"] = [d if isinstance(d, str) else known.get(h, known.get(a)) for d, h, a in zip(df.get("Div", [None] * len(df)), df.Home, df.Away)]
        df = df.dropna(subset=["Div"])
    return df.reset_index(drop=True)


class FormPredictor:
    """Engine primed with all history + trained model; scores a future fixture by team name."""

    def __init__(self, model: FormModel, games: pd.DataFrame, aliases: dict | None = None):
        self.model = model
        _, _, self.engine = build_features(games)
        self.names = {}
        for t in self.engine.loc:
            self.names.setdefault(_norm(t), t)
        self.aliases = {_norm(k): v for k, v in (aliases or {}).items()}
        self.last_date = games.Date.max()
        self.current_season = season_of(self.last_date)

    def resolve(self, name: str):
        n = _norm(name)
        if n in self.aliases:
            n = _norm(self.aliases[n])
        return self.names.get(n)

    def predict_fixture(self, home: str, away: str, min_games: int = 3):
        h, a = self.resolve(home), self.resolve(away)
        if not h or not a or h == a:
            return None
        lh, la = self.engine.loc[h], self.engine.loc[a]
        if lh[:2] != la[:2] or lh[0] != self.current_season:      # same league, current season only
            return None
        if min(len(self.engine.hist[h]), len(self.engine.hist[a])) < min_games:
            return None
        return self.model.predict(self.engine.features(h, a, lh[0], lh[1]))


    # ------------------------------------------------------------------ the statistics behind a selection
    def explain(self, home: str, away: str) -> dict | None:
        """Recent form, goals against and historic form for both teams, plus head-to-head, as plain numbers and text."""
        h, a = self.resolve(home), self.resolve(away)
        if not h or not a or h == a:
            return None
        e = self.engine
        lh = e.loc[h]
        T = e.table[(lh[0], lh[1])]
        order = sorted(T, key=lambda t: (-T[t][0], -(T[t][1] - T[t][2]), -T[t][1]))

        def team(t, side):
            hs = list(e.hist[t])
            l5, l10 = hs[-5:], hs[-10:]
            pts = lambda xs: sum(3 if g > c else 1 if g == c else 0 for g, c, *_ in xs)
            word = "".join("W" if g > c else "D" if g == c else "L" for g, c, *_ in l5) or "-"
            pts5, ga5, gf5 = pts(l5), sum(c for _, c, *_ in l5), sum(g for g, *_ in l5)
            ga10 = sum(c for _, c, *_ in l10) / len(l10) if l10 else None
            vv = list(e.venue[t][side])
            vga = sum(c for _, c in vv) / len(vv) if vv else None
            sp, sgf, sga, smp = T[t]
            pv = e.season_tot[t].get(prev_season(lh[0]))
            return {"form": word, "pts5": pts5, "gf5": gf5, "ga5": ga5, "ga_pg5": ga5 / len(l5) if l5 else None, "ga_pg10": ga10,
                    "venue_ga_pg": vga, "season_ppg": sp / smp if smp else None, "season_ga_pg": sga / smp if smp else None,
                    "position": order.index(t) + 1 if t in order else None, "played": smp,
                    "prev_ppg": pv[0] / pv[3] if pv and pv[3] >= 10 else None}

        th, ta = team(h, "h"), team(a, "a")
        prev = e.h2h[frozenset((h, a))][-4:]
        hw = sum(1 for hh, hg, ag in prev if (hg > ag if hh == h else ag > hg))
        h2h = {"meetings": len(prev), "home_team_wins": hw, "avg_goals": (sum(hg + ag for _, hg, ag in prev) / len(prev)) if prev else None}
        f = lambda v, nd=1: "n/a" if v is None else f"{v:.{nd}f}"
        text = (f"{home} form {th['form']} ({th['pts5']}pts, GA/g {f(th['ga_pg5'])}, at home {f(th['venue_ga_pg'])}, season ppg {f(th['season_ppg'], 2)}, "
                f"prev ppg {f(th['prev_ppg'], 2)}) v {away} form {ta['form']} ({ta['pts5']}pts, GA/g {f(ta['ga_pg5'])}, away {f(ta['venue_ga_pg'])}, "
                f"season ppg {f(ta['season_ppg'], 2)}, prev ppg {f(ta['prev_ppg'], 2)})"
                + (f"; H2H {h2h['meetings']}: {home} won {h2h['home_team_wins']}, {f(h2h['avg_goals'])} goals/g" if prev else "; no head-to-head in data"))
        return {"home": th, "away": ta, "h2h": h2h, "text": text}
