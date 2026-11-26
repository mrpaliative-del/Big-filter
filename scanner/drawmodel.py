"""Draw rating from football draw drivers (pooled over all leagues), plus the scanner that wraps it.

Drivers: market parity, low scoring, tight defences, draw-prone teams, draw runs, head-to-head draws, low tempo (shots on target),
corner parity and foul-heavy games, on top of the market's own draw probability. The rating is a calibrated probability.
Walk-forward (train 2025/26, 2,397 unseen 2026/27 games): log-loss 0.5441 vs 0.5513 flat and 0.5433 for the margin-free market; ratings
never exceeded ~36% and games rated 30-33% drew 33% of the time. A draw rating of 40%+ does not occur in calibrated models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .draw import calculate_kelly_stake
from .features import DRAW_COLS, STAT_COLS

DRIVER_COLS = ["lg", "m_gap", "ppg_gap", "rank_gap", "comb_gf", "comb_ga", "comb_cs", "comb_sot", "crn_gap"] + DRAW_COLS + STAT_COLS
# human-readable drivers: (feature, label shown, direction in which it favours a draw)
DRIVERS = [("m_gap", "Evenly matched (market)", "low"), ("ppg_gap", "Similar season form", "low"), ("comb_gf", "Low combined scoring", "low"),
           ("comb_ga", "Tight defences", "low"), ("drw10", "Draw-prone teams", "high"), ("h2h_draw", "Head-to-head draws", "high"),
           ("comb_sot", "Low chance creation", "low"), ("crn_gap", "Corner parity", "low"), ("comb_foul", "Foul-heavy, cagey game", "high")]


def _derived(f: dict, fair: np.ndarray) -> dict:
    d = {"lg": float(np.log(fair[1] / (1 - fair[1]))), "m_gap": float(abs(fair[0] - fair[2])), "ppg_gap": abs(f["ppg_h"] - f["ppg_a"]),
         "rank_gap": abs(f["rank_h"] - f["rank_a"]), "comb_gf": f["f_gf_h"] + f["f_gf_a"], "comb_ga": f["f_ga_h"] + f["f_ga_a"],
         "comb_cs": f["cs10_h"] + f["cs10_a"], "comb_sot": f["sotpg_h"] + f["sotpg_a"], "crn_gap": abs(f["crn_f_h"] - f["crn_f_a"]),
         "drw10": (f["drw10_h"] + f["drw10_a"]) / 2, "comb_foul": f["foul_h"] + f["foul_a"]}
    return d


def driver_features(f: dict, odds) -> dict:
    inv = np.array([1 / odds[0], 1 / odds[1], 1 / odds[2]])
    out = dict(f)
    out.update(_derived(f, inv / inv.sum()))
    return out


def driver_matrix(F: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    inv = np.column_stack([1 / g.AvgH, 1 / g.AvgD, 1 / g.AvgA])
    fair = inv / inv.sum(1, keepdims=True)
    X = F.copy()
    X["lg"] = np.log(fair[:, 1] / (1 - fair[:, 1]))
    X["m_gap"] = np.abs(fair[:, 0] - fair[:, 2])
    X["ppg_gap"], X["rank_gap"] = np.abs(F.ppg_h - F.ppg_a), np.abs(F.rank_h - F.rank_a)
    X["comb_gf"], X["comb_ga"], X["comb_cs"] = F.f_gf_h + F.f_gf_a, F.f_ga_h + F.f_ga_a, F.cs10_h + F.cs10_a
    X["comb_sot"], X["crn_gap"] = F.sotpg_h + F.sotpg_a, np.abs(F.crn_f_h - F.crn_f_a)
    X["drw10"], X["comb_foul"] = (F.drw10_h + F.drw10_a) / 2, F.foul_h + F.foul_a
    return X


class DrawDriverModel:
    def __init__(self, cols, mean, scale, w, b, ref, note=""):
        self.cols, self.mean, self.scale, self.w, self.b, self.ref, self.note = list(cols), np.array(mean), np.array(scale), np.array(w), float(b), ref, note

    def predict(self, feats: dict, odds) -> float:
        x = (np.array([driver_features(feats, odds)[c] for c in self.cols], float) - self.mean) / self.scale
        return float(1 / (1 + np.exp(-(x @ self.w + self.b))))

    def explain(self, feats: dict, odds) -> list[dict]:
        """Which draw drivers line up for this fixture (vs the 25th/75th percentile of past games)."""
        d = driver_features(feats, odds)
        out = []
        for name, label, direction in DRIVERS:
            lo, hi = self.ref[name]
            v = d[name]
            aligned = v <= lo if direction == "low" else v >= hi
            against = v >= hi if direction == "low" else v <= lo
            out.append({"driver": label, "value": round(float(v), 2), "state": "yes" if aligned else "against" if against else "neutral"})
        return out

    def save(self, path):
        Path(path).write_text(json.dumps({"note": self.note, "cols": self.cols, "mean": self.mean.tolist(), "scale": self.scale.tolist(), "w": self.w.tolist(),
                                          "b": self.b, "ref": self.ref}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        p = Path(path)
        if not p.is_file():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(d["cols"], d["mean"], d["scale"], d["w"], d["b"], d["ref"], d.get("note", ""))


def train_driver(games: pd.DataFrame, F: pd.DataFrame, mask=None, C: float = 0.05, note: str = "") -> DrawDriverModel:
    from sklearn.linear_model import LogisticRegression
    ok = games[["AvgH", "AvgD", "AvgA"]].notna().all(axis=1).values & (np.ones(len(games), bool) if mask is None else mask)
    X = driver_matrix(F[ok].reset_index(drop=True), games[ok].reset_index(drop=True))
    Z = X[DRIVER_COLS].values
    mean, scale = Z.mean(0), Z.std(0)
    scale[scale == 0] = 1.0
    y = (games[ok].FTHG.values == games[ok].FTAG.values).astype(int)
    m = LogisticRegression(C=C, max_iter=1000).fit((Z - mean) / scale, y)
    ref = {name: [float(np.quantile(X[name], 0.25)), float(np.quantile(X[name], 0.75))] for name, _, _ in DRIVERS}
    return DrawDriverModel(DRIVER_COLS, mean, scale, m.coef_[0], m.intercept_[0], ref, note)


class DrawScanner:
    """Third scanner: draw rating from the driver model; flags a pick above ``draw_min`` and a watchlist above ``watch_min``.

    ``legacy`` (optional) is the 4-league Dixon-Coles + booster blend of the original script, shown as a second opinion."""

    def __init__(self, driver: DrawDriverModel, form_predictor, legacy=None, draw_min: float = 0.40, watch_min: float = 0.30, bankroll: float = 1000.0):
        self.driver, self.form, self.legacy, self.draw_min, self.watch_min, self.bankroll = driver, form_predictor, legacy, draw_min, watch_min, bankroll

    def predict_fixture(self, home: str, away: str, odds, draw_min: float | None = None):
        line = self.draw_min if draw_min is None else draw_min
        ctx = self.form.features_for(home, away)
        if ctx is None:
            return None
        feats, div = ctx
        league = {"I2": "Italian Serie B", "F2": "French Ligue 2", "SP2": "Spanish Segunda", "E1": "English Championship"}.get(div, div)
        if not odds:
            return {"div": div, "league": league, "qualified": False, "watch": False, "reason": "no 1X2 odds for this fixture (the draw rating uses the market price)"}
        p = self.driver.predict(feats, odds)
        imp = 1 / odds[1]
        qualified, watch = p > line, (p >= min(self.watch_min, line) and not p > line)
        res = {"div": div, "league": league, "prob": p, "odds": odds[1], "implied": imp, "edge": p - imp, "fair_odds": round(1 / p, 2),
               "qualified": bool(qualified), "watch": bool(watch), "drivers": self.driver.explain(feats, odds),
               "reason": "" if qualified else ("watchlist: above %.0f%% but below your %.0f%% line" % (min(self.watch_min, line) * 100, line * 100) if watch
                                                  else "draw rating below %.0f%%" % (min(self.watch_min, line) * 100)),
               "stake": calculate_kelly_stake(p, odds[1], self.bankroll) if qualified else 0.0}
        if self.legacy is not None:
            lr = self.legacy.predict_fixture(home, away, odds)
            if lr and lr.get("prob") is not None:
                res.update(dc_prob=lr["dc_prob"], xgb_prob=lr["xgb_prob"], blend_prob=lr["prob"])
        return res
