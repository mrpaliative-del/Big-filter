"""Draw predictor (the third scanner): time-weighted Dixon-Coles + calibrated gradient-boosted classifier, blended.

Ported from the user's ``draw_predictor_pipeline.py`` with the same maths and rules, made faster (the Dixon-Coles likelihood is
vectorised) and embeddable. The booster is XGBoost when installed, else scikit-learn's HistGradientBoostingClassifier.

Rules (unchanged): a draw is a pick when blend probability - market implied draw probability (1/odds) >= MIN_EDGE and the
market draw odds are <= MAX_ODDS; stake = quarter Kelly capped at 3% of the bankroll.
"""
from __future__ import annotations

import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

from .model import _norm

LEAGUES = {"I2": "Italian Serie B", "F2": "French Ligue 2", "SP2": "Spanish Segunda", "E1": "English Championship"}
MIN_EDGE = 0.04
MAX_ODDS = 4.10
BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.03
XI = 0.0022                    # Dixon-Coles time decay per day (half-life ~315 days)
WINDOW = 5
FEATURE_COLS = ["h_sot_for", "h_sot_against", "a_sot_for", "a_sot_against", "sot_diff", "combined_corners", "combined_fouls",
                "market_draw_prob", "home_away_prob_gap"]
REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST", "HC", "AC", "HF", "AF", "AvgH", "AvgD", "AvgA"]


# ------------------------------------------------------------------------------ Dixon-Coles
def dixon_coles_tau(h_goals, a_goals, h_xg, a_xg, rho):
    """Scalar form (reference). The vectorised ``_tau_vec`` is what the fit uses."""
    if h_goals == 0 and a_goals == 0:
        return 1.0 - (h_xg * a_xg * rho)
    if h_goals == 0 and a_goals == 1:
        return 1.0 + (h_xg * rho)
    if h_goals == 1 and a_goals == 0:
        return 1.0 + (a_xg * rho)
    if h_goals == 1 and a_goals == 1:
        return 1.0 - rho
    return 1.0


def _tau_vec(hg, ag, hx, ax, rho):
    tau = np.ones_like(hx, dtype=float)
    m00, m01, m10, m11 = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1), (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - hx[m00] * ax[m00] * rho
    tau[m01] = 1.0 + hx[m01] * rho
    tau[m10] = 1.0 + ax[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def fit_dixon_coles_weighted(df, reference_date, xi=XI, maxiter=None):
    teams = np.sort(np.unique(np.concatenate([df["HomeTeam"].values, df["AwayTeam"].values])))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai = df["HomeTeam"].map(idx).values, df["AwayTeam"].map(idx).values
    hg, ag = df["FTHG"].values.astype(float), df["FTAG"].values.astype(float)
    w = np.exp(-xi * np.maximum((reference_date - df["Date"]).dt.days.values, 0))
    lg_h, lg_a = gammaln(hg + 1), gammaln(ag + 1)

    def nll(p):
        att, dfn, home_adv, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
        hx, ax = np.exp(att[hi] + dfn[ai] + home_adv), np.exp(att[ai] + dfn[hi])
        tau = _tau_vec(hg, ag, hx, ax, rho)
        if np.any(tau <= 0):
            return 1e10
        return -np.sum(w * (np.log(tau) + (hg * np.log(hx) - hx - lg_h) + (ag * np.log(ax) - ax - lg_a)))

    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])
    res = minimize(nll, x0, method="SLSQP", bounds=[(-3, 3)] * (2 * n) + [(0, 1)] + [(-0.5, 0.5)],
                   constraints=[{"type": "eq", "fun": lambda p: np.mean(p[:n])}], options={"maxiter": maxiter} if maxiter else None)
    f = res.x
    return {"teams": set(teams.tolist()), "attack": dict(zip(teams, f[:n])), "defense": dict(zip(teams, f[n:2 * n])),
            "home_adv": float(f[2 * n]), "rho": float(f[2 * n + 1])}


def predict_dc_prob(model, home_team, away_team, max_goals=6):
    """Probability of a draw."""
    if home_team not in model["teams"] or away_team not in model["teams"]:
        return None
    hx = np.exp(model["attack"][home_team] + model["defense"][away_team] + model["home_adv"])
    ax = np.exp(model["attack"][away_team] + model["defense"][home_team])
    rho = model["rho"]
    k = np.arange(max_goals + 1)
    ph, pa = np.exp(k * np.log(hx) - hx - gammaln(k + 1)), np.exp(k * np.log(ax) - ax - gammaln(k + 1))
    m = np.outer(ph, pa)
    m[0, 0] *= 1.0 - hx * ax * rho
    m[0, 1] *= 1.0 + hx * rho
    m[1, 0] *= 1.0 + ax * rho
    m[1, 1] *= 1.0 - rho
    m /= m.sum()
    return float(np.trace(m))


def calculate_kelly_stake(prob, odds, bankroll, fraction=KELLY_FRACTION, max_stake_pct=MAX_STAKE_PCT):
    b = odds - 1.0
    kelly_f = ((b * prob) - (1.0 - prob)) / b
    if kelly_f <= 0:
        return 0.0
    return round(bankroll * min(kelly_f * fraction, max_stake_pct), 2)


# ------------------------------------------------------------------------------ features
def to_script_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Repo history (Home/Away) -> the script's column names (HomeTeam/AwayTeam)."""
    out = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"}).copy()
    return out.dropna(subset=[c for c in REQUIRED if c in out.columns])


def _team_log_entry(sot_for, sot_against, corners, fouls):
    return (sot_for, sot_against, corners, fouls)


def build_features_and_latest_state(raw_df, window=WINDOW):
    df = raw_df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["Is_Draw"] = (df["FTHG"] == df["FTAG"]).astype(int)
    logs = defaultdict(list)
    feats = []
    for r in df.itertuples():
        h, a = r.HomeTeam, r.AwayTeam
        hh, aa = logs[h][-window:], logs[a][-window:]
        if len(hh) < window or len(aa) < window:
            feats.append({})
        else:
            hm, am = np.mean(hh, axis=0), np.mean(aa, axis=0)
            feats.append({"h_sot_for": hm[0], "h_sot_against": hm[1], "a_sot_for": am[0], "a_sot_against": am[1],
                          "sot_diff": abs(hm[0] - am[0]), "combined_corners": hm[2] + am[2], "combined_fouls": hm[3] + am[3],
                          "market_draw_prob": 1.0 / r.AvgD, "home_away_prob_gap": abs(1.0 / r.AvgH - 1.0 / r.AvgA)})
        logs[h].append((r.HST, r.AST, r.HC, r.HF))
        logs[a].append((r.AST, r.HST, r.AC, r.AF))
    full = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    return full, latest_form_from_logs(logs, window)


def latest_form_from_logs(logs, window=WINDOW):
    out = {}
    for t, lg in logs.items():
        if len(lg) >= window:
            m = np.mean(lg[-window:], axis=0)
            out[t] = {"sot_for": m[0], "sot_against": m[1], "corners": m[2], "fouls": m[3]}
    return out


def latest_form_from_history(df: pd.DataFrame, window=WINDOW) -> dict:
    """Fresh team form state from (script-schema) results; used at scan time so new results count without retraining."""
    logs = defaultdict(list)
    for r in df.sort_values("Date", kind="stable").itertuples():
        logs[r.HomeTeam].append((r.HST, r.AST, r.HC, r.HF))
        logs[r.AwayTeam].append((r.AST, r.HST, r.AC, r.AF))
    return latest_form_from_logs(logs, window)


def make_booster():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=42), "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=120, max_depth=3, learning_rate=0.03, random_state=42), "sklearn-histgb"


def train_calibrated_xgb(train_df, feature_cols=FEATURE_COLS):
    clean = train_df.dropna(subset=feature_cols)
    base, _ = make_booster()
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    model.fit(clean[feature_cols], clean["Is_Draw"])
    return model


def learn_optimal_blend_weights(full_df, feature_cols=FEATURE_COLS, eval_tail=150, dc_maxiter=None):
    clean = full_df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(clean) < eval_tail + 150:
        return 0.65, 0.35
    split = len(clean) - eval_tail
    train_part, val_part = clean.iloc[:split], clean.iloc[split:]
    dc = fit_dixon_coles_weighted(train_part, train_part["Date"].max(), maxiter=dc_maxiter)
    xgb = train_calibrated_xgb(train_part, feature_cols)
    y, dcp, xp = [], [], []
    for _, row in val_part.iterrows():
        p = predict_dc_prob(dc, row["HomeTeam"], row["AwayTeam"])
        if p is None:
            continue
        y.append(row["Is_Draw"]); dcp.append(p)
        xp.append(float(xgb.predict_proba(pd.DataFrame([row[feature_cols]]))[0, 1]))
    if len(y) < 50:
        return 0.65, 0.35
    y, dcp, xp = np.array(y), np.array(dcp), np.array(xp)
    res = minimize_scalar(lambda w: log_loss(y, np.clip(w * dcp + (1 - w) * xp, 1e-6, 1 - 1e-6)), bounds=(0.0, 1.0), method="bounded")
    w = round(float(res.x), 4)
    return w, round(1.0 - w, 4)


# ------------------------------------------------------------------------------ one league
@dataclass
class LeagueModel:
    div: str
    dc: dict
    cal: object
    dc_w: float
    xgb_w: float
    trained_through: pd.Timestamp
    n_games: int
    booster: str

    def predict(self, home, away, form, market_odds):
        """(dc_prob, xgb_prob, blend) or None if a team is unknown / lacks a form window."""
        if home not in form or away not in form:
            return None
        dc_p = predict_dc_prob(self.dc, home, away)
        if dc_p is None:
            return None
        hs, as_ = form[home], form[away]
        h_odds, d_odds, a_odds = market_odds
        x = pd.DataFrame([{"h_sot_for": hs["sot_for"], "h_sot_against": hs["sot_against"], "a_sot_for": as_["sot_for"],
                           "a_sot_against": as_["sot_against"], "sot_diff": abs(hs["sot_for"] - as_["sot_for"]),
                           "combined_corners": hs["corners"] + as_["corners"], "combined_fouls": hs["fouls"] + as_["fouls"],
                           "market_draw_prob": 1.0 / d_odds, "home_away_prob_gap": abs(1.0 / h_odds - 1.0 / a_odds)}])[FEATURE_COLS]
        xg_p = float(self.cal.predict_proba(x)[0, 1])
        return dc_p, xg_p, self.dc_w * dc_p + self.xgb_w * xg_p


def train_league(div: str, raw_df: pd.DataFrame, dc_maxiter=None) -> LeagueModel | None:
    """Fit one league exactly as the script does: features -> blend weights -> final DC and calibrated booster."""
    if len(raw_df) < 200:
        return None
    full, _ = build_features_and_latest_state(raw_df)
    dc_w, xgb_w = learn_optimal_blend_weights(full, dc_maxiter=dc_maxiter)
    dc = fit_dixon_coles_weighted(full, full["Date"].max(), maxiter=dc_maxiter)
    cal = train_calibrated_xgb(full)
    return LeagueModel(div, dc, cal, dc_w, xgb_w, full["Date"].max(), len(full), make_booster()[1])


# ------------------------------------------------------------------------------ the scanner
def evaluate_pick(prob, odds, min_edge=MIN_EDGE, max_odds=MAX_ODDS, bankroll=BANKROLL):
    imp = 1.0 / odds
    edge = prob - imp
    ok_price = odds <= max_odds
    qualified = bool(edge >= min_edge and ok_price)
    reason = "" if qualified else ("draw odds above %.2f" % max_odds if not ok_price else "edge below %.0f%%" % (min_edge * 100))
    return {"implied": imp, "edge": edge, "fair_odds": round(1.0 / prob, 2), "qualified": qualified, "reason": reason,
            "stake": calculate_kelly_stake(prob, odds, bankroll) if qualified else 0.0}


class DrawPredictor:
    """Trained league models + current results history; scores a fixture's draw by team name and 1X2 odds."""

    def __init__(self, models: dict[str, LeagueModel], history: pd.DataFrame, aliases: dict | None = None,
                 min_edge=MIN_EDGE, max_odds=MAX_ODDS, bankroll=BANKROLL):
        self.models, self.min_edge, self.max_odds, self.bankroll = models, min_edge, max_odds, bankroll
        self.aliases = {_norm(k): v for k, v in (aliases or {}).items()}
        self.form, self.names, self.div_of = {}, {}, {}
        hist = to_script_schema(history)
        for div in models:
            sub = hist[hist["Div"] == div]
            self.form[div] = latest_form_from_history(sub)
            for t in set(sub["HomeTeam"]) | set(sub["AwayTeam"]):
                self.names.setdefault(_norm(t), t)
                self.div_of[t] = div

    def resolve(self, name):
        n = _norm(name)
        return self.names.get(_norm(self.aliases.get(n, n)))

    def predict_fixture(self, home: str, away: str, odds):
        """Dict with probabilities, price, edge and the verdict; ``reason`` explains any non-pick. None if the league is not covered."""
        h, a = self.resolve(home), self.resolve(away)
        if not h or not a or self.div_of.get(h) != self.div_of.get(a) or self.div_of.get(h) not in self.models:
            return None
        div = self.div_of[h]
        if not odds:
            return {"div": div, "league": LEAGUES.get(div, div), "qualified": False, "reason": "no 1X2 odds for this fixture (the draw scanner prices against the market)"}
        lm = self.models[div]
        p = lm.predict(h, a, self.form[div], odds)
        if p is None:
            return {"div": div, "league": LEAGUES.get(div, div), "qualified": False, "reason": "fewer than 5 recent games for a team"}
        dc_p, xg_p, blend = p
        ev = evaluate_pick(blend, odds[1], self.min_edge, self.max_odds, self.bankroll)
        return {"div": div, "league": LEAGUES.get(div, div), "dc_prob": dc_p, "xgb_prob": xg_p, "prob": blend, "odds": odds[1],
                "dc_weight": lm.dc_w, "booster": lm.booster, **ev}

    # ---- persistence (the models are trained by train_draw_models.py)
    def save_models(self, path):
        pickle.dump({"version": 1, "models": self.models}, open(path, "wb"))

    @staticmethod
    def load_models(path):
        p = Path(path)
        if not p.is_file():
            return {}
        return pickle.load(open(p, "rb"))["models"]


# ------------------------------------------------------------------------------ the script's weekly flow
PICK_COLUMNS = ["League", "Date", "Time", "HomeTeam", "AwayTeam", "DC_Weight", "DC_Prob_Pct", "Cal_XGB_Pct", "Blend_Prob_Pct", "Fair_Odds",
                "Market_Odds", "Edge_Pct", "Kelly_Stake"]


def weekly_scan(fixtures_df: pd.DataFrame, histories: dict, min_edge=MIN_EDGE, max_odds=MAX_ODDS, bankroll=BANKROLL, log=print) -> pd.DataFrame:
    """The original script's loop: per league fit, score every upcoming fixture, keep edge >= min_edge. ``histories`` maps div -> results frame
    (script schema: HomeTeam/AwayTeam, needs HST/AST/HC/AC/HF/AF and Avg odds). Returns the picks, best edge first."""
    picks = []
    for div, league in LEAGUES.items():
        fx = fixtures_df[fixtures_df["Div"] == div]
        if fx.empty:
            log(f"  -> No upcoming fixtures found for {league}.")
            continue
        hist = histories.get(div)
        if hist is None or len(hist) < 200:
            log(f"  -> Insufficient historical matches for {league}.")
            continue
        lm = train_league(div, hist)
        _, form = build_features_and_latest_state(hist)
        log(f"  -> {league}: learned weights | Dixon-Coles {lm.dc_w} | booster {lm.xgb_w}")
        for _, r in fx.iterrows():
            h, a = r["HomeTeam"], r["AwayTeam"]
            odds = (r.get("AvgH", np.nan), r.get("AvgD", np.nan), r.get("AvgA", np.nan))
            if any(pd.isna(o) for o in odds) or odds[1] > max_odds:
                continue
            p = lm.predict(h, a, form, odds)
            if p is None:
                continue
            dc_p, xg_p, blend = p
            ev = evaluate_pick(blend, odds[1], min_edge, max_odds, bankroll)
            if ev["edge"] >= min_edge:
                picks.append({"League": league, "Date": r["Date"], "Time": r.get("Time", ""), "HomeTeam": h, "AwayTeam": a, "DC_Weight": lm.dc_w,
                              "DC_Prob_Pct": round(dc_p * 100, 2), "Cal_XGB_Pct": round(xg_p * 100, 2), "Blend_Prob_Pct": round(blend * 100, 2),
                              "Fair_Odds": ev["fair_odds"], "Market_Odds": odds[1], "Edge_Pct": round(ev["edge"] * 100, 2), "Kelly_Stake": ev["stake"]})
    if not picks:
        return pd.DataFrame(columns=PICK_COLUMNS)
    return pd.DataFrame(picks).sort_values(by="Edge_Pct", ascending=False).reset_index(drop=True)
