#!/usr/bin/env python3
"""Established football technical analysis, tested for value against the market.

    python technical_analysis.py data/fd_2025-26_2026-27.csv

A) Does an indicator predict goals BEYOND what the pre-match odds already say?
   (Poisson regression with the odds-implied expected goals as offset; fit on the first season,
    judged on the second; 95% bootstrap range on the log-likelihood gain per observation.)
   Indicators: Elo rating gap, rest days, home/away-split form, shots form, goal form.
B) Does the over/under price pin down total goals better than 1X2 alone?
C) Sharp-vs-soft value betting: best bookmaker price vs the Pinnacle no-vig fair price.
D) Early price vs the closing line (closing line value).
"""
import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from scanner.model import DEFAULT_RHO, _grid_1x2, lambdas_from_1x2

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ A) indicators
def indicator_test(df, K=6, SH=3.0):
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    lam = [lambdas_from_1x2((r.AvgH, r.AvgD, r.AvgA)) for r in df.itertuples()]
    ok = [l is not None for l in lam]
    df, lam = df[ok].reset_index(drop=True), [l for l in lam if l]
    df["lh"], df["la"] = [l[0] for l in lam], [l[1] for l in lam]
    seasons = sorted(df.Season.unique()); last = seasons[-1]
    elo, last_day = defaultdict(lambda: 1500.0), {}
    H = lambda: {k: deque(maxlen=K) for k in ("gf", "ga", "sf", "sa", "hgf", "hga", "agf", "aga")}
    hist = defaultdict(H)
    sm = lambda t, k: (sum(hist[t][k]) / (len(hist[t][k]) + SH)) if hist[t][k] else 0.0
    cols = ["Elo gap", "Rest-day gap", "Home/away split form", "Shots form", "Goal form"]
    X, Y, OFF, SEAS = [], [], [], []
    for r in df.itertuples():
        h, a = r.Home, r.Away
        dr = (elo[h] + 60 - elo[a]) / 400
        rest = lambda t: min((r.Date - last_day[t]).days, 10) if t in last_day else 7
        rg = (rest(h) - rest(a)) / 5
        fh = [dr, rg, sm(h, "hgf") + sm(a, "aga"), sm(h, "sf") + sm(a, "sa"), sm(h, "gf") + sm(a, "ga")]
        fa = [-dr, -rg, sm(a, "agf") + sm(h, "hga"), sm(a, "sf") + sm(h, "sa"), sm(a, "gf") + sm(h, "ga")]
        X += [fh, fa]; Y += [r.FTHG, r.FTAG]; OFF += [np.log(r.lh), np.log(r.la)]; SEAS += [r.Season] * 2
        # updates after the game
        gd = r.FTHG - r.FTAG
        G = 1 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
        we = 1 / (1 + 10 ** (-(elo[h] + 60 - elo[a]) / 400)); w = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        elo[h] += 20 * G * (w - we); elo[a] -= 20 * G * (w - we)
        last_day[h] = last_day[a] = r.Date
        for t, gf, ga, lf, la_, sf, sa in ((h, r.FTHG, r.FTAG, r.lh, r.la, r.HS, r.AS), (a, r.FTAG, r.FTHG, r.la, r.lh, r.AS, r.HS)):
            hist[t]["gf"].append(gf - lf); hist[t]["ga"].append(ga - la_)
            hist[t]["sf"].append(0.0 if pd.isna(sf) else 0.1 * sf - lf); hist[t]["sa"].append(0.0 if pd.isna(sa) else 0.1 * sa - la_)
        hist[h]["hgf"].append(r.FTHG - r.lh); hist[h]["hga"].append(r.FTAG - r.la)
        hist[a]["agf"].append(r.FTAG - r.la); hist[a]["aga"].append(r.FTHG - r.lh)
    X, Y, OFF, SEAS = np.array(X), np.array(Y, float), np.array(OFF), np.array(SEAS)
    trm, tem = SEAS != last, SEAS == last

    def fit(cols_idx):
        Xt = X[trm][:, cols_idx]; b = np.zeros(len(cols_idx))
        for _ in range(60):
            mu = np.exp(OFF[trm] + Xt @ b)
            b += np.linalg.solve((Xt * mu[:, None]).T @ Xt + np.eye(len(b)), Xt.T @ (Y[trm] - mu) - b)
        return b

    ll = lambda y, mu: y * np.log(mu) - mu
    base = ll(Y[tem], np.exp(OFF[tem]))
    print(f"A) INDICATORS vs THE ODDS (fit {seasons[0]}, test {last}, {tem.sum() // 2} games)")
    print(f"   {'indicator':24}{'coef':>8}{'gain/obs':>10}   95% range        verdict")
    for j, name in enumerate(cols + ["ALL TOGETHER"]):
        idx = [j] if j < len(cols) else list(range(len(cols)))
        b = fit(idx); g = ll(Y[tem], np.exp(OFF[tem] + X[tem][:, idx] @ b)) - base
        bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(300)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "adds value" if lo > 0 else "nothing beyond the odds"
        print(f"   {name:24}{(b[0] if len(idx) == 1 else float('nan')):8.3f}{g.mean():+10.5f}   [{lo:+.5f}, {hi:+.5f}]  {verdict}")


# ------------------------------------------------------------------ B) over/under pins totals
def totals_test(df):
    d = df.dropna(subset=["AvgH", "AvgD", "AvgA", "AvgO25", "AvgU25"]).reset_index(drop=True)
    k = np.arange(10); fact = np.cumprod(np.maximum(k, 1)).astype(float)
    xs, ys = np.indices((10, 10)); under = (xs + ys) <= 2

    def solve(tH, tD, tU):
        lo, hi = np.log(0.05), np.log(6.0); ch = ca = (lo + hi) / 2; half = (hi - lo) / 2
        for _ in range(5):
            gh = np.clip(np.exp(np.linspace(ch - half, ch + half, 25)), 0.02, 8); ga = np.clip(np.exp(np.linspace(ca - half, ca + half, 25)), 0.02, 8)
            ph = np.exp(-gh[:, None]) * gh[:, None] ** k / fact; pa = np.exp(-ga[:, None]) * ga[:, None] ** k / fact
            j = ph[:, None, :, None] * pa[None, :, None, :]
            LH, LA, rho = gh[:, None], ga[None, :], DEFAULT_RHO
            j[:, :, 0, 0] *= 1 - LH * LA * rho; j[:, :, 0, 1] *= 1 + LH * rho; j[:, :, 1, 0] *= 1 + LA * rho; j[:, :, 1, 1] *= 1 - rho
            j = np.clip(j, 0, None); j /= j.sum(axis=(2, 3), keepdims=True)
            loss = (j[:, :, xs > ys].sum(-1) - tH) ** 2 + (j[:, :, xs == ys].sum(-1) - tD) ** 2 + (j[:, :, under].sum(-1) - tU) ** 2
            i, jj = np.unravel_index(int(loss.argmin()), loss.shape); ch, ca, half = np.log(gh[i]), np.log(ga[jj]), half / 5
            best = (gh[i], ga[jj])
        return best

    from scanner.model import evaluate_triple_markets
    A, B, Y, M = [], [], [], []
    for r in d.itertuples():
        inv = np.array([1 / r.AvgH, 1 / r.AvgD, 1 / r.AvgA]); inv /= inv.sum()
        tU = (1 / r.AvgU25) / (1 / r.AvgU25 + 1 / r.AvgO25)
        l1 = lambdas_from_1x2((r.AvgH, r.AvgD, r.AvgA)); l2 = solve(inv[0], inv[1], tU)
        if not l1: continue
        m1, m2 = evaluate_triple_markets(*l1), evaluate_triple_markets(*l2)
        A.append((m1["1x_prob"], m1["under_2_5_prob"], m1["all_three_prob"] * 100)); B.append((m2["1x_prob"], m2["under_2_5_prob"], m2["all_three_prob"] * 100))
        M.append(tU * 100); hg, ag = r.FTHG, r.FTAG; Y.append((hg >= ag, hg + ag <= 2, hg >= ag and hg + ag <= 2))
    A, B, Y, M = np.array(A) / 100, np.array(B) / 100, np.array(Y, float), np.array(M) / 100
    ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(1 - np.clip(p, 1e-6, 1 - 1e-6)))
    print(f"\nB) USING THE OVER/UNDER PRICE TO PIN TOTAL GOALS ({len(Y)} games, log-loss, lower is better)")
    print(f"   {'':26}{'1X':>8}{'Under 2.5':>11}{'All3':>8}")
    print(f"   {'goals from 1X2 odds only':26}{ll(A[:, 0], Y[:, 0]):8.4f}{ll(A[:, 1], Y[:, 1]):11.4f}{ll(A[:, 2], Y[:, 2]):8.4f}")
    print(f"   {'1X2 + over/under odds':26}{ll(B[:, 0], Y[:, 0]):8.4f}{ll(B[:, 1], Y[:, 1]):11.4f}{ll(B[:, 2], Y[:, 2]):8.4f}")
    print(f"   {"bookmaker's own U2.5 price":26}{'':8}{ll(M, Y[:, 1]):11.4f}")


# ------------------------------------------------------------------ C) sharp vs soft value
def value_test(df):
    d = df.dropna(subset=["PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA"]).reset_index(drop=True)
    inv = np.column_stack([1 / d.PSH, 1 / d.PSD, 1 / d.PSA]); fair = inv / inv.sum(1, keepdims=True)
    best = d[["MaxH", "MaxD", "MaxA"]].values
    res = np.column_stack([d.FTHG > d.FTAG, d.FTHG == d.FTAG, d.FTHG < d.FTAG]).astype(float)
    ev = best * fair - 1; ret = np.where(res == 1, best - 1, -1)
    print(f"\nC) VALUE BETTING: best bookmaker price vs Pinnacle no-vig fair price ({len(d)} games with Pinnacle)")
    print(f"   {'min EV':>7}{'bets':>7}{'ROI':>8}   95% range")
    for t in (0.0, 0.02, 0.04, 0.06):
        sel = ev >= t; r = ret[sel]
        if len(r) < 20:
            print(f"   {t:7.2f}{len(r):7d}   (too few)"); continue
        bs = [r[rng.integers(0, len(r), len(r))].mean() for _ in range(300)]
        print(f"   {t:7.2f}{len(r):7d}{r.mean():8.3f}   [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")


# ------------------------------------------------------------------ D) closing line value
def clv_test(df):
    d = df.dropna(subset=["AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA"]).reset_index(drop=True)
    inv = np.column_stack([1 / d.AvgCH, 1 / d.AvgCD, 1 / d.AvgCA]); fair = inv / inv.sum(1, keepdims=True)
    early = d[["AvgH", "AvgD", "AvgA"]].values; clv = early * fair - 1
    res = np.column_stack([d.FTHG > d.FTAG, d.FTHG == d.FTAG, d.FTHG < d.FTAG]).astype(float)
    print(f"\nD) EARLY PRICE vs CLOSING LINE ({len(d)} games): expected value of the early average price at the closing fair odds")
    print(f"   all outcomes: {clv.mean():+.3f}   home {clv[:, 0].mean():+.3f}  draw {clv[:, 1].mean():+.3f}  away {clv[:, 2].mean():+.3f}")
    mv = np.log(d.AvgCH / d.AvgH).values            # shortening home price = money coming for the home side
    move_in = mv < -0.05
    ret = np.where(res[:, 0] == 1, d.AvgH - 1, -1)
    print(f"   home price shortened >5% by the close: {move_in.sum()} games, early-price ROI {ret[move_in].mean():+.3f} (not knowable in advance)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    args = ap.parse_args()
    data = pd.read_csv(args.csv, parse_dates=["Date"])
    indicator_test(data.copy()); totals_test(data.copy()); value_test(data.copy()); clv_test(data.copy())
