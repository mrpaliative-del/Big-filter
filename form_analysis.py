#!/usr/bin/env python3
"""Does recent form / match statistics add anything? Two walk-forward tests.

    python form_analysis.py data/fd_2025-26_2026-27.csv

Needs: Date, Season, Home, Away, FTHG, FTAG, HS, AS, HST, AST, AvgH, AvgD, AvgA.
 1. Odds path: Poisson regression of goals on rolling form (goals and shots-based) with the market-implied
    expected goals as an offset. Fitted on the first season, judged on the second (+ the early-season weeks).
 2. Ratings path: monthly refits with the current season's games, and shots-based blending, vs frozen ratings.
"""
import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from scanner.fit import fit_ratings
from scanner.model import Rating, evaluate_triple_markets, expected_goals, lambdas_from_1x2


def odds_path_test(df, K=6, SH=3.0):
    df = df.dropna(subset=["AvgH", "AvgD", "AvgA"]).sort_values("Date", kind="stable").reset_index(drop=True)
    lam = [lambdas_from_1x2((r.AvgH, r.AvgD, r.AvgA)) for r in df.itertuples()]
    keep = [l is not None for l in lam]
    df, lam = df[keep].reset_index(drop=True), [l for l in lam if l]
    df["lh"], df["la"] = [l[0] for l in lam], [l[1] for l in lam]
    seasons = sorted(df.Season.unique())
    first, last = seasons[0], seasons[-1]
    tr = df[df.Season == first]
    Z = pd.concat([pd.DataFrame({"s": tr.HS, "t": tr.HST, "g": tr.FTHG}), pd.DataFrame({"s": tr.AS, "t": tr.AST, "g": tr.FTAG})]).dropna()
    c = np.linalg.lstsq(np.column_stack([Z.t, Z.s - Z.t]), Z.g, rcond=None)[0]
    proxy = lambda s, t: c[0] * t + c[1] * (s - t)
    df["ph"], df["pa"] = proxy(df.HS, df.HST), proxy(df.AS, df.AST)
    hist = defaultdict(lambda: {k: deque(maxlen=K) for k in ("gf", "ga", "pf", "pa")})
    f = lambda t, k: (sum(hist[t][k]) / (len(hist[t][k]) + SH)) if hist[t][k] else 0.0
    X, Y, OFF, SEAS, EARLY = [], [], [], [], []
    cut = df[df.Season == last].Date.min() + pd.Timedelta(days=45)
    for r in df.itertuples():
        h, a = r.Home, r.Away
        X += [[f(h, "gf"), f(h, "pf"), f(a, "ga"), f(a, "pa")], [f(a, "gf"), f(a, "pf"), f(h, "ga"), f(h, "pa")]]
        Y += [r.FTHG, r.FTAG]; OFF += [np.log(r.lh), np.log(r.la)]; SEAS += [r.Season] * 2; EARLY += [r.Date < cut] * 2
        hist[h]["gf"].append(r.FTHG - r.lh); hist[h]["ga"].append(r.FTAG - r.la)
        hist[a]["gf"].append(r.FTAG - r.la); hist[a]["ga"].append(r.FTHG - r.lh)
        z = lambda v, m: 0.0 if pd.isna(v) else v - m
        hist[h]["pf"].append(z(r.ph, r.lh)); hist[h]["pa"].append(z(r.pa, r.la))
        hist[a]["pf"].append(z(r.pa, r.la)); hist[a]["pa"].append(z(r.ph, r.lh))
    X, Y, OFF, SEAS, EARLY = map(np.array, (X, np.array(Y, float), OFF, SEAS, EARLY))
    trm, tem = SEAS != last, SEAS == last
    b = np.zeros(4)
    for _ in range(50):
        mu = np.exp(OFF[trm] + X[trm] @ b)
        b += np.linalg.solve((X[trm] * mu[:, None]).T @ X[trm] + np.eye(4), X[trm].T @ (Y[trm] - mu) - b)
    ll = lambda y, mu: y * np.log(mu) - mu
    gain = ll(Y[tem], np.exp(OFF[tem] + X[tem] @ b)) - ll(Y[tem], np.exp(OFF[tem]))
    rng = np.random.default_rng(0)
    print(f"1) ODDS PATH - form on top of the market (fit {first}, test {last})")
    print("   coefficients: own goal-form %+.3f, own shots-form %+.3f, opp goals-against %+.3f, opp shots-against %+.3f" % tuple(b))
    for label, m in (("all test games", np.ones(len(gain), bool)), (f"early season (first 45 days)", EARLY[tem])):
        d = gain[m]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(400)]
        print(f"   {label:30} n={len(d):5d}  log-lik gain/obs {d.mean():+.5f}  95% [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]")
    print("   (a range that includes 0 means form adds nothing beyond the odds)")


def ratings_path_test(df):
    df = df.sort_values("Date", kind="stable")
    seasons = sorted(df.Season.unique()); first, last = seasons[0], seasons[-1]
    tr = df[df.Season == first]
    Z = pd.concat([pd.DataFrame({"s": tr.HS, "t": tr.HST, "g": tr.FTHG}), pd.DataFrame({"s": tr.AS, "t": tr.AST, "g": tr.FTAG})]).dropna()
    c = np.linalg.lstsq(np.column_stack([Z.t, Z.s - Z.t]), Z.g, rcond=None)[0]
    df["ph"], df["pa"] = c[0] * df.HST + c[1] * (df.HS - df.HST), c[0] * df.AST + c[1] * (df.AS - df.AST)
    start = df[df.Season == last].Date.min().normalize().replace(day=1)
    edges = list(pd.date_range(start, df.Date.max() + pd.Timedelta(days=1), freq="MS")) + [df.Date.max() + pd.Timedelta(days=1)]

    def matches(sub, w):
        out = []
        for r in sub.itertuples():
            hg, ag = float(r.FTHG), float(r.FTAG)
            if w > 0 and not (pd.isna(r.ph) or pd.isna(r.pa)):
                hg, ag = (1 - w) * hg + w * r.ph, (1 - w) * ag + w * r.pa
            out.append({"home": r.Home, "away": r.Away, "hg": hg, "ag": ag, "date": r.Date.date()})
        return out

    ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def run(w, hl, refit):
        P, Y, frozen = [], [], None
        for a, b in zip(edges[:-1], edges[1:]):
            if refit or frozen is None:
                cut = a if refit else start
                frozen = fit_ratings(matches(df[df.Date < cut], w), reg=8, min_games=8, half_life_days=hl, today=(cut - pd.Timedelta(days=1)).date())
            for r in df[(df.Date >= a) & (df.Date < b) & (df.Season == last)].itertuples():
                x, y = frozen.get(r.Home), frozen.get(r.Away)
                if x and y:
                    m = evaluate_triple_markets(*expected_goals(Rating(**x), Rating(**y)))
                    P.append((m["1x_prob"] / 100, m["under_2_5_prob"] / 100, m["all_three_prob"]))
                    Y.append((r.FTHG >= r.FTAG, r.FTHG + r.FTAG <= 2, r.FTHG >= r.FTAG and r.FTHG + r.FTAG <= 2))
        return np.array(P), np.array(Y, float)

    print(f"\n2) RATINGS PATH - start-of-season updating and shots statistics (test {last})")
    for label, w, hl, refit in (("frozen end-of-last-season ratings", 0, 365, False), ("monthly refit, goals, half-life 365", 0, 365, True),
                                ("monthly refit, goals, half-life 150", 0, 150, True), ("monthly refit, 50% shots-proxy", 0.5, 365, True)):
        P, Y = run(w, hl, refit)
        print(f"   {label:38} n={len(P):5d}  1X {ll(P[:, 0], Y[:, 0]):.4f}  U2.5 {ll(P[:, 1], Y[:, 1]):.4f}  All3 {ll(P[:, 2], Y[:, 2]):.4f}")
    print(f"   {'flat base rate (hindsight)':38} {'':12}  1X {ll(np.full(len(Y), Y[:, 0].mean()), Y[:, 0]):.4f}  U2.5 {ll(np.full(len(Y), Y[:, 1].mean()), Y[:, 1]):.4f}  All3 {ll(np.full(len(Y), Y[:, 2].mean()), Y[:, 2]):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    args = ap.parse_args()
    data = pd.read_csv(args.csv, parse_dates=["Date"])
    odds_path_test(data.copy())
    ratings_path_test(data.copy())
