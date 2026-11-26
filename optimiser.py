#!/usr/bin/env python3
"""SofaScore-style analysis as an All3 optimiser, tested against the market.

    python optimiser.py data/fd_2025-26_2026-27.csv

1. Builds pre-match features for every game from earlier games only: standings (position, points, goal
   difference, matches played), form, streaks, head-to-head, past match statistics, and psychology proxies
   (bounce-back, big-win complacency, late-season stakes, dead rubbers).
2. Trains a stacked model on 2025/26 (market probability + features) and judges it on unseen 2026/27 games,
   for two All3 definitions, against the market probability alone.
3. Scans the price data for real 3-way / 2-way arbitrage (best price per outcome across bookmakers).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scanner.features import GROUPS, build_features
from scanner.model import dixon_coles_matrix, lambdas_from_1x2

rng = np.random.default_rng(0)
x_, y_ = np.indices((10, 10))
TRIPLES = {
    "A: 1X + Home to score + Under 4.5": (lambda h, a: h >= a and h >= 1 and h + a <= 4, (x_ >= y_) & (x_ >= 1) & ((x_ + y_) <= 4)),
    "Current: 1X + Under 2.5": (lambda h, a: h >= a and h + a <= 2, (x_ >= y_) & ((x_ + y_) <= 2)),
}
def build(df):
    games, F, _ = build_features(df)
    return games, F


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def stacking(df, F):
    ok = [lambdas_from_1x2((r.AvgH, r.AvgD, r.AvgA)) for r in df.itertuples()]
    keep = np.array([o is not None for o in ok])
    df, F = df[keep].reset_index(drop=True), F[keep].reset_index(drop=True)
    mats = [dixon_coles_matrix(*o) for o in ok if o is not None]
    tr, te = (df.Season == "2025/26").values, (df.Season == "2026/27").values
    last = df.Season.iloc[-1]
    print(f"2) STACKED MODEL: market probability + SofaScore-style features  (train 2025/26 = {tr.sum()}, test {last} = {te.sum()} games)")
    for name, (fn, mask) in TRIPLES.items():
        y = np.array([fn(int(r.FTHG), int(r.FTAG)) for r in df.itertuples()], float)
        pm = np.array([m[mask].sum() for m in mats]); lm = np.log(np.clip(pm, 1e-4, 1 - 1e-4) / (1 - np.clip(pm, 1e-4, 1 - 1e-4)))
        base = LogisticRegression(C=100).fit(lm[tr, None], y[tr]); pb = base.predict_proba(lm[te, None])[:, 1]
        lb = logloss(pb, y[te])
        print(f"\n   {name}   (actual {y[te].mean():.3f}, market-only predicted {pm[te].mean():.3f})")
        print(f"   {'features added':26}{'log-loss gain':>14}   95% range            >=50% predicted: n, hit")
        sel0 = pb >= 0.5
        print(f"   {'market only (baseline)':26}{0:14.5f}   {'':20}   {sel0.sum():5d}, {y[te][sel0].mean() if sel0.any() else float('nan'):.3f}")
        sets = {**{g: c for g, c in GROUPS.items()}, "ALL (linear)": sum(GROUPS.values(), [])}
        for label, cols in sets.items():
            Z = F[cols].values; sc = StandardScaler().fit(Z[tr])
            X = np.column_stack([lm, sc.transform(Z)])
            m = LogisticRegression(C=0.05, max_iter=500).fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
            d = lb - logloss(p, y[te]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
            lo, hi = np.percentile(bs, [2.5, 97.5]); sel = p >= 0.5
            tag = "  <-- adds value" if lo > 0 else ""
            print(f"   {label:26}{d.mean():+14.5f}   [{lo:+.5f}, {hi:+.5f}]   {sel.sum():5d}, {y[te][sel].mean() if sel.any() else float('nan'):.3f}{tag}")
        Z = F[sum(GROUPS.values(), [])].values
        gb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_depth=3, l2_regularization=5.0, random_state=0)
        X = np.column_stack([lm, Z]); gb.fit(X[tr], y[tr]); p = gb.predict_proba(X[te])[:, 1]
        d = lb - logloss(p, y[te]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
        lo, hi = np.percentile(bs, [2.5, 97.5]); sel = p >= 0.5
        print(f"   {'ALL (gradient boosting)':26}{d.mean():+14.5f}   [{lo:+.5f}, {hi:+.5f}]   {sel.sum():5d}, {y[te][sel].mean() if sel.any() else float('nan'):.3f}")


def arbitrage(df):
    print("\n3) ARBITRAGE SCAN: best price per outcome across bookmakers, same snapshot")
    def scan(label, cols):
        d = df.dropna(subset=cols)
        book = sum(1 / d[c] for c in cols)
        arb = book < 1
        print(f"   {label:22}{len(d):6d} games   arbitrage in {arb.sum():4d} ({arb.mean():.1%});  average best-price overround {book.mean():.3f}"
              + (f";  mean locked profit when present {((1 / book[arb]) - 1).mean():.2%}, max {((1 / book[arb]) - 1).max():.2%}" if arb.any() else ""))
    scan("1X2 (3-way)", ["MaxH", "MaxD", "MaxA"]); scan("Over/Under 2.5 (2-way)", ["MaxO25", "MaxU25"]); scan("Asian handicap (2-way)", ["MaxAHH", "MaxAHA"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    args = ap.parse_args()
    data = pd.read_csv(args.csv, parse_dates=["Date"])
    print("1) building pre-match features (standings, form, streaks, head-to-head, match statistics, psychology)...")
    d, F = build(data)
    print(f"   {len(F.columns)} features for {len(d)} games\n")
    stacking(d, F)
    arbitrage(d)
