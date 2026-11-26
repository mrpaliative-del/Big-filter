#!/usr/bin/env python3
"""Walk-forward test of the draw scanner on real results.

    python draw_backtest.py data/fd_2025-26_2026-27.csv --test-from 2026-08-01

For each test month the league models are refitted on games BEFORE that month only (as the weekly script does), then the month's games are scored:
blend / Dixon-Coles / booster draw probabilities against the real average draw price (AvgD). Reports probability quality vs the market and the
return of the script's own pick rule (edge >= 4 points, draw odds <= 4.10).
"""
import argparse

import numpy as np
import pandas as pd

from scanner.draw import FEATURE_COLS, LEAGUES, MAX_ODDS, MIN_EDGE, build_features_and_latest_state, predict_dc_prob, to_script_schema, train_league
from picklog import wilson


def collect(df, test_from):
    rows = []
    for div in LEAGUES:
        sub = to_script_schema(df[df.Div == div]).sort_values("Date", kind="stable").reset_index(drop=True)
        if sub.empty:
            continue
        full, _ = build_features_and_latest_state(sub)                       # features per game use earlier games only
        months = pd.date_range(pd.Timestamp(test_from), full.Date.max() + pd.offsets.MonthBegin(1), freq="MS")
        for a, b in zip(months[:-1], months[1:]):
            lm = train_league(div, sub[sub.Date < a])
            if lm is None:
                continue
            for _, r in full[(full.Date >= a) & (full.Date < b)].dropna(subset=FEATURE_COLS).iterrows():
                dc = predict_dc_prob(lm.dc, r.HomeTeam, r.AwayTeam)
                if dc is None:
                    continue
                xg = float(lm.cal.predict_proba(pd.DataFrame([r[FEATURE_COLS]]))[0, 1])
                inv = np.array([1 / r.AvgH, 1 / r.AvgD, 1 / r.AvgA])
                rows.append(dict(div=div, date=r.Date, dc=dc, xgb=xg, blend=lm.dc_w * dc + lm.xgb_w * xg, odds=r.AvgD, imp=1 / r.AvgD,
                                 fair=inv[1] / inv.sum(), y=int(r.Is_Draw)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--test-from", default="2026-08-01")
    args = ap.parse_args()
    R = collect(pd.read_csv(args.csv, parse_dates=["Date"]), args.test_from)
    ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4)))
    n = len(R)
    print(f"{n} test games in {R["div"].nunique()} leagues from {args.test_from}; draw rate {R.y.mean():.3f}\n")
    print(f"{'draw probability':26}{'mean':>7}{'log-loss':>10}")
    print(f"{'flat base rate':26}{R.y.mean():7.3f}{ll(np.full(n, R.y.mean()), R.y):10.4f}")
    print(f"{'market, margin removed':26}{R.fair.mean():7.3f}{ll(R.fair, R.y):10.4f}")
    for k, lab in (("dc", "Dixon-Coles"), ("xgb", "calibrated booster"), ("blend", "BLEND (the scanner)")):
        print(f"{lab:26}{R[k].mean():7.3f}{ll(R[k], R.y):10.4f}")
    print(f"\nPick rule: edge = blend - 1/odds >= threshold, draw odds <= {MAX_ODDS}")
    print(f"{'edge >=':>8}{'picks':>7}{'hit':>7}{'95% range':>11}{'break-even':>11}{'ROI':>8}")
    ok = R[R.odds <= MAX_ODDS]
    print(f"{'all':>8}{len(ok):7d}{ok.y.mean():7.3f}{'':>11}{(1 / ok.odds).mean():11.3f}{(ok.y * ok.odds - 1).mean():+8.3f}   (every draw at <= {MAX_ODDS}, no model)")
    for t in (0.02, MIN_EDGE, 0.06, 0.08):
        p = ok[(ok.blend - ok.imp) >= t]
        if len(p) < 5:
            print(f"{t:8.2f}{len(p):7d}   (too few)")
            continue
        lo, hi = wilson(int(p.y.sum()), len(p))
        print(f"{t:8.2f}{len(p):7d}{p.y.mean():7.3f}{lo:6.2f}-{hi:.2f}{(1 / p.odds).mean():11.3f}{(p.y * p.odds - 1).mean():+8.3f}" + ("   <- the script's rule" if t == MIN_EDGE else ""))
    print("\nby league (script's rule):")
    for div, g in ok[(ok.blend - ok.imp) >= MIN_EDGE].groupby("div"):
        print(f"  {LEAGUES[div]:22}{len(g):4d} picks  hit {g.y.mean():.3f}  ROI {(g.y * g.odds - 1).mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
