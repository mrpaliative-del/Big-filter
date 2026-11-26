#!/usr/bin/env python3
"""What the scanner rules deliver on unseen games (statistics-only selection, prices from the market).

    python rules_report.py data/fd_2025-26_2026-27.csv --split 2026-07-01

Trains the form model and the draw-driver rating on games BEFORE the split, then applies the three scanners' rules to the games after it:
Scanner 1 (1X, Home +1.5, Under 2.5) and Scanner 2 (1X, 12, Over 1.5): All3 strictly above the line AND every leg priced >= the floor.
Scanner 3 (draw): draw rating above the line (watchlist at 30%+).
"""
import argparse

import numpy as np
import pandas as pd

from scanner.drawmodel import DRIVER_COLS, driver_matrix, train_driver
from scanner.features import build_features
from scanner.formmodel import outcomes, train
from scanner.model import evaluate_profile, lambdas_from_1x2

PRICE_MARGIN = 0.06


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--split", default="2026-07-01")
    ap.add_argument("--line", type=float, default=0.20)
    ap.add_argument("--floor", type=float, default=1.25)
    ap.add_argument("--draw-line", type=float, default=0.40)
    args = ap.parse_args()
    raw = pd.read_csv(args.csv, parse_dates=["Date"]).dropna(subset=["AvgH", "AvgD", "AvgA"]).reset_index(drop=True)
    g, F, _ = build_features(raw)
    sp = pd.Timestamp(args.split)
    tr, te = (g.Date < sp).values, (g.Date >= sp).values
    fm = train(g, F, tr)
    P = pd.DataFrame([fm.predict(F.iloc[i].to_dict()) for i in np.where(te)[0]])
    Y = pd.DataFrame([outcomes(int(r.FTHG), int(r.FTAG)) for r in g[te].itertuples()]).astype(float)
    pr = {"classic": [], "regular": []}
    for r in g[te].itertuples():
        lam = lambdas_from_1x2((r.AvgH, r.AvgD, r.AvgA))
        for prof in pr:
            legs = evaluate_profile(*lam, prof)["legs"]
            est = {k: max((1 / v) / (1 + PRICE_MARGIN), 1.01) for k, v in legs.items()}
            est["1x"] = 1 / (1 / r.AvgH + 1 / r.AvgD)
            if "12" in est:
                est["12"] = 1 / (1 / r.AvgH + 1 / r.AvgA)
            pr[prof].append(est)
    n = te.sum()
    print(f"train {tr.sum()} games, test {n} games from {args.split} (statistics-only selection, prices from the market)\n")
    print(f"{'scanner':34}{'rule':>34}{'games':>7}{'share':>7}{'All3 predicted':>15}{'actual':>8}")
    for name, key, keys, prof in (("1: 1X, Home +1.5, Under 2.5", "all3c", ["1x", "h15", "u25"], "classic"), ("2: 1X, 12, Over 1.5", "all3", ["1x", "12", "o15"], "regular")):
        pe = pd.DataFrame(pr[prof])
        ok_line = (P[key] > args.line).values
        ok_price = (pe[keys] >= args.floor).all(axis=1).values
        for lab, m in ((f"All3 > {args.line:.0%} only", ok_line), (f"+ every leg priced >= {args.floor:.2f} (the rule)", ok_line & ok_price), (f"  of which All3 >= 50%", ok_line & ok_price & (P[key] >= .5).values)):
            if m.sum():
                print(f"{name:34}{lab:>34}{m.sum():7d}{m.mean():7.1%}{P[key][m].mean():15.3f}{Y[key][m].mean():8.3f}")
            else:
                print(f"{name:34}{lab:>34}{0:7d}")
    dm = train_driver(g, F, tr)
    X = driver_matrix(F, g)
    p = 1 / (1 + np.exp(-(((X[DRIVER_COLS].values - dm.mean) / dm.scale) @ dm.w + dm.b)))[te] if False else (1 / (1 + np.exp(-(((X[DRIVER_COLS].values - dm.mean) / dm.scale) @ dm.w + dm.b))))[te]
    y = (g.FTHG == g.FTAG).values[te].astype(float)
    print(f"\nScanner 3 (draw rating): range {p.min():.3f}-{p.max():.3f}; draw rate overall {y.mean():.3f}")
    for lab, m in ((f"rating above {args.draw_line:.0%} (the rule)", p > args.draw_line), ("watchlist 30%+", p >= 0.30), ("watchlist 33%+", p >= 0.33)):
        print(f"  {lab:32}{m.sum():6d} games" + (f", predicted {p[m].mean():.3f}, actual draw rate {y[m].mean():.3f}" if m.sum() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
