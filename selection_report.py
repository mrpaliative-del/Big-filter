#!/usr/bin/env python3
"""Statistics first, price second: which common market selection is best, and what does a 1.30-1.35 price window deliver?

    python selection_report.py data/fd_2025-26_2026-27.csv --split 2026-07-01
    python selection_report.py data/fd_2025-26_2026-27.csv --write-calibration calibration_select.json

Step 1  how predictable is each common market (model log-loss gain over the base rate)?
Step 2  results of selections whose price is inside the window (train and test seasons).
Step 3  the engine: per game the in-window selection with the best calibrated edge; calibration learned BEFORE --split.
"""
import argparse

import numpy as np
import pandas as pd

from scanner.calibration import Calibration, apply_map, fit_isotonic
from scanner.model import dixon_coles_matrix, lambdas_from_1x2

x, y = np.indices((10, 10))
s = x + y
MK = {
    "home": ("Home win", x > y, lambda r: r.AvgH, lambda h, a: h > a),
    "away": ("Away win", y > x, lambda r: r.AvgA, lambda h, a: a > h),
    "1x": ("1X", x >= y, lambda r: 1 / (1 / r.AvgH + 1 / r.AvgD), lambda h, a: h >= a),
    "x2": ("X2", y >= x, lambda r: 1 / (1 / r.AvgD + 1 / r.AvgA), lambda h, a: a >= h),
    "12": ("12", x != y, lambda r: 1 / (1 / r.AvgH + 1 / r.AvgA), lambda h, a: h != a),
    "o25": ("Over 2.5", s >= 3, lambda r: r.AvgO25, lambda h, a: h + a >= 3),
    "u25": ("Under 2.5", s <= 2, lambda r: r.AvgU25, lambda h, a: h + a <= 2),
}
ll = lambda p, yv: -np.mean(yv * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - yv) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4)))


def build(df):
    rows = []
    for r in df.itertuples():
        lam = lambdas_from_1x2((r.AvgH, r.AvgD, r.AvgA))
        if not lam:
            continue
        m = dixon_coles_matrix(*lam)
        for k, (_, mask, pf, of) in MK.items():
            rows.append((r.Date, r.Index, k, float(m[mask].sum()), float(pf(r)), float(of(int(r.FTHG), int(r.FTAG)))))
    return pd.DataFrame(rows, columns=["date", "g", "market", "p", "price", "hit"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--split", default="2026-07-01")
    ap.add_argument("--lo", type=float, default=1.30)
    ap.add_argument("--hi", type=float, default=1.35)
    ap.add_argument("--write-calibration")
    args = ap.parse_args()
    df = pd.read_csv(args.csv, parse_dates=["Date"]).dropna(subset=["AvgH", "AvgD", "AvgA", "AvgO25", "AvgU25"]).reset_index(drop=True)
    R = build(df)
    if args.write_calibration:
        cal = Calibration({f"sel_{k}": fit_isotonic(g.p.values, g.hit.values, n_bins=15) for k, g in R.groupby("market") if k in ("home", "away", "1x", "x2", "12")},
                          note=f"Isotonic calibration of result-market probabilities from {len(df)} games.")
        cal.save(args.write_calibration)
        print(f"saved {args.write_calibration}")
    sp = pd.Timestamp(args.split)
    tr, te = R[R.date < sp], R[R.date >= sp]
    print(f"train {tr.g.nunique()} games, test {te.g.nunique()} games (from {args.split})\n")
    print("STEP 1  predictability (test): model log-loss gain over the base rate")
    res = []
    for k, g in te.groupby("market"):
        base = tr[tr.market == k].hit.mean()
        res.append((MK[k][0], base, ll(np.full(len(g), base), g.hit.values) - ll(g.p.values, g.hit.values), g[g.p >= g.p.quantile(.75)].hit.mean()))
    print(f"   {'market':10}{'base rate':>10}{'gain':>9}{'top-quartile hit':>18}")
    for n, b, gn, t in sorted(res, key=lambda r: -r[2]):
        print(f"   {n:10}{b:10.3f}{gn:9.4f}{t:18.3f}")
    print(f"\nSTEP 2  selections priced {args.lo:.2f}-{args.hi:.2f}")
    print(f"   {'market':10}{'season':>7}{'n':>6}{'hit':>7}{'break-even':>11}{'ROI':>8}")
    for k in MK:
        for lab, D in (("train", tr), ("test", te)):
            b = D[(D.market == k) & D.price.between(args.lo, args.hi)]
            if len(b) >= 20:
                print(f"   {MK[k][0]:10}{lab:>7}{len(b):6d}{b.hit.mean():7.3f}{(1 / b.price).mean():11.3f}{(b.hit * b.price - 1).mean():+8.3f}")
    a = te[te.price.between(args.lo, args.hi)]
    print(f"   {'ALL':10}{'test':>7}{len(a):6d}{a.hit.mean():7.3f}{(1 / a.price).mean():11.3f}{(a.hit * a.price - 1).mean():+8.3f}")
    cal = {k: fit_isotonic(g.p.values, g.hit.values, n_bins=15) for k, g in tr.groupby("market")}
    te = te.assign(pc=[apply_map(cal[k], p) for k, p in zip(te.market, te.p)])
    te["ev"] = te.pc * te.price - 1
    inb = te[te.price.between(args.lo, args.hi) & te.market.isin(["home", "away", "1x", "x2", "12"])]
    pick = inb.loc[inb.groupby("g").ev.idxmax()]
    print("\nSTEP 3  engine: per game the in-window result market with the best calibrated edge")
    for thr in (-9, 0.0, 0.02, 0.05):
        q = pick[pick.ev >= thr]
        lab = "all picks" if thr == -9 else f"edge >= {thr:+.2f}"
        print(f"   {lab:14}{len(q):5d} picks" + (f"  hit {q.hit.mean():.3f}  break-even {(1 / q.price).mean():.3f}  ROI {(q.hit * q.price - 1).mean():+.3f}" if len(q) else "  (none)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
