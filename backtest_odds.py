#!/usr/bin/env python3
"""Check (and calibrate) the odds-based path on real results.

    python backtest_odds.py data/fd_2025-26_2026-27.csv --split 2026-07-01
    python backtest_odds.py data/fd_2025-26_2026-27.csv --write-calibration calibration_odds.json

CSV columns: Date, Home, Away, FTHG, FTAG, AvgH, AvgD, AvgA (football-data.co.uk names).
Each game: 1X2 odds -> expected goals -> the three legs, compared with what happened.
--split learns the calibration on games BEFORE the date and judges it on games from that date on.
--write-calibration learns it from every game in the file and saves it.
"""
import argparse
import csv
from datetime import date

import numpy as np

from picklog import wilson
from scanner.calibration import MARKETS as KEYS, Calibration, fit_isotonic
from scanner.model import evaluate_triple_markets, lambdas_from_1x2

LEGS = ("1X", "H+1.5", "U2.5", "All3")


def load(path):
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        try:
            odds = tuple(float(r[k]) for k in ("AvgH", "AvgD", "AvgA"))
            hg, ag = int(float(r["FTHG"])), int(float(r["FTAG"]))
            d = date.fromisoformat(r["Date"][:10]) if r.get("Date") else None
        except (KeyError, ValueError):
            continue
        lam = lambdas_from_1x2(odds)
        if not lam:
            continue
        m = evaluate_triple_markets(*lam)
        out.append((d, np.array([m["1x_prob"], m["home_plus_1_5_prob"], m["under_2_5_prob"], m["all_three_prob"] * 100]) / 100,
                    np.array([hg >= ag, ag - hg <= 1, hg + ag <= 2, hg >= ag and hg + ag <= 2], float)))
    return out


def ll(P, y):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(P) + (1 - y) * np.log(1 - P))


def learn(rows):
    P, Y = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    return Calibration({k: fit_isotonic(P[:, i], Y[:, i], n_bins=20) for i, k in enumerate(KEYS)},
                       note=f"Isotonic calibration of ODDS-based leg probabilities, learned from {len(rows)} real games "
                            f"({min(r[0] for r in rows)} to {max(r[0] for r in rows)}).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--split", help="YYYY-MM-DD: learn before, test from this date")
    ap.add_argument("--write-calibration", metavar="PATH")
    ap.add_argument("--line", type=float, default=0.55)
    args = ap.parse_args()

    rows = load(args.csv)
    print(f"{len(rows)} games with usable odds")
    if args.write_calibration:
        cal = learn(rows)
        cal.save(args.write_calibration)
        print(f"saved -> {args.write_calibration}")
    if not args.split:
        return 0

    split = date.fromisoformat(args.split)
    train, test = [r for r in rows if r[0] < split], [r for r in rows if r[0] >= split]
    cal = learn(train)
    P, Y = np.array([r[1] for r in test]), np.array([r[2] for r in test])
    C = np.column_stack([[cal.apply(k, p) for p in P[:, i]] for i, k in enumerate(KEYS)])
    base = np.array([r[2] for r in train]).mean(axis=0)
    print(f"train {len(train)} games (before {split}), test {len(test)} games\n")
    print(f"{'test log-loss':18}" + "".join(f"{n:>9}" for n in LEGS))
    print(f"{'flat base rate':18}" + "".join(f"{ll(np.full(len(Y), base[i]), Y[:, i]):9.4f}" for i in range(4)))
    print(f"{'odds, raw':18}" + "".join(f"{ll(P[:, i], Y[:, i]):9.4f}" for i in range(4)))
    print(f"{'odds, calibrated':18}" + "".join(f"{ll(C[:, i], Y[:, i]):9.4f}" for i in range(4)))
    print("\npredicted (raw / calibrated) vs actual")
    for i, n in enumerate(LEGS):
        print(f"  {n:6} {P[:, i].mean():.3f} / {C[:, i].mean():.3f}  vs {Y[:, i].mean():.3f}")
    print(f"\nCommon line (every calibrated leg >= line)")
    print(f"{'line':>6}{'games':>7}{'share':>7}{'1X':>7}{'H+1.5':>7}{'U2.5':>7}{'all 3':>7}  U2.5 95% range")
    for line in (0.50, 0.52, args.line, 0.58, 0.60):
        sel = C[:, :3].min(axis=1) >= line
        if sel.sum() >= 5:
            h = Y[sel, :3].mean(axis=0)
            lo, hi = wilson(int(Y[sel, 2].sum()), int(sel.sum()))
            print(f"{line:6.2f}{sel.sum():7d}{sel.mean():7.1%}{h[0]:7.2f}{h[1]:7.2f}{h[2]:7.2f}{Y[sel, 3].mean():7.2f}  {lo:.2f}-{hi:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
