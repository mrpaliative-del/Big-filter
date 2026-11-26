#!/usr/bin/env python3
"""Train the draw-driver rating (pooled over all leagues) and report how it performs on unseen games.

    python train_draw_driver.py data/fd_2025-26_2026-27.csv --split 2026-07-01      # walk-forward report
    python train_draw_driver.py data/fd_2025-26_2026-27.csv -o draw_driver.json      # production fit on all games
"""
import argparse

import numpy as np
import pandas as pd

from scanner.drawmodel import DRIVER_COLS, driver_matrix, train_driver
from scanner.features import build_features


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--split")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    raw = pd.read_csv(args.csv, parse_dates=["Date"]).dropna(subset=["AvgH", "AvgD", "AvgA"]).reset_index(drop=True)
    g, F, _ = build_features(raw)
    if args.split:
        sp = pd.Timestamp(args.split)
        tr, te = (g.Date < sp).values, (g.Date >= sp).values
        model = train_driver(g, F, tr)
        X = driver_matrix(F, g)
        Z = (X[DRIVER_COLS].values - model.mean) / model.scale
        p = 1 / (1 + np.exp(-(Z @ model.w + model.b)))
        y = (g.FTHG == g.FTAG).values.astype(float)
        inv = np.column_stack([1 / g.AvgH, 1 / g.AvgD, 1 / g.AvgA])
        fair = (inv / inv.sum(1, keepdims=True))[:, 1]
        ll = lambda q, yy: -np.mean(yy * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - yy) * np.log(1 - np.clip(q, 1e-4, 1 - 1e-4)))
        pt, yt = p[te], y[te]
        print(f"train {tr.sum()} games, test {te.sum()} games from {args.split}; draw rate {yt.mean():.3f}")
        print(f"log-loss: flat {ll(np.full(len(yt), y[tr].mean()), yt):.4f} | market (margin removed) {ll(fair[te], yt):.4f} | draw-driver rating {ll(pt, yt):.4f}")
        print(f"rating range {pt.min():.3f}-{pt.max():.3f}; share at 30%+: {(pt >= .30).mean():.1%}, at 40%+: {(pt > .40).sum()} games")
        print("predicted level -> actual draw rate:")
        for lo, hi in ((0, .22), (.22, .25), (.25, .28), (.28, .30), (.30, .33), (.33, 1)):
            s = (pt >= lo) & (pt < hi)
            if s.sum() >= 5:
                print(f"  {lo:.2f}-{min(hi, 1):.2f}: {s.sum():4d} games, predicted {pt[s].mean():.3f}, actual {yt[s].mean():.3f}")
    if args.out:
        m = train_driver(g, F, note=f"draw-driver rating trained on {len(g)} games to {g.Date.max().date()}")
        m.save(args.out)
        print(f"saved {args.out}: {m.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
