#!/usr/bin/env python3
"""Train the technical-analysis form model from results history.

    python train_form_model.py data/fd_2025-26_2026-27.csv --split 2026-07-01   # honest walk-forward report
    python train_form_model.py data/fd_2025-26_2026-27.csv -o form_model.json    # production fit on all games

The report trains only on games BEFORE --split and judges on games from it on, with no odds used anywhere.
"""
import argparse

import numpy as np
import pandas as pd

from scanner.features import build_features
from scanner.formmodel import TARGETS, outcomes, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--split")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    games, F, _ = build_features(pd.read_csv(args.csv, parse_dates=["Date"]))
    if args.split:
        sp = pd.Timestamp(args.split)
        tr, te = (games.Date < sp).values, (games.Date >= sp).values
        model = train(games, F, tr, note=f"walk-forward: trained before {args.split}")
        Y = pd.DataFrame([outcomes(int(r.FTHG), int(r.FTAG)) for r in games.itertuples()]).astype(float)
        P = pd.DataFrame([model.predict(F.iloc[i].to_dict()) for i in np.where(te)[0]])
        ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4)))
        print(f"train {tr.sum()} games before {args.split}, test {te.sum()} games after; no odds used")
        print(f"{'target':8}{'actual':>8}{'predicted':>10}{'flat':>9}{'model':>9}")
        for k in TARGETS:
            y = Y[k].values[te]; flat = Y[k].values[tr].mean()
            print(f"{k:8}{y.mean():8.3f}{P[k].mean():10.3f}{ll(np.full(len(y), flat), y):9.4f}{ll(P[k].values, y):9.4f}")
        for key, label in (("all3", "regular: Home win AND Over 1.5"), ("all3c", "classic: 1X AND Under 2.5")):
            y = Y[key].values[te]
            print(f"\nAll3 {label} - by predicted level (actual rate overall {y.mean():.3f}):")
            for lo in (0.20, 0.30, 0.40, 0.50, 0.60):
                s_ = P[key].values >= lo
                if s_.sum() >= 5:
                    print(f"  predicted >= {lo:.2f}: {s_.sum():4d} games ({s_.mean():.1%}), predicted {P[key].values[s_].mean():.3f}, actual {y[s_].mean():.3f}")
    if args.out:
        model = train(games, F, note=f"trained on {len(games)} games to {games.Date.max().date()}")
        model.save(args.out)
        print(f"saved {args.out}: {model.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
