#!/usr/bin/env python3
"""Train the draw scanner's league models for use inside the main scanner.

    python train_draw_models.py                         # from the local history CSV (data/fd_2025-26_2026-27.csv)
    python train_draw_models.py --web                   # download 3 seasons per league first (internet), like the original script

Writes draw_models.pkl (pickle: only load files you made yourself). Run again after new results or when you update the history.
"""
import argparse
import os

import pandas as pd

from scanner.draw import LEAGUES, DrawPredictor, to_script_schema, train_league
from scanner.feeds import fetch_league_history

SEASONS = ["2425", "2526", "2627"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=os.getenv("HISTORY_PATHS", "data/fd_2025-26_2026-27.csv").split(",")[0])
    ap.add_argument("--web", action="store_true")
    ap.add_argument("-o", "--out", default="draw_models.pkl")
    args = ap.parse_args()
    local = pd.read_csv(args.history, parse_dates=["Date"])
    models = {}
    for div, name in LEAGUES.items():
        raw = to_script_schema(fetch_league_history(div, SEASONS)) if args.web else to_script_schema(local[local.Div == div])
        lm = train_league(div, raw)
        if lm is None:
            print(f"{name}: not enough games ({len(raw)}), skipped")
            continue
        models[div] = lm
        print(f"{name}: {lm.n_games} games to {lm.trained_through.date()}, weights DC {lm.dc_w} / booster {lm.xgb_w} ({lm.booster})")
    DrawPredictor(models, local).save_models(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
