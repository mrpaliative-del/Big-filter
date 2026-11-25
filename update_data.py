#!/usr/bin/env python3
"""Refresh the open-feed data (needs internet).

    python update_data.py                      # latest results -> data/fd_2025-26_2026-27.csv, fixtures odds -> data/fixtures_cache.csv
    python update_data.py --retrain            # also refit form_model.json on the updated history

Sources: football-data.co.uk (free for personal use; credit it). Run it after each matchday, or on a schedule.
"""
import argparse
import json
import os

import pandas as pd

from scanner.feeds import FIXTURES_URL, fetch_text, update_history


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=os.getenv("HISTORY_PATHS", "data/fd_2025-26_2026-27.csv").split(",")[0])
    ap.add_argument("--fixtures-cache", default="data/fixtures_cache.csv")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    added = update_history(args.history)
    print(f"history: {added} new games -> {args.history}")
    text = fetch_text(FIXTURES_URL)
    open(args.fixtures_cache, "w", encoding="utf-8").write(text)
    print(f"fixtures: saved {len(text.splitlines()) - 1} rows -> {args.fixtures_cache}")
    if args.retrain and added:
        from scanner.features import build_features
        from scanner.formmodel import train
        games, F, _ = build_features(pd.read_csv(args.history, parse_dates=["Date"]))
        model = train(games, F, note=f"trained on {len(games)} games to {games.Date.max().date()}")
        model.save("form_model.json")
        print("retrained form_model.json:", model.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
