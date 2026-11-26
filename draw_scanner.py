#!/usr/bin/env python3
"""The weekly draw scanner (your draw_predictor_pipeline.py flow, on the repo's faster port). Needs internet.

    python draw_scanner.py            # fixtures.csv + 3 seasons of the four leagues -> weekly_draw_picks_<date>.xlsx

The original script is kept unchanged in legacy/. The same model also runs inside the main scanner as the third scanner (DRAW).
"""
import sys
from datetime import datetime

import pandas as pd

from scanner.draw import LEAGUES, to_script_schema, weekly_scan
from scanner.feeds import FIXTURES_URL, fetch_league_history, fetch_text, parse_football_data_csv

SEASONS = ["2425", "2526", "2627"]


def main() -> int:
    print("Fetching upcoming fixtures from Football-Data.co.uk...")
    try:
        fx = pd.read_csv(FIXTURES_URL.replace("football-data", "www.football-data"), encoding="latin1")
    except Exception as exc:
        print(f"Failed to download fixtures.csv: {exc}")
        return 1
    histories = {}
    for div, name in LEAGUES.items():
        print(f"\nProcessing {name} ({div})...")
        histories[div] = to_script_schema(fetch_league_history(div, SEASONS))
    picks = weekly_scan(fx, histories)
    if picks.empty:
        print("\nScan complete: No +EV draw picks met your edge criteria this week.")
        return 0
    out = f"weekly_draw_picks_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    picks.to_excel(out, index=False)
    print(f"\nSUCCESS: Exported {len(picks)} value draw picks to {out}")
    print(picks[["League", "Date", "HomeTeam", "AwayTeam", "Blend_Prob_Pct", "Market_Odds", "Edge_Pct", "Kelly_Stake"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
