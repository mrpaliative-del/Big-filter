#!/usr/bin/env python3
"""Arbitrage calculator for 2-way and 3-way markets from several bookmakers' prices.

    python arbitrage.py quotes.csv [--stake 100]

quotes.csv: match,bookmaker,<outcome1>,<outcome2>[,<outcome3>]   e.g.  match,bookmaker,home,draw,away
For each match it takes the best price per outcome across bookmakers. If the inverse prices sum to less than 1
you can back every outcome and lock a profit regardless of the result; it prints the stake split.
Real-world limits: stale or erroneous quotes get voided, stakes are capped, and prices move while you place bets.
"""
import argparse
import csv
from collections import defaultdict


def find_arbitrage(prices: dict[str, list[tuple[str, float]]], total_stake: float = 100.0) -> dict:
    """prices: outcome -> [(bookmaker, decimal_price), ...]. Returns best prices, overround and stake split."""
    best = {o: max(q, key=lambda bp: bp[1]) for o, q in prices.items() if q}
    book = sum(1 / p for _, p in best.values())
    out = {"best": {o: {"bookmaker": b, "price": p} for o, (b, p) in best.items()}, "overround": book,
           "arbitrage": book < 1, "profit_pct": (1 / book - 1) * 100}
    stakes = {o: total_stake * (1 / p) / book for o, (_, p) in best.items()}   # equal payout whichever outcome lands
    out["stakes"] = stakes
    out["payout"] = {o: stakes[o] * best[o][1] for o in stakes}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--stake", type=float, default=100.0)
    args = ap.parse_args()
    games = defaultdict(lambda: defaultdict(list))
    with open(args.csv, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        outcomes = [c for c in rdr.fieldnames if c not in ("match", "bookmaker")]
        for r in rdr:
            for o in outcomes:
                if r[o]:
                    games[r["match"]][o].append((r["bookmaker"], float(r[o])))
    found = 0
    for match, prices in games.items():
        res = find_arbitrage(prices, args.stake)
        tag = f"ARBITRAGE +{res['profit_pct']:.2f}%" if res["arbitrage"] else f"no arbitrage (overround {res['overround']:.3f})"
        print(f"{match}: {tag}")
        if res["arbitrage"]:
            found += 1
            for o, s in res["stakes"].items():
                b = res["best"][o]
                print(f"   {o:6} {s:8.2f} @ {b['price']:.2f} ({b['bookmaker']})  -> returns {res['payout'][o]:.2f}")
    print(f"\n{found} arbitrage(s) in {len(games)} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
