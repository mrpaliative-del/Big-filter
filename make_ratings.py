#!/usr/bin/env python3
"""Build ratings.json from one or more results CSVs (files or folders).

    python make_ratings.py england.csv spain.csv -o ratings.json --since 2022-07-01
    python make_ratings.py data/ --check paste.txt --aliases aliases.json

CSV columns (case-insensitive): home/away team and goals (home_goals, FTHG,
hgoal ... or a combined "2-1" score column), plus a date column for recency
weighting. Works with football-data, engsoccerdata and footballcsv layouts.
"""
import argparse
import difflib
import json
import sys
from datetime import date
from pathlib import Path

from scanner.fit import fit_ratings, read_results
from scanner.model import Ratings, _norm
from scanner.parser import parse_fixtures


def collect(paths):
    for p in map(Path, paths):
        yield from (sorted(p.rglob("*.csv")) if p.is_dir() else [p])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="CSV files and/or folders")
    ap.add_argument("-o", "--out", default="ratings.json")
    ap.add_argument("--since", help="ignore games before YYYY-MM-DD (also drops rows without a date)")
    ap.add_argument("--as-of", help="reference date for recency weights (default: newest game in the data)")
    ap.add_argument("--min-games", type=float, default=8.0, help="min effective games per team (default 8)")
    ap.add_argument("--reg", type=float, default=8.0, help="shrinkage toward average (default 8.0; best out-of-sample on two datasets)")
    ap.add_argument("--half-life", type=float, default=365.0, help="recency half-life in days (default 365; 0 = off)")
    ap.add_argument("--aliases", help='JSON {"dataset name": "Flashscore name"} to rename teams')
    ap.add_argument("--check", metavar="PASTE.txt", help="report which teams in a Flashscore paste have ratings")
    args = ap.parse_args()

    since = date.fromisoformat(args.since) if args.since else None
    matches = []
    for f in collect(args.inputs):
        try:
            matches += read_results(f, since=since)
        except ValueError as exc:
            print(f"skip {f.name}: {exc}", file=sys.stderr)
    if not matches:
        print("No usable matches found.", file=sys.stderr)
        return 1

    dated = [m["date"] for m in matches if m["date"]]
    as_of = date.fromisoformat(args.as_of) if args.as_of else (max(dated) if dated else None)
    table = fit_ratings(matches, reg=args.reg, min_games=args.min_games,
                        half_life_days=args.half_life or None, today=as_of)
    if args.aliases:
        for src, dst in json.loads(Path(args.aliases).read_text(encoding="utf-8")).items():
            if src in table:
                table[dst] = table[src]

    note = f"Fitted from {len(matches)} matches up to {max(dated) if dated else '?'}. 1.0 = average; defence > 1 = leaky."
    payload = {"_comment": note, **dict(sorted(table.items()))}
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(matches)} matches -> {len(table)} rated teams -> {args.out}")

    if args.check:
        ratings = Ratings.from_json(args.out)
        fixtures = parse_fixtures(Path(args.check).read_text(encoding="utf-8"))["fixtures"]
        teams = sorted({t for f in fixtures for t in (f["home"], f["away"])})
        missing = [t for t in teams if ratings.get(t) is None]
        print(f"Paste coverage: {len(teams) - len(missing)}/{len(teams)} teams rated")
        known = {_norm(k): k for k in table}
        for t in missing[:40]:
            close = difflib.get_close_matches(_norm(t), list(known), n=1, cutoff=0.8)
            if close:
                print(f"  {t!r} -> maybe {known[close[0]]!r}  (add to --aliases if right)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
