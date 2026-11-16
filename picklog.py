#!/usr/bin/env python3
"""Log scanner picks, settle them against results, and report how the probabilities really did.

    python picklog.py settle results.csv          # fill in scores for logged picks (any results CSV fit.py reads)
    python picklog.py settle --score "Arsenal|Chelsea|2-1"   # or enter one by hand (repeatable)
    python picklog.py report [--min-status all|qualified]

Picks are appended by the app when "Save picks" / log_picks is on (default file: picks.csv, or $PICKS_PATH).
Note: on hosts with an ephemeral disk (e.g. free Render) picks.csv is lost on restart - download it first.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from scanner.fit import read_results
from scanner.model import _norm

DEFAULT_PATH = "picks.csv"
FIELDS = ["scan_date", "time", "competition", "home", "away", "source", "xg", "p_1x", "p_h15", "p_u25",
          "min_leg", "all3", "status", "odds", "hg", "ag", "hit_1x", "hit_h15", "hit_u25", "hit_all3"]


def _num(pct: str) -> float:
    return float(str(pct).rstrip("%")) / 100


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def append_picks(rows: list[dict], path: str | Path = DEFAULT_PATH, scan_date: date | None = None) -> int:
    """Append scanner rows (``run_pipeline()['ranked_fixtures']``). Skips ones already logged for that date."""
    path, scan_date = Path(path), scan_date or date.today()
    existing = _read(path)
    seen = {(r["scan_date"], _norm(r["home"]), _norm(r["away"])) for r in existing}
    added = 0
    for r in rows:
        key = (scan_date.isoformat(), _norm(r["home"]), _norm(r["away"]))
        if key in seen:
            continue
        seen.add(key)
        existing.append({
            "scan_date": scan_date.isoformat(), "time": r["time"], "competition": r["competition"],
            "home": r["home"], "away": r["away"], "source": r["source"], "xg": r["xg"],
            "p_1x": _num(r["prob_1x"]), "p_h15": _num(r["prob_h_plus_1_5"]), "p_u25": _num(r["prob_u2_5"]),
            "min_leg": _num(r["min_leg"]), "all3": _num(r["all_three"]), "status": r["status"],
            "odds": r.get("odds", ""), "hg": "", "ag": "", "hit_1x": "", "hit_h15": "", "hit_u25": "", "hit_all3": "",
        })
        added += 1
    if added:
        _write(path, [{k: r.get(k, "") for k in FIELDS} for r in existing])
    return added


def _apply_score(row: dict, hg: int, ag: int) -> None:
    row.update(hg=hg, ag=ag, hit_1x=int(hg >= ag), hit_h15=int(ag - hg <= 1), hit_u25=int(hg + ag <= 2),
               hit_all3=int(hg >= ag and hg + ag <= 2))


def settle(path: str | Path, results: list[dict], window_days: tuple[int, int] = (-1, 3)) -> int:
    """Fill scores for unsettled picks from ``results`` (dicts with home/away/hg/ag/date)."""
    path = Path(path)
    rows = _read(path)
    index: dict[tuple[str, str], list[dict]] = {}
    for m in results:
        index.setdefault((_norm(m["home"]), _norm(m["away"])), []).append(m)
    done = 0
    for r in rows:
        if r["hg"] != "":
            continue
        d0 = date.fromisoformat(r["scan_date"])
        for m in index.get((_norm(r["home"]), _norm(r["away"])), []):
            d = m.get("date")
            if d is None or d0 + timedelta(days=window_days[0]) <= d <= d0 + timedelta(days=window_days[1]):
                _apply_score(r, m["hg"], m["ag"])
                done += 1
                break
    if done:
        _write(path, rows)
    return done


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def report(path: str | Path = DEFAULT_PATH, only_qualified: bool = False) -> str:
    rows = [r for r in _read(Path(path)) if r["hg"] != ""]
    if only_qualified:
        rows = [r for r in rows if r["status"].startswith("★")]
    if not rows:
        return "No settled picks yet."
    out = [f"{len(rows)} settled picks" + (" (qualified only)" if only_qualified else "")]
    groups = [("all", rows)] + [(f"source={s}", [r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})]
    for name, grp in groups:
        if not grp:
            continue
        out.append(f"\n[{name}]  n={len(grp)}")
        out.append(f"{'leg':7}{'predicted':>10}{'actual':>8}{'95% range':>15}")
        for label, p_key, h_key in (("1X", "p_1x", "hit_1x"), ("H+1.5", "p_h15", "hit_h15"),
                                    ("U2.5", "p_u25", "hit_u25"), ("All3", "all3", "hit_all3")):
            hits = sum(int(r[h_key]) for r in grp)
            pred = sum(float(r[p_key]) for r in grp) / len(grp)
            lo, hi = wilson(hits, len(grp))
            out.append(f"{label:7}{pred:10.1%}{hits / len(grp):8.1%}{lo:8.0%}-{hi:.0%}")
        if len(grp) < 100:
            out.append("  (fewer than 100 picks: the range is wide, do not tune on this yet)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("settle")
    st.add_argument("results_csv", nargs="?")
    st.add_argument("--score", action="append", default=[], help='"Home|Away|2-1" (repeatable)')
    st.add_argument("--picks", default=os.getenv("PICKS_PATH", DEFAULT_PATH))
    rp = sub.add_parser("report")
    rp.add_argument("--picks", default=os.getenv("PICKS_PATH", DEFAULT_PATH))
    rp.add_argument("--only-qualified", action="store_true")
    args = ap.parse_args()

    if args.cmd == "settle":
        results = read_results(args.results_csv) if args.results_csv else []
        for item in args.score:
            home, away, sc = item.split("|")
            m = re.match(r"\s*(\d+)\s*-\s*(\d+)", sc)
            if not m:
                print(f"bad score {item!r}", file=sys.stderr)
                return 1
            results.append({"home": home, "away": away, "hg": int(m.group(1)), "ag": int(m.group(2)), "date": None})
        print(f"settled {settle(args.picks, results)} picks")
    else:
        print(report(args.picks, args.only_qualified))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
