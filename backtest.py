#!/usr/bin/env python3
"""Out-of-sample check of the ratings + scanner probabilities.

    python backtest.py league1.csv league2.csv ... [--train-from 2021-07-01] [--split 2024-07-01]

Fits ratings on games before --split, then scores games on/after it that the
model has never seen, for the three markets the scanner uses
(1X, Home +1.5, Under 2.5), against the flat league-average baseline.
"""
import argparse
import math
from datetime import date, timedelta

import numpy as np

from scanner.calibration import MARKETS as CAL_KEYS, Calibration, fit_isotonic
from scanner.fit import fit_ratings, read_results
from scanner.model import Rating, evaluate_triple_markets, expected_goals

MARKETS = ("1X", "H+1.5", "U2.5")


def outcomes(hg: int, ag: int) -> tuple[int, int, int]:
    return int(hg >= ag), int(ag - hg <= 1), int(hg + ag <= 2)


def probs(lh: float, la: float) -> tuple[float, float, float]:
    m = evaluate_triple_markets(lh, la)
    return m["1x_prob"] / 100, m["home_plus_1_5_prob"] / 100, m["under_2_5_prob"] / 100


def poisson_ll(g: int, lam: float) -> float:
    return g * math.log(lam) - lam - math.lgamma(g + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--train-from", default="2021-07-01")
    ap.add_argument("--split", default="2024-07-01")
    ap.add_argument("--half-life", type=float, default=365.0)
    ap.add_argument("--min-games", type=float, default=8.0)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.695)
    ap.add_argument("--cal-seasons", type=int, default=0, help="learn per-leg calibration from this many seasons before --split (pooled, walk-forward)")
    ap.add_argument("--write-calibration", metavar="PATH", help="save the learned calibration (implies --cal-seasons 3 if unset)")
    ap.add_argument("--target", type=float, help="realised hit-rate target (e.g. 0.695): walk-forward report of which markets can reach it")
    args = ap.parse_args()

    start, split = date.fromisoformat(args.train_from), date.fromisoformat(args.split)
    allm = []
    for path in args.csvs:
        allm += read_results(path, since=start)
    train = [m for m in allm if m["date"] < split]
    test = [m for m in allm if m["date"] >= split]
    print(f"train games: {len(train)}  test games: {len(test)}")

    fit = fit_ratings(train, reg=args.reg, min_games=args.min_games,
                      half_life_days=args.half_life, today=split - timedelta(days=1))
    print(f"rated teams: {len(fit)}")

    rows = []
    for m in test:
        rh, ra = fit.get(m["home"]), fit.get(m["away"])
        if not rh or not ra:
            continue
        lam_m = expected_goals(Rating(**rh), Rating(**ra))
        lam_f = expected_goals(None, None)
        rows.append((probs(*lam_m), probs(*lam_f), outcomes(m["hg"], m["ag"]),
                     poisson_ll(m["hg"], lam_m[0]) + poisson_ll(m["ag"], lam_m[1]),
                     poisson_ll(m["hg"], lam_f[0]) + poisson_ll(m["ag"], lam_f[1]),
                     m["hg"] + m["ag"],
                     evaluate_triple_markets(*lam_m)["all_three_prob"]))
    n = len(rows)
    if not n:
        print("No test games with both teams rated.")
        if args.write_calibration and not args.cal_seasons:
            args.cal_seasons = 3
        if args.cal_seasons:
            calibration_report(allm, split, args)
            return 0
        return 1
    print(f"scored games (both teams rated): {n}\n")

    P = np.array([r[0] for r in rows]); F = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows])
    print(f"Goal log-likelihood per game:  ratings {np.mean([r[3] for r in rows]):.3f}   flat {np.mean([r[4] for r in rows]):.3f}")
    print(f"\n{'market':6} {'actual':>7} {'ratings pred':>13} {'flat pred':>10} {'Brier ratings':>14} {'Brier flat':>11}")
    for i, name in enumerate(MARKETS):
        print(f"{name:6} {Y[:, i].mean():7.3f} {P[:, i].mean():13.3f} {F[:, i].mean():10.3f} "
              f"{np.mean((P[:, i] - Y[:, i]) ** 2):14.4f} {np.mean((F[:, i] - Y[:, i]) ** 2):11.4f}")

    score = P.min(axis=1)
    all3 = Y.all(axis=1)
    A3 = np.array([r[6] for r in rows])
    print(f"\nScanner ranking = min of the three probabilities.  All three legs hit overall: {all3.mean():.3f}")
    print(f"{'min_prob band':16} {'games':>6} {'avg min_prob':>13} {'all 3 hit':>10} {'weakest leg hit':>16}")
    for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, args.threshold), (args.threshold, 1.01)):
        sel = (score >= lo) & (score < hi)
        if sel.sum() < 5:
            continue
        weakest = Y[sel][np.arange(sel.sum()), P[sel].argmin(axis=1)].mean()
        label = f"{lo:.2f}-{min(hi, 1):.2f}" + ("  QUALIFIED" if lo == args.threshold else "")
        print(f"{label:16} {sel.sum():6d} {score[sel].mean():13.3f} {all3[sel].mean():10.3f} {weakest:16.3f}")

    print(f"\nCommon line: a game qualifies when EVERY leg's predicted probability is >= the line")
    print(f"{'line':>6} {'games':>6} {'1X hit':>7} {'H+1.5':>7} {'U2.5':>7} {'all 3':>7}")
    for line in (0.60, 0.65, args.threshold, 0.72):
        sel = score >= line
        if sel.sum() >= 5:
            h = Y[sel].mean(axis=0)
            print(f"{line:6.3f} {sel.sum():6d} {h[0]:7.3f} {h[1]:7.3f} {h[2]:7.3f} {all3[sel].mean():7.3f}")
    print(f"\nAll3 = P(all three legs land) - the scanner's default ranking.  Mean predicted {A3.mean():.3f}, actual {all3.mean():.3f}")
    print(f"{'All3 band':14} {'games':>6} {'predicted':>10} {'actual':>8}")
    for lo, hi in ((0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)):
        sel = (A3 >= lo) & (A3 < hi)
        if sel.sum() >= 5:
            print(f"{lo:.1f}-{min(hi, 1):.1f}" + (" QUALIFIED" if lo == .5 else "").ljust(9) + f"{sel.sum():6d} {A3[sel].mean():10.3f} {all3[sel].mean():8.3f}")

    if args.target:
        target_report(allm, split, args)
    if args.write_calibration and not args.cal_seasons:
        args.cal_seasons = 3
    if args.cal_seasons:
        calibration_report(allm, split, args)
    return 0


def _season_probs(allm, train_end, eval_start, eval_end, args):
    train = [m for m in allm if m["date"] < train_end]
    fit = fit_ratings(train, reg=args.reg, min_games=args.min_games,
                      half_life_days=args.half_life, today=train_end - timedelta(days=1))
    P, Y = [], []
    for m in allm:
        if not (eval_start <= m["date"] < eval_end):
            continue
        a, b = fit.get(m["home"]), fit.get(m["away"])
        if not a or not b:
            continue
        mk = evaluate_triple_markets(*expected_goals(Rating(**a), Rating(**b)))
        P.append((mk["1x_prob"] / 100, mk["home_plus_1_5_prob"] / 100, mk["under_2_5_prob"] / 100, mk["all_three_prob"]))
        Y.append(outcomes(m["hg"], m["ag"]) + (int(m["hg"] >= m["ag"] and m["hg"] + m["ag"] <= 2),))
    return np.array(P), np.array(Y, float)


def target_report(allm, split, args):
    """Walk-forward: choose cutoffs on the year BEFORE --split, confirm on games from --split on."""
    val_start = date(split.year - 1, split.month, split.day)
    Pv, Yv = _season_probs(allm, val_start, val_start, split, args)
    Pt, Yt = _season_probs(allm, split, split, date(2100, 1, 1), args)
    T = args.target
    print(f"\n=== Target {T:.1%} realised hit rate  (validation {val_start}..{split}, n={len(Pv)}; test n={len(Pt)}) ===")
    print(f"{'market':7}{'cutoff':>8}{'val n':>7}{'val hit':>9}{'test n':>8}{'test hit':>9}{'share of games':>16}")
    for i, name in enumerate(("1X", "H+1.5", "U2.5", "All3")):
        # lowest cutoff whose validation hit rate clears target + 1pt margin, with >= 100 games
        cut = next((c for c in np.arange(0.30, 0.96, 0.01)
                    if (Pv[:, i] >= c).sum() >= 100 and Yv[Pv[:, i] >= c, i].mean() >= T + 0.01), None)
        if cut is None:
            best = Yt[:, i][Pt[:, i] >= np.sort(Pt[:, i])[-100]].mean()
            print(f"{name:7}  UNREACHABLE: no cutoff gets >= {T:.1%} on validation (top 100 test games hit {best:.1%})")
            continue
        sv, st = Pv[:, i] >= cut, Pt[:, i] >= cut
        print(f"{name:7}{cut:8.2f}{sv.sum():7d}{Yv[sv, i].mean():9.3f}{st.sum():8d}{Yt[st, i].mean():9.3f}{st.mean():16.1%}")
    print("Note: a cutoff of ~0.5-0.7 for 1X / H+1.5 just reflects that those legs already land ~70% / ~85% of the time;")
    print("use a higher predicted level (see the By-level table in the README) if you want selective picks.")


def calibration_report(allm, split, args):
    """Pool the ``cal_seasons`` seasons before ``split`` (each predicted with ratings fitted only on
    earlier games), learn a monotone map per leg, then sweep a common line on the unseen games >= split."""
    Pv, Yv = [], []
    for k in range(args.cal_seasons, 0, -1):
        a, b = date(split.year - k, split.month, split.day), date(split.year - k + 1, split.month, split.day)
        P, Y = _season_probs(allm, a, a, b, args)
        if len(P):
            Pv.append(P)
            Yv.append(Y)
    Pv, Yv = np.vstack(Pv), np.vstack(Yv)
    cal = Calibration({k: fit_isotonic(Pv[:, i], Yv[:, i], n_bins=20) for i, k in enumerate(CAL_KEYS)},
                      note=f"Isotonic leg calibration pooled over {args.cal_seasons} seasons before {split} ({len(Pv)} games). "
                           "Learned from rating-based predictions.")
    print(f"\n=== Calibration learned from {len(Pv)} games ({args.cal_seasons} seasons before {split}) ===")
    if args.write_calibration:
        cal.save(args.write_calibration)
        print(f"saved -> {args.write_calibration}")
    Pt, Yt = _season_probs(allm, split, split, date(2100, 1, 1), args)
    if not len(Pt):
        print("(no games on/after the split to test on)")
        return
    C = np.column_stack([[cal.apply(k, p) for p in Pt[:, i]] for i, k in enumerate(CAL_KEYS)])
    print(f"Unseen games from {split}: {len(Pt)}.  Common line on CALIBRATED legs (qualify = every leg >= line)")
    print(f"{'line':>6}{'games':>7}{'share':>7}{'1X hit':>8}{'H+1.5':>8}{'U2.5':>8}{'all 3':>7}")
    for line in (0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60):
        sel = C[:, :3].min(axis=1) >= line
        if sel.sum() >= 5:
            h = Yt[sel, :3].mean(axis=0)
            print(f"{line:6.3f}{sel.sum():7d}{sel.mean():7.1%}{h[0]:8.3f}{h[1]:8.3f}{h[2]:8.3f}{Yt[sel, 3].mean():7.3f}")
        else:
            print(f"{line:6.3f}{sel.sum():7d}  (too few)")
    print("Calibrated All3 (predicted -> actual):")
    for lo in (0.30, 0.35, 0.40, 0.45):
        sel = C[:, 3] >= lo
        if sel.sum() >= 5:
            print(f"  >= {lo:.2f}: n={sel.sum():4d} predicted {C[sel, 3].mean():.3f} actual {Yt[sel, 3].mean():.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
