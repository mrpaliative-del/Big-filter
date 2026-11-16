"""Parse -> model -> filter -> rank."""
from __future__ import annotations

from .calibration import Calibration
from .model import Ratings, evaluate_triple_markets, expected_goals, lambdas_from_1x2
from .parser import parse_fixtures

DEFAULT_THRESHOLD = 0.695  # ONE common line every leg (1X, Home +1.5, Under 2.5) must reach
CALIBRATED_THRESHOLD = 0.55  # common line on CALIBRATED legs: weakest leg realised ~60% on unseen 2024-25
ALL3_THRESHOLD = 0.50      # default line when ranking by the joint chance (rank_by="all3")
CALIBRATED_ALL3_THRESHOLD = 0.45
DEFAULT_MIN_THRESHOLD = 0.20
DEFAULT_MAX_XG = 3.25


def _fmt_odds(odds) -> str:
    return "/".join(f"{o:g}" for o in odds) if odds else ""


def _pct(x: float) -> str:
    return f"{round(x * 100, 1)}%"


def run_pipeline(
    raw_text: str,
    threshold: float | None = None,
    min_threshold: float = DEFAULT_MIN_THRESHOLD,
    max_xg: float = DEFAULT_MAX_XG,
    ratings: Ratings | None = None,
    include_unrated: bool = False,
    rank_by: str = "min",
    calibration: Calibration | None = None,
    calibration_odds: Calibration | None = None,
    qualify_unvalidated: bool = False,
) -> dict:
    """Rank fixtures and tag the ones where every leg clears the same line.

    ``rank_by="min"`` (default): a fixture is ★ QUALIFIED when the weakest of its three
    legs (1X, Home +1.5, Under 2.5) is >= ``threshold`` - i.e. all three legs stand at
    that same percent or better. Ranked by that weakest leg, ties broken by All3.

    ``rank_by="all3"``: rank by the joint chance that all three land
    (= P(1X and Under 2.5), since Home +1.5 is implied by 1X).

    Expected goals come from the fixture's 1X2 odds, else team ratings, else - only if
    ``include_unrated`` - league-average constants (identical for every such fixture).

    Calibration is per source: ``calibration_odds`` for odds-based rows, ``calibration`` for
    ratings-based ones. Only odds-based rows can be ★ QUALIFIED: on 2,000+ unseen 2026/27 games the
    odds path held up, while ratings fitted on one season were no better than a flat guess for the
    three-leg combination. ``qualify_unvalidated=True`` lifts that restriction.
    """
    if rank_by not in ("min", "all3"):
        raise ValueError("rank_by must be 'min' or 'all3'")
    calibration = calibration or Calibration()
    calibration_odds = calibration_odds or Calibration()
    any_cal = bool(calibration or calibration_odds)
    if threshold is None:
        if rank_by == "min":
            threshold = CALIBRATED_THRESHOLD if any_cal else DEFAULT_THRESHOLD
        else:
            threshold = CALIBRATED_ALL3_THRESHOLD if any_cal else ALL3_THRESHOLD
    ratings = ratings or Ratings()
    parsed = parse_fixtures(raw_text)
    rows: list[dict] = []
    counts = {"odds": 0, "ratings": 0, "default": 0}
    skipped_no_data = 0

    for item in parsed["fixtures"]:
        lam, source = None, "default"
        if item.get("odds"):
            lam = lambdas_from_1x2(tuple(item["odds"]))
            if lam:
                source = "odds"
        if lam is None:
            h_r, a_r = ratings.get(item["home"]), ratings.get(item["away"])
            if h_r or a_r:
                lam, source = expected_goals(h_r, a_r), "ratings"
        if lam is None:
            if not include_unrated:
                skipped_no_data += 1
                continue
            lam = expected_goals(None, None)

        lam_h, lam_a = lam
        lam_total = lam_h + lam_a
        if lam_total > max_xg:
            continue

        m = evaluate_triple_markets(lam_h, lam_a)
        raw = {"1x": m["1x_prob"] / 100, "h15": m["home_plus_1_5_prob"] / 100,
               "u25": m["under_2_5_prob"] / 100, "all3": m["all_three_prob"]}
        cmap = calibration_odds if source == "odds" else calibration
        cal = {k: cmap.apply(k, v) for k, v in raw.items()}  # identity when no calibration loaded
        min_leg = min(cal["1x"], cal["h15"], cal["u25"])
        ranking_score = min_leg if rank_by == "min" else cal["all3"]
        if ranking_score < min_threshold:
            continue

        counts[source] += 1
        rows.append(
            {
                "time": item["time"],
                "competition": item["competition"],
                "home": item["home"],
                "away": item["away"],
                "match": f"{item['home']} vs {item['away']}",
                "xg": round(lam_total, 2),
                "score": ranking_score,
                "min_leg": _pct(min_leg),
                "all_three": _pct(cal["all3"]),
                "prob_1x": _pct(cal["1x"]),
                "prob_h_plus_1_5": _pct(cal["h15"]),
                "prob_u2_5": _pct(cal["u25"]),
                "raw_min_leg": _pct(m["min_prob"]),
                "status": "★ QUALIFIED" if ranking_score >= threshold and (source == "odds" or qualify_unvalidated) else "◆ MODERATE",
                "source": source,
                "odds": _fmt_odds(item.get("odds")),
                "_tie": m["all_three_prob"],
            }
        )

    rows.sort(key=lambda r: (r["score"], r["_tie"]), reverse=True)
    for r in rows:
        del r["_tie"]

    warnings: list[str] = []
    if skipped_no_data:
        warnings.append(
            f"{skipped_no_data} fixture(s) had no odds or team ratings and were skipped. "
            "Paste Flashscore's Odds page (Football » Odds) or add a ratings.json."
        )
    if counts["default"]:
        warnings.append(
            f"{counts['default']} fixture(s) use league-average inputs and share identical "
            "probabilities; their ranking is not meaningful."
        )

    if counts["ratings"] or counts["default"]:
        warnings.append(
            "Rows from team ratings (not odds) are never tagged QUALIFIED: on unseen 2026/27 games ratings "
            "fitted on one season were no better than a flat guess for the three-leg combination."
        )

    copyable = [
        f"{r['time']} | {r['competition']} | {r['match']} | xG:{r['xg']} | "
        f"1X:{r['prob_1x']} | H+1.5:{r['prob_h_plus_1_5']} | U2.5:{r['prob_u2_5']} | "
        f"MinLeg:{r['min_leg']} | All3:{r['all_three']} | {r['status']} | src:{r['source']}"
        for r in rows
    ]

    return {
        "rank_by": rank_by,
        "calibrated": any_cal,
        "threshold": threshold,
        "summary": {
            "total_leagues": parsed["leagues_count"],
            "total_fixtures": parsed["fixtures_count"],
            "ranked_count": len(rows),
            "qualified_count": sum(r["status"].startswith("★") for r in rows),
            "from_odds": counts["odds"],
            "from_ratings": counts["ratings"],
            "rated_fixtures": counts["odds"] + counts["ratings"],
            "skipped_no_data": skipped_no_data,
            "skipped_in_play": parsed["skipped_in_play"],
            "skipped_postponed": parsed["skipped_postponed"],
        },
        "warnings": warnings,
        "cleaned_schedule": parsed["cleaned_text"],
        "ranked_fixtures": rows,
        "copyable_block": "\n".join(copyable),
    }
