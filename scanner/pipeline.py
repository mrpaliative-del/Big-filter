"""Parse -> model -> filter -> rank."""
from __future__ import annotations

import numpy as np

from .calibration import Calibration
from .model import PROFILES, Ratings, evaluate_profile, expected_goals, lambdas_from_1x2
from .parser import parse_fixtures

DEFAULT_THRESHOLD = 0.695        # classic profile, raw numbers, ranked by weakest leg
CALIBRATED_THRESHOLD = 0.55      # classic profile on calibrated legs
ALL3_THRESHOLD = 0.50            # default line for All3 (regular profile; classic all3 mode without calibration)
CALIBRATED_ALL3_THRESHOLD = 0.45  # classic all3 mode with calibration
REGULAR_THRESHOLD = 0.50         # regular profile: All3 = Home win AND Over 1.5; 50%+ games landed ~64% (odds) / ~59% (form)
DEFAULT_MIN_THRESHOLD = 0.20
DEFAULT_MAX_XG = 6.0             # lower it to ~3.25 for Under-based (classic) selections
DEFAULT_PRICE_BAND = (1.20, 1.50)  # every leg of the regular profile must be priced inside this band
DEFAULT_PRICE_FLOOR = 1.25       # qualify_by="all3_floor": every leg must be priced at least this
FORM_TRUSTED_PROFILES = {"regular"}  # the form model is validated for All3 of the regular legs only
DEFAULT_PRICE_MARGIN = 0.06      # assumed bookmaker margin when a leg price has to be estimated from probability


def _fmt_odds(odds) -> str:
    return "/".join(f"{o:g}" for o in odds) if odds else ""


def _pct(x: float) -> str:
    return f"{round(x * 100, 1)}%"


def leg_prices(profile: str, probs: dict, odds, margin: float = DEFAULT_PRICE_MARGIN) -> dict:
    """Estimated bookmaker price of each leg.

    1X and 12 follow the fixture's own 1X2 prices when pasted (double chance = 1 / (1/H + 1/D) and 1 / (1/H + 1/A)); any
    other leg - or any leg when there are no odds - is fair odds 1/p reduced by an assumed bookmaker margin."""
    est = {k: max((1.0 / p) / (1.0 + margin), 1.01) for k, p in probs.items()}      # no bookmaker price is below ~1.01
    if odds:
        h, d, a = odds
        if "1x" in est:
            est["1x"] = 1.0 / (1 / h + 1 / d)
        if "12" in est:
            est["12"] = 1.0 / (1 / h + 1 / a)
    return est


def run_pipeline(
    raw_text: str,
    threshold: float | None = None,
    min_threshold: float = DEFAULT_MIN_THRESHOLD,
    max_xg: float = DEFAULT_MAX_XG,
    ratings: Ratings | None = None,
    include_unrated: bool = False,
    rank_by: str | None = None,
    calibration: Calibration | None = None,
    calibration_odds: Calibration | None = None,
    qualify_unvalidated: bool = False,
    profile: str = "regular",
    form=None,
    qualify_by: str = "band",
    price_band: tuple[float, float] = DEFAULT_PRICE_BAND,
    price_margin: float = DEFAULT_PRICE_MARGIN,
    price_floor: float = DEFAULT_PRICE_FLOOR,
    odds_lookup=None,
    selection_source: str = "odds_first",
    trust_form_all: bool = False,
) -> dict:
    """Rank fixtures by the chance that all three legs of a profile land, and tag the strong ones.

    profile="regular" (default): legs 1X, 12, Over 1.5 - markets every bookmaker lists. 1X and 12 together mean
      Home win, so All3 = Home win AND Over 1.5. Ranked by All3; ★ QUALIFIED at All3 >= 50%.
    profile="classic": the original 1X / Home +1.5 / Under 2.5 (ranked by weakest leg by default).

    Regular profile, qualify_by="band" (default): a fixture is ★ QUALIFIED when EVERY leg is priced inside price_band
    (default 1.20-1.50) and the probabilities come from odds or the form model. qualify_by="all3" qualifies on
    All3 >= threshold instead (50% default). The two rules do not overlap in practice: games with All3 >= 50%
    have legs priced below 1.20, and in-band games land All3 only ~29% in testing.

    selection_source="stats": the trained statistical model (recent form, goals against, historic form, standings, streaks,
    head-to-head, match statistics) is the ONLY selection tool. Pasted/feed odds are ignored for probabilities and selection and
    appear only as leg prices; fixtures whose teams are not in the statistical history are skipped. Rows carry the tier
    (A >= 50%, B >= 35%, C below) and the statistics behind them. "odds_first" (default here) uses odds, then the model.

    qualify_by="all3_floor" (regular profile): All3 >= threshold (50%) AND every leg priced >= price_floor (1.30). Games that
    pass the All3 test but fail the price floor are listed as "◇ ALL3 OK, PRICE BELOW FLOOR" (filtered out).
    With every leg at 1.30 or more the highest All3 seen in testing was 31%, so this combination almost never has a qualifier.

    Probabilities come from (in order): the fixture's 1X2 odds; the trained form model ("technical analysis" of
    previous games, no odds needed; regular profile only); team ratings; or - only if include_unrated -
    league-average constants. Odds-based and form-based rows can be QUALIFIED; ratings-based rows cannot
    (qualify_unvalidated=True lifts that). Calibration maps apply to the classic profile only.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    if rank_by is None:
        rank_by = "all3" if profile == "regular" else "min"
    if rank_by not in ("min", "all3"):
        raise ValueError("rank_by must be 'min' or 'all3'")
    if selection_source not in ("odds_first", "stats"):
        raise ValueError("selection_source must be 'odds_first' or 'stats'")
    stats_mode = selection_source == "stats"
    if qualify_by not in ("band", "all3", "all3_floor"):
        raise ValueError("qualify_by must be 'band', 'all3' or 'all3_floor'")
    band_mode = profile == "regular" and qualify_by == "band"
    floor_mode = qualify_by == "all3_floor"          # All3 strictly above the line AND every leg priced at least price_floor
    lo_p, hi_p = price_band
    calibration, calibration_odds = calibration or Calibration(), calibration_odds or Calibration()
    use_cal = profile == "classic"
    any_cal = use_cal and bool(calibration or calibration_odds)
    if threshold is None:
        if profile == "regular":
            threshold = REGULAR_THRESHOLD
        elif rank_by == "min":
            threshold = CALIBRATED_THRESHOLD if any_cal else DEFAULT_THRESHOLD
        else:
            threshold = CALIBRATED_ALL3_THRESHOLD if any_cal else ALL3_THRESHOLD
    ratings = ratings or Ratings()
    keys = [k for k, _ in PROFILES[profile]]
    labels = [lb for _, lb in PROFILES[profile]]
    parsed = parse_fixtures(raw_text)
    rows: list[dict] = []
    counts = {"odds": 0, "form": 0, "ratings": 0, "default": 0}
    skipped_no_data = 0

    feed_priced = 0
    skipped_no_stats = 0
    for item in parsed["fixtures"]:
        lam, source, probs, all3 = None, "default", None, None
        if not item.get("odds") and odds_lookup is not None:
            found = odds_lookup(item["home"], item["away"])
            if found:
                item = {**item, "odds": found}
                feed_priced += 1
        if item.get("odds") and not stats_mode:
            lam = lambdas_from_1x2(tuple(item["odds"]))
            if lam:
                source = "odds"
        if lam is None and form is not None:
            fp = form.predict_fixture(item["home"], item["away"])
            if fp and all(k in fp for k in keys):
                source, probs, all3 = "form", {k: fp[k] for k in keys}, fp["all3" if profile == "regular" else "all3c"]
        if stats_mode and probs is None:
            skipped_no_stats += 1                      # statistics are the only selection tool: no recent-form data, no selection
            continue
        if probs is None and lam is None:
            h_r, a_r = ratings.get(item["home"]), ratings.get(item["away"])
            if h_r or a_r:
                lam, source = expected_goals(h_r, a_r), "ratings"
        if probs is None and lam is None:
            if not include_unrated:
                skipped_no_data += 1
                continue
            lam = expected_goals(None, None)

        xg = None
        if lam is not None:
            xg = lam[0] + lam[1]
            if xg > max_xg:
                continue
            m = evaluate_profile(lam[0], lam[1], profile)
            probs, all3 = dict(m["legs"]), m["all3"]
            if use_cal:
                cmap = calibration_odds if source == "odds" else calibration
                probs = {k: cmap.apply(k, v) for k, v in probs.items()}
                all3 = cmap.apply("all3", all3)
        min_leg = min(probs.values())
        ranking_score = min_leg if rank_by == "min" else all3
        if ranking_score < min_threshold:
            continue

        counts[source] += 1
        trusted = source == "odds" or (source == "form" and (profile in FORM_TRUSTED_PROFILES or trust_form_all)) or qualify_unvalidated
        price_probs = probs
        if stats_mode and item.get("odds"):          # prices follow the MARKET, not our model: derive them from the odds
            lam_m = lambdas_from_1x2(tuple(item["odds"]))
            if lam_m:
                price_probs = evaluate_profile(lam_m[0], lam_m[1], profile)["legs"]
        est = leg_prices(profile, price_probs, item.get("odds"), price_margin)
        in_band = all(lo_p <= est[k] <= hi_p for k in keys)
        price_ok = all(est[k] >= price_floor for k in keys)
        all3_ok = all3 > threshold if floor_mode else all3 >= threshold
        if floor_mode:
            qualified = all3_ok and price_ok and trusted
        else:
            qualified = (in_band if band_mode else ranking_score >= threshold) and trusted
        status = "★ QUALIFIED" if qualified else "◆ MODERATE"
        if floor_mode and all3_ok and not price_ok:
            status = "◇ ALL3 OK, PRICE BELOW FLOOR"
        band = "HIGH" if all3 >= 0.50 else "QUALIFIES" if all3_ok else "-"
        row = {
            "time": item["time"], "competition": item["competition"], "home": item["home"], "away": item["away"],
            "match": f"{item['home']} vs {item['away']}",
            "xg": round(xg, 2) if xg is not None else None,
            "score": ranking_score, "profile": profile,
            "labels": labels, "legs": [{"key": k, "label": lb, "prob": probs[k], "pct": _pct(probs[k]), "price": round(est[k], 2)} for k, lb in zip(keys, labels)],
            "in_band": in_band, "combined_price": round(float(np.prod([est[k] for k in keys])), 2),
            "min_leg": _pct(min_leg), "all_three": _pct(all3), "all3_prob": all3,
            "status": status, "all3_ok": all3_ok, "price_ok": price_ok,
            "tier": "A" if all3 >= 0.50 else "B" if all3 >= 0.35 else "C", "band": band,
            "analysis": (form.explain(item["home"], item["away"]) or {}).get("text") if (form is not None and source == "form") else None,
            "source": source, "odds": _fmt_odds(item.get("odds")), "_tie": all3,
        }
        if profile == "classic":                     # legacy keys
            row.update(prob_1x=_pct(probs["1x"]), prob_h_plus_1_5=_pct(probs["h15"]), prob_u2_5=_pct(probs["u25"]))
        rows.append(row)

    rows.sort(key=lambda r: ((r["in_band"] if band_mode else 0), r["score"], r["_tie"]), reverse=True)
    for r in rows:
        del r["_tie"]

    warnings: list[str] = []
    if skipped_no_data:
        warnings.append(
            f"{skipped_no_data} fixture(s) had no odds, no form history and no team ratings and were skipped. "
            "Paste Flashscore's Odds page (Football » Odds) or add the teams to the history/aliases."
        )
    if stats_mode:
        warnings.append(
            "Statistics-only mode: the form model (recent form, goals against, historic form, standings, streaks, head-to-head, match statistics) is the only "
            "selection tool; odds are used only to price the legs for the 1.30 floor. Measured on unseen games its 50%+ All3 picks landed ~59% (1X + 12 + Over 1.5) against ~64% for "
            "odds-based probabilities; for the classic legs it over-predicts All3 by about 4 points, and its Under 2.5 has no skill."
        )
        if skipped_no_stats:
            warnings.append(f"{skipped_no_stats} fixture(s) skipped: no recent-form data for the teams (not in the history, a different league, or a name that needs aliases.json).")
    if feed_priced:
        warnings.append(f"{feed_priced} fixture(s) had no pasted odds and were priced from the open fixtures feed (average 1X2 prices, a Friday/Tuesday snapshot).")
    if floor_mode and rows:
        n_a, n_q = sum(r["all3_ok"] for r in rows), sum(r["status"].startswith("★") for r in rows)
        warnings.append(
            f"All3 above {threshold * 100:.0f}% and every leg priced >= {price_floor:.2f}: {n_q} qualified, {n_a - n_q} passed All3 but were filtered out on price. "
            "Strong favourites are priced short on at least one leg, so the floor removes them."
        )
    if band_mode and rows:
        warnings.append(
            f"Price-band mode: ★ means every leg is priced {lo_p:.2f}-{hi_p:.2f}. In testing those games landed All3 (Home win AND "
            "Over 1.5) only about 29%; games with All3 >= 50% are priced below 1.20 on at least one leg. Use qualify_by='all3' for the 50% rule."
        )
    if counts["form"]:
        warnings.append(
            f"{counts['form']} row(s) come from the trained form model (no odds). In testing its 50%+ All3 picks landed "
            "about 59%, against about 64% when odds are available."
        )
    if counts["ratings"] or counts["default"]:
        warnings.append(
            "Rows from team ratings (not odds or the form model) are never tagged QUALIFIED: on unseen 2026/27 games ratings "
            "fitted on one season were no better than a flat guess for the three-leg combination."
        )

    copyable = [
        f"{r['time']} | {r['competition']} | {r['match']} | xG:{r['xg'] if r['xg'] is not None else '-'} | "
        + " | ".join(f"{lg['label']}:{lg['pct']}@{lg['price']:.2f}" for lg in r["legs"])
        + f" | MinLeg:{r['min_leg']} | All3:{r['all_three']} | {'in band' if r['in_band'] else 'out of band'} | {r['status']} | src:{r['source']}"
        for r in rows
    ]

    return {
        "rank_by": rank_by, "profile": profile, "threshold": threshold, "calibrated": any_cal, "selection_source": selection_source,
        "qualify_by": qualify_by if profile == "regular" else "all3", "price_band": list(price_band), "price_floor": price_floor,
        "summary": {
            "total_leagues": parsed["leagues_count"], "total_fixtures": parsed["fixtures_count"],
            "ranked_count": len(rows), "qualified_count": sum(r["status"].startswith("★") for r in rows), "in_band_count": sum(r["in_band"] for r in rows),
            "from_odds": counts["odds"], "from_form": counts["form"], "from_ratings": counts["ratings"],
            "rated_fixtures": counts["odds"] + counts["form"] + counts["ratings"],
            "skipped_no_data": skipped_no_data, "skipped_no_stats": skipped_no_stats, "skipped_in_play": parsed["skipped_in_play"],
            "skipped_postponed": parsed["skipped_postponed"],
        },
        "warnings": warnings, "cleaned_schedule": parsed["cleaned_text"],
        "ranked_fixtures": rows, "copyable_block": "\n".join(copyable),
    }
