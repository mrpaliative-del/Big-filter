"""The three scanners on one paste, side by side, listed by kickoff time.

CLASSIC  : 1X, Home +1.5, Under 2.5 (the first scanner, unchanged rules: calibrated legs, every leg at the common line).
REGULAR  : 1X, 12, Over 1.5 (regular bookmaker markets): All3 line first, then every leg priced >= 1.30.
DRAW     : the draw predictor (Dixon-Coles + calibrated booster blend) for Serie B, Ligue 2, Segunda and the Championship:
           a draw is a pick when its probability beats the market draw price by >= 4 points at odds <= 4.10 (quarter-Kelly stake).
All three share the parser, kickoff ordering, odds sources (pasted or open feed), pick log and display.
"""
from __future__ import annotations

from .calibration import Calibration
from .model import Ratings
from .parser import parse_fixtures
from .pipeline import DEFAULT_MAX_XG, DEFAULT_MIN_THRESHOLD, DEFAULT_PRICE_FLOOR, DEFAULT_PRICE_MARGIN, REGULAR_THRESHOLD, run_pipeline


def _short(status: str) -> str:
    return "★" if status.startswith("★") else "◇ price<floor" if status.startswith("◇") else "◆"


def _classic_text(r) -> str:
    if not r:
        return "CLASSIC –"
    legs = " ".join(f"{lg['label']}:{lg['pct']}" for lg in r["legs"])
    return f"CLASSIC {legs} All3:{r['all_three']} tier {r['tier']} {_short(r['status'])}"


def _regular_text(r) -> str:
    if not r:
        return "REGULAR –"
    legs = " ".join(f"{lg['label']}:{lg['pct']}@{lg['price']:.2f}" for lg in r["legs"])
    return f"REGULAR {legs} All3:{r['all_three']} tier {r['tier']} {_short(r['status'])}"


def _draw_text(p) -> str:
    if not p:
        return "DRAW –"
    if p.get("prob") is None:
        return f"DRAW n/a ({p['reason']})"
    s = (f"DRAW p:{p['prob'] * 100:.1f}% (DC {p['dc_prob'] * 100:.1f}/GB {p['xgb_prob'] * 100:.1f}) @{p['price']:.2f} "
         f"fair {p['fair_odds']:.2f} edge {p['edge'] * 100:+.1f}%")
    return s + (f" stake {p['stake']:.2f} ★" if p["qualified"] else f" ({p['reason']}) ◆")


def _draw_panel(item: dict, draw, odds) -> dict | None:
    res = draw.predict_fixture(item["home"], item["away"], odds)
    if res is None:
        return None                                              # league not covered by the draw scanner
    has_p = res.get("prob") is not None
    status = "★ DRAW PICK" if res["qualified"] else ("◆ " if has_p else "◇ ") + res["reason"]
    panel = {"profile": "draw", "time": item["time"], "competition": item["competition"], "home": item["home"], "away": item["away"],
             "match": f"{item['home']} vs {item['away']}", "source": "odds" if odds else "none", "status": status, "league": res["league"],
             "reason": res.get("reason", ""), "qualified": res["qualified"], "prob": res.get("prob"), "dc_prob": res.get("dc_prob"),
             "xgb_prob": res.get("xgb_prob"), "edge": res.get("edge"), "fair_odds": res.get("fair_odds"), "price": res.get("odds"),
             "stake": res.get("stake", 0.0), "odds": "/".join(f"{o:g}" for o in odds) if odds else "", "tier": "-"}
    panel["pick"] = {"key": "draw", "p": res["prob"], "price": res["odds"]} if has_p else None
    return panel


def run_both(
    raw_text: str,
    ratings: Ratings | None = None,
    calibration: Calibration | None = None,
    calibration_odds: Calibration | None = None,
    form=None,
    odds_lookup=None,
    min_threshold: float = DEFAULT_MIN_THRESHOLD,
    max_xg: float = DEFAULT_MAX_XG,
    include_unrated: bool = False,
    classic_threshold: float | None = None,
    all3_min: float | None = None,
    price_floor: float | None = None,
    selection_source: str = "stats",
    price_margin: float = DEFAULT_PRICE_MARGIN,
    show_only_qualified: bool = False,
    draw=None,
    scanners: tuple = ("classic", "regular", "draw"),
) -> dict:
    stats_mode = selection_source == "stats"
    parsed = parse_fixtures(raw_text)
    if all3_min is None:
        all3_min = 0.20 if stats_mode else REGULAR_THRESHOLD
    if price_floor is None:
        price_floor = DEFAULT_PRICE_FLOOR                                    # the 1.30 leg-price target applies in both modes
    min_threshold = min(min_threshold, all3_min)
    common = dict(selection_source=selection_source, min_threshold=min_threshold, max_xg=max_xg, ratings=ratings, include_unrated=include_unrated, form=form,
                  odds_lookup=odds_lookup, price_margin=price_margin)
    classic = regular = None
    if "classic" in scanners:
        if stats_mode:   # both scanners qualify on All3 >= all3_min from the statistical model
            classic = run_pipeline(raw_text, threshold=all3_min, rank_by="all3", calibration=calibration, calibration_odds=calibration_odds, profile="classic",
                                   trust_form_all=True, **common)
        else:
            classic = run_pipeline(raw_text, threshold=classic_threshold, calibration=calibration, calibration_odds=calibration_odds, profile="classic", **common)
    if "regular" in scanners:
        if stats_mode:
            regular = run_pipeline(raw_text, threshold=all3_min, profile="regular", qualify_by="all3_floor" if price_floor > 1.0 else "all3",
                                   price_floor=price_floor, trust_form_all=True, **common)
        else:
            regular = run_pipeline(raw_text, threshold=all3_min, profile="regular", qualify_by="all3_floor", price_floor=price_floor, **common)

    merged: dict[tuple, dict] = {}
    for name, res in (("classic", classic), ("regular", regular)):
        if res is None:
            continue
        for r in res["ranked_fixtures"]:
            k = (r["time"], r["competition"], r["home"], r["away"])
            m = merged.setdefault(k, {"time": r["time"], "competition": r["competition"], "home": r["home"], "away": r["away"],
                                      "match": r["match"], "classic": None, "regular": None, "draw": None, "source": r["source"], "odds": r["odds"]})
            m[name] = r
            if r["source"] == "odds":
                m["source"], m["odds"] = "odds", r["odds"]
    if draw is not None and "draw" in scanners:                       # the third scanner scores every pasted fixture it covers
        for item in parsed["fixtures"]:
            odds = item.get("odds") or (odds_lookup(item["home"], item["away"]) if odds_lookup else None)
            panel = _draw_panel(item, draw, odds)
            if panel:
                k = (item["time"], item["competition"], item["home"], item["away"])
                m = merged.setdefault(k, {"time": item["time"], "competition": item["competition"], "home": item["home"], "away": item["away"],
                                          "match": panel["match"], "classic": None, "regular": None, "draw": None,
                                          "source": "odds" if odds else "none", "odds": panel["odds"]})
                m["draw"] = panel
    rows = sorted(merged.values(), key=lambda m: (m["time"], m["competition"], m["match"]))           # kickoff order
    is_q = lambda p: bool(p) and p["status"].startswith("★")
    if show_only_qualified:
        rows = [m for m in rows if is_q(m["classic"]) or is_q(m["regular"]) or is_q(m["draw"])]
    for m in rows:
        m["status"] = " / ".join([f"{n}: {_short(m[n]['status'])}" for n in ("classic", "regular") if m[n]] + (["draw: " + m["draw"]["status"][:1]] if m["draw"] else []))
        m["legs"] = []

    warnings = []
    for w in (classic["warnings"] if classic else []) + (regular["warnings"] if regular else []):
        if w not in warnings:
            warnings.append(w)
    if any(m["draw"] and m["draw"]["prob"] is not None for m in rows):
        warnings.append(
            "Draw scanner: it prices against the market draw price (1/odds is both a model input and the edge benchmark). In walk-forward testing "
            "(428 games, Aug-Nov 2026, four leagues, one season of training data) its probabilities did not beat the margin-free market "
            "(log-loss 0.5541 vs 0.5534) and the 4-point rule gave only 7 picks (3 landed): treat it as unproven."
        )
    panels = [p for m in rows for p in (m["classic"], m["regular"], m["draw"]) if p and (p.get("profile") != "draw" or p.get("pick"))]
    ps = (regular or classic or {}).get("summary") or {"skipped_no_data": 0, "skipped_no_stats": 0}
    summary = {
        "total_leagues": parsed["leagues_count"], "total_fixtures": parsed["fixtures_count"], "ranked_count": len(rows),
        "qualified_count": sum(is_q(m["classic"]) or is_q(m["regular"]) or is_q(m["draw"]) for m in rows),
        "qualified_classic": sum(is_q(m["classic"]) for m in rows), "qualified_regular": sum(is_q(m["regular"]) for m in rows),
        "qualified_draw": sum(is_q(m["draw"]) for m in rows),
        "in_band_count": sum(bool(m["regular"]) and m["regular"]["status"].startswith("◇") for m in rows),   # passed All3, filtered on price
        "from_odds": sum(m["source"] == "odds" for m in rows), "from_form": sum(m["source"] == "form" for m in rows),
        "from_ratings": sum(m["source"] == "ratings" for m in rows), "rated_fixtures": len(rows),
        "skipped_no_data": ps.get("skipped_no_data", 0), "skipped_no_stats": ps.get("skipped_no_stats", 0), "skipped_in_play": parsed["skipped_in_play"], "skipped_postponed": parsed["skipped_postponed"],
    }
    def stats_text(m):
        for k in ("regular", "classic"):
            if m[k] and m[k].get("analysis"):
                return f" | stats: {m[k]['analysis']}"
        return ""

    def line(m):
        parts = [m["time"], m["competition"], m["match"]]
        if "classic" in scanners:
            parts.append(_classic_text(m["classic"]))
        if "regular" in scanners:
            parts.append(_regular_text(m["regular"]))
        if draw is not None and "draw" in scanners:
            parts.append(_draw_text(m["draw"]))
        return " | ".join(parts) + f" | src:{m['source']}{stats_text(m)}"

    lines = [line(m) for m in rows]
    used_draw = draw is not None and "draw" in scanners
    return {
        "mode": "all" if used_draw and len(scanners) == 3 else "draw" if scanners == ("draw",) else "both", "rank_by": "time", "profile": "both",
        "threshold": all3_min, "calibrated": bool(classic and classic["calibrated"]),
        "qualify_by": "both", "price_floor": price_floor, "selection_source": selection_source, "summary": summary, "warnings": warnings,
        "cleaned_schedule": parsed["cleaned_text"], "ranked_fixtures": rows, "panels": panels, "copyable_block": "\n".join(lines),
    }
