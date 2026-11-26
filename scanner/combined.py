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
from .profile import fixture_profile, profile_text
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
    s = f"DRAW rating:{p['prob'] * 100:.1f}% @{p['price']:.2f} fair {p['fair_odds']:.2f} edge {p['edge'] * 100:+.1f}%"
    if p.get("blend_prob") is not None:
        s += f" (4-league DC/GB blend {p['blend_prob'] * 100:.1f}%)"
    if p["qualified"]:
        return s + (f" stake {p['stake']:.2f}" if p.get("stake") else "") + " ★"
    return s + (" ◇ watchlist" if p.get("watch") else " ◆")


def _draw_panel(item: dict, draw, odds, draw_min=None) -> dict | None:
    try:
        res = draw.predict_fixture(item["home"], item["away"], odds, draw_min=draw_min) if draw_min is not None else draw.predict_fixture(item["home"], item["away"], odds)
    except TypeError:
        res = draw.predict_fixture(item["home"], item["away"], odds)
    if res is None:
        return None                                              # fixture not scorable by the draw scanner
    has_p = res.get("prob") is not None
    watch = bool(res.get("watch"))
    status = "★ DRAW PICK" if res["qualified"] else "◇ DRAW WATCHLIST" if watch else ("◆ " if has_p else "◇ ") + res["reason"]
    panel = {"profile": "draw", "time": item["time"], "competition": item["competition"], "home": item["home"], "away": item["away"],
             "match": f"{item['home']} vs {item['away']}", "source": "odds" if odds else "none", "status": status, "league": res["league"],
             "reason": res.get("reason", ""), "qualified": res["qualified"], "watch": watch, "prob": res.get("prob"), "dc_prob": res.get("dc_prob"),
             "xgb_prob": res.get("xgb_prob"), "blend_prob": res.get("blend_prob"), "edge": res.get("edge"), "fair_odds": res.get("fair_odds"),
             "price": res.get("odds"), "stake": res.get("stake", 0.0), "drivers": res.get("drivers"),
             "odds": "/".join(f"{o:g}" for o in odds) if odds else "", "tier": "-"}
    panel["pick"] = {"key": "draw", "p": res["prob"], "price": res["odds"]} if has_p else None
    return panel


def _report_block(m, form_profile) -> str:
    """Plain-text report for one fixture in the recommended format."""
    lines = [f"⏰ {m['time']} · {m['competition']}", f"{m['match']}"]
    for key, name in (("classic", "SCANNER 1 · 1X · Home +1.5 · Under 2.5"), ("regular", "SCANNER 2 · 1X · 12 · Over 1.5")):
        r = m[key]
        if r:
            legs = " | ".join(f"{lg['label']} {lg['pct']} @{lg['price']:.2f}" for lg in r["legs"])
            lines.append(f"{name}: All3 {r['all_three']} [{r['status'][:1]}] {legs}")
    d = m["draw"]
    if d and d.get("prob") is not None:
        lines.append(f"SCANNER 3 · Draw: rating {d['prob'] * 100:.1f}% @{d['price']:.2f} [{d['status'][:1]}]")
    if form_profile:
        lines += ["", profile_text(form_profile)]
    return "\n".join(lines)


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
    draw_min: float | None = None,
    scanners: tuple = ("classic", "regular", "draw"),
) -> dict:
    stats_mode = selection_source == "stats"
    parsed = parse_fixtures(raw_text)
    if all3_min is None:
        all3_min = 0.20                                                       # All3 strictly above 20% (both scanners)
    if price_floor is None:
        price_floor = DEFAULT_PRICE_FLOOR                                    # the 1.30 leg-price target applies in both modes
    min_threshold = min(min_threshold, all3_min)
    common = dict(selection_source=selection_source, min_threshold=min_threshold, max_xg=max_xg, ratings=ratings, include_unrated=include_unrated, form=form,
                  odds_lookup=odds_lookup, price_margin=price_margin)
    classic = regular = None
    if "classic" in scanners:
        classic = run_pipeline(raw_text, threshold=classic_threshold if classic_threshold is not None else all3_min, rank_by="all3",
                               calibration=calibration, calibration_odds=calibration_odds, profile="classic",
                               qualify_by="all3_floor" if price_floor > 1.0 else "all3", price_floor=price_floor, trust_form_all=stats_mode, **common)
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
            panel = _draw_panel(item, draw, odds, draw_min)
            if panel:
                k = (item["time"], item["competition"], item["home"], item["away"])
                m = merged.setdefault(k, {"time": item["time"], "competition": item["competition"], "home": item["home"], "away": item["away"],
                                          "match": panel["match"], "classic": None, "regular": None, "draw": None,
                                          "source": "odds" if odds else "none", "odds": panel["odds"]})
                m["draw"] = panel
    rows = sorted(merged.values(), key=lambda m: (m["time"], m["competition"], m["match"]))           # kickoff order
    is_q = lambda p: bool(p) and p["status"].startswith("★")
    scored_count = len(rows)

    def miss_reason(p, name):
        legs = [lg for lg in p["legs"] if lg["price"] < price_floor]
        if p["status"].startswith("◇") and legs:
            return f"{name}: All3 {p['all_three']} but " + ", ".join(f"{lg['label']} priced {lg['price']:.2f}" for lg in legs) + f" (floor {price_floor:.2f})"
        return f"{name}: All3 {p['all_three']} is not above {all3_min * 100:.0f}%"

    near = []
    for m in rows:
        if any(m[k] and m[k]["status"].startswith("★") for k in ("classic", "regular")) or (m["draw"] and m["draw"]["qualified"]):
            continue
        best = max(((m[k], n) for k, n in (("classic", "Scanner 1"), ("regular", "Scanner 2")) if m[k]), key=lambda t: t[0]["all3_prob"], default=None)
        if best:
            near.append({"time": m["time"], "competition": m["competition"], "match": m["match"], "all3": best[0]["all3_prob"], "reason": miss_reason(best[0], best[1])})
    near_misses = sorted(near, key=lambda d: -d["all3"])[:5]
    watchlist = [m for m in rows if m["draw"] and m["draw"].get("watch")]
    if show_only_qualified:
        rows = [m for m in rows if is_q(m["classic"]) or is_q(m["regular"]) or is_q(m["draw"])]
    for m in rows:
        m["status"] = " / ".join([f"{n}: {_short(m[n]['status'])}" for n in ("classic", "regular") if m[n]] + (["draw: " + m["draw"]["status"][:1]] if m["draw"] else []))
        m["legs"] = []

    for m in rows + [w for w in watchlist if w not in rows]:          # team profile in the recommended format, for the fixtures shown
        m["profile_ratings"] = fixture_profile(form, m["home"], m["away"]) if form is not None else None
        m["profile_text"] = profile_text(m["profile_ratings"]) if m["profile_ratings"] else ""

    warnings = []
    for w in (classic["warnings"] if classic else []) + (regular["warnings"] if regular else []):
        if w not in warnings:
            warnings.append(w)
    if any(m["draw"] and m["draw"]["prob"] is not None for m in rows + watchlist):
        warnings.append(
            "Draw rating: from the draw drivers (parity, low scoring, tight defences, draw-prone teams, head-to-head draws, tempo) and the market price. "
            "Out of sample it matched the margin-free market (log-loss 0.5449 vs 0.5433). Draws rated 30-33% landed 32% (315 games); no calibrated model rates a "
            "draw above ~37% and the market's own highest draw probability was 33.5%, so a 40% line is very rarely met."
        )
    if stats_mode:
        warnings.append(
            "Measured on 2,397 unseen games (Jul-Nov 2026) with these rules: Scanner 1 qualified 14% of games and they landed All3 20% (the model said 31%); "
            "Scanner 2 qualified 8% and landed 29% (model 31%); only 4 games had All3 of 50% or more with every leg priced 1.25+; no draw rating reached 40% "
            "(ratings of 30%+ drew 31% of the time). Treat the All3 shown for Scanner 1 as optimistic."
        )
    panels = [p for m in rows for p in (m["classic"], m["regular"], m["draw"]) if p and (p.get("profile") != "draw" or p.get("pick"))]
    ps = (regular or classic or {}).get("summary") or {"skipped_no_data": 0, "skipped_no_stats": 0}
    summary = {
        "total_leagues": parsed["leagues_count"], "total_fixtures": parsed["fixtures_count"], "ranked_count": len(rows),
        "qualified_count": sum(is_q(m["classic"]) or is_q(m["regular"]) or is_q(m["draw"]) for m in rows),
        "qualified_classic": sum(is_q(m["classic"]) for m in rows), "qualified_regular": sum(is_q(m["regular"]) for m in rows),
        "qualified_draw": sum(is_q(m["draw"]) for m in rows), "watch_draw": len(watchlist), "scored_count": scored_count,
        "highly_expected": sum(any(m[k] and m[k]["all3_prob"] >= 0.50 and is_q(m[k]) for k in ("classic", "regular")) for m in rows),
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
    report = "\n\n".join(_report_block(m, m.get("profile_ratings")) for m in rows)
    used_draw = draw is not None and "draw" in scanners
    return {
        "mode": "all" if used_draw and len(scanners) == 3 else "draw" if scanners == ("draw",) else "both", "rank_by": "time", "profile": "both",
        "threshold": all3_min, "calibrated": bool(classic and classic["calibrated"]),
        "qualify_by": "both", "price_floor": price_floor, "selection_source": selection_source, "summary": summary, "warnings": warnings,
        "cleaned_schedule": parsed["cleaned_text"], "ranked_fixtures": rows, "watchlist": watchlist, "near_misses": near_misses, "panels": panels,
        "copyable_block": "\n".join(lines), "report_text": report, "draw_min": draw_min if draw_min is not None else getattr(draw, "draw_min", 0.40),
    }
