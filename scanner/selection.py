"""Single-selection mode: statistics first, price second.

1. Statistics: score the common result markets (Home win, Away win, 1X, X2, 12) with the best available model
   (the fixture's 1X2 odds, else the trained form model), calibrated against past results.
2. Price: keep only selections whose bookmaker price lies in the target window (default 1.30-1.35) and pick the most
   probable one. Prices come from the pasted 1X2 odds (1X = 1/(1/H+1/D) etc.); with no odds there is no price, so the
   row says what price you would need.
No edge over the bookmaker is claimed: in testing, selections priced 1.30-1.35 landed 72% against a 75.6% break-even.
"""
from __future__ import annotations

from .calibration import Calibration
from .model import lambdas_from_1x2
from .parser import parse_fixtures

import numpy as np

from .model import dixon_coles_matrix

SELECTIONS = {  # key: (label, price from 1X2 odds)
    "home": ("Home win", lambda h, d, a: h),
    "away": ("Away win", lambda h, d, a: a),
    "1x": ("1X (home or draw)", lambda h, d, a: 1 / (1 / h + 1 / d)),
    "x2": ("X2 (draw or away)", lambda h, d, a: 1 / (1 / d + 1 / a)),
    "12": ("12 (no draw)", lambda h, d, a: 1 / (1 / h + 1 / a)),
}
DEFAULT_WINDOW = (1.30, 1.35)


def _odds_probs(lam) -> dict:
    m = dixon_coles_matrix(*lam)
    x, y = np.indices(m.shape)
    return {"home": float(m[x > y].sum()), "away": float(m[y > x].sum()), "1x": float(m[x >= y].sum()),
            "x2": float(m[y >= x].sum()), "12": float(m[x != y].sum())}


def run_selection(raw_text: str, price_window: tuple[float, float] = DEFAULT_WINDOW, calibration: Calibration | None = None,
                  form=None, min_prob: float = 0.0) -> dict:
    lo, hi = price_window
    cal = calibration or Calibration()
    parsed = parse_fixtures(raw_text)
    rows, counts, skipped = [], {"odds": 0, "form": 0}, 0
    for item in parsed["fixtures"]:
        odds, probs, source = item.get("odds"), None, None
        if odds:
            lam = lambdas_from_1x2(tuple(odds))
            if lam:
                probs, source = _odds_probs(lam), "odds"
        if probs is None and form is not None:
            fp = form.predict_fixture(item["home"], item["away"])
            if fp:
                probs, source = {k: fp[k] for k in SELECTIONS}, "form"
        if probs is None:
            skipped += 1
            continue
        p = {k: cal.apply(f"sel_{k}", v) for k, v in probs.items()}
        cands = []
        for k, (label, pf) in SELECTIONS.items():
            price = pf(*odds) if odds else None
            fair = 1 / p[k]
            cands.append({"key": k, "label": label, "p": p[k], "fair_price": round(fair, 2), "price": round(price, 2) if price else None,
                          "ev": (p[k] * price - 1) if price else None, "in_window": bool(price and lo <= price <= hi)})
        by_stats = max(cands, key=lambda c: c["p"])
        window = [c for c in cands if c["in_window"]]
        if source == "odds":
            pick = max(window, key=lambda c: c["p"]) if window else None
            status = "★ PICK" if pick else "◆ NO SELECTION IN WINDOW"
        else:
            # no bookmaker price: the most probable selection whose fair price is in the window, to be checked at the bookmaker
            near = [c for c in cands if lo <= c["fair_price"] <= hi + 0.15]
            pick = max(near, key=lambda c: c["p"]) if near else None
            status = "◇ CHECK PRICE" if pick else "◆ NO SELECTION IN WINDOW"
        if pick is None and min_prob > 0 and by_stats["p"] < min_prob:
            continue
        counts[source] += 1
        rows.append({
            "time": item["time"], "competition": item["competition"], "home": item["home"], "away": item["away"],
            "match": f"{item['home']} vs {item['away']}", "source": source, "status": status, "pick": pick, "best_by_stats": by_stats,
            "candidates": cands, "score": (pick["p"] if pick else 0.0) + (1 if pick and source == "odds" else 0),
            "odds": "/".join(f"{o:g}" for o in odds) if odds else "", "xg": None, "all_three": "", "min_leg": "", "in_band": bool(pick),
            "legs": [{"key": pick["key"], "label": pick["label"], "prob": pick["p"], "pct": f"{pick['p'] * 100:.1f}%", "price": pick["price"] or pick["fair_price"]}] if pick else [],
            "profile": "single",
        })
    rows.sort(key=lambda r: r["score"], reverse=True)

    def line(r):
        b, pk = r["best_by_stats"], r["pick"]
        head = f"{r['time']} | {r['competition']} | {r['match']} | best by stats: {b['label']} {b['p'] * 100:.1f}% @{b['price'] if b['price'] else 'n/a'}"
        if pk:
            price = f"@{pk['price']:.2f}" if pk["price"] else f"needs price >= {pk['fair_price']:.2f}"
            ev = f" | edge {pk['ev'] * 100:+.1f}%" if pk["ev"] is not None else ""
            return f"{head} | PICK: {pk['label']} {pk['p'] * 100:.1f}% {price}{ev} | {r['status']} | src:{r['source']}"
        return f"{head} | no selection priced {lo:.2f}-{hi:.2f} | {r['status']} | src:{r['source']}"

    warnings = ["Statistics pick the market, the price window then filters it. In testing, selections priced "
                f"{lo:.2f}-{hi:.2f} landed about 72% against a ~75.6% break-even (return -4.8%): the window does not create value."]
    if counts["form"]:
        warnings.append(f"{counts['form']} row(s) come from the trained form model and have no pasted price: check the bookmaker pays at least the stated price.")
    picks = [r for r in rows if r["pick"]]
    return {
        "mode": "single", "rank_by": "selection", "profile": "single", "threshold": lo, "price_window": [lo, hi], "calibrated": bool(cal),
        "qualify_by": "window",
        "summary": {"total_leagues": parsed["leagues_count"], "total_fixtures": parsed["fixtures_count"], "ranked_count": len(rows),
                    "qualified_count": sum(r["status"] == "★ PICK" for r in rows), "in_band_count": len(picks),
                    "from_odds": counts["odds"], "from_form": counts["form"], "from_ratings": 0, "rated_fixtures": counts["odds"] + counts["form"],
                    "skipped_no_data": skipped, "skipped_in_play": parsed["skipped_in_play"], "skipped_postponed": parsed["skipped_postponed"]},
        "warnings": warnings, "cleaned_schedule": parsed["cleaned_text"], "ranked_fixtures": rows,
        "copyable_block": "\n".join(line(r) for r in rows),
    }
