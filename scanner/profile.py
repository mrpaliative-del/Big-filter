"""Team profile in the scanner's display format: 9 indicators per team, each rated against the same league.

Excellent / Good / Average / Poor / Bad come from where the team ranks among the teams of its league this season
(top 20% = Excellent ... bottom 20% = Bad). Fouls and cards are Lesser / Average / Plenty.
Indicators: goal record, corners, attack, near-goal chances, winning streak, fouls, cards, home-or-away form, head-to-head.
"""
from __future__ import annotations

import numpy as np

ROWS = ["Goal record", "Corner", "Attack", "Near goal chances", "Winning streak", "Fouls", "Cards", "Home and away form", "Head2head"]
GOOD = {"Excellent", "Good", "Lesser"}
WEAK = {"Poor", "Bad", "Plenty"}


def _mean(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else None


def _pct(v, pop):
    pop = [p for p in pop if p is not None]
    if v is None or len(pop) < 4:
        return None
    return (sum(p < v for p in pop) + 0.5 * sum(p == v for p in pop)) / len(pop)


def _grade(p, invert=False):
    if p is None:
        return "No data"
    if invert:                                    # fouls / cards: fewer is better
        return "Lesser" if p <= 0.35 else "Plenty" if p >= 0.65 else "Average"
    return "Excellent" if p >= 0.8 else "Good" if p >= 0.6 else "Average" if p >= 0.4 else "Poor" if p >= 0.2 else "Bad"


def _metrics(eng, t, side):
    hs = list(eng.hist[t])
    l5, l10 = hs[-5:], hs[-10:]
    st = list(eng.stats[t])[-5:]
    col = lambda i: [x[i] for x in st]
    vv = list(eng.venue[t][side])
    run = 0
    for g, c, *_ in reversed(hs):
        if g > c:
            run += 1
        else:
            break
    wins5 = sum(g > c for g, c, *_ in l5)
    sot, goals = _mean(col(4)), _mean([g for g, *_ in l5])
    return {
        "gd": _mean([g - c for g, c, *_ in l10]),
        "corners": _mean(col(0)),
        "attack": None if goals is None else goals + 0.3 * (sot or 0.0),
        "chances": sot,
        "streak": (wins5 + run) if hs else None,
        "fouls": _mean(col(2)),
        "cards": _mean(col(3)),
        "venue": _mean([3 if g > c else 1 if g == c else 0 for g, c in vv]) if vv else None,
        "run": run, "form": "".join("W" if g > c else "D" if g == c else "L" for g, c, *_ in l5), "last_sot": (st[-1][4] if st and not np.isnan(st[-1][4]) else None),
        "gf5": goals, "n10": len(l10),
    }


def _h2h(eng, team, other):
    prev = eng.h2h[frozenset((team, other))][-4:]
    if not prev:
        return "No data", "no earlier meeting in the data"
    w = sum(1 for h, hg, ag in prev if (hg > ag if h == team else ag > hg))
    d = sum(1 for _, hg, ag in prev if hg == ag)
    rate = (w + 0.5 * d) / len(prev)
    grade = "Excellent" if rate >= 0.75 else "Good" if rate >= 0.6 else "Average" if rate >= 0.4 else "Poor" if rate >= 0.25 else "Bad"
    return grade, f"{len(prev)} meeting(s): won {w}, drew {d}, lost {len(prev) - w - d}"


def fixture_profile(form, home: str, away: str, home_label: str | None = None, away_label: str | None = None):
    """Both teams' ratings in the display format, plus how many indicators favour the home side / weaken the away side."""
    h, a = form.resolve(home), form.resolve(away)
    if not h or not a or form.engine.loc.get(h) is None or form.engine.loc[h][:2] != form.engine.loc.get(a, (None, None))[:2]:
        return None
    eng = form.engine
    season, div = eng.loc[h][:2]
    teams = [t for t in eng.table[(season, div)] if len(eng.hist[t]) >= 3]
    pop = {"h": [_metrics(eng, t, "h") for t in teams], "a": [_metrics(eng, t, "a") for t in teams]}
    out = {}
    for key, t, side, other, label in (("home", h, "h", a, home_label or home), ("away", a, "a", h, away_label or away)):
        m = _metrics(eng, t, side)
        pp = pop[side]
        g = lambda k: _pct(m[k], [x[k] for x in pp])
        f = lambda v, nd=1, suf="": "n/a" if v is None else f"{v:.{nd}f}{suf}"
        h2h_grade, h2h_detail = _h2h(eng, t, other)
        venue = "home" if side == "h" else "away"
        out[key] = {"name": label, "form": m["form"], "rows": [
            ("Goal record", _grade(g("gd")), f"goal difference {f(m['gd'], 1)} per game (last {m['n10']})"),
            ("Corner", _grade(g("corners")), f"{f(m['corners'])} corners per game"),
            ("Attack", _grade(g("attack")), f"{f(m['gf5'])} goals + chances per game"),
            ("Near goal chances", _grade(g("chances")), f"{f(m['chances'])} shots on target per game; {f(m['last_sot'], 0)} in the last match"),
            ("Winning streak", _grade(g("streak")), f"form {m['form'] or '-'}, {m['run']} win(s) in a row"),
            ("Fouls", _grade(g("fouls"), True), f"{f(m['fouls'])} fouls per game"),
            ("Cards", _grade(g("cards"), True), f"{f(m['cards'])} card points per game"),
            ("Home and away form", _grade(g("venue")), f"{f(m['venue'], 2)} points per game {venue} (last 5)"),
            ("Head2head", h2h_grade, h2h_detail),
        ]}
    hs = sum(r[1] in GOOD for r in out["home"]["rows"])
    aw = sum(r[1] in WEAK for r in out["away"]["rows"])
    out["alignment"] = {"home_strong": hs, "away_weak": aw, "of": len(ROWS)}
    return out


def profile_text(prof: dict) -> str:
    """The recommended plain-text format."""
    lines = []
    for key, title in (("home", "HOME TEAM"), ("away", "AWAY TEAM")):
        t = prof[key]
        lines.append(f"√{title} ({t['name']})")
        for label, grade, _ in t["rows"]:
            lines.append(f"- {label}: {grade}")
        lines.append("")
    return "\n".join(lines).rstrip()
