"""Dixon-Coles scoreline model and the three-market evaluation.

Only numpy is required (scipy.stats.poisson was replaced by a closed-form pmf,
which removes a ~100 MB dependency and speeds up cold starts).
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BASE_GOALS = 1.35
HOME_SCORED, AWAY_SCORED = 1.08, 0.92
HOME_CONCEDED, AWAY_CONCEDED = 0.90, 1.12
DEFAULT_RHO = -0.11
DEFAULT_MAX_GOALS = 10


# --------------------------------------------------------------------------- #
# Team ratings (optional). Without them every fixture gets identical numbers.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rating:
    """Multipliers relative to an average team. defence > 1 means leaky."""
    attack: float = 1.0
    defence: float = 1.0


def _norm(name: str) -> str:
    """Case-, accent- and punctuation-insensitive key: 'St. Truiden' == 'st truiden', 'Köln' == 'Koln'."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", plain.casefold())).strip()


class Ratings:
    def __init__(self, table: dict[str, Rating] | None = None):
        self._table = {_norm(k): v for k, v in (table or {}).items()}

    def __len__(self) -> int:
        return len(self._table)

    def get(self, team: str) -> Rating | None:
        return self._table.get(_norm(team))

    @classmethod
    def from_json(cls, path: str | Path) -> "Ratings":
        """Load ``{"Team": {"attack": 1.2, "defence": 0.8}}``. Missing file -> empty."""
        p = Path(path)
        if not p.is_file():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        table: dict[str, Rating] = {}
        for team, v in data.items():
            if team.startswith("_"):  # allow "_comment" keys
                continue
            att, dfn = float(v.get("attack", 1.0)), float(v.get("defence", 1.0))
            if not (0.2 <= att <= 4.0 and 0.2 <= dfn <= 4.0):
                raise ValueError(f"Rating for {team!r} outside sane range 0.2-4.0")
            table[team] = Rating(att, dfn)
        log.info("Loaded %d team ratings from %s", len(table), p)
        return cls(table)


def expected_goals(home: Rating | None, away: Rating | None) -> tuple[float, float]:
    """(lambda_home, lambda_away). Reproduces the original constants when unrated."""
    h, a = home or Rating(), away or Rating()
    h_scored = max(0.5, BASE_GOALS * HOME_SCORED * h.attack)
    h_conceded = max(0.3, BASE_GOALS * HOME_CONCEDED * h.defence)
    a_scored = max(0.5, BASE_GOALS * AWAY_SCORED * a.attack)
    a_conceded = max(0.3, BASE_GOALS * AWAY_CONCEDED * a.defence)
    return (h_scored + a_conceded) / 2.0, (a_scored + h_conceded) / 2.0


# --------------------------------------------------------------------------- #
# Dixon-Coles
# --------------------------------------------------------------------------- #
def _poisson_pmf(lam: float, n: int) -> np.ndarray:
    k = np.arange(n)
    fact = np.cumprod(np.maximum(k, 1)).astype(float)  # 0!,1!,2!,...
    return np.exp(-lam) * lam**k / fact


def dixon_coles_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = DEFAULT_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Normalised scoreline matrix; ``m[x, y]`` = P(home x, away y)."""
    m = np.outer(_poisson_pmf(lambda_home, max_goals), _poisson_pmf(lambda_away, max_goals))
    m[0, 0] *= 1.0 - lambda_home * lambda_away * rho
    m[0, 1] *= 1.0 + lambda_home * rho
    m[1, 0] *= 1.0 + lambda_away * rho
    m[1, 1] *= 1.0 - rho
    np.clip(m, 0.0, None, out=m)
    total = m.sum()
    if total > 0:
        return m / total
    return np.full((max_goals, max_goals), 1.0 / (max_goals * max_goals))


@lru_cache(maxsize=4096)
def _markets_cached(lh: float, la: float, rho: float, max_goals: int) -> tuple[float, float, float, float]:
    m = dixon_coles_matrix(lh, la, rho, max_goals)
    x, y = np.indices(m.shape)
    return (
        float(m[x >= y].sum()),         # 1X  (home win or draw)
        float(m[(y - x) <= 1].sum()),   # home +1.5 handicap
        float(m[(x + y) <= 2].sum()),   # under 2.5 goals
        # All three legs together. Home +1.5 is implied by 1X, so this is P(1X and Under 2.5).
        float(m[(x >= y) & ((x + y) <= 2)].sum()),
    )


def evaluate_triple_markets(
    lambda_home: float,
    lambda_away: float,
    rho: float = DEFAULT_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> dict:
    p1x, ph, pu, pall = _markets_cached(round(lambda_home, 4), round(lambda_away, 4), rho, max_goals)
    return {
        "min_prob": min(p1x, ph, pu),
        "all_three_prob": pall,
        "1x_prob": round(p1x * 100, 2),
        "home_plus_1_5_prob": round(ph * 100, 2),
        "under_2_5_prob": round(pu * 100, 2),
    }


# --------------------------------------------------------------------------- #
# Market-implied expected goals (from 1X2 odds)
# --------------------------------------------------------------------------- #
def devig(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    """Proportional margin removal: (p_home, p_draw, p_away)."""
    inv = [1.0 / o for o in odds]
    total = sum(inv)
    return inv[0] / total, inv[1] / total, inv[2] / total


def _grid_1x2(lh: np.ndarray, la: np.ndarray, rho: float, max_goals: int):
    """P(home win), P(draw) for every (lh[i], la[j]) pair, fully vectorised."""
    k = np.arange(max_goals)
    fact = np.cumprod(np.maximum(k, 1)).astype(float)
    ph = np.exp(-lh[:, None]) * lh[:, None] ** k / fact          # (n, K)
    pa = np.exp(-la[:, None]) * la[:, None] ** k / fact          # (m, K)
    j = ph[:, None, :, None] * pa[None, :, None, :]              # (n, m, K, K)
    LH, LA = lh[:, None], la[None, :]
    j[:, :, 0, 0] *= 1.0 - LH * LA * rho
    j[:, :, 0, 1] *= 1.0 + LH * rho
    j[:, :, 1, 0] *= 1.0 + LA * rho
    j[:, :, 1, 1] *= 1.0 - rho
    np.clip(j, 0.0, None, out=j)
    j /= j.sum(axis=(2, 3), keepdims=True)
    x, y = np.indices((max_goals, max_goals))
    return j[:, :, x > y].sum(-1), j[:, :, x == y].sum(-1)


@lru_cache(maxsize=4096)
def lambdas_from_1x2(
    odds: tuple[float, float, float],
    rho: float = DEFAULT_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
    tol: float = 0.01,
) -> tuple[float, float] | None:
    """Find (lambda_home, lambda_away) whose Dixon-Coles 1X2 matches the de-vigged odds.

    Coarse-to-fine grid in log space (no scipy needed). Returns None if the odds
    cannot be matched within ``tol`` (e.g. absurd prices).
    """
    t_home, t_draw, _ = devig(odds)
    lo, hi = math.log(0.05), math.log(6.0)
    c_h = c_a = (lo + hi) / 2
    half = (hi - lo) / 2
    best = None
    for _ in range(5):
        gh = np.clip(np.exp(np.linspace(c_h - half, c_h + half, 25)), 0.02, 8.0)
        ga = np.clip(np.exp(np.linspace(c_a - half, c_a + half, 25)), 0.02, 8.0)
        p_home, p_draw = _grid_1x2(gh, ga, rho, max_goals)
        loss = (p_home - t_home) ** 2 + (p_draw - t_draw) ** 2
        i, j = np.unravel_index(int(loss.argmin()), loss.shape)
        best = (float(gh[i]), float(ga[j]), float(loss[i, j]))
        c_h, c_a, half = math.log(gh[i]), math.log(ga[j]), half / 5
    if best is None or math.sqrt(best[2]) > tol:
        return None
    return best[0], best[1]


# --------------------------------------------------------------------------- #
# Leg profiles: which three markets make up "All3"
# --------------------------------------------------------------------------- #
# "regular": only markets every bookmaker lists for almost every game. 1X and 12 together mean Home win,
#            so All3 = Home win AND Over 1.5 goals.
# "classic": the original 1X / Home +1.5 / Under 2.5 (Home +1.5 is implied by 1X, so it adds nothing).
PROFILES = {
    "regular": (("1x", "1X"), ("12", "12"), ("o15", "Over 1.5")),
    "classic": (("1x", "1X"), ("h15", "Home +1.5"), ("u25", "Under 2.5")),
}


def _leg_masks(profile: str, n: int):
    x, y = np.indices((n, n))
    s_ = x + y
    table = {
        "1x": x >= y, "12": x != y, "o15": s_ >= 2, "h15": (y - x) <= 1, "u25": s_ <= 2,
    }
    return [table[k] for k, _ in PROFILES[profile]]


@lru_cache(maxsize=4096)
def _profile_cached(lh: float, la: float, rho: float, max_goals: int, profile: str):
    m = dixon_coles_matrix(lh, la, rho, max_goals)
    masks = _leg_masks(profile, max_goals)
    legs = tuple(float(m[mk].sum()) for mk in masks)
    joint = float(m[np.logical_and.reduce(masks)].sum())
    return legs, joint


def evaluate_profile(lambda_home: float, lambda_away: float, profile: str = "regular",
                     rho: float = DEFAULT_RHO, max_goals: int = DEFAULT_MAX_GOALS) -> dict:
    """Probabilities of the three legs and of all three together for a profile."""
    legs, joint = _profile_cached(round(lambda_home, 4), round(lambda_away, 4), rho, max_goals, profile)
    keys = [k for k, _ in PROFILES[profile]]
    return {"legs": dict(zip(keys, legs)), "all3": joint, "min_leg": min(legs)}


def leg_outcomes(profile: str, hg: int, ag: int) -> dict:
    """What actually happened for each leg (and all three) of a profile."""
    res = {
        "1x": hg >= ag, "12": hg != ag, "o15": hg + ag >= 2, "h15": ag - hg <= 1, "u25": hg + ag <= 2,
    }
    out = {k: bool(res[k]) for k, _ in PROFILES[profile]}
    out["all3"] = all(out.values())
    return out
