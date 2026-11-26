"""Monotone (isotonic) calibration of the scanner's leg probabilities.

Learned from past games: maps a predicted probability to the rate at which that
leg actually landed, so a common line on the calibrated numbers means the same
*realised* percent for every leg. numpy only; stored as piecewise-linear knots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MARKETS = ("1x", "h15", "u25", "all3")


def fit_isotonic(pred: np.ndarray, hit: np.ndarray, n_bins: int = 25, smooth: float = 0.0) -> dict:
    """Pool-adjacent-violators on equal-count bins -> {"x": [...], "y": [...]} knots."""
    order = np.argsort(pred)
    p, y = np.asarray(pred, float)[order], np.asarray(hit, float)[order]
    edges = np.linspace(0, len(p), n_bins + 1).astype(int)
    xs = [p[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]
    ys = [y[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]
    ws = [b - a for a, b in zip(edges[:-1], edges[1:]) if b > a]
    blocks = [[xs[i], ys[i], ws[i], 1] for i in range(len(xs))]  # x_sum-weighted below
    out: list[list[float]] = []
    for blk in blocks:  # PAV on block means
        out.append(blk)
        while len(out) > 1 and out[-2][1] > out[-1][1]:
            b2, b1 = out.pop(), out.pop()
            w = b1[2] + b2[2]
            out.append([(b1[0] * b1[2] + b2[0] * b2[2]) / w, (b1[1] * b1[2] + b2[1] * b2[2]) / w, w, b1[3] + b2[3]])
    return {"x": [round(b[0], 5) for b in out], "y": [round(b[1], 5) for b in out]}


def apply_map(knots: dict | None, p: float) -> float:
    if not knots or len(knots["x"]) < 2:
        return p
    return float(np.interp(p, knots["x"], knots["y"]))


class Calibration:
    def __init__(self, maps: dict[str, dict] | None = None, note: str = ""):
        self.maps = maps or {}
        self.note = note

    def __bool__(self) -> bool:
        return bool(self.maps)

    def apply(self, market: str, p: float) -> float:
        return apply_map(self.maps.get(market), p)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"_comment": self.note, **self.maps}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        p = Path(path)
        if not p.is_file():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        note = data.pop("_comment", "")
        return cls({k: v for k, v in data.items() if k in MARKETS}, note)
