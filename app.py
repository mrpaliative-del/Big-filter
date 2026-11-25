import logging
import os

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Literal

from pydantic import BaseModel, Field, model_validator

import picklog
import ui
from bot import send_telegram_alert
from scanner.calibration import Calibration
from scanner.formmodel import FormModel, FormPredictor, load_history
from scanner.model import Ratings
from scanner.combined import run_both
from scanner.draw import DrawPredictor
from scanner.feeds import FIXTURES_URL, OddsFeed, fetch_text, parse_football_data_csv, update_history
from scanner.selection import run_selection
from scanner.pipeline import (
    REGULAR_THRESHOLD,
    DEFAULT_MAX_XG,
    DEFAULT_PRICE_FLOOR,
    DEFAULT_MIN_THRESHOLD,
    DEFAULT_THRESHOLD,
    run_pipeline,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_INPUT_CHARS = 200_000
RATINGS = Ratings.from_json(os.getenv("RATINGS_PATH", "ratings.json"))
PICKS_PATH = os.getenv("PICKS_PATH", "picks.csv")
CALIBRATION = Calibration.load(os.getenv("CALIBRATION_PATH", "calibration.json"))
CAL_SELECT = Calibration.load(os.getenv("CALIBRATION_SELECT_PATH", "calibration_select.json"))
CALIBRATION_ODDS = Calibration.load(os.getenv("CALIBRATION_ODDS_PATH", "calibration_odds.json"))
FORM_THRESHOLD = 0.20   # statistics-only default: All3 from 20% upward, ranked by tier


def _load_form():
    """Trained technical-analysis model + results history (needs form_model.json and the history CSVs)."""
    try:
        model = FormModel.load(os.getenv("FORM_MODEL_PATH", "form_model.json"))
        paths = [p for p in os.getenv("HISTORY_PATHS", "data/fd_2025-26_2026-27.csv").split(",") if p and os.path.isfile(p)]
        if model is None or not paths:
            return None
        aliases = {}
        if os.path.isfile(os.getenv("ALIASES_PATH", "aliases.json")):
            import json
            aliases = json.load(open(os.getenv("ALIASES_PATH", "aliases.json"), encoding="utf-8"))
        return FormPredictor(model, load_history(paths), aliases)
    except Exception as exc:  # the scanner still works from odds and ratings
        logging.getLogger(__name__).warning("form model not loaded: %s", exc)
        return None


FORM = _load_form()


def _load_feed():
    """Upcoming fixtures with odds (open football-data.co.uk feed), from the cached file if present."""
    try:
        path = os.getenv("FIXTURES_CACHE", "data/fixtures_cache.csv")
        if os.path.isfile(path):
            return OddsFeed.from_frame(parse_football_data_csv(open(path, encoding="utf-8-sig").read()))
    except Exception as exc:
        logging.getLogger(__name__).warning("fixtures feed not loaded: %s", exc)
    return OddsFeed()


ODDS_FEED = _load_feed()


def _load_draw():
    """The draw scanner: trained league models (train_draw_models.py) + the current results history for team form."""
    try:
        models = DrawPredictor.load_models(os.getenv("DRAW_MODELS_PATH", "draw_models.pkl"))
        hist = os.getenv("HISTORY_PATHS", "data/fd_2025-26_2026-27.csv").split(",")[0]
        if not models or not os.path.isfile(hist):
            return None
        import json
        import pandas as pd
        aliases = json.load(open(os.getenv("ALIASES_PATH", "aliases.json"), encoding="utf-8")) if os.path.isfile(os.getenv("ALIASES_PATH", "aliases.json")) else {}
        return DrawPredictor(models, pd.read_csv(hist, parse_dates=["Date"]), aliases)
    except Exception as exc:
        logging.getLogger(__name__).warning("draw scanner not loaded: %s", exc)
        return None


DRAW = _load_draw()


def _lookup(use_feed: bool = True):
    return ODDS_FEED.lookup if (use_feed and len(ODDS_FEED)) else None

app = FastAPI(
    title="Dixon-Coles Flashscore Scanner",
    description="Parser and probability ranking pipeline trained for Flashscore mobile fixtures.",
    version="4.0.0",
)


class ScanRequest(BaseModel):
    raw_text: str = Field(max_length=MAX_INPUT_CHARS)
    threshold: float | None = Field(None, ge=0, le=1)  # default depends on profile / rank_by
    profile: Literal["regular", "classic"] = "regular"
    rank_by: Literal["min", "all3"] | None = None
    mode: Literal["all", "both", "classic", "regular", "triple", "single", "draw"] = "all"
    use_draw: bool = True
    qualify_by: Literal["all3_floor", "band", "all3"] = "all3_floor"
    all3_min: float | None = Field(None, ge=0, le=1)   # None: 20% in statistics-only mode, 50% in odds-first mode
    selection_source: Literal["stats", "odds_first"] = "stats"
    price_floor: float = Field(DEFAULT_PRICE_FLOOR, ge=1.0, le=10)
    show_only_qualified: bool = False
    use_feed: bool = True
    price_min: float = Field(1.30, gt=1.0, le=10)
    price_max: float = Field(1.35, gt=1.0, le=10)
    min_threshold: float = Field(DEFAULT_MIN_THRESHOLD, ge=0, le=1)
    max_xg: float = Field(DEFAULT_MAX_XG, gt=0, le=10)
    include_unrated: bool = False
    use_calibration: bool = True
    use_form: bool = True
    log_picks: bool = False
    notify_telegram: bool = False

    @model_validator(mode="after")
    def _min_below_threshold(self):
        if self.price_min > self.price_max:
            raise ValueError("price_min must not exceed price_max")
        if self.threshold is not None and self.min_threshold > self.threshold:
            raise ValueError("min_threshold must not exceed threshold")
        return self


@app.post("/refresh-data")
def refresh_data():
    """Pull the latest results and fixtures odds from the open feed (needs internet), then reload the models' inputs."""
    global FORM, ODDS_FEED, DRAW
    history = os.getenv("HISTORY_PATHS", "data/fd_2025-26_2026-27.csv").split(",")[0]
    cache = os.getenv("FIXTURES_CACHE", "data/fixtures_cache.csv")
    try:
        added = update_history(history)
        text = fetch_text(FIXTURES_URL)
        open(cache, "w", encoding="utf-8").write(text)
    except Exception as exc:
        return JSONResponse(status_code=502, content={"ok": False, "error": str(exc)})
    FORM, ODDS_FEED, DRAW = _load_form(), _load_feed(), _load_draw()
    return {"ok": True, "new_games": added, "fixtures_with_odds": len(ODDS_FEED)}


@app.get("/health")
def health_check():
    return {"status": "ok", "ratings_loaded": len(RATINGS), "calibration_loaded": bool(CALIBRATION), "calibration_odds_loaded": bool(CALIBRATION_ODDS), "form_model_loaded": FORM is not None, "fixtures_feed_rows": len(ODDS_FEED), "draw_scanner_loaded": DRAW is not None}


@app.post("/scan")
def scan_api(payload: ScanRequest):
    if payload.mode in ("all", "both", "draw"):
        scanners = {"all": ("classic", "regular", "draw"), "both": ("classic", "regular"), "draw": ("draw",)}[payload.mode]
        result = run_both(payload.raw_text, draw=DRAW if payload.use_draw else None, scanners=scanners, ratings=RATINGS, calibration=CALIBRATION if payload.use_calibration else None,
                          calibration_odds=CALIBRATION_ODDS if payload.use_calibration else None, form=FORM if payload.use_form else None,
                          odds_lookup=_lookup(payload.use_feed), min_threshold=payload.min_threshold, max_xg=payload.max_xg,
                          include_unrated=payload.include_unrated, all3_min=payload.all3_min, price_floor=payload.price_floor,
                          show_only_qualified=payload.show_only_qualified, selection_source=payload.selection_source)
        if payload.log_picks:
            result["logged"] = picklog.append_picks(result["panels"], PICKS_PATH)
        return JSONResponse(content=result)
    if payload.mode == "single":
        result = run_selection(payload.raw_text, (payload.price_min, payload.price_max),
                               calibration=CAL_SELECT if payload.use_calibration else None, form=FORM if payload.use_form else None)
        if payload.log_picks:
            result["logged"] = picklog.append_picks(result["ranked_fixtures"], PICKS_PATH)
        return JSONResponse(content=result)
    prof = "classic" if payload.mode == "classic" else payload.profile if payload.mode == "regular" or payload.mode == "triple" else payload.profile
    if payload.mode in ("regular", "triple"):
        prof = "regular"
    result = run_pipeline(
        payload.raw_text,
        threshold=payload.threshold if prof == "classic" or payload.threshold is not None else payload.all3_min,
        min_threshold=payload.min_threshold,
        max_xg=payload.max_xg,
        ratings=RATINGS,
        include_unrated=payload.include_unrated,
        rank_by=payload.rank_by,
        profile=prof,
        qualify_by=payload.qualify_by,
        price_floor=payload.price_floor,
        odds_lookup=_lookup(payload.use_feed),
        price_band=(payload.price_min, payload.price_max),
        form=FORM if payload.use_form else None,
        calibration=CALIBRATION if payload.use_calibration else None,
        calibration_odds=CALIBRATION_ODDS if payload.use_calibration else None,
    )
    if payload.log_picks:
        result["logged"] = picklog.append_picks(result["ranked_fixtures"], PICKS_PATH)
    if payload.notify_telegram:
        result["telegram"] = send_telegram_alert(result["copyable_block"])
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
def mobile_ui():
    return ui.render_form(FORM_THRESHOLD, DEFAULT_MIN_THRESHOLD, DEFAULT_MAX_XG)


@app.post("/ui-scan", response_class=HTMLResponse)
def mobile_ui_scan(
    raw_text: str = Form(""),
    threshold: float = Form(FORM_THRESHOLD),
    min_threshold: float = Form(DEFAULT_MIN_THRESHOLD),
    max_xg: float = Form(DEFAULT_MAX_XG),
    include_unrated: bool = Form(False),
    log_picks: bool = Form(False),
    mode: str = Form("all"),
    selection_source: str = Form("stats"),
    price_floor: float = Form(1.30),
    show_only: bool = Form(False),
    use_feed: bool = Form(True),
    qualify_by: str = Form("band"),
    price_min: float = Form(1.30),
    price_max: float = Form(1.35),
):
    if len(raw_text) > MAX_INPUT_CHARS:
        return HTMLResponse(ui.render_error(f"Input too large (max {MAX_INPUT_CHARS:,} characters)."), status_code=413)
    if not (0 <= min_threshold <= threshold <= 1) or not (0 < max_xg <= 10):
        return HTMLResponse(ui.render_error("Thresholds must satisfy 0 ≤ Show ≤ Qualify ≤ 1, and 0 < Max xG ≤ 10."), status_code=422)
    if mode in ("all", "both", "draw"):
        scanners = {"all": ("classic", "regular", "draw"), "both": ("classic", "regular"), "draw": ("draw",)}[mode]
        res = run_both(raw_text, draw=DRAW, scanners=scanners, ratings=RATINGS, calibration=CALIBRATION, calibration_odds=CALIBRATION_ODDS, form=FORM, odds_lookup=_lookup(use_feed),
                       min_threshold=min_threshold, max_xg=max_xg, include_unrated=include_unrated, all3_min=threshold, price_floor=price_floor,
                       show_only_qualified=show_only, selection_source=selection_source if selection_source in ("stats", "odds_first") else "stats")
        if log_picks:
            res["logged"] = picklog.append_picks(res["panels"], PICKS_PATH)
        return ui.render_results(res, threshold, min_threshold)
    if mode == "single":
        res = run_selection(raw_text, (price_min, price_max), calibration=CAL_SELECT, form=FORM)
        if log_picks:
            res["logged"] = picklog.append_picks(res["ranked_fixtures"], PICKS_PATH)
        return ui.render_results(res, price_min, min_threshold)
    prof = "classic" if mode == "classic" else "regular"
    res = run_pipeline(raw_text, None if prof == "classic" else threshold, min_threshold, max_xg, ratings=RATINGS, include_unrated=include_unrated, calibration=CALIBRATION, calibration_odds=CALIBRATION_ODDS, form=FORM,
                       profile=prof, qualify_by=qualify_by if qualify_by in ("band", "all3", "all3_floor") else "all3_floor",
                       price_band=(price_min, price_max), price_floor=price_floor, odds_lookup=_lookup(use_feed))
    if log_picks:
        res["logged"] = picklog.append_picks(res["ranked_fixtures"], PICKS_PATH)
    return ui.render_results(res, threshold, min_threshold)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
