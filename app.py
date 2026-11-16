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
from scanner.model import Ratings
from scanner.pipeline import (
    CALIBRATED_THRESHOLD,
    DEFAULT_MAX_XG,
    DEFAULT_MIN_THRESHOLD,
    DEFAULT_THRESHOLD,
    run_pipeline,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_INPUT_CHARS = 200_000
RATINGS = Ratings.from_json(os.getenv("RATINGS_PATH", "ratings.json"))
PICKS_PATH = os.getenv("PICKS_PATH", "picks.csv")
CALIBRATION = Calibration.load(os.getenv("CALIBRATION_PATH", "calibration.json"))
CALIBRATION_ODDS = Calibration.load(os.getenv("CALIBRATION_ODDS_PATH", "calibration_odds.json"))
FORM_THRESHOLD = CALIBRATED_THRESHOLD if (CALIBRATION or CALIBRATION_ODDS) else DEFAULT_THRESHOLD

app = FastAPI(
    title="Dixon-Coles Flashscore Scanner",
    description="Parser and probability ranking pipeline trained for Flashscore mobile fixtures.",
    version="4.0.0",
)


class ScanRequest(BaseModel):
    raw_text: str = Field(max_length=MAX_INPUT_CHARS)
    threshold: float | None = Field(None, ge=0, le=1)  # default depends on rank_by
    rank_by: Literal["min", "all3"] = "min"
    min_threshold: float = Field(DEFAULT_MIN_THRESHOLD, ge=0, le=1)
    max_xg: float = Field(DEFAULT_MAX_XG, gt=0, le=10)
    include_unrated: bool = False
    use_calibration: bool = True
    log_picks: bool = False
    notify_telegram: bool = False

    @model_validator(mode="after")
    def _min_below_threshold(self):
        if self.threshold is not None and self.min_threshold > self.threshold:
            raise ValueError("min_threshold must not exceed threshold")
        return self


@app.get("/health")
def health_check():
    return {"status": "ok", "ratings_loaded": len(RATINGS), "calibration_loaded": bool(CALIBRATION), "calibration_odds_loaded": bool(CALIBRATION_ODDS)}


@app.post("/scan")
def scan_api(payload: ScanRequest):
    result = run_pipeline(
        payload.raw_text,
        threshold=payload.threshold,
        min_threshold=payload.min_threshold,
        max_xg=payload.max_xg,
        ratings=RATINGS,
        include_unrated=payload.include_unrated,
        rank_by=payload.rank_by,
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
):
    if len(raw_text) > MAX_INPUT_CHARS:
        return HTMLResponse(ui.render_error(f"Input too large (max {MAX_INPUT_CHARS:,} characters)."), status_code=413)
    if not (0 <= min_threshold <= threshold <= 1) or not (0 < max_xg <= 10):
        return HTMLResponse(ui.render_error("Thresholds must satisfy 0 ≤ Show ≤ Qualify ≤ 1, and 0 < Max xG ≤ 10."), status_code=422)
    res = run_pipeline(raw_text, threshold, min_threshold, max_xg, ratings=RATINGS, include_unrated=include_unrated, calibration=CALIBRATION, calibration_odds=CALIBRATION_ODDS)
    if log_picks:
        res["logged"] = picklog.append_picks(res["ranked_fixtures"], PICKS_PATH)
    return ui.render_results(res, threshold, min_threshold)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
