import os
import re
import numpy as np
from scipy.stats import poisson
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Dixon-Coles Flashscore Scanner",
    description="Parser and probability ranking pipeline trained for Flashscore mobile fixtures.",
    version="3.4.0",
)


class ScanRequest(BaseModel):
    raw_text: str
    threshold: float = 0.695
    min_threshold: float = 0.20
    max_xg: float = 3.25


def clean_and_parse_flashscore_fixtures(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        return {
            "cleaned_text": "",
            "leagues_count": 0,
            "fixtures_count": 0,
            "fixtures": [],
        }

    junk_keywords = [
        "bet on football", "refresh now", "flashscore", "livescore",
        "basketball", "tennis", "ice hockey", "field hockey", "handball",
        "volleyball", "baseball", "am. football", "rugby union", "cricket",
        "more sports", "top", "m.flashscore", "provided live", "mobile version",
        "advertisement", "all games", "live", "finished", "odds", "draws",
        "football |", "gamble responsibly", "copyright"
    ]

    lines = raw_text.split("\n")
    cleaned_schedule = []
    structured_fixtures = []
    current_comp = "GLOBAL LEAGUES"
    leagues_set = set()

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        lower_line = line_str.lower()
        if any(j in lower_line for j in junk_keywords) and not re.search(r"\d{2}:\d{2}", line_str):
            continue

        # Detect Flashscore competition headers (e.g., "NIGERIA: NPFL Standings" or "AFRICA: Africa Cup...")
        if ":" in line_str and " - " not in line_str and not re.search(r"^\d{2}:\d{2}", line_str):
            current_comp = line_str.replace("Standings", "").replace("Table", "").strip()
            leagues_set.add(current_comp)
            cleaned_schedule.append(f"\n--- {current_comp.upper()} ---")
            continue

        # Catch lines without time or colon that act as headers
        if ":" not in line_str[:5] and " - " not in line_str and len(line_str) > 3 and not re.search(r"\d+-\d+", line_str):
            current_comp = line_str.replace("Standings", "").replace("Table", "").strip()
            leagues_set.add(current_comp)
            cleaned_schedule.append(f"\n--- {current_comp.upper()} ---")
            continue

        time_str = "00:00"
        match_part = line_str

        # Flexible international time match (e.g., "16:00Enyimba - Ikorodu City" or "14:00Zambia U23...")
        time_match = re.search(r"\b(\d{2}:\d{2})\b", line_str)
        if time_match:
            time_str = time_match.group(1)
            parts_after_time = line_str.split(time_str, 1)
            if len(parts_after_time) > 1:
                match_part = parts_after_time[1].strip()

        # Remove match status prefixes or live scores (e.g. "32'", "Postponed", live score "1-0")
        match_part = re.sub(
            r"^(Postponed|Cancelled|\d+['\+]?)\s*", "", match_part, flags=re.IGNORECASE
        ).strip()

        if " - " in match_part:
            parts = match_part.split(" - ", 1)
            home_team = parts[0].strip()
            away_team = re.sub(r"\s*-\s*$", "", parts[1]).strip()
            # Clean trailing scores if present (e.g., "Team A 1-0" -> strip scores)
            away_team = re.split(r"\s+\d+-\d+", away_team)[0].strip()
            away_team = re.sub(r"\s+\d+$", "", away_team).strip()

            if home_team and away_team and home_team.lower() != away_team.lower():
                cleaned_line = f"{time_str} | {home_team} vs {away_team}"
                cleaned_schedule.append(cleaned_line)
                structured_fixtures.append(
                    {
                        "time": time_str,
                        "competition": current_comp.upper(),
                        "home": home_team,
                        "away": away_team,
                    }
                )

    return {
        "cleaned_text": "\n".join(cleaned_schedule).strip(),
        "leagues_count": len(leagues_set),
        "fixtures_count": len(structured_fixtures),
        "fixtures": structured_fixtures,
    }


def derive_team_parameters(is_home: bool) -> tuple[float, float]:
    base_scored = 1.35
    base_conceded = 1.35
    venue_mult_scored = 1.08 if is_home else 0.92
    venue_mult_conceded = 0.90 if is_home else 1.12
    scored = max(0.5, base_scored * venue_mult_scored)
    conceded = max(0.3, base_conceded * venue_mult_conceded)
    return scored, conceded


def compute_dixon_coles_matrix(
    lambda_home: float, lambda_away: float, rho: float = -0.11, max_goals: int = 8
) -> np.ndarray:
    matrix = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            base_prob = poisson.pmf(x, lambda_home) * poisson.pmf(y, lambda_away)
            if x == 0 and y == 0:
                tau = 1.0 - (lambda_home * lambda_away * rho)
            elif x == 0 and y == 1:
                tau = 1.0 + (lambda_home * rho)
            elif x == 1 and y == 0:
                tau = 1.0 + (lambda_away * rho)
            elif x == 1 and y == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            matrix[x, y] = max(0.0, base_prob * tau)

    total = np.sum(matrix)
    if total > 0:
        matrix /= total
    else:
        matrix = np.ones((max_goals, max_goals)) / (max_goals * max_goals)

    return matrix


def evaluate_triple_markets(dc_matrix: np.ndarray) -> dict:
    max_goals = dc_matrix.shape[0]
    prob_1x = 0.0
    prob_home_plus_1_5 = 0.0
    prob_under_2_5 = 0.0

    for x in range(max_goals):
        for y in range(max_goals):
            p = dc_matrix[x, y]
            if x >= y:
                prob_1x += p
            if (y - x) <= 1:
                prob_home_plus_1_5 += p
            if (x + y) <= 2:
                prob_under_2_5 += p

    min_prob = min(prob_1x, prob_home_plus_1_5, prob_under_2_5)

    return {
        "min_prob": min_prob,
        "1x_prob": round(prob_1x * 100, 2),
        "home_plus_1_5_prob": round(prob_home_plus_1_5 * 100, 2),
        "under_2_5_prob": round(prob_under_2_5 * 100, 2),
    }


def run_pipeline(raw_text: str, threshold: float = 0.695, min_threshold: float = 0.20, max_xg: float = 3.25) -> dict:
    cleaned_data = clean_and_parse_flashscore_fixtures(raw_text)
    scanned_results = []
    copyable_lines = []

    for item in cleaned_data["fixtures"]:
        h_sc, h_co = derive_team_parameters(is_home=True)
        a_sc, a_co = derive_team_parameters(is_home=False)

        lambda_home = (h_sc + a_co) / 2.0
        lambda_away = (a_sc + h_co) / 2.0
        lambda_total = lambda_home + lambda_away

        if lambda_total > max_xg:
            continue

        dc_matrix = compute_dixon_coles_matrix(lambda_home, lambda_away)
        metrics = evaluate_triple_markets(dc_matrix)

        if metrics["min_prob"] < min_threshold:
            continue

        if metrics["min_prob"] >= threshold:
            status_tag = "★ QUALIFIED"
        else:
            status_tag = "◆ MODERATE"

        row = {
            "time": item["time"],
            "competition": item["competition"],
            "match": f"{item['home']} vs {item['away']}",
            "xg": round(lambda_total, 2),
            "score": metrics["min_prob"],
            "prob_1x": f"{metrics['1x_prob']}%",
            "prob_h_plus_1_5": f"{metrics['home_plus_1_5_prob']}%",
            "prob_u2_5": f"{metrics['under_2_5_prob']}%",
            "status": status_tag,
        }
        scanned_results.append(row)

    scanned_results.sort(key=lambda x: x["score"], reverse=True)

    for row in scanned_results:
        copyable_lines.append(
            f"{row['time']} | {row['competition']} | {row['match']} | "
            f"xG:{row['xg']} | MinProb:{round(row['score']*100, 1)}% | "
            f"1X:{row['prob_1x']} | H+1.5:{row['prob_h_plus_1_5']} | U2.5:{row['prob_u2_5']} | {row['status']}"
        )

    return {
        "summary": {
            "total_leagues": cleaned_data["leagues_count"],
            "total_fixtures": cleaned_data["fixtures_count"],
            "ranked_count": len(scanned_results),
        },
        "cleaned_schedule": cleaned_data["cleaned_text"],
        "ranked_fixtures": scanned_results,
        "copyable_block": "\n".join(copyable_lines),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/scan")
def scan_api(payload: ScanRequest):
    result = run_pipeline(
        raw_text=payload.raw_text,
        threshold=payload.threshold,
        min_threshold=payload.min_threshold,
        max_xg=payload.max_xg,
    )
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
def mobile_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Flashscore Dixon-Coles Scanner</title>
        <style>
            body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; padding: 16px; margin: 0; }
            .container { max-width: 720px; margin: auto; }
            h2 { font-size: 1.25rem; margin-bottom: 8px; color: #38bdf8; }
            textarea { width: 100%; height: 200px; background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 8px; padding: 10px; box-sizing: border-box; font-size: 13px; font-family: monospace; }
            button { width: 100%; padding: 12px; margin-top: 10px; background: #0284c7; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 16px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Flashscore Mobile Fixture Scanner</h2>
            <form method="post" action="/ui-scan">
                <textarea name="raw_text" placeholder="Paste raw Flashscore mobile copy here..."></textarea>
                <button type="submit">Run Flashscore Pipeline</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/ui-scan", response_class=HTMLResponse)
def mobile_ui_scan(raw_text: str = Form(...)):
    res = run_pipeline(raw_text)
    s = res["summary"]
    copy_text = res["copyable_block"] or "No fixtures met the 20%-69%+ probability band."
    cleaned_preview = res["cleaned_schedule"] or "No valid fixtures parsed."

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Scan Results</title>
        <style>
            body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; padding: 16px; margin: 0; }}
            .container {{ max-width: 720px; margin: auto; }}
            h3 {{ color: #38bdf8; margin-top: 18px; margin-bottom: 6px; font-size: 1.05rem; }}
            .stats {{ display: flex; justify-content: space-between; background: #1e293b; padding: 12px; border-radius: 8px; font-weight: bold; font-size: 14px; border: 1px solid #334155; }}
            pre {{ background: #1e293b; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 13px; border: 1px solid #334155; white-space: pre-wrap; font-family: monospace; }}
            button {{ width: 100%; padding: 12px; margin-top: 8px; background: #10b981; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 15px; cursor: pointer; }}
            a {{ display: block; text-align: center; margin-top: 16px; color: #38bdf8; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="stats">
                <span>Leagues: {s['total_leagues']}</span>
                <span>Fixtures: {s['total_fixtures']}</span>
                <span>Ranked (20%+): {s['ranked_count']}</span>
            </div>

            <h3>Ranked Copyable Results (69% down to 20%)</h3>
            <pre id="copyBlock">{copy_text}</pre>
            <button onclick="navigator.clipboard.writeText(document.getElementById('copyBlock').innerText)">Copy Results</button>

            <h3>Cleaned Schedule</h3>
            <pre>{cleaned_preview}</pre>

            <a href="/">← Back</a>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
