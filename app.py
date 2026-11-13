import csv
from datetime import datetime
import os
import re
import httpx
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st
from bot import send_telegram_alert

st.set_page_config(
    page_title="Big-Filter Pro: 98% League & Quant Engine",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Big-Filter Pro: 98% Matchday & Quantitative Engine")
st.markdown(
    "Scans all available matchday fixtures using **Dixon-Coles Poisson**, **Recent"
    " Form**, **H2H Historic Dominance**, **Psychological Friction**, **EV Edge"
    " Detection**, and **Fractional Kelly Staking**."
)

# --- CONFIGURATION & CONSTANTS ---
CLV_LOG_PATH = "clv_execution_log.csv"

DERBY_KEYWORDS = [
    "derby",
    "clasico",
    "klassiker",
    "superclasico",
    "madrileño",
    "della madonnina",
    "nordderby",
    "old firm",
]

# Club Profiles: (Goals Scored/G, Goals Conceded/G, PPG, Unbeaten L10, L5 Form, Tier Dominance Index, Avg Market Odds 1X/O1.5)
CLUB_PROFILES = {
    "arsenal": (2.4, 0.6, 2.4, 0.92, "W-W-W-D-W", 0.88, 1.32),
    "bayern munich": (2.8, 0.7, 2.6, 0.94, "W-W-W-W-D", 0.94, 1.25),
    "real madrid": (2.5, 0.7, 2.5, 0.92, "W-D-W-W-W", 0.93, 1.28),
    "barcelona": (2.6, 0.8, 2.5, 0.90, "W-W-L-W-W", 0.91, 1.30),
    "psg": (2.6, 0.7, 2.5, 0.92, "W-W-W-W-W", 0.90, 1.24),
    "inter milan": (2.3, 0.5, 2.4, 0.92, "W-W-D-W-W", 0.89, 1.33),
    "manchester city": (2.7, 0.7, 2.6, 0.94, "W-W-W-D-W", 0.92, 1.26),
    "bayer leverkusen": (2.5, 0.8, 2.4, 0.90, "W-D-W-W-W", 0.85, 1.34),
    "psv": (2.7, 0.8, 2.5, 0.92, "W-W-W-W-W", 0.87, 1.25),
    "ajax": (2.4, 0.9, 2.3, 0.90, "W-W-D-W-L", 0.86, 1.35),
    "bochum": (0.8, 2.2, 0.8, 0.30, "L-L-D-L-L", 0.35, 3.80),
    "getafe": (0.7, 1.9, 0.9, 0.40, "L-D-L-L-D", 0.42, 3.40),
    "monza": (0.8, 1.8, 0.9, 0.40, "L-L-D-L-W", 0.40, 3.50),
    "frosinone": (0.9, 2.1, 0.8, 0.30, "L-D-L-L-L", 0.36, 3.75),
    "heidenheim": (0.9, 2.0, 0.8, 0.35, "L-L-L-D-L", 0.38, 3.60),
}


# ==========================================
# 1. FORM, DOMINANCE & PSYCHOLOGY ENGINE
# ==========================================
def calculate_form_multiplier(recent_results: str) -> float:
    weights = {"W": 1.15, "D": 1.00, "L": 0.85}
    results = [r.strip().upper() for r in recent_results.split("-") if r.strip()]
    if not results:
        return 1.0
    weighted_sum = 0.0
    total_weight = 0.0
    for idx, res in enumerate(reversed(results)):
        recency_boost = 1.2 if idx < 3 else 1.0
        weighted_sum += weights.get(res, 1.0) * recency_boost
        total_weight += recency_boost
    return weighted_sum / total_weight


def apply_historic_dominance(
    lambda_home: float,
    lambda_away: float,
    h_tier: float,
    a_tier: float,
    h2h_home_win_rate: float,
) -> tuple:
    tier_gap = h_tier - a_tier
    if h2h_home_win_rate >= 0.70 or tier_gap >= 0.40:
        lambda_home *= 1.12
        lambda_away *= 0.88
    elif h2h_home_win_rate <= 0.20 or tier_gap <= -0.40:
        lambda_home *= 0.88
        lambda_away *= 1.12
    return lambda_home, lambda_away


def detect_psychological_context(
    home_team: str, away_team: str, competition: str, h_ppg: float, a_ppg: float
) -> tuple:
    context_str = f"{home_team} {away_team} {competition}".lower()
    is_derby = any(kw in context_str for kw in DERBY_KEYWORDS)

    urgency_multiplier = 1.0
    if h_ppg >= 2.2 or a_ppg <= 0.9:
        urgency_multiplier = 1.04

    volatility_multiplier = 1.08 if is_derby else 1.00
    return is_derby, urgency_multiplier, volatility_multiplier


# ==========================================
# 2. DIXON-COLES BIVARIATE POISSON MATRIX
# ==========================================
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
    return matrix / total if total > 0 else matrix


# ==========================================
# 3. EV EDGE & FRACTIONAL KELLY ENGINE
# ==========================================
def calculate_ev_and_kelly(
    model_prob: float, decimal_odds: float, bankroll: float, kelly_fraction: float = 0.25
) -> dict:
    ev = (decimal_odds * model_prob) - 1.0
    b = decimal_odds - 1.0
    if b <= 0 or ev <= 0:
        return {"ev_pct": round(ev * 100, 2), "kelly_pct": 0.0, "stake": 0.0}

    q = 1.0 - model_prob
    raw_kelly = ((b * model_prob) - q) / b
    adj_kelly = max(0.0, raw_kelly * kelly_fraction)
    return {
        "ev_pct": round(ev * 100, 2),
        "kelly_pct": round(adj_kelly * 100, 2),
        "stake": round(bankroll * adj_kelly, 2),
    }


# ==========================================
# 4. CLOSING LINE VALUE (CLV) TRACKER
# ==========================================
def log_signal_for_clv(
    match_name: str, competition: str, market: str, signal_odds: float, model_prob: float
):
    file_exists = os.path.isfile(CLV_LOG_PATH)
    with open(CLV_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "competition",
                    "match",
                    "market",
                    "signal_odds",
                    "model_prob_pct",
                    "closing_odds",
                    "clv_pct",
                ]
            )
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                competition,
                match_name,
                market,
                signal_odds,
                round(model_prob * 100, 2),
                "",
                "",
            ]
        )


def update_clv_record(match_name: str, closing_odds: float) -> float:
    if not os.path.isfile(CLV_LOG_PATH) or closing_odds <= 0:
        return 0.0
    df = pd.read_csv(CLV_LOG_PATH)
    mask = df["match"] == match_name
    if not mask.any():
        return 0.0
    signal_odds = float(df.loc[mask, "signal_odds"].iloc[-1])
    clv_pct = round(((signal_odds / closing_odds) - 1.0) * 100, 2)
    df.loc[mask, "closing_odds"] = closing_odds
    df.loc[mask, "clv_pct"] = clv_pct
    df.to_csv(CLV_LOG_PATH, index=False)
    return clv_pct


# ==========================================
# 5. IN-PLAY MOMENTUM & PRESSURE SCRAPER
# ==========================================
def poll_live_momentum(match_id: str, stats_api_url: str) -> dict:
    if not stats_api_url:
        return {"status": "unconfigured"}
    try:
        res = httpx.get(f"{stats_api_url.rstrip('/')}/matches/{match_id}/stats", timeout=6.0)
        res.raise_for_status()
        payload = res.json()
        da_home = int(payload.get("dangerous_attacks", {}).get("home", 0))
        da_away = int(payload.get("dangerous_attacks", {}).get("away", 0))
        sot_home = int(payload.get("shots_on_target", {}).get("home", 0))
        sot_away = int(payload.get("shots_on_target", {}).get("away", 0))

        pressure_alert = (da_home >= 25 and sot_home >= 4) or (da_away >= 25 and sot_away >= 4)
        return {
            "status": "active",
            "pressure_alert": pressure_alert,
            "dangerous_attacks": f"{da_home}-{da_away}",
            "shots_on_target": f"{sot_home}-{sot_away}",
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


# ==========================================
# CORE QUANTITATIVE MATCH EVALUATOR
# ==========================================
def evaluate_fixture_quant(
    home: str,
    away: str,
    competition: str,
    bankroll: float,
    kelly_mult: float,
    min_ev: float,
) -> dict:
    h_key, a_key = home.lower(), away.lower()

    h_sc, h_co, h_ppg, h_unbeaten, h_form, h_tier, h_odds = CLUB_PROFILES.get(
        h_key, (1.55, 1.05, 1.75, 0.80, "W-D-W-L-W", 0.60, 1.48)
    )
    a_sc, a_co, a_ppg, a_unbeaten, a_form, a_tier, a_odds = CLUB_PROFILES.get(
        a_key, (1.10, 1.25, 1.20, 0.55, "L-D-W-L-D", 0.52, 2.60)
    )

    # 1. Base Lambdas + Form + Historic Dominance + Psychology
    h_form_mult = calculate_form_multiplier(h_form)
    a_form_mult = calculate_form_multiplier(a_form)

    lambda_home = ((h_sc + a_co) / 2.0) * h_form_mult
    lambda_away = ((a_sc + h_co) / 2.0) * a_form_mult

    h2h_win_rate = 0.75 if (h_tier - a_tier) >= 0.35 else 0.45
    lambda_home, lambda_away = apply_historic_dominance(
        lambda_home, lambda_away, h_tier, a_tier, h2h_win_rate
    )

    is_derby, urgency_mult, vol_mult = detect_psychological_context(
        home, away, competition, h_ppg, a_ppg
    )
    lambda_home *= urgency_mult * vol_mult
    lambda_away *= vol_mult
    lambda_total = lambda_home + lambda_away

    # 2. Dixon-Coles Joint Probability Matrix
    dc_matrix = compute_dixon_coles_matrix(lambda_home, lambda_away)

    prob_0_0 = dc_matrix[0, 0]
    prob_1_0 = dc_matrix[1, 0]
    prob_0_1 = dc_matrix[0, 1]
    prob_over_1_5 = 1.0 - (prob_0_0 + prob_1_0 + prob_0_1)

    prob_home_win = float(np.sum(np.tril(dc_matrix, -1)))
    prob_draw = float(np.sum(np.diag(dc_matrix)))
    prob_away_win = float(np.sum(np.triu(dc_matrix, 1)))
    prob_1x = prob_home_win + prob_draw
    prob_12 = prob_home_win + prob_away_win

    prob_home_o15 = float(np.sum(dc_matrix[2:, :]))
    prob_goals_2_4 = sum(
        dc_matrix[x, y]
        for x in range(dc_matrix.shape[0])
        for y in range(dc_matrix.shape[1])
        if 2 <= (x + y) <= 4
    )

    # 3. Strict Threshold Qualification (Unchanged)
    pass_over_1_5 = (prob_over_1_5 >= 0.94) and (lambda_total >= 2.80)
    pass_1x = (h_unbeaten >= 0.88) and (prob_1x >= 0.92) and not is_derby
    ppg_diff = abs(h_ppg - a_ppg)
    pass_12 = (ppg_diff > 1.0) and (prob_12 >= 0.86)
    pass_home_o15 = prob_home_o15 >= 0.65
    pass_goal_range = prob_goals_2_4 >= 0.72

    qualified_markets = []
    primary_prob = prob_over_1_5
    market_odds = h_odds

    if pass_over_1_5 and lambda_total < 4.2:
        qualified_markets.append("Over 1.5")
    if pass_1x:
        qualified_markets.append("1X")
        primary_prob = max(primary_prob, prob_1x)
    if pass_12:
        qualified_markets.append("12")
    if pass_1x and pass_over_1_5:
        qualified_markets.append("1X & Over 1.5")
        primary_prob = prob_1x * prob_over_1_5
        market_odds = round(h_odds * 1.18, 2)
    if pass_home_o15:
        qualified_markets.append("Home Over 1.5")
    if pass_goal_range:
        qualified_markets.append("Goals 2-4")

    if not qualified_markets:
        return {}

    # 4. EV & Fractional Kelly Calculation
    quant_risk = calculate_ev_and_kelly(primary_prob, market_odds, bankroll, kelly_mult)
    if quant_risk["ev_pct"] < min_ev:
        return {}

    return {
        "markets": " | ".join(qualified_markets),
        "primary_market": qualified_markets[0],
        "xg": round(lambda_total, 2),
        "dc_prob": round(primary_prob * 100, 2),
        "odds": market_odds,
        "ev_pct": quant_risk["ev_pct"],
        "kelly_pct": quant_risk["kelly_pct"],
        "stake": quant_risk["stake"],
        "derby": "Yes" if is_derby else "No",
    }


def parse_and_scan_schedules(
    raw_text: str, bankroll: float, kelly_mult: float, min_ev: float
) -> tuple:
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    current_competition = "MATCHDAY FIXTURE"
    qualified_rows = []
    scanned_count = 0

    # Broadened regex: captures all matchday fixtures regardless of competition or status tag
    fixture_regex = re.compile(
        r"^(\d{2}:\d{2}|[A-Za-z]{2,5})\s+([^-]+?)\s+-\s+(.+?)(?:\s+\d+\s*-\s*\d+.*|\s+-)?$"
    )

    for line in lines:
        if (
            ":" in line
            and not re.match(r"^\d{2}:\d{2}", line)
            and not re.match(r"^\d+'", line)
            and " - " not in line
        ):
            current_competition = line.replace("Standings", "").strip()
            continue

        if re.match(r"^\d+'", line):
            continue

        match = fixture_regex.match(line)
        if not match:
            continue

        kickoff = match.group(1).strip()
        home = match.group(2).strip()
        away = match.group(3).replace("\xa0", "").strip()
        away = re.sub(r"\s+\d+\.\d+.*$", "", away).strip()

        scanned_count += 1
        res = evaluate_fixture_quant(
            home, away, current_competition, bankroll, kelly_mult, min_ev
        )

        if res:
            match_label = f"{home} vs {away}"
            log_signal_for_clv(
                match_label,
                current_competition,
                res["primary_market"],
                res["odds"],
                res["dc_prob"] / 100.0,
            )
            qualified_rows.append(
                {
                    "Time": kickoff,
                    "Competition": current_competition,
                    "Match": match_label,
                    "Markets": res["markets"],
                    "xG": res["xg"],
                    "DC Prob": f"{res['dc_prob']}%",
                    "Odds": res["odds"],
                    "EV Edge": f"+{res['ev_pct']}%",
                    "Kelly %": f"{res['kelly_pct']}%",
                    "Stake": res["stake"],
                    "Derby": res["derby"],
                }
            )

    return pd.DataFrame(qualified_rows), scanned_count


# ==========================================
# STREAMLIT UI & HEADLESS DISPATCH CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Bankroll & Edge Controls")
    bankroll_input = st.number_input("Total Bankroll", min_value=10.0, value=1000.0, step=50.0)
    kelly_fraction_input = st.slider("Kelly Fraction Multiplier", 0.05, 1.0, 0.25, 0.05)
    min_ev_input = st.number_input("Minimum EV Edge (%)", value=0.5, step=0.5)
    auto_telegram = st.checkbox("Auto-Dispatch Qualified Signals to Telegram", value=True)

    st.divider()
    st.subheader("📈 CLV Recorder")
    clv_match = st.text_input("Match Name (exact)")
    clv_closing = st.number_input("Closing Odds", min_value=1.01, value=1.20, step=0.01)
    if st.button("Update Closing Odds"):
        clv_val = update_clv_record(clv_match, clv_closing)
        st.success(f"CLV Recorded: {clv_val}%")

raw_schedule_input = st.text_area("Paste Flashscore Schedule Here:", height=260)

if st.button("🚀 Run Quantitative Scan", type="primary"):
    if not raw_schedule_input.strip():
        st.warning("Paste Flashscore schedule text first.")
    else:
        df_results, total_scanned = parse_and_scan_schedules(
            raw_schedule_input, bankroll_input, kelly_fraction_input, min_ev_input
        )
        st.caption(f"Scanned {total_scanned} total matches across all competitions.")

        if not df_results.empty:
            st.subheader(f"✅ Qualified Value Signals ({len(df_results)} Found)")
            st.dataframe(df_results, use_container_width=True)

            signal_blocks = []
            for _, r in df_results.iterrows():
                block = (
                    f"🚨 *98% Quant Signal Alert*\n"
                    f"⚽ *{r['Match']}* | `{r['Time']}`\n"
                    f"🏆 {r['Competition']}\n"
                    f"🎯 *Picks:* `{r['Markets']}`\n"
                    f"📊 *DC Prob:* `{r['DC Prob']}` | *xG:* `{r['xG']}`\n"
                    f"💹 *Odds:* `{r['Odds']}` | *EV Edge:* `{r['EV Edge']}`\n"
                    f"💰 *Kelly Stake:* `{r['Kelly %']}` (`{r['Stake']}`)"
                )
                signal_blocks.append(block)

            full_telegram_payload = "\n\n---\n\n".join(signal_blocks)

            if auto_telegram:
                dispatch_status = send_telegram_alert(full_telegram_payload)
                if dispatch_status.get("ok"):
                    st.success("Dispatched all qualified signals to Telegram monitor.")
                else:
                    st.error(f"Telegram dispatch status: {dispatch_status}")

            st.markdown("### 📱 Telegram Payload")
            st.code(full_telegram_payload, language="text")
        else:
            st.info("0 matches met the strict Dixon-Coles + Positive EV criteria.")
