import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import re

st.set_page_config(page_title="Extended 98% League Filter", page_icon="🎯", layout="wide")

st.title("🎯 League Match 98% Filter & Extended Markets")
st.markdown(
    "Paste your Flashscore schedule below. Automatically excludes cups and qualifiers, "
    "evaluating regular league fixtures across core markets (**1X, 12, Over 1.5**) and extended value options (**Combos, Team Goals, Goal Ranges**)."
)

# --- ADVANCED MATHEMATICAL ENGINE & EXTENDED MARKETS ---
def evaluate_extended_markets(h_sc, h_co, a_sc, a_co, h_unbeaten_l10, h_ppg, a_ppg, h2h_draw_rate):
    # 1. Expected Goals (Lambda)
    lambda_home = (h_sc + a_co) / 2.0
    lambda_away = (a_sc + h_co) / 2.0
    lambda_total = lambda_home + lambda_away

    # 2. Over 1.5 Goals Filter
    prob_over_1_5 = 1.0 - (poisson.pmf(0, lambda_total) + poisson.pmf(1, lambda_total))
    pass_over_1_5 = (prob_over_1_5 >= 0.96) and (lambda_total >= 2.85)

    # 3. 1X Filter
    home_superiority = lambda_home - lambda_away
    pass_1x = (h_unbeaten_l10 >= 0.90) and (home_superiority > -0.3) and (h_sc >= 1.8)

    # 4. 12 (No Draw) Filter
    ppg_diff = abs(h_ppg - a_ppg)
    pass_12 = (ppg_diff > 1.0) and (h2h_draw_rate < 0.10)

    # 5. Extended Market calculations
    prob_home_over_1_5 = 1.0 - (poisson.pmf(0, lambda_home) + poisson.pmf(1, lambda_home))
    pass_home_over_1_5 = prob_home_over_1_5 >= 0.65

    prob_goal_range_2_4 = poisson.pmf(2, lambda_total) + poisson.pmf(3, lambda_total) + poisson.pmf(4, lambda_total)
    pass_goal_range = prob_goal_range_2_4 >= 0.75

    markets = []
    
    # Core Markets
    if pass_over_1_5 and lambda_total < 3.8:
        markets.append("Over 1.5")
    if pass_1x and h_unbeaten_l10 <= 0.96:
        markets.append("1X")
    if pass_12:
        markets.append("12")
        
    # Extended Markets
    if pass_1x and pass_over_1_5:
        markets.append("1X & Over 1.5 (Combo)")
    if pass_home_over_1_5:
        markets.append("Home Team Over 1.5")
    if pass_goal_range:
        markets.append("Goal Range: 2-4 Goals")

    return markets, round(lambda_total, 2), round(prob_over_1_5 * 100, 2), round(ppg_diff, 2)

# --- GLOBAL LEAGUE PROFILES ---
ELITE_LEAGUE_PROFILES = {
    "arsenal": (2.4, 0.6, 2.4, 0.92),
    "bayern munich": (2.8, 0.7, 2.6, 0.94),
    "real madrid": (2.5, 0.7, 2.5, 0.92),
    "barcelona": (2.6, 0.8, 2.5, 0.90),
    "psg": (2.6, 0.7, 2.5, 0.92),
    "inter milan": (2.3, 0.5, 2.4, 0.92),
    "manchester city": (2.7, 0.7, 2.6, 0.94),
    "bayer leverkusen": (2.5, 0.8, 2.4, 0.90),
    "psv": (2.7, 0.8, 2.5, 0.92),
    "ajax": (2.4, 0.9, 2.3, 0.90),
}

WEAK_AWAY_PROFILES = {
    "bochum": (0.8, 2.2, 0.8),
    "getafe": (0.7, 1.9, 0.9),
    "monza": (0.8, 1.8, 0.9),
    "frosinone": (0.9, 2.1, 0.8),
    "heidenheim": (0.9, 2.0, 0.8),
}

CUP_EXCLUSION_KEYWORDS = [
    "cup", "qualification", "qualifiers", "play-off", "playoffs", 
    "super cup", "shield", "trophy", "nations league", "friendly", 
    "copa", "u17", "u18", "u19", "u20", "u21", "u23", "women", "(am)", 
    "reserve", "postponed", "cancelled", "delayed"
]

def parse_and_scan_schedules(raw_text):
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    current_competition = "REGULAR LEAGUE"
    qualified_rows = []
    scanned_count = 0

    fixture_regex = re.compile(r'^(\d{2}:\d{2})\s*([^-]+?)\s+-\s+(.+?)(?:\s+\d+-\d+.*|\s+-)?$')

    for line in lines:
        if ":" in line and not re.match(r'^\d{2}:\d{2}', line) and not re.match(r"^\d+'", line):
            current_competition = line.replace("Standings", "").strip()
            continue

        if re.match(r"^\d+'", line) or re.search(r'\d+-\d+', line):
            continue

        match = fixture_regex.match(line)
        if not match:
            continue

        comp_check = current_competition.lower()
        if any(cup_word in comp_check for cup_word in CUP_EXCLUSION_KEYWORDS):
            continue

        kickoff = match.group(1).strip()
        home = match.group(2).strip()
        away = match.group(3).replace("\xa0", "").strip()
        full_context = f"{current_competition} {home} {away}".lower()

        if any(noise in full_context for noise in CUP_EXCLUSION_KEYWORDS):
            continue

        scanned_count += 1
        h_key = home.lower()
        a_key = away.lower()

        if h_key in ELITE_LEAGUE_PROFILES:
            h_sc, h_co, h_ppg, h_unbeaten = ELITE_LEAGUE_PROFILES[h_key]
        else:
            h_sc, h_co, h_ppg, h_unbeaten = (1.55, 1.05, 1.75, 0.80)

        if a_key in WEAK_AWAY_PROFILES:
            a_sc, a_co, a_ppg = WEAK_AWAY_PROFILES[a_key]
            h2h_draw = 0.05
        else:
            a_sc, a_co, a_ppg = (1.10, 1.25, 1.20)
            h2h_draw = 0.22

        markets, xg_tot, over15_prob, ppg_diff = evaluate_extended_markets(
            h_sc, h_co, a_sc, a_co, h_unbeaten, h_ppg, a_ppg, h2h_draw
        )

        if markets:
            qualified_rows.append({
                "Time": kickoff,
                "Competition": current_competition,
                "Match": f"{home} vs {away}",
                "Markets": " | ".join(markets),
                "xG": xg_tot,
                "Prob": f"{over15_prob}%"
            })

    return pd.DataFrame(qualified_rows), scanned_count

raw_schedule_input = st.text_area("Paste Flashscore Schedule Here:", height=260)

if st.button("🚀 Filter Extended Markets", type="primary"):
    if not raw_schedule_input.strip():
        st.warning("Please paste schedule text first.")
    else:
        df_results, total_scanned = parse_and_scan_schedules(raw_schedule_input)
        st.caption(f"Scanned {total_scanned} regular league matches (cups excluded).")

        if not df_results.empty:
            st.subheader(f"✅ Qualified Matches with Extended Markets ({len(df_results)} Found)")
            st.dataframe(df_results, use_container_width=True)

            st.markdown("### 📱 Telegram Copy Block")
            copy_blocks = [
                f"⚽ {r['Match']} | {r['Time']}\n🏆 {r['Competition']}\n🎯 Picks: {r['Markets']} (xG: {r['xG']})"
                for _, r in df_results.iterrows()
            ]
            st.code("\n\n".join(copy_blocks), language="text")
        else:
            st.info("0 regular league fixtures matched the strict criteria in this batch.")
