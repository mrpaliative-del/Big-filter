from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
LEAGUES = {
    "I2": "Italian Serie B",
    "F2": "French Ligue 2",
    "SP2": "Spanish Segunda",
    "E1": "English Championship",
}

SEASONS = ["2425", "2526", "2627"]
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

MIN_EDGE = 0.04
MAX_ODDS = 4.10
BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.03
OUTPUT_EXCEL = f"weekly_draw_picks_{datetime.now().strftime('%Y-%m-%d')}.xlsx"


# ==============================================================================
# 2. DIXON-COLES & KELLY MATHEMATICAL FUNCTIONS
# ==============================================================================
def dixon_coles_tau(h_goals, a_goals, h_xg, a_xg, rho):
    if h_goals == 0 and a_goals == 0:
        return 1.0 - (h_xg * a_xg * rho)
    elif h_goals == 0 and a_goals == 1:
        return 1.0 + (h_xg * rho)
    elif h_goals == 1 and a_goals == 0:
        return 1.0 + (a_xg * rho)
    elif h_goals == 1 and a_goals == 1:
        return 1.0 - rho
    return 1.0


def fit_dixon_coles_weighted(df, reference_date, xi=0.0022):
    teams = np.sort(
        np.unique(np.concatenate([df["HomeTeam"].values, df["AwayTeam"].values]))
    )
    n_teams = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}

    home_idx = df["HomeTeam"].map(team_to_idx).values
    away_idx = df["AwayTeam"].map(team_to_idx).values
    h_goals = df["FTHG"].values
    a_goals = df["FTAG"].values

    days_diff = (reference_date - df["Date"]).dt.days.values
    weights = np.exp(-xi * np.maximum(days_diff, 0))

    def neg_log_likelihood(params):
        attack = params[:n_teams]
        defense = params[n_teams : 2 * n_teams]
        home_adv = params[2 * n_teams]
        rho = params[2 * n_teams + 1]

        h_xg = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        a_xg = np.exp(attack[away_idx] + defense[home_idx])

        taus = np.array([
            dixon_coles_tau(hg, ag, hx, ax, rho)
            for hg, ag, hx, ax in zip(h_goals, a_goals, h_xg, a_xg)
        ])

        if np.any(taus <= 0):
            return 1e10

        log_lik = weights * (
            np.log(taus)
            + poisson.logpmf(h_goals, h_xg)
            + poisson.logpmf(a_goals, a_xg)
        )
        return -np.sum(log_lik)

    init_params = np.concatenate(
        [np.zeros(n_teams), np.zeros(n_teams), [0.25, -0.05]]
    )
    constraints = [{"type": "eq", "fun": lambda p: np.mean(p[:n_teams])}]
    bounds = [(-3, 3)] * (2 * n_teams) + [(0, 1)] + [(-0.5, 0.5)]

    res = minimize(
        neg_log_likelihood,
        init_params,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    fitted = res.x

    return {
        "teams": set(teams),
        "attack": dict(zip(teams, fitted[:n_teams])),
        "defense": dict(zip(teams, fitted[n_teams : 2 * n_teams])),
        "home_adv": fitted[2 * n_teams],
        "rho": fitted[2 * n_teams + 1],
    }


def predict_dc_prob(model, home_team, away_team, max_goals=6):
    if home_team not in model["teams"] or away_team not in model["teams"]:
        return None

    h_xg = np.exp(
        model["attack"][home_team]
        + model["defense"][away_team]
        + model["home_adv"]
    )
    a_xg = np.exp(model["attack"][away_team] + model["defense"][home_team])
    rho = model["rho"]

    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base = poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
            prob_matrix[h, a] = base * dixon_coles_tau(h, a, h_xg, a_xg, rho)

    prob_matrix /= np.sum(prob_matrix)
    return float(np.sum(np.diag(prob_matrix)))


def calculate_kelly_stake(
    prob,
    odds,
    bankroll,
    fraction=KELLY_FRACTION,
    max_stake_pct=MAX_STAKE_PCT,
):
    b = odds - 1.0
    q = 1.0 - prob
    kelly_f = ((b * prob) - q) / b
    if kelly_f <= 0:
        return 0.0
    stake_pct = min(kelly_f * fraction, max_stake_pct)
    return round(bankroll * stake_pct, 2)


# ==============================================================================
# 3. DATA LOADING, FEATURE ENGINEERING & WEIGHT OPTIMIZATION
# ==============================================================================
def fetch_merged_league_history(div_code, seasons):
    req_cols = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HST",
        "AST",
        "HC",
        "AC",
        "HF",
        "AF",
        "AvgH",
        "AvgD",
        "AvgA",
    ]
    frames = []
    for s in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{div_code}.csv"
        try:
            df = pd.read_csv(url, encoding="latin1")
            available = [c for c in req_cols if c in df.columns]
            if len(available) == len(req_cols):
                frames.append(df[req_cols].dropna())
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged["Date"] = pd.to_datetime(
        merged["Date"], dayfirst=True, format="mixed"
    )
    return merged.sort_values("Date").reset_index(drop=True)


def build_features_and_latest_state(raw_df, window=5):
    df = raw_df.sort_values("Date").reset_index(drop=True)
    df["Is_Draw"] = (df["FTR"] == "D").astype(int)

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    team_stats = {t: [] for t in teams}
    features = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        h_hist = pd.DataFrame(team_stats[h][-window:])
        a_hist = pd.DataFrame(team_stats[a][-window:])

        if len(h_hist) < window or len(a_hist) < window:
            features.append({})
        else:
            imp_h, imp_d, imp_a = (
                1.0 / row["AvgH"],
                1.0 / row["AvgD"],
                1.0 / row["AvgA"],
            )
            features.append({
                "h_sot_for": h_hist["sot_for"].mean(),
                "h_sot_against": h_hist["sot_against"].mean(),
                "a_sot_for": a_hist["sot_for"].mean(),
                "a_sot_against": a_hist["sot_against"].mean(),
                "sot_diff": abs(
                    h_hist["sot_for"].mean() - a_hist["sot_for"].mean()
                ),
                "combined_corners": (
                    h_hist["corners"].mean() + a_hist["corners"].mean()
                ),
                "combined_fouls": h_hist["fouls"].mean() + a_hist["fouls"].mean(),
                "market_draw_prob": imp_d,
                "home_away_prob_gap": abs(imp_h - imp_a),
            })

        team_stats[h].append({
            "sot_for": row["HST"],
            "sot_against": row["AST"],
            "corners": row["HC"],
            "fouls": row["HF"],
        })
        team_stats[a].append({
            "sot_for": row["AST"],
            "sot_against": row["HST"],
            "corners": row["AC"],
            "fouls": row["AF"],
        })

    feat_df = pd.DataFrame(features, index=df.index)
    full_df = pd.concat([df, feat_df], axis=1)

    latest_form = {}
    for t, logs in team_stats.items():
        if len(logs) >= window:
            recent = pd.DataFrame(logs[-window:])
            latest_form[t] = {
                "sot_for": recent["sot_for"].mean(),
                "sot_against": recent["sot_against"].mean(),
                "corners": recent["corners"].mean(),
                "fouls": recent["fouls"].mean(),
            }

    return full_df, latest_form


def train_calibrated_xgb(train_df, feature_cols):
    clean_train = train_df.dropna(subset=feature_cols)
    base_xgb = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb, method="sigmoid", cv=3
    )
    calibrated_model.fit(clean_train[feature_cols], clean_train["Is_Draw"])
    return calibrated_model


def learn_optimal_blend_weights(full_df, feature_cols, eval_tail=150):
    clean_df = full_df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(clean_df) < eval_tail + 150:
        return 0.65, 0.35

    split_idx = len(clean_df) - eval_tail
    train_part = clean_df.iloc[:split_idx]
    val_part = clean_df.iloc[split_idx:]

    dc_val_model = fit_dixon_coles_weighted(
        train_part, train_part["Date"].max(), xi=0.0022
    )
    xgb_val_model = train_calibrated_xgb(train_part, feature_cols)

    y_true, dc_preds, xgb_preds = [], [], []
    for _, row in val_part.iterrows():
        dc_p = predict_dc_prob(dc_val_model, row["HomeTeam"], row["AwayTeam"])
        if dc_p is None:
            continue
        X_row = pd.DataFrame([row[feature_cols]])
        xgb_p = float(xgb_val_model.predict_proba(X_row)[:, 1])

        y_true.append(row["Is_Draw"])
        dc_preds.append(dc_p)
        xgb_preds.append(xgb_p)

    if len(y_true) < 50:
        return 0.65, 0.35

    y_true = np.array(y_true)
    dc_preds = np.array(dc_preds)
    xgb_preds = np.array(xgb_preds)

    def loss_func(w):
        blended = np.clip(
            (w * dc_preds) + ((1.0 - w) * xgb_preds), 1e-6, 1.0 - 1e-6
        )
        return log_loss(y_true, blended)

    res = minimize_scalar(loss_func, bounds=(0.0, 1.0), method="bounded")
    dc_w = round(float(res.x), 4)
    return dc_w, round(1.0 - dc_w, 4)


# ==============================================================================
# 4. MAIN MULTI-LEAGUE EXECUTION PIPELINE
# ==============================================================================
def run_weekly_draw_scanner():
    feature_cols = [
        "h_sot_for",
        "h_sot_against",
        "a_sot_for",
        "a_sot_against",
        "sot_diff",
        "combined_corners",
        "combined_fouls",
        "market_draw_prob",
        "home_away_prob_gap",
    ]

    print("Fetching upcoming fixtures from Football-Data.co.uk...")
    try:
        fixtures_df = pd.read_csv(FIXTURES_URL, encoding="latin1")
    except Exception as e:
        print(f"Failed to download fixtures.csv: {e}")
        return

    all_picks = []

    for div_code, league_name in LEAGUES.items():
        print(f"\nProcessing {league_name} ({div_code})...")
        league_fixtures = fixtures_df[fixtures_df["Div"] == div_code].copy()

        if league_fixtures.empty:
            print(f"  -> No upcoming fixtures found for {league_name}.")
            continue

        raw_history = fetch_merged_league_history(div_code, SEASONS)
        if len(raw_history) < 200:
            print(f"  -> Insufficient historical matches for {league_name}.")
            continue

        full_df, latest_form = build_features_and_latest_state(
            raw_history, window=5
        )

        dc_w, xgb_w = learn_optimal_blend_weights(
            full_df, feature_cols, eval_tail=150
        )
        print(f"  -> Learned Weights | Dixon-Coles: {dc_w} | Calibrated XGB: {xgb_w}")

        dc_model = fit_dixon_coles_weighted(
            full_df, full_df["Date"].max(), xi=0.0022
        )
        cal_xgb_model = train_calibrated_xgb(full_df, feature_cols)

        for _, row in league_fixtures.iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            market_d = row.get("AvgD", np.nan)
            market_h = row.get("AvgH", np.nan)
            market_a = row.get("AvgA", np.nan)

            if pd.isna(market_d) or pd.isna(market_h) or pd.isna(market_a):
                continue
            if market_d > MAX_ODDS:
                continue
            if h not in latest_form or a not in latest_form:
                continue

            dc_prob = predict_dc_prob(dc_model, h, a)
            if dc_prob is None:
                continue

            h_stats, a_stats = latest_form[h], latest_form[a]
            imp_h, imp_d, imp_a = 1.0 / market_h, 1.0 / market_d, 1.0 / market_a

            X_live = pd.DataFrame([{
                "h_sot_for": h_stats["sot_for"],
                "h_sot_against": h_stats["sot_against"],
                "a_sot_for": a_stats["sot_for"],
                "a_sot_against": a_stats["sot_against"],
                "sot_diff": abs(h_stats["sot_for"] - a_stats["sot_for"]),
                "combined_corners": h_stats["corners"] + a_stats["corners"],
                "combined_fouls": h_stats["fouls"] + a_stats["fouls"],
                "market_draw_prob": imp_d,
                "home_away_prob_gap": abs(imp_h - imp_a),
            }])
            xgb_prob = float(cal_xgb_model.predict_proba(X_live)[:, 1])

            blend_prob = (dc_w * dc_prob) + (xgb_w * xgb_prob)
            edge = blend_prob - imp_d

            if edge >= MIN_EDGE:
                stake = calculate_kelly_stake(blend_prob, market_d, BANKROLL)
                all_picks.append({
                    "League": league_name,
                    "Date": row["Date"],
                    "Time": row.get("Time", ""),
                    "HomeTeam": h,
                    "AwayTeam": a,
                    "DC_Weight": dc_w,
                    "DC_Prob_Pct": round(dc_prob * 100, 2),
                    "Cal_XGB_Pct": round(xgb_prob * 100, 2),
                    "Blend_Prob_Pct": round(blend_prob * 100, 2),
                    "Fair_Odds": round(1.0 / blend_prob, 2),
                    "Market_Odds": market_d,
                    "Edge_Pct": round(edge * 100, 2),
                    "Kelly_Stake": stake,
                })

    if not all_picks:
        print("\nScan complete: No +EV draw picks met your edge criteria this week.")
        return

    results_df = pd.DataFrame(all_picks).sort_values(
        by="Edge_Pct", ascending=False
    )
    results_df.to_excel(OUTPUT_EXCEL, index=False)
    print("\n=======================================================")
    print(f"SUCCESS: Exported {len(results_df)} value draw picks to {OUTPUT_EXCEL}")
    print("=======================================================")
    print(
        results_df[[
            "League",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "Blend_Prob_Pct",
            "Market_Odds",
            "Edge_Pct",
            "Kelly_Stake",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    run_weekly_draw_scanner()
