import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train import STAGE2_FEATURES, DixonColesModel, make_calibrated_lgbm  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"


def load_artifacts():
    """Loads the feature table and both trained models once."""
    features_path = PROCESSED_DIR / "features_matches.csv"
    dc_path = MODELS_DIR / "dixon_coles_stage1.pkl"
    lgbm_path = MODELS_DIR / "lightgbm_stage2.pkl"

    for path, hint in [
        (features_path, "python -m src.features"),
        (dc_path, "python -m src.train"),
        (lgbm_path, "python -m src.train"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `{hint}` first.")

    df = pd.read_csv(features_path, parse_dates=["Date"])
    return df, joblib.load(dc_path), joblib.load(lgbm_path)


def get_latest_team_state(df, team_name, k_factor=20.0, home_advantage=65.0):
    """
    Extracts a club's current post-match Elo and 5-match rolling stats,
    including their most recently completed match.
    """
    team_matches = df[(df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)].copy()
    if team_matches.empty:
        known = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
        raise ValueError(
            f"Team '{team_name}' not found in the data. "
            f"Names must match the dataset exactly, for example: {known[:8]} ..."
        )

    # 1. Current Elo AFTER the most recent match finished
    last_row = team_matches.iloc[-1]
    h_elo, a_elo = float(last_row["HomeElo"]), float(last_row["AwayElo"])
    exp_home = 1.0 / (1.0 + 10.0 ** (-((h_elo + home_advantage) - a_elo) / 400.0))
    actual_home = 1.0 if last_row["FTR"] == "H" else (0.5 if last_row["FTR"] == "D" else 0.0)
    goal_diff = abs(last_row["FTHG"] - last_row["FTAG"])
    elo_shift = k_factor * np.log(max(goal_diff, 1) + 1.0) * (actual_home - exp_home)

    current_elo = (h_elo + elo_shift) if last_row["HomeTeam"] == team_name else (a_elo - elo_shift)

    # 2. Rolling stats across the last 5 completed matches
    last_5 = team_matches.tail(5)
    pts, xg_diff, g_diff = [], [], []

    for _, row in last_5.iterrows():
        hs = row.get("HS", 12.0) if pd.notna(row.get("HS", np.nan)) else 12.0
        as_ = row.get("AS", 10.0) if pd.notna(row.get("AS", np.nan)) else 10.0
        hst = row.get("HST", 4.5) if pd.notna(row.get("HST", np.nan)) else 4.5
        ast = row.get("AST", 3.8) if pd.notna(row.get("AST", np.nan)) else 3.8

        h_xg = 0.10 * hs + 0.30 * hst
        a_xg = 0.10 * as_ + 0.30 * ast

        if row["HomeTeam"] == team_name:
            pts.append(3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0))
            xg_diff.append(h_xg - a_xg)
            g_diff.append(row["FTHG"] - row["FTAG"])
        else:
            pts.append(3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0))
            xg_diff.append(a_xg - h_xg)
            g_diff.append(row["FTAG"] - row["FTHG"])

    return {
        "Elo": float(current_elo),
        "Roll_Points": float(np.mean(pts)),
        "Roll_xGDiff": float(np.mean(xg_diff)),
        "Roll_GoalDiff": float(np.mean(g_diff)),
    }


def days_since_last_match(df, team, match_date):
    """Rest days before match_date, capped at 1 to 14 like the training features."""
    mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    last = df.loc[mask, "Date"].max()
    if pd.isna(last):
        return 14
    return int(min(max((match_date - last).days, 1), 14))


def predict_fixture(home_team, away_team, artifacts, rest_home=None, rest_away=None, match_date=None):
    """
    Stage 1 expected goals and scoreline grid, plus Stage 2 calibrated probabilities.
    If match_date is given, only matches played before that date are used for
    Elo and form, and rest days are calculated automatically unless you pass them.
    """
    df, dc_model, lgbm_model = artifacts

    if match_date is not None:
        match_ts = pd.Timestamp(match_date)
        df = df[df["Date"] < match_ts]
        if rest_home is None:
            rest_home = days_since_last_match(df, home_team, match_ts)
        if rest_away is None:
            rest_away = days_since_last_match(df, away_team, match_ts)

    rest_home = 7 if rest_home is None else rest_home
    rest_away = 7 if rest_away is None else rest_away

    # 1. Stage 1 scout (Dixon-Coles)
    dc_out = dc_model.predict_match(home_team, away_team)

    # 2. Team form and Elo as of the match date
    h_state = get_latest_team_state(df, home_team)
    a_state = get_latest_team_state(df, away_team)

    # 3. The 20 features for the Stage 2 manager
    row_dict = {
        "dc_lam": dc_out["lam"],
        "dc_mu": dc_out["mu"],
        "dc_p_H": dc_out["p_H"],
        "dc_p_D": dc_out["p_D"],
        "dc_p_A": dc_out["p_A"],
        "dc_over_25": dc_out["over_25"],
        "HomeElo": h_state["Elo"],
        "AwayElo": a_state["Elo"],
        "EloDiff": h_state["Elo"] - a_state["Elo"],
        "HomeRestDays": rest_home,
        "AwayRestDays": rest_away,
        "RestDiff": rest_home - rest_away,
        "Home_Roll_Points": h_state["Roll_Points"],
        "Away_Roll_Points": a_state["Roll_Points"],
        "FormPointsDiff": h_state["Roll_Points"] - a_state["Roll_Points"],
        "Home_Roll_xGDiff": h_state["Roll_xGDiff"],
        "Away_Roll_xGDiff": a_state["Roll_xGDiff"],
        "xGDiff_Diff": h_state["Roll_xGDiff"] - a_state["Roll_xGDiff"],
        "Home_Roll_GoalDiff": h_state["Roll_GoalDiff"],
        "Away_Roll_GoalDiff": a_state["Roll_GoalDiff"],
    }

    X_input = pd.DataFrame([row_dict])[STAGE2_FEATURES]
    probs = lgbm_model.predict_proba(X_input)[0]

    best_idx = np.unravel_index(np.argmax(dc_out["matrix"]), dc_out["matrix"].shape)

    return {
        "Fixture": f"{home_team} vs {away_team}",
        "Expected_Goals": f"{dc_out['lam']:.2f} - {dc_out['mu']:.2f}",
        "Most_Likely_Score": f"{best_idx[0]}-{best_idx[1]} ({dc_out['matrix'][best_idx] * 100:.1f}%)",
        "P(Home)": round(float(probs[0]), 4),
        "P(Draw)": round(float(probs[1]), 4),
        "P(Away)": round(float(probs[2]), 4),
        "Matrix": dc_out["matrix"],
        "Rest_Home": rest_home,
        "Rest_Away": rest_away,
    }

BACKTEST_PATH = PROCESSED_DIR / "backtest_predictions.csv"
MIN_STAGE2_ROWS = 1000  # minimum past rows needed to fit the Stage 2 model


def load_replay_data():
    """Loads the feature table and the saved walk-forward Stage 1 outputs."""
    features_path = PROCESSED_DIR / "features_matches.csv"
    for path, hint in [(features_path, "python -m src.features"), (BACKTEST_PATH, "python -m src.train")]:
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `{hint}` first.")

    df = pd.read_csv(features_path, parse_dates=["Date"])
    stage2 = pd.read_csv(BACKTEST_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    return df, stage2


def earliest_replay_date(stage2):
    """Earliest date with enough earlier history to train Stage 2 fairly."""
    return stage2["Date"].iloc[MIN_STAGE2_ROWS].date()


def fit_models_as_of(df, stage2, cutoff):
    """Fits Stage 1 and Stage 2 using ONLY matches dated before `cutoff`."""
    cutoff = pd.Timestamp(cutoff)

    past = df[df["Date"] < cutoff].dropna(subset=["Home_Roll_Points", "Away_Roll_Points"])
    train2 = stage2[stage2["Date"] < cutoff]
    if len(train2) < MIN_STAGE2_ROWS:
        raise ValueError("Not enough earlier matches to train the model for that date.")

    dc_model = DixonColesModel(use_dc=True).fit(past)
    lgbm_model = make_calibrated_lgbm()
    lgbm_model.fit(train2[STAGE2_FEATURES], train2["Target"].values)
    return dc_model, lgbm_model


if __name__ == "__main__":
    artifacts = load_artifacts()

    # Usage: python -m src.predict "Arsenal" "Liverpool"
    if len(sys.argv) == 3:
        fixtures = [(sys.argv[1], sys.argv[2])]
    else:
        fixtures = [
            ("Arsenal", "Liverpool"),
            ("Manchester City", "Tottenham"),
            ("Chelsea", "Newcastle"),
        ]

    preds = [predict_fixture(h, a, artifacts) for h, a in fixtures]
    out_df = pd.DataFrame(preds).drop(columns=["Matrix"])
    print("\n=== UPCOMING EPL GAMEWEEK PREDICTIONS ===")
    print(out_df.to_string(index=False))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROCESSED_DIR / "upcoming_predictions.csv", index=False)