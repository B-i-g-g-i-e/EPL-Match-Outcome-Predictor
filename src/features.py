from pathlib import Path

import numpy as np
import pandas as pd

# Paths are built from this file's location (src/features.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def compute_elo_and_rest_days(df, k_factor=20.0, home_advantage=65.0, base_elo=1500.0):
    """
    Phase 3A: Iterates chronologically through all matches.
    Records PRE-MATCH Elo ratings and rest days BEFORE updating ratings with the match outcome.
    """
    elo_ratings = {}
    last_played_date = {}

    home_elos, away_elos = [], []
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        h_team, a_team = row["HomeTeam"], row["AwayTeam"]
        match_date = pd.to_datetime(row["Date"])

        # 1. Record current Elo BEFORE kickoff (defaults to 1500.0 for first appearance)
        h_elo = elo_ratings.get(h_team, base_elo)
        a_elo = elo_ratings.get(a_team, base_elo)
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # 2. Record days of rest since previous match (capped at 14 days for off-seasons)
        h_days = (match_date - last_played_date[h_team]).days if h_team in last_played_date else 14
        a_days = (match_date - last_played_date[a_team]).days if a_team in last_played_date else 14
        home_rest.append(min(max(h_days, 1), 14))
        away_rest.append(min(max(a_days, 1), 14))

        # 3. ONLY AFTER recording pre-match features, update Elo using the match result
        exp_home = 1.0 / (1.0 + 10.0 ** (-((h_elo + home_advantage) - a_elo) / 400.0))
        actual_home = 1.0 if row["FTR"] == "H" else (0.5 if row["FTR"] == "D" else 0.0)

        # Goal difference multiplier rewards decisive wins with diminishing returns
        goal_diff = abs(row["FTHG"] - row["FTAG"])
        margin_mult = np.log(max(goal_diff, 1) + 1.0)

        elo_shift = k_factor * margin_mult * (actual_home - exp_home)
        elo_ratings[h_team] = h_elo + elo_shift
        elo_ratings[a_team] = a_elo - elo_shift

        # Update last played date for both clubs
        last_played_date[h_team] = match_date
        last_played_date[a_team] = match_date

    df["HomeElo"] = home_elos
    df["AwayElo"] = away_elos
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    df["HomeRestDays"] = home_rest
    df["AwayRestDays"] = away_rest
    df["RestDiff"] = df["HomeRestDays"] - df["AwayRestDays"]
    return df


def compute_rolling_team_form(df, window=5):
    """
    Phase 3B: Computes 5-match rolling form (points, goals, shot-based xG proxy) per team.
    Uses .shift(1) so only matches completed strictly prior to kickoff are included.
    """
    # Safely fill any missing shot counts in older seasons with league-average defaults
    hs = df["HS"].fillna(12.0) if "HS" in df.columns else pd.Series(12.0, index=df.index)
    as_ = df["AS"].fillna(10.0) if "AS" in df.columns else pd.Series(10.0, index=df.index)
    hst = df["HST"].fillna(4.5) if "HST" in df.columns else pd.Series(4.5, index=df.index)
    ast = df["AST"].fillna(3.8) if "AST" in df.columns else pd.Series(3.8, index=df.index)

    # Create one perspective row for the Home team and one for the Away team
    home_rows = pd.DataFrame({
        "Match_ID": df["Match_ID"],
        "Date": df["Date"],
        "Team": df["HomeTeam"],
        "IsHome": 1,
        "GoalsFor": df["FTHG"],
        "GoalsAgainst": df["FTAG"],
        "Points": df["FTR"].map({"H": 3, "D": 1, "A": 0}),
        "xG_Proxy_For": 0.10 * hs + 0.30 * hst,
        "xG_Proxy_Against": 0.10 * as_ + 0.30 * ast,
    })

    away_rows = pd.DataFrame({
        "Match_ID": df["Match_ID"],
        "Date": df["Date"],
        "Team": df["AwayTeam"],
        "IsHome": 0,
        "GoalsFor": df["FTAG"],
        "GoalsAgainst": df["FTHG"],
        "Points": df["FTR"].map({"A": 3, "D": 1, "H": 0}),
        "xG_Proxy_For": 0.10 * as_ + 0.30 * ast,
        "xG_Proxy_Against": 0.10 * hs + 0.30 * hst,
    })

    # Combine and sort strictly by Team and Date
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(by=["Team", "Date", "Match_ID"]).reset_index(drop=True)

    # CRITICAL LEAKAGE GUARD: .shift(1) pushes rows down by 1 game before .rolling(window)
    grouped = long_df.groupby("Team")
    for col in ["GoalsFor", "GoalsAgainst", "Points", "xG_Proxy_For", "xG_Proxy_Against"]:
        long_df[f"Roll_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

    long_df["Roll_xGDiff"] = long_df["Roll_xG_Proxy_For"] - long_df["Roll_xG_Proxy_Against"]
    long_df["Roll_GoalDiff"] = long_df["Roll_GoalsFor"] - long_df["Roll_GoalsAgainst"]

    # Join the Home and Away rolling features back onto the main match table by Match_ID
    roll_cols = ["Roll_GoalsFor", "Roll_GoalsAgainst", "Roll_Points", "Roll_xGDiff", "Roll_GoalDiff"]
    home_feats = long_df[long_df["IsHome"] == 1][["Match_ID"] + roll_cols].rename(
        columns={c: f"Home_{c}" for c in roll_cols}
    )
    away_feats = long_df[long_df["IsHome"] == 0][["Match_ID"] + roll_cols].rename(
        columns={c: f"Away_{c}" for c in roll_cols}
    )

    df = df.merge(home_feats, on="Match_ID", how="left")
    df = df.merge(away_feats, on="Match_ID", how="left")

    # Compute differential features (Home advantage in form and xG)
    df["FormPointsDiff"] = df["Home_Roll_Points"] - df["Away_Roll_Points"]
    df["xGDiff_Diff"] = df["Home_Roll_xGDiff"] - df["Away_Roll_xGDiff"]
    return df


def build_features():
    in_path = PROCESSED_DIR / "canonical_matches.csv"
    if not in_path.is_file():
        raise FileNotFoundError(
            f"{in_path} not found. Run `python -m src.data_loader` first."
        )

    df = pd.read_csv(in_path, parse_dates=["Date"])

    print(f"[INFO] Computing Elo ratings, rest days, and 5-match rolling form for {len(df)} matches...")
    df = compute_elo_and_rest_days(df)
    df = compute_rolling_team_form(df, window=5)

    out_path = PROCESSED_DIR / "features_matches.csv"
    df.to_csv(out_path, index=False)

    print(f"[SUCCESS] Saved leakage-safe feature dataset to: {out_path}")
    print(f" -> Total columns: {len(df.columns)}")
    return df


if __name__ == "__main__":
    build_features()