import pandas as pd

from src.features import compute_elo_and_rest_days, compute_rolling_team_form

# Every engineered, pre-match column the pipeline produces
FEATURE_COLS = [
    "HomeElo", "AwayElo", "EloDiff",
    "HomeRestDays", "AwayRestDays", "RestDiff",
    "Home_Roll_GoalsFor", "Home_Roll_GoalsAgainst", "Home_Roll_Points",
    "Home_Roll_xGDiff", "Home_Roll_GoalDiff",
    "Away_Roll_GoalsFor", "Away_Roll_GoalsAgainst", "Away_Roll_Points",
    "Away_Roll_xGDiff", "Away_Roll_GoalDiff",
    "FormPointsDiff", "xGDiff_Diff",
]


def build_features(df):
    return compute_rolling_team_form(compute_elo_and_rest_days(df.copy()))


def test_no_future_or_current_match_leakage():
    """Changing Match 3's score must not alter pre-match features for Matches 1 to 3."""
    sample = pd.DataFrame({
        "Match_ID": [1, 2, 3, 4],
        "Date": pd.to_datetime(["2024-08-10", "2024-08-17", "2024-08-24", "2024-08-31"]),
        "HomeTeam": ["Arsenal", "Liverpool", "Arsenal", "Liverpool"],
        "AwayTeam": ["Liverpool", "Arsenal", "Chelsea", "Chelsea"],
        "FTHG": [1, 2, 1, 0],
        "FTAG": [1, 0, 0, 1],
        "FTR": ["D", "H", "H", "A"],
    })

    original = build_features(sample)

    tampered = sample.copy()
    tampered.loc[2, "FTHG"] = 99  # Match 3 becomes 99-0
    mutated = build_features(tampered)

    # Matches 1 to 3 must be identical, across ALL engineered features
    pd.testing.assert_frame_equal(
        original.iloc[:3][FEATURE_COLS].reset_index(drop=True),
        mutated.iloc[:3][FEATURE_COLS].reset_index(drop=True),
    )

    # Sanity check: Match 4 SHOULD notice, otherwise the test proves nothing
    assert original.loc[3, "AwayElo"] != mutated.loc[3, "AwayElo"]
    assert original.loc[3, "Away_Roll_GoalsAgainst"] != mutated.loc[3, "Away_Roll_GoalsAgainst"]