from pathlib import Path

import pandas as pd

# 1. CONFIGURATION
# Paths are built from this file's location (src/data_loader.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Seasons 2000/01 through 2026/27, matching files named like "2000_2001.csv"
FIRST_START_YEAR = 2000
LAST_START_YEAR = 2026
SEASONS = [f"{y}_{y + 1}" for y in range(FIRST_START_YEAR, LAST_START_YEAR + 1)]

# Columns every match row must have to be valid
REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Optional columns. Older seasons don't have all of these, which is fine.
OPTIONAL_STAT_COLS = ["HS", "AS", "HST", "AST", "HC", "AC", "B365H", "B365D", "B365A"]

TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield Utd",
    "Wolves": "Wolverhampton",
}


def find_season_files(seasons=SEASONS):
    """
    Phase 1A: Looks in data/raw for each season's CSV (e.g. 2000_2001.csv).
    Seasons with no file are skipped with a warning. The current season
    (2026_2027) may only be partly played or not downloaded yet.
    """
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    found = []
    for season in seasons:
        file_path = RAW_DIR / f"{season}.csv"
        if file_path.is_file():
            found.append((season, file_path))
        else:
            print(f"[WARN] No file for season {season}, skipping ({file_path.name})")

    if not found:
        raise FileNotFoundError(f"No season CSVs found in {RAW_DIR}")

    print(f"[INFO] Found {len(found)} season files in {RAW_DIR}")
    return found


def read_csv_safely(file_path):
    """Older football-data files are often not UTF-8, so fall back to latin-1."""
    try:
        return pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")


def validate_and_clean_season(file_path, season):
    """
    Phase 1B: Reads one raw CSV, checks required columns, drops empty rows,
    and fixes date formats.
    """
    df = read_csv_safely(file_path)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Season {season} is missing required columns: {missing}")

    # Drop blank rows and unplayed fixtures (no result yet)
    df = df.dropna(subset=REQUIRED_COLS).copy()

    available_optional = [col for col in OPTIONAL_STAT_COLS if col in df.columns]
    df = df[REQUIRED_COLS + available_optional].copy()

    df["Season"] = season.replace("_", "-")

    # Old files use DD/MM/YY, newer ones DD/MM/YYYY. format="mixed" handles both.
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"])

    # Tidy team names
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Sanity checks
    df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0)]
    df = df[df["FTR"].isin(["H", "D", "A"])]

    return df


def build_canonical_match_table(season_files):
    """
    Phase 2: Combines all seasons into one table, standardizes team names,
    and sorts strictly by date.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_seasons = [validate_and_clean_season(path, season) for season, path in season_files]
    canonical_df = pd.concat(all_seasons, ignore_index=True)

    canonical_df["HomeTeam"] = canonical_df["HomeTeam"].replace(TEAM_NAME_MAP)
    canonical_df["AwayTeam"] = canonical_df["AwayTeam"].replace(TEAM_NAME_MAP)

    # Oldest match first, newest last (the foundation of the no-leakage rule)
    canonical_df = canonical_df.sort_values(by=["Date", "HomeTeam"]).reset_index(drop=True)
    canonical_df.insert(0, "Match_ID", range(1, len(canonical_df) + 1))

    output_path = PROCESSED_DIR / "canonical_matches.csv"
    canonical_df.to_csv(output_path, index=False)

    print("\n[SUCCESS] Canonical match table built!")
    print(f" -> Saved to: {output_path}")
    print(f" -> Total valid matches: {len(canonical_df)}")
    print(f" -> Date range: {canonical_df['Date'].min().date()} to {canonical_df['Date'].max().date()}")

    return canonical_df


if __name__ == "__main__":
    files = find_season_files()
    build_canonical_match_table(files)