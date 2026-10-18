from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

FPL_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (EPL-Match-Outcome-Predictor)"}
LOCAL_TZ = "Africa/Nairobi"  # kick-off times are shown in this time zone

# FPL team names -> the names used in your dataset (check the right-hand side
# against the spellings in your CSVs)
FPL_TO_DATASET = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
}

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PRIOR_RATE_90 = {1: 0.005, 2: 0.04, 3: 0.12, 4: 0.30}
PRIOR_MINUTES = 450
LEAGUE_AVG_GOALS = 1.35


class LiveDataError(RuntimeError):
    """Raised when the live data can't be fetched."""


def _get(path):
    try:
        resp = requests.get(f"{FPL_URL}/{path}", headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as err:
        raise LiveDataError(f"Could not fetch live data from '{path}': {err}") from err


def fetch_bootstrap():
    return _get("bootstrap-static/")


def fetch_fixtures():
    return _get("fixtures/")


def weekend_start(today=None):
    """Saturday of the current weekend (the coming Saturday Monday to Friday)."""
    today = today or date.today()
    wd = today.weekday()
    if wd == 6:
        return today - timedelta(days=1)
    if wd == 5:
        return today
    return today + timedelta(days=5 - wd)


def _team_names(bootstrap):
    return {t["id"]: FPL_TO_DATASET.get(t["name"], t["name"]) for t in bootstrap["teams"]}


def all_team_names(bootstrap):
    return sorted(_team_names(bootstrap).values())


def get_team_meta(bootstrap):
    """Dataset team name -> FPL team code (used for badges) and 3-letter short name."""
    return {
        FPL_TO_DATASET.get(t["name"], t["name"]): {"code": t["code"], "short": t["short_name"]}
        for t in bootstrap["teams"]
    }


def get_upcoming_fixtures(bootstrap, fixtures):
    """Fixtures that haven't started, from this weekend onwards."""
    teams = _team_names(bootstrap)
    start = weekend_start()
    rows = []
    for fx in fixtures:
        if fx.get("finished") or fx.get("finished_provisional") or fx.get("started"):
            continue
        if not fx.get("kickoff_time"):
            continue
        kickoff = pd.Timestamp(fx["kickoff_time"]).tz_convert(LOCAL_TZ)
        if kickoff.date() < start:
            continue
        rows.append({
            "Kickoff": kickoff,
            "MatchDate": kickoff.date(),
            "Home": teams.get(fx["team_h"], "?"),
            "Away": teams.get(fx["team_a"], "?"),
            "Gameweek": fx.get("event"),
        })

    cols = ["Kickoff", "MatchDate", "Home", "Away", "Gameweek"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("Kickoff").reset_index(drop=True)


def get_results(bootstrap, fixtures):
    """Finished matches this season, with scores."""
    teams = _team_names(bootstrap)
    rows = []
    for fx in fixtures:
        done = fx.get("finished") or fx.get("finished_provisional")
        if not done or fx.get("team_h_score") is None or fx.get("team_a_score") is None:
            continue
        kickoff = fx.get("kickoff_time")
        when = pd.Timestamp(kickoff).tz_convert(LOCAL_TZ).tz_localize(None) if kickoff else pd.NaT
        rows.append({
            "Date": when,
            "Home": teams.get(fx["team_h"], "?"),
            "Away": teams.get(fx["team_a"], "?"),
            "HG": int(fx["team_h_score"]),
            "AG": int(fx["team_a_score"]),
            "Gameweek": fx.get("event"),
        })
    return pd.DataFrame(rows, columns=["Date", "Home", "Away", "HG", "AG", "Gameweek"])


def get_remaining(bootstrap, fixtures):
    """Every match still to be completed this season, including unscheduled ones."""
    teams = _team_names(bootstrap)
    rows = []
    for fx in fixtures:
        if fx.get("finished") or fx.get("finished_provisional"):
            continue
        rows.append({
            "Home": teams.get(fx["team_h"], "?"),
            "Away": teams.get(fx["team_a"], "?"),
            "Gameweek": fx.get("event"),
        })
    return pd.DataFrame(rows, columns=["Home", "Away", "Gameweek"])


def build_player_table(bootstrap):
    teams = _team_names(bootstrap)
    rows = []
    for p in bootstrap["elements"]:
        rows.append({
            "Player": p["web_name"],
            "Team": teams.get(p["team"], "?"),
            "Position": POSITIONS.get(p["element_type"], "?"),
            "pos_id": p["element_type"],
            "Goals": p.get("goals_scored", 0),
            "xG": float(p.get("expected_goals") or 0),
            "Minutes": p.get("minutes", 0),
            "Status": p.get("status", "a"),
            "ChancePlaying": p.get("chance_of_playing_next_round"),
            "News": p.get("news", ""),
            "PenaltyTaker": p.get("penalties_order") == 1,
        })
    return pd.DataFrame(rows)


def add_scoring_rates(players):
    df = players.copy()
    blended_goals = 0.5 * df["Goals"] + 0.5 * df["xG"]
    prior = df["pos_id"].map(PRIOR_RATE_90).fillna(0.05)
    df["Rate90"] = (blended_goals + prior * PRIOR_MINUTES / 90) / ((df["Minutes"] + PRIOR_MINUTES) / 90)
    team_max = df.groupby("Team")["Minutes"].transform("max").clip(lower=1)
    df["PlayShare"] = (df["Minutes"] / team_max).clip(0, 1)
    return df


def _availability(status, chance):
    if status in ("i", "s", "u", "n"):
        return 0.0
    if pd.notna(chance):
        return float(chance) / 100.0
    return 0.5 if status == "d" else 1.0


def players_to_watch(players, team, team_lambda, top_n=3):
    squad = players[players["Team"] == team].copy()
    if squad.empty:
        return squad

    games = squad["Minutes"].max() / 90.0
    team_per_game = (squad["Goals"].sum() + LEAGUE_AVG_GOALS * 4) / (games + 4)
    scale = float(np.clip(team_lambda / team_per_game, 0.4, 2.5))

    squad["Availability"] = [_availability(s, c) for s, c in zip(squad["Status"], squad["ChancePlaying"])]
    squad["ExpGoals"] = squad["Rate90"] * squad["PlayShare"] * squad["Availability"] * scale
    squad["P_Score"] = 1 - np.exp(-squad["ExpGoals"])
    return squad.sort_values("P_Score", ascending=False).head(top_n)