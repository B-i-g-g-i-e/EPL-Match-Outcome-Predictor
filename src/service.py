import numpy as np
import pandas as pd


def standings_from_matches(matches, home="Home", away="Away", hg="HG", ag="AG", teams=None):
    """League table from a table of finished matches. Pass `teams` to include clubs with 0 games."""
    cols = ["P", "W", "D", "L", "GF", "GA", "Pts"]
    if matches is None or len(matches) == 0:
        base = pd.DataFrame(columns=cols, dtype=float)
    else:
        h = pd.DataFrame({"Team": matches[home].values, "GF": matches[hg].values, "GA": matches[ag].values})
        a = pd.DataFrame({"Team": matches[away].values, "GF": matches[ag].values, "GA": matches[hg].values})
        long = pd.concat([h, a], ignore_index=True)
        long["W"] = (long["GF"] > long["GA"]).astype(int)
        long["D"] = (long["GF"] == long["GA"]).astype(int)
        long["L"] = (long["GF"] < long["GA"]).astype(int)
        long["Pts"] = 3 * long["W"] + long["D"]
        long["P"] = 1
        base = long.groupby("Team")[cols].sum()

    if teams is not None:
        base = base.reindex(teams).fillna(0)
    base = base.astype(int)
    base["GD"] = base["GF"] - base["GA"]
    base = base.sort_values(["Pts", "GD", "GF"], ascending=False)
    base.insert(0, "Pos", range(1, len(base) + 1))
    base.index.name = "Team"
    return base.reset_index()


def season_table(df, season):
    sub = df[df["Season"] == season]
    return standings_from_matches(sub, "HomeTeam", "AwayTeam", "FTHG", "FTAG")


def table_to_pos(table):
    return {r.Team: (r.Pos, r.Pts, r.P) for r in table.itertuples()}


def team_recent(df, team, n=10, as_of=None):
    """A team's last n matches before `as_of` (no later data is ever used)."""
    d = df if as_of is None else df[df["Date"] < pd.Timestamp(as_of)]
    d = d[(d["HomeTeam"] == team) | (d["AwayTeam"] == team)].sort_values("Date").tail(n)
    is_home = (d["HomeTeam"] == team).to_numpy()

    out = pd.DataFrame({
        "Date": d["Date"].to_numpy(),
        "Opponent": np.where(is_home, d["AwayTeam"], d["HomeTeam"]),
        "Venue": np.where(is_home, "H", "A"),
        "GF": np.where(is_home, d["FTHG"], d["FTAG"]),
        "GA": np.where(is_home, d["FTAG"], d["FTHG"]),
    })
    for name, h, a in [("Shots", "HS", "AS"), ("SoT", "HST", "AST"), ("Corners", "HC", "AC")]:
        out[name] = np.where(is_home, d[h], d[a]) if h in d.columns and a in d.columns else np.nan

    out["Result"] = np.select([out["GF"] > out["GA"], out["GF"] == out["GA"]], ["W", "D"], "L")
    out["Pts"] = out["Result"].map({"W": 3, "D": 1, "L": 0})
    return out.reset_index(drop=True)


def team_summary(recent):
    cols = ["GF", "GA", "Shots", "SoT", "Corners", "Pts"]
    if recent.empty:
        return {c: np.nan for c in cols}
    return recent[cols].astype(float).mean().to_dict()


def cumulative_points(df, season, teams):
    rows = []
    sub = df[df["Season"] == season]
    for t in teams:
        rec = team_recent(sub, t, n=100)
        rec["Points"] = rec["Pts"].cumsum()
        rec["Match"] = np.arange(1, len(rec) + 1)
        rec["Team"] = t
        rows.append(rec[["Team", "Match", "Points", "Opponent", "Result", "Date"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def result_table(df, seasons=None, teams=None, outcome="All"):
    d = df
    if seasons:
        d = d[d["Season"].isin(seasons)]
    if teams:
        d = d[d["HomeTeam"].isin(teams) | d["AwayTeam"].isin(teams)]
    if outcome != "All":
        d = d[d["FTR"] == {"Home wins": "H", "Draws": "D", "Away wins": "A"}[outcome]]
    out = pd.DataFrame({
        "Date": d["Date"].dt.date,
        "Season": d["Season"],
        "Home": d["HomeTeam"],
        "Score": d["FTHG"].astype(int).astype(str) + " - " + d["FTAG"].astype(int).astype(str),
        "Away": d["AwayTeam"],
        "Result": d["FTR"].map({"H": "Home win", "D": "Draw", "A": "Away win"}),
    })
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


def best_score_for_outcome(matrix, outcome):
    """Most likely scoreline that agrees with the predicted result (0 = home, 1 = draw, 2 = away)."""
    i, j = np.indices(matrix.shape)
    mask = (i > j) if outcome == 0 else (i == j) if outcome == 1 else (i < j)
    best = np.unravel_index(np.where(mask, matrix, -1.0).argmax(), matrix.shape)
    return int(best[0]), int(best[1]), float(matrix[best])


def calibration_curve(prob, hit, bins=10):
    prob, hit = np.asarray(prob, dtype=float), np.asarray(hit, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            rows.append({"Predicted": prob[m].mean(), "Observed": hit[m].mean(), "Matches": int(m.sum())})
    return pd.DataFrame(rows)