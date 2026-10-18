import numpy as np
import pandas as pd


def simulate_season(base_table, remaining, dc_model, n_sims=5000, seed=42):
    """
    Monte Carlo simulation of the rest of the season.

    base_table: DataFrame indexed by Team with columns P, Pts, GF, GA (the table so far).
    remaining:  DataFrame with Home and Away columns (matches still to play).
    Each remaining match is sampled from the Dixon-Coles scoreline grid.
    """
    teams = list(base_table.index)
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)

    pts = np.tile(base_table["Pts"].to_numpy(dtype=float), (n_sims, 1))
    gf = np.tile(base_table["GF"].to_numpy(dtype=float), (n_sims, 1))
    ga = np.tile(base_table["GA"].to_numpy(dtype=float), (n_sims, 1))

    played = 0
    for fx in remaining.itertuples():
        if fx.Home not in idx or fx.Away not in idx:
            continue
        grid = dc_model.predict_match(fx.Home, fx.Away)["matrix"]
        flat = grid.ravel() / grid.sum()
        draw = rng.choice(flat.size, size=n_sims, p=flat)
        hg, ag = draw // grid.shape[1], draw % grid.shape[1]
        h, a = idx[fx.Home], idx[fx.Away]

        pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg
        played += 1

    gd = gf - ga
    # Points, then goal difference, then goals scored, then a coin flip
    key = pts * 1e6 + gd * 1e3 + gf + rng.random(pts.shape) * 0.5
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(n), order.shape), axis=1)

    pos_probs = np.stack([(pos == p).sum(axis=0) for p in range(n)], axis=1) / n_sims

    table = pd.DataFrame({
        "Team": teams,
        "Played": base_table["P"].to_numpy(),
        "PtsNow": base_table["Pts"].to_numpy(),
        "ExpPts": pts.mean(axis=0),
        "ExpGD": gd.mean(axis=0),
        "AvgPos": (pos + 1).mean(axis=0),
        "Champion": pos_probs[:, 0],
        "Top4": pos_probs[:, : min(4, n)].sum(axis=1),
        "Relegation": pos_probs[:, max(n - 3, 0):].sum(axis=1),
    })
    table = table.sort_values(["ExpPts", "ExpGD"], ascending=False).reset_index(drop=True)
    table.insert(0, "Pos", range(1, n + 1))

    pos_dist = pd.DataFrame(pos_probs, index=teams, columns=range(1, n + 1)).loc[table["Team"]]
    return {"table": table, "pos_dist": pos_dist, "matches_simulated": played, "n_sims": n_sims}