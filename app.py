import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import log_loss

from src.live_data import (
    LOCAL_TZ,
    LiveDataError,
    add_scoring_rates,
    all_team_names,
    build_player_table,
    fetch_bootstrap,
    fetch_fixtures,
    get_remaining,
    get_results,
    get_team_meta,
    get_upcoming_fixtures,
    players_to_watch,
    weekend_start,
)
from src.predict import (
    PROCESSED_DIR,
    earliest_replay_date,
    fit_models_as_of,
    load_artifacts,
    load_replay_data,
    predict_fixture,
)
from src.season_sim import simulate_season
from src.service import (
    best_score_for_outcome,
    calibration_curve,
    cumulative_points,
    result_table,
    season_table,
    standings_from_matches,
    table_to_pos,
    team_recent,
    team_summary,
)
from src.train import DixonColesModel, compute_metrics  # noqa: F401  (needed to load saved models)
from ui import theme as T

st.set_page_config(page_title="EPL Match Predictor", page_icon="⚽", layout="wide")
T.inject_css()


# =====================================================
# CACHED LOADERS
# =====================================================
@st.cache_resource(show_spinner="Loading models and data...")
def get_artifacts():
    return load_artifacts()


@st.cache_resource(show_spinner=False)
def get_replay_data():
    return load_replay_data()


@st.cache_resource(show_spinner="Training the models on matches before that date...")
def get_models_as_of(cutoff_iso):
    replay_df, replay_stage2 = get_replay_data()
    return fit_models_as_of(replay_df, replay_stage2, cutoff_iso)


@st.cache_data(ttl=1800, show_spinner="Fetching live fixtures, results and player stats...")
def load_live():
    bootstrap = fetch_bootstrap()
    fixtures = fetch_fixtures()
    return {
        "upcoming": get_upcoming_fixtures(bootstrap, fixtures),
        "results": get_results(bootstrap, fixtures),
        "remaining": get_remaining(bootstrap, fixtures),
        "players": add_scoring_rates(build_player_table(bootstrap)),
        "meta": get_team_meta(bootstrap),
        "teams": all_team_names(bootstrap),
    }


@st.cache_data(ttl=1800, show_spinner="Forecasting every fixture...")
def forecast_all(fixtures, _artifacts):
    rows = []
    for fx in fixtures.itertuples():
        res = predict_fixture(fx.Home, fx.Away, _artifacts, match_date=fx.MatchDate)
        probs = {fx.Home: res["P(Home)"], "Draw": res["P(Draw)"], fx.Away: res["P(Away)"]}
        rows.append({
            "Kick-off": fx.Kickoff.strftime("%a %d %b, %H:%M"),
            "Match": f"{fx.Home} vs {fx.Away}",
            "Home win %": round(res["P(Home)"] * 100),
            "Draw %": round(res["P(Draw)"] * 100),
            "Away win %": round(res["P(Away)"] * 100),
            "Most likely": max(probs, key=probs.get),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner="Simulating the rest of the season...")
def run_season_sim(results, remaining, teams, n_sims, _dc_model):
    base = standings_from_matches(results, teams=teams).set_index("Team")
    return simulate_season(base, remaining, _dc_model, n_sims)


# =====================================================
# STARTUP
# =====================================================
try:
    artifacts = get_artifacts()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
except (AttributeError, ImportError) as err:
    st.error(
        "The saved models could not be loaded, usually because they were saved with a "
        "different library or Python version. Re-run `python -m src.train` in the same "
        f"environment the app runs in. Details: {err}"
    )
    st.stop()

df, dc_model, _ = artifacts
replay_df, replay_stage2 = get_replay_data()
last_data_date = df["Date"].max().date()
first_allowed = last_data_date + dt.timedelta(days=1)
recent_teams = sorted(df.tail(760)["HomeTeam"].unique())
known_teams = set(df["HomeTeam"]).union(df["AwayTeam"])
earliest_replay = earliest_replay_date(replay_stage2)

try:
    live, live_error = load_live(), None
except LiveDataError as err:
    live, live_error = None, str(err)
meta = live["meta"] if live else {}


def split_upcoming():
    """Upcoming fixtures whose teams exist in your data, and the ones that don't."""
    if live is None or live["upcoming"].empty:
        return pd.DataFrame(), pd.DataFrame()
    up = live["upcoming"]
    ok = up["Home"].isin(known_teams) & up["Away"].isin(known_teams)
    return up[ok].reset_index(drop=True), up[~ok]


def position_lookup():
    if live is not None and not live["results"].empty:
        table = standings_from_matches(live["results"], teams=live["teams"])
        return table_to_pos(table), "live table"
    season = df["Season"].iloc[-1]
    return table_to_pos(season_table(df, season)), f"{season} table"


def get_sim(n_sims=5000):
    if live is None:
        return None
    return run_season_sim(live["results"], live["remaining"], live["teams"], n_sims, dc_model)


def read_comparison():
    path = PROCESSED_DIR / "model_comparison.csv"
    return pd.read_csv(path, index_col=0) if path.is_file() else None


# =====================================================
# SHARED CHART / UI HELPERS
# =====================================================
def donut(res, home, away):
    fig = go.Figure(go.Pie(
        labels=[f"{home} win", "Draw", f"{away} win"],
        values=[res["P(Home)"], res["P(Draw)"], res["P(Away)"]],
        hole=0.62, sort=False, textinfo="percent",
        marker=dict(colors=[T.GREEN, T.GREY, T.BLUE], line=dict(color=T.NAVY, width=3)),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Win / draw / loss probability", showlegend=True,
                      annotations=[dict(text="Result<br>odds", showarrow=False, font=dict(size=14))])
    return T.style_fig(fig, 340)


def score_heatmap(matrix):
    fig = go.Figure(go.Heatmap(
        z=matrix[:6, :6] * 100, x=list(range(6)), y=list(range(6)),
        colorscale=[[0, "#0d1a3a"], [1, T.GREEN]], texttemplate="%{z:.1f}",
        hovertemplate="Home %{y} - Away %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(title="%"),
    ))
    fig.update_layout(title="Scoreline probabilities (%)", xaxis_title="Away goals", yaxis_title="Home goals")
    fig.update_yaxes(autorange="reversed")
    return T.style_fig(fig, 340)


def stats_bar(sum_h, sum_a, home, away, window):
    metrics = [("Goals scored", "GF"), ("Goals conceded", "GA"), ("Shots", "Shots"),
               ("Shots on target", "SoT"), ("Corners", "Corners"), ("Points", "Pts")]
    fig = go.Figure()
    for name, summ, color in [(home, sum_h, T.GREEN), (away, sum_a, T.BLUE)]:
        fig.add_trace(go.Bar(
            name=name, x=[m[0] for m in metrics], y=[summ[m[1]] for m in metrics],
            marker_color=color, hovertemplate="%{x}: %{y:.2f} per game<extra>" + name + "</extra>",
        ))
    fig.update_layout(barmode="group", title=f"Per-game averages, last {window} matches", yaxis_title="Per game")
    return T.style_fig(fig, 360)


def trend_chart(rec_h, rec_a, home, away, col, title):
    fig = go.Figure()
    for rec, name, color in [(rec_h, home, T.GREEN), (rec_a, away, T.BLUE)]:
        custom = np.stack([
            rec["Opponent"].to_numpy(),
            (rec["GF"].astype(str) + "-" + rec["GA"].astype(str)).to_numpy(),
            rec["Venue"].to_numpy(),
        ], axis=-1) if len(rec) else None
        fig.add_trace(go.Scatter(
            x=rec["Date"], y=rec[col], mode="lines+markers", name=name,
            line=dict(color=color, width=3), marker=dict(size=8), customdata=custom,
            hovertemplate="%{x|%d %b %Y} (%{customdata[2]})<br>vs %{customdata[0]} %{customdata[1]}<br>"
                          + title + ": %{y}<extra>" + name + "</extra>",
        ))
    fig.update_layout(title=title, yaxis_title=title)
    return T.style_fig(fig, 320)


def render_players(home_team, away_team, lam, mu):
    st.markdown("#### ⚽ Players to look out for")
    cols = st.columns(2)
    for col, team, team_lambda in zip(cols, (home_team, away_team), (lam, mu)):
        picks = players_to_watch(live["players"], team, team_lambda)
        with col:
            st.markdown(f"**{team}**")
            if picks.empty or picks["P_Score"].iloc[0] <= 0:
                st.write("Not enough data yet.")
                continue
            top = picks.iloc[0]
            tag = " (penalty taker)" if top["PenaltyTaker"] else ""
            st.metric(f"{top['Player']}{tag}", f"{top['P_Score'] * 100:.0f}%",
                      help="Estimated chance to score at any point in the match")
            if top["Status"] != "a" and top["News"]:
                st.caption(f"⚠️ {top['News']}")
            others = [f"{r.Player} ({r.P_Score * 100:.0f}%)" for r in picks.iloc[1:].itertuples()]
            if others:
                st.caption("Also: " + ", ".join(others))
    st.caption("Estimated from this season's goals and expected goals, adjusted for the match "
               "forecast and injury news. Unlike the match forecast, this has not been backtested.")


def render_match(last, window):
    res, home, away, date = last["res"], last["home"], last["away"], last["date"]
    probs = np.array([res["P(Home)"], res["P(Draw)"], res["P(Away)"]])
    k = int(probs.argmax())
    sh, sa, sp = best_score_for_outcome(res["Matrix"], k)
    outcome = [f"{home} win", "Draw", f"{away} win"][k]

    rec_h = team_recent(df, home, window, as_of=date)
    rec_a = team_recent(df, away, window, as_of=date)
    sum_h, sum_a = team_summary(rec_h), team_summary(rec_a)
    pos = last["pos"]

    c1, c2, c3 = st.columns([1.1, 1.3, 1.1])
    c1.markdown(T.team_card(home, meta, pos.get(home), sum_h, rec_h["Result"].tail(5).tolist()),
                unsafe_allow_html=True)
    c3.markdown(T.team_card(away, meta, pos.get(away), sum_a, rec_a["Result"].tail(5).tolist()),
                unsafe_allow_html=True)
    c2.markdown(
        "<div class='card' style='text-align:center'>"
        "<div class='kpi-label'>Predicted result</div>"
        f"<div class='kpi-value'>{outcome}</div>"
        f"<div class='scorebox'>{sh} - {sa}</div>"
        f"<div class='kpi-sub'>Most likely score for that result ({sp * 100:.1f}%)</div>"
        f"<div style='margin-top:12px'><span class='pill'>{home} {probs[0] * 100:.0f}%</span>"
        f"<span class='pill'>Draw {probs[1] * 100:.0f}%</span>"
        f"<span class='pill'>{away} {probs[2] * 100:.0f}%</span></div>"
        f"<div class='kpi-sub' style='margin-top:10px'>Expected goals {res['Expected_Goals']}</div>"
        f"<div class='kpi-sub'>{date:%A %d %B %Y} · {last['pos_label']}</div></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    a, b = st.columns(2)
    a.plotly_chart(donut(res, home, away))
    b.plotly_chart(score_heatmap(res["Matrix"]))

    st.plotly_chart(stats_bar(sum_h, sum_a, home, away, window))
    st.caption("Possession isn't available in the football-data.co.uk files, so shots, "
               "shots on target and corners are used instead.")

    l1, l2 = st.columns(2)
    l1.plotly_chart(trend_chart(rec_h, rec_a, home, away, "GF", "Goals scored"))
    l2.plotly_chart(trend_chart(rec_h, rec_a, home, away, "GA", "Goals conceded"))

    if last["mode"] != "Replay a past match" and live is not None:
        lam, mu = (float(x) for x in res["Expected_Goals"].split(" - "))
        render_players(home, away, lam, mu)

    if last["actual"] is not None:
        fh, fa, ftr = last["actual"]
        actual_k = {"H": 0, "D": 1, "A": 2}[ftr]
        st.divider()
        st.subheader(f"What actually happened: {home} {fh} - {fa} {away}")
        names = [f"a {home} win", "a draw", f"a {away} win"]
        st.write(f"The model's top pick was {names[k]} ({probs[k] * 100:.0f}%). "
                 f"The actual result was {names[actual_k]}, to which it gave {probs[actual_k] * 100:.0f}%.")
        (st.success if k == actual_k else st.warning)(
            "The model's top pick was right." if k == actual_k
            else "The top pick was wrong. That happens, which is why these are probabilities."
        )

    with st.expander("Details"):
        st.write(f"Rest days used: {home} {res['Rest_Home']}, {away} {res['Rest_Away']}.")
        st.write("Forecasts combine a goals model (Dixon-Coles) with a second model (LightGBM) that adds "
                 "recent form, Elo ratings and rest days. Only matches before the match date are used.")


# =====================================================
# PAGES
# =====================================================
def page_dashboard():
    T.hero("EPL Match Predictor",
           "Dixon-Coles + LightGBM forecasts, live fixtures and a season projection")
    comp = read_comparison()
    if comp is not None:
        rows = comp[~comp.index.str.contains("Bookmaker|Naive")]
        ll_value, ll_sub = f"{rows['LogLoss'].min():.4f}", rows["LogLoss"].idxmin()[3:]
    else:
        y = replay_stage2["Target"].astype(int).to_numpy()
        ll_value = f"{compute_metrics(y, replay_stage2[['dc_p_H', 'dc_p_D', 'dc_p_A']].to_numpy())['LogLoss']:.4f}"
        ll_sub = "Dixon-Coles, walk-forward"

    ok, _ = split_upcoming()
    k = st.columns(4)
    k[0].markdown(T.kpi("Matches in database", f"{len(df):,}", f"{df['Season'].nunique()} seasons"), unsafe_allow_html=True)
    k[1].markdown(T.kpi("Latest result in data", f"{last_data_date:%d %b %Y}"), unsafe_allow_html=True)
    k[2].markdown(T.kpi("Best backtest log loss", ll_value, ll_sub + " (lower is better)"), unsafe_allow_html=True)
    k[3].markdown(T.kpi("Fixtures ahead", len(ok), "from this weekend"), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 🏆 Title race")
        sim = get_sim()
        if sim is None:
            st.info("Live data is needed for the title race.")
        else:
            top = sim["table"].sort_values("Champion", ascending=False).head(8)
            fig = go.Figure(go.Bar(
                x=top["Champion"] * 100, y=top["Team"], orientation="h", marker_color=T.GREEN,
                hovertemplate="%{y}: %{x:.1f}% to win the league<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Chance of winning the league (%)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(T.style_fig(fig, 360))
    with right:
        st.markdown("#### 📅 Next matches")
        if live is None:
            st.error(live_error)
        elif ok.empty:
            st.info("No upcoming fixtures found.")
        else:
            st.dataframe(forecast_all(ok.head(8), artifacts), hide_index=True)


def page_predictor():
    T.hero("Match Predictor", "Pick a match, tune the parameters and get a full forecast")
    modes = ["Upcoming fixture", "Custom matchup", "Replay a past match"]
    mode = st.radio("Mode", modes, horizontal=True)

    with st.expander("⚙️ Match parameters"):
        window = st.slider("Recent-form window (matches)", 3, 15, 8,
                           help="How many recent matches feed the comparison cards and charts.")
        manual = st.checkbox("Set rest days manually (otherwise worked out from the date)")
        rest_home = rest_away = None
        if manual:
            r1, r2 = st.columns(2)
            rest_home = r1.slider("Home rest days", 1, 14, 7)
            rest_away = r2.slider("Away rest days", 1, 14, 7)

    sel = None
    if mode == modes[0]:
        ok, _ = split_upcoming()
        if live is None:
            st.error(live_error)
        elif ok.empty:
            st.info("No upcoming fixtures found.")
        else:
            labels = [f"{r.Kickoff:%a %d %b %H:%M} · {r.Home} vs {r.Away}" for r in ok.itertuples()]
            i = st.selectbox("Fixture", range(len(labels)), format_func=lambda j: labels[j], key="p_fix")
            fx = ok.iloc[i]
            sel = dict(home=fx["Home"], away=fx["Away"], date=fx["MatchDate"], season=None, actual=None)

    elif mode == modes[1]:
        d = st.date_input("Match date", value=max(dt.date.today(), first_allowed), min_value=first_allowed,
                          max_value=dt.date(2030, 12, 31), key="p_date")
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home team", recent_teams, index=0, key="p_home")
        away = c2.selectbox("Away team", recent_teams, index=1, key="p_away")
        sel = dict(home=home, away=away, date=d, season=None, actual=None)

    else:
        rep = replay_df[replay_df["Date"].dt.date >= earliest_replay]
        c1, c2 = st.columns([1, 2])
        season = c1.selectbox("Season", sorted(rep["Season"].unique(), reverse=True), key="p_season")
        sm = rep[rep["Season"] == season].sort_values("Date")
        teams_in = sorted(set(sm["HomeTeam"]).union(sm["AwayTeam"]))
        picked = c2.multiselect("Filter by team (optional)", teams_in, key="p_filter")
        if picked:
            sm = sm[sm["HomeTeam"].isin(picked) | sm["AwayTeam"].isin(picked)]
        sm = sm.reset_index(drop=True)
        labels = [f"{r.Date:%d %b %Y} · {r.HomeTeam} vs {r.AwayTeam}" for r in sm.itertuples()]
        if labels:
            i = st.selectbox("Match", range(len(labels)), format_func=lambda j: labels[j], key="p_rmatch")
            row = sm.iloc[i]
            sel = dict(home=row["HomeTeam"], away=row["AwayTeam"], date=row["Date"].date(), season=season,
                       actual=(int(row["FTHG"]), int(row["FTAG"]), row["FTR"]))
            st.caption("Both models are retrained on matches before this date only. "
                       "The result is revealed after the forecast.")
        else:
            st.info("No matches for that filter.")

    if sel and st.button("⚽ Predict match", type="primary"):
        if sel["home"] == sel["away"]:
            st.error("Please pick two different teams.")
        else:
            with st.spinner("Running both models..."):
                try:
                    if mode == modes[2]:
                        dcm, lgbm = get_models_as_of(sel["date"].isoformat())
                        res = predict_fixture((sel["home"]), sel["away"], (replay_df, dcm, lgbm),
                                              rest_home, rest_away, match_date=sel["date"])
                        prior = df[(df["Season"] == sel["season"]) & (df["Date"] < pd.Timestamp(sel["date"]))]
                        tbl = standings_from_matches(prior, "HomeTeam", "AwayTeam", "FTHG", "FTAG")
                        pos, pos_label = table_to_pos(tbl), f"table before {sel['date']:%d %b %Y}"
                    else:
                        res = predict_fixture(sel["home"], sel["away"], artifacts, rest_home, rest_away,
                                              match_date=sel["date"])
                        pos, pos_label = position_lookup()
                except ValueError as err:
                    st.error(str(err))
                else:
                    st.session_state["last"] = {**sel, "res": res, "mode": mode, "pos": pos, "pos_label": pos_label}
                    k = int(np.argmax([res["P(Home)"], res["P(Draw)"], res["P(Away)"]]))
                    sh, sa, _ = best_score_for_outcome(res["Matrix"], k)
                    st.session_state.setdefault("history", []).append({
                        "Time": dt.datetime.now().strftime("%H:%M:%S"),
                        "Mode": mode,
                        "Match": f"{sel['home']} vs {sel['away']}",
                        "Match date": sel["date"],
                        "Home %": round(res["P(Home)"] * 100, 1),
                        "Draw %": round(res["P(Draw)"] * 100, 1),
                        "Away %": round(res["P(Away)"] * 100, 1),
                        "Predicted": [sel["home"], "Draw", sel["away"]][k],
                        "Score": f"{sh}-{sa}",
                    })

    last = st.session_state.get("last")
    if last:
        st.divider()
        render_match(last, window)

    hist = st.session_state.get("history", [])
    if hist:
        st.divider()
        st.markdown("#### 🕘 Prediction history (this session)")
        hdf = pd.DataFrame(hist).iloc[::-1]
        st.dataframe(hdf, hide_index=True)
        c1, c2 = st.columns([1, 5])
        c1.download_button("Download CSV", hdf.to_csv(index=False).encode(), "prediction_history.csv", "text/csv")
        if c2.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()


def page_season():
    T.hero("Season Outlook", "Who wins the league, and how the table could finish")
    if live is None:
        st.error(live_error)
        return
    n_sims = st.select_slider("Number of simulated seasons", [1000, 2500, 5000, 10000, 20000], value=5000,
                              help="More simulations give smoother percentages but take longer.")
    sim = get_sim(n_sims)
    table, pos_dist = sim["table"], sim["pos_dist"]
    left = len(live["remaining"])
    st.caption(f"{len(live['results'])} matches played, {left} still to play, simulated {n_sims:,} times.")

    top = table.sort_values("Champion", ascending=False).head(3)
    cols = st.columns(3)
    for col, (rank, r) in zip(cols, enumerate(top.itertuples(), start=1)):
        label = "Projected champion" if rank == 1 else ("Runner-up" if rank == 2 else "Third favourite")
        col.markdown(
            "<div class='card' style='text-align:center'>"
            f"{T.crest(r.Team, meta)}<div class='kpi-label' style='margin-top:8px'>{label}</div>"
            f"<div class='team-name'>{r.Team}</div>"
            f"<div class='kpi-value'>{r.Champion * 100:.1f}%</div>"
            f"<div class='kpi-sub'>Projected {r.ExpPts:.0f} pts</div></div>",
            unsafe_allow_html=True,
        )
    st.write("")

    st.markdown("#### 📋 Projected final table")
    view = pd.DataFrame({
        "Pos": table["Pos"], "Team": table["Team"], "Played": table["Played"], "Pts now": table["PtsNow"],
        "Proj. pts": table["ExpPts"].round(1), "Proj. GD": table["ExpGD"].round(0).astype(int),
        "Avg finish": table["AvgPos"].round(1),
        "Champion %": (table["Champion"] * 100).round(1),
        "Top 4 %": (table["Top4"] * 100).round(1),
        "Relegation %": (table["Relegation"] * 100).round(1),
    })
    pct = lambda title: st.column_config.ProgressColumn(title, format="%.1f%%", min_value=0, max_value=100)
    st.dataframe(view, hide_index=True, column_config={
        "Champion %": pct("Champion %"), "Top 4 %": pct("Top 4 %"), "Relegation %": pct("Relegation %"),
    })

    st.markdown("#### 🔥 Where each team could finish")
    fig = go.Figure(go.Heatmap(
        z=pos_dist.values * 100, x=list(pos_dist.columns), y=list(pos_dist.index),
        colorscale=[[0, "#0d1a3a"], [1, T.GREEN]], colorbar=dict(title="%"),
        hovertemplate="%{y}<br>Finishing %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Final league position", xaxis=dict(dtick=1))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(T.style_fig(fig, 640))

    with st.expander("Current table"):
        st.dataframe(standings_from_matches(live["results"], teams=live["teams"]), hide_index=True)

    missing = [t for t in live["teams"] if t not in dc_model.params_["attack"]]
    if missing:
        st.warning("No fitted rating for " + ", ".join(missing) + ", so slightly below-average defaults are "
                   "used. Their odds are the least reliable.")
    st.caption("Method: every remaining match is sampled from the Dixon-Coles scoreline grid (the goals "
               "model only, because the second model gives result odds but not scorelines). Ratings are not "
               "updated during the simulated season, and injuries and transfers aren't considered. "
               "Champion odds have not been backtested, and they are very uncertain early in the season.")


def page_team():
    T.hero("Team Analysis", "Season record, form and trends for any club")
    c1, c2, c3 = st.columns([2, 2, 1])
    team = c1.selectbox("Team", recent_teams, key="t_team")
    seasons = sorted(df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]["Season"].unique(), reverse=True)
    season = c2.selectbox("Season", seasons, key="t_season")
    window = c3.slider("Form window", 3, 15, 8, key="t_window")

    sdf = df[df["Season"] == season]
    tbl = season_table(df, season)
    pos = table_to_pos(tbl)
    row = tbl[tbl["Team"] == team].iloc[0]
    rec_all = team_recent(sdf, team, n=100)
    rec = rec_all.tail(window)

    k = st.columns(5)
    k[0].markdown(T.kpi("League position", f"{int(row.Pos)}", f"{int(row.Pts)} points"), unsafe_allow_html=True)
    k[1].markdown(T.kpi("Played", int(row.P)), unsafe_allow_html=True)
    k[2].markdown(T.kpi("W - D - L", f"{int(row.W)}-{int(row.D)}-{int(row.L)}"), unsafe_allow_html=True)
    k[3].markdown(T.kpi("Goals for - against", f"{int(row.GF)}-{int(row.GA)}", f"GD {int(row.GD):+d}"), unsafe_allow_html=True)
    k[4].markdown(T.kpi("Points per game", f"{row.Pts / max(row.P, 1):.2f}"), unsafe_allow_html=True)
    st.write("")

    cc, cd = st.columns([1, 2])
    cc.markdown(T.team_card(team, meta, pos.get(team), team_summary(rec), rec["Result"].tail(5).tolist()),
                unsafe_allow_html=True)
    with cd:
        others = sorted((set(sdf["HomeTeam"]) | set(sdf["AwayTeam"])) - {team})
        compare = st.multiselect("Compare with (up to 4 teams)", others, max_selections=4, key="t_cmp")
        cum = cumulative_points(df, season, [team] + compare)
        fig = px.line(cum, x="Match", y="Points", color="Team", markers=True,
                      color_discrete_sequence=[T.GREEN, T.BLUE, T.AMBER, T.RED, "#c792ea"],
                      hover_data={"Opponent": True, "Result": True})
        fig.update_layout(title=f"Cumulative points, {season}")
        st.plotly_chart(T.style_fig(fig, 340))

    a, b = st.columns(2)
    split = rec_all.groupby("Venue")[["Pts", "GF", "GA"]].mean()
    fig = go.Figure()
    for venue, name, color in [("H", "Home", T.GREEN), ("A", "Away", T.BLUE)]:
        if venue in split.index:
            fig.add_trace(go.Bar(name=name, x=["Points", "Goals scored", "Goals conceded"],
                                 y=split.loc[venue, ["Pts", "GF", "GA"]], marker_color=color,
                                 hovertemplate="%{x}: %{y:.2f} per game<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", title="Home vs away (per game)")
    a.plotly_chart(T.style_fig(fig, 320))

    roll = rec_all.copy()
    roll["Scored (5-match avg)"] = roll["GF"].rolling(5, min_periods=1).mean()
    roll["Conceded (5-match avg)"] = roll["GA"].rolling(5, min_periods=1).mean()
    fig = px.line(roll, x="Date", y=["Scored (5-match avg)", "Conceded (5-match avg)"],
                  color_discrete_sequence=[T.GREEN, T.RED], markers=True)
    fig.update_layout(title="Goal trend", yaxis_title="Goals", legend_title="")
    b.plotly_chart(T.style_fig(fig, 320))

    st.markdown("#### Match log")
    venue = st.radio("Venue", ["All", "Home", "Away"], horizontal=True, key="t_venue")
    log = rec_all.copy()
    if venue != "All":
        log = log[log["Venue"] == venue[0]]
    log["Score"] = log["GF"].astype(str) + " - " + log["GA"].astype(str)
    st.dataframe(log[["Date", "Venue", "Opponent", "Score", "Result"]].iloc[::-1], hide_index=True)


def page_fixtures():
    T.hero("Fixtures & Results", "Upcoming matches with forecasts, and historical results")
    tab_up, tab_hist = st.tabs(["📅 Upcoming", "🗂️ Historical results"])

    with tab_up:
        ok, skipped = split_upcoming()
        if live is None:
            st.error(live_error)
            st.caption("Check your internet connection and refresh.")
        elif ok.empty:
            st.info("No upcoming fixtures found.")
        else:
            st.caption(f"Fixtures from {weekend_start():%A %d %B} onwards. Kick-off times are in {LOCAL_TZ} time.")
            f1, f2 = st.columns([2, 1])
            teams_f = f1.multiselect("Filter by team", sorted(set(ok["Home"]) | set(ok["Away"])), key="f_teams")
            days = f2.slider("Days ahead", 1, 120, 30, key="f_days")
            view = ok[ok["MatchDate"] <= dt.date.today() + dt.timedelta(days=days)]
            if teams_f:
                view = view[view["Home"].isin(teams_f) | view["Away"].isin(teams_f)]
            if view.empty:
                st.info("No fixtures match those filters.")
            else:
                table = forecast_all(view.reset_index(drop=True), artifacts)
                st.dataframe(table, hide_index=True, column_config={
                    c: st.column_config.ProgressColumn(c, format="%d%%", min_value=0, max_value=100)
                    for c in ["Home win %", "Draw %", "Away win %"]
                })
                st.download_button("Download CSV", table.to_csv(index=False).encode(), "fixture_forecasts.csv", "text/csv")
            if not skipped.empty:
                names = sorted((set(skipped["Home"]) | set(skipped["Away"])) - known_teams)
                st.warning(f"Skipped fixtures with team names not found in your data: {', '.join(names)}. "
                           "Add them to FPL_TO_DATASET in src/live_data.py.")

    with tab_hist:
        all_seasons = sorted(df["Season"].unique(), reverse=True)
        h1, h2, h3 = st.columns([2, 2, 1])
        seasons_f = h1.multiselect("Season", all_seasons, default=all_seasons[:1], key="h_seasons")
        teams_h = h2.multiselect("Team", sorted(known_teams), key="h_teams")
        outcome = h3.selectbox("Result", ["All", "Home wins", "Draws", "Away wins"], key="h_out")
        res_tbl = result_table(df, seasons_f, teams_h, outcome)
        st.caption(f"{len(res_tbl):,} matches")
        st.dataframe(res_tbl.head(1000), hide_index=True)


def page_performance():
    T.hero("Model Performance", "Honest walk-forward results: every prediction uses only earlier matches")
    comp = read_comparison()
    y = replay_stage2["Target"].astype(int).to_numpy()
    s1 = pd.DataFrame({
        "Simple Poisson": compute_metrics(y, replay_stage2[["pois_p_H", "pois_p_D", "pois_p_A"]].to_numpy()),
        "Dixon-Coles": compute_metrics(y, replay_stage2[["dc_p_H", "dc_p_D", "dc_p_A"]].to_numpy()),
    }).T

    if comp is not None:
        metric = st.selectbox("Metric (lower is better)", ["LogLoss", "Brier", "RPS"])
        colors = [T.AMBER if "Bookmaker" in m else T.GREEN for m in comp.index]
        fig = go.Figure(go.Bar(x=comp[metric], y=comp.index, orientation="h", marker_color=colors,
                               hovertemplate="%{y}<br>" + metric + ": %{x:.4f}<extra></extra>"))
        fig.update_layout(xaxis=dict(range=[comp[metric].min() * 0.97, comp[metric].max() * 1.01]))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(T.style_fig(fig, 340))
        st.dataframe(comp)
        st.caption("Bookmaker row covers only matches with valid Bet365 odds, so treat that comparison as approximate.")
    else:
        st.info("Showing Stage 1 results over all walk-forward matches. Add the three save lines to "
                "`train.py` and re-run `python -m src.train` to include LightGBM and the bookmaker benchmark.")
        st.dataframe(s1)

    c1, c2 = st.columns(2)
    stage2_path = PROCESSED_DIR / "backtest_stage2.csv"
    if stage2_path.is_file():
        s2 = pd.read_csv(stage2_path)
        source = c1.selectbox("Model for calibration chart", ["LightGBM (uncalibrated)", "LightGBM (calibrated)", "Dixon-Coles"])
        if source == "Dixon-Coles":
            prob, hit = replay_stage2["dc_p_H"].to_numpy(), (y == 0)
        else:
            prob = s2["lgbm_p_H" if "uncal" in source else "cal_p_H"].to_numpy()
            hit = (s2["Target"].to_numpy() == 0)
    else:
        prob, hit = replay_stage2["dc_p_H"].to_numpy(), (y == 0)
        source = "Dixon-Coles"
    curve = calibration_curve(prob, hit)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(color=T.GREY, dash="dash")))
    fig.add_trace(go.Scatter(x=curve["Predicted"], y=curve["Observed"], mode="lines+markers", name=source,
                             line=dict(color=T.GREEN), marker=dict(size=10), customdata=curve["Matches"],
                             hovertemplate="Predicted %{x:.0%}<br>Observed %{y:.0%}<br>%{customdata} matches<extra></extra>"))
    fig.update_layout(title="Home-win calibration", xaxis_title="Predicted probability", yaxis_title="Actual frequency")
    c1.plotly_chart(T.style_fig(fig, 340))

    rows = []
    for season, g in replay_stage2.groupby("Season"):
        p, t = g[["dc_p_H", "dc_p_D", "dc_p_A"]].to_numpy(), g["Target"].astype(int).to_numpy()
        rows.append({"Season": season, "Accuracy %": (p.argmax(axis=1) == t).mean() * 100,
                     "Log loss": log_loss(t, p, labels=[0, 1, 2])})
    per = pd.DataFrame(rows)
    fig = px.line(per, x="Season", y="Accuracy %", markers=True, color_discrete_sequence=[T.GREEN],
                  hover_data={"Log loss": ":.3f"})
    fig.update_layout(title="Result accuracy by season (Dixon-Coles)")
    c2.plotly_chart(T.style_fig(fig, 340))
    st.caption("A single season is only about 380 matches, so season-to-season swings are mostly noise.")


# =====================================================
# SIDEBAR + ROUTER
# =====================================================
PAGES = {
    "📊 Dashboard": page_dashboard,
    "🎯 Match Predictor": page_predictor,
    "🏆 Season Outlook": page_season,
    "🔬 Team Analysis": page_team,
    "📅 Fixtures": page_fixtures,
    "📈 Model Performance": page_performance,
}

with st.sidebar:
    st.markdown("## ⚽ EPL Predictor")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(f"Match data through {last_data_date:%d %b %Y}")
    st.caption("🟢 Live data connected" if live else "🔴 Live data unavailable")
    if st.button("🔄 Refresh live data"):
        st.cache_data.clear()
        st.rerun()

PAGES[page]()