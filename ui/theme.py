import html

import plotly.graph_objects as go
import streamlit as st

NAVY, PANEL, BORDER = "#0a1128", "#111b3a", "#1f3064"
GREEN, BLUE, RED, AMBER = "#00e676", "#4da3ff", "#ff5252", "#ffb300"
MUTED, TEXT, GREY = "#8892b0", "#e8eefc", "#6b7694"

# Club badges are hot-linked from the Premier League site by FPL team code.
# They are club trademarks: fine for a personal project, check the terms before
# using them in anything public or commercial.
CREST_URL = "https://resources.premierleague.com/premierleague/badges/70/t{code}.png"

CSS = """
<style>
.stApp { background: linear-gradient(160deg, #060b1d 0%, #0a1128 55%, #0d1a3a 100%); }
[data-testid="stSidebar"] { background: #070d22; border-right: 1px solid #1f3064; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
.hero { background: linear-gradient(120deg, #0f2a1f 0%, #111b3a 60%); border: 1px solid #1f3064;
        border-left: 4px solid #00e676; border-radius: 14px; padding: 18px 22px; margin-bottom: 16px; }
.hero h1 { margin: 0; font-size: 1.7rem; padding: 0; }
.hero p { margin: 4px 0 0 0; color: #8892b0; }
.card { background: #111b3a; border: 1px solid #1f3064; border-radius: 14px; padding: 16px;
        transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease; height: 100%; }
.card:hover { transform: translateY(-3px); border-color: #00e676; box-shadow: 0 8px 24px rgba(0,230,118,.12); }
.kpi-label { color: #8892b0; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #00e676; line-height: 1.25; }
.kpi-sub { color: #8892b0; font-size: .82rem; }
.crest { width: 56px; height: 56px; object-fit: contain; }
.badge { display: inline-flex; width: 56px; height: 56px; border-radius: 50%; align-items: center;
         justify-content: center; background: #1a2a55; border: 2px solid #00e676; font-weight: 700; color: #e8eefc; }
.team-name { font-size: 1.25rem; font-weight: 700; margin: 6px 0 2px 0; }
.chip { display: inline-block; width: 26px; height: 26px; line-height: 26px; text-align: center;
        border-radius: 6px; font-weight: 700; font-size: .8rem; margin-right: 4px; color: #fff; }
.chip-W { background: #1b8f4c; } .chip-D { background: #6b7694; } .chip-L { background: #c0392b; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; background: #1a2a55;
        color: #e8eefc; font-size: .82rem; margin: 2px; }
.stat-row { display: flex; justify-content: space-between; color: #8892b0; font-size: .88rem;
            padding: 4px 0; border-bottom: 1px solid #18264d; }
.stat-row b { color: #e8eefc; }
.scorebox { font-size: 3rem; font-weight: 800; line-height: 1.1; margin: 6px 0; }
div.stButton > button[kind="primary"] { background: #00e676; color: #04210f; font-weight: 700; border: none;
        border-radius: 10px; padding: .6rem 1.4rem; transition: all .15s ease; }
div.stButton > button[kind="primary"]:hover { background: #33ff9a; transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(0,230,118,.3); }
@media (max-width: 768px) { .scorebox { font-size: 2.2rem; } .kpi-value { font-size: 1.4rem; } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle=""):
    st.markdown(
        f"<div class='hero'><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def kpi(label, value, sub=""):
    return (
        f"<div class='card'><div class='kpi-label'>{html.escape(str(label))}</div>"
        f"<div class='kpi-value'>{html.escape(str(value))}</div>"
        f"<div class='kpi-sub'>{html.escape(str(sub))}</div></div>"
    )


def _fmt(x, d=1):
    try:
        if x != x:  # NaN
            return "–"
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "–"


def _ordinal(n):
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def crest(team, meta):
    info = (meta or {}).get(team)
    short = html.escape(info["short"] if info else team[:3].upper())
    if info:
        img = f"<img class='crest' src='{CREST_URL.format(code=info['code'])}' alt='{short}'/>"
    else:
        img = f"<div class='badge'>{short}</div>"
    return f"<div style='display:flex;justify-content:center'>{img}</div>"


def form_chips(results):
    if not results:
        return "<span class='kpi-sub'>No recent matches</span>"
    return "".join(f"<span class='chip chip-{r}'>{r}</span>" for r in results)


def team_card(team, meta, pos_info, summary, results, pos_label=""):
    if pos_info:
        pos, pts, _played = pos_info
        pos_txt = f"{_ordinal(pos)} · {int(pts)} pts"
    else:
        pos_txt = "Position n/a"
    if pos_label:
        pos_txt += f" · {pos_label}"
    rows = [
        ("Goals scored / game", _fmt(summary["GF"], 2)),
        ("Goals conceded / game", _fmt(summary["GA"], 2)),
        ("Shots / game", _fmt(summary["Shots"])),
        ("Shots on target / game", _fmt(summary["SoT"])),
        ("Points / game", _fmt(summary["Pts"], 2)),
    ]
    rows_html = "".join(f"<div class='stat-row'><span>{k}</span><b>{v}</b></div>" for k, v in rows)
    return (
        "<div class='card' style='text-align:center'>"
        f"{crest(team, meta)}<div class='team-name'>{html.escape(team)}</div>"
        f"<div class='kpi-sub'>{html.escape(pos_txt)}</div>"
        f"<div style='margin:10px 0'>{form_chips(results)}</div>{rows_html}</div>"
    )


def style_fig(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GREEN, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor="#18264d", zeroline=False)
    fig.update_yaxes(gridcolor="#18264d", zeroline=False)
    return fig