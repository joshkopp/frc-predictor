"""Alliance Recommender page. Loaded by streamlit_app.py (the router)."""
import datetime as dt
import os

import pandas as pd
import streamlit as st

import frc_alliance as fa
import ui


# ---------------------------------------------------------------- API key
def secret_key():
    for name in ("TBA_API_KEY", "TBA_KEY", "tba_key"):
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
        if os.environ.get(name):
            return os.environ[name]
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_events(year, key):
    return fa.list_events(year, key)


@st.cache_data(ttl=120, show_spinner="Pulling event data from The Blue Alliance...")
def cached_event(event_key, key):
    return fa.load_event(event_key, key)


hero_slot = st.empty()
hero_slot.html(ui.a_hero("", 0, 0, ""))

api_key = secret_key()
if not api_key:
    with st.container(key="keybox"):
        st.html('<div class="jk"><div class="jk-eyebrow">Connect</div></div>')
        api_key = st.text_input("The Blue Alliance API key", type="password", placeholder="Paste your TBA read key",
                                help="Free at thebluealliance.com/account. Stays in this session only.")
    if not api_key:
        st.html(ui.a_empty("One step first",
                           'Add a free <a href="https://www.thebluealliance.com/account" target="_blank">TBA read key</a> above '
                           '(or set <code>TBA_API_KEY</code> in Streamlit secrets) to load events.'))
        st.stop()

# ---------------------------------------------------------------- event + alliance controls
st.html(ui.section("Setup", "Pick your event", "Choose an event, then tell it who you are and who's already taken."))
with st.container(key="ctl"):
    e1, e2, e3 = st.columns([1, 3, 2], gap="medium")
    year = e1.number_input("Season", 1992, dt.date.today().year + 1, 2026)
    try:
        events = cached_events(int(year), api_key)
    except Exception as e:
        st.error(f"Couldn't load events: {e}")
        st.stop()
    labels = {f'{e["name"]} ({e["key"]})': e["key"] for e in events}
    names = {e["key"]: e["name"] for e in events}
    if not labels:
        st.html(ui.a_empty("No events", f"No events found for {int(year)}."))
        st.stop()
    choice = e2.selectbox("Event", list(labels))
    manual = e3.text_input("…or type an event key", placeholder="2026txhou")
    event_key = manual.strip().lower() or labels[choice]

try:
    event = cached_event(event_key, api_key)
except Exception as e:
    st.error(f"Couldn't load {event_key}: {e}")
    st.stop()

with st.container(key="ctl2"):
    include_playoffs = st.toggle("Include playoff matches in stats", False)
df, fields = fa.build_team_table(event, include_playoffs)
if df.empty:
    st.html(ui.a_empty("No teams yet", "No teams are registered for this event yet."))
    st.stop()
played = int(len({m["key"] for m in event["matches"] if (m.get("alliances") or {}).get("red", {}).get("score", -1) >= 0}))
hero_slot.html(ui.a_hero(names.get(event_key, event_key), len(df), played, ""))
if df["played"].fillna(0).sum() == 0:
    st.html('<div class="jk"><div class="jk-warn">No matches played yet, so stats are empty. Check back once quals start.</div></div>')

team_label = {k: f'{r.team} · {r["name"]}' if r["name"] else str(r.team) for k, r in df.iterrows()}
by_rank = df.sort_values("rank", na_position="last").index.tolist()

with st.container(key="ctl3"):
    a1, a2 = st.columns([1, 2], gap="medium")
    captain = a1.selectbox("Captain (you)", by_rank, format_func=team_label.get)
    picked = a2.multiselect("Already picked (draft mode)", [t for t in by_rank if t != captain],
                            format_func=team_label.get, placeholder="Mark teams as they're taken",
                            help="Mark teams as they're taken during alliance selection.")
    b1, b2 = st.columns([1, 2], gap="medium")
    points_weight = b1.slider("Points ◀ ▶ synergy", 0.0, 1.0, 0.6, 0.05,
                              help="1.0 = pure raw points, 0.0 = pure synergy")
    syn_fields = b2.multiselect("Synergy categories", fields, fa.default_synergy_fields(fields),
                                format_func=fa.pretty_field,
                                help="Scoring categories used to measure how well a partner covers your weak spots.")

ranked = fa.recommend(df, captain, syn_fields, picked, points_weight)

# ---------------------------------------------------------------- captain summary
cap = df.loc[captain]
weak = "—"
if syn_fields:
    norm = df[syn_fields].clip(lower=0)
    norm = (norm / norm.max().replace(0, pd.NA)).fillna(0).loc[captain]
    weak = fa.pretty_field(norm.idxmin())
st.html(ui.section("You", f"Team {int(cap['team'])}", cap["name"] or ""))
st.html(ui.a_kpis([
    ("—" if pd.isna(cap["rank"]) else f'<em>#</em>{int(cap["rank"])}', "Event rank", cap["record"] and f'Record {cap["record"]}' or "No record yet"),
    (f'{cap["opr"]:.1f}', "OPR", "Points added per match."),
    ("—" if pd.isna(cap["avg_score"]) else f'{cap["avg_score"]:.0f}', "Avg alliance score", f'{int(cap["played"] or 0)} matches played'),
    (f'<span style="font-size:.62em;letter-spacing:-.03em">{ui.esc(weak)}</span>', "Weakest area", "What a partner should cover."),
]))

if ranked.empty:
    st.html(ui.a_empty("Everyone's been picked", "No teams left in the pool."))
    st.stop()

# ---------------------------------------------------------------- top 3
st.html(ui.section("Recommendations", "Top picks", "Best partners right now, given who's still available."))
st.html(ui.a_picks([{"pick": int(r["pick"]), "team": int(r["team"]), "name": r["name"], "score": r["pick_score"],
                     "synergy": float(r["synergy"]), "points": float(r["points_score"]), "proj": r["proj_alliance_pts"],
                     "fit": r["best_fit"]} for _, r in ranked.head(3).iterrows()]))

# ---------------------------------------------------------------- full table
st.html(ui.section("Full board", f"All {len(ranked)} available teams", "Sort any column. Scores are 0-100 relative to this pool."))
show = ranked.reset_index(drop=True)
show["win_rate"] = show["win_rate"] * 100
comp_cols = [f for f in fields if f in show.columns]
show = show.rename(columns={f: fa.pretty_field(f) for f in comp_cols})
col_cfg = {
    "pick": st.column_config.NumberColumn("#", width="small"),
    "team": st.column_config.NumberColumn("Team", format="%d"),
    "name": "Name",
    "pick_score": st.column_config.ProgressColumn("Pick score", min_value=0, max_value=100, format="%.0f"),
    "synergy": st.column_config.ProgressColumn("Synergy", min_value=0, max_value=100, format="%.0f"),
    "points_score": st.column_config.ProgressColumn("Points", min_value=0, max_value=100, format="%.0f"),
    "proj_alliance_pts": st.column_config.NumberColumn("Proj. alliance pts", format="%.1f"),
    "best_fit": "Best fit",
    "opr": st.column_config.NumberColumn("OPR", format="%.1f"),
    "dpr": st.column_config.NumberColumn("DPR", format="%.1f", help="Lower is better"),
    "ccwm": st.column_config.NumberColumn("CCWM", format="%.1f"),
    "rank": st.column_config.NumberColumn("Rank", format="%d"),
    "record": "W-L-T",
    "win_rate": st.column_config.NumberColumn("Win %", format="%.0f%%"),
    "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f"),
    "recent_avg": st.column_config.NumberColumn("Last 4 avg", format="%.1f", help="Recent form: average of their last 4 matches"),
    "std_score": st.column_config.NumberColumn("Consistency σ", format="%.1f",
                                               help="Std. dev of alliance score - lower = more consistent"),
    "played": st.column_config.NumberColumn("Played", format="%d"),
}
for f in comp_cols:
    col_cfg[fa.pretty_field(f)] = st.column_config.NumberColumn(fa.pretty_field(f), format="%.1f")
order = ["pick", "team", "name", "pick_score", "synergy", "points_score", "proj_alliance_pts",
         "best_fit", "opr", "dpr", "ccwm", "rank", "record", "win_rate", "avg_score",
         "recent_avg", "std_score", "played", *[fa.pretty_field(f) for f in comp_cols]]
with st.container(key="board"):
    st.dataframe(show[order], column_config=col_cfg, hide_index=True, width="stretch",
                 height=min(38 * (len(show) + 1), 620))
    st.download_button("Download CSV", ranked.to_csv().encode(), f"{event_key}_{captain}_picks.csv",
                       icon=":material/download:", width="content")

# ---------------------------------------------------------------- compare
if fields:
    st.html(ui.section("Head to head", "Compare a candidate", "Their contribution in each scoring category next to yours."))
    with st.container(key="ctl4"):
        cand = st.selectbox("Candidate", ranked.index.tolist(), format_func=team_label.get)
        st.html(ui.a_compare([(fa.pretty_field(f), float(df.at[captain, f]), float(df.at[cand, f])) for f in fields],
                             f"You · {int(cap['team'])}", f"Candidate · {int(df.at[cand, 'team'])}"))

st.html(ui.section("Method", "How the scores work"))
st.html(ui.a_how())
