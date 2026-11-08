import datetime as dt
import os

import altair as alt
import pandas as pd
import streamlit as st

import frc_alliance as fa

st.set_page_config(page_title="FRC Alliance Recommender", layout="wide")
st.title("🤝 FRC Alliance Recommender")
st.caption("Ranks every team at an event as a partner for your alliance, by points and synergy.")


# ---------------------------------------------------------------- API key
def get_api_key():
    for name in ("TBA_API_KEY", "TBA_KEY", "tba_key"):
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
        if os.environ.get(name):
            return os.environ[name]
    return st.sidebar.text_input("TBA API key", type="password",
                                 help="Get one at thebluealliance.com/account")


api_key = get_api_key()
if not api_key:
    st.info("Add your TBA API key (sidebar, or `TBA_API_KEY` in Streamlit secrets) to start.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_events(year, key):
    return fa.list_events(year, key)


@st.cache_data(ttl=120, show_spinner="Pulling event data from TBA...")
def cached_event(event_key, key):
    return fa.load_event(event_key, key)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Event")
    year = st.number_input("Season", 1992, dt.date.today().year + 1, 2026)
    try:
        events = cached_events(int(year), api_key)
    except Exception as e:
        st.error(f"Couldn't load events: {e}")
        st.stop()
    labels = {f'{e["name"]} ({e["key"]})': e["key"] for e in events}
    manual = st.text_input("…or type an event key", placeholder="2026txhou")
    event_key = manual.strip().lower() or labels[st.selectbox("Event", list(labels))]
    include_playoffs = st.checkbox("Include playoff matches in stats", False)

try:
    event = cached_event(event_key, api_key)
except Exception as e:
    st.error(f"Couldn't load {event_key}: {e}")
    st.stop()

df, fields = fa.build_team_table(event, include_playoffs)
if df.empty:
    st.warning("No teams found for this event yet.")
    st.stop()
if df["played"].fillna(0).sum() == 0:
    st.warning("No matches played yet, so the stats are empty. Check back once quals start.")

team_label = {k: f'{r.team} – {r["name"]}' if r["name"] else str(r.team) for k, r in df.iterrows()}
by_rank = df.sort_values("rank", na_position="last").index.tolist()

with st.sidebar:
    st.header("Your alliance")
    captain = st.selectbox("Captain (you)", by_rank, format_func=team_label.get)
    picked = st.multiselect("Already picked (draft mode)", [t for t in by_rank if t != captain],
                            format_func=team_label.get,
                            help="Mark teams as they're taken during alliance selection.")
    st.header("Scoring")
    points_weight = st.slider("Points vs synergy", 0.0, 1.0, 0.6, 0.05,
                              help="1.0 = pure raw points, 0.0 = pure synergy")
    syn_fields = st.multiselect("Synergy categories", fields, fa.default_synergy_fields(fields),
                                format_func=fa.pretty_field,
                                help="Scoring categories used to measure how well a partner "
                                     "covers your weak spots.")

ranked = fa.recommend(df, captain, syn_fields, picked, points_weight)

# ---------------------------------------------------------------- captain summary
cap = df.loc[captain]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Captain", team_label[captain])
c2.metric("Rank", "—" if pd.isna(cap["rank"]) else int(cap["rank"]))
c3.metric("OPR", f'{cap["opr"]:.1f}')
c4.metric("Avg alliance score", "—" if pd.isna(cap["avg_score"]) else f'{cap["avg_score"]:.1f}')
if syn_fields:
    norm = df[syn_fields].clip(lower=0)
    norm = (norm / norm.max().replace(0, pd.NA)).fillna(0).loc[captain]
    c5.metric("Weakest area", fa.pretty_field(norm.idxmin()))

if ranked.empty:
    st.info("Everyone's been picked!")
    st.stop()

# ---------------------------------------------------------------- top 3
st.subheader("Top picks")
cols = st.columns(3)
for col, (k, r) in zip(cols, ranked.head(3).iterrows()):
    with col.container(border=True):
        st.markdown(f"### #{r['pick']} · {team_label[k]}")
        st.write(f"**Pick score {r['pick_score']:.0f}** · Synergy {r['synergy']:.0f} · "
                 f"Points {r['points_score']:.0f}")
        st.write(f"Projected alliance: **{r['proj_alliance_pts']:.0f} pts**")
        if r["best_fit"]:
            st.write(f"Covers your weak spot: **{r['best_fit']}**")

# ---------------------------------------------------------------- full table
st.subheader(f"All {len(ranked)} available teams")
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
    "recent_avg": st.column_config.NumberColumn("Last 4 avg", format="%.1f",
                                                help="Recent form: average of their last 4 matches"),
    "std_score": st.column_config.NumberColumn("Consistency σ", format="%.1f",
                                               help="Std. dev of alliance score - lower = more consistent"),
    "played": st.column_config.NumberColumn("Played", format="%d"),
}
for f in comp_cols:
    col_cfg[fa.pretty_field(f)] = st.column_config.NumberColumn(fa.pretty_field(f), format="%.1f")

order = ["pick", "team", "name", "pick_score", "synergy", "points_score", "proj_alliance_pts",
         "best_fit", "opr", "dpr", "ccwm", "rank", "record", "win_rate", "avg_score",
         "recent_avg", "std_score", "played", *[fa.pretty_field(f) for f in comp_cols]]
st.dataframe(show[order], column_config=col_cfg, hide_index=True, width="stretch",
             height=min(38 * (len(show) + 1), 700))

st.download_button("Download CSV", ranked.to_csv().encode(), f"{event_key}_{captain}_picks.csv")

# ---------------------------------------------------------------- compare
if fields:
    st.subheader("Compare a candidate")
    cand = st.selectbox("Candidate", ranked.index.tolist(), format_func=team_label.get)
    long = []
    for f in fields:
        long += [
            {"Category": fa.pretty_field(f), "Who": "You", "Points": df.at[captain, f]},
            {"Category": fa.pretty_field(f), "Who": "Candidate", "Points": df.at[cand, f]},
        ]
    chart = (alt.Chart(pd.DataFrame(long))
             .mark_bar()
             .encode(y=alt.Y("Category:N", sort=None, title=None),
                     x=alt.X("Points:Q", title="Contribution (component OPR)"),
                     color=alt.Color("Who:N", scale=alt.Scale(range=["#4c78a8", "#f58518"])),
                     yOffset="Who:N",
                     tooltip=["Who", "Category", alt.Tooltip("Points:Q", format=".1f")])
             .properties(height=max(40 * len(fields), 160)))
    st.altair_chart(chart, width="stretch")

with st.expander("How the scores work"):
    st.markdown("""
- **Points score**: projected alliance score = your OPR + their OPR + a typical 3rd robot
  (the median OPR of teams still available), scaled 0-100 across the pool.
- **Synergy**: for each scoring category we work out every team's contribution
  (component OPR from TBA score breakdowns). A partner earns synergy for being strong where
  *you* are weak, weighted by how many points that category is worth at this event.
  Scaled 0-100.
- **Pick score**: a blend of the two, set with the slider.
- **Best fit**: the category where they help you most.
""")
