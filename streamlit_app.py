"""FRC 2026 (REBUILT) match predictor: a leakage-safe ML demo.

Run locally:  streamlit run streamlit_app.py
"""
import json
import pathlib
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from src import config
from src.features import _row, TeamState

D = pathlib.Path(__file__).parent / "deploy"

st.set_page_config(page_title="FRC 2026 Match Predictor", page_icon="🤖", layout="wide")


@st.cache_resource
def load():
    meta = json.loads((D / "app_meta.json").read_text())
    raw = json.loads((D / "team_ratings.json").read_text())
    states = {}
    for num, r in raw.items():
        s = TeamState()
        s.n, s.rating, s.wins = r["n"], r["rating"], r["wins"]
        s.recent = deque(r["recent"], maxlen=meta["recent_n"])
        s.n_bd, s.comp = r["n_bd"], dict(r["comp"])
        s.end_sum, s.atower_sum = r["end_sum"], r["atower_sum"]
        states[f"frc{num}"] = s
    win, diff = xgb.Booster(), xgb.Booster()
    win.load_model(str(D / "win.json"))
    diff.load_model(str(D / "score_diff.json"))
    return meta, states, win, diff


meta, states, win_model, diff_model = load()
ctx = meta["ctx"]

NAMES = {
    "rating_sum": "overall rating (sum of 3 robots)", "rating_max": "best robot's rating",
    "rating_min": "weakest robot's rating", "recent": "recent scoring form",
    "winrate": "win rate", "tele_sum": "teleop (hub) scoring", "auto_sum": "autonomous (hub) scoring",
    "tower_sum": "tower scoring", "end_sum": "measured endgame tower ability",
    "atower_sum": "measured auto tower ability", "n_min": "experience of least-experienced robot",
    "n_mean": "average experience", "recent_mean": "recent scoring form", "bd_n_min": "breakdown history",
    "rating_sum_": "rating",
}


def label(feat):
    if feat in ("comp_level", "is_playoff"):
        return "playoff match" if feat == "is_playoff" else "match level"
    if feat.startswith("d_"):
        return "Red minus Blue: " + NAMES.get(feat[2:], feat[2:])
    side = "Red" if feat.startswith("a_") else "Blue"
    return f"{side}: " + NAMES.get(feat[2:], feat[2:])


def predict(red, blue, playoff, pool):
    red_k, blue_k = [f"frc{t}" for t in red], [f"frc{t}" for t in blue]
    row = _row(red_k, blue_k, pool, ctx)
    row["comp_level"] = 3 if playoff else 0
    row["is_playoff"] = int(playoff)
    X = pd.DataFrame([row])[meta["feature_cols"]]
    dm = xgb.DMatrix(X)
    wr = (0, meta["win_best_iteration"] + 1)      # same trees the trained model used (early stopping)
    dr = (0, meta["diff_best_iteration"] + 1)
    p_red = float(win_model.predict(dm, iteration_range=wr)[0])
    gap = float(diff_model.predict(dm, iteration_range=dr)[0])
    contrib = win_model.predict(dm, pred_contribs=True, iteration_range=wr)[0][:-1]   # last = bias term
    drivers = sorted(zip(meta["feature_cols"], contrib), key=lambda x: -abs(x[1]))[:5]
    return p_red, gap, drivers


st.title("🤖 FRC 2026 (REBUILT) Match Predictor")
st.caption(f"Machine-learning demo trained on {meta['official_matches']:,} official 2026 matches "
           f"(data through {meta['data_through']}). Not affiliated with FIRST or The Blue Alliance.")

tab_pred, tab_teams, tab_about = st.tabs(["Predict a match", "Team ratings", "How it works & limits"])

with tab_pred:
    st.write("Enter six team numbers. The model only uses what was known about each team *before* the match.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔴 Red alliance")
        red = [st.number_input(f"Red {i + 1}", 1, 99999, v, key=f"r{i}") for i, v in enumerate((254, 1678, 971))]
    with c2:
        st.subheader("🔵 Blue alliance")
        blue = [st.number_input(f"Blue {i + 1}", 1, 99999, v, key=f"b{i}") for i, v in enumerate((118, 148, 111))]
    playoff = st.checkbox("Playoff match")

    if st.button("Predict", type="primary"):
        all6 = [int(x) for x in red + blue]
        if len(set(all6)) != 6:
            st.error("Each of the six teams must be different.")
        else:
            unknown = [t for t in all6 if f"frc{t}" not in states]
            local = states
            if unknown:
                st.warning(f"No official 2026 data for {unknown}. They're treated as average, "
                           f"so the prediction is less reliable.")
                local = dict(states)
                for t in unknown:
                    local[f"frc{t}"] = TeamState()
            p_red, gap, drivers = predict([int(x) for x in red], [int(x) for x in blue], playoff, local)
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 Red win chance", f"{p_red:.0%}")
            m2.metric("🔵 Blue win chance", f"{1 - p_red:.0%}")
            sd = meta["score_diff_resid_std"]
            m3.metric("Predicted score gap (Red − Blue)", f"{gap:+.0f}",
                      help=f"Rough range {gap - 1.28 * sd:+.0f} to {gap + 1.28 * sd:+.0f}: about 80% of past errors "
                           f"on validation matches were this small. Not a guarantee.")
            st.progress(min(max(p_red, 0.0), 1.0))

            st.subheader("Team estimates")
            rows = []
            for side, teams_ in (("Red", red), ("Blue", blue)):
                for t in teams_:
                    s = local.get(f"frc{int(t)}", TeamState())
                    rows.append({"Alliance": side, "Team": int(t), "Matches": s.n,
                                 "Est. points vs average robot": round(s.rating),
                                 "Est. hub auto": round(s.comp["auto"]), "Est. hub teleop": round(s.comp["tele"]),
                                 "Measured avg endgame tower pts": round(s.end_sum / max(s.n_bd, 1), 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("These are *estimates inferred from alliance scores*, not measurements, "
                       "except the endgame tower column, which comes from per-robot tower levels.")

            st.subheader("What drove this prediction")
            for f, v in drivers:
                st.write(f"- **{label(f)}** pushed toward {'🔴 Red' if v > 0 else '🔵 Blue'} "
                         f"({v:+.2f} on the log-odds scale)")
            st.caption("This shows what the model leaned on. It is not proof of what causes wins.")

with tab_teams:
    st.write("Estimated scoring contribution per robot, from every official 2026 match.")
    minm = st.slider("Minimum matches played", 1, 60, 15)
    df = pd.DataFrame([{"Team": int(k[3:]), "Matches": s.n, "Est. points vs average robot": round(s.rating),
                        "Est. hub auto": round(s.comp["auto"]), "Est. hub teleop": round(s.comp["tele"]),
                        "Win rate": round(s.wins / max(s.n, 1), 2)} for k, s in states.items() if s.n >= minm])
    df = df.sort_values("Est. points vs average robot", ascending=False).reset_index(drop=True)
    df.index += 1
    q = st.text_input("Find a team number")
    if q.strip().isdigit():
        df = df[df.Team == int(q)]
    st.dataframe(df, width="stretch", height=500)

with tab_about:
    m = meta["metrics"]
    st.subheader("How well does it work?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Winner accuracy (held-out matches)", f"{m['xgboost_test']['accuracy']:.1%}")
    c2.metric("Accuracy on unseen events", f"{m['event_holdout_accuracy']:.1%}")
    c3.metric("Score-gap typical error", f"{m['score_diff_MAE']:.0f} pts",
              help=f"Guessing 'no gap' every time is off by {m['score_diff_baseline_MAE']:.0f} pts.")
    st.write(
        f"- Tested on the **latest 20% of official matches**, which the model never saw while training or choosing settings.\n"
        f"- A simple rating-gap baseline already gets {m['baseline_test']['accuracy']:.1%}. "
        f"The ML model's main advantage is better-calibrated probabilities "
        f"(log loss {m['xgboost_test']['log_loss']:.2f} vs {m['baseline_test']['log_loss']:.2f}), "
        f"meaning '70%' really means about 70%."
    )
    st.subheader("How it's built")
    st.write(
        "1. Match data from The Blue Alliance for official 2026 events; bad or inconsistent records are logged, never silently dropped.\n"
        "2. Matches are processed **in time order**. Each match's features come only from team history *before* it. "
        "Automated tests scramble the future and confirm the features don't change.\n"
        "3. Per-robot ratings are estimated from alliance scores; tower levels are per-robot measurements.\n"
        "4. XGBoost, LightGBM and logistic regression are compared on a chronological split; XGBoost is used here."
    )
    st.subheader("Limits (please read)")
    st.write(
        "- Only alliance-level scores exist, so per-robot contributions are **estimates**.\n"
        "- Ratings are end-of-data snapshots, so this is for *what-if* matchups, not re-predicting past matches.\n"
        "- Early in a team's season the model knows little about it.\n"
        "- Offseason and preseason events are excluded.\n"
        "- Score-gap numbers have roughly ±75 points of error and are least reliable for elite alliances; "
        "the win probability is the better-validated output.\n"
        "- No model can see robot breakdowns, strategy, or drive-team changes."
    )
    st.caption("Data: The Blue Alliance (thebluealliance.com). Model: XGBoost.")
