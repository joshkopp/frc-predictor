"""FRC 2026 (REBUILT) match predictor: a leakage-safe ML demo, themed to match joshkoppanalil.com.

Run locally:  streamlit run streamlit_app.py
"""
import json
import pathlib
from collections import deque

import pandas as pd
import streamlit as st
import xgboost as xgb

import ui
from src.features import _row, TeamState

ROOT = pathlib.Path(__file__).parent
D = ROOT / "deploy"

st.set_page_config(page_title="FRC 2026 Match Predictor · Josh Koppanalil", page_icon="🤖", layout="wide")
# CSS goes through st.markdown: st.html silently drops a stylesheet this large.
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load():
    meta = json.loads((D / "app_meta.json").read_text())
    stats = json.loads((D / "stats.json").read_text())
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
    return meta, stats, states, win, diff


meta, stats, states, win_model, diff_model = load()
ctx = meta["ctx"]
M = stats["models"]


def predict(red, blue, playoff, pool):
    row = _row([f"frc{t}" for t in red], [f"frc{t}" for t in blue], pool, ctx)
    row["comp_level"] = 3 if playoff else 0
    row["is_playoff"] = int(playoff)
    X = pd.DataFrame([row])[meta["feature_cols"]]
    dm = xgb.DMatrix(X)
    wr, dr = (0, meta["win_best_iteration"] + 1), (0, meta["diff_best_iteration"] + 1)   # trees chosen by early stopping
    p_red = float(win_model.predict(dm, iteration_range=wr)[0])
    gap = float(diff_model.predict(dm, iteration_range=dr)[0])
    contrib = win_model.predict(dm, pred_contribs=True, iteration_range=wr)[0][:-1]
    drivers = sorted(zip(meta["feature_cols"], contrib), key=lambda x: -abs(x[1]))[:5]
    return p_red, gap, drivers


# ===================================================================== top of page
st.html(ui.nav(meta["data_through"]))
st.html(ui.hero(stats, meta))
st.html(ui.kpis(stats, meta))
st.html(ui.marquee())

# ===================================================================== predictor
st.html(ui.section("Try it", "Predict a match",
                   "Enter six team numbers. The model only uses what was known about each team <b>before</b> the match."))
c1, c2 = st.columns(2)
with c1:
    st.html(ui.alliance_head("Red"))
    red = [st.number_input(f"Red {i + 1}", 1, 99999, v, key=f"r{i}") for i, v in enumerate((254, 1678, 971))]
with c2:
    st.html(ui.alliance_head("Blue"))
    blue = [st.number_input(f"Blue {i + 1}", 1, 99999, v, key=f"b{i}") for i, v in enumerate((118, 148, 111))]
playoff = st.checkbox("Playoff match")

if st.button("Predict the match", type="primary"):
    all6 = [int(x) for x in red + blue]
    if len(set(all6)) != 6:
        st.session_state["res"] = {"error": "Each of the six teams must be different."}
    else:
        unknown = [t for t in all6 if f"frc{t}" not in states]
        pool = dict(states)
        for t in unknown:
            pool[f"frc{t}"] = TeamState()
        p_red, gap, drivers = predict([int(x) for x in red], [int(x) for x in blue], playoff, pool)
        rows = []
        for side, ts in (("Red", red), ("Blue", blue)):
            for t in ts:
                s = pool[f"frc{int(t)}"]
                rows.append({"side": side, "team": int(t), "n": s.n, "rating": round(s.rating), "auto": round(s.comp["auto"]),
                             "tele": round(s.comp["tele"]), "tower": s.end_sum / max(s.n_bd, 1)})
        top = max(1, max(r["rating"] for r in rows))
        for r in rows:
            r["bar"] = max(0, r["rating"]) / top * 100
        conf = max(p_red, 1 - p_red)
        note = ""
        for b in stats["confidence"]:
            if b["lo"] <= conf < b["hi"] + 1e-9:
                note = (f'In testing, when the model was <b>{b["lo"] * 100:.0f}-{b["hi"] * 100:.0f}% sure</b>, it picked the right winner '
                        f'<b>{b["acc"] * 100:.1f}%</b> of the time ({b["n"]:,} matches).')
        st.session_state["res"] = {"p": p_red, "gap": gap, "rows": rows, "drivers": drivers, "note": note, "unknown": unknown}

res = st.session_state.get("res")
if res:
    if "error" in res:
        st.error(res["error"])
    else:
        st.html(ui.result(res["p"], res["gap"], meta["score_diff_resid_std"], res["rows"], res["drivers"], res["note"], res["unknown"]))

# ===================================================================== proof: how good
t = M["xgboost"]["test"]
st.html(ui.section("The scoreboard", "How good is it, really?",
                   f"Every number here comes from the <b>latest {stats['n_test_matches']:,} official matches</b>, which the model never saw while "
                   "learning or while choosing its settings. Accuracy is how often it picks the winner; log loss and Brier score judge the quality of the probabilities (lower is better)."))
order = [("Coin flip (reference)", None), ("Rating-gap baseline", "baseline_rating_logit"), ("Logistic regression", "logistic_all_features"),
         ("XGBoost (used in this app)", "xgboost"), ("LightGBM", "lightgbm")]
rows, vals = [], []
for name, key in order:
    if key is None:
        v = (0.5, stats["coin_flip_logloss"], 0.25, 0.5)
    else:
        x = M[key]["test"]
        v = (x["accuracy"], x["log_loss"], x["brier"], x["roc_auc"])
    vals.append(v)
    rows.append([name, ui.pct(v[0]), f"{v[1]:.3f}", f"{v[2]:.3f}", f"{v[3]:.3f}"])
real = list(range(1, len(vals)))
best = {(max(real, key=lambda i: vals[i][0]), 1), (min(real, key=lambda i: vals[i][1]), 2),
        (min(real, key=lambda i: vals[i][2]), 3), (max(real, key=lambda i: vals[i][3]), 4)}
tbl = ui.table(["Model", "Accuracy", "Log loss", "Brier", "ROC-AUC"], [[c for c in r] for r in rows], used=3, best=best)
tbl = tbl + (
    '<div class="jk"><p class="jk-muted" style="font-size:13px;margin-top:12px">Accuracy differences under about 1 point are within noise at this test size. '
    'The tuned XGBoost model\'s clear advantage is probability quality (log loss), meaning its "70%" really means about 70%.</p></div>')
tr_, va_ = M["xgboost"]["train"], M["xgboost"]["val"]
over = ui.hbars([{"label": "Train", "v": tr_["accuracy"], "text": ui.pct(tr_["accuracy"]), "dim": True},
                 {"label": "Validation", "v": va_["accuracy"], "text": ui.pct(va_["accuracy"]), "dim": True},
                 {"label": "Test (final exam)", "v": t["accuracy"], "text": ui.pct(t["accuracy"]), "hot": True}])
sm = stats["score_model"]
score = ui.hbars([{"label": "Guess 'no gap'", "v": sm["baseline_MAE_predict_zero"], "text": f"{sm['baseline_MAE_predict_zero']:.0f} pts", "dim": True},
                  {"label": "This model", "v": sm["test_MAE"], "text": f"{sm['test_MAE']:.0f} pts", "hot": True}], scale=sm["baseline_MAE_predict_zero"])
st.html(ui.grid("2", ui.card(tbl, "Model comparison", "Four models were compared on the same chronological split."),
                ui.card(over + '<div class="jk"><p class="jk-muted" style="font-size:13px;margin:12px 0 22px">Train and test accuracy are close, so it is not just memorizing.</p></div>'
                        + '<div class="jk"><div class="jk-eyebrow" style="margin:6px 0 4px">Score-gap error (lower is better)</div></div>' + score
                        + f'<div class="jk"><p class="jk-muted" style="font-size:13px;margin-top:10px">R² = {sm["test_R2"]:.2f}. Typical miss on the score gap is about {sm["test_MAE"]:.0f} points.</p></div>',
                        "Overfitting check", "A model that aces practice but flops on new matches is useless.")))

# ===================================================================== calibration + confidence
st.html(ui.section("Trust", "Can you trust the percentages?",
                   "A good forecaster is right about 70% of the time when it says 70%. The line below should hug the dashed diagonal."))
conf_items = [{"label": f"{b['lo'] * 100:.0f}-{b['hi'] * 100:.0f}% sure", "v": b["acc"], "text": ui.pct(b["acc"]), "hot": b["lo"] >= 0.9} for b in stats["confidence"]]
conf_html = ui.hbars(conf_items) + '<div class="jk"><div class="jk-stat" style="margin-top:14px"><span>Matches in each band</span><b>' + \
    " · ".join(f"{b['n']:,}" for b in stats["confidence"]) + '</b></div></div>'
st.html(ui.grid("2", ui.card(ui.calib_svg(stats["calibration"]), "Calibration",
                             "Each dot is 10% of test matches, grouped by predicted win probability. Closer to the diagonal is better."),
                ui.card(conf_html, "Accuracy by confidence", "How often it was right, depending on how sure it was. More confident should mean more accurate.")))

# ===================================================================== strengths / weaknesses
st.html(ui.section("Detail", "Where it's strong and where it struggles",
                   "Honest breakdowns on held-out matches. Small groups are noisy, so check the match counts."))
exp = ui.hbars([{"label": f"{e['label']} prior matches", "v": e["acc"], "text": ui.pct(e["acc"], 0)} for e in stats["by_experience"]])
exp += '<div class="jk"><p class="jk-muted" style="font-size:13px;margin-top:10px">Matches in each group: ' + " · ".join(f"{e['n']:,}" for e in stats["by_experience"]) + '</p></div>'
lvl = ui.hbars([{"label": e["label"], "v": e["acc"], "text": ui.pct(e["acc"], 0)} for e in stats["by_level"]])
lvl += '<div class="jk"><p class="jk-muted" style="font-size:13px;margin-top:10px">Matches: ' + " · ".join(f"{e['n']:,}" for e in stats["by_level"]) + '</p></div>'
st.html(ui.grid("2", ui.card(exp, "By team experience", "Accuracy by how few matches the least-experienced robot had played."),
                ui.card(lvl, "Qualification vs playoff", "Playoffs are tighter: stronger teams face each other.")))
wk = stats["by_week"]
st.html(ui.grid("2", ui.card(ui.vbars([{"label": w["label"], "v": w["acc"]} for w in wk], vmax=1.0, fmt=lambda v: f"{v * 100:.0f}%"),
                             "Accuracy by week", "Validation and test periods only. Weeks with under 30 matches are hidden."),
                ui.card(ui.vbars([{"label": f"Fold {i + 1}", "v": c["accuracy"]} for i, c in enumerate(stats["temporal_cv"])], vmax=1.0,
                                 fmt=lambda v: f"{v * 100:.0f}%")
                        + f'<div class="jk"><p class="jk-muted" style="font-size:13px;margin-top:12px">Each fold trains on earlier matches and tests on the next block. '
                          f'On events held out entirely: <b style="color:#f4f6f4">{stats["event_holdout"]["accuracy"] * 100:.1f}%</b>.</p></div>',
                        "Stability over time", "Expanding-window cross-validation: always train on the past, test on the future.")))

# ===================================================================== drivers
st.html(ui.section("Inside the model", "What does it look at?",
                   "SHAP scores show how much each input moved the model's predictions on average. It shows what the model uses, not what causes wins."))
mx = max(x["value"] for x in stats["shap"])
shap_html = ui.hbars([{"label": ui.label(x["feature"]), "v": x["value"], "text": f'{x["value"]:.2f}', "hot": i == 0} for i, x in enumerate(stats["shap"])], scale=mx)
how = ("<div class='jk'><ul class='jk-list'>"
       "<li><b>Rating gap</b>: each robot gets an estimated scoring contribution, updated after every match (with small-sample shrinkage).</li>"
       "<li><b>Auto, teleop, tower</b>: separate estimates for each scoring phase from the 2026 score breakdown.</li>"
       "<li><b>Recent form</b>: how an alliance's teams scored lately.</li>"
       "<li><b>Strict time order</b>: features for a match use only earlier matches. Tests scramble the future to prove it.</li></ul></div>")
st.html(ui.grid("2", ui.card(shap_html, "Top features", "Average impact on the win prediction (log-odds)."),
                ui.card(how, "How the features are built", "Everything is computed in chronological order.")))

# ===================================================================== data
dq = stats["data_quality"]
st.html(ui.section("The data", "What it was trained on",
                   "Match results from The Blue Alliance. Every record is validated, and nothing is silently dropped."))
funnel = ui.hbars([{"label": "Collected from TBA", "v": dq["collected"], "text": f"{dq['collected']:,}", "dim": True},
                   {"label": "Passed validation", "v": dq["valid"], "text": f"{dq['valid']:,}", "dim": True},
                   {"label": "Official events (model)", "v": dq["official_used"], "text": f"{dq['official_used']:,}", "hot": True}], scale=dq["collected"])
reasons = ui.stat_rows([(k.replace("_", " "), f"{v:,}") for k, v in sorted(dq["reasons"].items(), key=lambda kv: -kv[1])])
facts = ui.stat_rows([("Events with valid matches (all types)", f"{dq['events']:,}"), ("Teams rated", f"{meta['n_teams']:,}"),
                      ("Offseason/preseason excluded", f"{dq['offseason_excluded']:,}"), ("Mean alliance score", f"{stats['mean_score']:.0f}"),
                      ("Median alliance score", f"{stats['median_score']:.0f}"), ("Highest alliance score", f"{stats['max_score']:.0f}")])
st.html(ui.grid("3", ui.card(funnel, "Pipeline", "From raw records to training data."),
                ui.card(f'<div class="jk">{reasons}</div>', "Why records were excluded", "Logged in a file, never deleted silently."),
                ui.card(f'<div class="jk">{facts}</div>', "Dataset facts", "2026 REBUILT.")))
st.html(ui.grid("2", ui.card(ui.vbars([{"label": w["label"], "v": w["n"]} for w in stats["matches_per_week"]], fmt=lambda v: f"{v:,}"),
                             "Matches per week", "Official matches used by the model."),
                ui.card(ui.vbars([{"label": h["label"].split("-")[0], "v": h["n"]} for h in stats["score_hist"]], fmt=lambda v: f"{v:,}"),
                        "Alliance score distribution", "Alliance score in points (bucket start). Heavily skewed: most are low, a few elite alliances score very high.")))

# ===================================================================== leaderboard
st.html(ui.section("Teams", "Team ratings",
                   "Estimated scoring contribution per robot, in points above an average robot, from every official 2026 match. These are estimates inferred from alliance scores."))
board = sorted(((k, s) for k, s in states.items() if s.n >= 15), key=lambda kv: -kv[1].rating)
rank = {k: i + 1 for i, (k, _) in enumerate(board)}
q = st.text_input("Find a team number", key="teamq", placeholder="e.g. 254")
if q.strip().isdigit() and f"frc{int(q)}" in states:
    k = f"frc{int(q)}"
    s = states[k]
    rk = f"#{rank[k]} of {len(board)} teams with 15+ matches" if k in rank else "Not ranked (needs 15+ matches)"
    st.html('<div class="jk"><div class="jk-found jk-glass jk-in"><div class="jk-eyebrow">Team ' + str(int(q)) + '</div>'
            f'<div class="jk-grid4" style="margin-top:12px"><div><div class="jk-kv">{s.rating:+.0f}</div><div class="jk-kl">Est. points vs avg</div></div>'
            f'<div><div class="jk-kv">{s.n}</div><div class="jk-kl">Matches</div></div>'
            f'<div><div class="jk-kv">{s.wins / max(s.n, 1) * 100:.0f}<em>%</em></div><div class="jk-kl">Win rate</div></div>'
            f'<div><div class="jk-kv" style="font-size:22px;padding-top:8px">{rk}</div></div></div></div></div>')
elif q.strip():
    st.html('<div class="jk"><div class="jk-warn">No official 2026 data for that team.</div></div>')
top = board[:25]
mxr = max(s.rating for _, s in top)
lb_rows = [[f'<span class="rk">{i + 1}</span>', k[3:], f'{s.rating:+.0f}<span class="jk-mini"><i style="--w:{max(0, s.rating) / mxr * 100:.0f}%"></i></span>',
            f"{s.n}", f"{s.wins / max(s.n, 1) * 100:.0f}%", f"{s.comp['auto']:+.0f}", f"{s.comp['tele']:+.0f}"] for i, (k, s) in enumerate(top)]
st.html(ui.wide(ui.card(ui.table(["#", "Team", "Est. points vs avg", "Matches", "Win rate", "Hub auto", "Hub teleop"], lb_rows),
                        "Top 25 teams", "Minimum 15 matches played.")))

# ===================================================================== limits
st.html(ui.section("Read this", "How it works and where it falls short"))
works = ("<div class='jk'><ul class='jk-list'>"
         "<li>Official 2026 events only (regionals, districts, championships). Offseason and preseason events are kept out.</li>"
         "<li>Matches are processed in time order, and the data is split by time into train, validation and test.</li>"
         "<li>XGBoost, LightGBM and logistic regression are compared. The winner is picked on validation data only.</li>"
         "<li>Automated tests fail the build if future information leaks into a match's features.</li></ul></div>")
limits = ("<div class='jk'><ul class='jk-list'>"
          "<li>Only alliance-level scores exist, so each robot's contribution is an <b>estimate</b>.</li>"
          "<li>Ratings are end-of-data snapshots, so this is for what-if matchups, not for re-predicting past matches.</li>"
          "<li>Early in a team's season the model knows little about it.</li>"
          "<li>Score-gap numbers carry roughly ±75 points of error and are weakest for elite alliances. Trust the win probability more.</li>"
          "<li>It cannot see breakdowns, strategy, or drive-team changes.</li></ul></div>")
st.html(ui.grid("2", ui.card(works, "How it works"), ui.card(limits, "Limits")))
st.html(ui.footer())
