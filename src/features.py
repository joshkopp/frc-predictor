"""Leakage-safe feature engineering.

Matches are processed strictly in time order. For each match we:
  1. build features from team state that existed BEFORE the match
  2. only then update team state with that match's result

Because the only source of features is the state dict, and the state is updated
after the row is written, a match's own score/breakdown (or any later match) can
never reach its own features. tests + src/checks.py verify this by perturbation.

HONESTY NOTES
- Alliance-level numbers (total, hub auto, hub teleop, tower) are alliance totals. Per-robot
  "ratings" for them are ESTIMATES inferred from alliance totals, not measurements.
- Per-robot tower levels (auto/endgame) ARE per-robot measurements, but they are matched to
  teams by position (robot1..3 = team_keys order). That mapping is TBA's convention; the data
  itself can't prove it.
"""
from collections import deque
import math
import numpy as np
import pandas as pd
from . import config

LEVEL_CODE = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
COMPS = ("auto", "tele", "tower")   # hub auto points, hub teleop points, tower points


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class TeamState:
    __slots__ = ("n", "rating", "wins", "recent", "n_bd", "comp", "end_sum", "atower_sum")

    def __init__(self):
        self.n = 0
        self.rating = 0.0
        self.wins = 0
        self.recent = deque(maxlen=config.RECENT_N)
        self.n_bd = 0                          # matches with usable breakdown
        self.comp = {c: 0.0 for c in COMPS}    # estimated per-robot component contribution
        self.end_sum = 0.0                     # measured endgame tower points of THIS robot
        self.atower_sum = 0.0                  # measured auto tower points of THIS robot


def _side_features(teams, states, ctx, prefix):
    mu = ctx["mu"]
    ss = [states[t] for t in teams]
    ratings = [s.rating for s in ss]
    ns = [s.n for s in ss]
    k = config.RATING_PRIOR_MATCHES
    recent = [(sum(s.recent) + k * mu) / (len(s.recent) + k) for s in ss]
    winr = [(s.wins + 1) / (s.n + 2) for s in ss]
    f = {
        f"{prefix}_rating_sum": sum(ratings),
        f"{prefix}_rating_max": max(ratings),
        f"{prefix}_rating_min": min(ratings),
        f"{prefix}_n_min": min(ns),
        f"{prefix}_n_mean": float(np.mean(ns)),
        f"{prefix}_recent_mean": float(np.mean(recent)),
        f"{prefix}_winrate": float(np.mean(winr)),
    }
    if ctx["use_bd"]:
        for c in COMPS:
            f[f"{prefix}_{c}_sum"] = sum(s.comp[c] for s in ss)
        # measured per-robot tower ability, shrunk toward the league average for low-sample teams
        f[f"{prefix}_end_sum"] = sum((s.end_sum + k * ctx["mu_end"]) / (s.n_bd + k) for s in ss)
        f[f"{prefix}_atower_sum"] = sum((s.atower_sum + k * ctx["mu_atower"]) / (s.n_bd + k) for s in ss)
        f[f"{prefix}_bd_n_min"] = min(s.n_bd for s in ss)
    return f


def _row(a_teams, b_teams, states, ctx):
    f = {}
    f.update(_side_features(a_teams, states, ctx, "a"))
    f.update(_side_features(b_teams, states, ctx, "b"))
    for name in ("rating_sum", "rating_max", "rating_min", "recent_mean", "winrate"):
        f["d_" + name.replace("_mean", "")] = f[f"a_{name}"] - f[f"b_{name}"]
    if ctx["use_bd"]:
        for name in [f"{c}_sum" for c in COMPS] + ["end_sum", "atower_sum"]:
            f[f"d_{name}"] = f[f"a_{name}"] - f[f"b_{name}"]
    return f


def build_features(matches: pd.DataFrame, return_state=False):
    """matches must be sorted chronologically. Returns one row per match per orientation."""
    use_bd = "red_hub_auto" in matches.columns
    states = {}
    total, count = 0.0, 0
    ctot = {c: 0.0 for c in COMPS}; ccount = 0          # alliance-level component means
    end_tot = atower_tot = 0.0; rcount = 0              # per-robot tower means
    rows = []
    lr_floor, k = config.RATING_LR, config.RATING_PRIOR_MATCHES

    for m in matches.itertuples(index=False):
        red, blue = list(m.red_teams), list(m.blue_teams)
        for t in red + blue:
            if t not in states:
                states[t] = TeamState()
        ctx = {"mu": total / count if count else 0.0, "use_bd": use_bd,
               "mu_end": end_tot / rcount if rcount else 0.0,
               "mu_atower": atower_tot / rcount if rcount else 0.0}
        for c in COMPS:
            ctx["mu_" + c] = ctot[c] / ccount if ccount else 0.0

        # ---- features from PRE-match state (both orientations) ----
        base = {"match_key": m.match_key, "event_key": m.event_key, "time": m.time,
                "comp_level": LEVEL_CODE[m.comp_level], "is_playoff": int(m.comp_level != "qm")}
        for orient, a, b, a_sc, b_sc in (("orig", red, blue, m.red_score, m.blue_score),
                                         ("mirror", blue, red, m.blue_score, m.red_score)):
            r = dict(base)
            r.update(_row(a, b, states, ctx))
            r["orient"] = orient
            r["y_win"] = int(a_sc > b_sc)          # LABELS: use result, never as feature
            r["a_score"], r["b_score"] = a_sc, b_sc
            r["score_diff"] = a_sc - b_sc
            rows.append(r)

        # ---- only now update state with this match's result ----
        mu = ctx["mu"]
        for side, teams, own, opp in (("red", red, m.red_score, m.blue_score),
                                      ("blue", blue, m.blue_score, m.red_score)):
            pred = mu + sum(states[t].rating for t in teams)
            err = own - pred
            for t in teams:
                s = states[t]
                lr = max(lr_floor, 1.0 / (s.n + k))
                s.rating += lr * err / 3.0
                s.n += 1
                s.wins += int(own > opp)
                s.recent.append(own)
        total += m.red_score + m.blue_score
        count += 2

        if use_bd and _ok(m.red_hub_auto) and _ok(m.blue_hub_auto):
            for side, teams in (("red", red), ("blue", blue)):
                vals = {"auto": getattr(m, f"{side}_hub_auto"), "tele": getattr(m, f"{side}_hub_teleop"),
                        "tower": getattr(m, f"{side}_tower")}
                end_pts, auto_pts = getattr(m, f"{side}_end_pts"), getattr(m, f"{side}_auto_pts")
                for c in COMPS:
                    pred = ctx["mu_" + c] + sum(states[t].comp[c] for t in teams)
                    err = vals[c] - pred
                    for t in teams:
                        s = states[t]
                        s.comp[c] += max(lr_floor, 1.0 / (s.n_bd + k)) * err / 3.0
                    ctot[c] += vals[c]
                ccount += 1
                for i, t in enumerate(teams):
                    s = states[t]
                    s.n_bd += 1
                    s.end_sum += end_pts[i]
                    s.atower_sum += auto_pts[i]
                    end_tot += end_pts[i]; atower_tot += auto_pts[i]; rcount += 1

    out = pd.DataFrame(rows)
    if return_state:
        ctx = {"mu": total / count if count else 0.0, "use_bd": use_bd,
               "mu_end": end_tot / rcount if rcount else 0.0,
               "mu_atower": atower_tot / rcount if rcount else 0.0}
        for c in COMPS:
            ctx["mu_" + c] = ctot[c] / ccount if ccount else 0.0
        return out, {"states": states, "ctx": ctx}
    return out


def feature_columns(df):
    drop = {"match_key", "event_key", "time", "orient", "y_win", "a_score", "b_score", "score_diff"}
    return [c for c in df.columns if c not in drop]


def run():
    matches = pd.read_pickle(config.VALIDATED / "matches_valid.pkl")
    if config.OFFICIAL_ONLY and "official" in matches.columns:
        matches = matches[matches.official].reset_index(drop=True)
        print(f"Official events only: {len(matches)} matches (offseason/preseason excluded from model).")
    assert matches.time.is_monotonic_increasing, "matches must be time-sorted"
    feats, state = build_features(matches, return_state=True)
    feats.to_pickle(config.PROCESSED / "features.pkl")
    import joblib
    joblib.dump(state, config.PROCESSED / "team_state.joblib")
    print(f"Built {len(feats)} feature rows ({len(feats)//2} matches x 2 orientations), "
          f"{len(feature_columns(feats))} features.")
    return feats
