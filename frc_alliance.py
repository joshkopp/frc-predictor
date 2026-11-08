"""
FRC Alliance Recommender - core logic.

Pulls an event from The Blue Alliance, computes per-team stats (including
component OPRs from score breakdowns, so it works for any season's game),
and ranks every team as a potential alliance partner for a chosen captain.

Scores (all 0-100, relative to the best candidate at the event):
  - points_score : projected alliance score (captain + candidate + typical 3rd robot)
  - synergy      : how much the candidate covers the captain's weak scoring areas
  - pick_score   : weighted blend of the two (weight set by the user)
"""

import numpy as np
import pandas as pd
import requests

TBA = "https://www.thebluealliance.com/api/v3"

# Score-breakdown "...Points" fields that aren't real scoring categories
_SKIP_WORDS = ("total", "foul", "adjust", "penalty", "rp", "bonus")
# Phase totals that usually overlap with finer-grained fields
_PHASE_TOTALS = ("autoPoints", "teleopPoints")


# ---------------------------------------------------------------- TBA fetch

def tba_get(path, api_key):
    r = requests.get(f"{TBA}{path}", headers={"X-TBA-Auth-Key": api_key}, timeout=20)
    r.raise_for_status()
    return r.json()


def load_event(event_key, api_key):
    return {
        "teams": tba_get(f"/event/{event_key}/teams/simple", api_key) or [],
        "matches": tba_get(f"/event/{event_key}/matches", api_key) or [],
        "oprs": tba_get(f"/event/{event_key}/oprs", api_key) or {},
        "rankings": tba_get(f"/event/{event_key}/rankings", api_key) or {},
    }


def list_events(year, api_key):
    events = tba_get(f"/events/{year}/simple", api_key) or []
    return sorted(events, key=lambda e: (e.get("start_date") or "", e.get("name") or ""))


# ---------------------------------------------------------------- parsing

def alliance_rows(matches, include_playoffs=False):
    """One row per alliance per played match."""
    rows = []
    for m in matches:
        if not include_playoffs and m.get("comp_level") != "qm":
            continue
        al = m.get("alliances") or {}
        if (al.get("red") or {}).get("score", -1) < 0:  # not played yet
            continue
        sb = m.get("score_breakdown") or {}
        for color, opp in (("red", "blue"), ("blue", "red")):
            a = al[color]
            comps = {
                k: float(v)
                for k, v in (sb.get(color) or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            rows.append({
                "match": m["key"],
                "order": (m.get("actual_time") or m.get("time") or 0, m.get("match_number", 0)),
                "teams": a["team_keys"],
                "score": float(a["score"]),
                "opp_score": float(al[opp]["score"]),
                "won": m.get("winning_alliance") == color,
                "tied": m.get("winning_alliance") in ("", None),
                "comps": comps,
            })
    rows.sort(key=lambda r: r["order"])
    return rows


def point_fields(rows):
    """All scoring-category fields found in the breakdowns (in a stable order)."""
    seen = {}
    for r in rows:
        for k in r["comps"]:
            words = {w.lower() for w in _camel_words(k)}
            if k.endswith("Points") and not words & set(_SKIP_WORDS):
                seen[k] = True
    return list(seen)


def default_synergy_fields(fields):
    """Prefer finer-grained categories; drop phase totals if enough others exist."""
    finer = [f for f in fields if f not in _PHASE_TOTALS]
    return finer if len(finer) >= 2 else fields


# ---------------------------------------------------------------- stats

def component_oprs(rows, team_keys, fields):
    """Least-squares OPR for the total score and every breakdown field."""
    idx = {t: i for i, t in enumerate(team_keys)}
    A = np.zeros((len(rows), len(team_keys)))
    for r, row in enumerate(rows):
        for t in row["teams"]:
            if t in idx:
                A[r, idx[t]] = 1.0
    out = {}
    targets = {"_score": [row["score"] for row in rows],
               "_conceded": [row["opp_score"] for row in rows]}
    for f in fields:
        targets[f] = [row["comps"].get(f, 0.0) for row in rows]
    for name, y in targets.items():
        x, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
        out[name] = x
    return pd.DataFrame(out, index=team_keys)


def team_match_stats(rows, team_keys, recent_n=4):
    per = {t: [] for t in team_keys}
    for row in rows:
        for t in row["teams"]:
            if t in per:
                per[t].append(row)
    recs = {}
    for t, ms in per.items():
        scores = np.array([m["score"] for m in ms], float)
        recs[t] = {
            "played": len(ms),
            "avg_score": scores.mean() if len(ms) else np.nan,
            "std_score": scores.std() if len(ms) > 1 else np.nan,
            "recent_avg": scores[-recent_n:].mean() if len(ms) else np.nan,
            "win_rate": np.mean([m["won"] for m in ms]) if ms else np.nan,
        }
    return pd.DataFrame.from_dict(recs, orient="index")


def build_team_table(event, include_playoffs=False):
    """Everything we know about each team at the event, one row per team."""
    teams = {t["key"]: t for t in event["teams"]}
    rows = alliance_rows(event["matches"], include_playoffs)
    # include teams that only show up in match data
    for r in rows:
        for t in r["teams"]:
            teams.setdefault(t, {"key": t, "team_number": int(t[3:]), "nickname": ""})
    keys = sorted(teams, key=lambda k: teams[k].get("team_number", 0))
    fields = point_fields(rows)

    df = pd.DataFrame(index=keys)
    df["team"] = [teams[k].get("team_number") for k in keys]
    df["name"] = [teams[k].get("nickname") or "" for k in keys]

    if rows:
        comp = component_oprs(rows, keys, fields)
        df = df.join(team_match_stats(rows, keys))
    else:
        comp = pd.DataFrame(0.0, index=keys, columns=["_score", "_conceded", *fields])
        df = df.join(pd.DataFrame({"played": 0, "avg_score": np.nan, "std_score": np.nan,
                                   "recent_avg": np.nan, "win_rate": np.nan}, index=keys))

    # OPR/DPR/CCWM: prefer TBA's official numbers, fall back to our own
    tba = event.get("oprs") or {}
    df["opr"] = [(tba.get("oprs") or {}).get(k, comp.at[k, "_score"]) for k in keys]
    df["dpr"] = [(tba.get("dprs") or {}).get(k, comp.at[k, "_conceded"]) for k in keys]
    df["ccwm"] = [(tba.get("ccwms") or {}).get(k, df.at[k, "opr"] - df.at[k, "dpr"]) for k in keys]

    # Rankings
    rank, record = {}, {}
    for r in (event.get("rankings") or {}).get("rankings") or []:
        rank[r["team_key"]] = r.get("rank")
        rec = r.get("record") or {}
        record[r["team_key"]] = f'{rec.get("wins", 0)}-{rec.get("losses", 0)}-{rec.get("ties", 0)}'
    df["rank"] = [rank.get(k) for k in keys]
    df["record"] = [record.get(k, "") for k in keys]

    for f in fields:
        df[f] = comp[f]
    return df, fields


# ---------------------------------------------------------------- ranking

def recommend(df, captain, synergy_fields, picked=(), points_weight=0.6):
    """
    Rank every available team as a partner for `captain`.
    `picked` = teams already on an alliance (excluded from the pool).
    """
    pool = df.drop(index=[captain, *[p for p in picked if p in df.index and p != captain]])
    if pool.empty:
        return pool

    # --- points score: captain + candidate + typical 3rd robot from what's left
    filler = pool["opr"].median()
    proj = df.at[captain, "opr"] + pool["opr"] + filler

    # --- synergy: does the candidate score where the captain doesn't?
    if synergy_fields:
        sub = df[synergy_fields].clip(lower=0)
        norm = (sub / sub.max().replace(0, np.nan)).fillna(0)      # 0-1 per category
        weight = sub.mean() / sub.mean().sum() if sub.mean().sum() > 0 else sub.mean() * 0
        gap = 1 - norm.loc[captain]                                  # captain's weakness
        contrib = norm.loc[pool.index] * gap * weight                # per category
        raw = contrib.sum(axis=1)
        best_fit = contrib.idxmax(axis=1).where(raw > 0, "")
    else:
        raw = pd.Series(0.0, index=pool.index)
        best_fit = pd.Series("", index=pool.index)

    def to100(s):
        lo, hi = s.min(), s.max()
        return pd.Series(100.0, index=s.index) if hi - lo < 1e-9 else 100 * (s - lo) / (hi - lo)

    out = pool.copy()
    out["proj_alliance_pts"] = proj
    out["points_score"] = to100(proj)
    out["synergy"] = to100(raw) if raw.max() > 0 else 0.0
    out["best_fit"] = best_fit.map(pretty_field)
    out["pick_score"] = points_weight * out["points_score"] + (1 - points_weight) * out["synergy"]
    out = out.sort_values("pick_score", ascending=False)
    out.insert(0, "pick", range(1, len(out) + 1))
    return out


def _camel_words(s):
    """'endGameTowerPoints' -> ['end', 'Game', 'Tower', 'Points']"""
    words, cur = [], ""
    for ch in s:
        if ch.isupper() and cur and not cur[-1].isupper():
            words.append(cur)
            cur = ch
        else:
            cur += ch
    words.append(cur)
    return [w for w in words if w]


def pretty_field(f):
    """'autoFuelPoints' -> 'Auto Fuel'"""
    if not f:
        return ""
    f = f[:-6] if f.endswith("Points") else f
    return " ".join(w[:1].upper() + w[1:] for w in _camel_words(f))
