"""HTML builders for the themed page. Everything returns an HTML string for st.html()."""
import html as _h

NAMES = {
    "rating_sum": "overall rating (sum of 3 robots)", "rating_max": "best robot's rating",
    "rating_min": "weakest robot's rating", "recent": "recent scoring form", "winrate": "win rate",
    "tele_sum": "teleop (hub) scoring", "auto_sum": "autonomous (hub) scoring", "tower_sum": "tower scoring",
    "end_sum": "measured endgame tower ability", "atower_sum": "measured auto tower ability",
    "n_min": "experience of least-experienced robot", "n_mean": "average experience",
    "recent_mean": "recent scoring form", "bd_n_min": "breakdown history",
}


def esc(x):
    return _h.escape(str(x))


def label(feat):
    if feat == "is_playoff":
        return "playoff match"
    if feat == "comp_level":
        return "match level"
    if feat.startswith("d_"):
        return "Red minus Blue: " + NAMES.get(feat[2:], feat[2:])
    side = "Red" if feat.startswith("a_") else "Blue"
    return f"{side}: " + NAMES.get(feat[2:], feat[2:])


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def count(n, suffix=""):
    """Number that counts up from 0 (CSS only; ends on the true value even if unsupported)."""
    return f'<span class="jk-count" style="--to:{int(n)}" data-to="{int(n)}" aria-label="{int(n)}"></span>{suffix}'


# ------------------------------------------------------------------ page chrome
def nav(data_through):
    return (f'<div class="jk"><div class="jk-nav jk-glass jk-in"><a class="jk-brand" href="https://joshkoppanalil.com" target="_blank">JK<span>.</span></a>'
            f'<span class="sp"></span><span class="jk-pill"><i class="jk-dot"></i>2026 data through {esc(data_through)}</span>'
            f'<a class="jk-pill" href="https://joshkoppanalil.com" target="_blank">joshkoppanalil.com ↗</a></div></div>')


def hero(stats, meta):
    m = stats["models"]
    acc = m["xgboost"]["test"]["accuracy"]
    base = m["baseline_rating_logit"]["test"]["accuracy"]
    n_test = stats["n_test_matches"]
    ring = f'<div class="jk-ring" style="--to:{acc * 100:.1f}"></div>'
    whole, dec = f"{acc * 100:.1f}".split(".")
    bars = (hbars([{"label": "Coin flip", "v": 0.5, "text": "50.0%"},
                   {"label": "Rating-gap baseline", "v": base, "text": pct(base)},
                   {"label": "XGBoost (this app)", "v": acc, "text": pct(acc), "hot": True}], scale=1.0))
    return (
        '<div class="jk"><section class="jk-hero"><div>'
        '<div class="jk-eyebrow jk-in" style="--d:.1">Machine learning · FRC 2026 REBUILT</div>'
        '<h1 class="jk-in" style="--d:.2">Who wins the match?<span class="hd"></span><span class="l2">An ML model that calls FRC winners before they play.</span></h1>'
        f'<p class="intro jk-in" style="--d:.4">Trained on {meta["official_matches"]:,} official 2026 matches. It only sees what was known about each team '
        'before the match, and it is tested on matches it never saw while learning. Enter any six teams below.</p>'
        '<div class="jk-chips jk-in" style="--d:.55"><span class="chip jk-chip">XGBoost</span><span class="jk-chip">Time-ordered features</span>'
        '<span class="jk-chip">Leakage-tested</span><span class="jk-chip">Calibrated probabilities</span></div></div>'
        '<div class="jk-ringcard jk-glass jk-in" style="--d:.3"><div class="jk-ringwrap">' + ring +
        f'<div class="jk-ringtxt"><div class="jk-big">{count(int(whole))}<small>.{dec}%</small></div><div class="jk-ringlbl">winner accuracy</div></div></div>'
        f'<p class="jk-ringcap"><b>{round(acc * n_test):,} of {n_test:,}</b> held-out official matches called correctly. A coin flip gets 50%.</p>'
        + bars + '</div></section></div>')


def kpis(stats, meta):
    m = stats["models"]["xgboost"]["test"]
    conf = stats["confidence"][-1]
    cards = [
        ("Unseen-event accuracy", f'{stats["event_holdout"]["accuracy"] * 100:.1f}<em>%</em>', "Events held out entirely from training."),
        (f'Right when {int(conf["lo"] * 100)}%+ sure', f'{conf["acc"] * 100:.1f}<em>%</em>',
         f'On {conf["n"]:,} matches ({conf["share"] * 100:.0f}% of all) where the model was very confident.'),
        ("ROC-AUC", f'{m["roc_auc"]:.3f}', "How well it ranks winners above losers (0.5 = chance)."),
        ("Matches tested", f'{stats["n_test_matches"]:,}', f'Plus {stats["n_train_matches"]:,} to learn from, {meta["n_teams"]:,} teams rated.'),
    ]
    return '<div class="jk"><div class="jk-grid4" style="margin-top:6px">' + "".join(
        f'<div class="jk-kpi jk-glass jk-hover" style="--i:{i}"><div class="jk-kv">{v}</div><div class="jk-kl">{esc(l)}</div><div class="jk-ks">{esc(s)}</div></div>'
        for i, (l, v, s) in enumerate(cards)) + '</div></div>'


def marquee():
    a = ["Win probability", "Time-ordered features", "XGBoost", "Calibrated", "Leakage-tested", "Official events only", "REBUILT 2026", "Held-out testing"]
    b = ["Auto points", "Teleop points", "Tower levels", "Rolling form", "Alliance strength", "Shrinkage for small samples", "SHAP explanations", "Chronological split", "Reproducible"]
    row = lambda items: "".join(f"<span>{esc(x)}</span>" for x in items * 2)
    return ('<div class="jk"><div class="jk-mq" aria-hidden="true"><div class="jk-mqr"><div class="jk-mqt" style="--dur:80s">' + row(a) +
            '</div></div><div class="jk-mqr r"><div class="jk-mqt" style="--dur:90s">' + row(b) + '</div></div></div></div>')


def section(eyebrow, title, text=""):
    p = f"<p>{text}</p>" if text else ""
    return f'<div class="jk jk-sec jk-rv"><div class="jk-sech"><div class="jk-eyebrow">{esc(eyebrow)}</div><h2>{esc(title)}</h2>{p}</div></div>'


def alliance_head(side):
    cls = "red" if side == "Red" else "blue"
    return f'<div class="jk"><div class="jk-eyebrow {cls}">{side} alliance</div></div>'


# ------------------------------------------------------------------ charts
def hbars(items, scale=1.0, dim=False):
    """items: dicts with label, v (0..scale), text, optional hot."""
    out = ['<div class="jk">']
    for i, it in enumerate(items):
        w = max(0.0, min(1.0, it["v"] / scale)) * 100
        cls = ("dim " if (dim or it.get("dim")) else "") + ("hot" if it.get("hot") else "")
        out.append(f'<div class="jk-hb {cls}"><span class="t">{esc(it["label"])}</span>'
                   f'<span class="k"><i style="--w:{w:.1f}%;--i:{i}"></i></span><b>{esc(it["text"])}</b></div>')
    out.append("</div>")
    return "".join(out)


def vbars(items, vmax=None, fmt=lambda v: f"{v}"):
    vmax = vmax or max(it["v"] for it in items) or 1
    cols = "".join(
        f'<div class="jk-vb"><span class="v">{esc(fmt(it["v"]))}</span><div class="col"><i style="--h:{it["v"] / vmax * 100:.1f}%;--i:{i}"></i></div>'
        f'<span class="l">{esc(it["label"])}</span></div>' for i, it in enumerate(items))
    return f'<div class="jk"><div class="jk-vbs">{cols}</div></div>'


def calib_svg(cal):
    """Chart is an SVG embedded as an <img> data URI: Streamlit's st.html strips inline <svg>."""
    import base64
    P, W = 40, 270
    X = lambda v: P + v * W
    Y = lambda v: P + (1 - v) * W
    g = "".join(f'<line class="grid" x1="{X(t)}" y1="{P}" x2="{X(t)}" y2="{P + W}"/><line class="grid" x1="{P}" y1="{Y(t)}" x2="{P + W}" y2="{Y(t)}"/>'
                f'<text x="{X(t)}" y="{P + W + 16}" text-anchor="middle">{int(t * 100)}%</text><text x="{P - 8}" y="{Y(t) + 4}" text-anchor="end">{int(t * 100)}%</text>'
                for t in (0, .25, .5, .75, 1))
    pts = [(X(c["pred"]), Y(c["actual"])) for c in cal]
    path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" style="animation-delay:{i * 0.14 + 0.5:.2f}s"/>' for i, (x, y) in enumerate(pts))
    css = ("text{font:11px DM Sans,Arial,sans-serif;fill:#8b918e}.grid{stroke:rgba(255,255,255,.08);stroke-width:1}"
           ".diag{stroke:rgba(255,255,255,.35);stroke-width:1.5;stroke-dasharray:5 5;fill:none}"
           ".ln{fill:none;stroke:#b8f36b;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;"
           "animation:d 2s cubic-bezier(.3,.7,.3,1) .3s forwards}"
           "circle{fill:#10120b;stroke:#b8f36b;stroke-width:2.5;transform-box:fill-box;transform-origin:center;animation:p .5s cubic-bezier(.3,1.7,.5,1) both}"
           "@keyframes d{to{stroke-dashoffset:0}}@keyframes p{from{transform:scale(0)}}")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {P + W + 14} {P + W + 36}"><style>{css}</style>{g}'
           f'<line class="diag" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}"/><path class="ln" pathLength="1" d="{path}"/>{dots}'
           f'<text x="{P + W / 2}" y="{P + W + 32}" text-anchor="middle">Predicted win probability</text>'
           f'<text transform="translate(12 {P + W / 2}) rotate(-90)" text-anchor="middle">Actual win rate</text></svg>')
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f'<div class="jk"><img class="jk-cal" alt="Calibration chart: predicted vs actual win rate" src="{uri}"></div>'


def table(headers, rows, used=None, best=None):
    """rows: list of lists of cell strings. best: set of (row, col) to highlight."""
    best = best or set()
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{"best" if (r, c) in best else ""}">{cell}</td>' for c, cell in enumerate(row))
        body.append(f'<tr class="{"used" if r == used else ""}">{tds}</tr>')
    return f'<div class="jk"><div class="jk-tw"><table class="jk-tbl"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>'


def card(inner, title="", sub=""):
    t = f"<h3>{esc(title)}</h3>" if title else ""
    s = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<div class="jk-card jk-glass jk-hover">{t}{s}{inner}</div>'


def grid(cols, *cards):
    return f'<div class="jk"><div class="jk-grid{cols} jk-rv">{"".join(cards)}</div></div>'


def wide(inner_card):
    return f'<div class="jk"><div class="jk-rv">{inner_card}</div></div>'


def stat_rows(pairs):
    return "".join(f'<div class="jk-stat"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in pairs)


# ------------------------------------------------------------------ prediction result
def result(p_red, gap, sd, rows, drivers, conf_note, unknown):
    pr = p_red * 100
    warn = (f'<div class="jk-warn">No official 2026 data for team(s) {", ".join(str(u) for u in unknown)}. '
            'They are treated as average, so this prediction is less reliable.</div>') if unknown else ""
    tr = "".join(
        f'<tr><td><span class="jk-tag {"r" if r["side"] == "Red" else "b"}">{r["side"]}</span> {r["team"]}</td><td>{r["n"]}</td>'
        f'<td>{r["rating"]:+d}<span class="jk-mini"><i style="--w:{r["bar"]:.0f}%"></i></span></td><td>{r["auto"]:+d}</td><td>{r["tele"]:+d}</td><td>{r["tower"]:.1f}</td></tr>'
        for r in rows)
    why = "".join(
        f'<div><span class="jk-tag {"r" if v > 0 else "b"}">{"RED" if v > 0 else "BLUE"}</span><span><b>{esc(label(f))}</b></span></div>'
        for f, v in drivers)
    note = f'<div class="jk-note">{conf_note}</div>' if conf_note else ""
    return (
        '<div class="jk"><div class="jk-res jk-glass jk-in">' + warn +
        '<div class="jk-eyebrow">Prediction</div>'
        f'<div class="jk-vs"><i class="r" style="--w:{pr:.1f}%">Red {pr:.0f}%</i><i class="b">Blue {100 - pr:.0f}%</i></div>'
        '<div class="jk-vsl"><span>Red alliance</span><span>Blue alliance</span></div>'
        f'<div class="jk-tiles"><div class="jk-tile"><div class="v">{pr:.0f}%</div><div class="l">Red win chance</div></div>'
        f'<div class="jk-tile"><div class="v">{100 - pr:.0f}%</div><div class="l">Blue win chance</div></div>'
        f'<div class="jk-tile"><div class="v">{gap:+.0f}</div><div class="l">Score gap (Red − Blue)</div>'
        f'<div class="s">Rough range {gap - 1.28 * sd:+.0f} to {gap + 1.28 * sd:+.0f}; about 80% of past errors were this small. Not a guarantee.</div></div></div>'
        + note +
        '<div class="jk-eyebrow" style="margin-bottom:8px">Team estimates</div><div class="jk-tw"><table class="jk-tbl"><thead><tr>'
        '<th>Team</th><th>Matches</th><th>Est. points vs avg robot</th><th>Est. hub auto</th><th>Est. hub teleop</th><th>Measured endgame tower pts</th></tr></thead>'
        f'<tbody>{tr}</tbody></table></div>'
        '<p class="jk-muted" style="font-size:12.5px;margin:10px 0 22px">Estimates are inferred from alliance scores, not measured. Only the endgame tower column comes from per-robot data.</p>'
        '<div class="jk-eyebrow" style="margin-bottom:6px">What drove this prediction</div>'
        f'<div class="jk-why">{why}</div>'
        '<p class="jk-muted" style="font-size:12.5px;margin-top:12px">This shows what the model leaned on. It is not proof of what causes wins.</p>'
        '</div></div>')


def footer():
    return ('<div class="jk"><div class="jk-foot"><span>Built by Josh Koppanalil · <a href="https://joshkoppanalil.com" target="_blank">joshkoppanalil.com</a></span>'
            '<span>Data: The Blue Alliance. Not affiliated with FIRST or TBA.</span></div></div>')
