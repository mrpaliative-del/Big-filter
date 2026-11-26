"""HTML for the scanner: the input form and the results dashboard. All dynamic text is escaped; no external resources."""
from __future__ import annotations

from html import escape as _e

CSS = """
:root{--bg:#0a1020;--card:#111a2e;--card2:#0e1628;--line:#1f2b47;--text:#e8eef9;--mute:#8fa3c2;--teal:#22d3ee;--green:#34d399;--lime:#86efac;
--amber:#fbbf24;--orange:#fb923c;--red:#f87171;--violet:#a78bfa;--r:16px}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:radial-gradient(1200px 600px at 10% -10%,#16264a 0%,transparent 60%),radial-gradient(900px 500px at 100% 0,#0f2f3a 0%,transparent 55%),var(--bg);
color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:14px 12px 48px}
.wrap{max-width:980px;margin:auto}
.brand{display:flex;align-items:center;gap:10px;margin:4px 0 14px}
.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--teal),var(--violet));display:grid;place-items:center;font-weight:800;color:#06101f}
.brand h1{font-size:1.15rem;margin:0;letter-spacing:.02em}.brand p{margin:0;color:var(--mute);font-size:.82rem}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 14px}
.chip{border:1px solid var(--line);background:var(--card2);color:var(--mute);border-radius:999px;padding:4px 10px;font-size:.78rem}
.chip b{color:var(--text)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
.stat{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);border-radius:var(--r);padding:12px 14px}
.stat .n{font-size:1.7rem;font-weight:800;line-height:1.1}.stat .l{color:var(--mute);font-size:.78rem;margin-top:2px}
.n.g{color:var(--green)}.n.t{color:var(--teal)}.n.v{color:var(--violet)}.n.a{color:var(--amber)}
h2{font-size:.82rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin:22px 0 10px}
.card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);border-radius:var(--r);padding:14px;margin-bottom:12px;box-shadow:0 8px 24px rgba(0,0,0,.25)}
.fx-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.time{background:var(--teal);color:#06101f;font-weight:800;border-radius:10px;padding:3px 10px;font-variant-numeric:tabular-nums}
.league{color:var(--mute);font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}
.match{font-size:1.08rem;font-weight:700;width:100%}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}
.tile{border:1px solid var(--line);border-radius:14px;padding:12px;background:#0c1426}
.tile.ok{border-color:rgba(52,211,153,.55);box-shadow:inset 0 0 0 1px rgba(52,211,153,.18)}
.tile.warn{border-color:rgba(251,191,36,.5)}
.tile h3{margin:0 0 8px;font-size:.78rem;letter-spacing:.06em;color:var(--mute);text-transform:uppercase}
.top{display:flex;align-items:center;gap:12px}
.ring{--p:0;--c:var(--mute);width:64px;height:64px;border-radius:50%;flex:none;background:conic-gradient(var(--c) calc(var(--p)*1%),#1b2742 0);display:grid;place-items:center;position:relative}
.ring::after{content:"";position:absolute;inset:7px;border-radius:50%;background:#0c1426}
.ring span{position:relative;z-index:1;font-weight:800;font-size:.95rem}
.badge{display:inline-block;border-radius:999px;padding:3px 10px;font-size:.74rem;font-weight:800;letter-spacing:.03em}
.b-ok{background:rgba(52,211,153,.16);color:var(--green)}.b-warn{background:rgba(251,191,36,.16);color:var(--amber)}
.b-no{background:rgba(143,163,194,.14);color:var(--mute)}.b-hi{background:rgba(167,139,250,.2);color:var(--violet)}
.legs{margin:10px 0 0;padding:0;list-style:none}
.legs li{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:5px 0;border-top:1px solid #16213a;font-size:.9rem}
.price{border-radius:8px;padding:1px 8px;font-weight:700;font-variant-numeric:tabular-nums}.p-ok{background:rgba(52,211,153,.16);color:var(--green)}.p-low{background:rgba(248,113,113,.16);color:var(--red)}
.bar{position:relative;height:9px;border-radius:9px;background:#1b2742;margin:12px 0 6px}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:9px;background:linear-gradient(90deg,var(--teal),var(--violet))}
.bar u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--amber)}
.scale{display:flex;justify-content:space-between;color:var(--mute);font-size:.7rem}
.drv{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.drv span{font-size:.72rem;border-radius:8px;padding:2px 8px;border:1px solid var(--line);color:var(--mute)}
.drv .yes{color:var(--green);border-color:rgba(52,211,153,.5)}.drv .against{color:var(--red);border-color:rgba(248,113,113,.5)}
.mute{color:var(--mute);font-size:.82rem}
details{margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
summary{cursor:pointer;color:var(--teal);font-weight:700;font-size:.88rem;list-style:none}summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";color:var(--mute)}details[open] summary::before{content:"▾ "}
.prof{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:10px}
.prof h4{margin:0 0 6px;font-size:.95rem}.prof h4 em{color:var(--mute);font-style:normal;font-weight:500;font-size:.8rem}
.row{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:5px 0;border-top:1px solid #16213a;font-size:.88rem}
.row small{display:block;color:var(--mute);font-size:.7rem}
.g{border-radius:8px;padding:2px 9px;font-weight:800;font-size:.76rem;white-space:nowrap}
.Excellent,.Lesser{background:rgba(52,211,153,.18);color:var(--green)}.Good{background:rgba(134,239,172,.16);color:var(--lime)}
.Average{background:rgba(251,191,36,.15);color:var(--amber)}.Poor{background:rgba(251,146,60,.16);color:var(--orange)}
.Bad,.Plenty{background:rgba(248,113,113,.17);color:var(--red)}.NoData{background:#1b2742;color:var(--mute)}
.align{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.empty{text-align:center;padding:26px 14px}.empty b{font-size:1.05rem}
.btn{display:inline-block;border:0;border-radius:12px;padding:12px 16px;font-weight:800;font-size:1rem;cursor:pointer;background:linear-gradient(135deg,var(--teal),var(--violet));color:#06101f;width:100%;margin-top:12px}
.btn.alt{background:#16213a;color:var(--text);border:1px solid var(--line)}
a.back{display:block;text-align:center;margin-top:18px;color:var(--teal);text-decoration:none;font-weight:700}
textarea,input,select{width:100%;background:#0c1426;color:var(--text);border:1px solid var(--line);border-radius:12px;padding:11px;font-size:15px;font-family:inherit}
textarea{height:210px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
label{display:block;color:var(--mute);font-size:.76rem;margin:0 0 4px}
.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
.check{display:flex;gap:8px;align-items:center;color:var(--text);font-size:.88rem;margin-top:10px}.check input{width:auto}
.warn-box{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.35);border-radius:14px;padding:10px 12px;margin-bottom:14px;font-size:.84rem;color:#fde68a}
.warn-box li{margin:4px 0}
pre.rep{white-space:pre-wrap;background:#0c1426;border:1px solid var(--line);border-radius:12px;padding:12px;font-size:12.5px;color:#cfe0f7}
"""
LOGO = '<div class="logo">F</div>'


def _page(title: str, body: str, script: str = "") -> str:
    return ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_e(title)}</title><style>{CSS}</style></head><body><div class=\"wrap\">{body}</div>{script}</body></html>")


def _brand(sub: str) -> str:
    return f'<div class="brand">{LOGO}<div><h1>Fixture Scanner</h1><p>{_e(sub)}</p></div></div>'


# ------------------------------------------------------------------------------ form
_FORM_JS = """
<script>
(function () {
  var KEY = 'scanner_params_v11', texts = ['threshold','min_threshold','max_xg','price_min','price_max','price_floor','draw_min','qualify_by','mode','selection_source','show'],
      checks = ['include_unrated','log_picks','use_feed'], form = document.querySelector('form');
  function f(n) { return form.querySelector('[name=' + n + ']'); }
  try {
    var s = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (s) { texts.forEach(function (n) { if (s[n] != null && f(n)) f(n).value = s[n]; }); checks.forEach(function (n) { if (s[n] != null && f(n)) f(n).checked = !!s[n]; }); }
  } catch (e) {}
  form.addEventListener('submit', function () {
    try { var o = {}; texts.forEach(function (n) { if (f(n)) o[n] = f(n).value; }); checks.forEach(function (n) { if (f(n)) o[n] = f(n).checked; });
      localStorage.setItem(KEY, JSON.stringify(o)); } catch (e) {}
  });
})();
</script>
"""


def render_form(threshold: float, min_threshold: float, max_xg: float) -> str:
    body = _brand("Statistics-driven market selection · three scanners") + f"""
<div class="chips"><span class="chip">Scanner 1 <b>1X · Home +1.5 · Under 2.5</b></span><span class="chip">Scanner 2 <b>1X · 12 · Over 1.5</b></span>
<span class="chip">Scanner 3 <b>Draw</b></span></div>
<form method="post" action="/ui-scan" class="card">
<label>Paste the Flashscore page (fixtures, optionally with odds)</label>
<textarea name="raw_text" placeholder="Paste raw Flashscore mobile copy here..." required></textarea>
<div class="fgrid">
<div><label>Scanners</label><select name="mode"><option value="all">All three scanners</option><option value="both">Scanners 1 + 2</option><option value="draw">Draw scanner only</option>
<option value="classic">Scanner 1 only</option><option value="regular">Scanner 2 only</option><option value="single">Single selection</option></select></div>
<div><label>Selection tool</label><select name="selection_source"><option value="stats">Statistics only</option><option value="odds_first">Odds first</option></select></div>
<div><label>All3 above (0.20 = 20%)</label><input type="number" name="threshold" step="0.01" min="0" max="1" value="{threshold}"></div>
<div><label>Each leg price at least</label><input type="number" name="price_floor" step="0.01" min="1" max="10" value="1.25"></div>
<div><label>Draw rating above</label><input type="number" name="draw_min" step="0.01" min="0" max="1" value="0.40"></div>
<div><label>Show</label><select name="show"><option value="qualified">Qualified fixtures only</option><option value="all">All scored fixtures</option></select></div>
</div>
<label class="check"><input type="checkbox" name="use_feed" value="true" checked> Price fixtures without odds from the open fixtures feed</label>
<label class="check"><input type="checkbox" name="log_picks" value="true"> Save picks to picks.csv (to check results later)</label>
<details><summary>Advanced</summary><div class="fgrid">
<div><label>Show floor (All3)</label><input type="number" name="min_threshold" step="0.01" min="0" max="1" value="{min_threshold}"></div>
<div><label>Max xG</label><input type="number" name="max_xg" step="0.05" min="0.1" max="10" value="{max_xg}"></div>
<div><label>Single: price from</label><input type="number" name="price_min" step="0.01" min="1.01" max="10" value="1.30"></div>
<div><label>Single: price to</label><input type="number" name="price_max" step="0.01" min="1.01" max="10" value="1.35"></div>
<div><label>Scanner 2 qualify by</label><select name="qualify_by"><option value="all3_floor">All3 line + price floor</option><option value="band">Leg price band</option><option value="all3">All3 line only</option></select></div>
</div><label class="check"><input type="checkbox" name="include_unrated" value="true"> Include fixtures with no statistics (uninformative)</label></details>
<button class="btn" type="submit">Run scanners</button>
</form>
<p class="mute" style="text-align:center">Only fixtures that meet the scanner rules are shown. Probabilities are model estimates, not guarantees.</p>"""
    return _page("Fixture Scanner", body, _FORM_JS)


# ------------------------------------------------------------------------------ dashboard
def _ring(p: float, color: str) -> str:
    return f'<div class="ring" style="--p:{p * 100:.1f};--c:var({color})"><span>{p * 100:.1f}%</span></div>'


def _grade_class(g: str) -> str:
    return g.replace(" ", "")


def _scanner_tile(title: str, r, floor: float, line: float) -> str:
    if not r:
        return f'<div class="tile"><h3>{_e(title)}</h3><span class="badge b-no">Not scored</span><p class="mute">No statistics or odds for this fixture.</p></div>'
    st = r["status"]
    if st.startswith("★"):
        cls, badge, bc = "ok", "QUALIFIED", "b-ok"
    elif st.startswith("◇"):
        cls, badge, bc = "warn", f"PRICE BELOW {floor:.2f}", "b-warn"
    else:
        cls, badge, bc = "", f"BELOW {line * 100:.0f}%", "b-no"
    band = '<span class="badge b-hi">HIGHLY EXPECTED 50–100%</span> ' if r["all3_prob"] >= 0.50 else ""
    color = "--green" if st.startswith("★") else "--amber" if st.startswith("◇") else "--mute"
    legs = "".join(f'<li><span>{_e(lg["label"])} <span class="mute">{lg["prob"] * 100:.0f}%</span></span>'
                   f'<span class="price {"p-ok" if lg["price"] >= floor else "p-low"}">@{lg["price"]:.2f}</span></li>' for lg in r["legs"])
    return (f'<div class="tile {cls}"><h3>{_e(title)}</h3><div class="top">{_ring(r["all3_prob"], color)}'
            f'<div>{band}<span class="badge {bc}">{badge}</span><div class="mute">All3 probability</div></div></div><ul class="legs">{legs}</ul></div>')


def _draw_tile(d, draw_min: float, watch_min: float = 0.30) -> str:
    title = "Scanner 3 · Draw"
    if not d:
        return f'<div class="tile"><h3>{title}</h3><span class="badge b-no">Not covered</span><p class="mute">No draw rating for this fixture.</p></div>'
    if d.get("prob") is None:
        return f'<div class="tile"><h3>{title}</h3><span class="badge b-no">No rating</span><p class="mute">{_e(d["reason"])}</p></div>'
    p = d["prob"]
    cls, badge, bc = ("ok", "DRAW PICK", "b-ok") if d["qualified"] else ("warn", "WATCHLIST", "b-warn") if d.get("watch") else ("", f"BELOW {draw_min * 100:.0f}%", "b-no")
    color = "--green" if d["qualified"] else "--amber" if d.get("watch") else "--mute"
    drv = "".join(f'<span class="{_e(x["state"])}">{"✓ " if x["state"] == "yes" else "✗ " if x["state"] == "against" else ""}{_e(x["driver"])}</span>' for x in (d.get("drivers") or []))
    blend = f' · 4-league DC/GB {d["blend_prob"] * 100:.1f}%' if d.get("blend_prob") is not None else ""
    return (f'<div class="tile {cls}"><h3>{title}</h3><div class="top">{_ring(min(p, 1), color)}<div><span class="badge {bc}">{badge}</span>'
            f'<div class="mute">Draw rating · price @{d["price"]:.2f} · fair {d["fair_odds"]:.2f}</div></div></div>'
            f'<div class="bar"><i style="width:{min(p / 0.5, 1) * 100:.1f}%"></i><u style="left:{watch_min / 0.5 * 100:.1f}%"></u><u style="left:{draw_min / 0.5 * 100:.1f}%;background:var(--red)"></u></div>'
            f'<div class="scale"><span>0%</span><span>30% watch</span><span>{draw_min * 100:.0f}% line</span><span>50%</span></div>'
            f'<div class="mute" style="margin-top:6px">edge vs price {d["edge"] * 100:+.1f}%{blend}</div><div class="drv">{drv}</div></div>')


def _profile(m) -> str:
    pr = m.get("profile_ratings")
    if not pr:
        return ""
    cols = []
    for key, title in (("home", "HOME TEAM"), ("away", "AWAY TEAM")):
        t = pr[key]
        rows = "".join(f'<div class="row"><span>{_e(lab)}<small>{_e(det)}</small></span><span class="g {_grade_class(g)}">{_e(g)}</span></div>' for lab, g, det in t["rows"])
        cols.append(f'<div><h4>√{title} <em>{_e(t["name"])} · form {_e(t["form"] or "-")}</em></h4>{rows}</div>')
    a = pr["alignment"]
    return (f'<details><summary>Team profile · home strong on {a["home_strong"]}/{a["of"]} · away weak on {a["away_weak"]}/{a["of"]}</summary>'
            f'<div class="prof">{"".join(cols)}</div></details>')


def _fixture_card(m, res) -> str:
    floor, line, draw_min = res["price_floor"], res["threshold"], res.get("draw_min", 0.40)
    tiles = []
    if "classic" in res.get("scanners", ("classic", "regular", "draw")):
        tiles.append(_scanner_tile("Scanner 1 · 1X · Home +1.5 · Under 2.5", m["classic"], floor, line))
    if "regular" in res.get("scanners", ("classic", "regular", "draw")):
        tiles.append(_scanner_tile("Scanner 2 · 1X · 12 · Over 1.5", m["regular"], floor, line))
    if res.get("mode") in ("all", "draw"):
        tiles.append(_draw_tile(m["draw"], draw_min))
    return (f'<div class="card"><div class="fx-head"><span class="time">{_e(m["time"])}</span><span class="league">{_e(m["competition"])}</span>'
            f'<div class="match">{_e(m["match"])}</div></div><div class="tiles">{"".join(tiles)}</div>{_profile(m)}</div>')


def _top_picks(res) -> str:
    picks = []
    for m in res["ranked_fixtures"]:
        for k, name in (("classic", "Scanner 1"), ("regular", "Scanner 2")):
            r = m[k]
            if r and r["status"].startswith("★"):
                picks.append((r["all3_prob"], name, m, r))
    picks.sort(key=lambda t: -t[0])
    if not picks:
        return ""
    items = "".join(f'<div class="stat"><div class="n g">{p * 100:.1f}%</div><div class="l"><b>{_e(m["match"])}</b><br>{_e(name)} · {_e(m["time"])}</div></div>' for p, name, m, r in picks[:3])
    return f'<h2>Top picks by All3</h2><div class="stats">{items}</div>'


def _empty(res) -> str:
    near = "".join(f'<div class="row"><span><b>{_e(n["match"])}</b> <span class="mute">{_e(n["time"])}</span><small>{_e(n["reason"])}</small></span>'
                   f'<span class="g Average">{n["all3"] * 100:.0f}%</span></div>' for n in res.get("near_misses", []))
    return ('<div class="card empty"><b>No fixture meets the scanner rules</b><p class="mute">Nothing in this paste has All3 above '
            f'{res["threshold"] * 100:.0f}% with every leg priced at {res["price_floor"]:.2f} or more, or a draw rating above {res.get("draw_min", 0.4) * 100:.0f}%.</p></div>'
            + (f'<h2>Closest misses</h2><div class="card">{near}</div>' if near else ""))


def render_dashboard(res: dict) -> str:
    s, floor, line = res["summary"], res["price_floor"], res["threshold"]
    draw_min = res.get("draw_min", 0.40)
    res = {**res, "scanners": ("classic", "regular", "draw") if res.get("mode") == "all" else ("draw",) if res.get("mode") == "draw" else ("classic", "regular")}
    stats = (f'<div class="stats"><div class="stat"><div class="n">{s["total_fixtures"]}</div><div class="l">Fixtures read</div></div>'
             f'<div class="stat"><div class="n t">{s["scored_count"]}</div><div class="l">Scored</div></div>'
             f'<div class="stat"><div class="n g">{s["qualified_count"]}</div><div class="l">Qualified</div></div>'
             f'<div class="stat"><div class="n v">{s["highly_expected"]}</div><div class="l">Highly expected 50–100%</div></div>'
             f'<div class="stat"><div class="n a">{s["watch_draw"]}</div><div class="l">Draw watchlist</div></div></div>')
    chips = (f'<div class="chips"><span class="chip">All3 above <b>{line * 100:.0f}%</b></span><span class="chip">Each leg at least <b>{floor:.2f}</b></span>'
             f'<span class="chip">Draw rating above <b>{draw_min * 100:.0f}%</b></span><span class="chip">By kickoff time</span>'
             f'<span class="chip">Selection: <b>{"statistics only" if res.get("selection_source") == "stats" else "odds first"}</b></span></div>')
    warn = ""
    if res["warnings"]:
        warn = '<details class="warn-box"><summary>Read before you stake</summary><ul>' + "".join(f"<li>{_e(w)}</li>" for w in res["warnings"]) + "</ul></details>"
    logged = f'<p class="mute">Saved {res["logged"]} picks to the pick log.</p>' if "logged" in res else ""
    cards = "".join(_fixture_card(m, res) for m in res["ranked_fixtures"]) if res["ranked_fixtures"] else _empty(res)
    watch = ""
    wl = [m for m in res.get("watchlist", []) if m not in res["ranked_fixtures"]]
    if wl:
        watch = ('<h2>Draw watchlist · below your line</h2>' + "".join(
            f'<div class="card"><div class="fx-head"><span class="time">{_e(m["time"])}</span><span class="league">{_e(m["competition"])}</span><div class="match">{_e(m["match"])}</div></div>'
            f'<div class="tiles">{_draw_tile(m["draw"], draw_min)}</div>{_profile(m)}</div>' for m in wl))
    report = ('<h2>Report</h2><button class="btn" onclick="copyReport(this)">Copy report</button><details><summary>Show report text</summary>'
              f'<pre class="rep" id="rep">{_e(res.get("report_text", ""))}</pre></details>') if res.get("report_text") else ""
    body = (_brand("Results · listed by kickoff time") + chips + stats + warn + _top_picks(res) + "<h2>Fixtures by kickoff time</h2>" + cards + watch + report + logged
            + '<a class="back" href="/">← New scan</a>')
    js = """<script>
async function copyReport(b){var t=document.getElementById('rep').innerText;try{await navigator.clipboard.writeText(t);b.textContent='Copied ✓';}
catch(e){var r=document.createRange();r.selectNodeContents(document.getElementById('rep'));var s=getSelection();s.removeAllRanges();s.addRange(r);b.textContent='Selected - press copy';}}
</script>"""
    return _page("Scan results", body, js)


# ------------------------------------------------------------------------------ simple view (single selection / one scanner)
def render_results(res: dict, threshold: float, min_threshold: float) -> str:
    if res.get("rank_by") == "time":
        return render_dashboard(res)
    s = res["summary"]
    copy_text = res["copyable_block"] or "No fixtures met the scanner rules."
    warnings = "".join(f"<li>{_e(w)}</li>" for w in res.get("warnings", []))
    if res.get("rank_by") == "selection":
        heading = f"Single selections priced {res['price_window'][0]:.2f}-{res['price_window'][1]:.2f}"
    else:
        heading = f"Ranked by {'All3' if res.get('rank_by') == 'all3' else 'weakest leg'} (★ = {threshold * 100:g}%+)"
    saved = f"Saved {res['logged']} picks. " if "logged" in res else ""
    body = (_brand("Results") + f'<div class="stats"><div class="stat"><div class="n">{s["total_fixtures"]}</div><div class="l">Fixtures</div></div>'
            f'<div class="stat"><div class="n t">{s["ranked_count"]}</div><div class="l">Ranked</div></div><div class="stat"><div class="n g">{s["qualified_count"]}</div><div class="l">★</div></div></div>'
            + (f'<details class="warn-box"><summary>Read before you stake</summary><ul>{warnings}</ul></details>' if warnings else "")
            + f'<p class="mute">{_e(saved)}</p><h2>{_e(heading)}</h2><pre class="rep" id="rep">{_e(copy_text)}</pre>'
            '<button class="btn" onclick="copyReport(this)">Copy results</button><a class="back" href="/">← Back</a>')
    js = """<script>async function copyReport(b){var t=document.getElementById('rep').innerText;try{await navigator.clipboard.writeText(t);b.textContent='Copied ✓';}catch(e){b.textContent='Select and copy';}}</script>"""
    return _page("Scan results", body, js)


def render_error(message: str) -> str:
    return _page("Error", _brand("Something needs attention") + f'<div class="warn-box">{_e(message)}</div><a class="back" href="/">← Back</a>')
