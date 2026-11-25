"""HTML rendering for the mobile UI. All dynamic text is escaped."""
from __future__ import annotations

from html import escape

_CSS = """
body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#f8fafc;padding:16px;margin:0}
.container{max-width:720px;margin:auto}
h2{font-size:1.25rem;margin-bottom:8px;color:#38bdf8}
h3{color:#38bdf8;margin:18px 0 6px;font-size:1.05rem}
textarea{width:100%;height:200px;background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:8px;padding:10px;box-sizing:border-box;font-size:13px;font-family:monospace}
input[type=number]{width:100%;background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:8px;box-sizing:border-box}
label{font-size:12px;color:#94a3b8;display:block}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}
button{width:100%;padding:12px;margin-top:10px;background:#0284c7;color:#fff;border:0;border-radius:8px;font-weight:bold;font-size:16px;cursor:pointer}
button.copy{background:#10b981;font-size:15px;margin-top:8px}
.stats{display:flex;justify-content:space-between;gap:8px;background:#1e293b;padding:12px;border-radius:8px;font-weight:bold;font-size:14px;border:1px solid #334155}
.warn{background:#422006;border:1px solid #b45309;color:#fcd34d;padding:10px;border-radius:8px;margin-top:12px;font-size:13px}
pre{background:#1e293b;padding:12px;border-radius:8px;overflow-x:auto;font-size:13px;border:1px solid #334155;white-space:pre-wrap;font-family:monospace}
a{display:block;text-align:center;margin-top:16px;color:#38bdf8;text-decoration:none}
"""


def _page(title: str, body: str) -> str:
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f'<body><div class="container">{body}</div></body></html>'
    )


_FORM_JS = """
<script>
(function () {
  var KEY = 'scanner_params_v10', names = ['threshold', 'min_threshold', 'max_xg', 'price_min', 'price_max', 'qualify_by', 'mode', 'price_floor', 'selection_source'];
  var form = document.querySelector('form');
  function field(n) { return form.querySelector('[name=' + n + ']'); }
  try {
    var saved = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (saved) {
      names.forEach(function (n) { if (saved[n] != null) field(n).value = saved[n]; });
      field('include_unrated').checked = !!saved.include_unrated;
      field('log_picks').checked = !!saved.log_picks;
      field('show_only').checked = !!saved.show_only;
      if (saved.use_feed != null) field('use_feed').checked = !!saved.use_feed;
    }
  } catch (e) {}
  form.addEventListener('submit', function () {
    try {
      var out = { include_unrated: field('include_unrated').checked, log_picks: field('log_picks').checked, show_only: field('show_only').checked, use_feed: field('use_feed').checked };
      names.forEach(function (n) { out[n] = field(n).value; });
      localStorage.setItem(KEY, JSON.stringify(out));
    } catch (e) {}
  });
})();
</script>
"""


def render_form(threshold: float, min_threshold: float, max_xg: float) -> str:
    return _page(
        "Flashscore Dixon-Coles Scanner",
        f"""<h2>Flashscore Mobile Fixture Scanner</h2>
<form method="post" action="/ui-scan">
<textarea name="raw_text" placeholder="Paste raw Flashscore mobile copy here..." required></textarea>
<div class="row">
<div><label>All3 ≥ (20% stats / 50% odds)</label><input type="number" name="threshold" step="0.005" min="0" max="1" value="{threshold}"></div>
<div><label>Show ≥</label><input type="number" name="min_threshold" step="0.01" min="0" max="1" value="{min_threshold}"></div>
<div><label>Max xG</label><input type="number" name="max_xg" step="0.05" min="0.1" max="10" value="{max_xg}"></div>
<div><label>Price from</label><input type="number" name="price_min" step="0.01" min="1.01" max="10" value="1.30"></div>
<div><label>Price to</label><input type="number" name="price_max" step="0.01" min="1.01" max="10" value="1.35"></div>
<div><label>Selection tool</label><select name="selection_source" style="width:100%;background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:8px"><option value="stats">Statistics only (form, goals against, history)</option><option value="odds_first">Odds first, then statistics</option></select></div>
<div><label>Mode</label><select name="mode" style="width:100%;background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:8px"><option value="all">All three scanners</option><option value="both">Classic + 1X/12/Over 1.5</option><option value="draw">Draw scanner only</option><option value="classic">Classic only (1X, H+1.5, U2.5)</option><option value="regular">1X + 12 + Over 1.5 only</option><option value="single">Single selection</option></select></div>
<div><label>Leg price ≥</label><input type="number" name="price_floor" step="0.01" min="1" max="10" value="1.30"></div>
<div><label>1X+12+O1.5: qualify by</label><select name="qualify_by" style="width:100%;background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:8px"><option value="all3_floor">All3 line + price floor</option><option value="band">Leg price band</option><option value="all3">All3 line only</option></select></div>
</div>
<label style="margin-top:10px"><input type="checkbox" name="include_unrated" value="true"> Include fixtures with no odds/ratings (identical, uninformative scores)</label>
<label style="margin-top:6px"><input type="checkbox" name="show_only" value="true"> Show only qualified fixtures</label>
<label style="margin-top:6px"><input type="checkbox" name="use_feed" value="true" checked> Price fixtures without odds from the open fixtures feed</label>
<label style="margin-top:6px"><input type="checkbox" name="log_picks" value="true"> Save these picks to picks.csv (to check results later)</label>
<button type="submit">Run Flashscore Pipeline</button>
</form>""" + _FORM_JS,
    )


def render_results(res: dict, threshold: float, min_threshold: float) -> str:
    s = res["summary"]
    copy_text = res["copyable_block"] or "No fixtures met the probability band."
    preview = res["cleaned_schedule"] or "No valid fixtures parsed."
    if res.get("rank_by") == "time":
        heading = ("All three scanners (classic · 1X+12+Over 1.5 · draw), by kickoff time" if res.get("mode") == "all" else "Draw scanner, by kickoff time" if res.get("mode") == "draw" else "Both scanners, by kickoff time") + " (★ qualified · ◇ All3 ok, price below floor)"
    elif res.get("rank_by") == "selection":
        heading = f"Single selections priced {res['price_window'][0]:.2f}-{res['price_window'][1]:.2f}"
    else:
        heading = f"Ranked by {'All3' if res.get('rank_by') == 'all3' else 'weakest leg'} (★ = {threshold * 100:g}%+)"
    warnings = "".join(f'<div class="warn">{escape(w)}</div>' for w in res.get("warnings", []))
    return _page(
        "Scan Results",
        f"""<div class="stats"><span>Leagues: {s['total_leagues']}</span>
<span>Fixtures: {s['total_fixtures']}</span>
<span>Ranked ({min_threshold * 100:g}%+): {s['ranked_count']}</span>
<span>★ {s['qualified_count']}{(' (classic ' + str(s['qualified_classic']) + ' · 1X+12+O1.5 ' + str(s['qualified_regular']) + ((' · draw ' + str(s['qualified_draw'])) if 'qualified_draw' in s else '') + ')') if 'qualified_classic' in s else ''}</span><span>Filtered on price: {s['in_band_count']}</span></div>
<p style="font-size:12px;color:#94a3b8">{('Saved ' + str(res['logged']) + ' picks. ') if 'logged' in res else ''}From odds: {s['from_odds']} · form model: {s['from_form']} · ratings: {s['from_ratings']} · Skipped: {s['skipped_no_data']} no data, {s['skipped_in_play']} live/finished, {s['skipped_postponed']} postponed</p>
{warnings}
<h3>{heading}</h3>
<pre id="copyBlock">{escape(copy_text)}</pre>
<button class="copy" onclick="copyResults(this)">Copy Results</button>
<h3>Cleaned Schedule</h3>
<pre>{escape(preview)}</pre>
<a href="/">← Back</a>
<script>
async function copyResults(btn) {{
  const text = document.getElementById('copyBlock').innerText;
  try {{ await navigator.clipboard.writeText(text); btn.textContent = 'Copied ✓'; }}
  catch (e) {{
    const r = document.createRange(); r.selectNodeContents(document.getElementById('copyBlock'));
    const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); btn.textContent = 'Selected - press copy';
  }}
}}
</script>""",
    )


def render_error(message: str) -> str:
    return _page("Error", f'<div class="warn">{escape(message)}</div><a href="/">← Back</a>')
