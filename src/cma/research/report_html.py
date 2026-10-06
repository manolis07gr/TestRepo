"""Self-contained HTML rendering of an edge-evaluation result (charts as inline SVG).

``render_fragment`` returns page content without <html>/<head>/<body> (artifact-ready);
``write_html`` wraps it as a standalone document for local viewing.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cma.research.report import (
    edge_band,
    live_cell,
    live_primary,
    positive_maker_lags,
    risk_overlay_sentence,
)

LAT = (0, 100, 250, 500, 1000, 2000, 5000)
EVIDENCE = (
    (
        "Polymarket BTC 15-minute quotes reprice after large Binance moves with a median of "
        "~347 ms; spreads are usually one tick; a 43-feature walk-forward model did not beat the "
        "book-implied probability and lost ~0.12 payoff units per trade after fees.",
        "OpenMarket dataset paper, arXiv 2607.26245 (Jul 2026)",
        "medium-high",
    ),
    (
        "Polymarket introduced taker fees on 15-minute crypto markets in Jan 2026 explicitly to "
        "curb latency arbitrage, extended them to all crypto markets (0.07·C·p(1−p)) and runs a "
        "150 ms taker delay on crypto up/down markets.",
        "docs.polymarket.com trading fees and order lifecycle pages",
        "official",
    ),
    (
        "Kalshi crypto contracts settle on the simple average of the CF Benchmarks real-time "
        "index over the 60 s before the stated time; taker fee 0.07·C·P(1−P), rounded up per "
        "order; WebSocket market data requires API-key authentication.",
        "Kalshi contract terms (BTC.pdf), fee schedule (Jul 2026), API docs",
        "official",
    ),
    (
        "Bot-like wallets account for ~86% of taker dollars in 5-minute crypto markets; average "
        "fee ~0.96% on Polymarket vs 2.74% modelled for Kalshi (Jan–Jun 2026).",
        "Pantera Research Lab, “Crypto on the Clock”",
        "medium",
    ),
    (
        "Kalshi contracts under 10¢ lose over 60%; takers lose ~32% on average, makers ~10% "
        "(favourite–longshot bias, 300k+ contracts).",
        "Bürgi, Deng & Whelan, UCD WP 2025/19",
        "high",
    ),
    (
        "Cross-venue BTC contracts reference different prices (Kalshi BRTI 60 s mean vs "
        "Polymarket Binance BTCUSDT candles or Chainlink TWAP): basis trades, not arbitrage.",
        "venue rules; mapping registry equivalence check",
        "official",
    ),
)

STYLE = """
:root{
  /* Layout: one readable column for the memo; charts break out to a wider grid. */
  --paper:#f5f7f9; --sheet:#ffffff; --ink:#151a22; --ink-2:#465061; --ink-3:#7a8494;
  --rule:#dde2e8; --rule-strong:#c4ccd6; --accent:#0e5a87; --accent-soft:#e3eef6;
  --neg:#b4332f; --pos:#1f7a3a; --warn:#9a6400;
  --s1:#2a78d6; --s2:#eb6834; --grid:#e6e9ee; --surface:#fcfcfb;
  --f-display:"Source Serif 4", Georgia, "Times New Roman", serif;
  --f-body:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono:"IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0f1216; --sheet:#151a20; --ink:#eef1f5; --ink-2:#b8c0cc; --ink-3:#8a93a1;
    --rule:#262d36; --rule-strong:#36404c; --accent:#6fb3e3; --accent-soft:#16283a;
    --neg:#ef7c74; --pos:#5cc27a; --warn:#e2b04f;
    --s1:#3987e5; --s2:#d95926; --grid:#232a33; --surface:#151a20; color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --paper:#0f1216; --sheet:#151a20; --ink:#eef1f5; --ink-2:#b8c0cc; --ink-3:#8a93a1;
  --rule:#262d36; --rule-strong:#36404c; --accent:#6fb3e3; --accent-soft:#16283a;
  --neg:#ef7c74; --pos:#5cc27a; --warn:#e2b04f;
  --s1:#3987e5; --s2:#d95926; --grid:#232a33; --surface:#151a20; color-scheme:dark;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:15px;
  line-height:1.6;margin:0}
.wrap{max-width:960px;margin:0 auto;padding-inline:16px;padding-block:28px 64px}
.memo{max-width:68ch}
header.top{display:grid;gap:10px;padding-block:8px 20px;border-bottom:1px solid var(--rule)}
.eyebrow{font-family:var(--f-mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3)}
h1{font-family:var(--f-display);font-weight:600;font-size:clamp(28px,4.4vw,40px);
  line-height:1.15;margin:0;text-wrap:balance}
h2{font-family:var(--f-display);font-weight:600;font-size:23px;margin:0;text-wrap:balance}
h3{font-size:15px;font-weight:600;margin:0}
p{margin:0}
.lede{color:var(--ink-2);font-size:16px;max-width:68ch}
section{display:grid;gap:14px;padding-block:28px;border-bottom:1px solid var(--rule)}
section:last-of-type{border-bottom:0}
.verdicts{display:flex;flex-wrap:wrap;gap:10px}
.verdict{display:grid;gap:2px;padding:12px 14px;border:1px solid var(--rule-strong);
  border-radius:6px;background:var(--sheet);min-width:0;flex:1 1 220px}
.verdict .k{font-size:12px;color:var(--ink-3);letter-spacing:.04em;text-transform:uppercase}
.verdict .v{font-family:var(--f-mono);font-weight:600;font-size:15px}
.verdict .v.reject{color:var(--neg)} .verdict .v.more{color:var(--warn)}
.verdict .why{font-size:13px;color:var(--ink-2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{display:grid;gap:4px;padding:14px;background:var(--sheet);border:1px solid var(--rule);
  border-radius:6px;min-width:0}
.tile .label{font-size:13px;color:var(--ink-2)}
.tile .value{font-size:26px;font-weight:600;line-height:1.2}
.tile .note{font-size:12.5px;color:var(--ink-3)}
.neg{color:var(--neg)} .pos{color:var(--pos)}
ul.plain{margin:0;padding-left:18px;display:grid;gap:6px}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink-2)}
.key{display:inline-flex;align-items:center;gap:8px}
.key i{display:inline-block;width:18px;height:3px;border-radius:2px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:12px}
.panels.wide{grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
.panel .tablebox{margin-top:8px}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:6px;
  padding:10px 10px 6px;min-width:0}
.panel h3{font-size:13px;font-weight:600;color:var(--ink)}
.panel .sub{font-size:12px;color:var(--ink-3)}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{font-family:var(--f-mono);font-size:10px;fill:var(--ink-3)}
svg .zero{stroke:var(--rule-strong);stroke-width:1}
svg .gridline{stroke:var(--grid);stroke-width:1}
svg .s1{stroke:var(--s1);fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .s2{stroke:var(--s2);fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .d1{fill:var(--s1);stroke:var(--surface);stroke-width:2}
svg .d2{fill:var(--s2);stroke:var(--surface);stroke-width:2}
svg .hit{fill:transparent;cursor:crosshair}
svg .ci{stroke:var(--s1);stroke-width:1.5;stroke-linecap:round;opacity:.6}
.tablebox{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--sheet)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-weight:600;color:var(--ink-2);background:var(--accent-soft);position:sticky;top:0}
td{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
details{border:1px solid var(--rule);border-radius:6px;background:var(--sheet)}
details summary{cursor:pointer;padding:10px 12px;font-size:13px;color:var(--ink-2)}
details[open] summary{border-bottom:1px solid var(--rule)}
details .tablebox{border:0;border-radius:0}
.formula{font-family:var(--f-mono);font-size:13px;background:var(--accent-soft);
  padding:2px 6px;border-radius:4px;white-space:nowrap}
.evidence{display:grid;gap:10px}
.ev{display:grid;grid-template-columns:auto 1fr;gap:10px;align-items:start;min-width:0}
.cred{font-family:var(--f-mono);font-size:11px;padding:2px 7px;border-radius:999px;
  border:1px solid var(--rule-strong);color:var(--ink-2);white-space:nowrap}
.ev p{font-size:14px} .ev .src{font-size:12.5px;color:var(--ink-3)}
.tip{position:fixed;pointer-events:none;background:var(--sheet);color:var(--ink);
  border:1px solid var(--rule-strong);border-radius:6px;padding:8px 10px;font-size:12px;
  line-height:1.45;box-shadow:0 4px 18px rgba(0,0,0,.12);z-index:10;max-width:260px}
.tip b{font-family:var(--f-mono)}
code{font-family:var(--f-mono);font-size:13px}
a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
"""

SCRIPT = """
(function(){
  var tip=document.getElementById('tip');
  function show(e){var d=e.target.getAttribute('data-tip'); if(!d){return;}
    tip.innerHTML=d; tip.hidden=false; move(e);}
  function move(e){var x=e.clientX+14,y=e.clientY+14,w=tip.offsetWidth,h=tip.offsetHeight;
    if(x+w>window.innerWidth-8){x=e.clientX-w-14;} if(y+h>window.innerHeight-8){y=e.clientY-h-14;}
    tip.style.left=x+'px'; tip.style.top=y+'px';}
  function hide(){tip.hidden=true;}
  document.querySelectorAll('[data-tip]').forEach(function(el){
    el.addEventListener('mouseenter',show); el.addEventListener('mousemove',move);
    el.addEventListener('mouseleave',hide); el.addEventListener('focus',function(ev){
      var r=el.getBoundingClientRect(); show({target:el,clientX:r.right,clientY:r.top});});
    el.addEventListener('blur',hide);});
})();
"""


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _fmt(x: Any, nd: int = 2, sign: bool = False) -> str:
    v = _num(x)
    if v is None:
        return "n/a"
    if round(v, nd) == 0:
        v = 0.0  # no "−0" / "+0.00" for values that round to zero
        return f"{v:,.{nd}f}"
    s = f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}"
    return s.replace("-", "−")


def _cls(x: Any) -> str:
    v = _num(x)
    if v is None or abs(v) < 1e-12:
        return ""
    return "pos" if v > 0 else "neg"


GATE_LABELS = {
    "oos_net_pnl": "out-of-sample net P&L",
    "stress_latency_viable": "1 s latency stress",
    "stress_cost_viable": "1.5× fee stress",
    "profit_concentration": "profit concentration",
    "parameter_stability": "parameter stability",
    "sample_size": "sample size",
    "data_quality": "data quality",
    "mappings_approved_paper": "mapping approval",
    "forward_paper_period": "forward paper period",
}


def tidy_numbers(text: str) -> str:
    """Round over-long floats in stored reason strings (display only)."""
    return re.sub(r"-?\d+\.\d{5,}", lambda m: f"{float(m.group()):.2f}", text)


def _ms_label(ms: float) -> str:
    # non-breaking space keeps the number with its unit when the text wraps
    return f"{ms / 1000:g}\u00a0s" if ms >= 1000 else f"{ms:g}\u00a0ms"


def _attribution_table(attr: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(r['bucket']))}</td><td>{_fmt(r['contracts'], 0)}</td>"
        f"<td>{_fmt(r['perceived_c_per_contract'], 2, True)}</td>"
        f'<td class="{_cls(r["true_c_per_contract"])}">'
        f"{_fmt(r['true_c_per_contract'], 2, True)}</td></tr>"
        for r in attr.get("by_perceived_edge", [])
    )
    return (
        '<div class="tablebox"><table><thead><tr><th>signal saw</th><th>contracts</th>'
        "<th>perceived ¢</th><th>true ¢</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _weighted(attr: Mapping[str, Any], field: str) -> float | None:
    rows = attr.get("by_perceived_edge", [])
    q = sum(float(r["contracts"]) for r in rows if _num(r.get(field)) is not None)
    if q <= 0:
        return None
    return (
        sum(float(r["contracts"]) * float(r[field]) for r in rows if _num(r.get(field)) is not None)
        / q
    )


def _attribution_section(result: Mapping[str, Any], crit: int) -> str:
    panels = []
    summary: list[str] = []
    for mm in (350.0, 3000.0):
        row = next(
            (
                r
                for r in result.get("frontier", [])
                if r.get("model", "implied_vol") == "implied_vol"
                and float(r["mm_lag_ms"]) == mm
                and r["competitor_ms"] is not None
                and int(r["latency_ms"]) == crit
                and r.get("attribution")
            ),
            None,
        )
        if row is None:
            continue
        attr = row["attribution"]
        perceived, true = (
            _weighted(attr, "perceived_c_per_contract"),
            _weighted(attr, "true_c_per_contract"),
        )
        summary.append(
            f"with {_ms_label(mm)} makers, filled contracts looked worth "
            f"{_fmt(perceived, 2, True)}¢ each to the strategy and were truly worth "
            f"{_fmt(true, 2, True)}¢"
            if not summary
            else f"with {_ms_label(mm)} makers, {_fmt(perceived, 2, True)}¢ against "
            f"{_fmt(true, 2, True)}¢"
        )
        panels.append(
            f'<div class="panel"><h3>Makers re-quote in {_ms_label(mm)}</h3>'
            f'<div class="sub">{row["competitor_ms"]:g} ms competitor · {crit} ms latency · '
            f"{_fmt(row.get('contracts'), 0)} contracts</div>{_attribution_table(attr)}</div>"
        )
    if not panels:
        return ""
    return f"""
<section id="attribution">
  <h2>Where the edge goes</h2>
  <p class="memo">Each fill's net edge as the signal saw it, against its true net edge at fill
  time (true fair value minus price minus fee). In the model, {"; ".join(summary)}. The gap is
  adverse selection: the strategy trades hardest exactly when its own spot or volatility input
  is stale, and fast makers have already moved.</p>
  <div class="panels wide">{"".join(panels)}</div>
</section>"""


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    t = math.floor(lo / step) * step
    ticks = [round(t, 10)]
    while ticks[-1] < hi - 1e-9:  # the last tick must cover the maximum
        t += step
        ticks.append(round(t, 10))
    return ticks


def _panel_svg(
    series: Mapping[str, Mapping[int, Mapping[str, Any]]], y_lo: float, y_hi: float, title: str
) -> str:
    w, h = 300, 190
    left, right, top, bottom = 38, 10, 10, 30
    pw, ph = w - left - right, h - top - bottom
    ticks = _nice_ticks(y_lo, y_hi, 6)
    y_min, y_max = ticks[0], ticks[-1]

    def x(i: int) -> float:
        return left + pw * i / (len(LAT) - 1)

    def y(v: float) -> float:
        return top + ph * (1 - (v - y_min) / (y_max - y_min))

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for t in ticks:
        cls = "zero" if abs(t) < 1e-12 else "gridline"
        parts.append(
            f'<line class="{cls}" x1="{left}" x2="{w - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y(t) + 3:.1f}" text-anchor="end">'
            f"{_fmt(t, 1 if abs(t) < 10 else 0)}</text>"
        )
    for i, lat in enumerate(LAT):
        label = f"{lat // 1000}s" if lat >= 1000 else f"{lat}"
        parts.append(
            f'<text x="{x(i):.1f}" y="{h - bottom + 14}" text-anchor="middle">{label}</text>'
        )
    parts.append(
        f'<text x="{left + pw / 2:.1f}" y="{h - 4}" text-anchor="middle">'
        f"our outbound latency (ms)</text>"
    )
    for key, cls in (("none", "1"), ("comp", "2")):
        pts = series.get(key, {})
        coords = [
            (x(i), y(v), lat, row)
            for i, lat in enumerate(LAT)
            if (row := pts.get(lat)) is not None
            and (v := _num(row.get("expected_c_per_contract"))) is not None
        ]
        if not coords:
            continue
        path = " ".join(
            f"{'M' if j == 0 else 'L'}{cx:.1f},{cy:.1f}" for j, (cx, cy, _, _) in enumerate(coords)
        )
        parts.append(f'<path class="s{cls}" d="{path}"/>')
        for cx, cy, lat, row in coords:
            comp = (
                "no competitor" if key == "none" else f"competitor {row.get('competitor_ms'):g} ms"
            )
            tipt = (
                f"{title} · {comp}<br>latency <b>{lat} ms</b><br>expected net "
                f"<b>{_fmt(row.get('expected_c_per_contract'), 2, True)}¢</b>/contract<br>"
                f"contracts filled <b>{_fmt(row.get('contracts'), 0)}</b><br>realized net "
                f"<b>${_fmt(row.get('realized_net_pnl'), 0)}</b> (settlement noise)"
            )
            parts.append(f'<circle class="d{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/>')
            parts.append(
                f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="11" tabindex="0" '
                f'data-tip="{html.escape(tipt)}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _frontier_section(frontier: Sequence[Mapping[str, Any]]) -> str:
    rows = [r for r in frontier if r.get("model", "implied_vol") == "implied_vol"]
    by_mm: dict[float, dict[str, dict[int, Mapping[str, Any]]]] = {}
    for r in rows:
        key = "none" if r["competitor_ms"] is None else "comp"
        by_mm.setdefault(float(r["mm_lag_ms"]), {}).setdefault(key, {})[int(r["latency_ms"])] = r
    vals = [v for r in rows if (v := _num(r.get("expected_c_per_contract"))) is not None]
    lo, hi = (min([*vals, 0.0]), max([*vals, 0.0])) if vals else (-1.0, 1.0)
    pad = 0.04 * (hi - lo or 1.0)
    panels = []
    for mm in sorted(by_mm):
        title = f"maker lag {mm:g} ms"
        sub = "median re-quote delay after a move"
        panels.append(
            f'<div class="panel"><h3>{title}</h3><div class="sub">{sub}</div>'
            f"{_panel_svg(by_mm[mm], lo - pad, hi + pad, title)}</div>"
        )
    comp_ms = next((r["competitor_ms"] for r in rows if r["competitor_ms"] is not None), 120)
    table_rows = []
    for mm in sorted(by_mm):
        for key, label in (("none", "none"), ("comp", f"{comp_ms:g} ms")):
            cells = [by_mm[mm].get(key, {}).get(lat) for lat in LAT]
            table_rows.append(
                f"<tr><td>{mm:g} ms · {label}</td>"
                + "".join(
                    f'<td class="{_cls(c.get("expected_c_per_contract") if c else None)}">'
                    f"{_fmt(c.get('expected_c_per_contract') if c else None, 2, True)}</td>"
                    for c in cells
                )
                + "</tr>"
            )
    rv = [r for r in frontier if r.get("model") == "realized_vol"]
    rv_rows = []
    if rv:
        grp: dict[tuple[float, Any], dict[int, Any]] = {}
        for r in rv:
            grp.setdefault((float(r["mm_lag_ms"]), r["competitor_ms"]), {})[
                int(r["latency_ms"])
            ] = r
        for (mm, comp), d in sorted(grp.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            label = "none" if comp is None else f"{comp:g} ms"
            rv_rows.append(
                f"<tr><td>{mm:g} ms · {label}</td>"
                + "".join(
                    f'<td class="{_cls((d.get(lat) or {}).get("expected_c_per_contract"))}">'
                    f"{_fmt((d.get(lat) or {}).get('expected_c_per_contract'), 2, True)}</td>"
                    for lat in LAT
                )
                + "</tr>"
            )
    head = "".join(f"<th>{lat} ms</th>" for lat in LAT)
    return f"""
<section id="frontier">
  <h2>Edge frontier</h2>
  <p class="memo">Expected net edge per contract for the lead-lag taker, valued at the
  <em>true</em> fair price at fill time minus price minus fee, so settlement luck drops out.
  Each panel is one market-maker speed; lines compare a market with and without a faster
  arbitrageur. Our latency on the x-axis (0 ms is a diagnostic, not a viable setting).</p>
  <div class="legend" aria-hidden="true">
    <span class="key"><i style="background:var(--s1)"></i>no competing arbitrageur</span>
    <span class="key"><i style="background:var(--s2)"></i>arbitrageur at {comp_ms:g} ms</span>
  </div>
  <div class="panels">{"".join(panels)}</div>
  <details><summary>Table view: expected net ¢ per contract (implied-vol model)</summary>
  <div class="tablebox"><table><thead><tr><th>maker lag · competitor</th>{head}</tr></thead>
  <tbody>{"".join(table_rows)}</tbody></table></div></details>
  {('<details><summary>Model risk: same markets, realized-vol fair value (expected ¢ / contract)</summary><div class="tablebox"><table><thead><tr><th>maker lag · competitor</th>' + head + "</tr></thead><tbody>" + "".join(rv_rows) + "</tbody></table></div></details>") if rv_rows else ""}
</section>"""


def _band(cell: Mapping[str, Any]) -> str:
    return edge_band(cell).replace("-", "−")


def _live_panel_svg(
    row: Mapping[str, Any], lats: Sequence[int], y_lo: float, y_hi: float, title: str
) -> str:
    """Mean market-anchored edge by latency with ±2 SE whiskers (one move threshold)."""
    w, h = 300, 190
    left, right, top, bottom = 38, 14, 10, 30
    pw, ph = w - left - right, h - top - bottom
    ticks = _nice_ticks(y_lo, y_hi, 6)
    y_min, y_max = ticks[0], ticks[-1]

    def x(i: int) -> float:
        return left + pw * i / max(len(lats) - 1, 1)

    def y(v: float) -> float:
        return top + ph * (1 - (v - y_min) / (y_max - y_min))

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for t in ticks:
        cls = "zero" if abs(t) < 1e-12 else "gridline"
        parts.append(
            f'<line class="{cls}" x1="{left}" x2="{w - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y(t) + 3:.1f}" text-anchor="end">'
            f"{_fmt(t, 1 if abs(t) < 10 else 0)}</text>"
        )
    for i, lat in enumerate(lats):
        label = f"{lat // 1000}s" if lat >= 1000 else f"{lat}"
        parts.append(
            f'<text x="{x(i):.1f}" y="{h - bottom + 14}" text-anchor="middle">{label}</text>'
        )
    parts.append(
        f'<text x="{left + pw / 2:.1f}" y="{h - 4}" text-anchor="middle">'
        f"our latency after the move (ms)</text>"
    )
    pts = [
        (i, lat, cell, m)
        for i, lat in enumerate(lats)
        if (m := _num((cell := live_cell(row, lat)).get("mean_c"))) is not None
    ]
    for i, _, cell, m in pts:
        se = _num(cell.get("se_c"))
        if se is not None:
            parts.append(
                f'<line class="ci" x1="{x(i):.1f}" x2="{x(i):.1f}" '
                f'y1="{y(m - 2 * se):.1f}" y2="{y(m + 2 * se):.1f}"/>'
            )
    if pts:
        path = " ".join(
            f"{'M' if j == 0 else 'L'}{x(i):.1f},{y(m):.1f}" for j, (i, _, _, m) in enumerate(pts)
        )
        parts.append(f'<path class="s1" d="{path}"/>')
    for i, lat, cell, m in pts:
        share = _num(cell.get("share_positive"))
        tipt = (
            f"{title}<br>latency <b>{lat} ms</b><br>net <b>{_fmt(m, 2, True)}¢</b>/contract"
            f" ({html.escape(_band(cell))} at 2 SE)<br>positive in "
            f"<b>{_fmt(100 * share if share is not None else None, 0)}%</b> of "
            f"{_fmt(cell.get('n'), 0)} quotes on {_fmt(cell.get('moves'), 0)} moves"
        )
        parts.append(f'<circle class="d1" cx="{x(i):.1f}" cy="{y(m):.1f}" r="4"/>')
        parts.append(
            f'<circle class="hit" cx="{x(i):.1f}" cy="{y(m):.1f}" r="11" tabindex="0" '
            f'data-tip="{html.escape(tipt)}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _took(v: float | None, lat: int, *, first: bool) -> str:
    """'lost 1.23¢ per contract at 100 ms' / 'netted 0.42¢ at 250 ms' / 'broke even at ...'."""
    if v is None:
        return f"had no quote to take at {lat} ms"
    if round(v, 2) == 0:
        return f"broke even at {lat} ms"
    return (
        f"{'netted' if v > 0 else 'lost'} {abs(v):.2f}¢{' per contract' if first else ''} "
        f"at {lat} ms"
    )


def _lead_rows(
    live: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """(row to lead with, primary row): the primary threshold when it has enough moves for
    the decision rule, else the best-populated threshold, with the primary kept beside it."""
    from cma.research.live_study import MIN_DECISION_MOVES

    rows = [r for r in live.get("pooled", []) if live_cell(r, 100).get("mean_c") is not None]
    prim = live_primary(live)
    if not rows:
        return None, prim
    if prim is not None and prim in rows and prim["moves"] >= MIN_DECISION_MOVES:
        return prim, prim
    return max(rows, key=lambda r: r["moves"]), prim


def _cell_text(row: Mapping[str, Any], lat: int) -> str:
    cell = live_cell(row, lat)
    return f"{_band(cell)}¢ at {lat} ms"


def live_lede(live: Mapping[str, Any]) -> str:
    """Data-driven opening sentences on the live window (they open the page when present)."""
    dec = (live.get("decision") or {}).get("decision", "COLLECT_MORE_DATA")
    hours = _num(live.get("window", {}).get("hours")) or 0.0
    lead, prim = _lead_rows(live)
    if lead is None:
        reasons = (live.get("decision") or {}).get("reasons") or ["no samples"]
        return f"Not yet measurable on real Kalshi books ({reasons[0]})."
    significant = any(
        (m := _num(live_cell(r, lat).get("mean_c"))) is not None
        and (se := _num(live_cell(r, lat).get("se_c"))) is not None
        and m - 2 * se > 0
        for r in live.get("pooled", [])
        for lat in (100, 250)
    )
    opener = (
        "Not on real Kalshi books."
        if dec == "REJECT"
        else ("Not proven on real Kalshi books." if significant else "Not on this evidence.")
    )
    rt = lead.get("reaction_ms") or {}
    med, p75, early = (_num(rt.get(k)) for k in ("median", "p75", "share_already_moved"))
    thr = f"≥ {lead['threshold_bps']:g} bps"
    if med is not None and med <= 50 and early is not None:
        react = (
            f" Over {hours:.1f} hours of live order books, Kalshi had usually repriced before a "
            f"{thr} BTC move even reached us ({100 * early:.0f}% of quotes already at least "
            f"halfway, three quarters within {_fmt(p75, 0)} ms)"
        )
    else:
        react = (
            f" Over {hours:.1f} hours of live order books, Kalshi makers repriced half of a "
            f"{thr} BTC move within a median {_fmt(med, 0)} ms of our seeing it"
        )
    c100, c250 = live_cell(lead, 100), live_cell(lead, 250)
    m100, m250 = _num(c100.get("mean_c")), _num(c250.get("mean_c"))
    same = ((m100 or 0) > 0) == ((m250 or 0) > 0)
    se = _num(c100.get("se_c"))
    band = f", ±{2 * se:.2f}¢ at two standard errors" if se is not None else ""
    edge = (
        f", and taking the quote after it {_took(m100, 100, first=True)} "
        f"{'and' if same else 'but'} {_took(m250, 250, first=False)} after the fee "
        f"({_fmt(c100.get('moves'), 0)} moves{band})"
    )
    from cma.research.live_study import MIN_DECISION_MOVES

    if dec == "REJECT":
        return f"{opener}{react}{edge}."
    if prim is not None and prim is not lead:
        pc = live_cell(prim, 100)
        tail = (
            f". After ≥ {prim['threshold_bps']:g} bps moves it was {_band(pc)}¢ on only "
            f"{_fmt(pc.get('moves'), 0)} moves, too few to decide"
        )
    elif prim is not None and prim["moves"] < MIN_DECISION_MOVES:
        tail = ", on too few moves to decide"
    else:
        tail = ", and one window cannot settle it"
    return f"{opener}{react}{edge}{tail}, so the pre-registered call is to collect more data."


def live_verdict(live: Mapping[str, Any]) -> tuple[str, str, str]:
    dec = live.get("decision") or {}
    hours = _num(live.get("window", {}).get("hours")) or 0.0
    lead, prim = _lead_rows(live)
    if lead is None:
        why = (dec.get("reasons") or ["no samples"])[0]
    else:
        rows = [lead] + ([prim] if prim is not None and prim is not lead else [])
        why = "; ".join(
            f"≥ {r['threshold_bps']:g} bps: {_cell_text(r, 100)}, {_cell_text(r, 250)} "
            f"({_fmt(r['moves'], 0)} moves)"
            for r in rows
        )
        why = f"per contract after the fee, ±2 SE. {why}"
    return (
        f"Real Kalshi books, {hours:.1f} h live",
        str(dec.get("decision", "COLLECT_MORE_DATA")),
        why,
    )


def _live_section(live: Mapping[str, Any]) -> str:
    win, ref, con = live.get("window", {}), live.get("reference", {}), live.get("contracts", {})
    lats = [int(x) for x in live.get("config", {}).get("latencies_ms", [0, 100, 250, 500, 1000])]
    pooled = live.get("pooled", [])
    lead, prim = _lead_rows(live)
    lt = (lead or {}).get("lifetime_ms", {})
    rt = (lead or {}).get("reaction_ms") or {}
    c100, c250 = live_cell(lead, 100), live_cell(lead, 250)
    thr = f"≥\u00a0{(lead or {}).get('threshold_bps', 5):g}\u00a0bps"  # keep on one line
    base = live.get("baseline_anchored", {})
    base_abs = live.get("baseline", {})
    early = _num(rt.get("share_already_moved"))
    med = _num(rt.get("median"))

    def tile_edge(cell: Mapping[str, Any], label: str, extra: str = "") -> str:
        se = _num(cell.get("se_c"))
        return (
            f'<div class="tile"><span class="label">{label}</span>'
            f'<span class="value {_cls(cell.get("mean_c"))}">'
            f"{_fmt(cell.get('mean_c'), 2, True)}¢</span>"
            f'<span class="note">per contract after the fee · ±{_fmt(2 * se if se else None, 2)}¢'
            f" (2 SE, clustered by move) · {_fmt(cell.get('n'), 0)} quotes on "
            f"{_fmt(cell.get('moves'), 0)} moves{extra}</span></div>"
        )

    if med is not None and med <= 50 and early is not None:
        reaction_tile = (
            '<div class="tile"><span class="label">Already repriced when the move reached us'
            f'</span><span class="value">{_fmt(100 * early, 0)}%</span>'
            f'<span class="note">of quotes after {thr} moves: Kalshi\'s mid had covered half '
            f"the predicted repricing · three quarters within {_fmt(rt.get('p75'), 0)} ms · "
            f"quote life median {_fmt(lt.get('median'), 0)} ms</span></div>"
        )
    else:
        reaction_tile = (
            '<div class="tile"><span class="label">Median maker reaction</span>'
            f'<span class="value">{_fmt(med, 0)} ms</span>'
            f'<span class="note">for Kalshi\'s mid to cover half the predicted repricing of a '
            f"{thr} move · p25–p75 {_fmt(rt.get('p25'), 0)}–{_fmt(rt.get('p75'), 0)} ms · "
            f"{_fmt(100 * early if early is not None else None, 0)}% repriced before we saw the "
            f"move · quote life median {_fmt(lt.get('median'), 0)} ms</span></div>"
        )
    prim_note = ""
    if prim is not None and prim is not lead:
        prim_note = (
            f" · ≥ {prim['threshold_bps']:g} bps moves: {_band(live_cell(prim, 100))}¢ on "
            f"{_fmt(live_cell(prim, 100).get('moves'), 0)} moves"
        )
    tiles = (
        '<div class="tiles">'
        + reaction_tile
        + tile_edge(c100, f"Taking it 100 ms after a {thr} move", prim_note)
        + tile_edge(c250, f"Taking it 250 ms after a {thr} move")
        + '<div class="tile"><span class="label">Trading without news</span>'
        f'<span class="value {_cls(base.get("mean_c"))}">{_fmt(base.get("mean_c"), 2, True)}¢'
        "</span>"
        '<span class="note">same valuation at fixed 10 s times: half the spread plus the fee'
        "</span></div></div>"
    )
    vals = [0.0]
    for r in pooled:
        for lat in lats:
            cell = live_cell(r, lat)
            m, se = _num(cell.get("mean_c")), _num(cell.get("se_c")) or 0.0
            if m is not None:
                vals += [m - 2 * se, m + 2 * se]
    lo, hi = min(vals), max(vals)
    pad = 0.04 * (hi - lo or 1.0)
    panels = "".join(
        f'<div class="panel"><h3>Moves ≥ {r["threshold_bps"]:g} bps in 1 s</h3>'
        f'<div class="sub">{_fmt(r["samples"], 0)} quotes on {_fmt(r["moves"], 0)} moves · '
        f"median reaction {_fmt((r.get('reaction_ms') or {}).get('median'), 0)} ms</div>"
        f"{_live_panel_svg(r, lats, lo - pad, hi + pad, f'moves ≥ {r["threshold_bps"]:g} bps')}"
        "</div>"
        for r in pooled
    )
    head = "".join(f"<th>{lat} ms</th>" for lat in lats)

    def edge_row(first: str, r: Mapping[str, Any]) -> str:
        react = r.get("reaction_ms") or {}
        return (
            f"<tr><td>{first}</td><td>{_fmt(r['samples'], 0)}</td><td>{_fmt(r['moves'], 0)}</td>"
            f"<td>{_fmt(react.get('p25'), 0)} / {_fmt(react.get('median'), 0)} / "
            f"{_fmt(react.get('p75'), 0)}</td>"
            f"<td>{_fmt(r['lifetime_ms'].get('median'), 0)}</td>"
            + "".join(
                f'<td class="{_cls(live_cell(r, lat).get("mean_c"))}">'
                f"{html.escape(_band(live_cell(r, lat)))}</td>"
                for lat in lats
            )
            + "</tr>"
        )

    pooled_rows = "".join(edge_row(f"≥ {r['threshold_bps']:g} bps", r) for r in pooled)
    series_rows = "".join(
        edge_row(f"≥ {r['threshold_bps']:g} bps · {html.escape(str(r['series']))}", r)
        for r in live.get("summary", [])
    )
    table_head = (
        "<thead><tr><th>move size</th><th>quotes</th><th>moves</th>"
        "<th>reaction p25 / median / p75 ms</th><th>quote life median ms</th>"
        f"{head}</tr></thead>"
    )
    ll_rows = "".join(
        f"<tr><td>{html.escape(r['contract'].split(':')[-1])}</td>"
        f"<td>{_fmt(r['updates'], 0)}</td>"
        f"<td>{_fmt(r['result'].get('best_lag_ms'), 0)}</td>"
        f"<td>{_fmt(r['result'].get('p_value'), 3)}</td>"
        f"<td>{_fmt(r['result'].get('incremental_oos_r2'), 3)}</td>"
        f"<td>{'yes' if r['result'].get('qualifies') else 'no'}</td></tr>"
        for r in live.get("lead_lag", [])
    )
    ll_html = (
        '<h3>Lead-lag: Coinbase mid to contract mid</h3><div class="tablebox"><table><thead><tr>'
        "<th>contract</th><th>mid changes</th><th>peak lag ms</th><th>p</th>"
        "<th>out-of-sample ΔR²</th><th>clears fees</th></tr></thead>"
        f"<tbody>{ll_rows}</tbody></table></div>"
        '<p class="memo" style="font-size:13px;color:var(--ink-3)">Peak lag of the '
        "cross-correlation: positive when Coinbase moves first, negative when Kalshi does.</p>"
        if ll_rows
        else ""
    )
    dec = live.get("decision") or {}
    dvol = _num(ref.get("dvol_mean"))
    by_series = ", ".join(
        f"{html.escape(str(k))} {_fmt(v, 0)}"
        for k, v in (con.get("quote_updates_by_series") or {}).items()
    )
    return f"""
<section id="live">
  <h2>Real Kalshi books, {_fmt(win.get("hours"), 1)} hours live</h2>
  <p class="memo">Kalshi's authenticated WebSocket order books for
  {_fmt(con.get("with_quotes"), 0)} BTC above-strike contracts (top-of-book changes:
  {by_series or "n/a"}) recorded next to the Coinbase BTC-USD ticker
  ({_fmt(ref.get("updates"), 0)} updates) from {html.escape(str(win.get("start", ""))[:16])} to
  {html.escape(str(win.get("end", ""))[11:16])} UTC, {_fmt(live.get("raw_messages"), 0)} raw
  messages in all. For every Coinbase move and every contract whose fair value it shifts by at
  least 1¢, the quote a taker would hit is valued at Kalshi's own mid one second before the move
  plus the model's change in fair value (volatility from
  {"Deribit DVOL, " + _fmt(100 * dvol, 0) + "%" if dvol is not None else "realised volatility"}),
  minus the price and the taker fee. Times are receive times on this machine.</p>
  {tiles}
  <div class="legend" aria-hidden="true">
    <span class="key"><i style="background:var(--s1)"></i>mean net ¢ per contract</span>
    <span class="key"><i style="background:var(--s1);opacity:.6;width:3px;height:14px"></i>±2 standard errors, clustered by move</span>
  </div>
  <div class="panels wide">{panels}</div>
  <details><summary>Table view: net ¢ per contract after the fee, ±2 SE (all series)</summary>
  <div class="tablebox"><table>{table_head}<tbody>{pooled_rows}</tbody></table></div></details>
  <details><summary>By series</summary>
  <div class="tablebox"><table>{table_head}<tbody>{series_rows}</tbody></table></div></details>
  <p class="memo">Without news the same valuation nets {_fmt(base.get("mean_c"), 2, True)}¢
  (positive {_fmt(100 * (_num(base.get("share_positive")) or 0), 0)}% of the time). Valued at
  the model price alone the baseline is {_fmt(base_abs.get("mean_c"), 2, True)}¢, which is model
  and basis disagreement with the market rather than speed, hence the market anchor.</p>
  {ll_html}
  <p class="memo"><b>{html.escape(str(dec.get("decision", "")).replace("_", " "))}</b> by the rule
  fixed before the run: {html.escape(str(dec.get("rule", "")).replace(">=", "≥"))}</p>
</section>"""


def _grid_table(
    base: Mapping[str, Any], field: str, nd: int = 0, sign: bool = True, prefix: str = ""
) -> str:
    table = base.get(field) or {}
    head = "".join(f"<th>{lat} ms</th>" for lat in LAT)
    rows = []
    for cost, by_lat in table.items():
        cells = []
        for lat in LAT:
            v = by_lat.get(str(lat), by_lat.get(lat))
            cells.append(f'<td class="{_cls(v)}">{prefix}{_fmt(v, nd, sign)}</td>')
        rows.append(f"<tr><td>{html.escape(str(cost))}</td>{''.join(cells)}</tr>")
    return (
        f'<div class="tablebox"><table><thead><tr><th>cost scenario</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _ex_ante_table(base: Mapping[str, Any]) -> str:
    exa = base.get("ex_ante") or {}
    costs = list(dict.fromkeys(k.split("@")[0] for k in exa))
    head = "".join(f"<th>{lat} ms</th>" for lat in LAT)
    rows = []
    for cost in costs:
        cells = "".join(
            f'<td class="{_cls((exa.get(f"{cost}@{lat}") or {}).get("expected_c_per_contract"))}">'
            f"{_fmt((exa.get(f'{cost}@{lat}') or {}).get('expected_c_per_contract'), 2, True)}</td>"
            for lat in LAT
        )
        rows.append(f"<tr><td>{html.escape(cost)}</td>{cells}</tr>")
    return (
        f'<div class="tablebox"><table><thead><tr><th>cost scenario</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_fragment(result: Mapping[str, Any], live: Mapping[str, Any] | None = None) -> str:
    man = result.get("manifest", {})
    plan = result.get("plan", {})
    bases = result.get("base_cases", [])
    kal = next((b for b in bases if b["label"] == "kalshi_fee"), bases[0] if bases else {})
    poly = next((b for b in bases if b is not kal), None)
    frontier = result.get("frontier", [])
    crit = int(plan.get("criterion_latency_ms", 250))

    def front(mm: float, comp: Any, lat: int) -> Mapping[str, Any] | None:
        for r in frontier:
            if (
                r.get("model", "implied_vol") == "implied_vol"
                and float(r["mm_lag_ms"]) == mm
                and r["competitor_ms"] == comp
                and int(r["latency_ms"]) == lat
            ):
                found: Mapping[str, Any] = r
                return found
        return None

    doc_case = front(350.0, 120.0, crit)
    slow_case = front(3000.0, 120.0, crit)
    grid_lags = sorted(
        {float(r["mm_lag_ms"]) for r in frontier if r.get("model", "implied_vol") == "implied_vol"}
    )
    pos_comp = positive_maker_lags(frontier, crit, "any")
    need = _ms_label(pos_comp[0]) if pos_comp else "none in grid"
    buyable = [lat for lat in LAT if lat >= 100]

    def best_buyable(mm: float, comp: Any) -> tuple[float, int] | None:
        cells = [
            (v, lat)
            for lat in buyable
            if (c := front(mm, comp, lat)) is not None
            and (v := _num(c.get("expected_c_per_contract"))) is not None
        ]
        return max(cells) if cells else None

    alone = best_buyable(350.0, None)  # reported maker speed, no rival arbitrageur
    rival = best_buyable(350.0, 120.0)  # reported maker speed, 120 ms arbitrageur
    ex = (kal.get("ex_ante") or {}).get(f"base@{crit}", {}) if kal else {}
    tail = (
        ""
        if live
        else " No real venue data could be collected here, so the real-market call is to "
        "collect data first."
    )
    if pos_comp and alone is not None and rival is not None and rival[0] < 0:
        if alone[0] < 0:
            at_350 = (
                "At the reported ~350 ms the expected edge is negative at every latency we can "
                "buy (100 ms and up), with or without a faster arbitrageur."
            )
        else:
            best = "break-even" if alone[0] < 0.25 else "thin"
            at_350 = (
                f"At the reported ~350 ms the best case is {best} ({_fmt(alone[0], 2, True)}¢ "
                f"per contract at {alone[1]} ms with no rival arbitrageur), and one 120 ms "
                f"arbitrageur makes it negative at every latency "
                f"({_fmt((doc_case or {}).get('expected_c_per_contract'), 2, True)}¢ at "
                f"{crit} ms)."
            )
        lede = (
            "Not at the speeds these markets are reported to run. In a market model calibrated "
            "to public evidence, a lead-lag taker clears the 0.07·p(1−p) taker fee reliably "
            f"only once makers take about {need} or longer to re-quote after a BTC move. "
            f"{at_350}{tail}"
        )
    else:
        lede = (
            f"In a market model calibrated to public evidence, 350 ms makers and a 120 ms "
            f"arbitrageur leave {_fmt((doc_case or {}).get('expected_c_per_contract'), 2, True)}¢ "
            f"of expected edge per contract at {crit} ms after the 0.07·p(1−p) taker fee.{tail}"
        )

    if live:
        lede = (
            f"{live_lede(live)} In a market model calibrated to public evidence, a lead-lag "
            "taker clears the 0.07·p(1−p) taker fee reliably only once makers take about "
            f"{need} or longer to re-quote after a BTC move."
        )
    verdicts = [
        live_verdict(live)
        if live
        else (
            "Real markets (Kalshi, Polymarket)",
            result.get("real_market_decision", "COLLECT_MORE_DATA"),
            "no venue data reachable from this environment; collection plan below",
        ),
    ]
    for b in bases:
        label = (
            "Synthetic base case, Kalshi fees"
            if b["label"] == "kalshi_fee"
            else "Synthetic variant, Polymarket fee + 150 ms delay"
        )
        failed = [
            GATE_LABELS.get(r.split(":")[0], r.split(":")[0])
            for r in b.get("decision_reasons", [])[:4]
        ]
        why = "failed: " + ", ".join(failed) if failed else "all gates passed"
        verdicts.append((label, b["decision"], why))

    def vclass(d: str) -> str:
        return "reject" if d == "REJECT" else ("more" if d == "COLLECT_MORE_DATA" else "")

    verdict_html = "".join(
        f'<div class="verdict"><span class="k">{html.escape(k)}</span>'
        f'<span class="v {vclass(v)}">{html.escape(v.replace("_", " "))}</span>'
        f'<span class="why">{html.escape(w)}</span></div>'
        for k, v, w in verdicts
    )

    tiles = f"""
<div class="tiles">
  <div class="tile"><span class="label">Taker fee at a 50¢ price</span>
    <span class="value">1.75¢</span>
    <span class="note">per $1 contract, both venues (<span class="formula">0.07·p(1−p)</span>)</span></div>
  <div class="tile"><span class="label">Expected edge, 350 ms makers, {crit} ms latency</span>
    <span class="value {_cls((doc_case or {}).get("expected_c_per_contract"))}">{_fmt((doc_case or {}).get("expected_c_per_contract"), 2, True)}¢</span>
    <span class="note">per contract, with a 120 ms arbitrageur (published maker speed)</span></div>
  <div class="tile"><span class="label">Expected edge, 3 s makers, {crit} ms latency</span>
    <span class="value {_cls((slow_case or {}).get("expected_c_per_contract"))}">{_fmt((slow_case or {}).get("expected_c_per_contract"), 2, True)}¢</span>
    <span class="note">same competitor; makers ~9× slower than reported</span></div>
  <div class="tile"><span class="label">Smallest maker lag with positive edge</span>
    <span class="value">{html.escape(need)}</span>
    <span class="note">grid {" · ".join(_ms_label(x) for x in grid_lags)}; 120 ms competitor, {crit} ms latency</span></div>
</div>"""

    base_html = ""
    if kal:
        ci = kal.get("ci_mean_position_pnl", {})
        mp = kal.get("market_params", {})
        parts = kal.get("partitions", {})
        mt = kal.get("multiple_testing", {})
        reasons = "".join(
            f"<li>{html.escape(tidy_numbers(r))}</li>" for r in kal.get("decision_reasons", [])
        )
        base_html = f"""
<section id="base">
  <h2>Base case on a locked final test</h2>
  <p class="memo">{mp.get("hours")} hours of synthetic market calibrated to the evidence:
  makers re-quote one-tick markets about {mp.get("mm_lag_ms"):g} ms after a move,
  arbitrageurs act at {mp.get("competitor_latency_ms"):g} ms, Kalshi fees, contracts settle on
  the 60 s index average. Parameters were chosen on the validation window
  ({mt.get("n_hypotheses")} settings, every one logged, Benjamini–Hochberg
  {mt.get("n_rejected")} rejections). The final test
  ({html.escape(str(parts.get("final_test", ["", ""])[0])[:16])} to
  {html.escape(str(parts.get("final_test", ["", ""])[1])[:16])} UTC) was unlocked once.</p>
  <h3>Realized net P&amp;L by latency and cost stress ($, final test)</h3>
  {_grid_table(kal, "net_pnl_table")}
  <h3>Fee-adjusted 60-second mark-outs ($)</h3>
  {_grid_table(kal, "markout_60s_table")}
  <h3>Expected net ¢ per contract under the true model</h3>
  {_ex_ante_table(kal)}
  <p class="memo">Position limits apply throughout; the daily loss stop is off in the edge study
  so one bad hour cannot silence the rest of the sample. {html.escape(risk_overlay_sentence(kal))}</p>
  <p class="memo">Family-level bootstrap 95% interval for mean net P&amp;L per hourly ladder:
  <b>[{_fmt(ci.get("lower"), 2, True)}, {_fmt(ci.get("upper"), 2, True)}]</b> around
  {_fmt(ci.get("mean"), 2, True)} (n = {ci.get("n")} ladders). Expected edge at the declared
  scenario: <b class="{_cls(ex.get("expected_c_per_contract"))}">{_fmt(ex.get("expected_c_per_contract"), 2, True)}¢</b>
  per contract.</p>
  <h3>Why the gates said {html.escape(str(kal.get("decision")).replace("_", " "))}</h3>
  <ul class="plain">{reasons}</ul>
</section>"""
    poly_html = ""
    if poly:
        poly_html = f"""
<section id="venue">
  <h2>Polymarket fee and 150 ms taker delay</h2>
  <p class="memo">Same market dynamics, priced with Polymarket's crypto taker fee and its
  150 ms delay on marketable orders. Decision: <b>{html.escape(str(poly["decision"]).replace("_", " "))}</b>.</p>
  <h3>Expected net ¢ per contract under the true model</h3>
  {_ex_ante_table(poly)}
</section>"""

    hurdle_rows = []
    for r in result.get("hurdle", []):
        if r.get("window_s") != 0.35 or not str(r.get("venue_fee", "")).startswith("kalshi"):
            continue
        if r.get("moneyness_z") not in (0.0, 1.0):
            continue
        hurdle_rows.append(
            f"<tr><td>{_fmt(r['t_seconds'] / 60, 0)} min · z={_fmt(r['moneyness_z'], 1)}</td>"
            f"<td>{_fmt(r['p0'] * 100, 0)}¢</td><td>{_fmt(r['fee_per_contract'] * 100, 2)}¢</td>"
            f"<td>{_fmt(r['delta_per_bp'], 2)}¢</td><td>{_fmt(r['required_move_bps'], 1)}</td>"
            f"<td>{r['p_move_gauss']:.1e}</td><td>{r['p_move_fat']:.1e}</td>"
            f"<td>{_fmt(r['opportunities_per_hour_fat'], 1)}</td></tr>"
        )
    hurdle_html = f"""
<section id="hurdle">
  <h2>How big a BTC move pays for the fee</h2>
  <p class="memo">A maker quoting one tick around fair value has not reacted yet. Buying the
  stale ask only clears the fee plus a 1¢ margin if BTC moved at least
  <span class="formula">r* = σ√T · (Φ⁻¹(p*) − Φ⁻¹(p₀))</span>. The table shows that move and
  how often it happens inside a 350 ms reaction window (σ = 45%, Gaussian vs fat-tailed t₃);
  chances per hour are a rate per hour spent at that time to expiry, before competition.</p>
  <div class="tablebox"><table><thead><tr><th>time to expiry · moneyness</th><th>fair</th>
  <th>fee</th><th>¢ per bp</th><th>move (bp)</th><th>P(Gauss)</th><th>P(t₃)</th>
  <th>chances / h</th></tr></thead><tbody>{"".join(hurdle_rows)}</tbody></table></div>
  <p class="memo">Far from expiry the required move is several basis points inside a third of
  a second, which essentially only fat tails deliver. Near expiry delta grows and the
  hurdle falls below 1 bp, which is exactly where the fastest bots concentrate.</p>
</section>"""

    ll = result.get("lead_lag", {})
    ll_rows = []
    for r in ll.get("results", []):
        o = r["overall"]
        fails = [x for x in o.get("reasons", []) if str(x).startswith("FAIL")]
        ll_rows.append(
            f"<tr><td>{html.escape(r['contract'].split(':')[-1])}</td>"
            f"<td>{o.get('best_positive_lag_ms')}</td><td>{o.get('hy_best_lag_ms')}</td>"
            f"<td>{_fmt(o.get('p_value'), 3)}</td><td>{_fmt(o.get('incremental_oos_r2'), 3)}</td>"
            f"<td>{_fmt(o.get('hit_rate'), 2)}</td>"
            f"<td>{'yes' if o.get('qualifies') else 'no'}</td>"
            f"<td style='text-align:left;white-space:normal;min-width:16ch'>"
            f"{html.escape(fails[0][:110]) if fails else ''}</td></tr>"
        )
    st = result.get("structural", {})
    results_ll = ll.get("results", [])
    n_ll = len(results_ll)
    n_lead = sum(
        1
        for r in results_ll
        if any(str(x).startswith("PASS significance") for x in r["overall"].get("reasons", []))
        and (_num(r["overall"].get("incremental_oos_r2")) or 0.0) > 0
    )
    n_qual = sum(1 for r in results_ll if r["overall"].get("qualifies"))
    ll_head = (
        "Lead-lag is real, the money is not"
        if n_lead and not n_qual
        else "Lead-lag discovery on contract mids"
    )
    ll_html = f"""
<section id="leadlag">
  <h2>{ll_head}</h2>
  <p class="memo">In {n_lead} of {n_ll} contracts the discovery engine finds the reference
  market leading contract mids by a few hundred milliseconds, significant under a
  circular-shift null and predictive out of sample, consistent with the
  {ll.get("true_maker_lag_ms", 350):g} ms maker lag built into the data. {n_qual} of {n_ll}
  pass the economic gate: predicted moves rarely exceed the taker fee plus half the spread.</p>
  <div class="tablebox"><table><thead><tr><th>contract</th><th>CCF lag ms</th>
  <th>HY lag ms</th><th>p</th><th>ΔOOS R²</th><th>hit rate</th><th>qualifies</th>
  <th>first failing gate</th></tr></thead><tbody>{"".join(ll_rows)}</tbody></table></div>
  <p class="memo">Structural check: {st.get("family_samples", 0):,} snapshots of strike
  ladders, {st.get("violations_after_costs", 0)} monotonicity violations left after both
  taker fees.</p>
</section>"""

    if live:
        med = _num(((live_primary(live) or {}).get("reaction_ms") or {}).get("median"))
        next_items = (
            f"<li>Real makers repriced half of a ≥ 5 bps move within a median {_fmt(med, 0)} ms "
            f"in this window. The model needs about {html.escape(need)} or longer for edge with "
            "a 120 ms competitor present; a longer capture shows whether slower periods (news, "
            "thin hours, expiry days) exist.</li>"
            "<li>Extend the capture to ≥ 14 days with <code>scripts/live_study.py collect</code> "
            "across volatility regimes, and add Polymarket and Binance / Deribit feeds. Promote "
            "only through the out-of-sample and promotion gates, never from one window.</li>"
        )
        hours = _num(live.get("window", {}).get("hours")) or 0.0
        live_foot = (
            f" Live numbers come from one {hours:.1f} h window and are receive-time estimates, "
            "not P&amp;L."
        )
    else:
        next_items = (
            "<li>Measure the real maker reaction-lag distribution and competitor fill speed on "
            "Kalshi BTC ladders from collected books. In the model, edge appears only once "
            f"makers take about {html.escape(need)} or longer, with a 120 ms competitor "
            "present.</li><li>Collect ≥ 14 days with <code>cma collect</code> once the venue "
            "hosts are reachable, approve mappings, then run <code>cma stress</code> and "
            "<code>cma leadlag</code>. Publication is refused unless every latency (0–5 s) and "
            "≥ 2 adverse cost scenarios are present.</li>"
        )
        live_foot = ""

    ev_html = "".join(
        f'<div class="ev"><span class="cred">{html.escape(c)}</span><div><p>{html.escape(t)}</p>'
        f'<p class="src">{html.escape(s)}</p></div></div>'
        for t, s, c in EVIDENCE
    )

    return f"""<title>BTC Lead-Lag Edge Study</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>{STYLE}</style>
<div class="wrap">
<header class="top">
  <span class="eyebrow">Cross-market prediction trading · edge evaluation · {html.escape(str(man.get("experiment_id", "")))}</span>
  <h1>Can a faster BTC feed beat Kalshi and Polymarket quotes after fees?</h1>
  <p class="lede">{html.escape(lede)}</p>
</header>
<section id="decision">
  <h2>Decision</h2>
  <div class="verdicts">{verdict_html}</div>
  {tiles}
</section>
{_live_section(live) if live else ""}
{_frontier_section(frontier)}
{_attribution_section(result, crit)}
{base_html}
{poly_html}
{hurdle_html}
{ll_html}
<section id="evidence">
  <h2>Outside evidence used for calibration</h2>
  <div class="evidence">{ev_html}</div>
</section>
<section id="next">
  <h2>What would change the answer</h2>
  <ul class="plain memo">
    {next_items}
    <li>Look beyond pure latency taking: passive quoting, which pays far lower fees than
    taking on both venues, and near-expiry contracts where delta is high, are the remaining
    places the economics can work.</li>
  </ul>
  <p class="memo" style="color:var(--ink-3);font-size:13px">git {html.escape(str(man.get("git_commit", ""))[:12])} ·
  config {html.escape(str(man.get("config_hash", ""))[:12])} · dataset
  {html.escape(str(man.get("dataset_hash", ""))[:12])} · seed {html.escape(str(man.get("seed", "")))}.
  Synthetic results bound what is plausible; they are not estimates of real P&amp;L.{live_foot}</p>
</section>
</div>
<div class="tip" id="tip" hidden></div>
<script>{SCRIPT}</script>
"""


def write_html(
    result: Mapping[str, Any],
    out_dir: Path,
    *,
    fragment: bool = False,
    live: Mapping[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = render_fragment(result, live)
    if fragment:
        path = out_dir / "edge_evaluation.fragment.html"
        path.write_text(body, encoding="utf-8")
        return path
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, '
        'viewport-fit=cover"></head><body>' + body + "</body></html>"
    )
    path = out_dir / "edge_evaluation.html"
    path.write_text(doc, encoding="utf-8")
    return path


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
