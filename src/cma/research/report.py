"""Decision report rendering (scope s.30) from an ``evaluation.json`` result.

Produces ``DECISION_REPORT.md`` (human-readable) and ``decision.json`` (machine-readable
summary). Every number is taken from the evaluation result; nothing is re-computed here.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

LATENCIES = (0, 100, 250, 500, 1000, 2000, 5000)


def _f(x: Any, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    return f"{v:,.{nd}f}"


def _tidy(text: str) -> str:
    """Round over-long floats in stored reason strings (display only)."""
    return re.sub(r"-?\d+\.\d{5,}", lambda m: f"{float(m.group()):.2f}", text)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def frontier_table(rows: Sequence[Mapping[str, Any]], model: str, field: str, nd: int = 2) -> str:
    scen: dict[tuple[float, Any], dict[int, Any]] = {}
    for r in rows:
        if r.get("model", "implied_vol") != model:
            continue
        scen.setdefault((r["mm_lag_ms"], r["competitor_ms"]), {})[int(r["latency_ms"])] = r[field]
    body = []
    for (mm, comp), by_lat in sorted(scen.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        comp_s = "none" if comp is None else f"{comp:g} ms"
        body.append([f"{mm:g} ms", comp_s, *(_f(by_lat.get(lat), nd) for lat in LATENCIES)])
    return _table(["maker lag", "competitor", *(f"{lat} ms" for lat in LATENCIES)], body)


def _grid_rows(base: Mapping[str, Any], field: str, nd: int = 2) -> str:
    table = base.get(field) or {}
    rows = []
    for cost, by_lat in table.items():
        rows.append([cost, *(_f(by_lat.get(str(lat), by_lat.get(lat)), nd) for lat in LATENCIES)])
    return _table(["cost scenario", *(f"{lat} ms" for lat in LATENCIES)], rows)


def _base_cell(base: Mapping[str, Any], cost: str, lat: int) -> Mapping[str, Any] | None:
    for r in base.get("final_test_grid", []):
        if r["cost"] == cost and int(r["latency_ms"]) == lat:
            cell: Mapping[str, Any] = r
            return cell
    return None


def positive_maker_lags(
    frontier: Sequence[Mapping[str, Any]], latency_ms: int, competitor: Any
) -> list[float]:
    """Maker lags whose implied-vol expected edge is > 0 at ``latency_ms``.

    ``competitor``: None = scenarios without a competitor; "any" = with one.
    """
    out = set()
    for r in frontier:
        if r.get("model", "implied_vol") != "implied_vol" or int(r["latency_ms"]) != latency_ms:
            continue
        has_comp = r["competitor_ms"] is not None
        if (competitor is None) == has_comp:
            continue
        v = r.get("expected_c_per_contract")
        if v is not None and math.isfinite(float(v)) and float(v) > 0:
            out.add(float(r["mm_lag_ms"]))
    return sorted(out)


def _attribution_sources(
    result: Mapping[str, Any], latency_ms: int
) -> list[tuple[str, Mapping[str, Any], tuple[str, ...]]]:
    """Base cases (both tables) and frontier scenarios with a competitor (edge buckets)."""
    both = ("by_perceived_edge", "by_time_to_expiry")
    out: list[tuple[str, Mapping[str, Any], tuple[str, ...]]] = []
    for b in result.get("base_cases", []):
        attr = (b.get("edge_attribution") or {}).get(f"base@{latency_ms}")
        if attr:
            out.append((f"Base case `{b['label']}` (final test, {latency_ms} ms)", attr, both))
    rows = [
        r
        for r in result.get("frontier", [])
        if r.get("model", "implied_vol") == "implied_vol"
        and r.get("attribution")
        and r["competitor_ms"] is not None
    ]
    for r in sorted(rows, key=lambda r: float(r["mm_lag_ms"])):
        out.append(
            (
                f"Frontier: maker lag {r['mm_lag_ms']:g} ms, competitor "
                f"{r['competitor_ms']:g} ms, {r['latency_ms']} ms",
                r["attribution"],
                ("by_perceived_edge",),
            )
        )
    return out


def risk_overlay_sentence(base: Mapping[str, Any]) -> str:
    ov = base.get("production_risk_overlay") or {}
    if not ov:
        return ""
    stops = ov.get("daily_stop_activations") or ov.get("daily_stops") or []
    head = (
        f"With the production {ov.get('daily_loss_stop_pct', 2):g}% daily loss stop on, the "
        f"{ov.get('latency_ms')} ms base run"
    )
    if not stops:
        return f"{head} never hit the stop (net P&L {_f(ov.get('net_pnl'))})."
    denied = ov.get("orders_denied_by_stop")
    return (
        f"{head} hit the stop at {str(stops[0])[11:16]} UTC and traded "
        f"{ov.get('families_traded')} of {ov.get('families')} hourly ladders "
        f"(net P&L {_f(ov.get('net_pnl'))}"
        + (f"; {denied:,} orders refused" if denied else "")
        + ")."
    )


def _n(x: Any) -> str:
    return f"{int(x):,}" if isinstance(x, int | float) and math.isfinite(x) else "n/a"


def live_primary(live: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Pooled (all series) row of the live study at its primary move threshold."""
    from cma.research.live_study import PRIMARY_THRESHOLD_BPS

    return next(
        (r for r in live.get("pooled", []) if float(r["threshold_bps"]) == PRIMARY_THRESHOLD_BPS),
        None,
    )


def live_cell(row: Mapping[str, Any] | None, lat: int | str) -> Mapping[str, Any]:
    cell: Mapping[str, Any] = (row or {}).get("anchored_edge_by_latency", {}).get(str(lat), {})
    return cell


def edge_band(cell: Mapping[str, Any]) -> str:
    """``+1.23 ± 0.40`` (mean ± 2 move-clustered SE, cents)."""
    mean, se = cell.get("mean_c"), cell.get("se_c")
    if mean is None:
        return "n/a"
    return f"{mean:+.2f}" + (f" ± {2 * se:.2f}" if se is not None else "")


def live_markdown(live: Mapping[str, Any]) -> list[str]:
    """Condensed real-market section (full by-series tables live in reports/live_study)."""
    win = live.get("window", {})
    ref = live.get("reference", {})
    con = live.get("contracts", {})
    lats = [str(x) for x in live.get("config", {}).get("latencies_ms", [0, 100, 250, 500, 1000])]
    dec = live.get("decision", {})
    by_series = ", ".join(
        f"{k} {_n(v)}" for k, v in (con.get("quote_updates_by_series") or {}).items()
    )
    moves = " / ".join(_n(v) for v in (live.get("moves_by_threshold") or {}).values())
    out = ["## Real-market check: live Kalshi order books", ""]
    out.append(
        f"* Window {win.get('start')} to {win.get('end')} ({_f(win.get('hours'), 2)} h): "
        f"Kalshi authenticated WebSocket books for {_n(con.get('with_quotes'))} of "
        f"{_n(con.get('above_strike'))} BTC above-strike contracts (top-of-book changes: "
        f"{by_series or 'n/a'}), Coinbase BTC-USD ticker ({_n(ref.get('updates'))} updates), "
        f"{_n(live.get('raw_messages'))} raw messages in all."
    )
    dvol = ref.get("dvol_mean")
    out.append(
        "* Volatility input: "
        + (
            f"Deribit DVOL, mean {_f(100 * dvol, 0)}% ({_n(ref.get('dvol_points'))} one-minute "
            "closes)"
            if dvol is not None
            else "realised volatility (DVOL unavailable)"
        )
        + f"; realised over the window {_f(100 * (ref.get('realized_vol') or math.nan), 0)}%."
    )
    out.append(f"* Coinbase moves of at least 3 / 5 / 10 bps within 1 s (5 s cooldown): {moves}.")
    bq = (live.get("book_quality") or {}).get("KALSHI") or {}
    if bq:
        out.append(
            f"* Kalshi book replay: {_n(bq.get('snapshots'))} snapshots ({_n(bq.get('resets'))} "
            f"after reconnects), {_n(bq.get('deltas_applied'))} deltas, {_n(bq.get('crossed'))} "
            f"momentarily crossed states skipped, {_n(bq.get('gaps'))} sequence gaps; "
            f"{_n(bq.get('books_invalid_at_end'))} of {_n(bq.get('books'))} books invalid at "
            "the end."
        )
    out.append(
        "* Method: for every move and every contract whose fair value it shifts by at least 1¢ "
        "(fair 10–90¢, 90 s to 6 h before close), take the quote a taker would hit as observed "
        "L ms after the move (receive time on this machine), value it at Kalshi's own mid 1 s "
        "before the move plus the model's change in fair value, and subtract the price and the "
        "Kalshi taker fee (10 contracts, rounded up per order)."
    )
    out.append("")
    rows = []
    for r in live.get("pooled", []):
        rt = r.get("reaction_ms") or {}
        rows.append(
            [
                f"{r['threshold_bps']:g}",
                _n(r["samples"]),
                _n(r["moves"]),
                f"{_f(rt.get('p25'), 0)} / {_f(rt.get('median'), 0)} / {_f(rt.get('p75'), 0)}",
                _f(rt.get("share_already_moved"), 2),
                _f(r["lifetime_ms"].get("median"), 0),
                *(edge_band(live_cell(r, x)) for x in lats),
            ]
        )
    out.append(
        _table(
            [
                "move ≥ bps",
                "samples",
                "moves",
                "maker reaction p25 / median / p75 ms",
                "repriced before seen",
                "quote life median ms",
                *(f"{x} ms" for x in lats),
            ],
            rows,
        )
    )
    out.append("")
    out.append(
        "Maker reaction: time from seeing the move until Kalshi's mid covered half the model's "
        "predicted repricing (0 if it already had). Quote life: time until the quote a taker "
        "would hit changed for any reason. Latency cells: mean net ¢ per contract after the "
        "fee ± 2 standard errors clustered by move, all series pooled."
    )
    out.append("")
    b, babs = live.get("baseline_anchored", {}), live.get("baseline", {})
    out.append(
        f"* No-news baseline: the same valuation at fixed 10 s times nets "
        f"{_f(b.get('mean_c'))}¢ (positive {_f(100 * (b.get('share_positive') or 0), 0)}% of "
        "the time), the cost of crossing half the spread plus the fee without a signal."
    )
    out.append(
        f"* Valued at the model price alone (no market anchor) the baseline is "
        f"{_f(babs.get('mean_c'))}¢, positive {_f(100 * (babs.get('share_positive') or 0), 0)}% "
        "of the time: level disagreement between model and market, not latency edge."
    )
    for r in live.get("lead_lag", []):
        o = r["result"]
        out.append(
            f"* Lead-lag {r['contract'].split(':')[-1]} ({_n(r['updates'])} mid changes): "
            f"Coinbase leads by {o.get('best_positive_lag_ms')} ms, p {_f(o.get('p_value'), 3)}, "
            f"ΔOOS R² {_f(o.get('incremental_oos_r2'), 3)}, economic gate "
            f"{'passed' if o.get('qualifies') else 'failed'}."
        )
    reasons = dec.get("reasons", [])
    out.append(
        f"* Decision `{dec.get('decision')}`: {_tidy(reasons[0]) if reasons else 'n/a'}. "
        f"Rule fixed before the run: {dec.get('rule')}"
    )
    out.append("* By-series tables and the model-absolute view: `reports/live_study/summary.md`.")
    out.append("")
    return out


def decision_summary(
    result: Mapping[str, Any], live: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    bases = result.get("base_cases", [])
    dec = (live or {}).get("decision") or {}
    out: dict[str, Any] = {
        "real_market_decision": dec.get(
            "decision", result.get("real_market_decision", "COLLECT_MORE_DATA")
        ),
        "synthetic_decisions": {b["label"]: b["decision"] for b in bases},
        "synthetic_reasons": {b["label"]: b["decision_reasons"] for b in bases},
        "manifest": {
            k: result.get("manifest", {}).get(k)
            for k in ("experiment_id", "git_commit", "config_hash", "dataset_hash", "seed")
        },
    }
    if dec:
        out["real_market_reasons"] = dec.get("reasons", [])
        out["real_market_rule"] = dec.get("rule")
        out["real_market_window"] = (live or {}).get("window")
    return out


def render_markdown(result: Mapping[str, Any], live: Mapping[str, Any] | None = None) -> str:
    man = result.get("manifest", {})
    plan = result.get("plan", {})
    bases = result.get("base_cases", [])
    kal = next((b for b in bases if b["label"] == "kalshi_fee"), bases[0] if bases else {})
    poly = next((b for b in bases if b["label"] != kal.get("label")), None)
    frontier = result.get("frontier", [])
    ll = result.get("lead_lag", {})
    st = result.get("structural", {})
    hurdle = result.get("hurdle", [])
    crit_lat = int(plan.get("criterion_latency_ms", 250))
    base_cell = _base_cell(kal, "base", crit_lat) if kal else None
    bm = (base_cell or {}).get("metrics", {})
    ex = (kal.get("ex_ante") or {}).get(f"base@{crit_lat}", {}) if kal else {}

    lines: list[str] = []
    w = lines.append
    w("# Cross-market edge evaluation — decision report")
    w("")
    w(
        f"Experiment `{man.get('experiment_id')}` · git `{man.get('git_commit')}` · config "
        f"`{str(man.get('config_hash'))[:12]}` · dataset `{str(man.get('dataset_hash'))[:12]}` · "
        f"seed {man.get('seed')}"
    )
    w("")
    w("## Decision")
    w("")
    live_dec = (live or {}).get("decision") or {}
    if live_dec:
        reasons = live_dec.get("reasons", [])
        w(
            f"* **Real markets (Kalshi BTC order books, "
            f"{_f((live or {}).get('window', {}).get('hours'), 1)} h live): "
            f"`{live_dec.get('decision')}`** for the stale-quote taker: "
            f"{_tidy(reasons[0]) if reasons else 'n/a'}. See *Real-market check* below."
        )
    else:
        w(
            f"* **Real markets (Kalshi / Polymarket BTC contracts): "
            f"`{result.get('real_market_decision', 'COLLECT_MORE_DATA')}`.** No real venue data "
            "could be collected from this environment (market-data hosts are blocked by its "
            "network policy), so no real-market claim is made. The platform is ready to collect "
            "and evaluate as soon as access exists (see *Next steps*)."
        )
    for b in bases:
        w(
            f"* Synthetic base case `{b['label']}`: **`{b['decision']}`** — "
            + "; ".join(_tidy(r) for r in b.get("decision_reasons", [])[:4])
        )
    w("")
    w(
        "The synthetic study answers a narrower question than profitability: *how much quote "
        "staleness, competition and latency can a lead-lag taker afford after real venue fees?* "
        "It uses the full production pipeline (replay → features → signals → risk → latency-"
        "aware execution simulator → accounting) on a calibrated synthetic market with a known "
        "ground truth."
    )
    w("")

    if live:
        lines += live_markdown(live)
    w("## 1. Data, instruments and exclusions")
    w("")
    if kal:
        mp = kal.get("market_params", {})
        w(
            f"* Base case: synthetic BTC index (Markov-switching vol 40%/85%, jumps), reference "
            f"exchange mid (Coinbase-like, 5 Hz), hourly ladder of {mp.get('n_strikes', 9)} "
            f'"index > K" contracts settling on the 60 s average before expiry (Kalshi BRTI '
            f"semantics), {mp.get('hours')} hours, seed {mp.get('seed')}."
        )
        w(
            f"* Market makers re-quote one-tick markets from a view delayed by a log-normal lag "
            f"(median {mp.get('mm_lag_ms')} ms); competing arbitrageurs with "
            f"{mp.get('competitor_latency_ms')} ms latency take quotes stale by > fees + 2¢; noise "
            "traders hit the touch at random."
        )
        w(
            f"* Fee schedule: `{mp.get('fee_schedule_id', 'kalshi-standard')}` (variant: "
            "Polymarket crypto taker fee + 150 ms taker speed bump)."
        )
    w(
        "* Data-quality exclusions: none (synthetic data is gap-free by construction; the "
        "gap/duplicate/crossed-book machinery is exercised by the replay fixtures and tests)."
    )
    w("")

    w("## 2. Strategy, model and mapping versions")
    w("")
    w(
        "* Strategy `fv_taker` v1.0: fair YES probability from the observed reference mid with "
        "settlement-exact semantics (point vs 60 s trailing average), volatility from an "
        "implied-vol index feed; IOC limit orders at the price where marginal net edge still "
        "clears the threshold; positions held to settlement (fees paid once)."
    )
    if kal:
        w(f"* Selected parameters: `{json.dumps(kal.get('selected_params'))}`.")
    w(
        "* Mappings: synthetic contracts with exact semantics, status `REVIEWED` (never "
        "`APPROVED_PAPER`, so no synthetic result can start forward paper trading)."
    )
    w("")

    w("## 3. Train / validation / final-test segmentation")
    w("")
    if kal:
        parts = kal.get("partitions", {})
        w(_table(["partition", "start", "end"], [[k, v[0], v[1]] for k, v in parts.items()]))
        w("")
        mt = kal.get("multiple_testing", {})
        w(
            f"* Parameter search on **validation only**: {mt.get('n_hypotheses')} settings "
            f"(min net edge × vol multiplier), each logged as a hypothesis "
            f"({mt.get('test', 'one-sided t-test')}); Benjamini–Hochberg at "
            f"α={mt.get('alpha')}: {mt.get('n_rejected')} rejections."
        )
        for line in kal.get("final_test_access_log", []):
            w(f"* Final-test access log: {line}")
    w("")

    w(f"## 4. Final-test results (base cost, {crit_lat} ms)")
    w("")
    if bm:
        rows = [
            ["realized net P&L ($)", _f(bm.get("net_pnl"))],
            ["realized gross P&L ($)", _f(bm.get("gross_pnl"))],
            ["fees ($)", _f(bm.get("fees"))],
            ["slippage vs signal price ($)", _f(bm.get("slippage_cost"))],
            ["ex-ante expected net P&L ($, true model)", _f(ex.get("expected_net_pnl"))],
            ["ex-ante expected net ¢ / contract", _f(ex.get("expected_c_per_contract"))],
            [
                "fee-adjusted 60 s mark-out P&L ($)",
                _f((bm.get("markout_net_pnl") or {}).get("60000ms")),
            ],
            [
                "signals / orders / fills",
                f"{bm.get('n_signals')} / {bm.get('n_orders')} / {bm.get('n_fills')}",
            ],
            ["contracts filled", _f(bm.get("filled_contracts"), 0)],
            ["fill rate", _f(bm.get("fill_rate"), 3)],
            ["positions (contracts traded)", bm.get("n_positions")],
            ["hit rate (positions)", _f(bm.get("hit_rate"), 3)],
            ["profit factor", _f(bm.get("profit_factor"), 2)],
            ["max drawdown ($)", _f(bm.get("max_drawdown"))],
            ["CVaR 5% per position ($)", _f(bm.get("tail_loss_cvar5"))],
            [
                "mean / median signal net edge (bps)",
                f"{_f(bm.get('avg_net_edge_bps'), 1)} / {_f(bm.get('median_net_edge_bps'), 1)}",
            ],
        ]
        w(_table(["metric", "value"], rows))
        ci = kal.get("ci_mean_position_pnl", {})
        w("")
        w(
            "Risk limits: position limits (contract / family / portfolio exposure) apply; the "
            "NAV-triggered daily loss stop is disabled in the edge study so one bad hour cannot "
            "silence the rest of the sample. " + risk_overlay_sentence(kal)
        )
        w("")
        w(
            f"Bootstrap 95% CI of mean net P&L per {ci.get('unit', 'unit')}, n = {ci.get('n')}: "
            f"[{_f(ci.get('lower'))}, {_f(ci.get('upper'))}] around {_f(ci.get('mean'))}. "
            "Hold-to-settlement P&L of one hourly ladder is one correlated bet on BTC, so the "
            "family — not the contract — is the unit of independent evidence."
        )
    w("")

    w("## 5. Results by latency and cost / slippage stress (final test)")
    w("")
    if kal:
        w("Realized net P&L ($):")
        w("")
        w(_grid_rows(kal, "net_pnl_table"))
        w("")
        w(
            "Realized hold-to-settlement P&L is dominated by where BTC finished each hour, and "
            "each cost scenario trades a different subset of opportunities, so this table need "
            "not be monotone in costs. The mark-out and ex-ante tables below measure edge."
        )
        w("")
        w(
            "Fee-adjusted 60 s mark-out P&L ($) — low-variance edge estimate also available on "
            "real data:"
        )
        w("")
        w(_grid_rows(kal, "markout_60s_table"))
        w("")
        exa = kal.get("ex_ante", {})
        w(
            "Ex-ante expected net ¢ per contract under the true model (simulation-only diagnostic "
            "that removes settlement luck):"
        )
        w("")
        rows = []
        for cost in dict.fromkeys(k.split("@")[0] for k in exa):
            rows.append(
                [
                    cost,
                    *(
                        _f((exa.get(f"{cost}@{lat}") or {}).get("expected_c_per_contract"))
                        for lat in LATENCIES
                    ),
                ]
            )
        w(_table(["cost scenario", *(f"{lat} ms" for lat in LATENCIES)], rows))
        w("")
        mr = kal.get("model_risk", [])
        if mr:
            w(
                "Model risk — same final test, implied-vol vs realized-vol fair value "
                "(expected ¢ / contract):"
            )
            w("")
            w(
                _table(
                    ["latency", "implied vol", "realized vol"],
                    [
                        [
                            f"{r['latency_ms']} ms",
                            _f(r["implied_vol"]["expected_c_per_contract"]),
                            _f(r["realized_vol"]["expected_c_per_contract"]),
                        ]
                        for r in mr
                    ],
                )
            )
            w("")
    if poly:
        w(
            f"Venue variant `{poly['label']}` (Polymarket crypto fee 0.07·p(1−p), 150 ms taker "
            f"delay) — decision `{poly['decision']}`; realized net P&L ($):"
        )
        w("")
        w(_grid_rows(poly, "net_pnl_table"))
        w("")

    w("## 6. Where the edge goes (simulation-only attribution)")
    w("")
    w(
        "Each fill's net edge as the signal saw it (model fair value minus price, fees and "
        "buffers) versus its true net edge at fill time (true fair value minus price minus "
        "fee). When the perceived edge systematically exceeds the true edge, the strategy is "
        "adversely selected: it trades most when its own inputs (stale spot, vol) are wrong."
    )
    w("")
    for title, attr, keys in _attribution_sources(result, crit_lat):
        w(f"**{title}**")
        w("")
        for key, label in (
            ("by_perceived_edge", "perceived net edge"),
            ("by_time_to_expiry", "time to expiry"),
        ):
            if key not in keys:
                continue
            rows = [
                [
                    r["bucket"],
                    _f(r["contracts"], 0),
                    _f(r["perceived_c_per_contract"]),
                    _f(r["true_c_per_contract"]),
                ]
                for r in attr.get(key, [])
            ]
            if rows:
                w(_table([label, "contracts", "perceived ¢", "true ¢"], rows))
                w("")
    w("## 7. Lead-lag estimates and stability by regime")
    w("")
    w(
        f"Reference mid → contract mid, true maker lag {ll.get('true_maker_lag_ms')} ms "
        "(median) plus feed delays. Lags in ms; positive = reference leads."
    )
    w("")
    rows = []
    for r in ll.get("results", []):
        o = r["overall"]
        rows.append(
            [
                r["contract"].split(":")[-1],
                o.get("best_positive_lag_ms"),
                o.get("hy_best_lag_ms"),
                _f(o.get("corr_at_best"), 3),
                _f(o.get("corr_at_zero"), 3),
                _f(o.get("p_value"), 3),
                _f(o.get("incremental_oos_r2"), 3),
                _f(o.get("hit_rate"), 3),
                _f(o.get("economic_edge_estimate"), 4),
                o.get("qualifies"),
            ]
        )
    w(
        _table(
            [
                "contract",
                "CCF lag",
                "HY lag",
                "corr@lag",
                "corr@0",
                "p",
                "ΔOOS R²",
                "hit rate",
                "econ edge",
                "qualifies",
            ],
            rows,
        )
    )
    w("")
    for r in ll.get("results", [])[:2]:
        w(
            f"* `{r['contract'].split(':')[-1]}` by time to expiry: "
            + "; ".join(
                f"{k}: lag {v.get('best_positive_lag_ms')} ms, "
                f"ΔOOS R² {_f(v.get('incremental_oos_r2'), 3)}, qualifies {v.get('qualifies')}"
                for k, v in (r.get("by_time_to_expiry") or {}).items()
                if isinstance(v, dict)
            )
        )
        fails = [x for x in r["overall"].get("reasons", []) if x.startswith("FAIL")]
        if fails:
            w(f"  * Failing gate: {fails[0]}")
    w("")
    w(
        "A statistically real lead (significant, out-of-sample predictive) is necessary but not "
        "sufficient: the economic gate requires predicted moves larger than fees + half-spread."
    )
    w("")

    w("## 8. Profit / loss concentration")
    w("")
    if bm:
        c = bm.get("concentration", {})
        w(
            _table(
                ["diagnostic", "value"],
                [
                    ["top contract share of gains", _f(c.get("top_contract_share_of_gains"), 3)],
                    ["top family share of gains", _f(c.get("top_family_share_of_gains"), 3)],
                    ["top day share of gains", _f(c.get("top_day_share_of_gains"), 3)],
                    ["families / days", f"{c.get('n_families')} / {c.get('n_days')}"],
                ],
            )
        )
    w("")

    w("## 9. Structural (nested-strike) consistency")
    w("")
    w(
        f"{st.get('family_samples')} family snapshots scanned; {st.get('violations_after_costs')} "
        f"executable violations after both taker fees (net total {st.get('net_total')}). "
        f"{st.get('note', '')}"
    )
    w("")

    w("## 10. Edge frontier (simulation)")
    w("")
    w(
        "Ex-ante expected net ¢ per contract (true-model value minus price minus fees), "
        "implied-vol model, base fees, by maker reaction lag, competitor latency and *our* "
        "outbound latency (0 ms is diagnostic only):"
    )
    w("")
    w(frontier_table(frontier, "implied_vol", "expected_c_per_contract"))
    w("")
    for comp_label, comp in (
        ("no competing arbitrageur", None),
        ("a competing arbitrageur", "any"),
    ):
        lags = positive_maker_lags(frontier, crit_lat, comp)
        grid = sorted({float(r["mm_lag_ms"]) for r in frontier})
        w(
            f"* At {crit_lat} ms with {comp_label}: expected edge is positive only for maker lags "
            f"{', '.join(f'{x:g} ms' for x in lags) if lags else 'none'} "
            f"(grid: {', '.join(f'{x:g} ms' for x in grid)})."
        )
    w("")
    w("Contracts filled (same grid):")
    w("")
    w(frontier_table(frontier, "implied_vol", "contracts", 0))
    w("")
    if any(r.get("model") == "realized_vol" for r in frontier):
        w("Realized-vol model (model-risk comparison), expected ¢ / contract:")
        w("")
        w(frontier_table(frontier, "realized_vol", "expected_c_per_contract"))
        w("")

    w("## 11. Analytical cost hurdle")
    w("")
    w(
        "Move needed (bps of BTC) for a one-tick stale quote to clear the taker fee + 1¢ "
        "net, and the chance it happens inside a 350 ms reaction window (Gaussian vs "
        "variance-matched Student-t ν=3), σ = 45%. Opportunities per hour are a rate per hour "
        "spent at that time to expiry, before any competition:"
    )
    w("")
    rows = []
    for r in hurdle:
        if r["window_s"] != 0.35 or not str(r["venue_fee"]).startswith("kalshi"):
            continue
        rows.append(
            [
                _f(r["t_seconds"], 0),
                _f(r["moneyness_z"], 1),
                _f(r["p0"], 2),
                _f(r["fee_per_contract"] * 100, 2),
                _f(r["delta_per_bp"], 2),
                _f(r["required_move_bps"], 1),
                f"{r['p_move_gauss']:.1e}",
                f"{r['p_move_fat']:.1e}",
                _f(r["opportunities_per_hour_fat"], 1),
            ]
        )
    w(
        _table(
            [
                "T (s)",
                "z",
                "p0",
                "fee ¢",
                "¢ per bp",
                "move bps",
                "P gauss",
                "P fat",
                "opps/h (fat)",
            ],
            rows,
        )
    )
    w("")

    w("## 12. Known limitations and unresolved issues")
    w("")
    hours = _f((live or {}).get("window", {}).get("hours"), 1)
    real_data = (
        f"Real venue data: one {hours} h window of Kalshi BTC books and Coinbase BTC-USD (see "
        "*Real-market check*). One window is one volatility regime and one set of makers; the "
        "model-world numbers in this report bound what is plausible, they do not estimate "
        "real P&L."
        if live
        else "No real venue data was examined (network policy). All profitability numbers are "
        "model-world results; they bound what is plausible, they do not estimate real P&L."
    )
    kalshi_feed = (
        "Kalshi WebSocket data was collected with API credentials on this machine's network "
        "path; a co-located competitor sees and acts on moves sooner than the receive-time "
        "timeline used here."
        if live
        else "Kalshi WebSocket market data requires API credentials; without them the collector "
        "falls back to REST polling, which cannot measure sub-second lead-lag."
    )
    parsers = (
        "Kalshi WebSocket and Coinbase parsers have run on live payloads; Polymarket, Binance "
        "and Deribit parsers are still unverified against live traffic."
        if live
        else "Parsers were written from documented schemas without live payloads; schema drift "
        "is quarantined, not silently accepted, and must be checked on first live collection."
    )
    for item in (
        real_data,
        "Synthetic makers re-quote from a lagged view with one-tick spreads and fixed depth "
        "distributions; real quoting (inventory skew, widening into events, cancels) differs.",
        "Competition is modelled as one arbitrageur class with a single latency and threshold.",
        "The ex-ante 'truth' is the model world's fair value (current regime vol, exact 60 s "
        "averaging); it measures edge against well-informed makers, not against real ones.",
        "Fee formulas reflect official documentation as of 2026-10 (Kalshi 0.07·C·P·(1−P) "
        "rounded up per order; Polymarket crypto 0.07·C·p·(1−p)); maker rebates are not "
        "credited; verify per-series values at runtime.",
        kalshi_feed,
        "Cross-venue BTC contracts settle on different references (Kalshi BRTI 60 s average vs "
        "Polymarket Binance BTCUSDT candles or Chainlink TWAP): they are basis trades, never "
        "riskless equivalents; the mapping registry rejects them as equivalent.",
        parsers,
    ):
        w(f"* {item}")
    w("")
    w("## 13. Next steps to reach a real-market decision")
    w("")
    steps_live = (
        "Extend the live capture to >= 14 days (`scripts/live_study.py collect`) across "
        "volatility regimes, US and Asian hours and expiry days, then re-run "
        "`scripts/live_study.py analyze`; one window is not a sample of regimes.",
        "Add Polymarket BTC markets and Binance / Deribit reference feeds to the same capture "
        "to test the cross-venue and options-led signals on real books.",
        "Only if an edge cell stays positive on the long sample: approve mappings, build a "
        "dataset (`cma dataset build`), run `cma stress` and `cma leadlag`, and require every "
        "promotion gate before forward paper trading.",
        "Look beyond latency taking: passive quoting, which pays far lower fees than taking, "
        "and near-expiry contracts where delta is high.",
    )
    for item in (
        steps_live
        if live
        else (
            "Allow the market-data hosts in the environment network policy (Kalshi external-api, "
            "Polymarket clob/gamma/data-api, Coinbase exchange, Deribit) and add Kalshi API "
            "credentials via environment variables (never config files).",
            "`cma collect --duration 1209600` for ≥ 14 days to capture synchronized books "
            "(`KXBTCD`, Polymarket BTC markets, Coinbase BTC-USD, Deribit DVOL/options).",
            "Review and approve mappings (`config/mappings/registry`), build a dataset "
            "(`cma dataset build`), run `cma stress` and `cma leadlag`; publish only manifests "
            "that pass `validate_publication`.",
            "Measure the real maker reaction-lag distribution and competitor fill speed: the "
            "frontier above says whether any latency budget can be profitable before money is "
            "spent on infrastructure.",
        )
    ):
        w(f"* {item}")
    w("")
    return "\n".join(lines) + "\n"


def write_reports(
    result: Mapping[str, Any], out_dir: Path, *, live: Mapping[str, Any] | None = None
) -> list[Path]:
    """DECISION_REPORT.md, decision.json and the HTML page; ``live`` is a live-study summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "DECISION_REPORT.md"
    md.write_text(render_markdown(result, live), encoding="utf-8")
    js = out_dir / "decision.json"
    js.write_text(json.dumps(decision_summary(result, live), indent=1, default=str) + "\n")
    from cma.research.report_html import write_html

    page = write_html(result, out_dir, live=live)
    return [md, js, page]
