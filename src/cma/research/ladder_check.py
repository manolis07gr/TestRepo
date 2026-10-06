"""Same-venue consistency: Kalshi 15-minute BTC markets vs the hourly ladder (no model).

Method and decision rule: docs/EDGE_EVALUATION_METHOD.md section 8 (fixed before the first
run). At every hour C, the ``KXBTC15M`` market closing at C (YES when the 60 s BRTI average
before C is at least its strike K15) and the ``KXBTCD`` strikes K_lo < K15 <= K_hi (YES when
the same average is above the strike) settle on one number, so the payoffs nest:
YES(K_hi) <= YES(15-minute) <= YES(K_lo). Three pairs buy two contracts whose payoffs add up
to at least $1 in every outcome:

* ``a``: 15-minute YES at its ask + ladder NO on K_hi at 1 - bid(K_hi)
* ``b``: 15-minute NO at 1 - its bid + ladder YES on K_lo at ask(K_lo)
* ``c``: ladder YES on K_lo + ladder NO on K_hi (the ladder itself out of order)

A pair is a riskless profit when it costs less than $1 including the Kalshi taker fee on each
leg. An hour is an opportunity when a pair is profitable at some decision minute and still is
at the next minute's quotes, where the profit is measured. Minute candles carry no depth, so
history can reject the idea or justify a size check on live books, not size a trade.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cma.research.settlement_data import (
    Candle,
    SettledMarket,
    load_candles,
    load_ladder,
    load_markets,
)
from cma.research.settlement_study import fee_c

MINUTE = 60
PAIRS = ("a", "b", "c")
PAIR_LABELS = {
    "a": "15-minute YES + ladder NO above",
    "b": "15-minute NO + ladder YES below",
    "c": "ladder YES below + ladder NO above",
}


@dataclass(frozen=True)
class LadderConfig:
    first_minute: int = 2  # 15-minute candle k ends at open + 60 k; its first minute is empty
    last_minute: int = 13
    last_fill_minute: int = 14
    qty: int = 100
    min_opportunity_share: float = 0.02
    min_median_profit_c: float = 1.0


@dataclass(frozen=True)
class Book:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def _book(bid: float | None, ask: float | None) -> Book | None:
    if bid is None or ask is None or not 0.0 < bid < ask < 1.0:
        return None
    return Book(bid, ask)


def _closes(candles: Sequence[Sequence[Any]]) -> dict[int, Book]:
    """Closing YES book by candle end time (candle fields: end, bid o/h/l/c, ask o/h/l/c)."""
    out: dict[int, Book] = {}
    for c in candles:
        b = _book(c[4], c[8])
        if b is not None:
            out[int(c[0])] = b
    return out


@dataclass(frozen=True)
class Minute:
    k: int
    m15: Book
    lo: Book
    hi: Book


def pair_prices(q: Minute) -> dict[str, tuple[float, float]]:
    """Price paid for each leg of each pair (dollars per contract)."""
    return {
        "a": (q.m15.ask, 1.0 - q.hi.bid),
        "b": (1.0 - q.m15.bid, q.lo.ask),
        "c": (q.lo.ask, 1.0 - q.hi.bid),
    }


def riskless_profit_c(p1: float, p2: float, qty: int) -> float:
    """Cents per pair guaranteed by a pair paying >= $1: 1 - cost - both taker fees."""
    return 100.0 * (1.0 - p1 - p2) - fee_c(p1, qty) - fee_c(p2, qty)


def pair_payoff(pair: str, value: float, k15: float, k_lo: float, k_hi: float) -> int:
    """Dollars paid by a pair once the settlement value is known."""
    yes15 = value >= k15
    yes_lo, yes_hi = value > k_lo, value > k_hi
    if pair == "a":
        return int(yes15) + int(not yes_hi)
    if pair == "b":
        return int(not yes15) + int(yes_lo)
    return int(yes_lo) + int(not yes_hi)


@dataclass(frozen=True)
class Hour:
    close_ts: int
    k15: float
    k_lo: float
    k_hi: float
    spacing: int
    minutes: dict[int, Minute]  # k -> minute with all three books sane
    settlement: float | None
    result15: str


def build_hours(
    markets15: Sequence[SettledMarket],
    candles15: Mapping[str, Sequence[Candle]],
    ladder: Mapping[int, Mapping[str, Any]],
    cfg: LadderConfig,
) -> tuple[list[Hour], dict[str, int]]:
    """Pair every on-the-hour 15-minute market with its ladder strikes, minute by minute."""
    by_open = {m.open_ts: m for m in markets15}
    counts = {"hours_15m_on_the_hour": 0, "ladder_missing": 0, "no_sane_minute": 0}
    hours: list[Hour] = []
    for m in sorted(markets15, key=lambda m: m.close_ts):
        if m.close_ts % 3600 or not m.floor_strike:
            continue
        counts["hours_15m_on_the_hour"] += 1
        row = ladder.get(m.close_ts)
        if not row or "hi" not in row:
            counts["ladder_missing"] += 1
            continue
        b15 = _closes(candles15.get(m.ticker, []))
        blo, bhi = _closes(row["lo"]["candles"]), _closes(row["hi"]["candles"])
        minutes: dict[int, Minute] = {}
        for k in range(cfg.first_minute, cfg.last_fill_minute + 1):
            end = m.open_ts + MINUTE * k
            q15, qlo, qhi = b15.get(end), blo.get(end), bhi.get(end)
            if q15 is not None and qlo is not None and qhi is not None:
                minutes[k] = Minute(k, q15, qlo, qhi)
        if not any(cfg.first_minute <= k <= cfg.last_minute for k in minutes):
            counts["no_sane_minute"] += 1
            continue
        nxt = by_open.get(m.close_ts)
        settlement = m.expiration_value
        if settlement is None and nxt is not None:
            settlement = nxt.floor_strike  # the next market's strike is the same average
        hours.append(
            Hour(
                close_ts=m.close_ts,
                k15=m.floor_strike,
                k_lo=float(row["lo"]["strike"]),
                k_hi=float(row["hi"]["strike"]),
                spacing=int(row["spacing"]),
                minutes=minutes,
                settlement=settlement,
                result15=m.result,
            )
        )
    return hours, counts


@dataclass(frozen=True)
class Opportunity:
    close_ts: int
    month: str
    pair: str
    k: int
    profit_signal_c: float  # at the minute the gap was seen
    profit_fill_c: float  # riskless profit at the next minute's quotes (the trade)
    realised_c: float | None  # with the settlement value, when known


def find_opportunity(h: Hour, cfg: LadderConfig) -> Opportunity | None:
    """The first decision minute with a riskless pair that is still riskless one minute on."""
    for k in range(cfg.first_minute, cfg.last_minute + 1):
        q, q_next = h.minutes.get(k), h.minutes.get(k + 1)
        if q is None or q_next is None:
            continue
        now = {p: riskless_profit_c(*pp, cfg.qty) for p, pp in pair_prices(q).items()}
        fill_prices = pair_prices(q_next)
        nxt = {p: riskless_profit_c(*pp, cfg.qty) for p, pp in fill_prices.items()}
        hits = [p for p in PAIRS if now[p] > 0 and nxt[p] > 0]
        if not hits:
            continue
        pair = max(hits, key=lambda p: nxt[p])
        realised = None
        if h.settlement is not None:
            p1, p2 = fill_prices[pair]
            payoff = pair_payoff(pair, h.settlement, h.k15, h.k_lo, h.k_hi)
            realised = 100.0 * (payoff - p1 - p2) - fee_c(p1, cfg.qty) - fee_c(p2, cfg.qty)
        return Opportunity(
            close_ts=h.close_ts,
            month=datetime.fromtimestamp(h.close_ts, UTC).strftime("%Y-%m"),
            pair=pair,
            k=k,
            profit_signal_c=now[pair],
            profit_fill_c=nxt[pair],
            realised_c=realised,
        )
    return None


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(hours: Sequence[Hour], cfg: LadderConfig) -> dict[str, Any]:
    """Pre-fee crossings, after-fee gaps and the 15-minute mid against the ladder band."""
    minutes = [
        q for h in hours for k, q in h.minutes.items() if cfg.first_minute <= k <= cfg.last_minute
    ]
    crossed = dict.fromkeys(PAIRS, 0)
    positive = dict.fromkeys(PAIRS, 0)
    outside = 0
    for q in minutes:
        for p, (p1, p2) in pair_prices(q).items():
            if p1 + p2 < 1.0:
                crossed[p] += 1
            if riskless_profit_c(p1, p2, cfg.qty) > 0:
                positive[p] += 1
        if not q.hi.mid <= q.m15.mid <= q.lo.mid:
            outside += 1
    best_per_hour = [
        max(
            riskless_profit_c(*pp, cfg.qty)
            for k, q in h.minutes.items()
            if cfg.first_minute <= k <= cfg.last_minute
            for pp in pair_prices(q).values()
        )
        for h in hours
    ]
    n = len(minutes)
    return {
        "minutes": n,
        "pre_fee_crossed_share": {p: crossed[p] / n if n else None for p in PAIRS},
        "after_fee_positive_share": {p: positive[p] / n if n else None for p in PAIRS},
        "mid_outside_ladder_band_share": outside / n if n else None,
        "best_after_fee_gap_per_hour_c": {
            "median": _pct(best_per_hour, 50),
            "p90": _pct(best_per_hour, 90),
            "p99": _pct(best_per_hour, 99),
            "max": max(best_per_hour) if best_per_hour else None,
        },
    }


def ladder_decision(result: Mapping[str, Any], cfg: LadderConfig) -> dict[str, Any]:
    """The pre-registered rule (method section 8.4)."""
    paired = result["data"]["paired_hours"]
    opp = result["opportunities"]
    share = opp["share_of_hours"]
    median = opp["profit_fill_c"]["median"]
    if not paired:
        return {"decision": "REJECT", "reasons": ["no paired hours"]}
    rare = share is None or share < cfg.min_opportunity_share
    small = median is None or median < cfg.min_median_profit_c
    decision = "REJECT" if rare or small else "LIVE_SIZE_CHECK"
    reasons = [
        f"{opp['count']} opportunities in {paired} paired hours "
        f"({100 * (share or 0):.2f}%; the rule needs {100 * cfg.min_opportunity_share:.0f}%)",
        "median riskless profit "
        + ("n/a" if median is None else f"{median:.2f}¢")
        + f" per pair (the rule needs {cfg.min_median_profit_c:g}¢)",
    ]
    return {"decision": decision, "reasons": reasons}


def analyze(
    markets15: Sequence[SettledMarket],
    candles15: Mapping[str, Sequence[Candle]],
    ladder: Mapping[int, Mapping[str, Any]],
    cfg: LadderConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or LadderConfig()
    hours, counts = build_hours(markets15, candles15, ladder, cfg)
    opps = [o for h in hours if (o := find_opportunity(h, cfg)) is not None]
    by_month: dict[str, dict[str, Any]] = defaultdict(lambda: {"paired_hours": 0, "count": 0})
    for h in hours:
        by_month[datetime.fromtimestamp(h.close_ts, UTC).strftime("%Y-%m")]["paired_hours"] += 1
    for o in opps:
        by_month[o.month]["count"] += 1
    fills = [o.profit_fill_c for o in opps]
    realised = [o.realised_c for o in opps if o.realised_c is not None]
    spacing: dict[str, int] = defaultdict(int)
    for h in hours:
        spacing[str(h.spacing)] += 1
    consistent = sum(
        1
        for h in hours
        if h.settlement is not None and (h.settlement >= h.k15) == (h.result15 == "yes")
    )
    result: dict[str, Any] = {
        "config": asdict(cfg),
        "data": {
            **counts,
            "paired_hours": len(hours),
            "first_close": datetime.fromtimestamp(hours[0].close_ts, UTC).isoformat()
            if hours
            else None,
            "last_close": datetime.fromtimestamp(hours[-1].close_ts, UTC).isoformat()
            if hours
            else None,
            "ladder_spacing_hours": dict(spacing),
            "hours_with_settlement_value": sum(1 for h in hours if h.settlement is not None),
            "settlement_agrees_with_15m_result": consistent,
        },
        "opportunities": {
            "count": len(opps),
            "share_of_hours": len(opps) / len(hours) if hours else None,
            "by_pair": {p: sum(1 for o in opps if o.pair == p) for p in PAIRS},
            "profit_fill_c": {
                "median": _pct(fills, 50),
                "mean": sum(fills) / len(fills) if fills else None,
                "p90": _pct(fills, 90),
                "max": max(fills) if fills else None,
            },
            "realised_c": {
                "n": len(realised),
                "mean": sum(realised) / len(realised) if realised else None,
                "min": min(realised) if realised else None,
            },
            "by_month": dict(sorted(by_month.items())),
            "list": [asdict(o) for o in opps[:200]],
        },
        "description": describe(hours, cfg),
    }
    result["decision"] = ladder_decision(result, cfg)
    return result


def analyze_dir(data_dir: Path, cfg: LadderConfig | None = None) -> dict[str, Any]:
    return analyze(
        load_markets(data_dir, "KXBTC15M"),
        load_candles(data_dir, "KXBTC15M"),
        load_ladder(data_dir),
        cfg,
    )


# ----------------------------------------------------------------------------- report


def _share(v: float | None, nd: int = 2) -> str:
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def _c(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}¢"


def render_markdown(result: Mapping[str, Any]) -> str:
    d, o, ds = result["data"], result["opportunities"], result["description"]
    dec = result["decision"]
    w = [
        "# Same-venue consistency: Kalshi 15-minute BTC markets vs the hourly ladder",
        "",
        f"**Decision: {dec['decision']}** (rule fixed before the first run: "
        "docs/EDGE_EVALUATION_METHOD.md section 8).",
        "",
        *[f"* {r}" for r in dec["reasons"]],
        "",
        "## Data",
        "",
        f"* {d['paired_hours']} hours where the 15-minute market closing on the hour and the "
        f"hourly ladder strikes around its strike both have usable minute quotes "
        f"({str(d['first_close'])[:10]} to {str(d['last_close'])[:10]}; "
        f"{d['hours_15m_on_the_hour']} on-the-hour 15-minute markets, ladder strikes not "
        f"found for {d['ladder_missing']}, no usable minute in {d['no_sane_minute']}).",
        "* Ladder strike spacing by hour: "
        + ", ".join(f"${k}: {v}" for k, v in sorted(d["ladder_spacing_hours"].items()))
        + ".",
        f"* Settlement value known for {d['hours_with_settlement_value']} hours; it agrees "
        f"with the 15-minute result in {d['settlement_agrees_with_15m_result']}.",
        "",
        "## Opportunities (riskless after both taker fees, still there a minute later)",
        "",
        f"* {o['count']} of {d['paired_hours']} hours ({_share(o['share_of_hours'])}); by "
        "pair: " + ", ".join(f"{PAIR_LABELS[p]}: {n}" for p, n in o["by_pair"].items()) + ".",
        f"* Riskless profit at the next minute's quotes: median {_c(o['profit_fill_c']['median'])}"
        f", mean {_c(o['profit_fill_c']['mean'])}, 90th percentile "
        f"{_c(o['profit_fill_c']['p90'])}, largest {_c(o['profit_fill_c']['max'])} per pair "
        "(each pair pays at least $1).",
        f"* With the settlement value ({o['realised_c']['n']} of them): mean realised "
        f"{_c(o['realised_c']['mean'])}, worst {_c(o['realised_c']['min'])} per pair.",
        "",
        "| Month | Paired hours | Opportunities |",
        "|---|---:|---:|",
        *[f"| {mo} | {v['paired_hours']} | {v['count']} |" for mo, v in o["by_month"].items()],
        "",
        "## How close the prices come to an arbitrage",
        "",
        f"Over {ds['minutes']} hour-minutes with all three books quoted:",
        "",
        "| Pair | Crossed before fees | Riskless after both fees |",
        "|---|---:|---:|",
        *[
            f"| {PAIR_LABELS[p]} | {_share(ds['pre_fee_crossed_share'][p], 3)} "
            f"| {_share(ds['after_fee_positive_share'][p], 3)} |"
            for p in PAIRS
        ],
        "",
        f"* The 15-minute mid sits outside the ladder's band [mid above, mid below] in "
        f"{_share(ds['mid_outside_ladder_band_share'])} of minutes.",
        f"* Best after-fee result per hour (negative = no arbitrage): median "
        f"{_c(ds['best_after_fee_gap_per_hour_c']['median'])}, 90th percentile "
        f"{_c(ds['best_after_fee_gap_per_hour_c']['p90'])}, 99th "
        f"{_c(ds['best_after_fee_gap_per_hour_c']['p99'])}, best "
        f"{_c(ds['best_after_fee_gap_per_hour_c']['max'])}.",
        "",
        "Minute closing quotes only: no depth, and both legs are assumed to fill at the quoted "
        "price for 100 contracts.",
        "",
    ]
    return "\n".join(w)


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / "summary.json", out_dir / "summary.md"
    js.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    md.write_text(render_markdown(result), encoding="utf-8")
    return [js, md]
