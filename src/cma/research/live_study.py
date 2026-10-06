"""Real-market staleness study: do Kalshi BTC quotes lag a fast reference enough to pay fees?

Works directly on the raw store written by ``cma collect`` (no reviewed mappings needed),
on the *receive-time* timeline, i.e. what this machine could actually have acted on:

1. **Reference moves.** Coinbase BTC-USD mid; a move is a 1-second log return of at least
   ``threshold_bps`` (de-clustered with a cooldown).
2. **Stale-quote lifetime.** For every move and every in-play Kalshi "above strike"
   contract, how long the quote on the side a taker would hit (the ask after an up-move,
   the bid after a down-move) survives before it is re-quoted or taken.
3. **Executable edge.** The book as observed ``L`` ms after the move, valued at the model
   fair price at that time (log-normal digital on the 60 s settlement average, volatility
   from Deribit DVOL), minus the price and the venue taker fee, for a grid of latencies.
4. **Baseline.** The same "edge" at random times without a move. Positive move-conditional
   edge only counts above this baseline, which absorbs model and basis error.
5. **Lead-lag.** The platform's discovery engine (CCF/HY, permutation significance,
   out-of-sample and economic gates) on reference vs contract mids.

All numbers are model-relative estimates from one observation window; they are evidence
for or against stale quotes, not a backtest and not a promotion decision.
"""

from __future__ import annotations

import bisect
import json
import math
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cma.domain.enums import LiquidityRole, Venue
from cma.domain.fees import KALSHI_STANDARD, FeeSchedule, get_fee_schedule
from cma.domain.models import BookDeltaEvent, BookSnapshotEvent, MarketEvent, PredictionContract
from cma.domain.time import NS_PER_MS, NS_PER_S, iso_from_ns
from cma.features.fair_value import prob_above
from cma.ingestion.book import L2BookBuilder

ABOVE_STRIKE_TYPES = frozenset({"greater", "greater_or_equal"})
REFERENCE = "COINBASE:BTC-USD"
DVOL_URL = (
    "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    "?currency=BTC&resolution=60&start_timestamp={start}&end_timestamp={end}"
)


@dataclass(frozen=True)
class StudyConfig:
    thresholds_bps: tuple[float, ...] = (3.0, 5.0, 10.0)
    move_window_ms: int = 1_000
    cooldown_ms: int = 5_000
    latencies_ms: tuple[int, ...] = (0, 100, 250, 500, 1_000)
    max_lifetime_ms: int = 30_000
    min_tte_s: float = 90.0  # skip the settlement-averaging endgame
    max_tte_s: float = 6 * 3600.0
    fair_band: tuple[float, float] = (0.10, 0.90)
    min_fair_move: float = 0.01  # the move must be worth >= 1 cent on this contract
    order_qty: int = 10
    baseline_every_s: float = 10.0
    lead_lag_contracts: int = 4
    seed: int = 20261006


@dataclass(frozen=True)
class Quote:
    """Top of book as observed at receive time (YES side, canonical book)."""

    ts: NDArray[np.int64]
    bid: NDArray[np.float64]  # NaN when the side is empty
    ask: NDArray[np.float64]
    bid_qty: NDArray[np.float64]
    ask_qty: NDArray[np.float64]

    def index_at(self, t_ns: int) -> int:
        """Last observation at or before ``t_ns`` (-1 if none)."""
        return int(np.searchsorted(self.ts, t_ns, side="right")) - 1


@dataclass(frozen=True)
class AboveContract:
    contract_id: str
    instrument_id: str
    series: str
    strike: float
    close_ns: int
    fee: FeeSchedule


@dataclass
class Sample:
    """One (reference move, contract) observation."""

    t0_ns: int
    contract_id: str
    series: str
    threshold_bps: float
    move_bps: float
    direction: int  # +1 up-move (taker would buy YES at the ask), -1 down-move (sell at bid)
    tte_s: float
    fair_before: float
    fair_after: float
    stale_price: float  # the quote a taker would hit, as observed just before t0
    lifetime_ms: float | None  # None: survived beyond max_lifetime_ms
    edge_c: dict[int, float | None] = field(default_factory=dict)  # latency -> net c/contract
    qty: dict[int, float | None] = field(default_factory=dict)
    # market-anchored: Kalshi's own mid before the move + the model's predicted change.
    # Removes level disagreement (tails, vol, basis) and keeps the delta a latency trader
    # exploits; this is the primary latency measure.
    anchored_edge_c: dict[int, float | None] = field(default_factory=dict)


# ----------------------------------------------------------------------------- inputs


def above_contracts(contracts: Iterable[PredictionContract]) -> list[AboveContract]:
    """Kalshi "above strike" contracts with a usable strike and close time."""
    out: list[AboveContract] = []
    for c in contracts:
        md = c.settlement_metadata
        if c.venue is not Venue.KALSHI or md.get("strike_type") not in ABOVE_STRIKE_TYPES:
            continue
        strike = md.get("floor_strike")
        if strike is None or c.close_ts_ns is None:
            continue
        try:
            fee = get_fee_schedule(c.fee_schedule_id) if c.fee_schedule_id else KALSHI_STANDARD
        except KeyError:
            fee = KALSHI_STANDARD
        out.append(
            AboveContract(
                contract_id=c.contract_id,
                instrument_id=c.outcome_instruments.get("YES", c.contract_id),
                series=c.series_id or c.native_id.split("-")[0],
                strike=float(strike),
                close_ns=int(c.close_ts_ns),
                fee=fee,
            )
        )
    return out


def quote_series(events: Iterable[MarketEvent], instrument_id: str) -> Quote:
    """Receive-time top of book of one instrument (only while the book is valid)."""
    builder: L2BookBuilder | None = None
    ts: list[int] = []
    bid: list[float] = []
    ask: list[float] = []
    bq: list[float] = []
    aq: list[float] = []
    for ev in events:
        if ev.instrument_id != instrument_id or not isinstance(
            ev, BookSnapshotEvent | BookDeltaEvent
        ):
            continue
        if builder is None:
            builder = L2BookBuilder(
                venue=ev.venue, instrument_id=instrument_id, require_sequence=False
            )
        builder.apply(ev)
        if not builder.is_valid:
            continue
        bb, ba = builder.best_bid(), builder.best_ask()
        ts.append(ev.recv_ts_ns)
        bid.append(float(bb[0]) if bb else math.nan)
        bq.append(float(bb[1]) if bb else 0.0)
        ask.append(float(ba[0]) if ba else math.nan)
        aq.append(float(ba[1]) if ba else 0.0)
    return Quote(
        ts=np.asarray(ts, dtype=np.int64),
        bid=np.asarray(bid, dtype=float),
        ask=np.asarray(ask, dtype=float),
        bid_qty=np.asarray(bq, dtype=float),
        ask_qty=np.asarray(aq, dtype=float),
    )


def mid_from_quote(q: Quote) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    ok = np.isfinite(q.bid) & np.isfinite(q.ask)
    return q.ts[ok], ((q.bid + q.ask) / 2)[ok]


def fetch_dvol(start_ns: int, end_ns: int, *, timeout_s: float = 15.0) -> list[tuple[int, float]]:
    """Deribit DVOL (percent) 1-minute closes over [start, end]; [] when unavailable."""
    url = DVOL_URL.format(start=start_ns // NS_PER_MS - 120_000, end=end_ns // NS_PER_MS)
    req = urllib.request.Request(url, headers={"User-Agent": "cma-research"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            rows = json.load(resp)["result"]["data"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [(int(r[0]) * NS_PER_MS, float(r[4])) for r in rows]


def realized_vol(
    ts: NDArray[np.int64], mid: NDArray[np.float64], *, sample_s: float = 1.0
) -> float:
    """Annualised realised vol of the reference sampled on a fixed grid (fallback sigma)."""
    if ts.size < 3:
        return 0.5
    grid = np.arange(ts[0], ts[-1], int(sample_s * NS_PER_S), dtype=np.int64)
    if grid.size < 30:
        return 0.5
    idx = np.searchsorted(ts, grid, side="right") - 1
    r = np.diff(np.log(mid[np.maximum(idx, 0)]))
    return float(np.sqrt(np.mean(r * r) * (365.25 * 86400 / sample_s)))


def sigma_lookup(dvol: Sequence[tuple[int, float]], fallback: float) -> Callable[[int], float]:
    times = [t for t, _ in dvol]
    vals = [v / 100.0 for _, v in dvol]

    def at(t_ns: int) -> float:
        i = bisect.bisect_right(times, t_ns) - 1
        return vals[i] if i >= 0 else (vals[0] if vals else fallback)

    return at


# ----------------------------------------------------------------------------- moves


def detect_moves(
    ts: NDArray[np.int64],
    mid: NDArray[np.float64],
    *,
    threshold_bps: float,
    window_ms: int = 1_000,
    cooldown_ms: int = 5_000,
) -> list[tuple[int, int, float]]:
    """(t0, direction, move_bps): first time the trailing ``window`` return crosses the
    threshold; later crossings within ``cooldown`` belong to the same move."""
    out: list[tuple[int, int, float]] = []
    if ts.size < 2:
        return out
    win = window_ms * NS_PER_MS
    cooldown = cooldown_ms * NS_PER_MS
    j = 0
    last: int | None = None
    for i in range(ts.size):
        t = int(ts[i])
        while int(ts[j]) < t - win:
            j += 1
        r = math.log(mid[i] / mid[j]) * 1e4
        if abs(r) >= threshold_bps and (last is None or t - last >= cooldown):
            out.append((t, 1 if r > 0 else -1, float(r)))
            last = t
    return out


# ----------------------------------------------------------------------------- study


def _fair(c: AboveContract, spot: float, t_ns: int, sigma: float) -> float:
    return prob_above(
        spot=spot,
        strike=c.strike,
        now_ns=t_ns,
        observation_end_ns=c.close_ns,
        sigma=sigma,
        observation_method="AVG_60S_BEFORE",
    )


def _fee_c(c: AboveContract, price: float, qty: int) -> float:
    p = Decimal(str(round(price, 4)))
    if not Decimal(0) < p < Decimal(1):
        return math.inf
    total = c.fee.fee(price=p, quantity=Decimal(qty), role=LiquidityRole.TAKER)
    return 100.0 * float(total) / qty


def _ref_at(ref_ts: NDArray[np.int64], ref_mid: NDArray[np.float64], t_ns: int) -> float | None:
    i = int(np.searchsorted(ref_ts, t_ns, side="right")) - 1
    return None if i < 0 else float(ref_mid[i])


def stale_lifetime_ms(q: Quote, t0_ns: int, direction: int, horizon_ms: int) -> float | None:
    """Ms until the quote a taker would hit after the move is re-quoted or taken.

    Up-move: the best ask observed just before t0 must disappear or rise. Down-move: the
    best bid must disappear or fall. None if it survives the whole horizon.
    """
    i0 = q.index_at(t0_ns)
    if i0 < 0:
        return None
    side = q.ask if direction > 0 else q.bid
    stale = side[i0]
    if not math.isfinite(stale):
        return None
    end = t0_ns + horizon_ms * NS_PER_MS
    for k in range(i0 + 1, q.ts.size):
        if q.ts[k] > end:
            break
        v = side[k]
        gone = not math.isfinite(v) or (v > stale + 1e-9 if direction > 0 else v < stale - 1e-9)
        if gone:
            return (int(q.ts[k]) - t0_ns) / NS_PER_MS
    return None


def executable_edge_c(
    c: AboveContract,
    q: Quote,
    t_ns: int,
    *,
    direction: int,
    fair: float,
    qty: int,
) -> tuple[float | None, float | None]:
    """Net cents per contract (and size) of taking the observed book at ``t_ns``."""
    i = q.index_at(t_ns)
    if i < 0:
        return None, None
    if direction > 0:
        price, size = q.ask[i], q.ask_qty[i]
        gross = fair - price
    else:
        price, size = q.bid[i], q.bid_qty[i]
        gross = price - fair
    if not math.isfinite(price) or size <= 0:
        return None, None
    return 100.0 * gross - _fee_c(c, price, qty), float(size)


def _kalshi_mid(q: Quote, t_ns: int) -> float | None:
    i = q.index_at(t_ns)
    if i < 0 or not (math.isfinite(q.bid[i]) and math.isfinite(q.ask[i])):
        return None
    return float(q.bid[i] + q.ask[i]) / 2


def _anchored_fair(
    mid_anchor: float | None, fair_model_anchor: float, fair_model_now: float
) -> float | None:
    if mid_anchor is None:
        return None
    return min(1.0, max(0.0, mid_anchor + (fair_model_now - fair_model_anchor)))


def run_event_study(
    moves: Sequence[tuple[int, int, float]],
    threshold_bps: float,
    *,
    contracts: Sequence[AboveContract],
    quotes: Mapping[str, Quote],
    ref_ts: NDArray[np.int64],
    ref_mid: NDArray[np.float64],
    sigma_at: Callable[[int], float],
    cfg: StudyConfig,
) -> list[Sample]:
    samples: list[Sample] = []
    lo, hi = cfg.fair_band
    for t0, direction, move_bps in moves:
        before = _ref_at(ref_ts, ref_mid, t0 - cfg.move_window_ms * NS_PER_MS)
        after = _ref_at(ref_ts, ref_mid, t0)
        if before is None or after is None:
            continue
        sigma = sigma_at(t0)
        for c in contracts:
            tte_s = (c.close_ns - t0) / NS_PER_S
            if not cfg.min_tte_s <= tte_s <= cfg.max_tte_s:
                continue
            q = quotes.get(c.instrument_id)
            if q is None or q.ts.size == 0 or q.index_at(t0) < 0:
                continue
            f0, f1 = _fair(c, before, t0, sigma), _fair(c, after, t0, sigma)
            if not (lo <= f0 <= hi or lo <= f1 <= hi) or abs(f1 - f0) < cfg.min_fair_move:
                continue
            i0 = q.index_at(t0)
            stale = q.ask[i0] if direction > 0 else q.bid[i0]
            if not math.isfinite(stale):
                continue
            s = Sample(
                t0_ns=t0,
                contract_id=c.contract_id,
                series=c.series,
                threshold_bps=threshold_bps,
                move_bps=move_bps,
                direction=direction,
                tte_s=tte_s,
                fair_before=f0,
                fair_after=f1,
                stale_price=float(stale),
                lifetime_ms=stale_lifetime_ms(q, t0, direction, cfg.max_lifetime_ms),
            )
            t_anchor = t0 - cfg.move_window_ms * NS_PER_MS
            mid_anchor = _kalshi_mid(q, t_anchor)
            fair_anchor = _fair(c, before, t_anchor, sigma)
            for lat in cfg.latencies_ms:
                t = t0 + lat * NS_PER_MS
                spot = _ref_at(ref_ts, ref_mid, t)
                if spot is None:
                    s.edge_c[lat], s.qty[lat], s.anchored_edge_c[lat] = None, None, None
                    continue
                fair_t = _fair(c, spot, t, sigma)
                s.edge_c[lat], s.qty[lat] = executable_edge_c(
                    c, q, t, direction=direction, fair=fair_t, qty=cfg.order_qty
                )
                anchored = _anchored_fair(mid_anchor, fair_anchor, fair_t)
                s.anchored_edge_c[lat] = (
                    None
                    if anchored is None
                    else executable_edge_c(
                        c, q, t, direction=direction, fair=anchored, qty=cfg.order_qty
                    )[0]
                )
            samples.append(s)
    return samples


def baseline_edges(
    contracts: Sequence[AboveContract],
    *,
    quotes: Mapping[str, Quote],
    ref_ts: NDArray[np.int64],
    ref_mid: NDArray[np.float64],
    sigma_at: Callable[[int], float],
    cfg: StudyConfig,
) -> tuple[list[float], list[float]]:
    """Best-side "edge" (c/contract) at regular times regardless of moves.

    Returns (model-absolute, market-anchored). The first absorbs model/basis error; the
    second is roughly minus (half the spread + fee), the cost of trading without news.
    """
    if ref_ts.size == 0:
        return [], []
    out: list[float] = []
    anchored_out: list[float] = []
    step = int(cfg.baseline_every_s * NS_PER_S)
    lo, hi = cfg.fair_band
    for t in range(int(ref_ts[0]) + step, int(ref_ts[-1]), step):
        spot = _ref_at(ref_ts, ref_mid, t)
        if spot is None:
            continue
        sigma = sigma_at(t)
        for c in contracts:
            if not cfg.min_tte_s <= (c.close_ns - t) / NS_PER_S <= cfg.max_tte_s:
                continue
            q = quotes.get(c.instrument_id)
            if q is None or q.index_at(t) < 0:
                continue
            fair = _fair(c, spot, t, sigma)
            if not lo <= fair <= hi:
                continue
            t_anchor = t - cfg.move_window_ms * NS_PER_MS
            spot_anchor = _ref_at(ref_ts, ref_mid, t_anchor)
            anchored = (
                None
                if spot_anchor is None
                else _anchored_fair(
                    _kalshi_mid(q, t_anchor), _fair(c, spot_anchor, t_anchor, sigma), fair
                )
            )
            for target, value in ((out, fair), (anchored_out, anchored)):
                if value is None:
                    continue
                best = None
                for d in (1, -1):
                    e, _ = executable_edge_c(c, q, t, direction=d, fair=value, qty=cfg.order_qty)
                    if e is not None and (best is None or e > best):
                        best = e
                if best is not None:
                    target.append(best)
    return out, anchored_out


# ----------------------------------------------------------------------------- summary


def _pct(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _baseline_stats(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "share_positive": (sum(1 for e in values if e > 0) / len(values)) if values else None,
        "mean_c": float(np.mean(values)) if values else None,
        "p95_c": _pct(values, 95),
    }


def clustered_stats(pairs: Sequence[tuple[float, int]]) -> dict[str, Any]:
    """Mean, share > 0 and move-clustered standard error of ``(value, move id)`` pairs.

    Every in-play strike reacts to the same BTC move, so samples are not independent. The
    standard error is the cluster-robust (CR1) one with each move as a cluster; with one
    move it is undefined (None).
    """
    if not pairs:
        return {
            "n": 0,
            "moves": 0,
            "share_positive": None,
            "mean_c": None,
            "se_c": None,
            "mean_positive_c": None,
        }
    vals = np.asarray([v for v, _ in pairs], dtype=float)
    mean = float(vals.mean())
    resid: dict[int, float] = {}
    for v, move in pairs:
        resid[move] = resid.get(move, 0.0) + (v - mean)
    g = len(resid)
    se = math.sqrt(g / (g - 1) * sum(r * r for r in resid.values())) / vals.size if g > 1 else None
    pos = vals[vals > 0]
    return {
        "n": int(vals.size),
        "moves": g,
        "share_positive": float(pos.size / vals.size),
        "mean_c": mean,
        "se_c": se,
        "mean_positive_c": float(pos.mean()) if pos.size else None,
    }


def _summary_row(
    thr: float, series: str, grp: Sequence[Sample], cfg: StudyConfig
) -> dict[str, Any]:
    lives = [s.lifetime_ms for s in grp if s.lifetime_ms is not None]
    censored = sum(1 for s in grp if s.lifetime_ms is None)

    def by_latency(get: Callable[[Sample, int], float | None]) -> dict[str, Any]:
        return {
            str(lat): clustered_stats([(e, s.t0_ns) for s in grp if (e := get(s, lat)) is not None])
            for lat in cfg.latencies_ms
        }

    return {
        "threshold_bps": thr,
        "series": series,
        "samples": len(grp),
        "moves": len({s.t0_ns for s in grp}),
        "lifetime_ms": {
            "p25": _pct(lives, 25),
            "median": _pct(lives, 50),
            "p75": _pct(lives, 75),
            "p90": _pct(lives, 90),
            "share_beyond_horizon": censored / len(grp) if grp else None,
        },
        "edge_by_latency": by_latency(lambda s, lat: s.edge_c.get(lat)),
        "anchored_edge_by_latency": by_latency(lambda s, lat: s.anchored_edge_c.get(lat)),
    }


def summarize_samples(
    samples: Sequence[Sample], cfg: StudyConfig, *, pooled: bool = False
) -> list[dict[str, Any]]:
    """One row per (threshold, series): lifetimes and executable edge by latency.

    ``pooled`` gives one row per threshold over every series (series ``"all"``).
    """
    groups: dict[tuple[float, str], list[Sample]] = {}
    for s in samples:
        groups.setdefault((s.threshold_bps, "all" if pooled else s.series), []).append(s)
    return [_summary_row(thr, series, groups[thr, series], cfg) for thr, series in sorted(groups)]


# ----------------------------------------------------------------------------- decision

PRIMARY_THRESHOLD_BPS = 5.0
DECISION_LATENCIES_MS = (100, 250)  # what a well-placed server can buy; 0 ms is a diagnostic
MIN_DECISION_MOVES = 30


def live_decision(result: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered real-market call from one live window (rule fixed before the first run).

    * ``REJECT`` (the stale-quote taker on these books): no threshold has a positive pooled
      market-anchored mean edge at 100 or 250 ms, and at the primary 5 bps threshold the
      100 ms edge is negative by more than two move-clustered standard errors, on at least
      30 moves.
    * ``COLLECT_MORE_DATA``: anything else (too few moves, an inconclusive interval or a
      positive cell). A positive cell from one short window is a hypothesis to test on the
      >= 14-day collection with the out-of-sample and promotion gates, never a promotion.

    ``FORWARD_PAPER_CANDIDATE`` is never issued from a single live window.
    """
    rule = (
        "REJECT when no move threshold shows a positive pooled market-anchored edge at "
        f"{' or '.join(f'{x} ms' for x in DECISION_LATENCIES_MS)} and the "
        f"{PRIMARY_THRESHOLD_BPS:g} bps edge at {DECISION_LATENCIES_MS[0]} ms is negative by "
        f"more than 2 move-clustered SE on >= {MIN_DECISION_MOVES} moves; otherwise "
        "COLLECT_MORE_DATA. One live window never promotes to paper."
    )
    if "error" in result:
        return {"decision": "COLLECT_MORE_DATA", "reasons": [str(result["error"])], "rule": rule}
    cells: list[str] = []
    positive: list[str] = []
    primary: Mapping[str, Any] | None = None
    for row in result.get("pooled", []):
        thr = float(row["threshold_bps"])
        for lat in DECISION_LATENCIES_MS:
            st = row["anchored_edge_by_latency"].get(str(lat), {})
            mean, se = st.get("mean_c"), st.get("se_c")
            if mean is None:
                continue
            label = f"{thr:g} bps at {lat} ms"
            band = f" ± {2 * se:.2f}" if se is not None else ""
            cells.append(
                f"{label}: {mean:+.2f}¢{band} (2 SE), {st['n']} samples on {st['moves']} moves"
            )
            if mean > 0:
                positive.append(label)
            if thr == PRIMARY_THRESHOLD_BPS and lat == DECISION_LATENCIES_MS[0]:
                primary = st
    if primary is None or not primary.get("n"):
        reasons = [f"no {PRIMARY_THRESHOLD_BPS:g} bps samples at {DECISION_LATENCIES_MS[0]} ms"]
        return {"decision": "COLLECT_MORE_DATA", "reasons": reasons + cells, "rule": rule}
    if positive:
        reasons = [f"positive mean edge: {', '.join(positive)}"]
        return {"decision": "COLLECT_MORE_DATA", "reasons": reasons + cells, "rule": rule}
    if primary["moves"] < MIN_DECISION_MOVES:
        reasons = [f"only {primary['moves']} moves at {PRIMARY_THRESHOLD_BPS:g} bps"]
        return {"decision": "COLLECT_MORE_DATA", "reasons": reasons + cells, "rule": rule}
    se = primary.get("se_c")
    if se is None or primary["mean_c"] + 2 * se >= 0:
        reasons = ["negative but within 2 SE of zero at the primary cell"]
        return {"decision": "COLLECT_MORE_DATA", "reasons": reasons + cells, "rule": rule}
    reasons = ["negative at every threshold at 100 and 250 ms, beyond 2 SE at the primary cell"]
    return {"decision": "REJECT", "reasons": reasons + cells, "rule": rule}


def lead_lag_contracts(
    *,
    contracts: Sequence[AboveContract],
    quotes: Mapping[str, Quote],
    ref_ts: NDArray[np.int64],
    ref_mid: NDArray[np.float64],
    cfg: StudyConfig,
) -> list[dict[str, Any]]:
    from cma.models.lead_lag.discovery import LeadLagConfig, discover_lead_lag

    ranked = sorted(
        (c for c in contracts if c.instrument_id in quotes),
        key=lambda c: -quotes[c.instrument_id].ts.size,
    )[: cfg.lead_lag_contracts]
    ll_cfg = LeadLagConfig(
        grid_ms=100,
        max_lag_ms=5_000,
        horizons_ms=(500, 1_000, 2_000),
        permutation_samples=199,
        cost_hurdle=0.0175 + 0.005,
        seed=cfg.seed,
        min_obs=300,
    )
    out: list[dict[str, Any]] = []
    for c in ranked:
        y_ts, y_v = mid_from_quote(quotes[c.instrument_id])
        if y_ts.size < 50:
            continue
        res = discover_lead_lag(ref_ts, ref_mid, y_ts, y_v, ll_cfg).to_dict()
        for bulky in ("ccf", "hy_ccf", "horizon_results"):
            res.pop(bulky, None)
        out.append({"contract": c.contract_id, "updates": int(y_ts.size), "result": res})
    return out


def analyze(
    events: Sequence[MarketEvent],
    contracts: Sequence[PredictionContract],
    *,
    cfg: StudyConfig | None = None,
    dvol: Sequence[tuple[int, float]] | None = None,
) -> dict[str, Any]:
    """In-memory front end (tests, small captures): events -> quotes -> study."""
    by_instrument: dict[str, list[MarketEvent]] = {}
    for ev in events:  # one pass; per-instrument replays then touch only their own events
        by_instrument.setdefault(ev.instrument_id, []).append(ev)
    above = above_contracts(contracts)
    quotes = {
        inst: quote_series(by_instrument.get(inst, []), inst)
        for inst in {REFERENCE, *(c.instrument_id for c in above)}
    }
    return analyze_quotes(quotes, contracts, cfg=cfg, dvol=dvol)


def stream_quotes(raw_root: Path, instruments: Iterable[str]) -> tuple[dict[str, Quote], int]:
    """Receive-time top of book per instrument straight from the raw store.

    One pass, constant memory per book: only *changes* of the top of book are kept, so a
    multi-hour capture with millions of deep-book deltas reduces to compact arrays.
    Returns (quotes, raw messages read).
    """
    from array import array

    from cma.adapters.base import MalformedPayloadError
    from cma.storage.normalize import default_adapters
    from cma.storage.raw import RawReader

    table = default_adapters()
    wanted = set(instruments)
    builders: dict[str, L2BookBuilder] = {}
    cols: dict[str, tuple[array[int], array[float], array[float], array[float], array[float]]] = {}
    last: dict[str, tuple[float, float, float, float]] = {}
    n = 0
    for raw in RawReader(raw_root).iter_messages():
        n += 1
        adapter = table.get(raw.venue)
        if adapter is None:
            continue
        try:
            events = adapter.parse(raw)
        except MalformedPayloadError:
            continue
        for ev in events:
            inst = ev.instrument_id
            if inst not in wanted or not isinstance(ev, BookSnapshotEvent | BookDeltaEvent):
                continue
            b = builders.get(inst)
            if b is None:
                b = builders[inst] = L2BookBuilder(
                    venue=ev.venue, instrument_id=inst, require_sequence=False
                )
            b.apply(ev)
            if not b.is_valid:
                continue
            bb, ba = b.best_bid(), b.best_ask()
            top = (
                float(bb[0]) if bb else math.nan,
                float(bb[1]) if bb else 0.0,
                float(ba[0]) if ba else math.nan,
                float(ba[1]) if ba else 0.0,
            )
            prev = last.get(inst)
            if prev is not None and all(
                x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(prev, top, strict=True)
            ):
                continue
            last[inst] = top
            c = cols.get(inst)
            if c is None:
                c = cols[inst] = (array("q"), array("d"), array("d"), array("d"), array("d"))
            c[0].append(ev.recv_ts_ns)
            c[1].append(top[0])
            c[2].append(top[2])
            c[3].append(top[1])
            c[4].append(top[3])
    quotes = {
        inst: Quote(
            ts=np.asarray(c[0], dtype=np.int64),
            bid=np.asarray(c[1], dtype=float),
            ask=np.asarray(c[2], dtype=float),
            bid_qty=np.asarray(c[3], dtype=float),
            ask_qty=np.asarray(c[4], dtype=float),
        )
        for inst, c in cols.items()
    }
    return quotes, n


def analyze_quotes(
    quotes: Mapping[str, Quote],
    contracts: Sequence[PredictionContract],
    *,
    cfg: StudyConfig | None = None,
    dvol: Sequence[tuple[int, float]] | None = None,
) -> dict[str, Any]:
    """The study on receive-time quotes (reference + Kalshi YES books)."""
    cfg = cfg or StudyConfig()
    ref_q = quotes.get(REFERENCE)
    ref_ts, ref_mid = (
        mid_from_quote(ref_q) if ref_q is not None else (np.zeros(0, np.int64), np.zeros(0))
    )
    above = above_contracts(contracts)
    quotes = {c.instrument_id: quotes[c.instrument_id] for c in above if c.instrument_id in quotes}
    quotes = {k: v for k, v in quotes.items() if v.ts.size}
    if ref_ts.size == 0:
        return {"error": "no reference (Coinbase BTC-USD) data", "contracts": len(above)}
    start, end = int(ref_ts[0]), int(ref_ts[-1])
    rv = realized_vol(ref_ts, ref_mid)
    dvol_rows = list(dvol) if dvol is not None else fetch_dvol(start, end)
    sigma_at = sigma_lookup(dvol_rows, rv)
    samples: list[Sample] = []
    moves_found: dict[str, int] = {}
    for thr in cfg.thresholds_bps:
        moves = detect_moves(
            ref_ts,
            ref_mid,
            threshold_bps=thr,
            window_ms=cfg.move_window_ms,
            cooldown_ms=cfg.cooldown_ms,
        )
        moves_found[f"{thr:g}"] = len(moves)
        samples += run_event_study(
            moves,
            thr,
            contracts=above,
            quotes=quotes,
            ref_ts=ref_ts,
            ref_mid=ref_mid,
            sigma_at=sigma_at,
            cfg=cfg,
        )
    base, base_anchored = baseline_edges(
        above, quotes=quotes, ref_ts=ref_ts, ref_mid=ref_mid, sigma_at=sigma_at, cfg=cfg
    )
    updates = {c.series: 0 for c in above}
    for c in above:
        if c.instrument_id in quotes:
            updates[c.series] += int(quotes[c.instrument_id].ts.size)
    result: dict[str, Any] = {
        "window": {
            "start": iso_from_ns(start),
            "end": iso_from_ns(end),
            "hours": (end - start) / 3600 / NS_PER_S,
        },
        "reference": {
            "instrument": REFERENCE,
            "updates": int(ref_ts.size),
            "realized_vol": rv,
            "dvol_points": len(dvol_rows),
            "dvol_mean": float(np.mean([v for _, v in dvol_rows])) / 100 if dvol_rows else None,
        },
        "contracts": {
            "above_strike": len(above),
            "with_quotes": len(quotes),
            "quote_updates_by_series": updates,
        },
        "config": asdict(cfg),
        "moves_by_threshold": moves_found,
        "summary": summarize_samples(samples, cfg),
        "pooled": summarize_samples(samples, cfg, pooled=True),
        "baseline": _baseline_stats(base),
        "baseline_anchored": _baseline_stats(base_anchored),
        "lead_lag": lead_lag_contracts(
            contracts=above, quotes=quotes, ref_ts=ref_ts, ref_mid=ref_mid, cfg=cfg
        ),
    }
    result["decision"] = live_decision(result)
    return result


def _f(v: Any, nd: int = 0) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _edge_table(result: Mapping[str, Any], key: str, lat: Sequence[str]) -> list[str]:
    rows = [
        "| move ≥ bps | series | samples | lifetime p25 / median / p75 ms | beyond 30 s | "
        + " | ".join(f"{x} ms: share>0 / mean ¢" for x in lat)
        + " |",
        "|" + "---|" * (5 + len(lat)),
    ]
    for r in result.get("summary", []):
        lt = r["lifetime_ms"]
        cells = []
        for x in lat:
            e = r.get(key, {}).get(x, {})
            cells.append(f"{_f(e.get('share_positive'), 2)} / {_f(e.get('mean_c'), 2)}")
        rows.append(
            f"| {r['threshold_bps']:g} | {r['series']} | {r['samples']} | "
            f"{_f(lt['p25'])} / {_f(lt['median'])} / {_f(lt['p75'])} | "
            f"{_f(lt['share_beyond_horizon'], 2)} | " + " | ".join(cells) + " |"
        )
    return rows


def _pooled_table(result: Mapping[str, Any], lat: Sequence[str]) -> list[str]:
    rows = [
        "| move ≥ bps | samples | moves | lifetime median ms | "
        + " | ".join(f"{x} ms: mean ± 2 SE ¢ (share>0)" for x in lat)
        + " |",
        "|" + "---|" * (4 + len(lat)),
    ]
    for r in result.get("pooled", []):
        cells = []
        for x in lat:
            e = r["anchored_edge_by_latency"].get(x, {})
            band = f" ± {2 * e['se_c']:.2f}" if e.get("se_c") is not None else ""
            cells.append(f"{_f(e.get('mean_c'), 2)}{band} ({_f(e.get('share_positive'), 2)})")
        rows.append(
            f"| {r['threshold_bps']:g} | {r['samples']} | {r['moves']} | "
            f"{_f(r['lifetime_ms']['median'])} | " + " | ".join(cells) + " |"
        )
    return rows


def _baseline_line(name: str, b: Mapping[str, Any]) -> str:
    return (
        f"{name}: n={b.get('n')}, share with positive edge {_f(b.get('share_positive'), 2)}, "
        f"mean {_f(b.get('mean_c'), 2)} ¢, p95 {_f(b.get('p95_c'), 2)} ¢."
    )


def render_markdown(result: Mapping[str, Any]) -> str:
    w: list[str] = []
    win = result.get("window", {})
    ref = result.get("reference", {})
    vol = ref.get("dvol_mean") or ref.get("realized_vol") or float("nan")
    lat = [str(x) for x in result.get("config", {}).get("latencies_ms", [])]
    w.append("# Live staleness study: Kalshi BTC vs Coinbase")
    w.append("")
    w.append(
        f"Window {win.get('start')} to {win.get('end')} ({win.get('hours', 0):.2f} h). "
        f"Reference updates: {ref.get('updates')}; vol {vol:.3f}."
    )
    w.append(
        f"Quote updates by series: {result.get('contracts', {}).get('quote_updates_by_series')}"
    )
    w.append(f"Moves found by threshold (bps in 1 s): {result.get('moves_by_threshold')}")
    w.append("")
    dec = result.get("decision") or {}
    if dec:
        w.append(f"## Decision: `{dec.get('decision')}` (stale-quote taker on these books)")
        w.append("")
        w += [f"* {r}" for r in dec.get("reasons", [])]
        w.append("")
        w.append(f"Rule, fixed before the first live run: {dec.get('rule')}")
        w.append("")
    w.append("## Market-anchored executable edge after fees (primary latency measure)")
    w.append("")
    w.append(
        "Fair value = Kalshi's own mid 1 s before the move + the model's predicted change from "
        "the BTC move; edge = fair - price - taker fee, for the book observed L ms after the "
        "move. Lifetime = how long the quote a taker would hit survived. Standard errors are "
        "clustered by move (every in-play strike reacts to the same move)."
    )
    w.append("")
    if result.get("pooled"):
        w.append("All series pooled:")
        w.append("")
        w += _pooled_table(result, lat)
        w.append("")
        w.append("By series:")
        w.append("")
    w += _edge_table(result, "anchored_edge_by_latency", lat)
    w.append("")
    w.append(_baseline_line("No-move baseline (anchored)", result.get("baseline_anchored", {})))
    w.append("")
    w.append("## Model-absolute edge (secondary: includes level disagreement)")
    w.append("")
    w += _edge_table(result, "edge_by_latency", lat)
    w.append("")
    w.append(_baseline_line("No-move baseline (model-absolute)", result.get("baseline", {})))
    w.append("")
    w.append("## Lead-lag (reference mid -> contract mid)")
    w.append("")
    for r in result.get("lead_lag", []):
        o = r["result"]
        w.append(
            f"* {r['contract']} ({r['updates']} updates): lag {o.get('best_positive_lag_ms')} ms "
            f"(HY {o.get('hy_best_lag_ms')} ms), p {_f(o.get('p_value'), 3)}, ΔOOS R² "
            f"{_f(o.get('incremental_oos_r2'), 3)}, qualifies {o.get('qualifies')}"
        )
    if not result.get("lead_lag"):
        w.append("* not enough contract updates for lead-lag discovery")
    w.append("")
    return "\n".join(w) + "\n"


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "summary.json"
    js.write_text(json.dumps(result, indent=1, default=str) + "\n")
    md = out_dir / "summary.md"
    md.write_text(render_markdown(result))
    return [js, md]
