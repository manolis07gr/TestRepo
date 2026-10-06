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
            for lat in cfg.latencies_ms:
                t = t0 + lat * NS_PER_MS
                spot = _ref_at(ref_ts, ref_mid, t)
                if spot is None:
                    s.edge_c[lat], s.qty[lat] = None, None
                    continue
                fair_t = _fair(c, spot, t, sigma)
                s.edge_c[lat], s.qty[lat] = executable_edge_c(
                    c, q, t, direction=direction, fair=fair_t, qty=cfg.order_qty
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
) -> list[float]:
    """Best-side "edge" (c/contract) at regular times regardless of moves: model/basis noise."""
    if ref_ts.size == 0:
        return []
    out: list[float] = []
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
            best = None
            for d in (1, -1):
                e, _ = executable_edge_c(c, q, t, direction=d, fair=fair, qty=cfg.order_qty)
                if e is not None and (best is None or e > best):
                    best = e
            if best is not None:
                out.append(best)
    return out


# ----------------------------------------------------------------------------- summary


def _pct(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize_samples(samples: Sequence[Sample], cfg: StudyConfig) -> list[dict[str, Any]]:
    """One row per (threshold, series): lifetimes and executable edge by latency."""
    rows: list[dict[str, Any]] = []
    keys = sorted({(s.threshold_bps, s.series) for s in samples})
    for thr, series in keys:
        grp = [s for s in samples if s.threshold_bps == thr and s.series == series]
        lives = [s.lifetime_ms for s in grp if s.lifetime_ms is not None]
        censored = sum(1 for s in grp if s.lifetime_ms is None)
        by_lat: dict[str, Any] = {}
        for lat in cfg.latencies_ms:
            edges = [e for s in grp if (e := s.edge_c.get(lat)) is not None]
            pos = [e for e in edges if e > 0]
            by_lat[str(lat)] = {
                "n": len(edges),
                "share_positive": (len(pos) / len(edges)) if edges else None,
                "mean_c": float(np.mean(edges)) if edges else None,
                "mean_positive_c": float(np.mean(pos)) if pos else None,
            }
        rows.append(
            {
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
                "edge_by_latency": by_lat,
            }
        )
    return rows


def lead_lag_contracts(
    events: Sequence[MarketEvent],
    *,
    contracts: Sequence[AboveContract],
    quotes: Mapping[str, Quote],
    ref_ts: NDArray[np.int64],
    ref_mid: NDArray[np.float64],
    cfg: StudyConfig,
) -> list[dict[str, Any]]:
    from cma.models.lead_lag.discovery import LeadLagConfig, discover_lead_lag

    del events  # quotes already hold the contract series
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
    cfg = cfg or StudyConfig()
    by_instrument: dict[str, list[MarketEvent]] = {}
    for ev in events:  # one pass; per-instrument replays then touch only their own events
        by_instrument.setdefault(ev.instrument_id, []).append(ev)
    ref_q = quote_series(by_instrument.get(REFERENCE, []), REFERENCE)
    ref_ts, ref_mid = mid_from_quote(ref_q)
    above = above_contracts(contracts)
    quotes = {
        c.instrument_id: quote_series(by_instrument.get(c.instrument_id, []), c.instrument_id)
        for c in above
    }
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
    base = baseline_edges(
        above, quotes=quotes, ref_ts=ref_ts, ref_mid=ref_mid, sigma_at=sigma_at, cfg=cfg
    )
    updates = {c.series: 0 for c in above}
    for c in above:
        if c.instrument_id in quotes:
            updates[c.series] += int(quotes[c.instrument_id].ts.size)
    return {
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
        "baseline": {
            "n": len(base),
            "share_positive": (sum(1 for e in base if e > 0) / len(base)) if base else None,
            "mean_c": float(np.mean(base)) if base else None,
            "p95_c": _pct(base, 95),
        },
        "lead_lag": lead_lag_contracts(
            events, contracts=above, quotes=quotes, ref_ts=ref_ts, ref_mid=ref_mid, cfg=cfg
        ),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    w: list[str] = []
    win = result.get("window", {})
    ref = result.get("reference", {})
    vol = ref.get("dvol_mean") or ref.get("realized_vol") or float("nan")
    w.append("# Live staleness study: Kalshi BTC vs Coinbase")
    w.append("")
    w.append(
        f"Window {win.get('start')} to {win.get('end')} ({win.get('hours', 0):.2f} h). "
        f"Reference updates: {result.get('reference', {}).get('updates')}; "
        f"vol {vol:.3f}."
    )
    w.append(
        f"Quote updates by series: {result.get('contracts', {}).get('quote_updates_by_series')}"
    )
    w.append(f"Moves found by threshold (bps in 1 s): {result.get('moves_by_threshold')}")
    w.append("")
    w.append("## Stale-quote lifetime and executable edge after fees")
    w.append("")
    lat = [str(x) for x in result.get("config", {}).get("latencies_ms", [])]
    w.append(
        "| move ≥ bps | series | samples | lifetime p25 / median / p75 ms | beyond 30 s | "
        + " | ".join(f"{x} ms: +share / mean ¢" for x in lat)
        + " |"
    )
    w.append("|" + "---|" * (5 + len(lat)))
    for r in result.get("summary", []):
        lt = r["lifetime_ms"]

        def f(v: Any, nd: int = 0) -> str:
            return "n/a" if v is None else f"{v:.{nd}f}"

        cells = []
        for x in lat:
            e = r["edge_by_latency"].get(x, {})
            cells.append(f"{f(e.get('share_positive'), 2)} / {f(e.get('mean_c'), 2)}")
        w.append(
            f"| {r['threshold_bps']:g} | {r['series']} | {r['samples']} | "
            f"{f(lt['p25'])} / {f(lt['median'])} / {f(lt['p75'])} | "
            f"{f(lt['share_beyond_horizon'], 2)} | " + " | ".join(cells) + " |"
        )
    b = result.get("baseline", {})
    w.append("")
    w.append(
        f"Baseline (no move, every 10 s, best side): n={b.get('n')}, share positive "
        f"{b.get('share_positive')}, mean {b.get('mean_c')} ¢, p95 {b.get('p95_c')} ¢."
    )
    w.append("")
    w.append("## Lead-lag (reference mid -> contract mid)")
    w.append("")
    for r in result.get("lead_lag", []):
        o = r["result"].get("overall", r["result"])
        w.append(
            f"* {r['contract']} ({r['updates']} updates): lag {o.get('best_positive_lag_ms')} ms, "
            f"p {o.get('p_value')}, ΔOOS R² {o.get('incremental_oos_r2')}, qualifies "
            f"{o.get('qualifies')}"
        )
    w.append("")
    return "\n".join(w) + "\n"


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "summary.json"
    js.write_text(json.dumps(result, indent=1, default=str) + "\n")
    md = out_dir / "summary.md"
    md.write_text(render_markdown(result))
    return [js, md]
