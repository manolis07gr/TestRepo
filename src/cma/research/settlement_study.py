"""Hold-to-settlement study on settled Kalshi BTC 15-minute markets.

Question: when the options-style model and Kalshi's quote disagree by more than the fee,
does buying the side the model favours and holding it to settlement make money? The method
and the decision rule were fixed before the first full run (section 7 of
docs/EDGE_EVALUATION_METHOD.md):

* Contract: ``KXBTC15M`` pays $1 when the 60 s BRTI average before close is at least the
  60 s BRTI average before open (the floor strike).
* Decision minutes: candles ending 2..13 minutes after open, sane book, fair in 10-90c.
* Model: log-normal digital on the settlement average; spot = Coinbase close x the
  Coinbase-to-BRTI basis (median of the last 8 quarter-hour marks); volatility = Deribit
  DVOL (primary) or 60-minute realised volatility.
* Trade: first minute per market whose after-fee edge on the better side is >= theta; one
  trade of 100 contracts, held to settlement; filled at the NEXT minute's quote (primary),
  the same minute's (optimistic) or two minutes later (stress).
* Chronological 60/40 split; the in-sample t-statistic picks the specification that is
  tested out of sample; standard errors cluster by UTC day.

Everything is on 1-minute bars, so this tests minute-scale strategies only; the latency
question belongs to :mod:`cma.research.live_study`.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cma.domain.enums import LiquidityRole
from cma.domain.fees import KALSHI_STANDARD
from cma.domain.time import NS_PER_S
from cma.features.fair_value import prob_above
from cma.research.live_study import clustered_stats
from cma.research.settlement_data import (
    Candle,
    SettledMarket,
    coinbase_path,
    dvol_path,
    load_candles,
    load_chunked_rows,
    load_markets,
)

MINUTE = 60
MINUTES_PER_YEAR = 365.25 * 24 * 60
MARKET_MINUTES = 15


@dataclass(frozen=True)
class SettlementConfig:
    series: str = "KXBTC15M"
    first_minute: int = 2  # decision candles k: end = open + 60 k
    last_minute: int = 13
    last_fill_minute: int = 14  # the final candle ends at close: never a fill
    max_spread: float = 0.10
    fair_band: tuple[float, float] = (0.10, 0.90)
    thresholds_c: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
    vol_inputs: tuple[str, ...] = ("dvol", "rv60")
    primary_vol: str = "dvol"
    qty: int = 100
    basis_marks: int = 8
    rv_window_min: int = 60
    rv_min_returns: int = 30
    max_ref_staleness_s: int = 300
    max_dvol_staleness_s: int = 3600
    in_sample_frac: float = 0.60
    min_trades: int = 200
    fee_stress: float = 1.5
    stress_delay_min: int = 2
    checkpoints_min_before_close: tuple[int, ...] = (10, 5, 2)
    max_top_day_share: float = 0.70
    min_neighbor_frac: float = 0.60
    min_month_trades: int = 50
    min_months_positive_frac: float = 0.5


# ----------------------------------------------------------------------------- inputs


class Reference:
    """Coinbase 1-minute candles and DVOL with as-of lookups (no look-ahead).

    A value "at t" comes from the last bar that had finished by t: Coinbase candles and
    DVOL rows are stamped with their start, so the bar starting at t - 60 s is the latest
    one complete at t."""

    def __init__(
        self,
        coinbase: Sequence[Sequence[float]],
        dvol: Sequence[Sequence[float]],
        cfg: SettlementConfig,
    ) -> None:
        self.cfg = cfg
        self.cb_start: NDArray[np.int64] = np.asarray([int(r[0]) for r in coinbase], np.int64)
        self.cb_close: NDArray[np.float64] = np.asarray([r[4] for r in coinbase], float)
        self.cb_typical: NDArray[np.float64] = np.asarray(
            [(r[1] + r[2] + r[3] + r[4]) / 4.0 for r in coinbase], float
        )
        self.dv_ts: NDArray[np.int64] = np.asarray([int(r[0]) for r in dvol], np.int64)
        self.dv_val: NDArray[np.float64] = np.asarray([r[1] for r in dvol], float)
        self._rv = self._rolling_rv()

    def _rolling_rv(self) -> NDArray[np.float64]:
        """Annualised realised vol of the ``rv_window_min`` 1-minute returns ending at each
        bar (consecutive minutes only); NaN with fewer than ``rv_min_returns`` returns."""
        n = self.cb_close.size
        out = np.full(n, np.nan)
        if n < 2:
            return out
        r = np.diff(np.log(self.cb_close))
        ok = np.diff(self.cb_start) == MINUTE
        sq = np.where(ok, r * r, 0.0)
        cs = np.concatenate(([0.0], np.cumsum(sq)))
        cc = np.concatenate(([0], np.cumsum(ok.astype(np.int64))))
        w = self.cfg.rv_window_min
        for i in range(1, n):  # return j = i - 1 ends at bar i
            lo = max(0, i - w)
            cnt = int(cc[i] - cc[lo])
            if cnt >= self.cfg.rv_min_returns:
                out[i] = math.sqrt(float(cs[i] - cs[lo]) / cnt * MINUTES_PER_YEAR)
        return out

    def _cb_index(self, t: int) -> int | None:
        i = int(np.searchsorted(self.cb_start, t - MINUTE, side="right")) - 1
        if i < 0 or (t - MINUTE) - int(self.cb_start[i]) > self.cfg.max_ref_staleness_s:
            return None
        return i

    def spot(self, t: int) -> float | None:
        i = self._cb_index(t)
        return None if i is None else float(self.cb_close[i])

    def minute_average(self, mark: int) -> float | None:
        """Coinbase typical price of the minute ending exactly at ``mark``."""
        i = int(np.searchsorted(self.cb_start, mark - MINUTE, side="left"))
        if i < self.cb_start.size and int(self.cb_start[i]) == mark - MINUTE:
            return float(self.cb_typical[i])
        return None

    def sigma(self, t: int, kind: str) -> float | None:
        if kind == "dvol":
            i = int(np.searchsorted(self.dv_ts, t - MINUTE, side="right")) - 1
            if i < 0 or (t - MINUTE) - int(self.dv_ts[i]) > self.cfg.max_dvol_staleness_s:
                return None
            return float(self.dv_val[i]) / 100.0
        if kind == "rv60":
            j = self._cb_index(t)
            if j is None or math.isnan(self._rv[j]):
                return None
            return float(self._rv[j])
        raise ValueError(f"unknown volatility input {kind!r}")


class Basis:
    """Coinbase-to-BRTI log basis from the quarter-hour marks already published at t.

    Every market's floor strike is the BRTI 60 s average before its open, and its
    expiration value the same average before its close; against the Coinbase typical price
    of that minute each gives one basis observation."""

    def __init__(self, markets: Sequence[SettledMarket], ref: Reference, n: int) -> None:
        marks: dict[int, float] = {}
        for m in markets:
            if m.floor_strike:
                marks[m.open_ts] = m.floor_strike
            if m.expiration_value:
                marks.setdefault(m.close_ts, m.expiration_value)
        ts: list[int] = []
        vals: list[float] = []
        for t in sorted(marks):
            cb = ref.minute_average(t)
            if cb:
                ts.append(t)
                vals.append(math.log(marks[t] / cb))
        self.ts: NDArray[np.int64] = np.asarray(ts, np.int64)
        self.vals: NDArray[np.float64] = np.asarray(vals, float)
        self.n = n

    def at(self, t: int) -> float | None:
        j = int(np.searchsorted(self.ts, t, side="right"))
        if j == 0:
            return None
        return float(np.median(self.vals[max(0, j - self.n) : j]))

    def stats_bp(self) -> dict[str, Any]:
        if self.vals.size == 0:
            return {"marks": 0}
        bp = self.vals * 1e4
        return {
            "marks": int(bp.size),
            "median_bp": float(np.median(bp)),
            "p05_bp": float(np.percentile(bp, 5)),
            "p95_bp": float(np.percentile(bp, 95)),
            "mean_abs_bp": float(np.mean(np.abs(bp))),
        }


@lru_cache(maxsize=4096)
def _fee_c(price_milli: int, qty: int) -> float:
    """Kalshi taker fee in cents per contract for one ``qty`` order at a YES-equivalent
    price of ``price_milli`` / 1000 (order rounding included)."""
    p = Decimal(price_milli) / Decimal(1000)
    total = KALSHI_STANDARD.fee(price=p, quantity=Decimal(qty), role=LiquidityRole.TAKER)
    return 100.0 * float(total) / qty


def fee_c(price: float, qty: int) -> float:
    return _fee_c(round(float(price) * 1000), qty)


@dataclass(frozen=True)
class Minute:
    """One decision/fill candle of a market: closing YES bid/ask and model fair values."""

    k: int
    end_ts: int
    bid: float | None
    ask: float | None
    fair: Mapping[str, float | None]

    def sane(self, max_spread: float) -> bool:
        return (
            self.bid is not None
            and self.ask is not None
            and 0.0 < self.bid < self.ask < 1.0
            and self.ask - self.bid <= max_spread + 1e-9
        )


def market_minutes(
    m: SettledMarket,
    candles: Sequence[Candle],
    ref: Reference,
    basis: Basis,
    cfg: SettlementConfig,
) -> dict[int, Minute]:
    """Candles 1..15 of a market by minute index, with fair values on decision minutes."""
    out: dict[int, Minute] = {}
    for c in candles:
        end = int(c[0])
        k, rem = divmod(end - m.open_ts, MINUTE)
        if rem or not 1 <= k <= MARKET_MINUTES:
            continue
        fair: dict[str, float | None] = dict.fromkeys(cfg.vol_inputs)
        if cfg.first_minute <= k <= cfg.last_minute and m.floor_strike:
            spot = ref.spot(end)
            b = basis.at(end)
            if spot is not None and b is not None:
                for kind in cfg.vol_inputs:
                    sigma = ref.sigma(end, kind)
                    if sigma is not None and sigma > 0:
                        fair[kind] = prob_above(
                            spot=spot * math.exp(b),
                            strike=m.floor_strike,
                            now_ns=end * NS_PER_S,
                            observation_end_ns=m.close_ts * NS_PER_S,
                            sigma=sigma,
                            observation_method="AVG_60S_BEFORE",
                        )
        out[k] = Minute(k=k, end_ts=end, bid=c[4], ask=c[8], fair=fair)
    return out


# ----------------------------------------------------------------------------- trades


@dataclass(frozen=True)
class Trade:
    ticker: str
    day: str  # UTC date of the market's close
    month: str
    close_ts: int
    side: int  # +1 bought YES, -1 bought NO
    k: int  # decision minute
    signal_edge_c: float
    fill: float  # price paid per contract (dollars)
    fee_c: float
    payoff: int
    pnl_c: float


def best_side(fair: float, bid: float, ask: float, qty: int) -> tuple[int, float]:
    """(+1 buy YES at the ask | -1 buy NO at 1 - bid, after-fee edge in cents)."""
    e_yes = 100.0 * (fair - ask) - fee_c(ask, qty)
    e_no = 100.0 * (bid - fair) - fee_c(1.0 - bid, qty)
    return (1, e_yes) if e_yes >= e_no else (-1, e_no)


def first_signal(
    minutes: Mapping[int, Minute], vol: str, theta_c: float, cfg: SettlementConfig
) -> tuple[int, int, float] | None:
    """(minute, side, edge) of the first decision minute whose edge is >= theta."""
    lo, hi = cfg.fair_band
    for k in range(cfg.first_minute, cfg.last_minute + 1):
        mi = minutes.get(k)
        if mi is None or not mi.sane(cfg.max_spread):
            continue
        f = mi.fair.get(vol)
        if f is None or not lo <= f <= hi:
            continue
        bid, ask = mi.bid, mi.ask
        if bid is None or ask is None:  # sane() already excludes this
            continue
        side, edge = best_side(f, bid, ask, cfg.qty)
        if edge >= theta_c:
            return k, side, edge
    return None


def fill_trade(
    m: SettledMarket,
    minutes: Mapping[int, Minute],
    signal: tuple[int, int, float],
    delay_min: int,
    cfg: SettlementConfig,
) -> Trade | None:
    """The trade filled ``delay_min`` minutes after the signal (None: no sane quote)."""
    k, side, edge = signal
    kf = k + delay_min
    if kf > cfg.last_fill_minute:
        return None
    mf = minutes.get(kf)
    if mf is None or not mf.sane(cfg.max_spread):
        return None
    bid, ask = mf.bid, mf.ask
    if bid is None or ask is None:  # sane() already excludes this
        return None
    price = ask if side == 1 else 1.0 - bid
    won = (side == 1) == (m.result == "yes")
    fee = fee_c(price, cfg.qty)
    close = datetime.fromtimestamp(m.close_ts, UTC)
    return Trade(
        ticker=m.ticker,
        day=close.strftime("%Y-%m-%d"),
        month=close.strftime("%Y-%m"),
        close_ts=m.close_ts,
        side=side,
        k=k,
        signal_edge_c=edge,
        fill=price,
        fee_c=fee,
        payoff=int(won),
        pnl_c=100.0 * (int(won) - price) - fee,
    )


def trade_stats(trades: Sequence[Trade], *, fee_mult: float = 1.0) -> dict[str, Any]:
    """Mean P&L in cents per contract with a day-clustered (CR1) standard error."""
    days = sorted({t.day for t in trades})
    day_id = {d: i for i, d in enumerate(days)}
    pnl = [t.pnl_c - (fee_mult - 1.0) * t.fee_c for t in trades]
    cs = clustered_stats([(v, day_id[t.day]) for v, t in zip(pnl, trades, strict=True)])
    mean, se = cs["mean_c"], cs["se_c"]
    return {
        "n": cs["n"],
        "days": cs["moves"],
        "mean_c": mean,
        "se_c": se,
        "t": (mean / se) if mean is not None and se else None,
        "win_rate": (sum(t.payoff for t in trades) / len(trades)) if trades else None,
        "mean_signal_edge_c": float(np.mean([t.signal_edge_c for t in trades])) if trades else None,
        "mean_fill": float(np.mean([t.fill for t in trades])) if trades else None,
        "total_c": float(sum(pnl)),
        "yes_share": (sum(1 for t in trades if t.side == 1) / len(trades)) if trades else None,
    }


def by_group(trades: Sequence[Trade], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        groups[str(getattr(t, key))].append(t)
    return {g: trade_stats(v) for g, v in sorted(groups.items())}


def top_day_share(trades: Sequence[Trade]) -> float | None:
    """Largest single-day P&L as a share of the total (None when the total is <= 0)."""
    per_day: dict[str, float] = defaultdict(float)
    for t in trades:
        per_day[t.day] += t.pnl_c
    total = sum(per_day.values())
    if total <= 0 or not per_day:
        return None
    return max(per_day.values()) / total


# ----------------------------------------------------------------------------- diagnostics


def _brier(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    q = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def ols_hc1(y: NDArray[np.float64], x: NDArray[np.float64]) -> tuple[list[float], list[float]]:
    """OLS coefficients and heteroskedasticity-robust (HC1) standard errors."""
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    e = y - x @ beta
    meat = (x * (e * e)[:, None]).T @ x
    cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    return [float(b) for b in beta], [float(s) for s in np.sqrt(np.diag(cov))]


CALIBRATION_EDGES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0)


def checkpoint_diagnostics(
    markets: Sequence[SettledMarket],
    minutes_by_ticker: Mapping[str, Mapping[int, Minute]],
    minutes_before_close: int,
    cfg: SettlementConfig,
) -> dict[str, Any]:
    """Who forecasts settlement better at a fixed time before close: Kalshi's mid or the
    model? Plus calibration by price bucket (the favourite-longshot question)."""
    k = MARKET_MINUTES - minutes_before_close
    rows: list[tuple[float, float, float, float, float, float]] = []
    for m in markets:
        mi = minutes_by_ticker.get(m.ticker, {}).get(k)
        if mi is None or not mi.sane(cfg.max_spread):
            continue
        fd, fr = mi.fair.get("dvol"), mi.fair.get("rv60")
        if fd is None or fr is None:
            continue
        bid, ask = mi.bid, mi.ask
        if bid is None or ask is None:  # sane() already excludes this
            continue
        outcome = 1.0 if m.result == "yes" else 0.0
        rows.append((outcome, bid, ask, (bid + ask) / 2.0, fd, fr))
    if len(rows) < 30:
        return {"minutes_before_close": minutes_before_close, "n": len(rows)}
    a = np.asarray(rows, float)
    ys, bids, asks, mids, fds, frs = (a[:, i] for i in range(6))
    out: dict[str, Any] = {
        "minutes_before_close": minutes_before_close,
        "n": len(rows),
        "base_rate": float(ys.mean()),
        "brier": {
            "kalshi_mid": _brier(mids, ys),
            "model_dvol": _brier(fds, ys),
            "model_rv60": _brier(frs, ys),
        },
        "log_loss": {
            "kalshi_mid": _log_loss(mids, ys),
            "model_dvol": _log_loss(fds, ys),
            "model_rv60": _log_loss(frs, ys),
        },
    }
    regressions: dict[str, Any] = {}
    for name, f in (("dvol", fds), ("rv60", frs)):
        x = np.column_stack([np.ones_like(mids), mids, f - mids])
        beta, se = ols_hc1(ys, x)
        regressions[name] = {
            "mid_coef": beta[1],
            "mid_se": se[1],
            "gap_coef": beta[2],
            "gap_se": se[2],
            "gap_t": beta[2] / se[2] if se[2] else None,
        }
    out["outcome_on_mid_and_model_gap"] = regressions
    buckets = []
    for lo, hi in itertools.pairwise(CALIBRATION_EDGES):
        sel = (mids >= lo) & ((mids < hi) if hi < 1.0 else (mids <= hi))
        n = int(sel.sum())
        if n == 0:
            continue
        yy = ys[sel]
        freq = float(yy.mean())
        buy_yes = [
            100.0 * (yv - av) - fee_c(av, cfg.qty) for yv, av in zip(yy, asks[sel], strict=True)
        ]
        buy_no = [
            100.0 * ((1.0 - yv) - (1.0 - bv)) - fee_c(1.0 - bv, cfg.qty)
            for yv, bv in zip(yy, bids[sel], strict=True)
        ]
        buckets.append(
            {
                "mid_lo": lo,
                "mid_hi": hi,
                "n": n,
                "mean_mid": float(mids[sel].mean()),
                "yes_freq": freq,
                "yes_freq_se": math.sqrt(max(freq * (1 - freq), 1e-12) / n),
                "buy_yes_pnl_c": float(np.mean(buy_yes)),
                "buy_no_pnl_c": float(np.mean(buy_no)),
            }
        )
    out["calibration"] = buckets
    return out


# ----------------------------------------------------------------------------- study


def split_markets(
    markets: Sequence[SettledMarket], frac: float
) -> tuple[list[SettledMarket], list[SettledMarket]]:
    ordered = sorted(markets, key=lambda m: (m.close_ts, m.ticker))
    cut = int(len(ordered) * frac)
    return ordered[:cut], ordered[cut:]


def simulate(
    markets: Sequence[SettledMarket],
    minutes_by_ticker: Mapping[str, Mapping[int, Minute]],
    *,
    vol: str,
    theta_c: float,
    delay_min: int,
    cfg: SettlementConfig,
) -> tuple[list[Trade], int]:
    """Trades of one specification and the number of signals that found no fill."""
    trades: list[Trade] = []
    unfilled = 0
    for m in markets:
        minutes = minutes_by_ticker.get(m.ticker)
        if not minutes:
            continue
        sig = first_signal(minutes, vol, theta_c, cfg)
        if sig is None:
            continue
        t = fill_trade(m, minutes, sig, delay_min, cfg)
        if t is None:
            unfilled += 1
        else:
            trades.append(t)
    return trades, unfilled


def _spec_key(vol: str, theta: float) -> str:
    return f"{vol}|{theta:g}"


def select_spec(grid: Mapping[str, Mapping[str, Any]], cfg: SettlementConfig) -> str | None:
    """Highest in-sample t-statistic among specifications with enough trades."""
    best: tuple[float, str] | None = None
    for key, row in grid.items():
        s = row["in_sample"]
        if s["n"] < cfg.min_trades or s["t"] is None:
            continue
        if best is None or s["t"] > best[0]:
            best = (s["t"], key)
    return None if best is None else best[1]


def _num(v: float | None, nd: int, *, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def settlement_decision(result: Mapping[str, Any], cfg: SettlementConfig) -> dict[str, Any]:
    """The pre-registered rule (method section 7.7)."""
    sel = result.get("selected")
    if not sel:
        return {"decision": "REJECT", "reasons": ["no specification has enough trades"]}
    oos = sel["out_of_sample"]
    gates = {
        "oos_mean_above_2se": bool(oos["t"] is not None and oos["t"] > 2.0),
        "fees_x1_5_positive": bool((sel["oos_fee_stress"]["mean_c"] or 0.0) > 0.0),
        "two_minute_fill_positive": bool((sel["oos_delay_stress"]["mean_c"] or 0.0) > 0.0),
        "min_trades": bool(oos["n"] >= cfg.min_trades),
        "top_day_share": bool(
            sel["oos_top_day_share"] is not None
            and sel["oos_top_day_share"] <= cfg.max_top_day_share
        ),
        "neighbours_profitable": bool(
            sel["oos_neighbour_frac"] is not None
            and sel["oos_neighbour_frac"] >= cfg.min_neighbor_frac
        ),
        "months_positive": bool(
            sel["oos_months_positive_frac"] is not None
            and sel["oos_months_positive_frac"] >= cfg.min_months_positive_frac
        ),
    }
    mean = oos["mean_c"]
    if mean is None or mean <= 0:
        decision = "REJECT"
    elif all(gates.values()):
        decision = "FORWARD_PAPER_CANDIDATE"
    else:
        decision = "COLLECT_MORE_DATA"
    se = oos["se_c"]
    ins = sel["in_sample"]
    reasons = [
        f"selected {sel['spec']} (in-sample t {_num(ins['t'], 2)}, {ins['n']} trades)",
        f"out of sample: {_num(mean, 2, signed=True)}c per contract"
        + (f" ± {2 * se:.2f} (2 SE)" if se else "")
        + f" on {oos['n']} trades",
    ]
    reasons += [f"gate {name}: {'pass' if ok else 'FAIL'}" for name, ok in gates.items()]
    return {"decision": decision, "gates": gates, "reasons": reasons}


def analyze(
    markets: Sequence[SettledMarket],
    candles: Mapping[str, Sequence[Candle]],
    coinbase: Sequence[Sequence[float]],
    dvol: Sequence[Sequence[float]],
    cfg: SettlementConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or SettlementConfig()
    usable = [
        m
        for m in markets
        if m.series == cfg.series
        and m.strike_type in {"greater", "greater_or_equal"}
        and m.floor_strike
        and candles.get(m.ticker)
    ]
    ref = Reference(coinbase, dvol, cfg)
    basis = Basis(markets, ref, cfg.basis_marks)
    minutes_by_ticker = {
        m.ticker: market_minutes(m, candles[m.ticker], ref, basis, cfg) for m in usable
    }
    in_s, out_s = split_markets(usable, cfg.in_sample_frac)

    grid: dict[str, dict[str, Any]] = {}
    oos_trades: dict[str, list[Trade]] = {}
    for vol in cfg.vol_inputs:
        for theta in cfg.thresholds_c:
            key = _spec_key(vol, theta)
            tin, uin = simulate(
                in_s, minutes_by_ticker, vol=vol, theta_c=theta, delay_min=1, cfg=cfg
            )
            tout, uout = simulate(
                out_s, minutes_by_ticker, vol=vol, theta_c=theta, delay_min=1, cfg=cfg
            )
            oos_trades[key] = tout
            grid[key] = {
                "vol": vol,
                "theta_c": theta,
                "in_sample": trade_stats(tin) | {"unfilled": uin},
                "out_of_sample": trade_stats(tout) | {"unfilled": uout},
            }

    selected: dict[str, Any] | None = None
    sel_key = select_spec(grid, cfg)
    if sel_key is not None:
        vol, theta = grid[sel_key]["vol"], grid[sel_key]["theta_c"]
        tout = oos_trades[sel_key]
        t_delay, _ = simulate(
            out_s,
            minutes_by_ticker,
            vol=vol,
            theta_c=theta,
            delay_min=cfg.stress_delay_min,
            cfg=cfg,
        )
        t_same, _ = simulate(out_s, minutes_by_ticker, vol=vol, theta_c=theta, delay_min=0, cfg=cfg)
        i = cfg.thresholds_c.index(theta)
        neighbours = [cfg.thresholds_c[j] for j in (i - 1, i + 1) if 0 <= j < len(cfg.thresholds_c)]
        n_pos = sum(
            1
            for th in neighbours
            if (grid[_spec_key(vol, th)]["out_of_sample"]["mean_c"] or 0.0) > 0.0
        )
        months = {
            mo: s for mo, s in by_group(tout, "month").items() if s["n"] >= cfg.min_month_trades
        }
        selected = {
            "spec": sel_key,
            "vol": vol,
            "theta_c": theta,
            "in_sample": grid[sel_key]["in_sample"],
            "out_of_sample": grid[sel_key]["out_of_sample"],
            "oos_fee_stress": trade_stats(tout, fee_mult=cfg.fee_stress),
            "oos_delay_stress": trade_stats(t_delay),
            "oos_same_minute_fill": trade_stats(t_same),
            "oos_by_month": months,
            "oos_by_side": by_group(tout, "side"),
            "oos_by_minute": by_group(tout, "k"),
            "oos_top_day_share": top_day_share(tout),
            "oos_neighbours": neighbours,
            "oos_neighbour_frac": (n_pos / len(neighbours)) if neighbours else None,
            "oos_months_positive_frac": (
                sum(1 for s in months.values() if (s["mean_c"] or 0.0) > 0) / len(months)
                if months
                else None
            ),
        }

    def span(ms: Sequence[SettledMarket]) -> dict[str, Any]:
        if not ms:
            return {"markets": 0}
        return {
            "markets": len(ms),
            "first_close": datetime.fromtimestamp(ms[0].close_ts, UTC).isoformat(),
            "last_close": datetime.fromtimestamp(ms[-1].close_ts, UTC).isoformat(),
        }

    settled_consistent = sum(
        1
        for m in usable
        if m.expiration_value is not None
        and m.floor_strike is not None
        and (m.expiration_value >= m.floor_strike) == (m.result == "yes")
    )
    result: dict[str, Any] = {
        "config": asdict(cfg),
        "data": {
            "markets_listed": len(markets),
            "markets_usable": len(usable),
            "in_sample": span(in_s),
            "out_of_sample": span(out_s),
            "yes_rate": (sum(1 for m in usable if m.result == "yes") / len(usable))
            if usable
            else None,
            "result_matches_settlement_values": settled_consistent,
            "coinbase_minutes": int(ref.cb_start.size),
            "dvol_minutes": int(ref.dv_ts.size),
            "basis": basis.stats_bp(),
        },
        "grid": grid,
        "selected": selected,
        "diagnostics": [
            checkpoint_diagnostics(usable, minutes_by_ticker, mb, cfg)
            for mb in cfg.checkpoints_min_before_close
        ],
    }
    result["decision"] = settlement_decision(result, cfg)
    return result


def analyze_dir(data_dir: Path, cfg: SettlementConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SettlementConfig()
    return analyze(
        load_markets(data_dir, cfg.series),
        load_candles(data_dir, cfg.series),
        load_chunked_rows(coinbase_path(data_dir)),
        load_chunked_rows(dvol_path(data_dir)),
        cfg,
    )


# ----------------------------------------------------------------------------- report


def _pm(s: Mapping[str, Any]) -> str:
    """'+0.12 ± 0.34' (mean ± 2 SE, cents) or 'n/a'."""
    if s.get("mean_c") is None:
        return "n/a"
    se = s.get("se_c")
    return f"{s['mean_c']:+.2f}" + (f" ± {2 * se:.2f}" if se else "")


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _span_text(span: Mapping[str, Any]) -> str:
    first, last = str(span.get("first_close", ""))[:10], str(span.get("last_close", ""))[:10]
    return f"{span.get('markets', 0)} markets, {first} to {last}"


def render_markdown(result: Mapping[str, Any]) -> str:
    d = result["data"]
    dec = result["decision"]
    w: list[str] = ["# Hold-to-settlement study: Kalshi BTC 15-minute markets", ""]
    w.append(
        f"**Decision: {dec['decision']}** (rule fixed before the first full run: "
        "docs/EDGE_EVALUATION_METHOD.md section 7)."
    )
    w.append("")
    w += [f"* {r}" for r in dec["reasons"]]
    w += [
        "",
        "## Data",
        "",
        f"* {d['markets_usable']} settled `KXBTC15M` markets with candles "
        f"(of {d['markets_listed']} listed); YES settled {_pct(d['yes_rate'])} of the time. "
        f"Kalshi's result agrees with its own settlement values in "
        f"{d['result_matches_settlement_values']} markets.",
        f"* In-sample: {_span_text(d['in_sample'])}. Out-of-sample: "
        f"{_span_text(d['out_of_sample'])}.",
    ]
    b = d["basis"]
    if b.get("marks"):
        w.append(
            f"* Coinbase-to-BRTI basis over {b['marks']} quarter-hour marks: median "
            f"{b['median_bp']:+.2f} bp (5–95%: {b['p05_bp']:+.2f} to {b['p95_bp']:+.2f} bp)."
        )
    w.append(
        f"* Reference data: {d['coinbase_minutes']} Coinbase minutes, {d['dvol_minutes']} DVOL "
        "minutes."
    )
    w += [
        "",
        "## Strategy grid",
        "",
        "Buy the side the model favours when its edge after the taker fee is at least θ; one "
        "trade per market, filled at the next minute's quote, held to settlement. Cents per "
        "contract after fees, ± 2 standard errors clustered by day.",
        "",
        "| Volatility | θ (¢) | In-sample trades | In-sample P&L | t | Out-of-sample trades "
        "| Out-of-sample P&L | t | Win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["grid"].values():
        i, o = row["in_sample"], row["out_of_sample"]
        w.append(
            f"| {row['vol']} | {row['theta_c']:g} | {i['n']} | {_pm(i)} | {_num(i['t'], 1)} "
            f"| {o['n']} | {_pm(o)} | {_num(o['t'], 1)} | {_pct(o['win_rate'])} |"
        )
    sel = result.get("selected")
    if sel:
        w += [
            "",
            f"## Selected specification out of sample ({sel['spec']})",
            "",
            "| Variant | Trades | P&L (¢/contract) | t |",
            "|---|---:|---:|---:|",
        ]
        for name, key in (
            ("Next-minute fill (primary)", "out_of_sample"),
            ("Fees × 1.5", "oos_fee_stress"),
            ("Fill two minutes later", "oos_delay_stress"),
            ("Same-minute fill (optimistic)", "oos_same_minute_fill"),
        ):
            s = sel[key]
            w.append(f"| {name} | {s['n']} | {_pm(s)} | {_num(s['t'], 1)} |")
        w += ["", "| Month | Trades | P&L (¢/contract) |", "|---|---:|---:|"]
        for mo, s in sel["oos_by_month"].items():
            w.append(f"| {mo} | {s['n']} | {_pm(s)} |")
        sides = sel["oos_by_side"]
        w.append("")
        w.append(
            "* Bought YES: "
            + (f"{sides['1']['n']} trades, {_pm(sides['1'])}¢" if "1" in sides else "none")
            + "; bought NO: "
            + (f"{sides['-1']['n']} trades, {_pm(sides['-1'])}¢" if "-1" in sides else "none")
            + "."
        )
        w.append(
            f"* Largest day's share of the P&L: {_pct(sel['oos_top_day_share'])}; neighbouring "
            f"thresholds profitable: {_pct(sel['oos_neighbour_frac'])}; months positive: "
            f"{_pct(sel['oos_months_positive_frac'])}."
        )
    w += [
        "",
        "## Who forecasts settlement better?",
        "",
        "Brier score (lower is better) of Kalshi's mid and of the model, and the regression of "
        "the outcome on the mid and the model–mid gap: a gap coefficient near zero means the "
        "model adds nothing the price does not already contain.",
        "",
        "| Minutes before close | Markets | Brier: Kalshi mid | Brier: model (DVOL) "
        "| Brier: model (60-min vol) | Gap coefficient (DVOL) | Gap coefficient (60-min vol) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for diag in result["diagnostics"]:
        if "brier" not in diag:
            continue
        br, rg = diag["brier"], diag["outcome_on_mid_and_model_gap"]

        def coef(r: Mapping[str, Any]) -> str:
            return f"{r['gap_coef']:+.3f} ± {2 * r['gap_se']:.3f}"

        w.append(
            f"| {diag['minutes_before_close']} | {diag['n']} | {br['kalshi_mid']:.4f} "
            f"| {br['model_dvol']:.4f} | {br['model_rv60']:.4f} | {coef(rg['dvol'])} "
            f"| {coef(rg['rv60'])} |"
        )
    five = next((x for x in result["diagnostics"] if x.get("minutes_before_close") == 5), None)
    if five and five.get("calibration"):
        w += [
            "",
            "## Calibration 5 minutes before close",
            "",
            "| Kalshi mid | Markets | Mean mid | YES settled | Buy YES at ask (¢) "
            "| Buy NO at 1 − bid (¢) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for c in five["calibration"]:
            w.append(
                f"| {c['mid_lo']:.2f}–{c['mid_hi']:.2f} | {c['n']} | {c['mean_mid']:.3f} "
                f"| {c['yes_freq']:.3f} ± {2 * c['yes_freq_se']:.3f} | {c['buy_yes_pnl_c']:+.2f} "
                f"| {c['buy_no_pnl_c']:+.2f} |"
            )
    w += [
        "",
        "Minute bars only: this tests minute-scale strategies, not latency. Fills assume the "
        "quoted price had at least 100 contracts behind it.",
        "",
    ]
    return "\n".join(w)


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "summary.json"
    md = out_dir / "summary.md"
    js.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    md.write_text(render_markdown(result), encoding="utf-8")
    return [js, md]
