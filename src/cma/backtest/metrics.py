"""Trading and predictive metrics (scope s.13). Predictive and trading metrics are kept apart.

Per-position ("trade") P&L is the contract-level net P&L: realized (including settlement)
minus fees. Per-fill mark-outs measure adverse selection: the move of the observed mid
after the fill, signed by the fill side.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np

from cma.backtest.core import CoreRecords, TradingCore
from cma.domain.enums import LiquidityRole, Side
from cma.domain.models import Fill
from cma.domain.numbers import ZERO
from cma.domain.time import NS_PER_DAY, NS_PER_S


def _f(x: Decimal) -> float:
    return float(x)


@dataclass
class TradingMetrics:
    gross_pnl: float
    fees: float
    net_pnl: float
    slippage_cost: float  # vs the signal's executable price at decision time
    n_signals: int
    n_orders: int
    n_fills: int
    n_positions: int
    filled_contracts: float
    fill_rate: float  # filled qty / ordered qty
    order_fill_ratio: float  # orders with any fill / orders
    hit_rate: float  # share of positions with positive net P&L
    profit_factor: float
    mean_position_pnl: float
    median_position_pnl: float
    max_drawdown: float
    sharpe_daily: float  # caveat: few days -> unstable
    turnover: float  # sum |qty * price|
    avg_net_edge_bps: float
    median_net_edge_bps: float
    maker_share: float
    time_in_market_frac: float
    tail_loss_cvar5: float
    markout_bps: dict[str, float] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    concentration: dict[str, Any] = field(default_factory=dict)
    risk_rejections: dict[str, int] = field(default_factory=dict)
    suppressions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization (NaN/inf encoded as strings) for byte comparison."""
        return json.dumps(_jsonable(self.to_dict()), sort_keys=True, separators=(",", ":")).encode()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


def position_pnls(records: CoreRecords, core: TradingCore) -> dict[str, float]:
    """Net P&L per contract (realized + settlement - fees; open positions marked)."""
    marks = core.marks()
    out: dict[str, float] = {}
    for cid, pos in core.portfolio.positions.items():
        pnl = pos.realized_pnl - pos.fees
        if pos.quantity != ZERO:
            pnl += pos.unrealized(marks.get(cid, pos.avg_cost))
        out[cid] = _f(pnl)
    return out


def max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float(np.max(peak - arr))


def cvar(values: Sequence[float], level: float = 0.05) -> float:
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=float))
    k = max(1, math.ceil(level * arr.size))
    return float(np.mean(arr[:k]))


def markouts_bps(records: CoreRecords, horizons: Sequence[int]) -> dict[str, float]:
    """Mean signed mark-out per horizon in bps of $1 face (positive = favourable)."""
    out: dict[str, float] = {}
    fills = {f.fill_id: f for f in records.fills}
    for h in horizons:
        vals = []
        for (fid, hh), mid in records.markouts.items():
            if hh != h or fid not in fills:
                continue
            f = fills[fid]
            sign = 1 if f.side is Side.BUY else -1
            vals.append(sign * _f(mid - f.price) * 1e4)
        out[f"{h}ms"] = float(np.mean(vals)) if vals else float("nan")
    return out


def concentration(
    pnls: Mapping[str, float], fills: Sequence[Fill], family_of: Any
) -> dict[str, Any]:
    total = sum(pnls.values())
    positives = sorted((v for v in pnls.values() if v > 0), reverse=True)
    pos_total = sum(positives)
    by_family: dict[str, float] = defaultdict(float)
    for cid, v in pnls.items():
        by_family[family_of(cid) or cid] += v
    by_day: dict[int, float] = defaultdict(float)
    first_fill: dict[str, int] = {}
    for f in fills:
        first_fill.setdefault(f.contract_id, f.fill_ts_ns)
    for cid, v in pnls.items():
        if cid in first_fill:
            by_day[first_fill[cid] // NS_PER_DAY] += v
    fam_pos = sorted((v for v in by_family.values() if v > 0), reverse=True)
    day_pos = sorted((v for v in by_day.values() if v > 0), reverse=True)
    return {
        "total_net": total,
        "top_contract_share_of_gains": positives[0] / pos_total if pos_total > 0 else 0.0,
        "top_family_share_of_gains": (fam_pos[0] / sum(fam_pos)) if fam_pos else 0.0,
        "top_day_share_of_gains": (day_pos[0] / sum(day_pos)) if day_pos else 0.0,
        "n_families": len(by_family),
        "n_days": len(by_day),
    }


def compute_metrics(
    core: TradingCore, horizons_ms: Sequence[int] = (1_000, 5_000, 30_000)
) -> TradingMetrics:
    rec = core.records
    pf = core.portfolio
    fees = _f(pf.fees_total)
    marks = core.marks()
    unreal = _f(pf.unrealized(marks)) if pf.open_positions() else 0.0
    gross = _f(pf.realized_total) + unreal
    net = gross - fees
    pnls = position_pnls(rec, core)
    vals = list(pnls.values())
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v < 0]

    ordered_qty = sum(_f(o.quantity) for o in rec.orders)
    filled_qty = sum(_f(f.quantity) for f in rec.fills)
    orders_with_fill = len({f.order_id for f in rec.fills})
    meta = {o.order_id: o for o in rec.orders}
    slip = 0.0
    for f in rec.fills:
        o = meta.get(f.order_id)
        if o is None or o.signal_executable_price is None:
            continue
        sign = 1 if f.side is Side.BUY else -1
        slip += sign * _f(f.price - o.signal_executable_price) * _f(f.quantity)

    equity = [_f(p.nav) for p in rec.equity]
    daily: dict[int, float] = {}
    for p in rec.equity:
        daily[p.ts_ns // NS_PER_DAY] = _f(p.nav)
    day_navs = [daily[k] for k in sorted(daily)]
    day_rets = np.diff(day_navs) if len(day_navs) > 1 else np.array([])
    sharpe = (
        float(np.mean(day_rets) / np.std(day_rets, ddof=1) * math.sqrt(365))
        if day_rets.size > 1 and np.std(day_rets, ddof=1) > 0
        else float("nan")
    )
    edges = [_f(s.net_edge) * 1e4 for s in rec.signals]
    maker = sum(_f(f.quantity) for f in rec.fills if f.liquidity_role is LiquidityRole.MAKER)
    tim = 0.0
    if rec.equity:
        span = max(1, rec.equity[-1].ts_ns - rec.equity[0].ts_ns)
        held = sum(1 for p in rec.equity if p.open_risk > 0)
        tim = held / len(rec.equity) if span > 0 else 0.0
    lat = [(o.arrival_ts_ns - o.decision_ts_ns) / 1e6 for o in rec.orders]
    return TradingMetrics(
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        slippage_cost=slip,
        n_signals=len(rec.signals),
        n_orders=len(rec.orders),
        n_fills=len(rec.fills),
        n_positions=len(pnls),
        filled_contracts=filled_qty,
        fill_rate=filled_qty / ordered_qty if ordered_qty else 0.0,
        order_fill_ratio=orders_with_fill / len(rec.orders) if rec.orders else 0.0,
        hit_rate=len(wins) / len(vals) if vals else 0.0,
        profit_factor=(sum(wins) / -sum(losses)) if losses else (math.inf if wins else 0.0),
        mean_position_pnl=float(np.mean(vals)) if vals else 0.0,
        median_position_pnl=float(np.median(vals)) if vals else 0.0,
        max_drawdown=max_drawdown(equity),
        sharpe_daily=sharpe,
        turnover=sum(_f(f.quantity * f.price) for f in rec.fills),
        avg_net_edge_bps=float(np.mean(edges)) if edges else 0.0,
        median_net_edge_bps=float(np.median(edges)) if edges else 0.0,
        maker_share=maker / filled_qty if filled_qty else 0.0,
        time_in_market_frac=tim,
        tail_loss_cvar5=cvar(vals),
        markout_bps=markouts_bps(rec, horizons_ms),
        latency_ms={
            "decision_to_arrival_p50": float(np.median(lat)) if lat else float("nan"),
            "decision_to_arrival_max": float(np.max(lat)) if lat else float("nan"),
        },
        concentration=concentration(pnls, rec.fills, core._family_of),
        risk_rejections={k.value: v for k, v in rec.risk_rejections.items()},
        suppressions={k.value: v for k, v in core.signal_engine.suppressions.items()},
    )


def seconds(ns: int) -> float:
    return ns / NS_PER_S
