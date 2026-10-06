"""Executable edge decomposition (scope s.10.1).

    expected_net_edge = fair_value_edge - fees - expected_slippage
                        - adverse_selection_buffer - uncertainty_buffer

All amounts are per contract in probability units ($ per $1 face). ``fair_value_edge`` is
measured against the touch (best ask for buys, best bid for sells); slippage is the extra
cost of walking deeper levels (VWAP - touch). Quantity is sized level by level: a level is
used only while its *marginal* net edge clears the threshold, which guarantees the
aggregate net edge does too.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from cma.domain.enums import LiquidityRole, Side
from cma.domain.fees import FeeSchedule
from cma.domain.models import BookLevel, BookSnapshot
from cma.domain.numbers import BPS, ZERO


@dataclass(frozen=True, slots=True)
class EdgeBreakdown:
    side: Side
    fair: Decimal
    best_price: Decimal
    vwap: Decimal
    quantity: Decimal
    limit_price: Decimal  # worst level price used (IOC limit)
    gross_edge: Decimal
    fee: Decimal
    slippage: Decimal
    adverse_selection: Decimal
    uncertainty: Decimal

    @property
    def expected_cost(self) -> Decimal:
        return self.fee + self.slippage + self.adverse_selection + self.uncertainty

    @property
    def net_edge(self) -> Decimal:
        return self.gross_edge - self.expected_cost

    @property
    def net_edge_bps(self) -> Decimal:
        return self.net_edge / BPS


def _marginal_net(
    fair: Decimal,
    side: Side,
    price: Decimal,
    fee_schedule: FeeSchedule,
    buffers: Decimal,
) -> Decimal:
    fee = fee_schedule.fee_per_contract(price=price, role=LiquidityRole.TAKER)
    raw = fair - price if side is Side.BUY else price - fair
    return raw - fee - buffers


def compute_edge(
    *,
    fair: Decimal,
    side: Side,
    levels: Sequence[BookLevel],
    target_quantity: Decimal,
    fee_schedule: FeeSchedule,
    adverse_selection_bps: Decimal = ZERO,
    uncertainty_bps: Decimal = ZERO,
    min_marginal_net_edge_bps: Decimal | None = None,
    max_levels: int = 5,
) -> EdgeBreakdown | None:
    """Edge for trading ``side`` against ``levels`` (asks for BUY, bids for SELL).

    If ``min_marginal_net_edge_bps`` is given, deeper levels are only used while their
    marginal net edge is at least that threshold (the touch is always evaluated). Returns
    None when there is no liquidity.
    """
    if not levels or target_quantity <= ZERO:
        return None
    adverse = adverse_selection_bps * BPS
    uncertainty = uncertainty_bps * BPS
    buffers = adverse + uncertainty
    threshold = None if min_marginal_net_edge_bps is None else min_marginal_net_edge_bps * BPS
    best = levels[0].price
    qty = ZERO
    notional = ZERO
    fee_total = ZERO
    worst = best
    for i, lvl in enumerate(levels[:max_levels]):
        if qty >= target_quantity:
            break
        if (
            i > 0
            and threshold is not None
            and _marginal_net(fair, side, lvl.price, fee_schedule, buffers) < threshold
        ):
            break
        take = min(lvl.quantity, target_quantity - qty)
        if take <= ZERO:
            continue
        qty += take
        notional += take * lvl.price
        fee_total += fee_schedule.fee(price=lvl.price, quantity=take, role=LiquidityRole.TAKER)
        worst = lvl.price
    if qty == ZERO:
        return None
    vwap = notional / qty
    gross = fair - best if side is Side.BUY else best - fair
    slippage = vwap - best if side is Side.BUY else best - vwap
    return EdgeBreakdown(
        side=side,
        fair=fair,
        best_price=best,
        vwap=vwap,
        quantity=qty,
        limit_price=worst,
        gross_edge=gross,
        fee=fee_total / qty,
        slippage=slippage,
        adverse_selection=adverse,
        uncertainty=uncertainty,
    )


def best_edge(
    *,
    fair: Decimal,
    book: BookSnapshot,
    target_quantity: Decimal,
    fee_schedule: FeeSchedule,
    adverse_selection_bps: Decimal = ZERO,
    uncertainty_bps: Decimal = ZERO,
    min_marginal_net_edge_bps: Decimal | None = None,
    max_levels: int = 5,
) -> EdgeBreakdown | None:
    """Evaluate both sides; return the one with the larger net edge (None if no book)."""
    candidates = []
    for side, levels in ((Side.BUY, book.asks), (Side.SELL, book.bids)):
        eb = compute_edge(
            fair=fair,
            side=side,
            levels=levels,
            target_quantity=target_quantity,
            fee_schedule=fee_schedule,
            adverse_selection_bps=adverse_selection_bps,
            uncertainty_bps=uncertainty_bps,
            min_marginal_net_edge_bps=min_marginal_net_edge_bps,
            max_levels=max_levels,
        )
        if eb is not None:
            candidates.append(eb)
    if not candidates:
        return None
    return max(candidates, key=lambda e: (e.net_edge, e.side is Side.BUY))
