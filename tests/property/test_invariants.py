"""Scope s.19 invariants as hypothesis properties."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cma.domain.enums import BookSide, DeltaMode, LiquidityRole, OrderType, Side, TimeInForce, Venue
from cma.domain.fees import KALSHI_STANDARD
from cma.domain.models import BookDeltaEvent, Fill, LevelChange, SimOrder
from cma.execution.simulator.core import ExecutionSimulator
from cma.ingestion.book import L2BookBuilder
from cma.portfolio.ledger import Portfolio
from tests.factories import D, snapshot

pytestmark = pytest.mark.property
INST = "KALSHI:PROP"

changes = st.lists(
    st.tuples(st.booleans(), st.integers(1, 99), st.integers(-60, 60)), min_size=1, max_size=40
)


@settings(max_examples=150, deadline=None)
@given(changes)
def test_valid_book_is_never_crossed_and_sizes_non_negative(
    ops: list[tuple[bool, int, int]],
) -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    b.apply(snapshot(INST, 1, 0, [("0.40", "50")], [("0.60", "50")]))
    for seq, (bid, cents, qty) in enumerate(ops, start=2):
        b.apply(
            BookDeltaEvent(
                venue=Venue.KALSHI,
                instrument_id=INST,
                source_ts_ns=seq,
                recv_ts_ns=seq,
                sequence=seq,
                payload_hash=f"p{seq}",
                changes=(
                    LevelChange(
                        BookSide.BID if bid else BookSide.ASK, Decimal(cents) / 100, Decimal(qty)
                    ),
                ),
                mode=DeltaMode.INCREMENT,
            )
        )
        snap = b.snapshot()
        assert all(lvl.quantity > 0 for lvl in (*snap.bids, *snap.asks))
        if b.is_valid and snap.best_bid and snap.best_ask:
            assert snap.best_bid.price < snap.best_ask.price


@settings(max_examples=80, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(1, 99), st.integers(1, 40)),
        min_size=1,
        max_size=8,
        unique_by=lambda t: t[0],
    ),
    st.integers(1, 100),
    st.integers(1, 100),
)
def test_larger_taker_orders_never_get_a_better_vwap(
    levels: list[tuple[int, int]], q1: int, q2: int
) -> None:
    small, large = sorted((q1, q2))
    asks = [(str(Decimal(p) / 100), str(q)) for p, q in sorted(levels)]
    vwaps = []
    for i, qty in enumerate((small, large)):
        sim = ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD)
        sim.on_venue_event(snapshot(INST, 1, 0, [], asks), 0)
        o = SimOrder(
            order_id=f"o{i}",
            venue=Venue.KALSHI,
            contract_id=INST,
            instrument_id=INST,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=D(qty),
            tif=TimeInForce.IOC,
            submit_ts_ns=1,
            arrival_ts_ns=1,
        )
        sim.submit(o)
        sim.process_arrival(o.order_id, 1)
        vwaps.append(o.avg_fill_price)
    if vwaps[0] is not None and vwaps[1] is not None:
        assert vwaps[1] >= vwaps[0]


@settings(max_examples=120, deadline=None)
@given(
    st.lists(
        st.tuples(st.booleans(), st.integers(1, 99), st.integers(1, 50), st.integers(0, 30)),
        min_size=1,
        max_size=30,
    ),
    st.integers(1, 99),
)
def test_cash_positions_and_pnl_reconcile_to_nav(
    fills: list[tuple[bool, int, int, int]], mark_cents: int
) -> None:
    pf = Portfolio(initial_cash=D(1_000))
    for i, (buy, cents, qty, fee_cents) in enumerate(fills):
        pf.apply_fill(
            Fill(
                fill_id=f"f{i}",
                order_id="o",
                venue=Venue.KALSHI,
                contract_id=INST,
                instrument_id=INST,
                side=Side.BUY if buy else Side.SELL,
                fill_ts_ns=i,
                price=Decimal(cents) / 100,
                quantity=D(qty),
                fee=Decimal(fee_cents) / 100,
                fee_schedule_version="t",
                liquidity_role=LiquidityRole.TAKER,
            )
        )
    marks = {INST: Decimal(mark_cents) / 100}
    assert abs(pf.reconcile(marks)) < D("1e-12")
    assert pf.fees_total == sum((Decimal(f[3]) / 100 for f in fills), D(0))
