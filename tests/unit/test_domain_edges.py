"""Edge cases of the domain layer and simulator branches (validation, limits, lifecycle)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from cma.domain.binary import complement, from_yes_order, yes_book_side, yes_payoff
from cma.domain.enums import (
    BookSide,
    ContractStatus,
    LiquidityRole,
    MappingStatus,
    Operator,
    OrderState,
    OrderType,
    Outcome,
    Side,
    TimeInForce,
    Venue,
)
from cma.domain.errors import InvalidPriceError
from cma.domain.fees import (
    KALSHI_STANDARD,
    NotionalBpsFeeSchedule,
    ScaledFeeSchedule,
    ZeroFeeSchedule,
    get_fee_schedule,
    known_fee_schedules,
    register_fee_schedule,
)
from cma.domain.models import (
    BookLevel,
    BookSnapshot,
    ContractMapping,
    Fill,
    Instrument,
    MarketEvent,
    PredictionContract,
    RawMessage,
    Settlement,
    Signal,
    SimOrder,
    TradeEvent,
)
from cma.domain.numbers import (
    bps_to_decimal,
    ceil_to,
    decimal_to_bps,
    floor_to,
    from_float,
    is_on_tick,
    quantize,
    snap_to_tick,
    to_decimal,
    to_float,
)
from cma.domain.time import (
    ManualClock,
    SystemClock,
    datetime_from_ns,
    iso_from_ns,
    local_to_utc_ns,
    ms_from_ns,
    ns_from_datetime,
    ns_from_iso8601,
    ns_from_ms,
    ns_from_s,
    utc_ns_to_local,
)
from cma.execution.simulator.core import ExecutionSimulator, fills_total_quantity
from cma.storage.events_io import read_events_jsonl, sort_by_recv, write_events_jsonl
from tests.factories import D, delta, snapshot, trade

pytestmark = pytest.mark.unit
INST = "KALSHI:FIXTURE-BOOK"


def test_numbers_helpers() -> None:
    assert to_decimal(" 1.50 ") == D("1.50")
    with pytest.raises(ValueError, match="not a decimal"):
        to_decimal("x1")
    with pytest.raises(ValueError, match="non-finite"):
        to_decimal(Decimal("NaN"))
    with pytest.raises(TypeError):
        to_decimal(True)
    with pytest.raises(ValueError):
        from_float(float("inf"), D("0.01"))
    assert to_float(D("0.25")) == 0.25
    assert quantize(D("0.125"), D("0.01")) == D("0.12")
    assert ceil_to(D("0.121"), D("0.01")) == D("0.13")
    assert floor_to(D("0.129"), D("0.01")) == D("0.12")
    assert is_on_tick(D("0.45"), D("0.01")) and not is_on_tick(D("0.455"), D("0.01"))
    assert snap_to_tick(D("0.456"), D("0.01")) == D("0.46")
    with pytest.raises(InvalidPriceError):
        is_on_tick(D("0.4"), D("0"))
    with pytest.raises(InvalidPriceError):
        snap_to_tick(D("0.4"), D("-1"))
    assert bps_to_decimal(100) == D("0.01")
    assert decimal_to_bps(D("0.01")) == D(100)


def test_time_helpers_and_clocks() -> None:
    dt = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    ns = ns_from_datetime(dt)
    assert datetime_from_ns(ns) == dt
    with pytest.raises(ValueError, match="naive"):
        ns_from_datetime(datetime(2026, 1, 1))
    assert ns_from_iso8601("2026-10-06T12:00:00+0000") == ns
    assert ns_from_iso8601("2026-10-06 12:00:00.5Z") == ns + 500_000_000
    with pytest.raises(ValueError, match="naive"):
        ns_from_iso8601("2026-10-06T12:00:00")
    with pytest.raises(ValueError, match="unsupported"):
        ns_from_iso8601("yesterday")
    assert iso_from_ns(ns) == "2026-10-06T12:00:00.000000000Z"
    assert ns_from_ms(5) == 5_000_000 and ns_from_s(2) == 2_000_000_000
    assert ms_from_ns(1_500_000) == 1.5
    assert utc_ns_to_local(ns, "America/New_York").hour == 8
    with pytest.raises(ValueError, match="naive"):
        local_to_utc_ns(dt, "America/New_York")
    assert SystemClock().now_ns() > 0
    c = ManualClock(10)
    assert c.advance(5) == 15
    with pytest.raises(ValueError):
        c.advance(-1)
    with pytest.raises(ValueError):
        c.set(1)


def test_fee_schedule_variants_and_registry() -> None:
    bps = NotionalBpsFeeSchedule(schedule_id="ref-bps", version="1", taker_bps=D(10))
    assert bps.fee(price=D("100"), quantity=D(2), role=LiquidityRole.TAKER) == D("0.20")
    assert bps.fee(price=D("100"), quantity=D(2), role=LiquidityRole.MAKER) == D(0)
    with pytest.raises(ValueError):
        bps.raw_fee(price=D("-1"), quantity=D(1), role=LiquidityRole.TAKER)
    with pytest.raises(ValueError):
        KALSHI_STANDARD.raw_fee(price=D("0.5"), quantity=D(-1), role=LiquidityRole.TAKER)
    assert ZeroFeeSchedule().fee(price=D("0.3"), quantity=D(9), role=LiquidityRole.TAKER) == 0
    with pytest.raises(ValueError):
        ScaledFeeSchedule(base=KALSHI_STANDARD, factor=D(-1))
    scaled = ScaledFeeSchedule(base=KALSHI_STANDARD, factor=D(2))
    assert scaled.schedule_id == "kalshi-standard*2" and scaled.version == KALSHI_STANDARD.version
    register_fee_schedule(bps, replace=True)
    with pytest.raises(ValueError, match="already registered"):
        register_fee_schedule(bps)
    assert get_fee_schedule("ref-bps") is bps and "ref-bps" in known_fee_schedules()
    with pytest.raises(KeyError, match="unknown fee schedule"):
        get_fee_schedule("nope")


def test_binary_helpers() -> None:
    assert complement(D("0.3")) == D("0.7")
    assert from_yes_order(Outcome.YES, Side.BUY, D("0.3")) == (Side.BUY, D("0.3"))
    assert yes_book_side(Outcome.NO, BookSide.BID) is BookSide.ASK
    assert yes_book_side(Outcome.YES, BookSide.BID) is BookSide.BID
    assert yes_payoff(D(-5), D(1)) == D(-5)


def test_model_validation_rules() -> None:
    with pytest.raises(InvalidPriceError):
        Instrument(
            venue=Venue.COINBASE,
            instrument_id="COINBASE:BTC-USD",
            native_id="BTC-USD",
            kind=__import__("cma.domain.enums", fromlist=["InstrumentKind"]).InstrumentKind.SPOT,
            tick_size=D(0),
        )
    with pytest.raises(InvalidPriceError):
        BookLevel(D("-0.1"), D(1))
    with pytest.raises(InvalidPriceError):
        BookLevel(D("0.1"), D(-1))
    with pytest.raises(InvalidPriceError, match="sorted"):
        BookSnapshot(
            venue=Venue.KALSHI,
            instrument_id=INST,
            source_ts_ns=1,
            recv_ts_ns=1,
            sequence=1,
            bids=(BookLevel(D("0.40"), D(1)), BookLevel(D("0.41"), D(1))),
            asks=(),
        )
    snap = BookSnapshot(
        venue=Venue.KALSHI,
        instrument_id=INST,
        source_ts_ns=None,
        recv_ts_ns=7,
        sequence=None,
        bids=(BookLevel(D("0.40"), D(3)), BookLevel(D("0.39"), D(4))),
        asks=(BookLevel(D("0.45"), D(5)),),
    )
    assert snap.depth(BookSide.BID) == D(7) and snap.depth(BookSide.BID, 1) == D(3)
    assert snap.quantity_at(BookSide.ASK, D("0.45")) == D(5)
    assert snap.quantity_at(BookSide.ASK, D("0.46")) == 0
    assert snap.source_or_recv_ts_ns == 7
    empty = BookSnapshot(
        venue=Venue.KALSHI,
        instrument_id=INST,
        source_ts_ns=1,
        recv_ts_ns=1,
        sequence=1,
        bids=(),
        asks=(),
    )
    assert empty.mid is None and empty.spread is None and empty.best_bid is None
    with pytest.raises(ValueError):
        TradeEvent(
            venue=Venue.KALSHI,
            instrument_id=INST,
            source_ts_ns=1,
            recv_ts_ns=-1,
            payload_hash="h",
            price=D("0.5"),
            size=D(1),
            aggressor_side=None,
            trade_id="t",
        )
    with pytest.raises(InvalidPriceError):
        TradeEvent(
            venue=Venue.KALSHI,
            instrument_id=INST,
            source_ts_ns=1,
            recv_ts_ns=1,
            payload_hash="h",
            price=D("0.5"),
            size=D(0),
            aggressor_side=None,
            trade_id="t",
        )
    raw = RawMessage(venue=Venue.KALSHI, stream="ws", recv_ts_ns=5, payload="{}")
    assert raw.dedup_key.endswith("|5") and len(raw.payload_hash) == 64
    keyed = RawMessage(
        venue=Venue.KALSHI, stream="ws", recv_ts_ns=5, payload="{}", idempotency_key="sid1:seq2"
    )
    assert keyed.dedup_key == "KALSHI|ws|sid1:seq2"
    with pytest.raises(InvalidPriceError):
        PredictionContract(
            venue=Venue.KALSHI,
            contract_id=INST,
            native_id="x",
            event_id="e",
            title="t",
            yes_semantics="y",
            no_semantics="n",
            open_ts_ns=None,
            close_ts_ns=None,
            resolve_ts_ns=None,
            status=ContractStatus.OPEN,
            tick_size=D(1),
        )
    common = dict(
        venue=Venue.KALSHI,
        contract_id=INST,
        underlyings=("BTC-USD",),
        observation_start_ns=None,
        observation_end_ns=10,
        observation_method="POINT",
        timezone="UTC",
        resolution_source="X",
    )
    with pytest.raises(ValueError, match="strike"):
        ContractMapping(operator=Operator.BETWEEN, strikes=(D(1),), **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="increasing"):
        ContractMapping(operator=Operator.BETWEEN, strikes=(D(2), D(1)), **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="version"):
        ContractMapping(operator=Operator.GT, strikes=(D(1),), version=0, **common)  # type: ignore[arg-type]
    m = ContractMapping(operator=Operator.UP, strikes=(), **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = m.strike
    assert MappingStatus.APPROVED_PAPER.at_least(MappingStatus.REVIEWED)
    with pytest.raises(ValueError, match="must pay 1"):
        Settlement(
            venue=Venue.KALSHI,
            contract_id=INST,
            outcome=__import__(
                "cma.domain.enums", fromlist=["SettlementOutcome"]
            ).SettlementOutcome.YES,
            yes_value=D("0.5"),
            settled_ts_ns=1,
        )
    with pytest.raises(ValueError, match="quantity"):
        Fill(
            fill_id="f",
            order_id="o",
            venue=Venue.KALSHI,
            contract_id=INST,
            instrument_id=INST,
            side=Side.BUY,
            fill_ts_ns=1,
            price=D("0.5"),
            quantity=D(0),
            fee=D(0),
            fee_schedule_version="x",
            liquidity_role=LiquidityRole.TAKER,
        )
    with pytest.raises(ValueError, match="negative"):
        Fill(
            fill_id="f",
            order_id="o",
            venue=Venue.KALSHI,
            contract_id=INST,
            instrument_id=INST,
            side=Side.BUY,
            fill_ts_ns=1,
            price=D("0.5"),
            quantity=D(1),
            fee=D(-1),
            fee_schedule_version="x",
            liquidity_role=LiquidityRole.TAKER,
        )
    with pytest.raises(ValueError, match="expires"):
        Signal(
            signal_id="s",
            strategy_id="x",
            strategy_version="1",
            asof_ts_ns=10,
            venue=Venue.KALSHI,
            contract_id=INST,
            instrument_id=INST,
            side=Side.BUY,
            fair_probability=D("0.6"),
            executable_price=D("0.5"),
            gross_edge=D("0.1"),
            expected_cost=D(0),
            net_edge=D("0.1"),
            confidence=1.0,
            expiry_ts_ns=5,
            quantity=D(1),
            limit_price=D("0.5"),
        )


def _order(**kw: object) -> SimOrder:
    base: dict[str, object] = dict(
        order_id="o1",
        venue=Venue.KALSHI,
        contract_id=INST,
        instrument_id=INST,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=D(10),
        tif=TimeInForce.IOC,
        submit_ts_ns=0,
        arrival_ts_ns=0,
        limit_price=D("0.50"),
    )
    base.update(kw)
    return SimOrder(**base)  # type: ignore[arg-type]


def test_sim_order_lifecycle_guards() -> None:
    with pytest.raises(ValueError):
        _order(quantity=D(0))
    with pytest.raises(ValueError, match="limit price"):
        _order(limit_price=None)
    with pytest.raises(ValueError, match="arrive"):
        _order(submit_ts_ns=5, arrival_ts_ns=1)
    o = _order()
    with pytest.raises(ValueError):
        o.record_fill(D(0), D("0.5"), 1)
    with pytest.raises(ValueError, match="overfill"):
        o.record_fill(D(11), D("0.5"), 1)
    with pytest.raises(InvalidPriceError):
        o.record_fill(D(1), D("0.51"), 1)
    sell = _order(side=Side.SELL)
    with pytest.raises(InvalidPriceError):
        sell.record_fill(D(1), D("0.49"), 1)
    o.record_fill(D(10), D("0.5"), 2)
    assert o.state is OrderState.FILLED
    with pytest.raises(ValueError, match="terminal"):
        o.transition(OrderState.CANCELED, 3)


def test_simulator_lifecycle_branches(tmp_path: Path) -> None:
    sim = ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD, queue_model="fifo_proxy")
    with pytest.raises(ValueError):
        ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD, queue_model="magic")
    sim.on_venue_event(snapshot(INST, 1, 0, [("0.48", "40")], [("0.50", "10")]), 0)
    gtd = _order(
        order_id="g",
        tif=TimeInForce.GTD,
        limit_price=D("0.48"),
        arrival_ts_ns=1,
        submit_ts_ns=1,
        gtd_expiry_ts_ns=100,
    )
    sim.submit(gtd)
    with pytest.raises(ValueError, match="duplicate"):
        sim.submit(gtd)
    with pytest.raises(ValueError, match="before its arrival"):
        sim.process_arrival("g", 0)
    sim.process_arrival("g", 1)
    assert sim.process_arrival("g", 2) == []  # already arrived
    # unexplained size decrease improves queue position under fifo_proxy
    sim.on_venue_event(delta(INST, 2, 3, [(BookSide.BID, "0.48", "-20")]), 3)
    assert gtd.queue_ahead is not None and gtd.queue_ahead < D(40)
    assert sim.expire_order("g", 100) and gtd.state is OrderState.EXPIRED
    assert not sim.expire_order("g", 101) and not sim.process_cancel_arrival("g", 101)
    assert sim.request_cancel("g", 102, 1) is None
    # status update closes the venue book -> arrivals rejected
    from cma.domain.models import StatusEvent

    sim.on_venue_event(
        StatusEvent(
            venue=Venue.KALSHI,
            instrument_id=INST,
            source_ts_ns=5,
            recv_ts_ns=5,
            payload_hash="st",
            status=ContractStatus.CLOSED,
        ),
        5,
    )
    late = _order(order_id="late", arrival_ts_ns=6, submit_ts_ns=6)
    sim.submit(late)
    assert sim.process_arrival("late", 6) == [] and late.state is OrderState.REJECTED
    # invalid venue book -> reject
    sim2 = ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD)
    sim2.on_venue_event(snapshot(INST, 1, 0, [("0.48", "40")], [("0.50", "10")]), 0)
    sim2.on_venue_event(delta(INST, 5, 1, [(BookSide.BID, "0.48", "1")]), 1)  # gap
    o = _order(order_id="x", arrival_ts_ns=2, submit_ts_ns=2)
    sim2.submit(o)
    assert sim2.process_arrival("x", 2) == [] and o.reject_reason == "BOOK_INVALID"
    # all-or-none policy: GTC rests whole order instead of partially taking
    sim3 = ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD, allow_partial_fills=False)
    sim3.on_venue_event(snapshot(INST, 1, 0, [("0.48", "40")], [("0.50", "5")]), 0)
    g2 = _order(order_id="aon", tif=TimeInForce.GTC, arrival_ts_ns=1, submit_ts_ns=1)
    sim3.submit(g2)
    assert sim3.process_arrival("aon", 1) == [] and g2.state is OrderState.OPEN
    assert sim3.close_instrument(INST, 9) == ["aon"]
    # market order on a non-binary venue book without price bounds
    sim4 = ExecutionSimulator(
        fee_resolver=lambda _c: ZeroFeeSchedule(),
        binary_price_bounds=False,
        stress_slippage_ticks=1,
    )
    sim4.on_venue_event(snapshot(INST, 1, 0, [], [("0.50", "5")]), 0)
    mo = _order(
        order_id="m",
        order_type=OrderType.MARKET,
        limit_price=None,
        quantity=D(5),
        arrival_ts_ns=1,
        submit_ts_ns=1,
    )
    sim4.submit(mo)
    fills = sim4.process_arrival("m", 1)
    assert fills_total_quantity(fills) == D(5) and fills[0].price == D("0.51")
    # a trade with unknown aggressor at our price does not fill a resting order
    sim5 = ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD)
    sim5.on_venue_event(snapshot(INST, 1, 0, [("0.48", "0")], [("0.50", "10")]), 0)
    rest = _order(
        order_id="r",
        side=Side.SELL,
        tif=TimeInForce.GTC,
        limit_price=D("0.52"),
        arrival_ts_ns=1,
        submit_ts_ns=1,
    )
    sim5.submit(rest)
    sim5.process_arrival("r", 1)
    assert sim5.on_venue_event(trade(INST, 2, "0.52", "100", None), 2) == []
    filled = sim5.on_venue_event(trade(INST, 3, "0.53", "4", None), 3)  # trade-through
    assert fills_total_quantity(filled) == D(4)

    # events_io round trip helpers
    path = tmp_path / "ev.jsonl.gz"
    evs: list[MarketEvent] = [
        trade(INST, 5, "0.5", "1", Side.BUY, tid="b"),
        trade(INST, 1, "0.5", "1", Side.SELL, tid="a"),
    ]
    write_events_jsonl(path, evs)
    back = sort_by_recv(read_events_jsonl(path))
    assert [e.recv_ts_ns for e in back] == [1, 5]
