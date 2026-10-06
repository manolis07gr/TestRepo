"""T011, T015-T025: gates, edge maths, thresholds, TTL, latency, fills, queue, fees."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cma.domain.enums import (
    BookSide,
    ExecutionMode,
    LiquidityRole,
    OrderState,
    OrderType,
    ReasonCode,
    Side,
    TimeInForce,
    Venue,
)
from cma.domain.fees import (
    KALSHI_MAKER_FEE,
    KALSHI_STANDARD,
    POLYMARKET_CRYPTO_PILOT,
    POLYMARKET_CRYPTO_TAKER,
    ScaledFeeSchedule,
    ZeroFeeSchedule,
)
from cma.domain.models import BookLevel, SimOrder
from cma.domain.time import NS_PER_MS
from cma.execution.latency import LatencyModel, LatencyProfile
from cma.execution.simulator.core import ExecutionSimulator
from cma.signals.edge import compute_edge
from cma.signals.engine import FairValueEstimate, GateInputs, SignalEngine, Suppression
from tests.factories import (
    FIXTURES,
    D,
    ScriptedStrategy,
    config,
    delta,
    fixture_events,
    ms,
    order,
    run_replay,
    snapshot,
    trade,
)

pytestmark = pytest.mark.unit
INST = "KALSHI:FIXTURE-BOOK"
ZERO_FEES = ZeroFeeSchedule()


def _sim(**kw: object) -> ExecutionSimulator:
    return ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD, **kw)  # type: ignore[arg-type]


def _order(
    side: Side,
    qty: str,
    limit: str | None,
    *,
    arrival: int,
    tif: TimeInForce = TimeInForce.IOC,
    expiry: int | None = None,
    oid: str = "o1",
) -> SimOrder:
    return SimOrder(
        order_id=oid,
        venue=Venue.KALSHI,
        contract_id=INST,
        instrument_id=INST,
        side=side,
        order_type=OrderType.LIMIT if limit is not None else OrderType.MARKET,
        quantity=D(qty),
        tif=tif,
        submit_ts_ns=arrival,
        arrival_ts_ns=arrival,
        limit_price=None if limit is None else D(limit),
        expiry_ts_ns=expiry,
    )


def _engine(min_bps: str = "100", **sig: object) -> SignalEngine:
    cfg = config(signal={"min_net_edge_bps": min_bps, **sig})
    return SignalEngine(
        config=cfg.signal,
        mode=ExecutionMode.BACKTEST,
        can_trade=lambda _c, _m: True,
        fee_resolver=lambda _c: ZERO_FEES,
    )


def _est(fair: str, ts: int = 100, watermark: int = 100) -> FairValueEstimate:
    return FairValueEstimate(
        strategy_id="s",
        strategy_version="1",
        venue=Venue.KALSHI,
        contract_id=INST,
        instrument_id=INST,
        asof_ts_ns=ts,
        fair_probability=D(fair),
        feature_watermark_ns=watermark,
    )


def _valid_book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
    from cma.ingestion.book import L2BookBuilder

    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    b.apply(snapshot(INST, 1, 0, bids, asks))
    return b.snapshot()


# ---------------------------------------------------------------- T011


def test_T011_stale_reference_suppresses_signal() -> None:
    eng = _engine()
    book = _valid_book([("0.40", "10")], [("0.50", "10")])
    out = eng.evaluate(
        _est("0.90"),
        book,
        decision_ts_ns=100,
        gates=GateInputs(data_fresh=False, freshness_detail="reference age 5s"),
    )
    assert isinstance(out, Suppression) and out.reason is ReasonCode.STALE_DATA


def test_T011_core_gate_uses_max_reference_age() -> None:
    from cma.backtest.core import TradingCore

    events = [snapshot(INST, 1, ms(0), [("0.40", "10")], [("0.50", "10")])]
    core: TradingCore = run_replay(events, [])
    fresh = core._gates(_est("0.9", ts=ms(10), watermark=ms(10)), ms(10))
    stale = core._gates(_est("0.9", ts=ms(2_000), watermark=ms(0)), ms(2_000))
    assert fresh.data_fresh
    assert not stale.data_fresh and "reference age" in stale.freshness_detail


# ---------------------------------------------------------------- T015


def test_T015_gross_and_net_edge_match_hand_computation() -> None:
    eb = compute_edge(
        fair=D("0.60"),
        side=Side.BUY,
        levels=[BookLevel(D("0.50"), D(10)), BookLevel(D("0.52"), D(10))],
        target_quantity=D(15),
        fee_schedule=KALSHI_STANDARD,
        adverse_selection_bps=D(20),
        uncertainty_bps=D(10),
    )
    assert eb is not None
    # fees: ceil(0.07*10*.5*.5)=0.18 ; ceil(0.07*5*.52*.48=0.08736)=0.09 -> 0.27/15
    assert eb.fee == D("0.27") / 15
    assert eb.gross_edge == D("0.10")
    assert eb.vwap == (D("5.00") + D("2.60")) / 15
    assert eb.slippage == eb.vwap - D("0.50")
    assert eb.adverse_selection == D("0.0020") and eb.uncertainty == D("0.0010")
    expected_net = D("0.10") - D("0.27") / 15 - (eb.vwap - D("0.50")) - D("0.003")
    assert eb.net_edge == expected_net
    assert eb.limit_price == D("0.52") and eb.quantity == D(15)


def test_T015_sell_side_edge() -> None:
    eb = compute_edge(
        fair=D("0.40"),
        side=Side.SELL,
        levels=[BookLevel(D("0.45"), D(10))],
        target_quantity=D(10),
        fee_schedule=ZERO_FEES,
    )
    assert eb is not None and eb.gross_edge == D("0.05") and eb.net_edge == D("0.05")


# ---------------------------------------------------------------- T016


def test_T016_threshold_equality_emits_and_below_suppresses() -> None:
    eng = _engine(min_bps="100")
    book = _valid_book([("0.40", "10")], [("0.50", "10")])
    at = eng.evaluate(_est("0.51"), book, decision_ts_ns=100)  # net exactly 100 bps
    assert not isinstance(at, Suppression)
    assert at.net_edge == D("0.01")
    below = eng.evaluate(_est("0.5099"), book, decision_ts_ns=100)  # 99 bps
    assert isinstance(below, Suppression) and below.reason is ReasonCode.BELOW_THRESHOLD
    # deterministic: same inputs -> same signal id
    again = eng.evaluate(_est("0.51"), book, decision_ts_ns=100)
    assert not isinstance(again, Suppression) and again.signal_id == at.signal_id


# ---------------------------------------------------------------- T017


def test_T017_expired_signal_cannot_create_order() -> None:
    eng = _engine(ttl_ms=50)
    book = _valid_book([("0.40", "10")], [("0.50", "10")])
    sig = eng.evaluate(_est("0.60"), book, decision_ts_ns=100)
    assert not isinstance(sig, Suppression)
    assert sig.expiry_ts_ns == 100 + 50 * NS_PER_MS
    events = [snapshot(INST, 1, ms(0), [("0.40", "10")], [("0.50", "10")])]
    core = run_replay(events, [])
    assert core.order_from_signal(sig, sig.expiry_ts_ns + 1) is None
    assert core.records.risk_rejections[ReasonCode.SIGNAL_EXPIRED] == 1


def test_T017_order_arriving_after_ttl_expires_unfilled() -> None:
    sim = _sim()
    sim.on_venue_event(snapshot(INST, 1, 0, [("0.40", "10")], [("0.50", "10")]), 0)
    o = _order(Side.BUY, "5", "0.50", arrival=200, expiry=150)
    sim.submit(o)
    assert sim.process_arrival("o1", 200) == []
    assert o.state is OrderState.EXPIRED and o.filled_quantity == 0


# ---------------------------------------------------------------- T018


def test_T018_arrival_is_decision_plus_compute_plus_outbound() -> None:
    model = LatencyModel(LatencyProfile(outbound_ms=250, compute_ms=7))
    submit, arrival = model.schedule_order(1_000_000)
    assert submit == 1_000_000 + 7 * NS_PER_MS
    assert arrival == 1_000_000 + 257 * NS_PER_MS
    jit1 = LatencyModel(LatencyProfile(outbound_ms=100, jitter="lognormal"), seed=3)
    jit2 = LatencyModel(LatencyProfile(outbound_ms=100, jitter="lognormal"), seed=3)
    assert [jit1.outbound_ns() for _ in range(5)] == [jit2.outbound_ns() for _ in range(5)]


def test_T018_replay_records_arrival_times() -> None:
    events = [
        snapshot(INST, 1, ms(0), [("0.40", "10")], [("0.50", "10")]),
        delta(INST, 2, ms(1), [(BookSide.BID, "0.40", "1")]),
    ]
    strat = ScriptedStrategy([(ms(0), lambda ctx: [order(INST, Side.BUY, "1", "0.50")])])
    core = run_replay(events, [strat], outbound_ms=250, compute_ms=5)
    rec = core.records.orders[0]
    assert rec.submit_ts_ns == rec.decision_ts_ns + 5 * NS_PER_MS
    assert rec.arrival_ts_ns == rec.submit_ts_ns + 250 * NS_PER_MS


# ---------------------------------------------------------------- T019


@pytest.mark.parametrize(("latency_ms", "expected_key"), [(0, "fills_at_0ms"), (100, "fills_at_100ms")])
def test_T019_no_fill_against_liquidity_gone_before_arrival(
    latency_ms: int, expected_key: str
) -> None:
    exp = json.loads((FIXTURES / "latency_disappearing_liquidity.expected.json").read_text())
    events = fixture_events("latency_disappearing_liquidity.jsonl")
    o = exp["order"]
    strat = ScriptedStrategy(
        [(exp["decision_ts_ns"], lambda ctx: [order(INST, Side.BUY, o["quantity"], o["limit"])])]
    )
    core = run_replay(events, [strat], outbound_ms=latency_ms)
    filled = sum((f.quantity for f in core.records.fills), D(0))
    assert filled == D(exp[expected_key])
    for f in core.records.fills:
        assert f.price <= D(o["limit"])


# ---------------------------------------------------------------- T020 / T021 / T022


def _partial_sim() -> ExecutionSimulator:
    sim = _sim()
    for ev in fixture_events("partial_depth.jsonl"):
        sim.on_venue_event(ev, ev.venue_ts_ns)
    return sim


def test_T020_marketable_order_walks_levels_with_vwap() -> None:
    exp = json.loads((FIXTURES / "partial_depth.expected.json").read_text())
    sim = _partial_sim()
    o = _order(Side.BUY, exp["market_order"]["quantity"], None, arrival=10)
    sim.submit(o)
    fills = sim.process_arrival("o1", 10)
    assert [f.price for f in fills] == [D("0.50"), D("0.51"), D("0.53")]
    assert o.filled_quantity == D(exp["market_filled"])
    assert o.avg_fill_price == Decimal(exp["market_vwap"])
    assert all(f.liquidity_role is LiquidityRole.TAKER for f in fills)


def test_T021_insufficient_depth_partial_fill_and_remainder_by_tif() -> None:
    exp = json.loads((FIXTURES / "partial_depth.expected.json").read_text())
    sim = _partial_sim()
    lo = exp["limit_order"]
    ioc = _order(Side.BUY, lo["quantity"], lo["limit"], arrival=10)
    sim.submit(ioc)
    sim.process_arrival("o1", 10)
    assert ioc.filled_quantity == D(exp["limit_filled"])
    assert ioc.avg_fill_price == Decimal(exp["limit_vwap"])
    assert ioc.state is OrderState.CANCELED and ioc.remaining == D(20)

    sim2 = _partial_sim()
    gtc = _order(Side.BUY, lo["quantity"], lo["limit"], arrival=10, tif=TimeInForce.GTC)
    sim2.submit(gtc)
    sim2.process_arrival("o1", 10)
    assert gtc.state is OrderState.PARTIALLY_FILLED and gtc.remaining == D(20)
    assert sim2.resting_orders(INST) == [gtc]

    sim3 = _partial_sim()
    fok = _order(Side.BUY, lo["quantity"], lo["limit"], arrival=10, tif=TimeInForce.FOK)
    sim3.submit(fok)
    assert sim3.process_arrival("o1", 10) == []
    assert fok.state is OrderState.REJECTED and fok.filled_quantity == 0


def test_T022_fill_price_never_violates_limit_even_under_stress_slippage() -> None:
    sim = ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD, stress_slippage_ticks=1)
    for ev in fixture_events("partial_depth.jsonl"):
        sim.on_venue_event(ev, ev.venue_ts_ns)
    o = _order(Side.BUY, "50", "0.51", arrival=10)
    sim.submit(o)
    fills = sim.process_arrival("o1", 10)
    # 0.50 level fills at 0.51 (stressed); 0.51 level would cost 0.52 > limit -> stop
    assert [(f.price, f.quantity) for f in fills] == [(D("0.51"), D(10))]
    sell = _order(Side.SELL, "10", "0.49", arrival=11, oid="o2")
    sim.submit(sell)
    assert sim.process_arrival("o2", 11) == []  # best bid 0.48 < limit 0.49


@settings(max_examples=60, deadline=None)
@given(
    asks=st.lists(
        st.tuples(st.integers(1, 99), st.integers(1, 50)), min_size=1, max_size=6, unique_by=lambda t: t[0]
    ),
    limit=st.integers(1, 99),
    qty=st.integers(1, 200),
    side_buy=st.booleans(),
)
def test_T022_property_fills_respect_limit_and_quantity(
    asks: list[tuple[int, int]], limit: int, qty: int, side_buy: bool
) -> None:
    levels = sorted(asks)
    book_asks = [(str(Decimal(p) / 100), str(q)) for p, q in levels]
    book_bids = [(str(Decimal(p) / 100), str(q)) for p, q in reversed(levels)]
    sim = _sim()
    if side_buy:
        sim.on_venue_event(snapshot(INST, 1, 0, [], book_asks), 0)
    else:
        sim.on_venue_event(snapshot(INST, 1, 0, book_bids, []), 0)
    lim = str(Decimal(limit) / 100)
    o = _order(Side.BUY if side_buy else Side.SELL, str(qty), lim, arrival=1)
    sim.submit(o)
    fills = sim.process_arrival("o1", 1)
    assert o.filled_quantity <= o.quantity
    for f in fills:
        assert (f.price <= D(lim)) if side_buy else (f.price >= D(lim))


# ---------------------------------------------------------------- T023 / T024


def _resting_bid_sim(queue_model: str = "conservative") -> tuple[ExecutionSimulator, SimOrder]:
    sim = ExecutionSimulator(fee_resolver=lambda _c: KALSHI_STANDARD, queue_model=queue_model)
    sim.on_venue_event(snapshot(INST, 1, 0, [("0.48", "40")], [("0.50", "10")]), 0)
    o = _order(Side.BUY, "20", "0.48", arrival=10, tif=TimeInForce.GTC)
    sim.submit(o)
    assert sim.process_arrival("o1", 10) == []
    assert o.state is OrderState.OPEN and o.queue_ahead == D(40)
    return sim, o


def test_T023_touch_alone_never_fills_resting_order() -> None:
    sim, o = _resting_bid_sim()
    # best ask comes down to 0.49 (touching close), bids unchanged: no fill
    sim.on_venue_event(delta(INST, 2, 20, [(BookSide.ASK, "0.49", "5")]), 20)
    assert o.filled_quantity == 0
    # a trade AT our price smaller than the queue ahead: still no fill
    assert sim.on_venue_event(trade(INST, 30, "0.48", "30", Side.SELL), 30) == []
    assert o.queue_ahead == D(10)
    # the next 25 at our price: 10 consumes the queue, 15 fills us (as maker, at our price)
    fills = sim.on_venue_event(trade(INST, 40, "0.48", "25", Side.SELL), 40)
    assert [(f.price, f.quantity, f.liquidity_role) for f in fills] == [
        (D("0.48"), D(15), LiquidityRole.MAKER)
    ]
    # a trade through our price fills the rest
    fills = sim.on_venue_event(trade(INST, 50, "0.47", "100", Side.SELL), 50)
    assert sum(f.quantity for f in fills) == D(5) and o.state is OrderState.FILLED


def test_T023_trade_through_mode_ignores_trades_at_our_price() -> None:
    sim, o = _resting_bid_sim("trade_through")
    assert sim.on_venue_event(trade(INST, 30, "0.48", "500", Side.SELL), 30) == []
    assert o.filled_quantity == 0
    assert sim.on_venue_event(trade(INST, 40, "0.47", "5", Side.SELL), 40)[0].quantity == D(5)


def test_T023_crossing_liquidity_fills_resting_order_at_its_price() -> None:
    sim, o = _resting_bid_sim()
    fills = sim.on_venue_event(delta(INST, 2, 20, [(BookSide.ASK, "0.47", "8")]), 20)
    assert [(f.price, f.quantity) for f in fills] == [(D("0.48"), D(8))]


def test_T024_order_can_fill_while_cancel_is_in_flight() -> None:
    sim, o = _resting_bid_sim()
    cancel_arrival = sim.request_cancel("o1", 100, 100 * NS_PER_MS)
    assert cancel_arrival == 100 + 100 * NS_PER_MS
    assert o.state is OrderState.CANCEL_PENDING
    fills = sim.on_venue_event(trade(INST, 150, "0.47", "7", Side.SELL), 150)  # before arrival
    assert sum(f.quantity for f in fills) == D(7)
    assert sim.process_cancel_arrival("o1", cancel_arrival)
    assert o.state is OrderState.CANCELED and o.filled_quantity == D(7)
    assert sim.on_venue_event(trade(INST, cancel_arrival + 1, "0.40", "50", Side.SELL), 0) == []


def test_T024_replay_cancel_latency_window() -> None:
    events = [
        snapshot(INST, 1, ms(0), [("0.48", "0")], [("0.50", "10")]),
        delta(INST, 2, ms(1), [(BookSide.ASK, "0.50", "1")]),
        trade(INST, ms(150), "0.47", "10", Side.SELL),  # inside cancel window
        delta(INST, 3, ms(400), [(BookSide.ASK, "0.50", "1")]),
    ]

    def place(ctx):  # type: ignore[no-untyped-def]
        return [order(INST, Side.BUY, "10", "0.48", tif=TimeInForce.GTC)]

    def cancel_all(ctx):  # type: ignore[no-untyped-def]
        from tests.factories import cancel

        return [cancel(o.order_id) for o in ctx.working_orders(INST)]

    strat = ScriptedStrategy([(ms(0), place), (ms(1), cancel_all)])
    core = run_replay(events, [strat], outbound_ms=50, cancel_ms=200)
    assert sum(f.quantity for f in core.records.fills) == D(10)


# ---------------------------------------------------------------- T025


def test_T025_fee_schedules_produce_exact_expected_fees() -> None:
    t, m = LiquidityRole.TAKER, LiquidityRole.MAKER
    assert KALSHI_STANDARD.fee(price=D("0.50"), quantity=D(100), role=t) == D("1.75")
    assert KALSHI_STANDARD.fee(price=D("0.37"), quantity=D(10), role=t) == D("0.17")
    assert KALSHI_STANDARD.fee(price=D("0.50"), quantity=D(100), role=m) == D(0)
    assert KALSHI_MAKER_FEE.fee(price=D("0.50"), quantity=D(100), role=m) == D("0.44")
    assert POLYMARKET_CRYPTO_TAKER.fee(price=D("0.50"), quantity=D(100), role=t) == D("1.75")
    assert POLYMARKET_CRYPTO_TAKER.fee(price=D("0.50"), quantity=D(100), role=m) == D(0)
    assert POLYMARKET_CRYPTO_PILOT.fee(price=D("0.50"), quantity=D(100), role=t) == D("0.78125")
    stressed = ScaledFeeSchedule(base=KALSHI_STANDARD, factor=D("1.5"))
    assert stressed.fee(price=D("0.50"), quantity=D(100), role=t) == D("2.63")


def test_T025_order_level_rounding_accumulates_to_single_fill_cost() -> None:
    acc = D(0)
    charged = []
    for _ in range(2):
        c, acc = KALSHI_STANDARD.order_fee_increment(
            acc, price=D("0.50"), quantity=D(5), role=LiquidityRole.TAKER
        )
        charged.append(c)
    assert charged == [D("0.09"), D("0.09")]
    assert sum(charged) == KALSHI_STANDARD.fee(
        price=D("0.50"), quantity=D(10), role=LiquidityRole.TAKER
    )


def test_T025_fee_version_persisted_on_every_fill() -> None:
    sim = _partial_sim()
    o = _order(Side.BUY, "30", "0.51", arrival=10)
    sim.submit(o)
    fills = sim.process_arrival("o1", 10)
    assert fills and all(f.fee_schedule_version == KALSHI_STANDARD.tag for f in fills)
    # 10@0.50 raw .175 + 20@0.51 raw .34986 -> order total ceil(.5249) = .53
    assert sum(f.fee for f in fills) == D("0.53")
