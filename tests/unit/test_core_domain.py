"""T001-T005: probability bounds, YES/NO normalization, decimal accounting,
timestamp preservation and the no-lookahead watermark."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cma.domain.binary import complement, from_yes_order, no_levels_to_yes, to_yes_order
from cma.domain.enums import (
    ContractStatus,
    ExecutionMode,
    LiquidityRole,
    Outcome,
    SettlementOutcome,
    Side,
    Venue,
)
from cma.domain.errors import ImmutableTimestampError, InvalidProbabilityError, LookaheadError
from cma.domain.fees import KALSHI_STANDARD
from cma.domain.models import BookLevel, FeatureVector, Fill, Settlement, Signal
from cma.domain.numbers import from_float, to_decimal, validate_probability
from cma.features.state import RefSeries
from cma.features.watermark import assert_watermark, check_feature_vector
from cma.portfolio.ledger import Portfolio
from cma.signals.engine import FairValueEstimate, SignalEngine
from cma.storage.events_io import event_from_dict, event_to_dict
from tests.factories import D, config, snapshot, trade

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- T001


@pytest.mark.parametrize("value", ["0", "1", "0.5", "0.0001", "0.9999", 0, 1])
def test_T001_valid_probabilities_accepted(value: str | int) -> None:
    assert Decimal(0) <= validate_probability(value) <= Decimal(1)


@pytest.mark.parametrize("value", ["-0.0001", "1.0001", "2", "NaN", "Infinity", "abc", 0.5])
def test_T001_invalid_probabilities_rejected(value: object) -> None:
    with pytest.raises(InvalidProbabilityError):
        validate_probability(value)  # type: ignore[arg-type]


@given(st.decimals(allow_nan=False, allow_infinity=False, places=6))
def test_T001_probability_bounds_property(d: Decimal) -> None:
    if Decimal(0) <= d <= Decimal(1):
        assert validate_probability(d) == d
    else:
        with pytest.raises(InvalidProbabilityError):
            validate_probability(d)


def test_T001_domain_objects_reject_out_of_range_probabilities() -> None:
    with pytest.raises(InvalidProbabilityError):
        Settlement(
            venue=Venue.KALSHI,
            contract_id="KALSHI:X",
            outcome=SettlementOutcome.SCALAR,
            yes_value=D("1.5"),
            settled_ts_ns=1,
        )
    with pytest.raises(InvalidProbabilityError):
        complement(D("1.2"))
    with pytest.raises(InvalidProbabilityError):
        KALSHI_STANDARD.fee(price=D("1.01"), quantity=D(1), role=LiquidityRole.TAKER)
    with pytest.raises(InvalidProbabilityError):
        Signal(
            signal_id="s",
            strategy_id="x",
            strategy_version="1",
            asof_ts_ns=1,
            venue=Venue.KALSHI,
            contract_id="KALSHI:X",
            instrument_id="KALSHI:X",
            side=Side.BUY,
            fair_probability=D("1.1"),
            executable_price=D("0.5"),
            gross_edge=D("0.6"),
            expected_cost=D(0),
            net_edge=D("0.6"),
            confidence=1.0,
            expiry_ts_ns=2,
            quantity=D(1),
            limit_price=D("0.5"),
        )


# ---------------------------------------------------------------- T002


def test_T002_yes_no_order_transformation_round_trip() -> None:
    assert to_yes_order(Outcome.NO, Side.BUY, D("0.35")) == (Side.SELL, D("0.65"))
    assert to_yes_order(Outcome.NO, Side.SELL, D("0.35")) == (Side.BUY, D("0.65"))
    assert to_yes_order(Outcome.YES, Side.BUY, D("0.35")) == (Side.BUY, D("0.35"))
    for outcome in Outcome:
        for side in Side:
            yes_side, yes_px = to_yes_order(outcome, side, D("0.27"))
            assert from_yes_order(outcome, yes_side, yes_px) == (side, D("0.27"))


@pytest.mark.parametrize("q", ["0.01", "0.35", "0.5", "0.99"])
@pytest.mark.parametrize("yes_wins", [True, False])
def test_T002_buying_no_has_identical_payoff_to_selling_yes(q: str, yes_wins: bool) -> None:
    qty = D(10)
    side, yes_px = to_yes_order(Outcome.NO, Side.BUY, D(q))
    pf = Portfolio(initial_cash=D(1000))
    pf.apply_fill(_fill("KALSHI:X", side, yes_px, qty))
    pnl = pf.settle(
        Settlement(
            venue=Venue.KALSHI,
            contract_id="KALSHI:X",
            outcome=SettlementOutcome.YES if yes_wins else SettlementOutcome.NO,
            yes_value=D(1) if yes_wins else D(0),
            settled_ts_ns=1,
        )
    )
    direct_no_pnl = -qty * D(q) if yes_wins else qty * (1 - D(q))
    assert pnl == direct_no_pnl
    assert pf.cash - D(1000) == direct_no_pnl


def test_T002_no_book_maps_to_yes_book_on_tick_and_sorted() -> None:
    no_bids = [BookLevel(D("0.60"), D(5)), BookLevel(D("0.58"), D(7))]
    yes_asks = no_levels_to_yes(no_bids)
    assert [lvl.price for lvl in yes_asks] == [D("0.40"), D("0.42")]
    assert all(lvl.price % D("0.01") == 0 for lvl in yes_asks)
    # fees are symmetric, so the transformation preserves fee economics
    for p in ("0.12", "0.4", "0.73"):
        no_fee = KALSHI_STANDARD.fee(price=D(p), quantity=D(37), role=LiquidityRole.TAKER)
        yes_fee = KALSHI_STANDARD.fee(price=1 - D(p), quantity=D(37), role=LiquidityRole.TAKER)
        assert no_fee == yes_fee


# ---------------------------------------------------------------- T003


def _fill(cid: str, side: Side, price: Decimal, qty: Decimal, fee: Decimal = D(0)) -> Fill:
    return Fill(
        fill_id=f"f-{cid}-{side}-{price}-{qty}",
        order_id="o",
        venue=Venue.KALSHI,
        contract_id=cid,
        instrument_id=cid,
        side=side,
        fill_ts_ns=1,
        price=price,
        quantity=qty,
        fee=fee,
        fee_schedule_version=KALSHI_STANDARD.tag,
        liquidity_role=LiquidityRole.TAKER,
    )


def test_T003_repeated_fills_reconcile_exactly_to_the_cent() -> None:
    pf = Portfolio(initial_cash=D("10000.00"))
    expected_cash = D("10000.00")
    expected_fees = D(0)
    for i in range(500):
        side = Side.BUY if i % 3 else Side.SELL
        price = D("0.33") if i % 2 else D("0.67")
        qty = D("0.07") * (i % 5 + 1)
        fee = KALSHI_STANDARD.fee(price=price, quantity=qty, role=LiquidityRole.TAKER)
        pf.apply_fill(_fill("KALSHI:X", side, price, qty, fee))
        signed = qty if side is Side.BUY else -qty
        expected_cash -= signed * price + fee
        expected_fees += fee
    assert pf.cash == expected_cash  # exact, no drift
    assert pf.fees_total == expected_fees
    assert pf.fees_total == pf.fees_total.quantize(D("0.01"))  # every fee is whole cents
    assert abs(pf.reconcile({"KALSHI:X": D("0.5")})) < D("1e-12")


def test_T003_floats_cannot_silently_enter_accounting() -> None:
    assert 0.1 + 0.2 != 0.3  # the drift we are protecting against
    with pytest.raises(TypeError):
        to_decimal(0.1)  # type: ignore[arg-type]
    assert from_float(0.1, D("0.0001")) + from_float(0.2, D("0.0001")) == D("0.3")


# ---------------------------------------------------------------- T004


def test_T004_timestamps_distinct_immutable_and_preserved() -> None:
    ev = trade("KALSHI:X", 1_000, "0.5", "3", Side.BUY, delay_ns=250)
    assert (ev.source_ts_ns, ev.recv_ts_ns, ev.process_ts_ns) == (1_000, 1_250, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.source_ts_ns = 5  # type: ignore[misc]
    stamped = ev.with_process_ts(9_999)
    assert ev.process_ts_ns is None  # original untouched
    assert (stamped.source_ts_ns, stamped.recv_ts_ns, stamped.process_ts_ns) == (
        1_000,
        1_250,
        9_999,
    )
    with pytest.raises(ImmutableTimestampError):
        stamped.with_process_ts(10_000)
    # serialization round trip keeps source/recv distinct
    back = event_from_dict(event_to_dict(ev))
    assert (back.source_ts_ns, back.recv_ts_ns) == (1_000, 1_250)
    assert back.event_id == ev.event_id


def test_T004_venue_time_never_after_receive_time() -> None:
    skewed = snapshot("KALSHI:X", 1, 5_000, [("0.4", "1")], [("0.6", "1")], delay_ns=-100)
    assert skewed.source_ts_ns == 5_000  # raw source timestamp is not corrected...
    assert skewed.venue_ts_ns == skewed.recv_ts_ns == 4_900  # ...but cannot be exploited


# ---------------------------------------------------------------- T005


def test_T005_watermark_after_decision_is_rejected() -> None:
    assert assert_watermark({"btc": 100, "book": 90}, 100) == 100
    with pytest.raises(LookaheadError):
        assert_watermark({"btc": 101, "book": 90}, 100)
    with pytest.raises(LookaheadError):
        assert_watermark({}, 100)
    fv = FeatureVector(
        asof_ts_ns=100,
        contract_id="KALSHI:X",
        mapping_version=1,
        feature_version="f1",
        values={"ret_1s": 0.001},
        source_watermarks={"btc": 150},
    )
    with pytest.raises(LookaheadError):
        check_feature_vector(fv, 100)


def test_T005_signal_engine_refuses_lookahead_estimates() -> None:
    cfg = config()
    eng = SignalEngine(
        config=cfg.signal,
        mode=ExecutionMode.BACKTEST,
        can_trade=lambda _c, _m: True,
        fee_resolver=lambda _c: KALSHI_STANDARD,
    )
    est = FairValueEstimate(
        strategy_id="s",
        strategy_version="1",
        venue=Venue.KALSHI,
        contract_id="KALSHI:X",
        instrument_id="KALSHI:X",
        asof_ts_ns=100,
        fair_probability=D("0.9"),
        feature_watermark_ns=101,
    )
    book = snapshot("KALSHI:X", 1, 0, [("0.4", "10")], [("0.5", "10")])
    from cma.ingestion.book import L2BookBuilder

    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id="KALSHI:X")
    b.apply(book)
    with pytest.raises(LookaheadError):
        eng.evaluate(est, b.snapshot(), decision_ts_ns=100)


def test_T005_reference_series_never_returns_future_points() -> None:
    s = RefSeries()
    for t, p in [(10, 100.0), (20, 101.0), (30, 102.0)]:
        s.append(t, t + 1, p)
    assert s.price_at(25) == (20, 101.0)
    assert s.price_at(9) is None
    assert s.price_at(30) == (30, 102.0)
    assert s.price_at(29, max_age_ns=5) is None  # stale beyond max age, never forward-filled


def test_T004_status_event_round_trip() -> None:
    from cma.domain.models import StatusEvent

    ev = StatusEvent(
        venue=Venue.KALSHI,
        instrument_id="KALSHI:X",
        source_ts_ns=1,
        recv_ts_ns=2,
        payload_hash="h",
        status=ContractStatus.CLOSED,
    )
    assert event_from_dict(event_to_dict(ev)) == ev
