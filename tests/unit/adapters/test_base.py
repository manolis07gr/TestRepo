"""Shared adapter helpers: strict JSON/decimal parsing, timestamps, books, keys."""

from __future__ import annotations

from decimal import Decimal

import pytest

from cma.adapters.base import (
    EventFactory,
    MalformedPayloadError,
    SupportsResync,
    VenueAdapter,
    build_book_side,
    cents_to_probability,
    connection_scoped,
    epoch_auto_to_ns,
    epoch_to_ns,
    iso_to_ns,
    load_json,
    malformed_guard,
    resolve_idempotency_key,
    to_decimal,
    to_probability,
)
from cma.adapters.crypto import BinanceAdapter, CoinbaseAdapter, DeribitAdapter
from cma.adapters.kalshi import KalshiAdapter
from cma.adapters.polymarket import PolymarketAdapter
from cma.domain.enums import BookSide, QualityFlag, Side, Venue
from cma.domain.errors import CMAError

pytestmark = pytest.mark.unit


def test_load_json_parses_numbers_as_exact_decimals() -> None:
    data = load_json('{"p": 0.1, "q": 3, "s": "0.4500"}')
    assert data["p"] == Decimal("0.1")
    assert isinstance(data["p"], Decimal)
    assert data["q"] == 3
    assert isinstance(data["q"], int)


@pytest.mark.parametrize("payload", ["{not json", '{"p": NaN}', '{"p": Infinity}', ""])
def test_load_json_rejects_invalid_and_non_finite(payload: str) -> None:
    with pytest.raises(MalformedPayloadError):
        load_json(payload)


def test_malformed_payload_error_is_a_cma_error() -> None:
    assert issubclass(MalformedPayloadError, CMAError)


def test_to_decimal_refuses_floats_and_bools() -> None:
    with pytest.raises(MalformedPayloadError):
        to_decimal(0.1, "x")
    with pytest.raises(MalformedPayloadError):
        to_decimal(True, "x")
    with pytest.raises(MalformedPayloadError):
        to_decimal("NaN", "x")
    assert to_decimal(" 0.25 ", "x") == Decimal("0.25")


@pytest.mark.parametrize("value", ["1.0001", "-0.01", Decimal("1.5")])
def test_probabilities_outside_unit_interval_are_malformed(value: object) -> None:
    with pytest.raises(MalformedPayloadError, match=r"outside \[0, 1\]"):
        to_probability(value, "price")


def test_cents_conversion_is_exact_and_integral() -> None:
    assert cents_to_probability(45, "p") == Decimal("0.45")
    assert cents_to_probability(Decimal("100"), "p") == Decimal(1)
    with pytest.raises(MalformedPayloadError):
        cents_to_probability(Decimal("45.5"), "p")
    with pytest.raises(MalformedPayloadError):
        cents_to_probability(150, "p")


def test_epoch_conversions() -> None:
    assert epoch_to_ns(1_791_318_600, "s", "t") == 1_791_318_600_000_000_000
    assert epoch_to_ns("1791318600123", "ms", "t") == 1_791_318_600_123_000_000
    assert epoch_auto_to_ns(1_791_318_600, "t") == 1_791_318_600_000_000_000
    assert epoch_auto_to_ns(1_791_318_600_123, "t") == 1_791_318_600_123_000_000
    assert epoch_auto_to_ns(1_791_318_600_123_456, "t") == 1_791_318_600_123_456_000
    with pytest.raises(MalformedPayloadError):
        epoch_to_ns(-1, "s", "t")


def test_iso_timestamps_require_offsets_and_accept_short_offsets() -> None:
    assert iso_to_ns("2026-10-06T20:30:00Z", "t") == 1_791_318_600_000_000_000
    assert iso_to_ns("2026-10-06 22:30:00+02", "t") == 1_791_318_600_000_000_000
    assert iso_to_ns("2026-10-06T20:30:00.123456789Z", "t") == 1_791_318_600_123_456_789
    with pytest.raises(MalformedPayloadError):
        iso_to_ns("2026-10-06T20:30:00", "t")


def test_book_sides_are_sorted_best_first_and_validated() -> None:
    levels = [
        (Decimal("0.43"), Decimal(5)),
        (Decimal("0.45"), Decimal(1)),
        (Decimal("0.44"), Decimal(0)),
    ]
    bids = build_book_side(levels, side=BookSide.BID, what="t")
    asks = build_book_side(levels, side=BookSide.ASK, what="t")
    assert [lvl.price for lvl in bids] == [Decimal("0.45"), Decimal("0.43")]
    assert [lvl.price for lvl in asks] == [Decimal("0.43"), Decimal("0.45")]
    with pytest.raises(MalformedPayloadError, match="duplicate"):
        build_book_side([(Decimal("0.4"), Decimal(1))] * 2, side=BookSide.BID, what="t")
    with pytest.raises(MalformedPayloadError, match="negative"):
        build_book_side([(Decimal("0.4"), Decimal(-1))], side=BookSide.BID, what="t")


def test_connection_scoped_keys_are_namespaced_by_connection() -> None:
    key = connection_scoped("ob|2|7")
    assert resolve_idempotency_key(key, "kalshi:1:1") == "kalshi:1:1|ob|2|7"
    assert resolve_idempotency_key(key, "kalshi:9:2") != resolve_idempotency_key(key, "kalshi:1:1")
    assert resolve_idempotency_key("trade|42", "kalshi:1:1") == "trade|42"
    assert resolve_idempotency_key(None, "c") is None


def test_event_factory_numbers_events_and_flags_missing_source_time() -> None:
    factory = EventFactory(Venue.KALSHI, 10, "abc")
    a = factory.trade(
        instrument_id="KALSHI:X",
        source_ts_ns=None,
        price=Decimal("0.5"),
        size=Decimal(1),
        aggressor_side=Side.BUY,
        trade_id="1",
    )
    b = factory.trade(
        instrument_id="KALSHI:X",
        source_ts_ns=5,
        price=Decimal("0.5"),
        size=Decimal(1),
        aggressor_side=Side.BUY,
        trade_id="2",
    )
    assert (a.sub_index, b.sub_index) == (0, 1)
    assert a.event_id != b.event_id
    assert QualityFlag.SOURCE_TS_MISSING in a.quality_flags
    assert QualityFlag.SOURCE_TS_MISSING not in b.quality_flags
    with pytest.raises(MalformedPayloadError):  # zero size violates the trade invariant
        factory.trade(
            instrument_id="KALSHI:X",
            source_ts_ns=5,
            price=Decimal("0.5"),
            size=Decimal(0),
            aggressor_side=None,
            trade_id="3",
        )


def test_malformed_guard_converts_low_level_errors() -> None:
    with pytest.raises(MalformedPayloadError, match="KeyError"), malformed_guard(Venue.KALSHI, "x"):
        _ = {}["missing"]


def test_every_adapter_satisfies_the_protocol() -> None:
    adapters = [
        KalshiAdapter(),
        PolymarketAdapter(),
        CoinbaseAdapter(),
        BinanceAdapter(),
        DeribitAdapter(),
    ]
    for adapter in adapters:
        assert isinstance(adapter, VenueAdapter)
    assert isinstance(KalshiAdapter(), SupportsResync)
    assert not isinstance(CoinbaseAdapter(), SupportsResync)
    assert {a.venue for a in adapters} == {
        Venue.KALSHI,
        Venue.POLYMARKET,
        Venue.COINBASE,
        Venue.BINANCE,
        Venue.DERIBIT,
    }


def test_idempotency_keys_never_raise_on_garbage() -> None:
    for adapter in (KalshiAdapter(), PolymarketAdapter(), CoinbaseAdapter(), BinanceAdapter()):
        assert adapter.idempotency_key("ws", "{garbage") is None
        assert adapter.idempotency_key("ws", "[1, 2]") is None
