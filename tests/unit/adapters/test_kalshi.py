"""Kalshi parsing: contracts, YES-book construction, deltas, trades, keys (fixtures)."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from cma.adapters.base import MalformedPayloadError, load_json
from cma.adapters.kalshi import (
    KalshiAdapter,
    build_get_snapshot_command,
    build_subscriptions,
    fee_schedule_for_series,
    orderbook_stream,
    parse_events_page,
    parse_market_response,
    parse_markets_page,
    parse_series,
    parse_ticker_message,
)
from cma.domain.enums import BookSide, ContractStatus, DeltaMode, QualityFlag, Side, Venue
from cma.domain.models import BookDeltaEvent, BookLevel, BookSnapshotEvent, TradeEvent
from cma.domain.time import ns_from_iso8601
from cma.ingestion.book import BookManager
from tests.unit.adapters.helpers import (
    KALSHI_INST,
    KALSHI_TICKER,
    RECV_NS,
    T0_NS,
    fixture_text,
    raw_msg,
)

pytestmark = pytest.mark.unit
D = Decimal
ADAPTER = KalshiAdapter()


def kalshi(name: str, *, stream: str = "ws") -> list:  # type: ignore[type-arg]
    return ADAPTER.parse(raw_msg(Venue.KALSHI, fixture_text("kalshi", name), stream=stream))


EXPECTED_BIDS = (
    BookLevel(D("0.45"), D("35.50")),
    BookLevel(D("0.44"), D(120)),
    BookLevel(D("0.43"), D(500)),
)
EXPECTED_ASKS = (BookLevel(D("0.47"), D(80)), BookLevel(D("0.49"), D(250)))


# ------------------------------------------------------------------ contracts


def test_market_to_prediction_contract() -> None:
    series = parse_series(load_json(fixture_text("kalshi", "series.json")))
    c = parse_market_response(load_json(fixture_text("kalshi", "market.json")), series=series)
    assert c.venue is Venue.KALSHI
    assert c.contract_id == KALSHI_INST
    assert c.native_id == KALSHI_TICKER
    assert c.event_id == "KXBTCD-26OCT0617"
    assert c.series_id == "KXBTCD"
    assert c.title.startswith("Bitcoin price")
    assert c.yes_semantics == "$111,000 or above"
    assert c.no_semantics == "NOT ($111,000 or above)"  # no_sub_title repeats the strike text
    assert c.open_ts_ns == ns_from_iso8601("2026-10-05T21:00:00Z")
    assert c.close_ts_ns == ns_from_iso8601("2026-10-06T21:00:00Z")
    assert c.resolve_ts_ns == ns_from_iso8601("2026-10-06T21:05:00Z")
    assert c.status is ContractStatus.OPEN
    assert c.tick_size == D("0.001")  # smallest step of the tapered price_ranges grid
    assert c.can_close_early is True
    assert "BRTI" in c.rules_text
    assert "Not all cryptocurrency" in c.rules_text
    assert c.fee_schedule_id == "kalshi-standard"
    meta = c.settlement_metadata
    assert meta["strike_type"] == "greater"
    assert meta["floor_strike"] == "110999.99"  # exact decimal text, never a float
    assert meta["price_ranges"][1] == {"start": "0.1000", "end": "0.9000", "step": "0.0100"}
    assert meta["settlement_sources"][0]["name"] == "CF Benchmarks"
    assert meta["fee_schedule_mapped"] is True
    assert dict(c.outcome_instruments) == {"YES": KALSHI_INST}


def test_market_tick_falls_back_to_deprecated_cents_then_default() -> None:
    market = load_json(fixture_text("kalshi", "market.json"))["market"]
    market.pop("price_ranges")
    assert parse_market_response({"market": market}).tick_size == D("0.01")
    market.pop("tick_size")
    assert parse_market_response({"market": market}).tick_size == D("0.01")
    market["tick_size"] = 5
    assert parse_market_response({"market": market}).tick_size == D("0.05")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("initialized", ContractStatus.UNOPENED),
        ("active", ContractStatus.OPEN),
        ("closed", ContractStatus.CLOSED),
        ("determined", ContractStatus.CLOSED),
        ("finalized", ContractStatus.SETTLED),
        ("settled", ContractStatus.SETTLED),
        ("something-new", ContractStatus.UNKNOWN),
    ],
)
def test_market_status_mapping(status: str, expected: ContractStatus) -> None:
    market = load_json(fixture_text("kalshi", "market.json"))["market"]
    market["status"] = status
    assert parse_market_response({"market": market}).status is expected


def test_markets_page_cursor_and_events_page() -> None:
    contracts, cursor = parse_markets_page(load_json(fixture_text("kalshi", "markets_page_1.json")))
    assert [c.native_id for c in contracts] == [KALSHI_TICKER]
    assert cursor == "CgwI2e3JxgYQ0"
    _, last = parse_markets_page(load_json(fixture_text("kalshi", "markets_page_2.json")))
    assert last is None
    events, _ = parse_events_page(load_json(fixture_text("kalshi", "events_page.json")))
    assert events[0].series_ticker == "KXBTCD"
    assert events[0].mutually_exclusive is False
    assert len(events[0].markets) == 2
    assert events[0].markets[0].settlement_metadata["event_mutually_exclusive"] is False


def test_series_fee_type_maps_to_registered_schedules() -> None:
    assert fee_schedule_for_series("quadratic", D(1)) == ("kalshi-standard", True)
    assert fee_schedule_for_series("quadratic_with_maker_fees", D(1)) == (
        "kalshi-maker-fee-series",
        True,
    )
    assert fee_schedule_for_series("quadratic", D("0.5")) == ("kalshi-index-series", True)
    assert fee_schedule_for_series("quadratic", D(2)) == ("kalshi-standard", False)
    series = parse_series(load_json(fixture_text("kalshi", "series.json")))
    assert series.fee_schedule_id == "kalshi-standard"
    assert series.fee_multiplier == D(1)


def test_market_missing_ticker_is_malformed() -> None:
    market = load_json(fixture_text("kalshi", "market.json"))["market"]
    del market["ticker"]
    with pytest.raises(MalformedPayloadError):
        parse_market_response({"market": market})


# ------------------------------------------------------------------ books


def test_rest_orderbook_fp_builds_canonical_yes_book() -> None:
    (snap,) = kalshi("orderbook_rest_fp.json", stream=orderbook_stream(KALSHI_TICKER))
    assert isinstance(snap, BookSnapshotEvent)
    assert snap.instrument_id == KALSHI_INST
    assert snap.bids == EXPECTED_BIDS  # ascending ladder re-sorted best-first
    assert snap.asks == EXPECTED_ASKS  # NO bid q -> YES ask 1 - q
    assert snap.sequence is None
    assert snap.source_ts_ns is None
    assert QualityFlag.SOURCE_TS_MISSING in snap.quality_flags


def test_rest_orderbook_legacy_cents_fallback() -> None:
    (snap,) = kalshi("orderbook_rest_legacy_cents.json", stream=orderbook_stream(KALSHI_TICKER))
    assert isinstance(snap, BookSnapshotEvent)
    assert [lvl.price for lvl in snap.bids] == [D("0.45"), D("0.44"), D("0.43")]
    assert [lvl.price for lvl in snap.asks] == [D("0.47"), D("0.49")]


def test_ws_snapshot_fp_and_legacy_produce_the_same_book() -> None:
    (new,) = kalshi("ws_orderbook_snapshot.json")
    (old,) = kalshi("ws_orderbook_snapshot_legacy.json")
    assert isinstance(new, BookSnapshotEvent)
    assert isinstance(old, BookSnapshotEvent)
    assert new.sequence == 1
    assert old.sequence == 1
    assert new.bids == EXPECTED_BIDS
    assert new.asks == EXPECTED_ASKS
    assert [lvl.price for lvl in old.bids] == [lvl.price for lvl in new.bids]
    assert new.source_ts_ns == T0_NS + 5_000_000  # sending_ts_ms fallback (no ts_ms)
    assert old.source_ts_ns is None
    assert QualityFlag.SOURCE_TS_MISSING in old.quality_flags
    assert new.recv_ts_ns == RECV_NS


def test_ws_delta_yes_side_changes_yes_bid_incrementally() -> None:
    (delta,) = kalshi("ws_orderbook_delta_yes.json")
    assert isinstance(delta, BookDeltaEvent)
    assert delta.mode is DeltaMode.INCREMENT
    assert delta.sequence == 2
    assert delta.changes[0].side is BookSide.BID
    assert delta.changes[0].price == D("0.45")
    assert delta.changes[0].quantity == D("-10.50")
    assert delta.source_ts_ns == T0_NS + 250_000_000  # ts_ms (matching engine) preferred


def test_ws_delta_no_side_changes_yes_ask_at_complement() -> None:
    (delta,) = kalshi("ws_orderbook_delta_no.json")
    assert isinstance(delta, BookDeltaEvent)
    change = delta.changes[0]
    assert change.side is BookSide.ASK
    assert change.price == D("0.47")
    assert change.quantity == D(20)


def test_snapshot_plus_deltas_reconstruct_expected_book() -> None:
    books = BookManager()
    for name in (
        "ws_orderbook_snapshot.json",
        "ws_orderbook_delta_yes.json",
        "ws_orderbook_delta_no.json",
    ):
        for event in kalshi(name):
            assert books.apply(event).applied
    book = books.get(KALSHI_INST)
    assert book is not None
    assert book.is_valid
    assert book.levels(BookSide.BID)[0] == (D("0.45"), D("25.00"))
    assert book.levels(BookSide.ASK)[0] == (D("0.47"), D("100.00"))


def test_legacy_delta_without_timestamp_is_flagged() -> None:
    (delta,) = kalshi("ws_orderbook_delta_legacy.json")
    assert delta.source_ts_ns is None
    assert QualityFlag.SOURCE_TS_MISSING in delta.quality_flags
    assert isinstance(delta, BookDeltaEvent)
    assert delta.changes[0].quantity == D(-20)


# ------------------------------------------------------------------ trades / ticker


def test_ws_trade_taker_no_is_sell_at_yes_price() -> None:
    (trade,) = kalshi("ws_trade.json")
    assert isinstance(trade, TradeEvent)
    assert trade.instrument_id == KALSHI_INST
    assert trade.price == D("0.46")
    assert trade.size == D("136.00")
    assert trade.aggressor_side is Side.SELL
    assert trade.trade_id == "d91bc706-ee49-470d-82d8-11418bda6fed"
    assert trade.source_ts_ns == T0_NS + 1_123_000_000


def test_ws_trade_legacy_cents_taker_yes_is_buy() -> None:
    (trade,) = kalshi("ws_trade_legacy.json")
    assert isinstance(trade, TradeEvent)
    assert trade.price == D("0.46")
    assert trade.size == D(10)
    assert trade.aggressor_side is Side.BUY
    assert trade.source_ts_ns == T0_NS + 2_000_000_000  # ts in Unix seconds


def test_trade_with_only_no_price_uses_complement_and_block_trades_have_no_aggressor() -> None:
    msg = json.loads(fixture_text("kalshi", "ws_trade.json"))
    del msg["msg"]["yes_price_dollars"]
    msg["msg"]["is_block_trade"] = True
    (trade,) = ADAPTER.parse(raw_msg(Venue.KALSHI, json.dumps(msg)))
    assert isinstance(trade, TradeEvent)
    assert trade.price == D("0.4600")
    assert trade.aggressor_side is None


def test_inconsistent_yes_no_prices_are_malformed() -> None:
    msg = json.loads(fixture_text("kalshi", "ws_trade.json"))
    msg["msg"]["no_price_dollars"] = "0.5000"
    with pytest.raises(MalformedPayloadError, match="!= 1"):
        ADAPTER.parse(raw_msg(Venue.KALSHI, json.dumps(msg)))


def test_rest_trades_page() -> None:
    trades = kalshi("trades_page.json", stream="rest:trades")
    assert [t.trade_id for t in trades] == [
        "d91bc706-ee49-470d-82d8-11418bda6fed",
        "0c7e8f3a-91b2-4d55-8a1e-5c0f2b9d7e61",
    ]
    assert trades[0].source_ts_ns == ns_from_iso8601("2026-10-06T20:30:01.123456Z")
    assert trades[1].size == D("2.50")
    assert trades[1].aggressor_side is Side.BUY
    assert trades[0].sub_index == 0
    assert trades[1].sub_index == 1


def test_ticker_and_control_frames_emit_no_events() -> None:
    assert kalshi("ws_ticker.json") == []
    assert kalshi("ws_subscribed.json") == []
    assert kalshi("ws_error.json") == []
    ticker = parse_ticker_message(json.loads(fixture_text("kalshi", "ws_ticker.json")))
    assert ticker.yes_bid == D("0.45")
    assert ticker.yes_ask == D("0.47")
    assert ticker.volume == D("15234.00")
    assert ticker.source_ts_ns == T0_NS + 2_500_000_000


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m["msg"].update(price_dollars="1.2000"), "outside"),
        (lambda m: m["msg"].update(side="maybe"), "unknown side"),
        (lambda m: m.pop("seq"), "seq"),
        (lambda m: m["msg"].pop("market_ticker"), "market_ticker"),
    ],
)
def test_bad_deltas_are_malformed(mutate: object, match: str) -> None:
    msg = json.loads(fixture_text("kalshi", "ws_orderbook_delta_yes.json"))
    mutate(msg)  # type: ignore[operator]
    with pytest.raises(MalformedPayloadError, match=match):
        ADAPTER.parse(raw_msg(Venue.KALSHI, json.dumps(msg)))


def test_parse_is_deterministic() -> None:
    raw = raw_msg(Venue.KALSHI, fixture_text("kalshi", "ws_orderbook_snapshot.json"))
    first, second = ADAPTER.parse(raw), ADAPTER.parse(raw)
    assert first == second
    assert first[0].event_id == second[0].event_id


def test_foreign_venue_message_is_rejected() -> None:
    with pytest.raises(MalformedPayloadError):
        ADAPTER.parse(raw_msg(Venue.COINBASE, fixture_text("kalshi", "ws_trade.json")))


# ------------------------------------------------------------------ keys / commands


def test_idempotency_keys() -> None:
    assert ADAPTER.idempotency_key("ws", fixture_text("kalshi", "ws_orderbook_delta_yes.json")) == (
        "conn:ob|2|2"
    )
    assert ADAPTER.idempotency_key("ws", fixture_text("kalshi", "ws_trade.json")) == (
        "trade|d91bc706-ee49-470d-82d8-11418bda6fed"
    )
    assert ADAPTER.idempotency_key("ws", fixture_text("kalshi", "ws_ticker.json")) is None
    assert (
        ADAPTER.idempotency_key("rest:trades", fixture_text("kalshi", "trades_page.json")) is None
    )


def test_subscription_frames_use_one_orderbook_sid_per_market() -> None:
    frames = [json.loads(f) for f in build_subscriptions(["A", "B"])]
    assert frames[0] == {
        "id": 1,
        "cmd": "subscribe",
        "params": {"channels": ["orderbook_delta"], "market_tickers": ["A"]},
    }
    assert frames[1]["params"] == {"channels": ["orderbook_delta"], "market_tickers": ["B"]}
    assert frames[2] == {
        "id": 3,
        "cmd": "subscribe",
        "params": {"channels": ["trade"], "market_tickers": ["A", "B"]},
    }


def test_resync_requests_in_band_snapshot_for_the_gapped_subscription() -> None:
    raw = raw_msg(Venue.KALSHI, fixture_text("kalshi", "ws_orderbook_delta_no.json"))
    (frame,) = ADAPTER.resync_messages(KALSHI_INST, raw, 7)
    assert json.loads(frame) == json.loads(build_get_snapshot_command(7, 2, [KALSHI_TICKER]))
    assert json.loads(frame)["params"]["action"] == "get_snapshot"
    assert ADAPTER.resync_messages(KALSHI_INST, raw_msg(Venue.KALSHI, "{oops"), 8) == []
