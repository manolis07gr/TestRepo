"""Crypto adapters: Coinbase ticker/match, Binance aggTrade/bookTicker, Deribit summaries."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from cma.adapters.base import MalformedPayloadError, load_json
from cma.adapters.crypto import (
    BinanceAdapter,
    CoinbaseAdapter,
    DeribitAdapter,
    DeribitClient,
    parse_book_summary,
    parse_instrument_name,
)
from cma.adapters.crypto.binance import build_subscribe as binance_subscribe
from cma.adapters.crypto.binance import combined_stream_url
from cma.adapters.crypto.coinbase import build_subscribe as coinbase_subscribe
from cma.adapters.crypto.deribit import book_summary_stream
from cma.domain.enums import QualityFlag, Side, Venue
from cma.domain.models import BookSnapshotEvent, TradeEvent
from cma.domain.time import ManualClock, ns_from_iso8601
from cma.ingestion.book import BookManager
from cma.ingestion.rest import HttpFetcher
from tests.unit.adapters.helpers import T0_NS, fixture_text, mock_client, raw_msg

pytestmark = pytest.mark.unit
D = Decimal
COINBASE = CoinbaseAdapter()
BINANCE = BinanceAdapter()


def coinbase(name: str) -> list:  # type: ignore[type-arg]
    return COINBASE.parse(raw_msg(Venue.COINBASE, fixture_text("coinbase", name)))


def binance(name: str) -> list:  # type: ignore[type-arg]
    return BINANCE.parse(raw_msg(Venue.BINANCE, fixture_text("binance", name)))


# ------------------------------------------------------------------ Coinbase


def test_coinbase_ticker_is_a_top_of_book_snapshot_without_trade() -> None:
    (snap,) = coinbase("ws_ticker.json")
    assert isinstance(snap, BookSnapshotEvent)
    assert snap.instrument_id == "COINBASE:BTC-USD"
    assert snap.bids[0].price == D("110502.36")
    assert snap.bids[0].quantity == D("0.46688654")
    assert snap.asks[0].price == D("110502.37")
    assert snap.asks[0].quantity == D("1.56637040")
    assert snap.sequence == 37475248783
    assert snap.source_ts_ns == ns_from_iso8601("2026-10-06T20:30:00.061769Z")
    assert QualityFlag.TOP_OF_BOOK_ONLY in snap.quality_flags


def test_coinbase_match_aggressor_is_opposite_of_maker_side() -> None:
    (trade,) = coinbase("ws_match.json")
    assert isinstance(trade, TradeEvent)
    assert trade.aggressor_side is Side.BUY  # maker sold -> buyer aggressed
    assert trade.trade_id == "370843402"
    assert trade.size == D("0.01500000")
    msg = json.loads(fixture_text("coinbase", "ws_match.json"))
    msg["side"] = "buy"
    (down,) = COINBASE.parse(raw_msg(Venue.COINBASE, json.dumps(msg)))
    assert isinstance(down, TradeEvent)
    assert down.aggressor_side is Side.SELL


def test_coinbase_last_match_shares_trade_identity_with_match() -> None:
    (match,) = coinbase("ws_match.json")
    (last,) = coinbase("ws_last_match.json")
    assert isinstance(match, TradeEvent)
    assert isinstance(last, TradeEvent)
    assert (match.instrument_id, match.trade_id) == (last.instrument_id, last.trade_id)
    key = COINBASE.idempotency_key
    assert key("ws", fixture_text("coinbase", "ws_match.json")) == key(
        "ws", fixture_text("coinbase", "ws_last_match.json")
    )
    assert (
        key("ws", fixture_text("coinbase", "ws_ticker.json")) == "conn:ticker|BTC-USD|37475248783"
    )


def test_coinbase_control_messages_and_bad_payloads() -> None:
    assert coinbase("ws_subscriptions.json") == []
    assert coinbase("ws_heartbeat.json") == []
    msg = json.loads(fixture_text("coinbase", "ws_ticker.json"))
    del msg["best_bid_size"]
    with pytest.raises(MalformedPayloadError, match="best_bid_size"):
        COINBASE.parse(raw_msg(Venue.COINBASE, json.dumps(msg)))
    assert json.loads(coinbase_subscribe(["BTC-USD"])) == {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["ticker", "matches", "heartbeat"],
    }


def test_coinbase_ticker_snapshots_replace_each_other_in_book_manager() -> None:
    books = BookManager()
    (first,) = coinbase("ws_ticker.json")
    msg = json.loads(fixture_text("coinbase", "ws_ticker.json"))
    msg.update(sequence=37475248799, best_bid="110503.00", best_ask="110503.50")
    (second,) = COINBASE.parse(raw_msg(Venue.COINBASE, json.dumps(msg)))
    assert books.apply(first).applied
    assert books.apply(second).applied
    assert not books.apply(first).applied  # older sequence never overwrites newer state
    book = books.get("COINBASE:BTC-USD")
    assert book is not None
    assert book.best_bid() == (D("110503.00"), D("0.46688654"))


# ------------------------------------------------------------------ Binance


def test_binance_aggtrade_combined_stream() -> None:
    (trade,) = binance("ws_aggtrade_combined.json")
    assert isinstance(trade, TradeEvent)
    assert trade.instrument_id == "BINANCE:BTCUSDT"
    assert trade.aggressor_side is Side.SELL  # m=true: buyer is maker -> taker sold
    assert trade.source_ts_ns == T0_NS + 14_000_000  # T (trade time), not E
    assert trade.trade_id == "3170212834"
    assert trade.price == D("110498.12000000")


def test_binance_raw_aggtrade_buyer_taker() -> None:
    (trade,) = binance("ws_aggtrade_raw.json")
    assert isinstance(trade, TradeEvent)
    assert trade.instrument_id == "BINANCE:ETHUSDT"
    assert trade.aggressor_side is Side.BUY


def test_binance_book_ticker_top_of_book_without_source_time() -> None:
    (snap,) = binance("ws_bookticker_combined.json")
    assert isinstance(snap, BookSnapshotEvent)
    assert snap.sequence == 72845125318
    assert snap.source_ts_ns is None
    assert {QualityFlag.SOURCE_TS_MISSING, QualityFlag.TOP_OF_BOOK_ONLY} <= snap.quality_flags
    assert snap.bids[0].price == D("110498.11000000")
    assert snap.asks[0].quantity == D("0.88213000")


def test_binance_control_frames_keys_and_urls() -> None:
    assert binance("ws_subscribe_response.json") == []
    key = BINANCE.idempotency_key
    assert (
        key("ws", fixture_text("binance", "ws_aggtrade_combined.json"))
        == "aggTrade|BTCUSDT|3170212834"
    )
    assert key("ws", fixture_text("binance", "ws_bookticker_combined.json")).startswith("conn:")  # type: ignore[union-attr]
    assert json.loads(binance_subscribe(["BTCUSDT"])) == {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@aggTrade", "btcusdt@bookTicker"],
        "id": 1,
    }
    assert combined_stream_url(["BTCUSDT", "ETHUSDT"]).endswith(
        "?streams=btcusdt@aggTrade/btcusdt@bookTicker/ethusdt@aggTrade/ethusdt@bookTicker"
    )
    bad = json.dumps(
        {
            "stream": "x",
            "data": {"e": "aggTrade", "s": "BTCUSDT", "a": 1, "p": "1", "q": "1", "T": 1},
        }
    )
    with pytest.raises(MalformedPayloadError, match="'m'"):
        BINANCE.parse(raw_msg(Venue.BINANCE, bad))


# ------------------------------------------------------------------ Deribit


def test_deribit_instrument_names() -> None:
    call = parse_instrument_name("BTC-27DEC24-100000-C")
    assert call.underlying == "BTC"
    assert call.strike == D(100000)
    assert call.is_call
    assert call.expiry_ts_ns == ns_from_iso8601("2024-12-27T08:00:00Z")
    put = parse_instrument_name("BTC-5JUL24-90000-P")
    assert not put.is_call
    assert put.expiry_ts_ns == ns_from_iso8601("2024-07-05T08:00:00Z")
    linear = parse_instrument_name("XRP_USDC-30AUG24-0d625-C")
    assert linear.quote_currency == "USDC"
    assert linear.strike == D("0.625")
    for bad in ("BTC-PERPETUAL", "BTC-31FEB24-1-C", "BTC-27XYZ24-100-C"):
        with pytest.raises(MalformedPayloadError):
            parse_instrument_name(bad)


def test_deribit_book_summary_to_option_quotes() -> None:
    quotes = parse_book_summary(load_json(fixture_text("deribit", "book_summary_option.json")))
    assert [q.instrument_name for q in quotes] == [
        "BTC-27DEC24-100000-C",
        "BTC-5JUL24-90000-P",
        "XRP_USDC-30AUG24-0d625-C",
    ]  # the perpetual row is skipped
    call = quotes[0]
    assert call.mark_iv == D("0.5231")  # percent -> fraction
    assert (call.bid, call.ask, call.mid) == (D("0.052"), D("0.053"), D("0.0525"))
    assert call.underlying_price == D("110621.42")
    assert call.quote_currency == "BTC"
    assert call.ts_ns == 1_791_318_600_123_000_000
    assert call.instrument_id == "DERIBIT:BTC-27DEC24-100000-C"
    assert quotes[1].bid is None
    assert quotes[1].mid is None
    assert quotes[2].quote_currency == "USDC"


def test_deribit_error_response_and_adapter() -> None:
    with pytest.raises(MalformedPayloadError, match="too_many_requests"):
        parse_book_summary(load_json(fixture_text("deribit", "error_response.json")))
    adapter = DeribitAdapter()
    raw = raw_msg(
        Venue.DERIBIT,
        fixture_text("deribit", "book_summary_option.json"),
        stream=book_summary_stream("btc"),
    )
    assert adapter.parse(raw) == []
    with pytest.raises(MalformedPayloadError):
        adapter.parse(raw_msg(Venue.DERIBIT, "{bad", stream=book_summary_stream("BTC")))


def test_deribit_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v2/public/get_book_summary_by_currency"
        assert dict(request.url.params) == {"currency": "BTC", "kind": "option"}
        return httpx.Response(200, text=fixture_text("deribit", "book_summary_option.json"))

    async def go() -> int:
        client = DeribitClient(
            HttpFetcher(mock_client(httpx.MockTransport(handler)), clock=ManualClock(T0_NS))
        )
        return len(await client.get_option_quotes("btc"))

    assert asyncio.run(go()) == 3
