"""Binance spot WebSocket adapter (combined streams ``wss://stream.binance.com:9443/stream``).

The formats follow Binance's spot WebSocket documentation as of October 2026 and were
implemented without live access; validate before relying on them.

* Frames on the combined endpoint arrive wrapped as ``{"stream": ..., "data": {...}}``.
  Raw single-stream frames (no wrapper) are accepted too.
* ``aggTrade`` -> :class:`TradeEvent`. ``T`` (trade time, ms) is the source time, ``a``
  is the trade id, and ``m`` (buyer is maker) gives the aggressor: true means the taker
  sold, so SELL; false means BUY. Plain ``trade`` frames are handled the same way.
* ``bookTicker`` -> top-of-book :class:`BookSnapshotEvent` (``TOP_OF_BOOK_ONLY``) with
  ``u`` (order book update id) as ``sequence``. Spot bookTicker has no timestamp, so
  ``source_ts_ns`` is None (``SOURCE_TS_MISSING``) and only the receive time stamps it.
* Subscription responses ``{"result": ..., "id": ...}`` and error frames produce no
  events.

Instrument ids: ``instrument_key(Venue.BINANCE, "BTCUSDT")``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from cma.adapters.base import (
    EventFactory,
    MalformedPayloadError,
    as_mapping,
    build_book_side,
    check_venue,
    connection_scoped,
    epoch_to_ns,
    load_json,
    malformed_guard,
    opt_bool,
    opt_id,
    req_str,
    require,
    safe_idempotency_key,
    to_decimal,
    to_int,
    to_quantity,
)
from cma.domain.enums import BookSide, QualityFlag, Side, Venue
from cma.domain.models import MarketEvent, RawMessage, instrument_key

VENUE: Final = Venue.BINANCE
BINANCE_WS_URL: Final = "wss://stream.binance.com:9443/stream"
DEFAULT_STREAMS: Final = ("aggTrade", "bookTicker")
_BOOK_TICKER_KEYS: Final = frozenset({"u", "s", "b", "B", "a", "A"})


def binance_instrument(symbol: str) -> str:
    return instrument_key(VENUE, symbol.upper())


def stream_names(symbols: Sequence[str], streams: Sequence[str] = DEFAULT_STREAMS) -> list[str]:
    return [f"{symbol.lower()}@{stream}" for symbol in symbols for stream in streams]


def build_subscribe(
    symbols: Sequence[str], streams: Sequence[str] = DEFAULT_STREAMS, *, request_id: int = 1
) -> str:
    """``{"method": "SUBSCRIBE", "params": ["btcusdt@aggTrade", ...], "id": 1}``."""
    return json.dumps(
        {"method": "SUBSCRIBE", "params": stream_names(symbols, streams), "id": request_id},
        separators=(",", ":"),
    )


def combined_stream_url(
    symbols: Sequence[str], streams: Sequence[str] = DEFAULT_STREAMS, base: str = BINANCE_WS_URL
) -> str:
    return f"{base}?streams={'/'.join(stream_names(symbols, streams))}"


def _unwrap(msg: Mapping[str, Any]) -> Mapping[str, Any]:
    if "data" in msg and "stream" in msg:
        return as_mapping(msg["data"], "Binance combined-stream data")
    return msg


def _price(value: Any, what: str) -> Decimal:
    price = to_decimal(value, what)
    if price <= 0:
        raise MalformedPayloadError(f"{what}: non-positive price {price}")
    return price


class BinanceAdapter:
    @property
    def venue(self) -> Venue:
        return VENUE

    def parse(self, raw: RawMessage) -> list[MarketEvent]:
        check_venue(raw, VENUE)
        factory = EventFactory.for_raw(raw)
        with malformed_guard(VENUE, raw.stream):
            data = _unwrap(as_mapping(load_json(raw.payload), "Binance message"))
            if "result" in data and "id" in data:
                return []  # subscription acknowledgement
            event_type = data.get("e")
            if event_type in ("aggTrade", "trade"):
                return [self._trade(data, factory, aggregated=event_type == "aggTrade")]
            if event_type == "bookTicker" or (
                event_type is None and data.keys() >= _BOOK_TICKER_KEYS
            ):
                return [self._book_ticker(data, factory)]
            return []

    @staticmethod
    def _trade(data: Mapping[str, Any], factory: EventFactory, *, aggregated: bool) -> MarketEvent:
        what = f"Binance {'aggTrade' if aggregated else 'trade'}"
        trade_id = opt_id(data, "a" if aggregated else "t", what)
        if trade_id is None:
            raise MalformedPayloadError(f"{what}: missing trade id")
        buyer_is_maker = opt_bool(data, "m", what)
        if buyer_is_maker is None:
            raise MalformedPayloadError(f"{what}: missing 'm'")
        return factory.trade(
            instrument_id=binance_instrument(req_str(data, "s", what)),
            source_ts_ns=epoch_to_ns(require(data, "T", what), "ms", f"{what}.T"),
            price=_price(require(data, "p", what), f"{what}.p"),
            size=to_quantity(require(data, "q", what), f"{what}.q", positive=True),
            aggressor_side=Side.SELL if buyer_is_maker else Side.BUY,
            trade_id=trade_id,
        )

    @staticmethod
    def _book_ticker(data: Mapping[str, Any], factory: EventFactory) -> MarketEvent:
        what = "Binance bookTicker"
        bid = (
            _price(require(data, "b", what), f"{what}.b"),
            to_quantity(require(data, "B", what), what),
        )
        ask = (
            _price(require(data, "a", what), f"{what}.a"),
            to_quantity(require(data, "A", what), what),
        )
        source_ts = None
        for key in ("T", "E"):  # present on futures bookTicker; absent on spot
            if data.get(key) is not None:
                source_ts = epoch_to_ns(data[key], "ms", f"{what}.{key}")
                break
        return factory.snapshot(
            instrument_id=binance_instrument(req_str(data, "s", what)),
            source_ts_ns=source_ts,
            bids=build_book_side([bid], side=BookSide.BID, what=what),
            asks=build_book_side([ask], side=BookSide.ASK, what=what),
            sequence=to_int(require(data, "u", what), f"{what}.u"),
            flags=(QualityFlag.TOP_OF_BOOK_ONLY,),
        )

    def idempotency_key(self, stream: str, payload: str) -> str | None:
        def compute() -> str | None:
            msg = load_json(payload)
            if not isinstance(msg, dict):
                return None
            data = _unwrap(msg)
            symbol = data.get("s")
            if data.get("e") == "aggTrade" and data.get("a") is not None:
                return f"aggTrade|{symbol}|{data['a']}"
            if data.get("e") == "trade" and data.get("t") is not None:
                return f"trade|{symbol}|{data['t']}"
            if data.get("u") is not None and data.keys() >= _BOOK_TICKER_KEYS:
                return connection_scoped(f"bookTicker|{symbol}|{data['u']}")
            return None

        return safe_idempotency_key(compute)
