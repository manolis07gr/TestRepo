"""Coinbase Exchange WebSocket adapter (``wss://ws-feed.exchange.coinbase.com``).

The public channels used are ``ticker``, ``matches`` and ``heartbeat``; level2, level3
and full require authentication. The formats follow Coinbase's documentation as of
October 2026 and were implemented without live access; validate before relying on them.

* ``ticker`` -> top-of-book :class:`BookSnapshotEvent` flagged ``TOP_OF_BOOK_ONLY``.
  Each ticker is a complete snapshot. ``sequence`` is the product sequence and ``time``
  the source timestamp. A ticker also repeats the last trade, but it never produces a
  trade event; trades come only from ``matches``, so nothing is counted twice.
* ``match`` / ``last_match`` -> :class:`TradeEvent`. ``side`` is the MAKER's side, so the
  aggressor is the opposite: a sell maker means a buy aggressor. ``last_match`` (sent on
  subscribe) shares the ``trade_id`` of the earlier ``match`` and is de-duplicated by
  its idempotency key and by the trade deduplicator.
* ``subscriptions``, ``heartbeat``, ``error`` and unknown types produce no events.

Instrument ids: ``instrument_key(Venue.COINBASE, "BTC-USD")``.
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
    iso_to_ns,
    load_json,
    malformed_guard,
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

VENUE: Final = Venue.COINBASE
COINBASE_WS_URL: Final = "wss://ws-feed.exchange.coinbase.com"
DEFAULT_CHANNELS: Final = ("ticker", "matches", "heartbeat")


def coinbase_instrument(product_id: str) -> str:
    return instrument_key(VENUE, product_id)


def build_subscribe(product_ids: Sequence[str], channels: Sequence[str] = DEFAULT_CHANNELS) -> str:
    """``{"type": "subscribe", "product_ids": [...], "channels": [...]}``."""
    return json.dumps(
        {"type": "subscribe", "product_ids": list(product_ids), "channels": list(channels)},
        separators=(",", ":"),
    )


def _price(value: Any, what: str) -> Decimal:
    price = to_decimal(value, what)
    if price <= 0:
        raise MalformedPayloadError(f"{what}: non-positive price {price}")
    return price


def _top_level(
    obj: Mapping[str, Any], price_key: str, size_key: str, what: str
) -> list[tuple[Decimal, Decimal]]:
    if obj.get(price_key) in (None, ""):
        return []
    price = _price(obj[price_key], f"{what}.{price_key}")
    size = to_quantity(require(obj, size_key, what), f"{what}.{size_key}")
    return [(price, size)]


class CoinbaseAdapter:
    @property
    def venue(self) -> Venue:
        return VENUE

    def parse(self, raw: RawMessage) -> list[MarketEvent]:
        check_venue(raw, VENUE)
        factory = EventFactory.for_raw(raw)
        with malformed_guard(VENUE, raw.stream):
            msg = as_mapping(load_json(raw.payload), "Coinbase message")
            mtype = msg.get("type")
            if not isinstance(mtype, str):
                raise MalformedPayloadError("Coinbase message without 'type'")
            if mtype == "ticker":
                return [self._ticker(msg, factory)]
            if mtype in ("match", "last_match"):
                return [self._match(msg, factory)]
            return []

    @staticmethod
    def _ticker(msg: Mapping[str, Any], factory: EventFactory) -> MarketEvent:
        what = "Coinbase ticker"
        product = req_str(msg, "product_id", what)
        seq = msg.get("sequence")
        return factory.snapshot(
            instrument_id=coinbase_instrument(product),
            source_ts_ns=None if msg.get("time") in (None, "") else iso_to_ns(msg["time"], what),
            bids=build_book_side(
                _top_level(msg, "best_bid", "best_bid_size", what), side=BookSide.BID, what=what
            ),
            asks=build_book_side(
                _top_level(msg, "best_ask", "best_ask_size", what), side=BookSide.ASK, what=what
            ),
            sequence=None if seq is None else to_int(seq, f"{what}.sequence"),
            flags=(QualityFlag.TOP_OF_BOOK_ONLY,),
        )

    @staticmethod
    def _match(msg: Mapping[str, Any], factory: EventFactory) -> MarketEvent:
        what = f"Coinbase {msg.get('type')}"
        product = req_str(msg, "product_id", what)
        trade_id = opt_id(msg, "trade_id", what)
        if trade_id is None:
            raise MalformedPayloadError(f"{what}: missing trade_id")
        maker_side = req_str(msg, "side", what).lower()
        if maker_side not in ("buy", "sell"):
            raise MalformedPayloadError(f"{what}: unknown side {maker_side!r}")
        seq = msg.get("sequence")
        return factory.trade(
            instrument_id=coinbase_instrument(product),
            source_ts_ns=None if msg.get("time") in (None, "") else iso_to_ns(msg["time"], what),
            price=_price(require(msg, "price", what), f"{what}.price"),
            size=to_quantity(require(msg, "size", what), f"{what}.size", positive=True),
            aggressor_side=Side.BUY if maker_side == "sell" else Side.SELL,
            trade_id=trade_id,
            sequence=None if seq is None else to_int(seq, f"{what}.sequence"),
        )

    def idempotency_key(self, stream: str, payload: str) -> str | None:
        def compute() -> str | None:
            msg = load_json(payload)
            if not isinstance(msg, dict):
                return None
            mtype, product = msg.get("type"), msg.get("product_id")
            if mtype == "ticker" and msg.get("sequence") is not None:
                return connection_scoped(f"ticker|{product}|{msg['sequence']}")
            if mtype in ("match", "last_match") and msg.get("trade_id") is not None:
                return f"trade|{product}|{msg['trade_id']}"
            return None

        return safe_idempotency_key(compute)
