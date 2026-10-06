"""Polymarket adapter: Gamma listings, CLOB books/trades and the market WS channel.

The formats follow Polymarket's documentation as of October 2026: Gamma ``/markets``
and ``/events``, CLOB ``/book`` and ``/prices-history``, Data API ``/v2/trades``, and
the CLOB WebSocket ``market`` channel. Everything here was written WITHOUT live access,
so validate it against live payloads. Drift raises
:class:`~cma.adapters.base.MalformedPayloadError` and is quarantined.

Instruments
    A Polymarket market (``conditionId``) has two outcome TOKENS, and each token has its
    own CLOB book. Books, deltas and trades are keyed by the token instrument
    (``instrument_key(POLYMARKET, asset_id)``), and prices and sides are in that token's
    terms. The YES and NO token books mirror one unified book, because one resting order
    shows up in both. They must never be summed, and this adapter never turns a NO book
    into a YES book. The canonical YES book of a contract is its YES token's book
    (``PredictionContract.outcome_instruments["YES"]``); for Up/Down markets that is the
    "Up" token. Any re-expression in YES terms is left to downstream code.

WebSocket ``market`` channel
    A frame is a single object or a JSON array of objects.

    * ``book`` -> :class:`BookSnapshotEvent`. ``checksum`` is the book ``hash``,
      ``sequence`` is None (Polymarket books have no sequence numbers), and
      ``source_ts`` is ``timestamp`` in ms. Ladders arrive in arbitrary order and are
      re-sorted.
    * ``price_change`` -> :class:`BookDeltaEvent` in ABSOLUTE mode: ``size`` is the new
      total resting size and "0" removes the level. Side BUY maps to BID, SELL to ASK.
      The current format (since 2025-09-15) is ``price_changes: [{asset_id, ...}]``,
      grouped here into one event per asset. The legacy per-asset ``changes`` format is
      still accepted.
    * ``last_trade_price`` -> :class:`TradeEvent`. ``side`` is the aggressor's side on
      that token. The feed has no trade id, so one is synthesized deterministically from
      the token, timestamp, price, size, side and fee rate.
    * ``market_resolved`` -> :class:`StatusEvent` SETTLED, on the contract and on the
      listed tokens.
    * ``tick_size_change`` is a recorded no-op. It is validated, but the tick is listing
      metadata, not book state, and later prices carry the new grid (see
      :func:`parse_tick_size_change`).
    * ``best_bid_ask`` is validated and produces no event. A top-of-book summary must
      never overwrite the L2 book.
    * ``new_market`` produces no event; it is a discovery hint for the collector.
    * The keepalive frames "PING" and "PONG" are ignored, and so are unknown event types.

Fees
    ``fee_schedule_id`` follows ``feesEnabled``/``feeSchedule``:
    ``polymarket-no-fee`` only when fees are disabled; ``polymarket-crypto-taker`` for
    crypto markets; the sports/finance schedule when the category is obvious; otherwise
    the schedule registered with the same (rate, exponent). Anything else gets
    :data:`POLYMARKET_UNMAPPED_FEE_SCHEDULE`, which is deliberately not registered, so fee
    lookups fail closed. The raw ``feeSchedule`` always goes into ``settlement_metadata``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

from cma.adapters.base import (
    EventFactory,
    MalformedPayloadError,
    as_list,
    as_mapping,
    build_book_side,
    check_venue,
    connection_scoped,
    epoch_auto_to_ns,
    epoch_to_ns,
    iso_to_ns,
    jsonable,
    load_json,
    malformed_guard,
    opt_bool,
    opt_id,
    opt_str,
    req_str,
    require,
    safe_idempotency_key,
    to_decimal,
    to_int,
    to_probability,
    to_quantity,
)
from cma.domain.enums import BookSide, ContractStatus, DeltaMode, Side, Venue
from cma.domain.models import (
    BookDeltaEvent,
    BookSnapshotEvent,
    LevelChange,
    MarketEvent,
    PredictionContract,
    RawMessage,
    TradeEvent,
    instrument_key,
    payload_digest,
    stable_id,
)
from cma.domain.numbers import ONE, ZERO

VENUE: Final = Venue.POLYMARKET
POLYMARKET_DEFAULT_FEE_SCHEDULE: Final = "polymarket-no-fee"
POLYMARKET_UNMAPPED_FEE_SCHEDULE: Final = "polymarket-unmapped-fee"
DEFAULT_TICK: Final = Decimal("0.01")

REST_BOOK_STREAM: Final = "rest:book"
REST_TRADES_STREAM: Final = "rest:trades"
REST_PRICES_STREAM: Final = "rest:prices-history"
REST_MARKETS_STREAM: Final = "rest:gamma-markets"
REST_EVENTS_STREAM: Final = "rest:gamma-events"

_KEEPALIVE_FRAMES: Final = frozenset({"PING", "PONG", ""})
_UPDOWN_SLUG_RE = re.compile(
    r"^(?P<coin>[a-z0-9]+)-updown-(?P<interval>\d+[mhd])-(?P<start>\d{9,11})$"
)
_CRYPTO_WORDS: Final = frozenset(
    {"crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "dogecoin", "doge"}
)
_CATEGORY_FEES: Final[Mapping[str, str]] = {
    "crypto": "polymarket-crypto-taker",
    "sports": "polymarket-sports-taker",
    "finance": "polymarket-finance-taker",
    "politics": "polymarket-finance-taker",
}
_RATE_FEES: Final[Mapping[tuple[Decimal, int], str]] = {
    (Decimal("0.07"), 1): "polymarket-crypto-taker",
    (Decimal("0.072"), 1): "polymarket-crypto-taker-2026-03-30",
    (Decimal("0.05"), 1): "polymarket-sports-taker",
    (Decimal("0.04"), 1): "polymarket-finance-taker",
    (Decimal("0.25"), 2): "polymarket-crypto-15m-pilot",
}


def polymarket_instrument(native_id: str) -> str:
    """Instrument id of a Polymarket token (asset id) or contract (condition id)."""
    return instrument_key(VENUE, native_id)


# --------------------------------------------------------------------------------------
# WebSocket / CLOB market data
# --------------------------------------------------------------------------------------


def _book_side(raw_side: Any, what: str) -> BookSide:
    side = str(raw_side).upper()
    if side == "BUY":
        return BookSide.BID
    if side == "SELL":
        return BookSide.ASK
    raise MalformedPayloadError(f"{what}: unknown side {raw_side!r}")


def _trade_side(raw_side: Any, what: str) -> Side | None:
    if raw_side is None or raw_side == "":
        return None
    side = str(raw_side).upper()
    if side == "BUY":
        return Side.BUY
    if side == "SELL":
        return Side.SELL
    raise MalformedPayloadError(f"{what}: unknown side {raw_side!r}")


def _timestamp_ms(obj: Mapping[str, Any], what: str) -> int | None:
    value = obj.get("timestamp")
    return None if value in (None, "") else epoch_to_ns(value, "ms", f"{what}.timestamp")


def _levels(rows: Any, what: str) -> list[tuple[Decimal, Decimal]]:
    out = []
    for i, row in enumerate(as_list(rows, what)):
        level = as_mapping(row, f"{what}[{i}]")
        out.append(
            (
                to_probability(require(level, "price", what), f"{what}[{i}].price"),
                to_quantity(require(level, "size", what), f"{what}[{i}].size"),
            )
        )
    return out


def parse_book(obj: Mapping[str, Any], factory: EventFactory) -> BookSnapshotEvent:
    """WS ``book`` message or CLOB ``GET /book`` response -> token-book snapshot."""
    what = "Polymarket book"
    asset = opt_id(obj, "asset_id", what)
    if asset is None:
        raise MalformedPayloadError(f"{what}: missing asset_id")
    bids_raw = obj.get("bids", obj.get("buys"))
    asks_raw = obj.get("asks", obj.get("sells"))
    if bids_raw is None or asks_raw is None:
        raise MalformedPayloadError(f"{what}: missing bids/asks")
    return factory.snapshot(
        instrument_id=polymarket_instrument(asset),
        source_ts_ns=_timestamp_ms(obj, what),
        bids=build_book_side(_levels(bids_raw, f"{what}.bids"), side=BookSide.BID, what=what),
        asks=build_book_side(_levels(asks_raw, f"{what}.asks"), side=BookSide.ASK, what=what),
        checksum=opt_str(obj, "hash", what),
    )


def _price_change(obj: Mapping[str, Any], factory: EventFactory) -> list[BookDeltaEvent]:
    what = "Polymarket price_change"
    ts = _timestamp_ms(obj, what)
    grouped: dict[str, list[LevelChange]] = {}
    hashes: dict[str, str | None] = {}
    if obj.get("price_changes") is not None:
        for i, row in enumerate(as_list(obj["price_changes"], f"{what}.price_changes")):
            change = as_mapping(row, f"{what}.price_changes[{i}]")
            asset = opt_id(change, "asset_id", what)
            if asset is None:
                raise MalformedPayloadError(f"{what}.price_changes[{i}]: missing asset_id")
            grouped.setdefault(asset, []).append(_level_change(change, f"{what}[{i}]"))
            hashes[asset] = opt_str(change, "hash", what)
    elif obj.get("changes") is not None:  # legacy per-asset format
        asset = opt_id(obj, "asset_id", what)
        if asset is None:
            raise MalformedPayloadError(f"{what}: legacy format without asset_id")
        for i, row in enumerate(as_list(obj["changes"], f"{what}.changes")):
            grouped.setdefault(asset, []).append(
                _level_change(as_mapping(row, f"{what}.changes[{i}]"), f"{what}[{i}]")
            )
        hashes[asset] = opt_str(obj, "hash", what)
    else:
        raise MalformedPayloadError(f"{what}: neither 'price_changes' nor 'changes'")
    return [
        factory.delta(
            instrument_id=polymarket_instrument(asset),
            source_ts_ns=ts,
            changes=tuple(changes),
            mode=DeltaMode.ABSOLUTE,
            checksum=hashes.get(asset),
        )
        for asset, changes in grouped.items()
    ]


def _level_change(change: Mapping[str, Any], what: str) -> LevelChange:
    return LevelChange(
        _book_side(require(change, "side", what), what),
        to_probability(require(change, "price", what), f"{what}.price"),
        to_quantity(require(change, "size", what), f"{what}.size"),
    )


def _ltp_trade_id(obj: Mapping[str, Any]) -> str:
    fields = ("asset_id", "timestamp", "price", "size", "side", "fee_rate_bps", "transaction_hash")
    return stable_id("polymarket-ltp", *(str(obj.get(f, "")) for f in fields))


def _last_trade(obj: Mapping[str, Any], factory: EventFactory) -> TradeEvent:
    what = "Polymarket last_trade_price"
    asset = opt_id(obj, "asset_id", what)
    if asset is None:
        raise MalformedPayloadError(f"{what}: missing asset_id")
    return factory.trade(
        instrument_id=polymarket_instrument(asset),
        source_ts_ns=_timestamp_ms(obj, what),
        price=to_probability(require(obj, "price", what), f"{what}.price"),
        size=to_quantity(require(obj, "size", what), f"{what}.size", positive=True),
        aggressor_side=_trade_side(obj.get("side"), what),
        trade_id=_ltp_trade_id(obj),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TickSizeChange:
    asset_id: str
    market: str | None
    old_tick_size: Decimal | None
    new_tick_size: Decimal
    source_ts_ns: int | None


def parse_tick_size_change(obj: Mapping[str, Any]) -> TickSizeChange:
    what = "Polymarket tick_size_change"
    asset = opt_id(obj, "asset_id", what)
    if asset is None:
        raise MalformedPayloadError(f"{what}: missing asset_id")
    new_tick = to_decimal(require(obj, "new_tick_size", what), f"{what}.new_tick_size")
    if not ZERO < new_tick < ONE:
        raise MalformedPayloadError(f"{what}: invalid tick {new_tick}")
    old = obj.get("old_tick_size")
    return TickSizeChange(
        asset_id=asset,
        market=opt_str(obj, "market", what),
        old_tick_size=None if old in (None, "") else to_decimal(old, f"{what}.old_tick_size"),
        new_tick_size=new_tick,
        source_ts_ns=_timestamp_ms(obj, what),
    )


def _validate_best_bid_ask(obj: Mapping[str, Any]) -> None:
    what = "Polymarket best_bid_ask"
    if opt_id(obj, "asset_id", what) is None:
        raise MalformedPayloadError(f"{what}: missing asset_id")
    for key in ("best_bid", "best_ask"):
        if obj.get(key) not in (None, ""):
            to_probability(obj[key], f"{what}.{key}")


def _market_resolved(obj: Mapping[str, Any], factory: EventFactory) -> list[MarketEvent]:
    what = "Polymarket market_resolved"
    market = opt_id(obj, "market", what)
    if market is None:
        raise MalformedPayloadError(f"{what}: missing market")
    winner_asset = opt_id(obj, "winning_asset_id", what)
    winner = opt_str(obj, "winning_outcome", what)
    detail = f"winning_outcome={winner or ''};winning_asset_id={winner_asset or ''}"
    ts = _timestamp_ms(obj, what)
    instruments = [polymarket_instrument(market)]
    for key in ("assets_ids", "asset_ids"):
        if obj.get(key) is not None:
            instruments += [polymarket_instrument(str(a)) for a in as_list(obj[key], what)]
            break
    return [
        factory.status(
            instrument_id=inst, source_ts_ns=ts, status=ContractStatus.SETTLED, detail=detail
        )
        for inst in instruments
    ]


def _parse_item(obj: Mapping[str, Any], factory: EventFactory) -> list[MarketEvent]:
    etype = obj.get("event_type")
    if not isinstance(etype, str):
        raise MalformedPayloadError("Polymarket message without 'event_type'")
    if etype == "book":
        return [parse_book(obj, factory)]
    if etype == "price_change":
        return list(_price_change(obj, factory))
    if etype == "last_trade_price":
        return [_last_trade(obj, factory)]
    if etype == "tick_size_change":
        parse_tick_size_change(obj)
        return []
    if etype == "best_bid_ask":
        _validate_best_bid_ask(obj)
        return []
    if etype == "market_resolved":
        return _market_resolved(obj, factory)
    return []  # new_market and unknown event types


def _data_trade(row: Mapping[str, Any], factory: EventFactory, what: str) -> TradeEvent:
    def pick(*keys: str) -> Any:
        for key in keys:
            if row.get(key) not in (None, ""):
                return row[key]
        return None

    token = pick("token_id", "asset")
    if token is None:
        raise MalformedPayloadError(f"{what}: missing token_id")
    price = to_probability(pick("price"), f"{what}.price")
    size = to_quantity(pick("size"), f"{what}.size", positive=True)
    ts_raw = pick("timestamp")
    trade_id = stable_id(
        "polymarket-data",
        pick("transaction_hash", "transactionHash") or "",
        token,
        pick("proxy_wallet", "proxyWallet") or "",
        pick("side") or "",
        price,
        size,
        ts_raw if ts_raw is not None else "",
    )
    return factory.trade(
        instrument_id=polymarket_instrument(str(token)),
        source_ts_ns=None if ts_raw is None else epoch_auto_to_ns(ts_raw, f"{what}.timestamp"),
        price=price,
        size=size,
        aggressor_side=None,  # the Data API side is the wallet's side, not the aggressor's
        trade_id=trade_id,
    )


def parse_data_trades(data: Any, factory: EventFactory) -> tuple[list[TradeEvent], str | None]:
    """Data API ``/v2/trades`` page (``{data|trades, pagination.next_cursor}``) or v1 list."""
    if isinstance(data, list):
        rows, cursor = data, None
    else:
        page = as_mapping(data, "Polymarket trades page")
        rows_any = page.get("data", page.get("trades"))
        rows = as_list(rows_any if rows_any is not None else [], "Polymarket trades")
        pagination = page.get("pagination")
        cursor = (
            opt_str(as_mapping(pagination, "pagination"), "next_cursor", "pagination")
            if pagination is not None
            else None
        )
    trades = [
        _data_trade(as_mapping(row, f"Polymarket trades[{i}]"), factory, f"Polymarket trades[{i}]")
        for i, row in enumerate(rows)
    ]
    return trades, cursor


@dataclass(frozen=True, slots=True)
class PricePoint:
    ts_ns: int
    price: Decimal


def parse_prices_history(data: Any) -> list[PricePoint]:
    """CLOB ``/prices-history`` -> points (``t`` seconds, ``p`` probability)."""
    what = "Polymarket prices-history"
    obj = as_mapping(data, what)
    points = []
    for i, row in enumerate(as_list(obj.get("history") or [], what)):
        item = as_mapping(row, f"{what}[{i}]")
        points.append(
            PricePoint(
                ts_ns=epoch_to_ns(require(item, "t", what), "s", f"{what}[{i}].t"),
                price=to_probability(require(item, "p", what), f"{what}[{i}].p"),
            )
        )
    return points


class PolymarketAdapter:
    """:class:`~cma.adapters.base.VenueAdapter` for the CLOB market channel and REST pages.

    Streams: ``ws*`` carries WebSocket frames, ``rest:book`` carries CLOB book responses
    and ``rest:trades`` carries Data API trade pages. Other ``rest:*`` streams (Gamma
    listings, price history) produce no market events.
    """

    @property
    def venue(self) -> Venue:
        return VENUE

    def parse(self, raw: RawMessage) -> list[MarketEvent]:
        check_venue(raw, VENUE)
        factory = EventFactory.for_raw(raw)
        stream = raw.stream
        with malformed_guard(VENUE, stream):
            if stream == REST_BOOK_STREAM:
                return [parse_book(as_mapping(load_json(raw.payload), "book"), factory)]
            if stream == REST_TRADES_STREAM:
                trades, _ = parse_data_trades(load_json(raw.payload), factory)
                return list(trades)
            if stream.startswith("rest:"):
                return []
            if raw.payload.strip().upper() in _KEEPALIVE_FRAMES:
                return []
            data = load_json(raw.payload)
            items = data if isinstance(data, list) else [data]
            events: list[MarketEvent] = []
            for i, item in enumerate(items):
                events.extend(_parse_item(as_mapping(item, f"frame[{i}]"), factory))
            return events

    def idempotency_key(self, stream: str, payload: str) -> str | None:
        if stream.startswith("rest:") or payload.strip().upper() in _KEEPALIVE_FRAMES:
            return None
        return safe_idempotency_key(lambda: _frame_key(payload))

    def affected_instruments(self, raw: RawMessage) -> list[str] | None:
        """Book attribution for quarantined payloads (see ``SupportsAttribution``).

        Polymarket books carry no sequence numbers, so a lost ``book``/``price_change``
        would otherwise go unnoticed.
        """
        if raw.stream.startswith("rest:") and raw.stream != REST_BOOK_STREAM:
            return []
        if raw.payload.strip().upper() in _KEEPALIVE_FRAMES:
            return []
        try:
            data = load_json(raw.payload)
        except MalformedPayloadError:
            return None
        affected: list[str] = []
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict) or not isinstance(item.get("event_type"), str):
                return None
            if item["event_type"] not in ("book", "price_change"):
                continue
            assets = [item.get("asset_id")]
            changes = item.get("price_changes")
            if isinstance(changes, list):
                assets = [c.get("asset_id") if isinstance(c, dict) else None for c in changes]
            if not assets or not all(isinstance(a, str | int) and a != "" for a in assets):
                return None
            affected += [polymarket_instrument(str(a)) for a in assets]
        return list(dict.fromkeys(affected))


def _item_key(obj: Any) -> tuple[str, bool] | None:
    """(key, connection_scoped) for one message object, or None when it has no key."""
    if not isinstance(obj, dict):
        return None
    etype = obj.get("event_type")
    ts = obj.get("timestamp")
    if etype == "book":
        asset, digest = obj.get("asset_id"), obj.get("hash")
        if asset is None or digest is None or ts is None:
            return None
        return f"book|{asset}|{digest}|{ts}", True  # re-sent unchanged after reconnects
    if etype == "price_change":
        if obj.get("price_changes") is not None:
            hashes = ",".join(
                str(c.get("hash")) for c in obj["price_changes"] if isinstance(c, dict)
            )
            return f"pc|{obj.get('market')}|{ts}|{hashes}", False
        return f"pc|{obj.get('asset_id')}|{ts}|{obj.get('hash')}", False
    if etype == "last_trade_price":
        return f"trade|{obj.get('asset_id')}|{_ltp_trade_id(obj)}", False
    if etype == "tick_size_change":
        return f"tick|{obj.get('asset_id')}|{ts}|{obj.get('new_tick_size')}", False
    return None


def _frame_key(payload: str) -> str | None:
    data = load_json(payload)
    items = data if isinstance(data, list) else [data]
    keys: list[str] = []
    scoped = False
    for item in items:
        item_key = _item_key(item)
        if item_key is None:
            return None
        keys.append(item_key[0])
        scoped = scoped or item_key[1]
    if not keys:
        return None
    key = keys[0] if len(keys) == 1 else "multi|" + payload_digest("\n".join(keys))[:32]
    return connection_scoped(key) if scoped else key


# --------------------------------------------------------------------------------------
# Gamma listing metadata
# --------------------------------------------------------------------------------------


def _json_list(value: Any, what: str) -> list[Any]:
    """Gamma encodes ``outcomes``/``outcomePrices``/``clobTokenIds`` as JSON strings."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = load_json(value)
    return as_list(value, what)


def _yes_index(outcomes: Sequence[str], what: str) -> tuple[int, str]:
    lowered = [o.strip().lower() for o in outcomes]
    for label, basis in (("yes", "yes_label"), ("up", "up_label")):
        if label in lowered:
            return lowered.index(label), basis
    return 0, "first_outcome"


def _event(market: Mapping[str, Any]) -> Mapping[str, Any] | None:
    events = market.get("events")
    if isinstance(events, list) and events and isinstance(events[0], Mapping):
        return events[0]
    return None


def _category(
    market: Mapping[str, Any], event: Mapping[str, Any] | None, slug: str | None
) -> str | None:
    candidates: list[str] = []
    for holder in (market, event):
        if holder is None:
            continue
        category = holder.get("category")
        if isinstance(category, str):
            candidates.append(category)
        tags = holder.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, Mapping):
                    candidates += [str(tag.get(k, "")) for k in ("slug", "label")]
                elif isinstance(tag, str):
                    candidates.append(tag)
    words = {c.strip().lower() for c in candidates if c}
    for category in ("crypto", "sports", "finance", "politics"):
        if category in words:
            return category
    if words & _CRYPTO_WORDS:
        return "crypto"
    if slug and _UPDOWN_SLUG_RE.match(slug):
        return "crypto"
    return None


def select_fee_schedule(market: Mapping[str, Any]) -> tuple[str, str]:
    """(fee schedule id, basis) from ``feesEnabled``/``feeSchedule``/category."""
    what = "Polymarket market fees"
    if not opt_bool(market, "feesEnabled", what):
        return POLYMARKET_DEFAULT_FEE_SCHEDULE, "fees_disabled"
    category = _category(market, _event(market), opt_str(market, "slug", what))
    if category == "crypto":
        return _CATEGORY_FEES["crypto"], "category:crypto"
    if category is not None:
        return _CATEGORY_FEES[category], f"category:{category}"
    schedule = market.get("feeSchedule")
    if isinstance(schedule, Mapping) and schedule.get("rate") is not None:
        rate = to_decimal(schedule["rate"], f"{what}.rate")
        exponent = to_int(schedule.get("exponent", 1), f"{what}.exponent")
        for (r, e), schedule_id in _RATE_FEES.items():
            if r == rate and e == exponent:
                return schedule_id, "fee_schedule_rate"
    return POLYMARKET_UNMAPPED_FEE_SCHEDULE, "unmapped"


def _status(market: Mapping[str, Any], what: str) -> ContractStatus:
    if opt_bool(market, "closed", what):
        resolution = (opt_str(market, "umaResolutionStatus", what) or "").lower()
        return ContractStatus.SETTLED if resolution == "resolved" else ContractStatus.CLOSED
    if opt_bool(market, "acceptingOrders", what) is False:
        return ContractStatus.CLOSED
    if opt_bool(market, "active", what):
        return ContractStatus.OPEN
    return ContractStatus.UNKNOWN


def parse_gamma_market(
    market: Mapping[str, Any], *, fee_schedule_id: str | None = None
) -> PredictionContract:
    """Gamma market object -> :class:`PredictionContract` (``contract_id`` = conditionId).

    Up/Down markets have slugs ``{coin}-updown-{5m|15m|4h}-{unix_start}``. For them the
    observation window starts at ``unix_start`` (``settlement_metadata``). It does NOT
    start at Gamma's ``startDate``, which is the object's creation time (~24 h earlier).
    """
    what = "Polymarket market"
    with malformed_guard(VENUE, what):
        condition_id = req_str(market, "conditionId", what)
        what = f"Polymarket market {condition_id}"
        question = opt_str(market, "question", what) or condition_id
        outcomes = [str(o) for o in _json_list(market.get("outcomes"), f"{what}.outcomes")]
        if len(outcomes) != 2:
            raise MalformedPayloadError(f"{what}: expected 2 outcomes, got {len(outcomes)}")
        tokens = [str(t) for t in _json_list(market.get("clobTokenIds"), f"{what}.clobTokenIds")]
        if tokens and len(tokens) != 2:
            raise MalformedPayloadError(f"{what}: expected 2 token ids, got {len(tokens)}")
        yes_idx, yes_basis = _yes_index(outcomes, what)
        no_idx = 1 - yes_idx
        tick_raw = market.get("orderPriceMinTickSize")
        tick = DEFAULT_TICK if tick_raw in (None, "") else to_decimal(tick_raw, f"{what}.tick")
        end_ns = None if market.get("endDate") in (None, "") else iso_to_ns(market["endDate"], what)
        created_ns = (
            None if market.get("startDate") in (None, "") else iso_to_ns(market["startDate"], what)
        )
        accepting = market.get("acceptingOrdersTimestamp")
        open_ns = created_ns if accepting in (None, "") else iso_to_ns(accepting, what)
        slug = opt_str(market, "slug", what)
        event = _event(market)
        if fee_schedule_id is not None:
            schedule, basis = fee_schedule_id, "override"
        else:
            schedule, basis = select_fee_schedule(market)
        metadata: dict[str, Any] = {
            "source": "polymarket:gamma",
            "resolution_source": opt_str(market, "resolutionSource", what) or "",
            "outcome_labels": outcomes,
            "yes_outcome": outcomes[yes_idx],
            "yes_mapping": yes_basis,
            "neg_risk": bool(opt_bool(market, "negRisk", what)),
            "fees_enabled": bool(opt_bool(market, "feesEnabled", what)),
            "fee_schedule_basis": basis,
        }
        optional = {
            "neg_risk_market_id": market.get("negRiskMarketID"),
            "question_id": market.get("questionID"),
            "slug": slug,
            "uma_resolution_status": market.get("umaResolutionStatus"),
            "order_min_size": market.get("orderMinSize"),
            "fee_schedule": market.get("feeSchedule"),
            "game_start_time": market.get("gameStartTime"),
            "group_item_title": market.get("groupItemTitle"),
            "created_ts_ns": created_ns,
            "category": _category(market, event, slug),
        }
        metadata.update({k: jsonable(v) for k, v in optional.items() if v not in (None, "")})
        updown = _UPDOWN_SLUG_RE.match(slug or "")
        if updown is not None:
            metadata["updown"] = {
                "coin": updown.group("coin"),
                "interval": updown.group("interval"),
            }
            metadata["observation_window_start_ns"] = int(updown.group("start")) * 1_000_000_000
            if end_ns is not None:
                metadata["observation_window_end_ns"] = end_ns
        event_id = (
            (opt_id(event, "id", what) if event is not None else None)
            or opt_id(market, "eventId", what)
            or condition_id
        )
        series_id = None
        if event is not None:
            series_id = opt_str(event, "seriesSlug", what) or opt_str(event, "slug", what)
        instruments = (
            {
                "YES": polymarket_instrument(tokens[yes_idx]),
                "NO": polymarket_instrument(tokens[no_idx]),
            }
            if tokens
            else {}
        )
        return PredictionContract(
            venue=VENUE,
            contract_id=polymarket_instrument(condition_id),
            native_id=condition_id,
            event_id=event_id,
            title=question,
            yes_semantics=outcomes[yes_idx],
            no_semantics=outcomes[no_idx],
            open_ts_ns=open_ns,
            close_ts_ns=end_ns,
            resolve_ts_ns=end_ns,
            status=_status(market, what),
            tick_size=tick,
            series_id=series_id,
            rules_text=opt_str(market, "description", what) or "",
            can_close_early=False,
            fee_schedule_id=schedule,
            settlement_metadata=metadata,
            outcome_instruments=instruments,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PolymarketEvent:
    event_id: str
    slug: str | None
    title: str
    neg_risk: bool | None
    markets: tuple[PredictionContract, ...]


def parse_gamma_event(event: Mapping[str, Any]) -> PolymarketEvent:
    what = "Polymarket event"
    event_id = opt_id(event, "id", what)
    if event_id is None:
        raise MalformedPayloadError(f"{what}: missing id")
    markets = []
    for i, row in enumerate(as_list(event.get("markets") or [], f"{what}.markets")):
        market = dict(as_mapping(row, f"{what}.markets[{i}]"))
        market.setdefault("events", [{k: v for k, v in event.items() if k != "markets"}])
        markets.append(parse_gamma_market(market))
    return PolymarketEvent(
        event_id=event_id,
        slug=opt_str(event, "slug", what),
        title=opt_str(event, "title", what) or "",
        neg_risk=opt_bool(event, "negRisk", what),
        markets=tuple(markets),
    )
