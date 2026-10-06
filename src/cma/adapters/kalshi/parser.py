"""Kalshi adapter: markets, events, series, order books and trades -> canonical types.

The payload formats follow Kalshi's Trade API v2 documentation as of October 2026. REST
covers ``/markets``, ``/events``, ``/series/{ticker}``, ``/markets/{ticker}/orderbook``
and ``/markets/trades``. WebSocket covers the ``orderbook_delta`` (snapshot + delta),
``trade`` and ``ticker`` channels. Everything here was written WITHOUT live access, so
validate it against live payloads. Drift raises
:class:`~cma.adapters.base.MalformedPayloadError`, and the collector quarantines it.

Units (March 2026 change)
    The primary path uses fixed-point strings. Prices are ``*_dollars`` (up to 4
    decimals) and quantities are ``*_fp`` (2 decimals; fractional contracts exist).
    Legacy integer cents and integer counts are a fallback for old recordings only. The
    contract tick is the smallest ``price_ranges`` step; the full ranges go into
    ``settlement_metadata``. The deprecated ``tick_size`` (in cents) is a fallback.

Canonical YES book
    Kalshi publishes two BID ladders, YES bids and NO bids, and no asks. A NO bid at q is
    a YES ask at 1 - q, so the book is ``bids = yes ladder`` and
    ``asks = {1 - q: no ladder}``. Ladders arrive worst-to-best (ascending) and are
    always re-sorted. WebSocket deltas are INCREMENT mode: ``delta_fp`` is the signed
    change in resting size, and a ``no`` delta at q changes the YES ask at 1 - q.

Timestamps
    The source timestamp is ``ts_ms`` (matching-engine time). The fallbacks, in order,
    are ``ts`` (seconds or ISO) and ``sending_ts_ms`` (gateway send time, present from
    October 2026). Send time is never earlier than the event, so it is a conservative
    watermark. If none is present, ``source_ts_ns`` is None and the event is flagged
    ``SOURCE_TS_MISSING``.

Sequencing
    ``seq`` counts per subscription (``sid``), and both restart on every connection.
    Orderbook subscriptions therefore use one market per sid
    (:func:`cma.adapters.kalshi.commands.build_subscriptions`). sid/seq idempotency keys
    are connection-scoped. After a gap, :meth:`KalshiAdapter.resync_messages` requests
    an in-band re-snapshot (``update_subscription`` / ``get_snapshot``).

Trades
    The aggressor is the taker's outcome side, from ``taker_outcome_side`` or else the
    legacy ``taker_side``. "yes" means the taker bought YES: BUY at the YES price. "no"
    means the taker bought NO, i.e. sold YES: SELL at 1 - no_price. Block trades
    (``is_block_trade``) are negotiated off-book and carry no aggressor.
    ``taker_book_side`` is not used: its semantics are not pinned down well enough to
    map onto the canonical YES book.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

from cma.adapters.base import (
    EventFactory,
    MalformedPayloadError,
    as_list,
    as_mapping,
    build_book_side,
    cents_to_probability,
    check_venue,
    complement_probability,
    connection_scoped,
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
from cma.adapters.kalshi.commands import build_get_snapshot_command
from cma.domain.binary import no_levels_to_yes
from cma.domain.enums import BookSide, ContractStatus, DeltaMode, Side, Venue
from cma.domain.models import (
    BookLevel,
    BookSnapshotEvent,
    LevelChange,
    MarketEvent,
    PredictionContract,
    RawMessage,
    TradeEvent,
    instrument_key,
)
from cma.domain.numbers import ONE, ZERO

VENUE: Final = Venue.KALSHI
KALSHI_FEE_SCHEDULE_ID: Final = "kalshi-standard"
DEFAULT_TICK: Final = Decimal("0.01")

REST_ORDERBOOK_STREAM_PREFIX: Final = "rest:orderbook:"
REST_TRADES_STREAM: Final = "rest:trades"
REST_MARKETS_STREAM: Final = "rest:markets"
REST_EVENTS_STREAM: Final = "rest:events"
REST_SERIES_STREAM: Final = "rest:series"

_STATUS: Final[Mapping[str, ContractStatus]] = {
    "unopened": ContractStatus.UNOPENED,
    "initialized": ContractStatus.UNOPENED,
    "open": ContractStatus.OPEN,
    "active": ContractStatus.OPEN,
    "paused": ContractStatus.CLOSED,
    "inactive": ContractStatus.CLOSED,
    "closed": ContractStatus.CLOSED,
    "determined": ContractStatus.CLOSED,
    "disputed": ContractStatus.CLOSED,
    "amended": ContractStatus.CLOSED,
    "settled": ContractStatus.SETTLED,
    "finalized": ContractStatus.SETTLED,
}

# (fee_type, multiplier) -> registered schedule id (cma.domain.fees)
_SERIES_FEES: Final[Mapping[tuple[str, Decimal], str]] = {
    ("quadratic", Decimal(1)): "kalshi-standard",
    ("quadratic_with_maker_fees", Decimal(1)): "kalshi-maker-fee-series",
    ("quadratic", Decimal("0.5")): "kalshi-index-series",
}

_LADDER_KEYS: Final[Mapping[str, tuple[tuple[str, bool], ...]]] = {
    # (field, is_legacy_cents) in preference order
    "yes": (("yes_dollars_fp", False), ("yes_dollars", False), ("yes", True)),
    "no": (("no_dollars_fp", False), ("no_dollars", False), ("no", True)),
}

_CONTROL_TYPES: Final = frozenset(
    {"subscribed", "unsubscribed", "ok", "error", "list_subscriptions", "update_subscription"}
)


def kalshi_instrument(market_ticker: str) -> str:
    """Instrument/contract id of a Kalshi market (canonical YES book)."""
    return instrument_key(VENUE, market_ticker)


def orderbook_stream(market_ticker: str) -> str:
    """Raw-store stream of polled REST orderbook pages (the ticker is not in the body)."""
    return f"{REST_ORDERBOOK_STREAM_PREFIX}{market_ticker}"


def ticker_of_instrument(instrument_id: str) -> str:
    prefix = f"{VENUE.value}:"
    if not instrument_id.startswith(prefix):
        raise ValueError(f"not a Kalshi instrument id: {instrument_id!r}")
    return instrument_id[len(prefix) :]


# --------------------------------------------------------------------------------------
# Field helpers (fixed-point first, legacy fallback)
# --------------------------------------------------------------------------------------


def _price(
    obj: Mapping[str, Any], dollars_key: str, cents_key: str, what: str, *, required: bool = True
) -> Decimal | None:
    value = obj.get(dollars_key)
    if value is not None:
        return to_probability(value, f"{what}.{dollars_key}")
    legacy = obj.get(cents_key)
    if legacy is not None:
        return cents_to_probability(legacy, f"{what}.{cents_key}")
    if required:
        raise MalformedPayloadError(f"{what}: missing {dollars_key!r} (or legacy {cents_key!r})")
    return None


def _quantity(
    obj: Mapping[str, Any],
    fp_key: str,
    legacy_key: str,
    what: str,
    *,
    signed: bool = False,
    positive: bool = False,
) -> Decimal:
    value = obj.get(fp_key)
    key = fp_key
    if value is None:
        value, key = obj.get(legacy_key), legacy_key
    if value is None:
        raise MalformedPayloadError(f"{what}: missing {fp_key!r} (or legacy {legacy_key!r})")
    return to_quantity(value, f"{what}.{key}", signed=signed, positive=positive)


def _ladder(body: Mapping[str, Any], side: str, what: str) -> list[tuple[Decimal, Decimal]]:
    for key, cents in _LADDER_KEYS[side]:
        rows = body.get(key)
        if rows is None:
            continue
        levels = []
        for i, row in enumerate(as_list(rows, f"{what}.{key}")):
            pair = as_list(row, f"{what}.{key}[{i}]")
            if len(pair) != 2:
                raise MalformedPayloadError(f"{what}.{key}[{i}]: expected [price, quantity]")
            label = f"{what}.{key}[{i}]"
            price = (
                cents_to_probability(pair[0], label) if cents else to_probability(pair[0], label)
            )
            levels.append((price, to_quantity(pair[1], label)))
        return levels
    return []  # empty sides may be omitted


def yes_book(
    yes_bids: list[tuple[Decimal, Decimal]], no_bids: list[tuple[Decimal, Decimal]], what: str
) -> tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]]:
    """Canonical YES book from Kalshi's two bid ladders."""
    bids = build_book_side(yes_bids, side=BookSide.BID, what=f"{what} yes bids")
    no_side = build_book_side(no_bids, side=BookSide.BID, what=f"{what} no bids")
    with malformed_guard(VENUE, what):
        asks = no_levels_to_yes(no_side)  # NO bids (desc) -> YES asks (asc)
    return bids, asks


def _source_ts(
    body: Mapping[str, Any], envelope: Mapping[str, Any] | None, what: str
) -> int | None:
    if body.get("ts_ms") is not None:
        return epoch_to_ns(body["ts_ms"], "ms", f"{what}.ts_ms")
    ts = body.get("ts")
    if ts is not None:
        if isinstance(ts, str) and not ts.strip().lstrip("-").isdigit():
            return iso_to_ns(ts, f"{what}.ts")
        return epoch_to_ns(ts, "s", f"{what}.ts")
    for holder in (body, envelope):
        if holder is not None and holder.get("sending_ts_ms") is not None:
            return epoch_to_ns(holder["sending_ts_ms"], "ms", f"{what}.sending_ts_ms")
    return None


def _taker_aggressor(body: Mapping[str, Any], what: str) -> Side | None:
    if opt_bool(body, "is_block_trade", what):
        return None
    raw = body.get("taker_outcome_side")
    if raw is None:
        raw = body.get("taker_side")
    if raw is None:
        return None
    side = str(raw).lower()
    if side == "yes":
        return Side.BUY
    if side == "no":
        return Side.SELL
    raise MalformedPayloadError(f"{what}: unknown taker side {raw!r}")


def _trade(
    body: Mapping[str, Any], factory: EventFactory, *, source_ts_ns: int | None, what: str
) -> TradeEvent:
    trade_id = opt_id(body, "trade_id", what)
    if trade_id is None:
        raise MalformedPayloadError(f"{what}: missing trade_id")
    ticker = opt_str(body, "market_ticker", what) or req_str(body, "ticker", what)
    yes = _price(body, "yes_price_dollars", "yes_price", what, required=False)
    no = _price(body, "no_price_dollars", "no_price", what, required=False)
    if yes is None and no is None:
        raise MalformedPayloadError(f"{what}: trade without yes/no price")
    if yes is not None and no is not None and yes + no != ONE:
        raise MalformedPayloadError(f"{what}: yes price {yes} + no price {no} != 1")
    price = yes if yes is not None else complement_probability(no or ZERO, what)
    return factory.trade(
        instrument_id=kalshi_instrument(ticker),
        source_ts_ns=source_ts_ns,
        price=price,
        size=_quantity(body, "count_fp", "count", what, positive=True),
        aggressor_side=_taker_aggressor(body, what),
        trade_id=trade_id,
    )


# --------------------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------------------


def parse_rest_orderbook(data: Any, *, ticker: str, factory: EventFactory) -> BookSnapshotEvent:
    """``GET /markets/{ticker}/orderbook`` -> YES-book snapshot (no sequence, no source ts)."""
    what = f"Kalshi orderbook {ticker}"
    page = as_mapping(data, what)
    book = page.get("orderbook_fp")
    if book is None:
        book = page.get("orderbook")
    if book is None:
        raise MalformedPayloadError(f"{what}: missing 'orderbook_fp'")
    body = as_mapping(book, what)
    bids, asks = yes_book(_ladder(body, "yes", what), _ladder(body, "no", what), what)
    return factory.snapshot(
        instrument_id=kalshi_instrument(ticker), source_ts_ns=None, bids=bids, asks=asks
    )


def _rest_trade_ts(body: Mapping[str, Any], what: str) -> int | None:
    if body.get("created_time") is not None:
        return iso_to_ns(body["created_time"], f"{what}.created_time")
    return _source_ts(body, None, what)


def parse_trades_page(data: Any, *, factory: EventFactory) -> tuple[list[TradeEvent], str | None]:
    """``GET /markets/trades`` page -> (trades, next cursor or None)."""
    page = as_mapping(data, "Kalshi trades page")
    rows = page.get("trades")
    trades = []
    for i, row in enumerate(as_list(rows if rows is not None else [], "Kalshi trades")):
        what = f"Kalshi trades[{i}]"
        body = as_mapping(row, what)
        trades.append(_trade(body, factory, source_ts_ns=_rest_trade_ts(body, what), what=what))
    return trades, opt_str(page, "cursor", "Kalshi trades page")


@dataclass(frozen=True, slots=True, kw_only=True)
class KalshiTicker:
    """Top-of-book summary from the ``ticker`` channel (no depth; informational only)."""

    market_ticker: str
    last_price: Decimal | None
    yes_bid: Decimal | None
    yes_ask: Decimal | None
    volume: Decimal | None
    open_interest: Decimal | None
    source_ts_ns: int | None


def parse_ticker_message(msg: Mapping[str, Any]) -> KalshiTicker:
    what = "Kalshi ticker"
    body = as_mapping(require(msg, "msg", what), what)

    def qty(fp_key: str, legacy_key: str) -> Decimal | None:
        value = body.get(fp_key, body.get(legacy_key))
        return None if value is None else to_quantity(value, f"{what}.{fp_key}")

    return KalshiTicker(
        market_ticker=req_str(body, "market_ticker", what),
        last_price=_price(body, "price_dollars", "price", what, required=False),
        yes_bid=_price(body, "yes_bid_dollars", "yes_bid", what, required=False),
        yes_ask=_price(body, "yes_ask_dollars", "yes_ask", what, required=False),
        volume=qty("volume_fp", "volume"),
        open_interest=qty("open_interest_fp", "open_interest"),
        source_ts_ns=_source_ts(body, msg, what),
    )


class KalshiAdapter:
    """:class:`~cma.adapters.base.VenueAdapter` for Kalshi WS frames and REST pages.

    Streams: ``ws*`` carries WebSocket frames. ``rest:orderbook:<ticker>`` carries polled
    orderbook pages, and ``rest:trades`` carries trade pages. Other ``rest:*`` streams
    (markets, events, series) are listing metadata and produce no market events; see
    :func:`parse_markets_page`.
    """

    @property
    def venue(self) -> Venue:
        return VENUE

    def parse(self, raw: RawMessage) -> list[MarketEvent]:
        check_venue(raw, VENUE)
        factory = EventFactory.for_raw(raw)
        stream = raw.stream
        with malformed_guard(VENUE, stream):
            if stream.startswith(REST_ORDERBOOK_STREAM_PREFIX):
                ticker = stream[len(REST_ORDERBOOK_STREAM_PREFIX) :]
                if not ticker:
                    raise MalformedPayloadError(f"orderbook stream without ticker: {stream!r}")
                return [
                    parse_rest_orderbook(load_json(raw.payload), ticker=ticker, factory=factory)
                ]
            if stream == REST_TRADES_STREAM:
                trades, _ = parse_trades_page(load_json(raw.payload), factory=factory)
                return list(trades)
            if stream.startswith("rest:"):
                return []
            return self._parse_ws(load_json(raw.payload), factory)

    def _parse_ws(self, data: Any, factory: EventFactory) -> list[MarketEvent]:
        msg = as_mapping(data, "Kalshi WS message")
        mtype = msg.get("type")
        if not isinstance(mtype, str):
            raise MalformedPayloadError("Kalshi WS message without 'type'")
        if mtype == "orderbook_snapshot":
            return [self._snapshot(msg, factory)]
        if mtype == "orderbook_delta":
            return [self._delta(msg, factory)]
        if mtype == "trade":
            what = "Kalshi trade"
            body = as_mapping(require(msg, "msg", what), what)
            return [_trade(body, factory, source_ts_ns=_source_ts(body, msg, what), what=what)]
        if mtype in ("ticker", "ticker_v2"):
            parse_ticker_message(msg)  # validated and recorded raw; no depth -> no book event
            return []
        if mtype in _CONTROL_TYPES:
            return []
        return []  # unknown message types are ignored (forward compatible)

    @staticmethod
    def _snapshot(msg: Mapping[str, Any], factory: EventFactory) -> BookSnapshotEvent:
        what = "Kalshi orderbook_snapshot"
        body = as_mapping(require(msg, "msg", what), what)
        ticker = req_str(body, "market_ticker", what)
        seq = msg.get("seq")
        bids, asks = yes_book(_ladder(body, "yes", what), _ladder(body, "no", what), what)
        return factory.snapshot(
            instrument_id=kalshi_instrument(ticker),
            source_ts_ns=_source_ts(body, msg, what),
            bids=bids,
            asks=asks,
            sequence=None if seq is None else to_int(seq, f"{what}.seq"),
        )

    @staticmethod
    def _delta(msg: Mapping[str, Any], factory: EventFactory) -> MarketEvent:
        what = "Kalshi orderbook_delta"
        body = as_mapping(require(msg, "msg", what), what)
        ticker = req_str(body, "market_ticker", what)
        side = req_str(body, "side", what).lower()
        price = _price(body, "price_dollars", "price", what)
        assert price is not None
        delta = _quantity(body, "delta_fp", "delta", what, signed=True)
        if side == "yes":
            change = LevelChange(BookSide.BID, price, delta)
        elif side == "no":
            change = LevelChange(BookSide.ASK, complement_probability(price, what), delta)
        else:
            raise MalformedPayloadError(f"{what}: unknown side {side!r}")
        return factory.delta(
            instrument_id=kalshi_instrument(ticker),
            source_ts_ns=_source_ts(body, msg, what),
            changes=(change,),
            mode=DeltaMode.INCREMENT,
            sequence=to_int(require(msg, "seq", what), f"{what}.seq"),
        )

    def idempotency_key(self, stream: str, payload: str) -> str | None:
        if stream.startswith("rest:"):
            return None
        return safe_idempotency_key(lambda: _ws_key(payload))

    def affected_instruments(self, raw: RawMessage) -> list[str] | None:
        """Book attribution for quarantined payloads (see ``SupportsAttribution``)."""
        if raw.stream.startswith(REST_ORDERBOOK_STREAM_PREFIX):
            return [kalshi_instrument(raw.stream[len(REST_ORDERBOOK_STREAM_PREFIX) :])]
        if raw.stream.startswith("rest:"):
            return []
        try:
            msg = load_json(raw.payload)
        except MalformedPayloadError:
            return None
        if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
            return None
        if msg["type"] not in ("orderbook_snapshot", "orderbook_delta"):
            return []
        body = msg.get("msg")
        ticker = body.get("market_ticker") if isinstance(body, dict) else None
        return [kalshi_instrument(ticker)] if isinstance(ticker, str) and ticker else None

    def sequence_scope(self, raw: RawMessage) -> tuple[str, int] | None:
        """Orderbook ``seq`` counts per subscription (``sid``) on one connection.

        Kalshi folds every market subscribed on the orderbook channel into one
        subscription (later subscribe commands are answered ``ok`` and join it), so the
        scope is ``(connection, sid)``, not the market (see ``SupportsSequenceScope``).
        """
        if not raw.stream.startswith("ws"):
            return None
        sid = _SID_RE.search(raw.payload)
        seq = _SEQ_RE.search(raw.payload)
        if sid is None or seq is None:
            return None
        return f"{raw.connection_id}|{sid.group(1)}", int(seq.group(1))

    def resync_messages(self, instrument_id: str, raw: RawMessage, request_id: int) -> list[str]:
        """``update_subscription``/``get_snapshot`` for the gapped market's subscription."""

        def build() -> str | None:
            msg = load_json(raw.payload)
            if not isinstance(msg, dict) or msg.get("sid") is None:
                return None
            sid = to_int(msg["sid"], "sid")
            return build_get_snapshot_command(
                request_id, sid, [ticker_of_instrument(instrument_id)]
            )

        frame = safe_idempotency_key(build)
        return [] if frame is None else [frame]


_SID_RE: Final = re.compile(r'"sid"\s*:\s*(\d+)')
_SEQ_RE: Final = re.compile(r'"seq"\s*:\s*(\d+)')


def _ws_key(payload: str) -> str | None:
    msg = load_json(payload)
    if not isinstance(msg, dict):
        return None
    mtype = msg.get("type")
    if mtype in ("orderbook_snapshot", "orderbook_delta"):
        if msg.get("sid") is None or msg.get("seq") is None:
            return None
        sid, seq = to_int(msg["sid"], "sid"), to_int(msg["seq"], "seq")
        return connection_scoped(f"ob|{sid}|{seq}")  # sid/seq restart per connection
    if mtype == "trade":
        body = msg.get("msg")
        if isinstance(body, dict):
            trade_id = opt_id(body, "trade_id", "trade")
            if trade_id is not None:
                return f"trade|{trade_id}"
    return None


# --------------------------------------------------------------------------------------
# Listing metadata: markets, events, series
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class KalshiSeries:
    """Series metadata relevant to contracts: fees and settlement sources."""

    ticker: str
    title: str
    category: str | None
    frequency: str | None
    fee_type: str | None
    fee_multiplier: Decimal | None
    fee_schedule_id: str
    fee_schedule_mapped: bool
    settlement_sources: tuple[Mapping[str, str], ...]
    contract_url: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class KalshiEvent:
    event_ticker: str
    series_ticker: str | None
    title: str
    sub_title: str | None
    mutually_exclusive: bool | None
    category: str | None
    markets: tuple[PredictionContract, ...]


def fee_schedule_for_series(
    fee_type: str | None, fee_multiplier: Decimal | None
) -> tuple[str, bool]:
    """(registered fee schedule id, whether the series' fee terms mapped exactly).

    An unmapped combination falls back to ``kalshi-standard`` and is flagged. Downstream
    fee-sensitive research should refuse flagged contracts until a matching schedule is
    registered in :mod:`cma.domain.fees`.
    """
    if fee_type is None:
        return KALSHI_FEE_SCHEDULE_ID, False
    key = (fee_type.lower(), fee_multiplier if fee_multiplier is not None else Decimal(1))
    for (ftype, mult), schedule in _SERIES_FEES.items():
        if key[0] == ftype and key[1] == mult:
            return schedule, True
    return KALSHI_FEE_SCHEDULE_ID, False


def parse_series(data: Any) -> KalshiSeries:
    """``GET /series/{ticker}`` (``{"series": {...}}``) or a bare series object."""
    obj = as_mapping(data, "Kalshi series")
    series = as_mapping(obj["series"], "Kalshi series") if "series" in obj else obj
    what = "Kalshi series"
    with malformed_guard(VENUE, what):
        multiplier = series.get("fee_multiplier")
        fee_multiplier = (
            None if multiplier is None else to_decimal(multiplier, f"{what}.fee_multiplier")
        )
        fee_type = opt_str(series, "fee_type", what)
        schedule, mapped = fee_schedule_for_series(fee_type, fee_multiplier)
        sources = []
        for i, src in enumerate(as_list(series.get("settlement_sources") or [], what)):
            entry = as_mapping(src, f"{what}.settlement_sources[{i}]")
            sources.append({str(k): str(v) for k, v in entry.items() if v is not None})
        return KalshiSeries(
            ticker=req_str(series, "ticker", what),
            title=opt_str(series, "title", what) or "",
            category=opt_str(series, "category", what),
            frequency=opt_str(series, "frequency", what),
            fee_type=fee_type,
            fee_multiplier=fee_multiplier,
            fee_schedule_id=schedule,
            fee_schedule_mapped=mapped,
            settlement_sources=tuple(sources),
            contract_url=opt_str(series, "contract_url", what),
        )


def _tick_and_ranges(market: Mapping[str, Any], what: str) -> tuple[Decimal, list[dict[str, str]]]:
    ranges_raw = market.get("price_ranges")
    if ranges_raw:
        ranges = []
        for i, item in enumerate(as_list(ranges_raw, f"{what}.price_ranges")):
            entry = as_mapping(item, f"{what}.price_ranges[{i}]")
            start = to_decimal(require(entry, "start", what), f"{what}.price_ranges.start")
            end = to_decimal(require(entry, "end", what), f"{what}.price_ranges.end")
            step = to_decimal(require(entry, "step", what), f"{what}.price_ranges.step")
            if step <= ZERO or start < ZERO or end > ONE or start >= end:
                raise MalformedPayloadError(f"{what}: invalid price range {entry!r}")
            ranges.append({"start": str(start), "end": str(end), "step": str(step)})
        return min(Decimal(r["step"]) for r in ranges), ranges
    if market.get("tick_size") is not None:  # deprecated: integer cents
        return to_decimal(market["tick_size"], f"{what}.tick_size") / 100, []
    return DEFAULT_TICK, []


_METADATA_FIELDS: Final = (
    "market_type",
    "strike_type",
    "floor_strike",
    "cap_strike",
    "custom_strike",
    "functional_strike",
    "early_close_condition",
    "settlement_timer_seconds",
    "expiration_time",
    "expected_expiration_time",
    "latest_expiration_time",
    "result",
    "settlement_value_dollars",
    "expiration_value",
    "price_level_structure",
    "response_price_units",
    "notional_value_dollars",
    "exchange_index",
)


def parse_market(
    market: Mapping[str, Any],
    *,
    series: KalshiSeries | None = None,
    series_ticker: str | None = None,
    event: Mapping[str, Any] | None = None,
    fee_schedule_id: str | None = None,
) -> PredictionContract:
    """Kalshi market object -> :class:`PredictionContract` (listing-time rule metadata).

    ``series`` adds settlement sources and the series fee schedule. ``event`` (an event
    object listing this market) adds its series ticker and mutual-exclusivity flag. An
    explicit ``fee_schedule_id`` overrides both.
    """
    what = "Kalshi market"
    with malformed_guard(VENUE, what):
        ticker = req_str(market, "ticker", what)
        what = f"Kalshi market {ticker}"
        event_ticker = req_str(market, "event_ticker", what)
        series_id = (
            opt_str(market, "series_ticker", what)
            or (opt_str(event, "series_ticker", what) if event is not None else None)
            or series_ticker
            or (series.ticker if series is not None else None)
            or event_ticker.split("-", 1)[0]
        )
        title = opt_str(market, "title", what) or ticker
        yes_semantics = (
            opt_str(market, "yes_sub_title", what) or opt_str(market, "subtitle", what) or title
        )
        no_sub = opt_str(market, "no_sub_title", what)
        no_semantics = no_sub if no_sub and no_sub != yes_semantics else f"NOT ({yes_semantics})"
        rules = "\n\n".join(
            r
            for r in (
                opt_str(market, "rules_primary", what),
                opt_str(market, "rules_secondary", what),
            )
            if r
        )
        tick, ranges = _tick_and_ranges(market, what)
        status_raw = (opt_str(market, "status", what) or "").lower()

        def ts(key: str) -> int | None:
            value = market.get(key)
            return None if value in (None, "") else iso_to_ns(value, f"{what}.{key}")

        metadata: dict[str, Any] = {"source": "kalshi:market", "status_raw": status_raw}
        for key in _METADATA_FIELDS:
            value = market.get(key)
            if value not in (None, ""):
                metadata[key] = jsonable(value)
        if ranges:
            metadata["price_ranges"] = ranges
        if event is not None:
            mutually_exclusive = opt_bool(event, "mutually_exclusive", what)
            if mutually_exclusive is not None:
                metadata["event_mutually_exclusive"] = mutually_exclusive
            event_title = opt_str(event, "title", what)
            if event_title:
                metadata["event_title"] = event_title
        schedule = KALSHI_FEE_SCHEDULE_ID
        if series is not None:
            schedule = series.fee_schedule_id
            metadata["settlement_sources"] = [dict(s) for s in series.settlement_sources]
            metadata["fee_type"] = series.fee_type
            metadata["fee_multiplier"] = (
                None if series.fee_multiplier is None else str(series.fee_multiplier)
            )
            metadata["fee_schedule_mapped"] = series.fee_schedule_mapped
        return PredictionContract(
            venue=VENUE,
            contract_id=kalshi_instrument(ticker),
            native_id=ticker,
            event_id=event_ticker,
            title=title,
            yes_semantics=yes_semantics,
            no_semantics=no_semantics,
            open_ts_ns=ts("open_time"),
            close_ts_ns=ts("close_time"),
            resolve_ts_ns=ts("expected_expiration_time") or ts("expiration_time"),
            status=_STATUS.get(status_raw, ContractStatus.UNKNOWN),
            tick_size=tick,
            series_id=series_id,
            rules_text=rules,
            can_close_early=bool(opt_bool(market, "can_close_early", what)),
            fee_schedule_id=fee_schedule_id or schedule,
            settlement_metadata=metadata,
            outcome_instruments={"YES": kalshi_instrument(ticker)},
        )


def parse_market_response(data: Any, **kwargs: Any) -> PredictionContract:
    """``GET /markets/{ticker}`` -> contract."""
    obj = as_mapping(data, "Kalshi market response")
    return parse_market(
        as_mapping(require(obj, "market", "Kalshi market response"), "market"), **kwargs
    )


def parse_markets_page(
    data: Any,
    *,
    series: KalshiSeries | None = None,
    series_ticker: str | None = None,
    fee_schedule_id: str | None = None,
) -> tuple[list[PredictionContract], str | None]:
    """``GET /markets`` page -> (contracts, next cursor or None)."""
    page = as_mapping(data, "Kalshi markets page")
    rows = page.get("markets")
    contracts = [
        parse_market(
            as_mapping(m, "Kalshi market"),
            series=series,
            series_ticker=series_ticker,
            fee_schedule_id=fee_schedule_id,
        )
        for m in as_list(rows if rows is not None else [], "Kalshi markets")
    ]
    return contracts, opt_str(page, "cursor", "Kalshi markets page")


def parse_events_page(
    data: Any, *, series: KalshiSeries | None = None
) -> tuple[list[KalshiEvent], str | None]:
    """``GET /events`` page (``with_nested_markets=true`` for contracts) -> events, cursor."""
    page = as_mapping(data, "Kalshi events page")
    events = []
    rows = page.get("events")
    for i, row in enumerate(as_list(rows if rows is not None else [], "Kalshi events")):
        what = f"Kalshi events[{i}]"
        event = as_mapping(row, what)
        markets = tuple(
            parse_market(as_mapping(m, f"{what}.markets"), series=series, event=event)
            for m in as_list(event.get("markets") or [], f"{what}.markets")
        )
        events.append(
            KalshiEvent(
                event_ticker=req_str(event, "event_ticker", what),
                series_ticker=opt_str(event, "series_ticker", what),
                title=opt_str(event, "title", what) or "",
                sub_title=opt_str(event, "sub_title", what),
                mutually_exclusive=opt_bool(event, "mutually_exclusive", what),
                category=opt_str(event, "category", what),
                markets=markets,
            )
        )
    return events, opt_str(page, "cursor", "Kalshi events page")
