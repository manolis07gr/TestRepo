"""Canonical data model (scope section 7).

Venue payloads never leak past the adapters: everything downstream speaks these types.

Conventions
-----------
* Timestamps are integer UTC nanoseconds. ``source_ts_ns`` (venue/exchange clock),
  ``recv_ts_ns`` (our receive clock) and ``process_ts_ns`` (our processing clock) are kept
  distinct and are never overwritten.
* Binary prediction contracts are represented by a canonical **YES book**: prices are YES
  probabilities in [0, 1]; a NO bid at q is a YES ask at 1 - q. ``Side.BUY`` means buying
  YES exposure; ``Side.SELL`` means selling YES / buying NO.
* Instrument ids are globally unique strings ``"<VENUE>:<native id>"`` (see :func:`instrument_key`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, ClassVar, Self

from cma.domain.enums import (
    BLOCKING_QUALITY_FLAGS,
    BookSide,
    ContractStatus,
    DeltaMode,
    EventType,
    InstrumentKind,
    LiquidityRole,
    MappingStatus,
    Operator,
    OrderState,
    OrderType,
    Outcome,
    QualityFlag,
    SettlementOutcome,
    Side,
    TimeInForce,
    Venue,
)
from cma.domain.errors import ImmutableTimestampError, InvalidPriceError
from cma.domain.numbers import ONE, ZERO, validate_probability


def instrument_key(venue: Venue, native_id: str) -> str:
    """Globally unique instrument/contract key, e.g. ``"KALSHI:KXBTCD-26OCT0617-T110999.99"``."""
    return f"{venue.value}:{native_id}"


def stable_id(*parts: object, length: int = 24) -> str:
    """Deterministic identifier derived from ``parts`` (idempotent under replay)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return digest[:length]


def payload_digest(payload: bytes | str) -> str:
    data = payload.encode() if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def _frozen_mapping(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(data or {}))


# --------------------------------------------------------------------------------------
# Instruments and books
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Instrument:
    venue: Venue
    instrument_id: str  # instrument_key(venue, native_id)
    native_id: str
    kind: InstrumentKind
    tick_size: Decimal
    quote_currency: str = "USD"
    underlying: str | None = None  # e.g. "BTC-USD"
    contract_id: str | None = None  # for binary-contract instruments
    outcome: Outcome | None = None  # for venues with one token per outcome

    def __post_init__(self) -> None:
        if self.tick_size <= ZERO:
            raise InvalidPriceError(f"tick_size must be positive for {self.instrument_id}")


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        if not self.price.is_finite() or self.price < ZERO:
            raise InvalidPriceError(f"invalid level price {self.price}")
        if not self.quantity.is_finite() or self.quantity < ZERO:
            raise InvalidPriceError(f"invalid level quantity {self.quantity}")


@dataclass(frozen=True, slots=True)
class LevelChange:
    side: BookSide
    price: Decimal
    quantity: Decimal  # absolute size or signed increment, see DeltaMode


@dataclass(frozen=True, slots=True, kw_only=True)
class BookSnapshot:
    """Immutable view of an L2 book. Bids best-first (descending), asks best-first (ascending)."""

    venue: Venue
    instrument_id: str
    source_ts_ns: int | None
    recv_ts_ns: int
    sequence: int | None
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    quality_flags: frozenset[QualityFlag] = frozenset()
    checksum: str | None = None

    def __post_init__(self) -> None:
        for levels, descending in ((self.bids, True), (self.asks, False)):
            for a, b in itertools.pairwise(levels):
                if (a.price <= b.price) if descending else (a.price >= b.price):
                    raise InvalidPriceError(
                        f"{self.instrument_id}: levels not strictly sorted best-first"
                    )

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def spread(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price

    @property
    def is_crossed(self) -> bool:
        return bool(self.bids and self.asks and self.bids[0].price > self.asks[0].price)

    @property
    def is_locked(self) -> bool:
        return bool(self.bids and self.asks and self.bids[0].price == self.asks[0].price)

    @property
    def is_valid(self) -> bool:
        """Usable for signals/matching: not crossed/locked and no blocking quality flag."""
        if self.is_crossed or self.is_locked:
            return False
        return not (self.quality_flags & BLOCKING_QUALITY_FLAGS)

    def levels(self, side: BookSide) -> tuple[BookLevel, ...]:
        return self.bids if side is BookSide.BID else self.asks

    def depth(self, side: BookSide, max_levels: int | None = None) -> Decimal:
        levels = self.levels(side)
        if max_levels is not None:
            levels = levels[:max_levels]
        return sum((lvl.quantity for lvl in levels), ZERO)

    def quantity_at(self, side: BookSide, price: Decimal) -> Decimal:
        for lvl in self.levels(side):
            if lvl.price == price:
                return lvl.quantity
        return ZERO

    @property
    def source_or_recv_ts_ns(self) -> int:
        return self.source_ts_ns if self.source_ts_ns is not None else self.recv_ts_ns


# --------------------------------------------------------------------------------------
# Market events (normalized). Scope entity "MarketEvent".
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketEvent:
    """Normalized market event envelope. Timestamps are distinct and immutable."""

    event_type: ClassVar[EventType]

    venue: Venue
    instrument_id: str
    source_ts_ns: int | None
    recv_ts_ns: int
    payload_hash: str
    sequence: int | None = None
    sub_index: int = 0  # position of this event within its raw payload
    process_ts_ns: int | None = None
    quality_flags: frozenset[QualityFlag] = frozenset()

    def __post_init__(self) -> None:
        if self.recv_ts_ns < 0:
            raise ValueError("recv_ts_ns must be non-negative")
        if self.source_ts_ns is not None and self.source_ts_ns < 0:
            raise ValueError("source_ts_ns must be non-negative")
        if self.process_ts_ns is not None and self.process_ts_ns < 0:
            raise ValueError("process_ts_ns must be non-negative")

    @property
    def event_id(self) -> str:
        """Deterministic id: identical raw payloads always map to identical event ids."""
        return stable_id(
            self.venue.value,
            self.instrument_id,
            self.event_type.value,
            self.payload_hash,
            self.sub_index,
        )

    @property
    def venue_ts_ns(self) -> int:
        """Best estimate of when the event happened at the venue.

        The source timestamp when present, never later than our receive time (a venue
        clock running ahead of ours is a data-quality issue, flagged upstream, not
        something a simulation may exploit).
        """
        if self.source_ts_ns is None:
            return self.recv_ts_ns
        return min(self.source_ts_ns, self.recv_ts_ns)

    def with_process_ts(self, process_ts_ns: int) -> Self:
        """Return a copy stamped with the processing time; refuses to overwrite."""
        if self.process_ts_ns is not None:
            raise ImmutableTimestampError(f"process_ts already set on event {self.event_id}")
        return dataclasses.replace(self, process_ts_ns=process_ts_ns)

    def with_flags(self, *flags: QualityFlag) -> Self:
        return dataclasses.replace(self, quality_flags=self.quality_flags | frozenset(flags))


@dataclass(frozen=True, slots=True, kw_only=True)
class BookSnapshotEvent(MarketEvent):
    event_type: ClassVar[EventType] = EventType.BOOK_SNAPSHOT
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    checksum: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BookDeltaEvent(MarketEvent):
    event_type: ClassVar[EventType] = EventType.BOOK_DELTA
    changes: tuple[LevelChange, ...]
    mode: DeltaMode
    checksum: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TradeEvent(MarketEvent):
    """Scope entity "Trade". ``aggressor_side`` is in canonical YES terms for binaries."""

    event_type: ClassVar[EventType] = EventType.TRADE
    price: Decimal
    size: Decimal
    aggressor_side: Side | None
    trade_id: str

    def __post_init__(self) -> None:
        MarketEvent.__post_init__(self)
        if self.size <= ZERO:
            raise InvalidPriceError(f"trade size must be positive, got {self.size}")
        if self.price < ZERO:
            raise InvalidPriceError(f"trade price must be non-negative, got {self.price}")


Trade = TradeEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusEvent(MarketEvent):
    event_type: ClassVar[EventType] = EventType.STATUS
    status: ContractStatus
    detail: str = ""


# --------------------------------------------------------------------------------------
# Raw capture envelope (immutable, replayable)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RawMessage:
    """A message exactly as received from a venue, with our receive metadata."""

    venue: Venue
    stream: str  # e.g. "ws:orderbook_delta", "rest:markets"
    recv_ts_ns: int
    payload: str  # exact text received
    connection_id: str = ""
    connection_seq: int = 0  # receive order within the connection
    idempotency_key: str | None = None  # natural unique key when the payload has one

    @property
    def payload_hash(self) -> str:
        return payload_digest(self.payload)

    @property
    def dedup_key(self) -> str:
        if self.idempotency_key is not None:
            return f"{self.venue.value}|{self.stream}|{self.idempotency_key}"
        return f"{self.venue.value}|{self.stream}|{self.payload_hash}|{self.recv_ts_ns}"


# --------------------------------------------------------------------------------------
# Contracts and mappings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionContract:
    """A binary prediction contract with its listing-time rule metadata."""

    venue: Venue
    contract_id: str  # instrument_key(venue, native id)
    native_id: str
    event_id: str
    title: str
    yes_semantics: str
    no_semantics: str
    open_ts_ns: int | None
    close_ts_ns: int | None
    resolve_ts_ns: int | None  # expected resolution / expiration time
    status: ContractStatus
    tick_size: Decimal
    series_id: str | None = None
    rules_text: str = ""
    can_close_early: bool = False
    fee_schedule_id: str = ""
    settlement_metadata: Mapping[str, Any] = field(default_factory=dict)
    outcome_instruments: Mapping[str, str] = field(default_factory=dict)  # Outcome -> instrument

    def __post_init__(self) -> None:
        if self.tick_size <= ZERO or self.tick_size >= ONE:
            raise InvalidPriceError(f"invalid tick size {self.tick_size} for {self.contract_id}")
        object.__setattr__(self, "settlement_metadata", _frozen_mapping(self.settlement_metadata))
        object.__setattr__(self, "outcome_instruments", _frozen_mapping(self.outcome_instruments))

    __hash__ = None  # type: ignore[assignment]  # contains mappings


@dataclass(frozen=True, slots=True, kw_only=True)
class ContractMapping:
    """Exact resolution semantics linking a contract to underlying variables (scope s.9)."""

    venue: Venue
    contract_id: str
    underlyings: tuple[str, ...]  # e.g. ("BTC-USD",)
    operator: Operator
    strikes: tuple[Decimal, ...]  # 1 strike for GT/GE/LT/LE, 2 for BETWEEN, 0 for UP/DOWN
    observation_start_ns: int | None  # window start (None for point observations)
    observation_end_ns: int  # observation instant / window end, UTC ns
    observation_method: str  # e.g. "POINT", "AVG_60S_BEFORE", "CANDLE_CLOSE_1M"
    timezone: str  # IANA timezone the rule text is written in
    resolution_source: str  # e.g. "CF_BENCHMARKS_BRTI", "BINANCE_BTCUSDT"
    rounding_rule: str = "NONE"
    early_close_rule: str = "NONE"
    outcome_semantics: str = ""
    event_family: str = ""  # risk-aggregation key (same variable & observation time)
    version: int = 1
    review_status: MappingStatus = MappingStatus.DRAFT
    reviewer: str | None = None
    reviewed_at_ns: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        expected = {
            Operator.BETWEEN: 2,
            Operator.UP: 0,
            Operator.DOWN: 0,
        }.get(self.operator, 1)
        if len(self.strikes) != expected:
            raise ValueError(
                f"{self.contract_id}: operator {self.operator} needs {expected} strike(s), "
                f"got {len(self.strikes)}"
            )
        if self.operator is Operator.BETWEEN and self.strikes[0] >= self.strikes[1]:
            raise ValueError(f"{self.contract_id}: BETWEEN strikes must be increasing")
        if self.version < 1:
            raise ValueError("mapping version must be >= 1")
        if (
            self.observation_start_ns is not None
            and self.observation_start_ns > self.observation_end_ns
        ):
            raise ValueError(f"{self.contract_id}: observation window start after end")

    @property
    def strike(self) -> Decimal:
        if len(self.strikes) != 1:
            raise ValueError(f"{self.contract_id} has {len(self.strikes)} strikes")
        return self.strikes[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class Settlement:
    venue: Venue
    contract_id: str
    outcome: SettlementOutcome
    yes_value: Decimal  # payoff per YES contract in [0, 1] (VOID: see policy)
    settled_ts_ns: int
    source: str = ""

    def __post_init__(self) -> None:
        validate_probability(self.yes_value, name="settlement yes_value")
        if self.outcome is SettlementOutcome.YES and self.yes_value != ONE:
            raise ValueError("YES settlement must pay 1")
        if self.outcome is SettlementOutcome.NO and self.yes_value != ZERO:
            raise ValueError("NO settlement must pay 0")


# --------------------------------------------------------------------------------------
# Research / decision objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureVector:
    asof_ts_ns: int
    contract_id: str
    mapping_version: int
    feature_version: str
    values: Mapping[str, float]
    source_watermarks: Mapping[str, int]  # source name -> max source timestamp used
    stale_sources: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _frozen_mapping(self.values))
        object.__setattr__(self, "source_watermarks", _frozen_mapping(self.source_watermarks))

    __hash__ = None  # type: ignore[assignment]

    @property
    def max_watermark_ns(self) -> int:
        return max(self.source_watermarks.values(), default=0)


@dataclass(frozen=True, slots=True, kw_only=True)
class Signal:
    """A trade intention that passed edge and quality gates (scope entity "Signal")."""

    signal_id: str
    strategy_id: str
    strategy_version: str
    asof_ts_ns: int
    venue: Venue
    contract_id: str
    instrument_id: str
    side: Side  # canonical YES side
    fair_probability: Decimal
    executable_price: Decimal
    gross_edge: Decimal  # per contract, in probability (= $ per $1 face) units
    expected_cost: Decimal
    net_edge: Decimal
    confidence: float
    expiry_ts_ns: int
    quantity: Decimal
    limit_price: Decimal
    mapping_version: int = 0
    feature_watermark_ns: int = 0
    group_id: str | None = None  # multi-leg (structural / cross-venue) package id
    reason: str = ""

    def __post_init__(self) -> None:
        validate_probability(self.fair_probability, name="fair_probability")
        validate_probability(self.executable_price, name="executable_price")
        validate_probability(self.limit_price, name="limit_price")
        if self.quantity <= ZERO:
            raise ValueError("signal quantity must be positive")
        if self.expiry_ts_ns < self.asof_ts_ns:
            raise ValueError("signal expires before it is generated")

    def is_expired(self, now_ns: int) -> bool:
        return now_ns > self.expiry_ts_ns


@dataclass(slots=True, kw_only=True)
class SimOrder:
    """Simulated order. Mutable only through its transition methods."""

    order_id: str
    venue: Venue
    contract_id: str
    instrument_id: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    tif: TimeInForce
    submit_ts_ns: int
    arrival_ts_ns: int
    limit_price: Decimal | None = None
    signal_id: str | None = None
    strategy_id: str = ""
    expiry_ts_ns: int | None = None  # signal TTL: arrival after this is rejected
    gtd_expiry_ts_ns: int | None = None
    reduce_only: bool = False
    group_id: str | None = None
    state: OrderState = OrderState.PENDING_ARRIVAL
    filled_quantity: Decimal = ZERO
    notional_filled: Decimal = ZERO
    queue_ahead: Decimal | None = None
    cancel_request_ts_ns: int | None = None
    cancel_arrival_ts_ns: int | None = None
    reject_reason: str | None = None
    last_update_ts_ns: int = 0

    def __post_init__(self) -> None:
        if self.quantity <= ZERO:
            raise ValueError("order quantity must be positive")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit order requires a limit price")
        if self.arrival_ts_ns < self.submit_ts_ns:
            raise ValueError("order cannot arrive before it is submitted")

    @property
    def remaining(self) -> Decimal:
        return self.quantity - self.filled_quantity

    @property
    def avg_fill_price(self) -> Decimal | None:
        if self.filled_quantity == ZERO:
            return None
        return self.notional_filled / self.filled_quantity

    def record_fill(self, quantity: Decimal, price: Decimal, ts_ns: int) -> None:
        if quantity <= ZERO:
            raise ValueError("fill quantity must be positive")
        if quantity > self.remaining:
            raise ValueError(
                f"overfill on {self.order_id}: {quantity} > remaining {self.remaining}"
            )
        if self.limit_price is not None:
            if self.side is Side.BUY and price > self.limit_price:
                raise InvalidPriceError(f"buy fill {price} above limit {self.limit_price}")
            if self.side is Side.SELL and price < self.limit_price:
                raise InvalidPriceError(f"sell fill {price} below limit {self.limit_price}")
        self.filled_quantity += quantity
        self.notional_filled += quantity * price
        self.last_update_ts_ns = ts_ns
        if self.remaining == ZERO:
            self.state = OrderState.FILLED
        elif self.state is not OrderState.CANCEL_PENDING:
            self.state = OrderState.PARTIALLY_FILLED

    def transition(self, state: OrderState, ts_ns: int, reason: str | None = None) -> None:
        if self.state.is_terminal:
            raise ValueError(f"order {self.order_id} already terminal ({self.state})")
        self.state = state
        self.last_update_ts_ns = ts_ns
        if reason is not None:
            self.reject_reason = reason


@dataclass(frozen=True, slots=True, kw_only=True)
class Fill:
    fill_id: str
    order_id: str
    venue: Venue
    contract_id: str
    instrument_id: str
    side: Side
    fill_ts_ns: int
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_schedule_version: str
    liquidity_role: LiquidityRole
    simulated: bool = True
    strategy_id: str = ""
    signal_id: str | None = None

    def __post_init__(self) -> None:
        if self.quantity <= ZERO:
            raise ValueError("fill quantity must be positive")
        if self.fee < ZERO:
            raise ValueError("fees cannot be negative (rebates are modelled separately)")
