"""Venue adapter contract and the strict payload-parsing helpers every adapter shares.

Adapters are the only code that understands vendor JSON. They turn a :class:`RawMessage`
into canonical :mod:`cma.domain.models` events, and listing payloads into
:class:`~cma.domain.models.PredictionContract` metadata; everything downstream is
venue-agnostic (scope s.26).

Rules every adapter follows
---------------------------
* ``parse`` is pure and deterministic: the same ``RawMessage`` always yields the same
  events, with the same ``event_id``s (derived from ``payload_hash`` + ``sub_index``). It never
  reads a clock. ``recv_ts_ns`` comes from the raw envelope, and ``process_ts_ns`` is
  stamped later by the ingestion layer.
* Numbers go from their JSON text straight into :class:`~decimal.Decimal`
  (``json.loads(parse_float=Decimal)``), so binary floats never touch prices or sizes.
* Unparseable or schema-violating payloads raise :class:`MalformedPayloadError`. That
  includes a probability outside [0, 1]. The collector quarantines these payloads and
  keeps running. Unknown *message types* are ignored (forward compatible), but a known
  type with bad fields is always an error.
* ``idempotency_key`` returns the payload's natural unique key when it has one. Some keys
  are unique only within one WebSocket connection, so they carry
  :data:`CONNECTION_SCOPED_PREFIX`. Examples: Kalshi ``sid``/``seq`` restart on every
  connection, and venues legitimately re-send an unchanged snapshot after a reconnect.
  The feed session namespaces these keys with its connection id via
  :func:`resolve_idempotency_key`.

Schema provenance
-----------------
Every venue payload schema in :mod:`cma.adapters` was implemented from the venues'
documented formats (as of October 2026) WITHOUT live access to the venues. The build
environment cannot reach venue hosts. Validate the schemas against live payloads before
relying on collected data. Schema drift shows up as quarantined messages; it is never
silently "repaired".
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any, Final, NoReturn, Protocol, runtime_checkable

from cma.domain.enums import BookSide, ContractStatus, DeltaMode, QualityFlag, Side, Venue
from cma.domain.errors import CMAError
from cma.domain.models import (
    BookDeltaEvent,
    BookLevel,
    BookSnapshotEvent,
    LevelChange,
    MarketEvent,
    RawMessage,
    StatusEvent,
    TradeEvent,
)
from cma.domain.numbers import ONE, ZERO
from cma.domain.time import NS_PER_MS, NS_PER_S, NS_PER_US, ns_from_iso8601

CONNECTION_SCOPED_PREFIX: Final = "conn:"


class MalformedPayloadError(CMAError):
    """A venue payload is unparseable or violates the documented schema."""

    def __init__(self, message: str, *, venue: Venue | None = None) -> None:
        super().__init__(message)
        self.venue = venue


@runtime_checkable
class VenueAdapter(Protocol):
    """Pure translation of one venue's raw messages into canonical market events."""

    @property
    def venue(self) -> Venue: ...

    def parse(self, raw: RawMessage) -> list[MarketEvent]:
        """Canonical events for ``raw``; raises :class:`MalformedPayloadError`."""
        ...

    def idempotency_key(self, stream: str, payload: str) -> str | None:
        """Natural unique key of ``payload`` (never raises; ``None`` when there is none)."""
        ...


@runtime_checkable
class SupportsResync(Protocol):
    """Optional adapter capability: in-band re-snapshot after a sequence gap.

    ``raw`` is the message that revealed the gap. The returned frames are sent on the
    live connection. An empty list means "cannot resync in band", and the session then
    reconnects instead.
    """

    def resync_messages(self, instrument_id: str, raw: RawMessage, request_id: int) -> list[str]:
        """Frames that request a fresh snapshot for ``instrument_id``."""
        ...


@runtime_checkable
class SupportsAttribution(Protocol):
    """Optional adapter capability: which books a (possibly malformed) payload targets.

    The ingestion layer uses it when a payload is quarantined. A book update that could
    not be applied leaves that book untrustworthy, so the book is invalidated and
    re-snapshotted (fail closed). ``None`` means "cannot tell"; ``[]`` means "touches no
    book" (trades, tickers, control frames).
    """

    def affected_instruments(self, raw: RawMessage) -> list[str] | None:
        """Instrument ids whose order book ``raw`` would have updated."""
        ...


@runtime_checkable
class SupportsSequenceScope(Protocol):
    """Optional adapter capability: where a book message's sequence number counts.

    Some venues number messages per *subscription*, not per book; Kalshi folds every
    market of its orderbook channel into one subscription, so one market's sequence
    numbers are never contiguous. The ingestion layer then checks continuity per scope,
    applies book updates without a per-book sequence check, and treats a gap as a
    possible loss for every book of that scope (fail closed).
    """

    def sequence_scope(self, raw: RawMessage) -> tuple[str, int] | None:
        """``(scope key, sequence)`` of a sequenced book message, else None."""
        ...


def connection_scoped(key: str) -> str:
    """Mark ``key`` as unique only within the connection that delivered it."""
    return CONNECTION_SCOPED_PREFIX + key


def resolve_idempotency_key(key: str | None, connection_id: str) -> str | None:
    """Namespace connection-scoped keys with ``connection_id``; leave global keys alone."""
    if key is None:
        return None
    if key.startswith(CONNECTION_SCOPED_PREFIX):
        return f"{connection_id}|{key[len(CONNECTION_SCOPED_PREFIX) :]}"
    return key


# --------------------------------------------------------------------------------------
# JSON and typed field access
# --------------------------------------------------------------------------------------


def _reject_constant(name: str) -> NoReturn:
    raise MalformedPayloadError(f"non-finite JSON number {name!r}")


def load_json(payload: str | bytes) -> Any:
    """Parse JSON with exact decimals; NaN/Infinity and invalid text are malformed."""
    try:
        return json.loads(payload, parse_float=Decimal, parse_constant=_reject_constant)
    except MalformedPayloadError:
        raise
    except (ValueError, RecursionError) as exc:  # JSONDecodeError, UnicodeDecodeError
        raise MalformedPayloadError(f"invalid JSON: {exc}") from exc


def _type_name(value: object) -> str:
    return "null" if value is None else type(value).__name__


def as_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MalformedPayloadError(f"{what}: expected a JSON object, got {_type_name(value)}")
    return value


def as_list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise MalformedPayloadError(f"{what}: expected a JSON array, got {_type_name(value)}")
    return value


def require(obj: Mapping[str, Any], key: str, what: str) -> Any:
    """Field ``key`` (present and not null)."""
    value = obj.get(key)
    if value is None:
        raise MalformedPayloadError(f"{what}: missing required field {key!r}")
    return value


def req_str(obj: Mapping[str, Any], key: str, what: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MalformedPayloadError(
            f"{what}: field {key!r} must be a non-empty string, got {_type_name(value)}"
        )
    return value


def opt_str(obj: Mapping[str, Any], key: str, what: str) -> str | None:
    """Optional string field; empty strings read as ``None``; other types are malformed."""
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MalformedPayloadError(
            f"{what}: field {key!r} must be a string, got {_type_name(value)}"
        )
    return value if value.strip() else None


def opt_id(obj: Mapping[str, Any], key: str, what: str) -> str | None:
    """Identifier that venues encode either as a string or as an integer."""
    value = obj.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise MalformedPayloadError(f"{what}: field {key!r} must be an identifier")
    if isinstance(value, str | int):
        return str(value)
    raise MalformedPayloadError(f"{what}: field {key!r} must be an identifier")


def opt_bool(obj: Mapping[str, Any], key: str, what: str) -> bool | None:
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise MalformedPayloadError(f"{what}: field {key!r} must be a boolean")


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

_INT_RE = re.compile(r"[+-]?\d+")


def to_decimal(value: Any, what: str) -> Decimal:
    """Exact decimal from a JSON number (Decimal/int) or numeric string; floats refused."""
    if isinstance(value, bool):
        raise MalformedPayloadError(f"{what}: expected a number, got bool")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise MalformedPayloadError(f"{what}: empty numeric string")
        try:
            result = Decimal(text)
        except InvalidOperation as exc:
            raise MalformedPayloadError(f"{what}: not a decimal number: {value!r}") from exc
    else:
        # floats are refused on purpose: parse JSON with load_json (parse_float=Decimal)
        raise MalformedPayloadError(f"{what}: expected a decimal number, got {_type_name(value)}")
    if not result.is_finite():
        raise MalformedPayloadError(f"{what}: non-finite number {value!r}")
    return result


def to_int(value: Any, what: str) -> int:
    if isinstance(value, bool):
        raise MalformedPayloadError(f"{what}: expected an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value.is_finite() and value == value.to_integral_value():
        return int(value)
    if isinstance(value, str) and _INT_RE.fullmatch(value.strip()):
        return int(value.strip())
    raise MalformedPayloadError(f"{what}: expected an integer, got {value!r}")


def to_probability(value: Any, what: str) -> Decimal:
    """A price/probability in [0, 1] (venue dollar prices on $1-face contracts)."""
    d = to_decimal(value, what)
    if d < ZERO or d > ONE:
        raise MalformedPayloadError(f"{what}: probability {d} outside [0, 1]")
    return d


def cents_to_probability(value: Any, what: str) -> Decimal:
    """Legacy integer-cent price -> probability (45 -> 0.45)."""
    d = to_decimal(value, what)
    if d != d.to_integral_value():
        raise MalformedPayloadError(f"{what}: cent price must be an integer, got {d}")
    return to_probability(d / 100, what)


def to_quantity(value: Any, what: str, *, signed: bool = False, positive: bool = False) -> Decimal:
    d = to_decimal(value, what)
    if not signed and d < ZERO:
        raise MalformedPayloadError(f"{what}: negative quantity {d}")
    if positive and d <= ZERO:
        raise MalformedPayloadError(f"{what}: quantity must be positive, got {d}")
    return d


def complement_probability(price: Decimal, what: str) -> Decimal:
    """1 - p for a validated probability (a NO price expressed in YES terms)."""
    return ONE - to_probability(price, what)


# --------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------

_EPOCH_FACTORS: Final[Mapping[str, int]] = {
    "s": NS_PER_S,
    "ms": NS_PER_MS,
    "us": NS_PER_US,
    "ns": 1,
}
_SHORT_OFFSET_RE = re.compile(r"(.*\d{2}:\d{2}:\d{2}(?:\.\d+)?)([+-]\d{2})")


def epoch_to_ns(value: Any, unit: str, what: str) -> int:
    """Unix epoch number/numeric string in ``unit`` (s, ms, us, ns) -> integer UTC ns."""
    factor = _EPOCH_FACTORS.get(unit)
    if factor is None:
        raise ValueError(f"unknown epoch unit {unit!r}")
    d = to_decimal(value, what)
    if d < ZERO:
        raise MalformedPayloadError(f"{what}: negative epoch timestamp {d}")
    return int((d * factor).to_integral_value(rounding=ROUND_FLOOR))


def epoch_auto_to_ns(value: Any, what: str) -> int:
    """Epoch timestamp whose unit is undocumented: inferred from its magnitude.

    Only for fields whose unit the venue does not document. Present-day epochs are
    ~1.7e9 s, ~1.7e12 ms, ~1.7e15 us and ~1.7e18 ns, so the magnitude ranges are disjoint.
    """
    d = to_decimal(value, what)
    if d < ZERO:
        raise MalformedPayloadError(f"{what}: negative epoch timestamp {d}")
    if d < Decimal("1e11"):
        unit = "s"
    elif d < Decimal("1e14"):
        unit = "ms"
    elif d < Decimal("1e17"):
        unit = "us"
    else:
        unit = "ns"
    return epoch_to_ns(d, unit, what)


def iso_to_ns(value: Any, what: str) -> int:
    """ISO-8601 timestamp with explicit offset (also tolerates a bare ``+HH`` offset)."""
    if not isinstance(value, str) or not value.strip():
        raise MalformedPayloadError(f"{what}: expected an ISO-8601 timestamp, got {value!r}")
    text = value.strip()
    m = _SHORT_OFFSET_RE.fullmatch(text)
    if m is not None:
        text = f"{m.group(1)}{m.group(2)}:00"
    try:
        return ns_from_iso8601(text)
    except ValueError as exc:
        raise MalformedPayloadError(f"{what}: {exc}") from exc


# --------------------------------------------------------------------------------------
# Books and events
# --------------------------------------------------------------------------------------


def build_book_side(
    levels: Iterable[tuple[Decimal, Decimal]], *, side: BookSide, what: str
) -> tuple[BookLevel, ...]:
    """Best-first levels for ``side``. Zero sizes are dropped; duplicates/negatives refused.

    Venues publish ladders in different orders (often worst-to-best). The ordering here
    never depends on the payload.
    """
    merged: dict[Decimal, Decimal] = {}
    for price, qty in levels:
        if qty < ZERO:
            raise MalformedPayloadError(f"{what}: negative size {qty} at price {price}")
        if price in merged:
            raise MalformedPayloadError(f"{what}: duplicate price level {price}")
        merged[price] = qty
    ordered = sorted(
        ((p, q) for p, q in merged.items() if q > ZERO),
        key=lambda pq: pq[0],
        reverse=side is BookSide.BID,
    )
    return tuple(BookLevel(p, q) for p, q in ordered)


class EventFactory:
    """Builds the canonical events of one raw payload, numbered in emission order.

    Every event carries the raw envelope's venue, receive time and payload hash. The
    ``sub_index`` makes event ids unique within the payload and identical across
    replays. A missing source timestamp is flagged ``SOURCE_TS_MISSING``.
    """

    __slots__ = ("_next", "payload_hash", "recv_ts_ns", "venue")

    def __init__(self, venue: Venue, recv_ts_ns: int, payload_hash: str) -> None:
        self.venue = venue
        self.recv_ts_ns = recv_ts_ns
        self.payload_hash = payload_hash
        self._next = 0

    @classmethod
    def for_raw(cls, raw: RawMessage) -> EventFactory:
        return cls(raw.venue, raw.recv_ts_ns, raw.payload_hash)

    def _index(self) -> int:
        index = self._next
        self._next += 1
        return index

    @staticmethod
    def _flags(source_ts_ns: int | None, flags: Iterable[QualityFlag]) -> frozenset[QualityFlag]:
        out = set(flags)
        if source_ts_ns is None:
            out.add(QualityFlag.SOURCE_TS_MISSING)
        return frozenset(out)

    def snapshot(
        self,
        *,
        instrument_id: str,
        source_ts_ns: int | None,
        bids: tuple[BookLevel, ...],
        asks: tuple[BookLevel, ...],
        sequence: int | None = None,
        checksum: str | None = None,
        flags: Iterable[QualityFlag] = (),
    ) -> BookSnapshotEvent:
        with malformed_guard(self.venue, "snapshot"):
            return BookSnapshotEvent(
                venue=self.venue,
                instrument_id=instrument_id,
                source_ts_ns=source_ts_ns,
                recv_ts_ns=self.recv_ts_ns,
                payload_hash=self.payload_hash,
                sequence=sequence,
                sub_index=self._index(),
                quality_flags=self._flags(source_ts_ns, flags),
                bids=bids,
                asks=asks,
                checksum=checksum,
            )

    def delta(
        self,
        *,
        instrument_id: str,
        source_ts_ns: int | None,
        changes: tuple[LevelChange, ...],
        mode: DeltaMode,
        sequence: int | None = None,
        checksum: str | None = None,
        flags: Iterable[QualityFlag] = (),
    ) -> BookDeltaEvent:
        with malformed_guard(self.venue, "delta"):
            return BookDeltaEvent(
                venue=self.venue,
                instrument_id=instrument_id,
                source_ts_ns=source_ts_ns,
                recv_ts_ns=self.recv_ts_ns,
                payload_hash=self.payload_hash,
                sequence=sequence,
                sub_index=self._index(),
                quality_flags=self._flags(source_ts_ns, flags),
                changes=changes,
                mode=mode,
                checksum=checksum,
            )

    def trade(
        self,
        *,
        instrument_id: str,
        source_ts_ns: int | None,
        price: Decimal,
        size: Decimal,
        aggressor_side: Side | None,
        trade_id: str,
        sequence: int | None = None,
        flags: Iterable[QualityFlag] = (),
    ) -> TradeEvent:
        with malformed_guard(self.venue, "trade"):
            return TradeEvent(
                venue=self.venue,
                instrument_id=instrument_id,
                source_ts_ns=source_ts_ns,
                recv_ts_ns=self.recv_ts_ns,
                payload_hash=self.payload_hash,
                sequence=sequence,
                sub_index=self._index(),
                quality_flags=self._flags(source_ts_ns, flags),
                price=price,
                size=size,
                aggressor_side=aggressor_side,
                trade_id=trade_id,
            )

    def status(
        self,
        *,
        instrument_id: str,
        source_ts_ns: int | None,
        status: ContractStatus,
        detail: str = "",
        flags: Iterable[QualityFlag] = (),
    ) -> StatusEvent:
        with malformed_guard(self.venue, "status"):
            return StatusEvent(
                venue=self.venue,
                instrument_id=instrument_id,
                source_ts_ns=source_ts_ns,
                recv_ts_ns=self.recv_ts_ns,
                payload_hash=self.payload_hash,
                sub_index=self._index(),
                quality_flags=self._flags(source_ts_ns, flags),
                status=status,
                detail=detail,
            )


@contextmanager
def malformed_guard(venue: Venue | None, what: str) -> Iterator[None]:
    """Convert low-level parsing/validation errors into :class:`MalformedPayloadError`."""
    try:
        yield
    except MalformedPayloadError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError, AttributeError) as exc:
        prefix = f"{venue.value} " if venue is not None else ""
        raise MalformedPayloadError(
            f"{prefix}{what}: {type(exc).__name__}: {exc}", venue=venue
        ) from exc


def check_venue(raw: RawMessage, venue: Venue) -> None:
    if raw.venue is not venue:
        raise MalformedPayloadError(
            f"{venue.value} adapter received a {raw.venue.value} message", venue=venue
        )


def jsonable(value: Any) -> Any:
    """Metadata value -> JSON-native value (Decimals become exact strings)."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def safe_idempotency_key(compute: Callable[[], str | None]) -> str | None:
    """Run a key computation and map any parsing failure to ``None`` (keys never raise)."""
    try:
        return compute()
    except (MalformedPayloadError, KeyError, IndexError, TypeError, ValueError, ArithmeticError):
        return None
