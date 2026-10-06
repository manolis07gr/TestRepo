"""L2 order-book reconstruction from snapshot + incremental deltas (scope s.8).

Rules (fail closed):
* A book is unusable until a snapshot has been applied.
* A delta whose sequence is <= the last applied sequence is a duplicate/old message and
  can never mutate the book (T009).
* A sequence gap invalidates the book and requests a re-snapshot (T008); deltas are
  ignored until the snapshot arrives (optionally buffered and replayed after it).
* A crossed book (best bid > best ask) is flagged and, by default, invalidated (T010).
* Negative resulting sizes invalidate the book (integrity failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from cma.domain.enums import BookSide, DeltaMode, QualityFlag, Venue
from cma.domain.models import (
    BookDeltaEvent,
    BookLevel,
    BookSnapshot,
    BookSnapshotEvent,
    LevelChange,
    MarketEvent,
)
from cma.domain.numbers import ZERO


class BookState(StrEnum):
    AWAITING_SNAPSHOT = "AWAITING_SNAPSHOT"
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class ApplyResult:
    applied: bool
    reason: str = ""
    needs_snapshot: bool = False


@dataclass
class BookCounters:
    snapshots: int = 0
    deltas_applied: int = 0
    duplicates_ignored: int = 0
    gaps: int = 0
    crossed: int = 0
    negative_quantity: int = 0
    ignored_while_invalid: int = 0
    off_tick: int = 0


@dataclass
class L2BookBuilder:
    venue: Venue
    instrument_id: str
    tick_size: Decimal | None = None
    require_sequence: bool = True
    crossed_policy: str = "invalidate"  # or "flag"
    buffer_while_awaiting: bool = False
    max_buffer: int = 10_000
    state: BookState = BookState.AWAITING_SNAPSHOT
    last_sequence: int | None = None
    last_source_ts_ns: int | None = None
    last_recv_ts_ns: int = 0
    counters: BookCounters = field(default_factory=BookCounters)
    _bids: dict[Decimal, Decimal] = field(default_factory=dict)
    _asks: dict[Decimal, Decimal] = field(default_factory=dict)
    _flags: set[QualityFlag] = field(default_factory=lambda: {QualityFlag.AWAITING_SNAPSHOT})
    _buffer: list[BookDeltaEvent] = field(default_factory=list)

    # ------------------------------------------------------------------ public API

    @property
    def is_valid(self) -> bool:
        return self.state is BookState.VALID and not self._blocking_flags()

    @property
    def flags(self) -> frozenset[QualityFlag]:
        return frozenset(self._flags)

    def apply(self, event: MarketEvent) -> ApplyResult:
        if isinstance(event, BookSnapshotEvent):
            return self.apply_snapshot(event)
        if isinstance(event, BookDeltaEvent):
            return self.apply_delta(event)
        return ApplyResult(applied=False, reason="not_a_book_event")

    def apply_snapshot(self, event: BookSnapshotEvent) -> ApplyResult:
        if (
            self.state is BookState.VALID
            and event.sequence is not None
            and self.last_sequence is not None
            and event.sequence <= self.last_sequence
        ):
            self.counters.duplicates_ignored += 1
            return ApplyResult(applied=False, reason="stale_snapshot")
        self._bids = {lvl.price: lvl.quantity for lvl in event.bids if lvl.quantity > ZERO}
        self._asks = {lvl.price: lvl.quantity for lvl in event.asks if lvl.quantity > ZERO}
        self.last_sequence = event.sequence
        self._touch(event)
        self._flags = set()
        self.state = BookState.VALID
        self.counters.snapshots += 1
        for lvl in (*event.bids, *event.asks):
            self._check_tick(lvl.price)
        self._check_crossed()
        if self.buffer_while_awaiting and self._buffer:
            pending, self._buffer = self._buffer, []
            for delta in sorted(pending, key=lambda d: d.sequence or 0):
                if delta.sequence is not None and event.sequence is not None:
                    if delta.sequence <= event.sequence:
                        continue
                self.apply_delta(delta)
        return ApplyResult(applied=True, reason="snapshot")

    def apply_delta(self, event: BookDeltaEvent) -> ApplyResult:
        if self.state is not BookState.VALID:
            self.counters.ignored_while_invalid += 1
            if self.buffer_while_awaiting and len(self._buffer) < self.max_buffer:
                self._buffer.append(event)
            return ApplyResult(applied=False, reason="awaiting_snapshot", needs_snapshot=True)

        seq = event.sequence
        if seq is None:
            if self.require_sequence:
                return self._invalidate(QualityFlag.SEQUENCE_GAP, "missing_sequence")
        elif self.last_sequence is not None:
            if seq <= self.last_sequence:
                self.counters.duplicates_ignored += 1
                return ApplyResult(applied=False, reason="stale_sequence")
            if seq != self.last_sequence + 1:
                self.counters.gaps += 1
                return self._invalidate(QualityFlag.SEQUENCE_GAP, "sequence_gap")

        for change in event.changes:
            if not self._apply_change(change, event.mode):
                self.counters.negative_quantity += 1
                if seq is not None:
                    self.last_sequence = seq
                return self._invalidate(QualityFlag.NEGATIVE_QUANTITY, "negative_quantity")

        if seq is not None:
            self.last_sequence = seq
        self._touch(event)
        self.counters.deltas_applied += 1
        self._check_crossed()
        if self.state is not BookState.VALID:
            return ApplyResult(applied=True, reason="crossed", needs_snapshot=True)
        return ApplyResult(applied=True, reason="delta")

    def invalidate(self, flag: QualityFlag, reason: str = "") -> ApplyResult:
        """External invalidation (disconnect, staleness, checksum mismatch...)."""
        return self._invalidate(flag, reason or flag.value.lower())

    def snapshot(self, max_levels: int | None = None) -> BookSnapshot:
        bids = sorted(self._bids.items(), key=lambda kv: kv[0], reverse=True)
        asks = sorted(self._asks.items(), key=lambda kv: kv[0])
        if max_levels is not None:
            bids, asks = bids[:max_levels], asks[:max_levels]
        flags = set(self._flags)
        if self.state is BookState.AWAITING_SNAPSHOT:
            flags.add(QualityFlag.AWAITING_SNAPSHOT)
        return BookSnapshot(
            venue=self.venue,
            instrument_id=self.instrument_id,
            source_ts_ns=self.last_source_ts_ns,
            recv_ts_ns=self.last_recv_ts_ns,
            sequence=self.last_sequence,
            bids=tuple(BookLevel(p, q) for p, q in bids),
            asks=tuple(BookLevel(p, q) for p, q in asks),
            quality_flags=frozenset(flags),
        )

    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        if not self._bids:
            return None
        p = max(self._bids)
        return p, self._bids[p]

    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        if not self._asks:
            return None
        p = min(self._asks)
        return p, self._asks[p]

    def quantity_at(self, side: BookSide, price: Decimal) -> Decimal:
        book = self._bids if side is BookSide.BID else self._asks
        return book.get(price, ZERO)

    def levels(self, side: BookSide) -> list[tuple[Decimal, Decimal]]:
        """Best-first (price, quantity) pairs for ``side``."""
        if side is BookSide.BID:
            return sorted(self._bids.items(), key=lambda kv: kv[0], reverse=True)
        return sorted(self._asks.items(), key=lambda kv: kv[0])

    # ------------------------------------------------------------------ internals

    def _blocking_flags(self) -> bool:
        return bool(
            self._flags
            & {
                QualityFlag.SEQUENCE_GAP,
                QualityFlag.CROSSED,
                QualityFlag.LOCKED,
                QualityFlag.NEGATIVE_QUANTITY,
                QualityFlag.DISCONNECTED,
                QualityFlag.STALE,
                QualityFlag.CHECKSUM_MISMATCH,
                QualityFlag.AWAITING_SNAPSHOT,
            }
        )

    def _touch(self, event: MarketEvent) -> None:
        if event.source_ts_ns is not None:
            self.last_source_ts_ns = event.source_ts_ns
        self.last_recv_ts_ns = max(self.last_recv_ts_ns, event.recv_ts_ns)

    def _invalidate(self, flag: QualityFlag, reason: str) -> ApplyResult:
        self.state = BookState.INVALID
        self._flags.add(flag)
        return ApplyResult(applied=False, reason=reason, needs_snapshot=True)

    def _apply_change(self, change: LevelChange, mode: DeltaMode) -> bool:
        book = self._bids if change.side is BookSide.BID else self._asks
        if mode is DeltaMode.ABSOLUTE:
            new_qty = change.quantity
        else:
            new_qty = book.get(change.price, ZERO) + change.quantity
        if new_qty < ZERO:
            return False
        self._check_tick(change.price)
        if new_qty == ZERO:
            book.pop(change.price, None)
        else:
            book[change.price] = new_qty
        return True

    def _check_tick(self, price: Decimal) -> None:
        if self.tick_size is not None and price % self.tick_size != ZERO:
            self.counters.off_tick += 1
            self._flags.add(QualityFlag.OFF_TICK)

    def _check_crossed(self) -> None:
        bb, ba = self.best_bid(), self.best_ask()
        self._flags.discard(QualityFlag.LOCKED)
        if bb is None or ba is None:
            self._flags.discard(QualityFlag.CROSSED)
            return
        if bb[0] > ba[0]:
            self.counters.crossed += 1
            self._flags.add(QualityFlag.CROSSED)
            if self.crossed_policy == "invalidate":
                self.state = BookState.INVALID
        elif bb[0] == ba[0]:
            self._flags.add(QualityFlag.LOCKED)
        elif self.crossed_policy == "flag":
            self._flags.discard(QualityFlag.CROSSED)


@dataclass
class BookManager:
    """Holds one builder per instrument; creates them lazily with shared settings."""

    require_sequence: bool = True
    crossed_policy: str = "invalidate"
    tick_sizes: dict[str, Decimal] = field(default_factory=dict)
    sequence_free_venues: frozenset[Venue] = frozenset()
    books: dict[str, L2BookBuilder] = field(default_factory=dict)

    def builder(self, venue: Venue, instrument_id: str) -> L2BookBuilder:
        b = self.books.get(instrument_id)
        if b is None:
            b = L2BookBuilder(
                venue=venue,
                instrument_id=instrument_id,
                tick_size=self.tick_sizes.get(instrument_id),
                require_sequence=self.require_sequence and venue not in self.sequence_free_venues,
                crossed_policy=self.crossed_policy,
            )
            self.books[instrument_id] = b
        return b

    def apply(self, event: MarketEvent) -> ApplyResult:
        if not isinstance(event, BookSnapshotEvent | BookDeltaEvent):
            return ApplyResult(applied=False, reason="not_a_book_event")
        return self.builder(event.venue, event.instrument_id).apply(event)

    def invalidate_venue(self, venue: Venue, flag: QualityFlag) -> list[str]:
        hit = []
        for key, b in self.books.items():
            if b.venue is venue:
                b.invalidate(flag)
                hit.append(key)
        return hit

    def get(self, instrument_id: str) -> L2BookBuilder | None:
        return self.books.get(instrument_id)
