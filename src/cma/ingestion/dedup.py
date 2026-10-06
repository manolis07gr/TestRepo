"""Bounded LRU de-duplication of normalized events (scope s.19: idempotent under replay).

The collector uses three layers of idempotency:

1. Raw messages: :class:`cma.storage.raw.RawRecorder` refuses a message whose dedup key
   was already recorded. Such a message is never parsed again.
2. Trades: :class:`TradeDeduplicator` drops a trade whose ``(instrument_id, trade_id)``
   was already seen. This catches the same trade arriving through different raw
   messages, e.g. a WS print and a REST backfill page, or Coinbase ``last_match`` after a
   reconnect.
3. Books: the book builder ignores stale or duplicate sequence numbers.

:class:`EventDeduplicator` drops exact duplicate events (same ``event_id``) when
overlapping recordings are merged. Do not use it on live book snapshots: a venue may
legitimately re-send an identical snapshot after a reconnect.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterable

from cma.domain.models import MarketEvent, TradeEvent


class BoundedLRUSet[K: Hashable]:
    """Set with a capacity; the least recently seen key is evicted first."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: OrderedDict[K, None] = OrderedDict()

    def add(self, key: K) -> bool:
        """Insert ``key``; True if it was new, False if already present (refreshed)."""
        if key in self._data:
            self._data.move_to_end(key)
            return False
        self._data[key] = None
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
        return True

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def discard(self, key: K) -> None:
        self._data.pop(key, None)

    def __len__(self) -> int:
        return len(self._data)


class TradeDeduplicator:
    """Drops trades already seen by ``(instrument_id, trade_id)``."""

    def __init__(self, capacity: int = 200_000) -> None:
        self._seen: BoundedLRUSet[tuple[str, str]] = BoundedLRUSet(capacity)
        self.duplicates = 0

    def is_duplicate(self, trade: TradeEvent) -> bool:
        """True if seen before; otherwise registers the trade and returns False."""
        if self._seen.add((trade.instrument_id, trade.trade_id)):
            return False
        self.duplicates += 1
        return True

    def __len__(self) -> int:
        return len(self._seen)


class EventDeduplicator:
    """Drops events already seen by deterministic ``event_id``."""

    def __init__(self, capacity: int = 500_000) -> None:
        self._seen: BoundedLRUSet[str] = BoundedLRUSet(capacity)
        self.duplicates = 0

    def is_duplicate(self, event: MarketEvent) -> bool:
        if self._seen.add(event.event_id):
            return False
        self.duplicates += 1
        return True

    def filter[E: MarketEvent](self, events: Iterable[E]) -> list[E]:
        return [e for e in events if not self.is_duplicate(e)]
