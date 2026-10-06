"""Normalization stage between adapters and consumers (scope s.5, s.8).

* stamps ``process_ts_ns`` from an injected clock (never overwriting source/recv),
* flags clock drift: a source timestamp *ahead* of our receive clock by more than
  ``max_clock_drift_ms`` marks the event CLOCK_DRIFT (degraded) - it is not corrected,
* flags missing source timestamps,
* drops exact replays of an already-processed event id (idempotency under replay),
* re-validates canonical prediction prices are probabilities (defence in depth).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from cma.config import DataQualityConfig
from cma.domain.enums import QualityFlag
from cma.domain.errors import InvalidProbabilityError
from cma.domain.models import BookDeltaEvent, BookSnapshotEvent, MarketEvent, TradeEvent
from cma.domain.numbers import validate_probability
from cma.domain.time import NS_PER_MS, Clock


@dataclass
class NormalizerStats:
    processed: int = 0
    duplicates: int = 0
    clock_drift: int = 0
    missing_source_ts: int = 0
    rejected: int = 0
    max_drift_ms: float = 0.0


@dataclass
class Normalizer:
    config: DataQualityConfig
    clock: Clock
    probability_instruments: frozenset[str] | None = None  # None => infer from venue
    dedup_window: int = 200_000
    stats: NormalizerStats = field(default_factory=NormalizerStats)
    _seen: OrderedDict[str, None] = field(default_factory=OrderedDict)

    def _is_probability(self, ev: MarketEvent) -> bool:
        if self.probability_instruments is not None:
            return ev.instrument_id in self.probability_instruments
        return ev.venue.value in ("KALSHI", "POLYMARKET")

    def _check_prices(self, ev: MarketEvent) -> None:
        if isinstance(ev, BookSnapshotEvent):
            for lvl in (*ev.bids, *ev.asks):
                validate_probability(lvl.price)
        elif isinstance(ev, BookDeltaEvent):
            for ch in ev.changes:
                validate_probability(ch.price)
        elif isinstance(ev, TradeEvent):
            validate_probability(ev.price)

    def process(self, ev: MarketEvent) -> MarketEvent | None:
        eid = ev.event_id
        if eid in self._seen:
            self.stats.duplicates += 1
            return None
        if self._is_probability(ev):
            try:
                self._check_prices(ev)
            except InvalidProbabilityError:
                self.stats.rejected += 1
                return None
        flags: list[QualityFlag] = []
        if ev.source_ts_ns is None:
            self.stats.missing_source_ts += 1
            flags.append(QualityFlag.SOURCE_TS_MISSING)
        else:
            ahead_ms = (ev.source_ts_ns - ev.recv_ts_ns) / NS_PER_MS
            self.stats.max_drift_ms = max(self.stats.max_drift_ms, ahead_ms)
            if ahead_ms > self.config.max_clock_drift_ms:
                self.stats.clock_drift += 1
                flags.append(QualityFlag.CLOCK_DRIFT)
        out = ev.with_flags(*flags) if flags else ev
        if out.process_ts_ns is None:
            out = out.with_process_ts(self.clock.now_ns())
        self._seen[eid] = None
        if len(self._seen) > self.dedup_window:
            self._seen.popitem(last=False)
        self.stats.processed += 1
        return out

    def process_all(self, events: Iterable[MarketEvent]) -> Iterator[MarketEvent]:
        for ev in events:
            out = self.process(ev)
            if out is not None:
                yield out
