"""Observed market state: what a strategy legitimately knows at its decision time.

Updated strictly in observation (receive-time) order. Reference prices are stored as
floats for statistics (explicit boundary conversion); books stay Decimal.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from cma.domain.enums import QualityFlag, Venue
from cma.domain.models import (
    BookDeltaEvent,
    BookSnapshot,
    BookSnapshotEvent,
    MarketEvent,
    TradeEvent,
)
from cma.domain.time import NS_PER_S
from cma.ingestion.book import BookManager

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass
class RefSeries:
    """Append-only observed price series for one reference instrument."""

    max_points: int = 500_000
    source_ts: list[int] = field(default_factory=list)
    recv_ts: list[int] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    out_of_order: int = 0

    def append(self, source_ts_ns: int, recv_ts_ns: int, price: float) -> bool:
        if not math.isfinite(price) or price <= 0:
            return False
        if self.source_ts and source_ts_ns < self.source_ts[-1]:
            self.out_of_order += 1
            return False
        if self.source_ts and source_ts_ns == self.source_ts[-1]:
            self.recv_ts[-1] = recv_ts_ns
            self.prices[-1] = price
            return True
        self.source_ts.append(source_ts_ns)
        self.recv_ts.append(recv_ts_ns)
        self.prices.append(price)
        if len(self.source_ts) > self.max_points:
            cut = len(self.source_ts) - self.max_points // 2
            del self.source_ts[:cut], self.recv_ts[:cut], self.prices[:cut]
        return True

    def __len__(self) -> int:
        return len(self.prices)

    def latest(self) -> tuple[int, float] | None:
        if not self.prices:
            return None
        return self.source_ts[-1], self.prices[-1]

    def index_at(self, ts_ns: int) -> int:
        """Index of the last point with source_ts <= ts_ns (-1 if none)."""
        return bisect.bisect_right(self.source_ts, ts_ns) - 1

    def price_at(self, ts_ns: int, max_age_ns: int | None = None) -> tuple[int, float] | None:
        i = self.index_at(ts_ns)
        if i < 0:
            return None
        if max_age_ns is not None and ts_ns - self.source_ts[i] > max_age_ns:
            return None
        return self.source_ts[i], self.prices[i]

    def log_return(self, asof_ns: int, horizon_ns: int, max_age_ns: int) -> float | None:
        now = self.price_at(asof_ns, max_age_ns)
        past = self.price_at(asof_ns - horizon_ns, max_age_ns)
        if now is None or past is None:
            return None
        return math.log(now[1] / past[1])

    def realized_vol(
        self, asof_ns: int, window_ns: int, sample_ns: int = NS_PER_S, min_samples: int = 30
    ) -> float | None:
        """Annualised vol from log returns sampled on a fixed grid within the window."""
        n = int(window_ns // sample_ns)
        if n < min_samples:
            return None
        start = asof_ns - n * sample_ns
        i0 = self.index_at(start)
        if i0 < 0:
            return None
        i1 = self.index_at(asof_ns)
        ts = np.asarray(self.source_ts[i0 : i1 + 1], dtype=np.int64)
        px = np.asarray(self.prices[i0 : i1 + 1], dtype=np.float64)
        grid = start + sample_ns * np.arange(n + 1, dtype=np.int64)
        idx = np.searchsorted(ts, grid, side="right") - 1
        sampled = px[np.maximum(idx, 0)]
        r = np.diff(np.log(sampled))
        var_per_sample = float(np.mean(r * r))
        return math.sqrt(var_per_sample * (SECONDS_PER_YEAR * NS_PER_S / sample_ns))

    def integral(self, start_ns: int, end_ns: int) -> float | None:
        """Time integral of the step-interpolated price over [start, end] (price*seconds)."""
        if end_ns <= start_ns:
            return 0.0
        i = self.index_at(start_ns)
        if i < 0:
            return None
        total = 0.0
        t = start_ns
        while True:
            nxt = self.source_ts[i + 1] if i + 1 < len(self.source_ts) else end_ns
            seg_end = min(nxt, end_ns)
            total += self.prices[i] * (seg_end - t) / NS_PER_S
            if seg_end >= end_ns:
                return total
            t = seg_end
            i += 1

    def trim_before(self, ts_ns: int) -> None:
        i = bisect.bisect_left(self.source_ts, ts_ns) - 1
        if i > 0:
            del self.source_ts[:i], self.recv_ts[:i], self.prices[:i]


@dataclass
class MarketState:
    reference_instruments: frozenset[str] = frozenset()
    sequence_free_venues: frozenset[Venue] = frozenset()
    books: BookManager = field(init=False)
    ref: dict[str, RefSeries] = field(default_factory=dict)
    last_trade: dict[str, TradeEvent] = field(default_factory=dict)
    last_recv_ns: dict[str, int] = field(default_factory=dict)
    last_venue_recv_ns: dict[Venue, int] = field(default_factory=dict)
    clock_drift_flagged: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.books = BookManager(sequence_free_venues=self.sequence_free_venues)

    def on_event(self, event: MarketEvent) -> None:
        self.last_recv_ns[event.instrument_id] = event.recv_ts_ns
        self.last_venue_recv_ns[event.venue] = event.recv_ts_ns
        if QualityFlag.CLOCK_DRIFT in event.quality_flags:
            self.clock_drift_flagged[event.instrument_id] = event.recv_ts_ns
        if isinstance(event, BookSnapshotEvent | BookDeltaEvent):
            self.books.apply(event)
            if event.instrument_id in self.reference_instruments:
                builder = self.books.get(event.instrument_id)
                if builder is not None and builder.is_valid:
                    bb, ba = builder.best_bid(), builder.best_ask()
                    if bb is not None and ba is not None:
                        mid = float((bb[0] + ba[0]) / 2)
                        self._series(event.instrument_id).append(
                            event.venue_ts_ns, event.recv_ts_ns, mid
                        )
        elif isinstance(event, TradeEvent):
            self.last_trade[event.instrument_id] = event
            if (
                event.instrument_id in self.reference_instruments
                and self.books.get(event.instrument_id) is None
            ):
                self._series(event.instrument_id).append(
                    event.venue_ts_ns, event.recv_ts_ns, float(event.price)
                )

    def _series(self, instrument_id: str) -> RefSeries:
        s = self.ref.get(instrument_id)
        if s is None:
            s = RefSeries()
            self.ref[instrument_id] = s
        return s

    def book(self, instrument_id: str, max_levels: int | None = None) -> BookSnapshot | None:
        b = self.books.get(instrument_id)
        return None if b is None else b.snapshot(max_levels)

    def age_ns(self, instrument_id: str, now_ns: int) -> int | None:
        t = self.last_recv_ns.get(instrument_id)
        return None if t is None else now_ns - t

    def venue_age_ns(self, venue: Venue, now_ns: int) -> int | None:
        t = self.last_venue_recv_ns.get(venue)
        return None if t is None else now_ns - t
