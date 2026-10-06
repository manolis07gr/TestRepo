"""Feed and book health (scope s.16).

``FeedHealth`` tracks connection state, message rate, reconnects, sequence gaps,
source->receive latency (p50/p99), last-message age, duplicates and quarantined
payloads. ``BookHealth`` covers valid/crossed/locked/empty state and snapshot age.
``aggregate_status`` rolls everything into one OK / DEGRADED / DOWN status. Every
"now" is passed in explicitly (injected clock), so reports are deterministic in tests.

``IncidentLog`` persists data-quality incidents to ``data_quality_incidents``: feed
outages, sequence gaps, integrity failures, quarantined book updates and failed polls.
Each incident is recorded as a [start, end] window, so affected periods can be excluded
or stress-tested explicitly.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from cma.domain.enums import BookSide, QualityFlag, Venue
from cma.domain.models import stable_id
from cma.domain.time import NS_PER_MS, NS_PER_S
from cma.ingestion.book import L2BookBuilder
from cma.storage.db import Database


class ConnectionState(StrEnum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class HealthStatus(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


def _ms(ns: int) -> float:
    return ns / NS_PER_MS


class LatencyStats:
    """Rolling sample of source->receive latencies (ns), nearest-rank percentiles in ms.

    Negative latencies mean the venue clock is ahead of ours. They are counted, not
    clipped (scope s.8: never silently correct source timestamps).
    """

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._samples: deque[int] = deque(maxlen=capacity)
        self.total = 0
        self.negative = 0

    def add(self, latency_ns: int) -> None:
        self._samples.append(latency_ns)
        self.total += 1
        if latency_ns < 0:
            self.negative += 1

    def __len__(self) -> int:
        return len(self._samples)

    def percentile_ms(self, q: float) -> float | None:
        if not self._samples:
            return None
        if not 0.0 <= q <= 1.0:
            raise ValueError("percentile must be within [0, 1]")
        ordered = sorted(self._samples)
        rank = min(len(ordered), max(1, math.ceil(q * len(ordered))))  # nearest rank
        return _ms(ordered[rank - 1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": len(self._samples),
            "total": self.total,
            "p50_ms": self.percentile_ms(0.5),
            "p99_ms": self.percentile_ms(0.99),
            "max_ms": _ms(max(self._samples)) if self._samples else None,
            "negative": self.negative,
        }


@dataclass
class FeedHealth:
    """Mutable health counters of one feed (WebSocket session or REST poller)."""

    name: str
    venue: Venue
    state: ConnectionState = ConnectionState.IDLE
    connects: int = 0
    reconnects: int = 0
    connect_failures: int = 0
    disconnects: int = 0
    messages: int = 0
    duplicates: int = 0
    duplicate_trades: int = 0
    quarantined: int = 0
    sequence_gaps: int = 0
    integrity_failures: int = 0
    resyncs: int = 0
    callback_errors: int = 0
    poll_errors: int = 0
    last_message_ns: int | None = None
    last_state_change_ns: int | None = None
    last_error: str | None = None
    last_disconnect_reason: str | None = None
    last_backoff_s: float | None = None
    rate_window_ns: int = 10 * NS_PER_S
    latency: LatencyStats = field(default_factory=LatencyStats)
    _recent: deque[int] = field(default_factory=deque, repr=False)

    @property
    def is_up(self) -> bool:
        return self.state is ConnectionState.CONNECTED

    def set_state(self, state: ConnectionState, now_ns: int) -> None:
        if state is not self.state:
            self.state = state
            self.last_state_change_ns = now_ns

    def on_connected(self, now_ns: int) -> None:
        self.connects += 1
        if self.connects > 1:
            self.reconnects += 1
        self.set_state(ConnectionState.CONNECTED, now_ns)

    def on_disconnected(self, now_ns: int, reason: str) -> None:
        self.disconnects += 1
        self.last_disconnect_reason = reason
        self.set_state(ConnectionState.DISCONNECTED, now_ns)

    def record_connect_failure(self, now_ns: int, error: str) -> None:
        self.connect_failures += 1
        self.last_error = error
        self.set_state(ConnectionState.DISCONNECTED, now_ns)

    def record_message(self, now_ns: int) -> None:
        self.messages += 1
        self.last_message_ns = now_ns
        self._recent.append(now_ns)
        self._trim(now_ns)

    def _trim(self, now_ns: int) -> None:
        horizon = now_ns - self.rate_window_ns
        while self._recent and self._recent[0] <= horizon:
            self._recent.popleft()

    def message_rate(self, now_ns: int) -> float:
        """Messages per second over the trailing ``rate_window_ns``."""
        self._trim(now_ns)
        return len(self._recent) / (self.rate_window_ns / NS_PER_S)

    def last_message_age_ms(self, now_ns: int) -> float | None:
        return None if self.last_message_ns is None else _ms(now_ns - self.last_message_ns)

    def to_dict(self, now_ns: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "venue": self.venue.value,
            "state": self.state.value,
            "connects": self.connects,
            "reconnects": self.reconnects,
            "connect_failures": self.connect_failures,
            "disconnects": self.disconnects,
            "messages": self.messages,
            "message_rate_per_s": self.message_rate(now_ns),
            "last_message_age_ms": self.last_message_age_ms(now_ns),
            "duplicates": self.duplicates,
            "duplicate_trades": self.duplicate_trades,
            "quarantined": self.quarantined,
            "sequence_gaps": self.sequence_gaps,
            "integrity_failures": self.integrity_failures,
            "resyncs": self.resyncs,
            "callback_errors": self.callback_errors,
            "poll_errors": self.poll_errors,
            "last_error": self.last_error,
            "last_disconnect_reason": self.last_disconnect_reason,
            "last_backoff_s": self.last_backoff_s,
            "latency": self.latency.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BookHealth:
    """Point-in-time health of one reconstructed book."""

    instrument_id: str
    venue: Venue
    state: str
    valid: bool
    crossed: bool
    locked: bool
    empty: bool
    one_sided: bool
    flags: tuple[str, ...]
    last_sequence: int | None
    bid_levels: int
    ask_levels: int
    snapshot_age_ms: float | None
    update_age_ms: float | None

    @classmethod
    def from_builder(
        cls, builder: L2BookBuilder, *, now_ns: int, last_snapshot_ns: int | None = None
    ) -> BookHealth:
        bids = builder.levels(BookSide.BID)
        asks = builder.levels(BookSide.ASK)
        flags = builder.flags
        updated = builder.last_recv_ts_ns if builder.counters.snapshots > 0 else None
        return cls(
            instrument_id=builder.instrument_id,
            venue=builder.venue,
            state=builder.state.value,
            valid=builder.is_valid,
            crossed=QualityFlag.CROSSED in flags,
            locked=QualityFlag.LOCKED in flags,
            empty=not bids and not asks,
            one_sided=bool(bids) != bool(asks),
            flags=tuple(sorted(f.value for f in flags)),
            last_sequence=builder.last_sequence,
            bid_levels=len(bids),
            ask_levels=len(asks),
            snapshot_age_ms=None if last_snapshot_ns is None else _ms(now_ns - last_snapshot_ns),
            update_age_ms=None if updated is None else _ms(now_ns - updated),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "venue": self.venue.value,
            "state": self.state,
            "valid": self.valid,
            "crossed": self.crossed,
            "locked": self.locked,
            "empty": self.empty,
            "one_sided": self.one_sided,
            "flags": list(self.flags),
            "last_sequence": self.last_sequence,
            "bid_levels": self.bid_levels,
            "ask_levels": self.ask_levels,
            "snapshot_age_ms": self.snapshot_age_ms,
            "update_age_ms": self.update_age_ms,
        }


def aggregate_status(
    feeds: Iterable[FeedHealth],
    books: Iterable[BookHealth] = (),
    *,
    now_ns: int,
    stale_after_ns: int | None = None,
) -> HealthStatus:
    """DOWN: no feed is up. DEGRADED: some feed is down/silent or some book is invalid."""
    feed_list = list(feeds)
    if not feed_list:
        return HealthStatus.DOWN
    up = [f for f in feed_list if f.is_up]
    if not up:
        return HealthStatus.DOWN
    if len(up) < len(feed_list):
        return HealthStatus.DEGRADED
    if stale_after_ns is not None:
        for f in feed_list:
            if f.last_message_ns is None or now_ns - f.last_message_ns > stale_after_ns:
                return HealthStatus.DEGRADED
    if any(not b.valid for b in books):
        return HealthStatus.DEGRADED
    return HealthStatus.OK


@dataclass(frozen=True, slots=True, kw_only=True)
class Incident:
    """A data-quality window: [start_ns, end_ns], where end_ns is None while ongoing."""

    incident_id: str
    venue: Venue
    kind: str
    start_ns: int
    instrument_id: str | None = None
    end_ns: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "venue": self.venue.value,
            "kind": self.kind,
            "instrument_id": self.instrument_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "detail": self.detail,
        }


class IncidentLog:
    """Open/close data-quality incidents; persisted when a database is given.

    One incident per ``(venue, kind, subject)`` is open at a time, where the subject is
    an instrument id or a feed name, so repeated triggers extend the same window.
    Incident ids are deterministic in their contents, so replays do not duplicate rows.
    """

    def __init__(self, db: Database | None = None, *, keep_last: int = 1_000) -> None:
        self._db = db
        self._open: dict[tuple[str, str, str], Incident] = {}
        self.closed: deque[Incident] = deque(maxlen=keep_last)

    def open(
        self,
        venue: Venue,
        kind: str,
        start_ns: int,
        *,
        instrument_id: str | None = None,
        subject: str | None = None,
        detail: str = "",
    ) -> Incident:
        key = (venue.value, kind, subject or instrument_id or "")
        current = self._open.get(key)
        if current is not None:
            return current
        incident = Incident(
            incident_id=stable_id("incident", *key, start_ns),
            venue=venue,
            kind=kind,
            start_ns=start_ns,
            instrument_id=instrument_id,
            detail=detail,
        )
        self._open[key] = incident
        if self._db is not None:
            self._db.execute(
                "INSERT INTO data_quality_incidents (incident_id, venue, instrument_id, kind, "
                "start_ns, end_ns, detail, created_at_ns) VALUES (?, ?, ?, ?, ?, NULL, ?, ?) "
                "ON CONFLICT (incident_id) DO NOTHING",
                (
                    incident.incident_id,
                    venue.value,
                    instrument_id,
                    kind,
                    start_ns,
                    detail,
                    start_ns,
                ),
            )
        return incident

    def close(
        self,
        venue: Venue,
        kind: str,
        end_ns: int,
        *,
        instrument_id: str | None = None,
        subject: str | None = None,
    ) -> Incident | None:
        incident = self._open.pop((venue.value, kind, subject or instrument_id or ""), None)
        if incident is None:
            return None
        done = Incident(
            incident_id=incident.incident_id,
            venue=incident.venue,
            kind=incident.kind,
            start_ns=incident.start_ns,
            instrument_id=incident.instrument_id,
            end_ns=max(end_ns, incident.start_ns),
            detail=incident.detail,
        )
        self.closed.append(done)
        if self._db is not None:
            self._db.execute(
                "UPDATE data_quality_incidents SET end_ns = ? WHERE incident_id = ?",
                (done.end_ns, done.incident_id),
            )
        return done

    def close_instrument(self, venue: Venue, instrument_id: str, end_ns: int) -> list[Incident]:
        """Close every open incident of ``instrument_id`` (its book is valid again)."""
        keys = [k for k in self._open if k[0] == venue.value and k[2] == instrument_id]
        out = []
        for _, kind, subject in keys:
            done = self.close(venue, kind, end_ns, subject=subject)
            if done is not None:
                out.append(done)
        return out

    def open_incidents(self) -> list[Incident]:
        return sorted(self._open.values(), key=lambda i: (i.start_ns, i.incident_id))
