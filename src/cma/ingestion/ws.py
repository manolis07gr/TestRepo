"""Live feed sessions: WebSocket lifecycle plus REST-polling fallbacks (scope s.8, T044).

Message pipeline (shared by both session kinds, see :class:`MessagePipeline`)::

    text -> RawMessage(recv_ts from the injected Clock, connection id/seq, idempotency key)
         -> RawRecorder.append        (duplicate -> dropped, never parsed again)
         -> VenueAdapter.parse        (malformed -> Quarantine, session continues)
         -> process_ts stamp, trade dedup
         -> BookManager.apply         (stale/duplicate sequence -> not applied)
         -> on_events(events)         (downstream callback)

:class:`FeedSession` (one WebSocket connection):

* connects through an injectable :class:`WsTransport`. Production uses
  :class:`WebsocketsTransport`; tests use :class:`ScriptedTransport`.
* sends its subscription frames on every (re)connect. A provider may recompute them,
  e.g. after market discovery.
* reconnects with bounded exponential backoff and jitter (seeded RNG, injected sleep,
  ``max_attempts``). The attempt counter resets once a connection delivers data.
* fails closed. A connection start invalidates the session's books (AWAITING_SNAPSHOT);
  a disconnect immediately invalidates them with DISCONNECTED. Signals stay suppressed
  until a fresh snapshot from the new connection makes a book valid again.
  :meth:`FeedSession.is_ready` is "connected AND book valid".
* recovers books after sequence gaps, other integrity failures and quarantined book
  updates. It asks for a snapshot in band when the adapter supports it
  (:class:`~cma.adapters.base.SupportsResync`, e.g. Kalshi ``get_snapshot``).
  Otherwise it waits for a natural snapshot and reconnects after ``resync_timeout_s``.

:class:`PollingSession` polls one REST target at a time, e.g. the Kalshi public
orderbook when no WebSocket credentials exist. It feeds every response through the same
pipeline. A failed poll invalidates that target's book.

Both session kinds report data-quality windows to an optional
:class:`~cma.ingestion.health.IncidentLog`: feed outages, book integrity failures,
quarantined book updates and failed polls.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Literal, Protocol

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from cma.adapters.base import (
    MalformedPayloadError,
    SupportsAttribution,
    SupportsResync,
    VenueAdapter,
    resolve_idempotency_key,
)
from cma.domain.enums import QualityFlag, Venue
from cma.domain.errors import CMAError
from cma.domain.models import (
    BookDeltaEvent,
    BookSnapshotEvent,
    MarketEvent,
    RawMessage,
    TradeEvent,
)
from cma.domain.time import NS_PER_S, Clock, ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.dedup import TradeDeduplicator
from cma.ingestion.health import BookHealth, ConnectionState, FeedHealth, IncidentLog
from cma.ingestion.quarantine import Quarantine
from cma.ingestion.rest import BackoffPolicy, SleepFn
from cma.security import redact
from cma.storage.raw import RawRecorder

log = logging.getLogger(__name__)

type EventSink = Callable[[list[MarketEvent]], None]
type SubscriptionProvider = Callable[[], Awaitable[Sequence[str]]]
type HeaderProvider = Callable[[], Mapping[str, str]]
type TargetProvider = Callable[[], Awaitable[Sequence[str]]]

# book-apply outcomes that leave a book needing a fresh snapshot
_INTEGRITY_FAILURES = frozenset(
    {"sequence_gap", "missing_sequence", "negative_quantity", "crossed"}
)


class TransportClosed(CMAError):
    """The WebSocket connection is closed (by the peer, the network or locally)."""


class FeedExhaustedError(CMAError):
    """A feed gave up after its maximum number of consecutive reconnect attempts."""


class WsConnection(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str:
        """Next text frame; raises :class:`TransportClosed` once the connection is gone."""
        ...

    async def close(self) -> None: ...


class WsTransport(Protocol):
    async def connect(self, url: str, headers: Mapping[str, str]) -> WsConnection: ...


# --------------------------------------------------------------------------------------
# Production transport (websockets >= 13, asyncio implementation)
# --------------------------------------------------------------------------------------


class _WebsocketsConnection:
    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws

    async def send(self, message: str) -> None:
        try:
            await self._ws.send(message)
        except ConnectionClosed as exc:
            raise TransportClosed(str(exc)) from exc

    async def recv(self) -> str:
        try:
            data = await self._ws.recv()
        except ConnectionClosed as exc:
            raise TransportClosed(str(exc)) from exc
        return data if isinstance(data, str) else bytes(data).decode("utf-8")

    async def close(self) -> None:
        await self._ws.close()


class WebsocketsTransport:
    """WebSocket transport on the ``websockets`` library (protocol pings handled by it)."""

    def __init__(
        self,
        *,
        open_timeout_s: float = 10.0,
        ping_interval_s: float | None = 20.0,
        ping_timeout_s: float | None = 20.0,
        max_size: int = 16 * 1024 * 1024,
    ) -> None:
        self._open_timeout_s = open_timeout_s
        self._ping_interval_s = ping_interval_s
        self._ping_timeout_s = ping_timeout_s
        self._max_size = max_size

    async def connect(self, url: str, headers: Mapping[str, str]) -> WsConnection:
        ws = await ws_connect(
            url,
            additional_headers=dict(headers) if headers else None,
            open_timeout=self._open_timeout_s,
            ping_interval=self._ping_interval_s,
            ping_timeout=self._ping_timeout_s,
            max_size=self._max_size,
        )
        return _WebsocketsConnection(ws)


# --------------------------------------------------------------------------------------
# Scripted transport (deterministic test double; also used by collector smoke tests)
# --------------------------------------------------------------------------------------

type ScriptItem = str | BaseException | Callable[[], object]
type ConnectionScript = Sequence[ScriptItem] | BaseException


class ScriptedConnection:
    """Replays a script of frames.

    A ``str`` item is delivered as a frame. An exception instance is raised from
    ``recv``. A callable runs (and is awaited if it returns an awaitable) before the next
    item. When the script runs out, ``recv`` either blocks until the connection is closed
    (``on_exhausted="block"``, a quiet but healthy feed) or reports a close (``"close"``).
    """

    def __init__(
        self,
        items: Sequence[ScriptItem],
        *,
        url: str,
        headers: Mapping[str, str],
        on_exhausted: Literal["block", "close"] = "block",
        clock: ManualClock | None = None,
        step_ns: int = 0,
    ) -> None:
        self.url = url
        self.headers = dict(headers)
        self.sent: list[str] = []
        self.closed = False
        self.delivered = 0
        self._items: deque[ScriptItem] = deque(items)
        self._on_exhausted = on_exhausted
        self._clock = clock
        self._step_ns = step_ns
        self._closed_event = asyncio.Event()

    async def send(self, message: str) -> None:
        if self.closed:
            raise TransportClosed("send on a closed connection")
        self.sent.append(message)

    async def recv(self) -> str:
        while self._items:
            if self.closed:
                raise TransportClosed("connection closed")
            item = self._items.popleft()
            if isinstance(item, str):
                await asyncio.sleep(0)  # yield like a real socket read
                if self._clock is not None and self._step_ns:
                    self._clock.advance(self._step_ns)
                self.delivered += 1
                return item
            if isinstance(item, BaseException):
                self.closed = True
                self._closed_event.set()
                raise item
            result = item()
            if inspect.isawaitable(result):
                await result
        if self.closed:
            raise TransportClosed("connection closed")
        if self._on_exhausted == "close":
            self.closed = True
            raise TransportClosed("script exhausted")
        await self._closed_event.wait()
        raise TransportClosed("connection closed")

    async def close(self) -> None:
        self.closed = True
        self._closed_event.set()


class ScriptedTransport:
    """Hands out :class:`ScriptedConnection` objects in order, per URL if given a mapping.

    A script entry that is an exception makes that ``connect`` attempt fail. When no
    scripts remain, ``connect`` raises ``ConnectionRefusedError``.
    """

    def __init__(
        self,
        scripts: Sequence[ConnectionScript] | Mapping[str, Sequence[ConnectionScript]],
        *,
        on_exhausted: Literal["block", "close"] = "block",
        clock: ManualClock | None = None,
        step_ns: int = 0,
    ) -> None:
        if isinstance(scripts, Mapping):
            self._by_url: dict[str, deque[ConnectionScript]] | None = {
                url: deque(items) for url, items in scripts.items()
            }
            self._queue: deque[ConnectionScript] = deque()
        else:
            self._by_url = None
            self._queue = deque(scripts)
        self._on_exhausted: Literal["block", "close"] = on_exhausted
        self._clock = clock
        self._step_ns = step_ns
        self.attempts: list[tuple[str, dict[str, str]]] = []
        self.connections: list[ScriptedConnection] = []

    def _queue_for(self, url: str) -> deque[ConnectionScript]:
        if self._by_url is None:
            return self._queue
        if url in self._by_url:
            return self._by_url[url]
        for prefix, queue in self._by_url.items():
            if url.startswith(prefix):
                return queue
        return deque()

    async def connect(self, url: str, headers: Mapping[str, str]) -> WsConnection:
        self.attempts.append((url, dict(headers)))
        queue = self._queue_for(url)
        if not queue:
            raise ConnectionRefusedError(f"no scripted connection left for {url}")
        script = queue.popleft()
        if isinstance(script, BaseException):
            raise script
        conn = ScriptedConnection(
            script,
            url=url,
            headers=headers,
            on_exhausted=self._on_exhausted,
            clock=self._clock,
            step_ns=self._step_ns,
        )
        self.connections.append(conn)
        return conn


# --------------------------------------------------------------------------------------
# Shared message pipeline
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineResult:
    """What happened to one raw message."""

    events: list[MarketEvent] = field(default_factory=list)
    duplicate: bool = False
    quarantined: bool = False
    snapshots_applied: list[str] = field(default_factory=list)
    needs_snapshot: list[str] = field(default_factory=list)


class MessagePipeline:
    """record -> parse -> stamp -> dedup -> apply -> deliver, for one venue adapter."""

    def __init__(
        self,
        *,
        adapter: VenueAdapter,
        clock: Clock,
        books: BookManager,
        health: FeedHealth,
        recorder: RawRecorder | None = None,
        quarantine: Quarantine | None = None,
        trade_dedup: TradeDeduplicator | None = None,
        on_events: EventSink | None = None,
    ) -> None:
        self.adapter = adapter
        self.clock = clock
        self.books = books
        self.health = health
        self.recorder = recorder
        self.quarantine = quarantine
        self.trade_dedup = trade_dedup
        self.on_events = on_events
        self.instruments: set[str] = set()
        self.last_snapshot_ns: dict[str, int] = {}

    def handle(self, raw: RawMessage) -> PipelineResult:
        """Record ``raw`` (dropping duplicates) and process it."""
        if self.recorder is not None and not self.recorder.append(raw):
            self.health.duplicates += 1
            return PipelineResult(duplicate=True)
        return self.process(raw)

    def process(self, raw: RawMessage) -> PipelineResult:
        """Parse and apply ``raw`` without recording it (replay entry point)."""
        try:
            events = self.adapter.parse(raw)
        except MalformedPayloadError as exc:
            self._quarantine(raw, exc)
            return PipelineResult(quarantined=True, needs_snapshot=self._invalidate_affected(raw))
        except Exception as exc:  # an adapter bug must not take the feed down
            log.exception("%s adapter failed on %s message", raw.venue.value, raw.stream)
            self._quarantine(raw, exc)
            return PipelineResult(quarantined=True, needs_snapshot=self._invalidate_affected(raw))

        result = PipelineResult()
        process_ts = self.clock.now_ns()
        for event in events:
            if event.process_ts_ns is None:
                event = event.with_process_ts(process_ts)
            if event.source_ts_ns is not None:
                self.health.latency.add(event.recv_ts_ns - event.source_ts_ns)
            if isinstance(event, TradeEvent):
                if self.trade_dedup is not None and self.trade_dedup.is_duplicate(event):
                    self.health.duplicate_trades += 1
                    continue
            elif isinstance(event, BookSnapshotEvent | BookDeltaEvent) and not self._apply_book(
                event, result
            ):
                continue  # stale/duplicate or awaiting a snapshot: not applied, not forwarded
            result.events.append(event)
        if result.events and self.on_events is not None:
            try:
                self.on_events(result.events)
            except Exception:
                self.health.callback_errors += 1
                log.exception("downstream event callback failed")
        return result

    def _apply_book(
        self, event: BookSnapshotEvent | BookDeltaEvent, result: PipelineResult
    ) -> bool:
        instrument = event.instrument_id
        self.instruments.add(instrument)
        try:
            outcome = self.books.apply(event)
        except Exception:
            log.exception("book apply failed for %s", instrument)
            self.books.builder(event.venue, instrument).invalidate(
                QualityFlag.CHECKSUM_MISMATCH, "apply_error"
            )
            self.health.integrity_failures += 1
            result.needs_snapshot.append(instrument)
            return False
        if isinstance(event, BookSnapshotEvent) and outcome.applied:
            self.last_snapshot_ns[instrument] = event.recv_ts_ns
            result.snapshots_applied.append(instrument)
        if outcome.reason in ("sequence_gap", "missing_sequence"):
            self.health.sequence_gaps += 1
        if outcome.reason in _INTEGRITY_FAILURES:
            self.health.integrity_failures += 1
            result.needs_snapshot.append(instrument)
        return outcome.applied

    def _invalidate_affected(self, raw: RawMessage) -> list[str]:
        """Fail closed after a quarantined payload that may have carried a book update.

        A sequenced venue's books detect a lost delta at the next sequence number, but an
        unparseable snapshot would leave them waiting forever, and a sequence-free venue
        (Polymarket) cannot detect a lost update at all. Books the payload targets
        (adapter attribution) are therefore invalidated and re-snapshotted. When the
        target is unknown on a sequence-free venue, every book of this pipeline is.
        """
        affected: list[str] | None = None
        if isinstance(self.adapter, SupportsAttribution):
            try:
                affected = self.adapter.affected_instruments(raw)
            except Exception:
                affected = None
        if affected is None:
            if raw.venue not in self.books.sequence_free_venues:
                return []
            affected = sorted(self.instruments)
        for instrument in affected:
            self.instruments.add(instrument)
            self.books.builder(raw.venue, instrument).invalidate(
                QualityFlag.SEQUENCE_GAP, "quarantined_update"
            )
        return list(affected)

    def _quarantine(self, raw: RawMessage, error: BaseException) -> None:
        self.health.quarantined += 1
        if self.quarantine is not None:
            self.quarantine.put(raw, error)
        else:
            log.warning("dropped malformed %s message: %s", raw.venue.value, error)

    def book_health(self, now_ns: int) -> list[BookHealth]:
        out = []
        for instrument in sorted(self.instruments):
            builder = self.books.get(instrument)
            if builder is not None:
                out.append(
                    BookHealth.from_builder(
                        builder,
                        now_ns=now_ns,
                        last_snapshot_ns=self.last_snapshot_ns.get(instrument),
                    )
                )
        return out


# --------------------------------------------------------------------------------------
# WebSocket session
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeepAlive:
    """Application-level heartbeat (e.g. Polymarket requires a text ``PING`` every 10 s)."""

    interval_s: float
    message: str = "PING"


def _describe(exc: BaseException) -> str:
    return redact(f"{type(exc).__name__}: {exc}")


async def _close_quietly(conn: WsConnection) -> None:
    with suppress(Exception):
        await conn.close()


class FeedSession:
    """One WebSocket feed with fail-closed reconnect/resubscribe semantics (T044)."""

    def __init__(
        self,
        *,
        name: str,
        url: str,
        adapter: VenueAdapter,
        transport: WsTransport,
        clock: Clock,
        books: BookManager,
        subscriptions: Sequence[str] | SubscriptionProvider = (),
        headers: Mapping[str, str] | HeaderProvider | None = None,
        recorder: RawRecorder | None = None,
        quarantine: Quarantine | None = None,
        trade_dedup: TradeDeduplicator | None = None,
        on_events: EventSink | None = None,
        backoff: BackoffPolicy | None = None,
        rng: random.Random | None = None,
        sleep: SleepFn | None = None,
        stream: str = "ws",
        on_sequence_gap: Literal["resync", "reconnect", "ignore"] = "resync",
        invalidate_scope: Literal["venue", "session"] = "venue",
        idle_timeout_s: float | None = None,
        keepalive: KeepAlive | None = None,
        keepalive_sleep: SleepFn | None = None,
        resync_timeout_s: float = 10.0,
        health: FeedHealth | None = None,
        incidents: IncidentLog | None = None,
    ) -> None:
        self.name = name
        self.url = url
        self.stream = stream
        self._incidents = incidents
        self.venue: Venue = adapter.venue
        self._adapter = adapter
        self._transport = transport
        self._clock = clock
        self._books = books
        if callable(subscriptions):
            self._subscriptions: SubscriptionProvider = subscriptions
        else:
            frames = tuple(subscriptions)

            async def _static() -> Sequence[str]:
                return frames

            self._subscriptions = _static
        if headers is None or callable(headers):
            self._headers: HeaderProvider | None = headers
        else:
            fixed = dict(headers)
            self._headers = lambda: fixed
        self._backoff = backoff or BackoffPolicy()
        self._rng = rng or random.Random(0)
        self._sleep: SleepFn = sleep or asyncio.sleep
        self._on_gap = on_sequence_gap
        self._invalidate_scope = invalidate_scope
        self._idle_timeout_s = idle_timeout_s
        self._keepalive = keepalive
        self._keepalive_sleep: SleepFn = keepalive_sleep or asyncio.sleep
        self._resync_timeout_ns = int(resync_timeout_s * NS_PER_S)
        self.health = health or FeedHealth(name=name, venue=self.venue)
        self.pipeline = MessagePipeline(
            adapter=adapter,
            clock=clock,
            books=books,
            health=self.health,
            recorder=recorder,
            quarantine=quarantine,
            trade_dedup=trade_dedup,
            on_events=on_events,
        )
        self._conn: WsConnection | None = None
        self._connected = False
        self._running = False
        self._stop_requested = False
        self._connection_id = ""
        self._conn_seq = 0
        self._conn_counter = 0
        self._cmd_id = 0
        self._outbox: list[str] = []
        self._pending_resync: dict[str, int] = {}
        self._reconnect_requested = False
        self._close_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ public API

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connection_id(self) -> str:
        return self._connection_id

    def is_ready(self, instrument_id: str) -> bool:
        """Connected AND the instrument's book is valid (fresh snapshot since connect)."""
        if not self._connected:
            return False
        builder = self._books.get(instrument_id)
        return builder is not None and builder.is_valid

    async def run(self) -> None:
        """Connect, stream and reconnect until :meth:`stop` (or attempts are exhausted)."""
        if self._running:
            raise RuntimeError(f"feed {self.name} is already running")
        self._running = True
        self._stop_requested = False
        attempt = 0
        first = True
        try:
            while not self._stop_requested:
                if not first:
                    attempt += 1
                    limit = self._backoff.max_attempts
                    if limit is not None and attempt > limit:
                        self.health.set_state(ConnectionState.FAILED, self._clock.now_ns())
                        raise FeedExhaustedError(
                            f"{self.name}: giving up after {limit} reconnect attempts"
                        )
                    delay = self._backoff.delay(attempt, self._rng)
                    self.health.last_backoff_s = delay
                    log.info("%s: reconnect attempt %d in %.3fs", self.name, attempt, delay)
                    await self._sleep(delay)
                    if self._stopping():
                        break
                first = False
                conn = await self._connect()
                if conn is None:
                    continue
                if await self._pump(conn):
                    attempt = 0
        finally:
            self._running = False
            self._connected = False
            if self.health.state is not ConnectionState.FAILED:
                self.health.set_state(ConnectionState.STOPPED, self._clock.now_ns())

    async def stop(self) -> None:
        """Stop after the current frame; closes the live connection to unblock reads."""
        self._stop_requested = True
        conn = self._conn
        if conn is not None:
            await _close_quietly(conn)

    def request_stop(self) -> None:
        """Synchronous :meth:`stop` (for callbacks running inside the event loop)."""
        self._stop_requested = True
        conn = self._conn
        if conn is not None:
            task = asyncio.get_running_loop().create_task(_close_quietly(conn))
            self._close_tasks.add(task)
            task.add_done_callback(self._close_tasks.discard)

    def book_health(self, now_ns: int) -> list[BookHealth]:
        return self.pipeline.book_health(now_ns)

    def _stopping(self) -> bool:
        return self._stop_requested  # a method so type narrowing does not survive awaits

    # ------------------------------------------------------------------ connection

    async def _connect(self) -> WsConnection | None:
        now = self._clock.now_ns()
        self.health.set_state(ConnectionState.CONNECTING, now)
        try:
            headers = dict(self._headers()) if self._headers is not None else {}
            conn = await self._transport.connect(self.url, headers)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.health.record_connect_failure(self._clock.now_ns(), _describe(exc))
            log.warning("%s: connect to %s failed: %s", self.name, self.url, _describe(exc))
            return None
        self._conn_counter += 1
        self._connection_id = f"{self.name}:{now}:{self._conn_counter}"
        self._conn_seq = 0
        self._outbox.clear()
        self._pending_resync.clear()
        self._reconnect_requested = False
        # deltas of a new connection may only extend a snapshot of that same connection
        self._invalidate_books(QualityFlag.AWAITING_SNAPSHOT)
        try:
            frames = list(await self._subscriptions())
            for frame in frames:
                await conn.send(frame)
        except asyncio.CancelledError:
            await _close_quietly(conn)
            raise
        except Exception as exc:
            self.health.record_connect_failure(self._clock.now_ns(), _describe(exc))
            log.warning("%s: subscribe failed: %s", self.name, _describe(exc))
            await _close_quietly(conn)
            return None
        self._cmd_id = len(frames)
        self._conn = conn
        self._connected = True
        self.health.on_connected(self._clock.now_ns())
        if self._incidents is not None:
            self._incidents.close(
                self.venue, "FEED_DISCONNECTED", self._clock.now_ns(), subject=self.name
            )
        log.info(
            "%s: connected (%s), %d subscription frames",
            self.name,
            self._connection_id,
            len(frames),
        )
        return conn

    async def _pump(self, conn: WsConnection) -> bool:
        """Read frames until the connection ends; returns whether any frame arrived."""
        delivered = False
        reason = "closed"
        keepalive = self._start_keepalive(conn)
        try:
            while not self._stop_requested:
                if self._idle_timeout_s is not None:
                    text = await asyncio.wait_for(conn.recv(), timeout=self._idle_timeout_s)
                else:
                    text = await conn.recv()
                delivered = True
                self._on_frame(text)
                while self._outbox:
                    await conn.send(self._outbox.pop(0))
                if self._reconnect_requested:
                    reason = "resync by reconnect"
                    break
            else:
                reason = "stopped"
        except TransportClosed as exc:
            reason = f"closed: {exc}" if not self._stop_requested else "stopped"
        except TimeoutError:
            reason = "idle timeout"
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        except Exception as exc:
            reason = f"error: {_describe(exc)}"
            log.warning("%s: connection error: %s", self.name, reason)
        finally:
            if keepalive is not None:
                keepalive.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await keepalive
            self._conn = None
            self._connected = False
            self._invalidate_books(QualityFlag.DISCONNECTED)
            self.health.on_disconnected(self._clock.now_ns(), reason)
            if self._incidents is not None and not self._stop_requested:
                self._incidents.open(
                    self.venue,
                    "FEED_DISCONNECTED",
                    self._clock.now_ns(),
                    subject=self.name,
                    detail=reason,
                )
            log.info("%s: disconnected (%s)", self.name, reason)
            await _close_quietly(conn)
        return delivered

    def _start_keepalive(self, conn: WsConnection) -> asyncio.Task[None] | None:
        if self._keepalive is None:
            return None
        return asyncio.get_running_loop().create_task(self._keepalive_loop(conn, self._keepalive))

    async def _keepalive_loop(self, conn: WsConnection, keepalive: KeepAlive) -> None:
        while True:
            await self._keepalive_sleep(keepalive.interval_s)
            try:
                await conn.send(keepalive.message)
            except Exception:
                return

    def _invalidate_books(self, flag: QualityFlag) -> None:
        if self._invalidate_scope == "venue":
            self._books.invalidate_venue(self.venue, flag)
            return
        for instrument in self.pipeline.instruments:
            builder = self._books.get(instrument)
            if builder is not None:
                builder.invalidate(flag)

    # ------------------------------------------------------------------ frames

    def _idempotency_key(self, text: str) -> str | None:
        try:
            return self._adapter.idempotency_key(self.stream, text)
        except Exception:  # keys are best effort; never let them drop a frame
            return None

    def _on_frame(self, text: str) -> None:
        now = self._clock.now_ns()
        self._conn_seq += 1
        raw = RawMessage(
            venue=self.venue,
            stream=self.stream,
            recv_ts_ns=now,
            payload=text,
            connection_id=self._connection_id,
            connection_seq=self._conn_seq,
            idempotency_key=resolve_idempotency_key(
                self._idempotency_key(text), self._connection_id
            ),
        )
        self.health.record_message(now)
        result = self.pipeline.handle(raw)
        for instrument in result.snapshots_applied:
            self._pending_resync.pop(instrument, None)
            if self._incidents is not None:
                self._incidents.close_instrument(self.venue, instrument, now)
        for instrument in result.needs_snapshot:
            if self._incidents is not None:
                kind = "QUARANTINED_UPDATE" if result.quarantined else "BOOK_INTEGRITY"
                self._incidents.open(self.venue, kind, now, instrument_id=instrument)
            self._request_resync(instrument, raw, now)
        self._check_resync_timeouts(now)

    def _request_resync(self, instrument: str, raw: RawMessage, now_ns: int) -> None:
        """Get ``instrument`` a fresh snapshot (its book is already invalid: fail closed).

        ``reconnect`` policy: reconnect at once. ``resync`` policy: ask in band when the
        adapter can; otherwise wait for a natural snapshot. Either way, reconnect if none
        arrives within ``resync_timeout_s``, which bounds the reconnect rate.
        """
        if self._on_gap == "ignore" or instrument in self._pending_resync:
            return
        self.health.resyncs += 1
        if self._on_gap == "reconnect":
            log.info("%s: %s needs a snapshot; reconnecting", self.name, instrument)
            self._reconnect_requested = True
            return
        frames: list[str] = []
        if isinstance(self._adapter, SupportsResync):
            self._cmd_id += 1
            try:
                frames = self._adapter.resync_messages(instrument, raw, self._cmd_id)
            except Exception:
                log.exception("%s: building resync frames failed", self.name)
        if frames:
            log.info("%s: requesting in-band snapshot for %s", self.name, instrument)
            self._outbox.extend(frames)
        else:
            log.info("%s: %s awaiting a fresh snapshot", self.name, instrument)
        self._pending_resync[instrument] = now_ns

    def _check_resync_timeouts(self, now_ns: int) -> None:
        for instrument, requested_ns in self._pending_resync.items():
            if now_ns - requested_ns > self._resync_timeout_ns:
                log.warning(
                    "%s: no snapshot for %s after resync; reconnecting", self.name, instrument
                )
                self._reconnect_requested = True
                return


# --------------------------------------------------------------------------------------
# REST polling session
# --------------------------------------------------------------------------------------


class PollingSession:
    """Polls ``fetch(target)`` for every target each ``interval_s`` (REST fallback feed).

    Each response becomes a :class:`RawMessage` built by ``fetch`` and goes through the
    shared pipeline. If a fetch fails after its retries, or its payload is quarantined,
    the target's book (``instrument_of(target)``) is invalidated with DISCONNECTED. The
    book stays unusable until the next successful poll (fail closed).
    """

    def __init__(
        self,
        *,
        name: str,
        adapter: VenueAdapter,
        fetch: Callable[[str], Awaitable[RawMessage]],
        targets: Sequence[str] | TargetProvider,
        clock: Clock,
        books: BookManager,
        interval_s: float,
        instrument_of: Callable[[str], str] | None = None,
        recorder: RawRecorder | None = None,
        quarantine: Quarantine | None = None,
        trade_dedup: TradeDeduplicator | None = None,
        on_events: EventSink | None = None,
        sleep: SleepFn | None = None,
        health: FeedHealth | None = None,
        incidents: IncidentLog | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._incidents = incidents
        self.name = name
        self.venue: Venue = adapter.venue
        self._fetch = fetch
        if callable(targets):
            self._targets: TargetProvider = targets
        else:
            fixed = tuple(targets)

            async def _static() -> Sequence[str]:
                return fixed

            self._targets = _static
        self._clock = clock
        self._books = books
        self.interval_s = interval_s
        self._instrument_of = instrument_of
        self._sleep: SleepFn = sleep or asyncio.sleep
        self.health = health or FeedHealth(name=name, venue=self.venue)
        self.pipeline = MessagePipeline(
            adapter=adapter,
            clock=clock,
            books=books,
            health=self.health,
            recorder=recorder,
            quarantine=quarantine,
            trade_dedup=trade_dedup,
            on_events=on_events,
        )
        self._ok_instruments: set[str] = set()
        self._stop_requested = False
        self.polls = 0

    def is_ready(self, instrument_id: str) -> bool:
        if instrument_id not in self._ok_instruments:
            return False
        builder = self._books.get(instrument_id)
        return builder is not None and builder.is_valid

    async def poll_once(self) -> list[MarketEvent]:
        """One pass over all targets; returns the events delivered downstream."""
        self.polls += 1
        now = self._clock.now_ns()
        try:
            targets = list(await self._targets())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.health.poll_errors += 1
            self.health.last_error = _describe(exc)
            self.health.set_state(ConnectionState.DISCONNECTED, now)
            log.warning("%s: target discovery failed: %s", self.name, _describe(exc))
            return []
        events: list[MarketEvent] = []
        any_ok = False
        for target in targets:
            instrument = self._instrument_of(target) if self._instrument_of else None
            try:
                raw = await self._fetch(target)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.health.poll_errors += 1
                self.health.last_error = _describe(exc)
                log.warning("%s: poll of %s failed: %s", self.name, target, _describe(exc))
                self._mark_failed(instrument)
                continue
            self.health.record_message(raw.recv_ts_ns)
            result = self.pipeline.handle(raw)
            if result.quarantined:
                self._mark_failed(instrument)
                continue
            any_ok = True
            if self.health.state is not ConnectionState.CONNECTED:
                # a pass over many targets can take a minute; one good book means we are up
                self.health.on_connected(raw.recv_ts_ns)
            if instrument is not None:
                self._ok_instruments.add(instrument)
                if self._incidents is not None:
                    self._incidents.close_instrument(self.venue, instrument, raw.recv_ts_ns)
            events.extend(result.events)
        now = self._clock.now_ns()
        if any_ok or not targets:
            if self.health.state is not ConnectionState.CONNECTED:
                self.health.on_connected(now)
        else:
            self.health.set_state(ConnectionState.DISCONNECTED, now)
        return events

    def _mark_failed(self, instrument: str | None, *, stopping: bool = False) -> None:
        if instrument is None:
            return
        if self._incidents is not None and not stopping:
            self._incidents.open(
                self.venue, "POLL_FAILED", self._clock.now_ns(), instrument_id=instrument
            )
        self._ok_instruments.discard(instrument)
        builder = self._books.get(instrument)
        if builder is not None:
            builder.invalidate(QualityFlag.DISCONNECTED, "poll_failed")

    async def run(self) -> None:
        self._stop_requested = False
        self.health.set_state(ConnectionState.CONNECTING, self._clock.now_ns())
        try:
            while not self._stop_requested:
                await self.poll_once()
                if self._stopping():
                    break
                await self._sleep(self.interval_s)
        finally:
            for instrument in list(self._ok_instruments):
                self._mark_failed(instrument, stopping=True)
            self.health.set_state(ConnectionState.STOPPED, self._clock.now_ns())

    async def stop(self) -> None:
        self._stop_requested = True

    def _stopping(self) -> bool:
        return self._stop_requested

    def book_health(self, now_ns: int) -> list[BookHealth]:
        return self.pipeline.book_health(now_ns)
