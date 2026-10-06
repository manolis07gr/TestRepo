"""Collector: runs feed sessions and pollers over one shared recorder/quarantine/book set.

:class:`Collector` runs every feed concurrently. One feed failing (e.g. reconnect
attempts exhausted) marks that feed FAILED and leaves the rest running. The collector
flushes the raw recorder periodically and exposes :meth:`Collector.is_ready` and an
aggregate :meth:`Collector.health_status` (OK / DEGRADED / DOWN).

:func:`build_collector` wires the enabled venues of an :class:`~cma.config.AppConfig`:

* ``kalshi``: a WebSocket session when API credentials exist (the env vars named in
  ``credential_env.key_id`` plus ``private_key_path`` or ``private_key``) and
  ``cryptography`` is installed. Otherwise :class:`KalshiRestBookPoller` polls the
  public orderbook endpoint; this is the default research path, because Kalshi's
  WebSocket needs auth even for public data. Markets come from ``instruments``
  (tickers) plus the open markets of ``series``.
* ``polymarket``: the CLOB ``market`` channel with the 10 s ``PING`` keepalive. Assets
  come from ``instruments`` (token ids) plus the markets of the Gamma event slugs listed
  in ``series``.
* ``coinbase``: ``ticker`` + ``matches`` + ``heartbeat`` for ``instruments`` (products).
* ``binance``: ``aggTrade`` + ``bookTicker`` for ``instruments`` (symbols).
* ``deribit``: polls ``get_book_summary_by_currency`` for ``instruments`` (currencies,
  default BTC and ETH); raw pages are recorded and the quotes validated.

Discovered contracts are upserted into the :class:`~cma.storage.contracts.ContractStore`
(when a database is given), and their tick sizes are registered with the book manager.
Data-quality incidents (outages, gaps, quarantined book updates, failed polls) go to
``data_quality_incidents`` through :class:`~cma.ingestion.health.IncidentLog`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

import httpx

from cma.adapters.base import MalformedPayloadError
from cma.adapters.crypto.binance import BINANCE_WS_URL, BinanceAdapter
from cma.adapters.crypto.binance import build_subscribe as binance_subscribe
from cma.adapters.crypto.coinbase import COINBASE_WS_URL, CoinbaseAdapter
from cma.adapters.crypto.coinbase import build_subscribe as coinbase_subscribe
from cma.adapters.crypto.deribit import DERIBIT_REST_URL, DeribitAdapter, DeribitClient
from cma.adapters.kalshi.client import (
    KALSHI_REST_URL,
    KALSHI_WS_URL,
    KalshiRestClient,
    KalshiSigner,
    signing_available,
    ws_auth_headers,
)
from cma.adapters.kalshi.commands import build_subscriptions
from cma.adapters.kalshi.parser import KalshiAdapter, KalshiSeries
from cma.adapters.kalshi.poller import KalshiRestBookPoller
from cma.adapters.polymarket.client import (
    POLYMARKET_CLOB_URL,
    POLYMARKET_KEEPALIVE,
    POLYMARKET_WS_MARKET_URL,
    PolymarketRestClient,
    build_market_subscription,
)
from cma.adapters.polymarket.parser import PolymarketAdapter
from cma.config import AppConfig, VenueConfig
from cma.domain.enums import Venue
from cma.domain.models import PredictionContract, RawMessage
from cma.domain.time import NS_PER_S, Clock, SystemClock
from cma.ingestion.book import BookManager
from cma.ingestion.dedup import TradeDeduplicator
from cma.ingestion.health import (
    BookHealth,
    ConnectionState,
    FeedHealth,
    HealthStatus,
    IncidentLog,
    aggregate_status,
)
from cma.ingestion.quarantine import Quarantine
from cma.ingestion.rest import HttpFetcher, RateLimiter, RestError, RetryPolicy, SleepFn
from cma.ingestion.ws import (
    EventSink,
    FeedSession,
    PollingSession,
    WebsocketsTransport,
    WsTransport,
)
from cma.storage.contracts import ContractStore
from cma.storage.db import Database
from cma.storage.raw import RawRecorder

log = logging.getLogger(__name__)

type Closer = Callable[[], Awaitable[None] | None]


class Feed(Protocol):
    """What the collector needs from a feed (WebSocket session or REST poller)."""

    name: str
    health: FeedHealth

    async def run(self) -> None: ...

    async def stop(self) -> None: ...

    def is_ready(self, instrument_id: str) -> bool: ...

    def book_health(self, now_ns: int) -> list[BookHealth]: ...


class Collector:
    """Owns the feeds plus shared recorder/quarantine/books; aggregates their health."""

    def __init__(
        self,
        *,
        clock: Clock,
        books: BookManager,
        recorder: RawRecorder | None = None,
        quarantine: Quarantine | None = None,
        contract_store: ContractStore | None = None,
        feeds: Sequence[Feed] = (),
        flush_interval_s: float = 1.0,
        stale_after_s: float | None = 60.0,
        closers: Sequence[Closer] = (),
        incidents: IncidentLog | None = None,
    ) -> None:
        self.clock = clock
        self.incidents = incidents
        self.books = books
        self.recorder = recorder
        self.quarantine = quarantine
        self.contract_store = contract_store
        self._feeds: list[Feed] = list(feeds)
        self._flush_interval_s = flush_interval_s
        self._stale_after_ns = None if stale_after_s is None else int(stale_after_s * NS_PER_S)
        self._closers: list[Closer] = list(closers)
        self._task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def feeds(self) -> tuple[Feed, ...]:
        return tuple(self._feeds)

    def add_feed(self, feed: Feed) -> None:
        if self._running:
            raise RuntimeError("cannot add feeds to a running collector")
        self._feeds.append(feed)

    def add_closer(self, closer: Closer) -> None:
        self._closers.append(closer)

    # ------------------------------------------------------------------ lifecycle

    async def run(self) -> None:
        """Run every feed until all of them stop (see :meth:`stop`)."""
        if self._running:
            raise RuntimeError("collector is already running")
        self._running = True
        tasks = [asyncio.create_task(self._run_feed(f), name=f"feed:{f.name}") for f in self._feeds]
        flusher = asyncio.create_task(self._flush_loop(), name="collector:flush")
        try:
            await asyncio.gather(*tasks)
        finally:
            flusher.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(flusher, *tasks, return_exceptions=True)
            self.flush()
            self._running = False

    async def start(self) -> None:
        """Run in a background task (returns immediately)."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("collector already started")
        self._task = asyncio.get_running_loop().create_task(self.run(), name="collector")
        await asyncio.sleep(0)

    async def stop(self, *, timeout_s: float = 5.0) -> None:
        """Ask every feed to stop; cancel whatever has not stopped after ``timeout_s``."""
        for feed in self._feeds:
            with suppress(Exception):
                await feed.stop()
        task, self._task = self._task, None
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
            except TimeoutError:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self.flush()

    async def aclose(self) -> None:
        """Stop, flush and release owned resources (HTTP clients, recorder)."""
        await self.stop()
        for closer in reversed(self._closers):
            with suppress(Exception):
                result = closer()
                if inspect.isawaitable(result):
                    await result
        if self.recorder is not None:
            self.recorder.close()

    def flush(self) -> None:
        if self.recorder is not None:
            try:
                self.recorder.flush()
            except Exception:
                log.exception("raw recorder flush failed")

    async def _run_feed(self, feed: Feed) -> None:
        try:
            await feed.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("feed %s failed: %s", feed.name, exc)
            feed.health.last_error = str(exc)
            feed.health.set_state(ConnectionState.FAILED, self.clock.now_ns())

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval_s)
            self.flush()

    # ------------------------------------------------------------------ status

    def is_ready(self, instrument_id: str) -> bool:
        """True when some feed is live for ``instrument_id`` and its book is valid."""
        return any(feed.is_ready(instrument_id) for feed in self._feeds)

    def book_health(self) -> list[BookHealth]:
        now = self.clock.now_ns()
        return [b for feed in self._feeds for b in feed.book_health(now)]

    def health_status(self) -> HealthStatus:
        now = self.clock.now_ns()
        return aggregate_status(
            [f.health for f in self._feeds],
            [b for feed in self._feeds for b in feed.book_health(now)],
            now_ns=now,
            stale_after_ns=self._stale_after_ns,
        )

    def health_report(self) -> dict[str, Any]:
        now = self.clock.now_ns()
        return {
            "status": self.health_status().value,
            "now_ns": now,
            "feeds": [f.health.to_dict(now) for f in self._feeds],
            "books": [b.to_dict() for feed in self._feeds for b in feed.book_health(now)],
            "quarantine": self.quarantine.counters() if self.quarantine is not None else {},
            "open_incidents": (
                [i.to_dict() for i in self.incidents.open_incidents()]
                if self.incidents is not None
                else []
            ),
            "recorder": (vars(self.recorder.stats).copy() if self.recorder is not None else {}),
        }


# --------------------------------------------------------------------------------------
# Discovery (which markets/assets to subscribe)
# --------------------------------------------------------------------------------------


class _Registry:
    """Shared side effects of discovery: persist contracts, register tick sizes."""

    def __init__(self, books: BookManager, store: ContractStore | None) -> None:
        self._books = books
        self._store = store

    def register(self, contract: PredictionContract) -> None:
        if self._store is not None:
            try:
                self._store.upsert(contract)
            except Exception:
                log.exception("contract upsert failed for %s", contract.contract_id)
        for instrument in {contract.contract_id, *contract.outcome_instruments.values()}:
            self._books.tick_sizes[instrument] = contract.tick_size


def _native(instrument_id: str) -> str:
    return instrument_id.split(":", 1)[1] if ":" in instrument_id else instrument_id


class KalshiDiscovery:
    """Configured tickers plus the open markets of configured series (TTL-cached)."""

    def __init__(
        self,
        client: KalshiRestClient,
        *,
        tickers: Sequence[str],
        series: Sequence[str],
        clock: Clock,
        registry: _Registry,
        refresh_s: float = 300.0,
        status: str = "open",
    ) -> None:
        self._client = client
        self._static = list(tickers)
        self._series = list(series)
        self._clock = clock
        self._registry = registry
        self._refresh_ns = int(refresh_s * NS_PER_S)
        self._status = status
        self._cache: list[str] | None = None
        self._cache_ns = 0
        self._subscribed: set[str] = set()

    async def tickers(self) -> list[str]:
        now = self._clock.now_ns()
        if self._cache is not None and now - self._cache_ns < self._refresh_ns:
            return self._cache
        try:
            found = await self._discover()
        except (RestError, MalformedPayloadError, httpx.HTTPError) as exc:
            if self._cache is None:
                raise
            log.warning(
                "Kalshi discovery failed (%s); keeping %d cached markets", exc, len(self._cache)
            )
            return self._cache
        self._cache, self._cache_ns = found, now
        return found

    async def _discover(self) -> list[str]:
        out = list(self._static)
        for series_ticker in self._series:
            series: KalshiSeries | None = None
            try:
                series = await self._client.get_series(series_ticker)
            except (RestError, MalformedPayloadError) as exc:
                log.warning("Kalshi series %s metadata unavailable: %s", series_ticker, exc)
            contracts = await self._client.list_markets(
                series_ticker=series_ticker, status=self._status, series=series
            )
            for contract in contracts:
                self._registry.register(contract)
                out.append(contract.native_id)
        return list(dict.fromkeys(out))

    async def subscriptions(self) -> list[str]:
        """Frames for a fresh connection (every currently known market)."""
        tickers = await self.tickers()
        self._subscribed = set(tickers)
        return build_subscriptions(tickers)

    async def new_tickers(self) -> list[str]:
        """Markets listed since the connection subscribed (marked subscribed on return)."""
        fresh = [t for t in await self.tickers() if t not in self._subscribed]
        self._subscribed.update(fresh)
        return fresh


class PolymarketDiscovery:
    """Configured token ids plus the tokens of configured Gamma event slugs."""

    def __init__(
        self,
        client: PolymarketRestClient,
        *,
        assets: Sequence[str],
        event_slugs: Sequence[str],
        clock: Clock,
        registry: _Registry,
        refresh_s: float = 300.0,
    ) -> None:
        self._client = client
        self._static = list(assets)
        self._slugs = list(event_slugs)
        self._clock = clock
        self._registry = registry
        self._refresh_ns = int(refresh_s * NS_PER_S)
        self._cache: list[str] | None = None
        self._cache_ns = 0

    async def assets(self) -> list[str]:
        now = self._clock.now_ns()
        if self._cache is not None and now - self._cache_ns < self._refresh_ns:
            return self._cache
        out = list(self._static)
        try:
            for slug in self._slugs:
                for event in await self._client.list_events(slug=slug, closed=False):
                    for contract in event.markets:
                        self._registry.register(contract)
                        out += [_native(i) for i in contract.outcome_instruments.values()]
        except (RestError, MalformedPayloadError, httpx.HTTPError) as exc:
            if self._cache is None and not self._static:
                raise
            log.warning("Polymarket discovery failed: %s", exc)
            return self._cache if self._cache is not None else list(self._static)
        self._cache, self._cache_ns = list(dict.fromkeys(out)), now
        return self._cache

    async def subscriptions(self) -> list[str]:
        assets = await self.assets()
        return [build_market_subscription(assets)] if assets else []


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


class _Wiring:
    def __init__(
        self,
        *,
        config: AppConfig,
        clock: Clock,
        transport: WsTransport,
        http: httpx.AsyncClient,
        books: BookManager,
        recorder: RawRecorder,
        quarantine: Quarantine,
        trade_dedup: TradeDeduplicator,
        registry: _Registry,
        incidents: IncidentLog,
        sleep: SleepFn | None,
        seed: int,
        on_events: EventSink | None,
        kalshi_poll_interval_s: float,
        deribit_poll_interval_s: float,
        discovery_refresh_s: float,
    ) -> None:
        self.config = config
        self.clock = clock
        self.transport = transport
        self.http = http
        self.books = books
        self.recorder = recorder
        self.quarantine = quarantine
        self.trade_dedup = trade_dedup
        self.registry = registry
        self.incidents = incidents
        self.sleep = sleep
        self.seed = seed
        self.on_events = on_events
        self.kalshi_poll_interval_s = kalshi_poll_interval_s
        self.deribit_poll_interval_s = deribit_poll_interval_s
        self.discovery_refresh_s = discovery_refresh_s

    def rng(self, venue: Venue) -> random.Random:
        return random.Random(f"{self.seed}:{venue.value}")

    def fetcher(self, venue: Venue, vc: VenueConfig) -> HttpFetcher:
        return HttpFetcher(
            self.http,
            clock=self.clock,
            limiter=RateLimiter(vc.rate_limit_per_s, clock=self.clock, sleep=self.sleep),
            retry=RetryPolicy(),
            sleep=self.sleep,
            rng=self.rng(venue),
            connection_id=f"rest:{venue.value.lower()}:{self.clock.now_ns()}",
        )

    def session(self, **kwargs: Any) -> FeedSession:
        venue: Venue = kwargs["adapter"].venue
        return FeedSession(
            transport=self.transport,
            clock=self.clock,
            books=self.books,
            recorder=self.recorder,
            quarantine=self.quarantine,
            trade_dedup=self.trade_dedup,
            on_events=self.on_events,
            rng=self.rng(venue),
            sleep=self.sleep,
            incidents=self.incidents,
            **kwargs,
        )

    # ---------------------------------------------------------------- venues

    def kalshi(self, vc: VenueConfig) -> Feed | None:
        client = KalshiRestClient(
            self.fetcher(Venue.KALSHI, vc),
            base_url=vc.rest_url or KALSHI_REST_URL,
            recorder=self.recorder,
        )
        discovery = KalshiDiscovery(
            client,
            tickers=vc.instruments,
            series=vc.series,
            clock=self.clock,
            registry=self.registry,
            refresh_s=self.discovery_refresh_s,
        )
        if not vc.instruments and not vc.series:
            log.warning("kalshi enabled without instruments or series; skipped")
            return None
        signer = self._kalshi_signer(vc)
        if signer is not None:
            ws_url = vc.ws_url or KALSHI_WS_URL
            log.info("kalshi: authenticated WebSocket feed")
            return self.session(
                name="kalshi-ws",
                url=ws_url,
                adapter=KalshiAdapter(),
                subscriptions=discovery.subscriptions,
                headers=lambda: ws_auth_headers(signer, ws_url),
                on_sequence_gap="resync",
                subscription_refresh=discovery.new_tickers,
                subscription_frames=lambda tickers, first_id: build_subscriptions(
                    tickers, start_id=first_id
                ),
                subscription_refresh_s=min(60.0, self.discovery_refresh_s),
            )
        log.info("kalshi: no usable WebSocket credentials; REST orderbook poller fallback")
        return KalshiRestBookPoller(
            client=client,
            tickers=discovery.tickers,
            clock=self.clock,
            books=self.books,
            interval_s=self.kalshi_poll_interval_s,
            recorder=self.recorder,
            quarantine=self.quarantine,
            trade_dedup=self.trade_dedup,
            on_events=self.on_events,
            sleep=self.sleep,
            incidents=self.incidents,
        )

    def _kalshi_signer(self, vc: VenueConfig) -> KalshiSigner | None:
        key_env = vc.credential_env.get("key_id")
        if not key_env:
            return None
        signer = KalshiSigner.from_env(
            key_id_env=key_env,
            private_key_path_env=vc.credential_env.get("private_key_path"),
            private_key_env=vc.credential_env.get("private_key"),
            clock=self.clock,
        )
        if signer is not None and not signing_available():
            log.warning("kalshi credentials found but 'cryptography' is not installed")
            return None
        return signer

    def polymarket(self, vc: VenueConfig) -> Feed | None:
        if not vc.instruments and not vc.series:
            log.warning("polymarket enabled without instruments or series; skipped")
            return None
        rest_url = vc.rest_url or POLYMARKET_CLOB_URL
        client = PolymarketRestClient(
            self.fetcher(Venue.POLYMARKET, vc), clob_url=rest_url, recorder=self.recorder
        )
        discovery = PolymarketDiscovery(
            client,
            assets=vc.instruments,
            event_slugs=vc.series,
            clock=self.clock,
            registry=self.registry,
            refresh_s=self.discovery_refresh_s,
        )
        return self.session(
            name="polymarket-ws",
            url=vc.ws_url or POLYMARKET_WS_MARKET_URL,
            adapter=PolymarketAdapter(),
            subscriptions=discovery.subscriptions,
            keepalive=POLYMARKET_KEEPALIVE,
        )

    def coinbase(self, vc: VenueConfig) -> Feed | None:
        if not vc.instruments:
            log.warning("coinbase enabled without instruments; skipped")
            return None
        return self.session(
            name="coinbase-ws",
            url=vc.ws_url or COINBASE_WS_URL,
            adapter=CoinbaseAdapter(),
            subscriptions=[coinbase_subscribe(vc.instruments)],
        )

    def binance(self, vc: VenueConfig) -> Feed | None:
        if not vc.instruments:
            log.warning("binance enabled without instruments; skipped")
            return None
        return self.session(
            name="binance-ws",
            url=vc.ws_url or BINANCE_WS_URL,
            adapter=BinanceAdapter(),
            subscriptions=[binance_subscribe(vc.instruments)],
        )

    def deribit(self, vc: VenueConfig) -> Feed | None:
        client = DeribitClient(
            self.fetcher(Venue.DERIBIT, vc), base_url=vc.rest_url or DERIBIT_REST_URL
        )

        async def fetch(currency: str) -> RawMessage:
            page = await client.get_book_summary_page(currency)
            return client.raw_message(currency, page)

        return PollingSession(
            name="deribit-book-summary",
            adapter=DeribitAdapter(),
            fetch=fetch,
            targets=list(vc.instruments) or ["BTC", "ETH"],
            clock=self.clock,
            books=self.books,
            interval_s=self.deribit_poll_interval_s,
            recorder=self.recorder,
            quarantine=self.quarantine,
            sleep=self.sleep,
            incidents=self.incidents,
        )


def build_collector(
    config: AppConfig,
    *,
    clock: Clock | None = None,
    transport: WsTransport | None = None,
    http_client: httpx.AsyncClient | None = None,
    db: Database | None = None,
    raw_root: Path | str | None = None,
    quarantine_root: Path | str | None = None,
    sleep: SleepFn | None = None,
    seed: int | None = None,
    on_events: EventSink | None = None,
    kalshi_poll_interval_s: float = 2.0,
    deribit_poll_interval_s: float = 30.0,
    discovery_refresh_s: float = 300.0,
) -> Collector:
    """A collector for every enabled venue in ``config`` (no I/O happens until run).

    Injected ``transport``/``http_client``/``clock``/``sleep`` make the whole stack
    testable without network access. Resources created here (HTTP client, recorder) are
    released by :meth:`Collector.aclose`.
    """
    clock = clock or SystemClock()
    books = BookManager(sequence_free_venues=frozenset({Venue.POLYMARKET, Venue.REFERENCE}))
    recorder = RawRecorder(Path(raw_root or config.storage.raw_dir), db=db)
    quarantine = Quarantine(Path(quarantine_root or config.storage.quarantine_dir), clock=clock)
    store = ContractStore(db, clock=clock) if db is not None else None
    incidents = IncidentLog(db)
    owns_http = http_client is None
    http = http_client or httpx.AsyncClient(timeout=10.0)
    wiring = _Wiring(
        config=config,
        clock=clock,
        transport=transport or WebsocketsTransport(),
        http=http,
        books=books,
        recorder=recorder,
        quarantine=quarantine,
        trade_dedup=TradeDeduplicator(),
        registry=_Registry(books, store),
        incidents=incidents,
        sleep=sleep,
        seed=config.simulation.seed if seed is None else seed,
        on_events=on_events,
        kalshi_poll_interval_s=kalshi_poll_interval_s,
        deribit_poll_interval_s=deribit_poll_interval_s,
        discovery_refresh_s=discovery_refresh_s,
    )
    builders: dict[Venue, Callable[[VenueConfig], Feed | None]] = {
        Venue.KALSHI: wiring.kalshi,
        Venue.POLYMARKET: wiring.polymarket,
        Venue.COINBASE: wiring.coinbase,
        Venue.BINANCE: wiring.binance,
        Venue.DERIBIT: wiring.deribit,
    }
    collector = Collector(
        clock=clock,
        books=books,
        recorder=recorder,
        quarantine=quarantine,
        contract_store=store,
        closers=[http.aclose] if owns_http else [],
        incidents=incidents,
    )
    for name, vc in sorted(config.venues.items()):
        if not vc.enabled:
            continue
        try:
            venue = Venue(name.upper())
        except ValueError:
            log.warning("unknown venue %r in config; skipped", name)
            continue
        builder = builders.get(venue)
        if builder is None:
            log.warning("venue %s has no live collector; skipped", venue.value)
            continue
        feed = builder(vc)
        if feed is not None:
            collector.add_feed(feed)
    return collector
