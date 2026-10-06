"""Kalshi REST orderbook poller: the default research path without WS credentials.

Each poll fetches the public ``GET /markets/{ticker}/orderbook`` for every ticker. The
response text is recorded under stream ``rest:orderbook:<ticker>`` (the body does not
name the market), parsed into a YES-book :class:`BookSnapshotEvent` and applied to the
shared :class:`~cma.ingestion.book.BookManager`.

What a polled snapshot carries:
* ``sequence`` is None: REST books are not sequenced, so they can never be merged with
  WebSocket deltas.
* ``source_ts_ns`` is None, flagged ``SOURCE_TS_MISSING``: the endpoint returns no
  exchange timestamp, and ``recv_ts_ns`` (response arrival) is the only clock.

The snapshot can be up to one interval plus latency old. Staleness gates downstream use
``recv_ts_ns``. A poll that fails after its retries invalidates that market's book
(DISCONNECTED) until the next good poll.
"""

from __future__ import annotations

from collections.abc import Sequence

from cma.adapters.kalshi.client import KalshiRestClient
from cma.adapters.kalshi.parser import KalshiAdapter, kalshi_instrument, orderbook_stream
from cma.domain.enums import Venue
from cma.domain.models import RawMessage
from cma.domain.time import Clock
from cma.ingestion.book import BookManager
from cma.ingestion.dedup import TradeDeduplicator
from cma.ingestion.health import FeedHealth, IncidentLog
from cma.ingestion.quarantine import Quarantine
from cma.ingestion.rest import SleepFn
from cma.ingestion.ws import EventSink, PollingSession, TargetProvider
from cma.storage.raw import RawRecorder


class KalshiRestBookPoller(PollingSession):
    """Polls Kalshi's public orderbook endpoint at ``interval_s`` for each ticker."""

    def __init__(
        self,
        *,
        client: KalshiRestClient,
        tickers: Sequence[str] | TargetProvider,
        clock: Clock,
        books: BookManager,
        interval_s: float = 2.0,
        depth: int | None = None,
        recorder: RawRecorder | None = None,
        quarantine: Quarantine | None = None,
        trade_dedup: TradeDeduplicator | None = None,
        on_events: EventSink | None = None,
        sleep: SleepFn | None = None,
        health: FeedHealth | None = None,
        incidents: IncidentLog | None = None,
        name: str = "kalshi-rest-orderbook",
    ) -> None:
        self._client = client
        self._depth = depth
        super().__init__(
            name=name,
            adapter=KalshiAdapter(),
            fetch=self._fetch,
            targets=tickers,
            clock=clock,
            books=books,
            interval_s=interval_s,
            instrument_of=kalshi_instrument,
            recorder=recorder,
            quarantine=quarantine,
            trade_dedup=trade_dedup,
            on_events=on_events,
            sleep=sleep,
            health=health,
            incidents=incidents,
        )

    async def _fetch(self, ticker: str) -> RawMessage:
        page = await self._client.get_orderbook_page(ticker, depth=self._depth)
        return RawMessage(
            venue=Venue.KALSHI,
            stream=orderbook_stream(ticker),
            recv_ts_ns=page.recv_ts_ns,
            payload=page.text,
            connection_id=self._client.connection_id,
            connection_seq=page.seq,
        )
