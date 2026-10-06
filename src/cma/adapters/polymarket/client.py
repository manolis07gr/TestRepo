"""Polymarket REST clients (Gamma, CLOB, Data API) and WS subscription frames.

Endpoints (October 2026): Gamma ``https://gamma-api.polymarket.com`` (``/markets`` and
``/events``, offset pagination), CLOB ``https://clob.polymarket.com`` (``/book`` and
``/prices-history``), Data API ``https://data-api.polymarket.com/v2/trades`` (cursor
pagination; v1 ``/trades`` is retired on 2026-10-24) and the WebSocket
``wss://ws-subscriptions-clob.polymarket.com/ws/market``. Clients must send a text
``PING`` every 10 s; see :data:`POLYMARKET_KEEPALIVE`.

None of these were exercised against the live venue; validate before relying on them.
Listings fetch all pages before returning or recording anything.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

from cma.adapters.base import EventFactory, MalformedPayloadError, as_list, as_mapping, load_json
from cma.adapters.polymarket.parser import (
    REST_BOOK_STREAM,
    REST_EVENTS_STREAM,
    REST_MARKETS_STREAM,
    REST_PRICES_STREAM,
    REST_TRADES_STREAM,
    PolymarketAdapter,
    PolymarketEvent,
    PricePoint,
    parse_data_trades,
    parse_gamma_event,
    parse_gamma_market,
    parse_prices_history,
)
from cma.domain.enums import Venue
from cma.domain.models import BookSnapshotEvent, PredictionContract, RawMessage, TradeEvent
from cma.ingestion.rest import FetchedPage, HttpFetcher, QueryValue, RestError
from cma.ingestion.ws import KeepAlive
from cma.storage.raw import RawRecorder

log = logging.getLogger(__name__)

POLYMARKET_CLOB_URL: Final = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL: Final = "https://gamma-api.polymarket.com"
POLYMARKET_DATA_URL: Final = "https://data-api.polymarket.com"
POLYMARKET_WS_MARKET_URL: Final = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_KEEPALIVE: Final = KeepAlive(interval_s=10.0, message="PING")


def build_market_subscription(
    asset_ids: Sequence[str], *, custom_feature_enabled: bool = True
) -> str:
    """``{"assets_ids": [...], "type": "market", "custom_feature_enabled": true}``."""
    frame: dict[str, Any] = {"assets_ids": list(asset_ids), "type": "market"}
    if custom_feature_enabled:
        frame["custom_feature_enabled"] = True
    return json.dumps(frame, separators=(",", ":"))


class PolymarketRestClient:
    """Typed access to Gamma listings, CLOB books/history and Data API trades."""

    def __init__(
        self,
        fetcher: HttpFetcher,
        *,
        clob_url: str = POLYMARKET_CLOB_URL,
        gamma_url: str = POLYMARKET_GAMMA_URL,
        data_url: str = POLYMARKET_DATA_URL,
        recorder: RawRecorder | None = None,
        page_size: int = 500,
        max_pages: int = 200,
    ) -> None:
        self._fetcher = fetcher
        self._clob = clob_url.rstrip("/")
        self._gamma = gamma_url.rstrip("/")
        self._data = data_url.rstrip("/")
        self._recorder = recorder
        self._page_size = page_size
        self._max_pages = max_pages
        self._adapter = PolymarketAdapter()
        self.skipped: list[str] = []

    @property
    def connection_id(self) -> str:
        return self._fetcher.connection_id

    def _raw(self, stream: str, page: FetchedPage) -> RawMessage:
        return RawMessage(
            venue=Venue.POLYMARKET,
            stream=stream,
            recv_ts_ns=page.recv_ts_ns,
            payload=page.text,
            connection_id=self._fetcher.connection_id,
            connection_seq=page.seq,
        )

    def _record(self, raws: Sequence[RawMessage]) -> None:
        if self._recorder is not None:
            for raw in raws:
                self._recorder.append(raw)

    async def _offset_pages(
        self, url: str, params: Mapping[str, QueryValue]
    ) -> list[tuple[FetchedPage, list[Any]]]:
        pages: list[tuple[FetchedPage, list[Any]]] = []
        offset = 0
        while True:
            query = {**params, "limit": self._page_size, "offset": offset}
            page = await self._fetcher.get(url, params=query, what=f"Polymarket GET {url}")
            rows = as_list(load_json(page.text), f"Polymarket {url} page")
            pages.append((page, rows))
            if len(rows) < self._page_size:
                return pages
            offset += len(rows)
            if len(pages) >= self._max_pages:
                raise RestError(f"Polymarket {url}: more than {self._max_pages} pages")

    async def list_markets(
        self, *, strict: bool = False, **filters: QueryValue
    ) -> list[PredictionContract]:
        """Gamma ``/markets`` (e.g. ``closed=False``, ``slug=...``) -> contracts.

        Non-binary or malformed markets are skipped and listed in :attr:`skipped`.
        With ``strict=True`` they raise instead.
        """
        pages = await self._offset_pages(f"{self._gamma}/markets", filters)
        contracts = []
        for _, rows in pages:
            for row in rows:
                try:
                    contracts.append(parse_gamma_market(as_mapping(row, "Gamma market")))
                except MalformedPayloadError as exc:
                    if strict:
                        raise
                    self.skipped.append(str(exc))
                    log.warning("skipping Gamma market: %s", exc)
        self._record([self._raw(REST_MARKETS_STREAM, page) for page, _ in pages])
        return contracts

    async def list_events(self, **filters: QueryValue) -> list[PolymarketEvent]:
        """Gamma ``/events`` (e.g. ``slug=...``, ``closed=False``) with nested markets."""
        pages = await self._offset_pages(f"{self._gamma}/events", filters)
        events = [
            parse_gamma_event(as_mapping(row, "Gamma event")) for _, rows in pages for row in rows
        ]
        self._record([self._raw(REST_EVENTS_STREAM, page) for page, _ in pages])
        return events

    async def get_book(self, token_id: str) -> BookSnapshotEvent:
        """CLOB ``/book?token_id=`` -> snapshot of that token's book."""
        page = await self._fetcher.get(
            f"{self._clob}/book", params={"token_id": token_id}, what="Polymarket GET /book"
        )
        raw = self._raw(REST_BOOK_STREAM, page)
        events = self._adapter.parse(raw)
        self._record([raw])
        snapshot = events[0]
        assert isinstance(snapshot, BookSnapshotEvent)
        return snapshot

    async def get_prices_history(
        self,
        token_id: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = None,
        fidelity: int | None = None,
    ) -> list[PricePoint]:
        """CLOB ``/prices-history`` (``start_ts``/``end_ts`` in Unix seconds)."""
        page = await self._fetcher.get(
            f"{self._clob}/prices-history",
            params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "interval": interval,
                "fidelity": fidelity,
            },
            what="Polymarket GET /prices-history",
        )
        points = parse_prices_history(load_json(page.text))
        self._record([self._raw(REST_PRICES_STREAM, page)])
        return points

    async def get_trades(
        self,
        *,
        condition_id: str | None = None,
        token_id: str | None = None,
        limit: int = 500,
    ) -> list[TradeEvent]:
        """Data API ``/v2/trades`` (cursor pagination) -> trades without aggressor side."""
        url = f"{self._data}/v2/trades"
        cursor: str | None = None
        collected: list[tuple[RawMessage, Any]] = []
        seen: set[str] = set()
        while True:
            params: dict[str, QueryValue] = {
                "condition_id": condition_id,
                "token_id": token_id,
                "limit": limit,
                "cursor": cursor,
            }
            page = await self._fetcher.get(url, params=params, what="Polymarket GET /v2/trades")
            raw = self._raw(REST_TRADES_STREAM, page)
            data = load_json(page.text)
            _, cursor = parse_data_trades(data, EventFactory.for_raw(raw))
            collected.append((raw, data))
            if not cursor:
                break
            if cursor in seen:
                raise MalformedPayloadError("Polymarket /v2/trades: pagination cursor repeated")
            seen.add(cursor)
            if len(collected) >= self._max_pages:
                raise RestError(f"Polymarket /v2/trades: more than {self._max_pages} pages")
        trades: list[TradeEvent] = []
        for raw, data in collected:
            batch, _ = parse_data_trades(data, EventFactory.for_raw(raw))
            trades.extend(batch)
        self._record([raw for raw, _ in collected])
        return trades
