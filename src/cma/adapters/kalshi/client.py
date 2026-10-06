"""Kalshi REST client and request signing (Trade API v2).

* Base URLs (Oct 2026): REST ``https://external-api.kalshi.com/trade-api/v2``
  (recommended; ``https://api.elections.kalshi.com/trade-api/v2`` still works) and WS
  ``wss://external-api-ws.kalshi.com/trade-api/ws/v2``.
* Market data REST endpoints are public. The WebSocket requires API-key auth at the
  handshake, even for public channels. Without credentials, the collector therefore
  falls back to :class:`~cma.adapters.kalshi.poller.KalshiRestBookPoller`.
* Signing: RSA-PSS (MGF1/SHA-256, salt length = digest length) over
  ``timestamp_ms + METHOD + path``. The path has no query string. The signature goes in
  the headers ``KALSHI-ACCESS-KEY`` / ``KALSHI-ACCESS-SIGNATURE`` /
  ``KALSHI-ACCESS-TIMESTAMP``. ``cryptography`` is imported lazily and is needed only
  when a request is actually signed. Secrets come only from the environment
  (:func:`cma.security.load_secret`) and are never logged.
* Paginated listings fetch every page before anything is parsed, recorded or returned
  ("never emit partial page results"). A failure part-way raises and leaves the raw
  store untouched.
"""

from __future__ import annotations

import base64
import importlib.util
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from cma.adapters.base import EventFactory, MalformedPayloadError, as_mapping, load_json, opt_str
from cma.adapters.kalshi.parser import (
    REST_EVENTS_STREAM,
    REST_MARKETS_STREAM,
    REST_SERIES_STREAM,
    REST_TRADES_STREAM,
    KalshiAdapter,
    KalshiEvent,
    KalshiSeries,
    orderbook_stream,
    parse_events_page,
    parse_market_response,
    parse_markets_page,
    parse_series,
    parse_trades_page,
)
from cma.domain.enums import Venue
from cma.domain.errors import CMAError
from cma.domain.models import BookSnapshotEvent, PredictionContract, RawMessage, TradeEvent
from cma.domain.time import NS_PER_MS, Clock
from cma.ingestion.rest import FetchedPage, HttpFetcher, QueryValue, RestError
from cma.security import Secret, load_secret
from cma.storage.raw import RawRecorder

KALSHI_REST_URL: Final = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_WS_URL: Final = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
KALSHI_LEGACY_REST_URL: Final = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_LEGACY_WS_URL: Final = "wss://api.elections.kalshi.com/trade-api/ws/v2"


class KalshiAuthError(CMAError):
    """Kalshi credentials are missing/invalid or signing is unavailable."""


def signing_available() -> bool:
    """Whether the optional ``cryptography`` dependency needed for signing is installed."""
    return importlib.util.find_spec("cryptography") is not None


def _rsa_pss_sha256_signer(private_key_pem: str) -> Callable[[bytes], bytes]:
    try:
        from cryptography.hazmat.primitives import (  # type: ignore[import-not-found,unused-ignore]
            hashes,
            serialization,
        )
        from cryptography.hazmat.primitives.asymmetric import (  # type: ignore[import-not-found,unused-ignore]
            padding,
            rsa,
        )
    except ImportError as exc:
        raise KalshiAuthError(
            "Kalshi request signing needs the 'cryptography' package (RSA-PSS/SHA-256); "
            "install it, or run without Kalshi credentials to use the REST poller fallback"
        ) from exc
    try:
        key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    except (ValueError, TypeError) as exc:
        raise KalshiAuthError("Kalshi private key could not be loaded (expected RSA PEM)") from exc
    if not isinstance(key, rsa.RSAPrivateKey):
        raise KalshiAuthError("Kalshi private key must be an RSA key")
    pss = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)

    def sign(message: bytes) -> bytes:
        signature: bytes = key.sign(message, pss, hashes.SHA256())
        return signature

    return sign


class KalshiSigner:
    """Builds Kalshi auth headers. The key material never appears in ``repr`` or logs."""

    __slots__ = ("_clock", "_key_id", "_private_key", "_sign_fn")

    def __init__(
        self,
        key_id: Secret,
        private_key_pem: Secret,
        *,
        clock: Clock,
        sign_fn: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self._key_id = key_id
        self._private_key = private_key_pem
        self._clock = clock
        self._sign_fn = sign_fn

    @classmethod
    def from_env(
        cls,
        *,
        key_id_env: str,
        private_key_path_env: str | None = None,
        private_key_env: str | None = None,
        clock: Clock,
    ) -> KalshiSigner | None:
        """Signer from environment variables; None when credentials are not configured.

        The key id comes from ``key_id_env``. The PEM comes either from the file named
        by ``private_key_path_env`` or inline from ``private_key_env``.
        """
        key_id = load_secret(key_id_env, required=False)
        if key_id is None:
            return None
        pem: Secret | None = None
        if private_key_env:
            pem = load_secret(private_key_env, required=False)
        if pem is None and private_key_path_env:
            path = load_secret(private_key_path_env, required=False)
            if path is not None:
                try:
                    text = Path(path.reveal()).expanduser().read_text(encoding="utf-8")
                except OSError as exc:
                    raise KalshiAuthError(
                        f"cannot read the Kalshi private key file named by {private_key_path_env}"
                    ) from exc
                pem = Secret("KALSHI_PRIVATE_KEY", text)
        if pem is None:
            return None
        return cls(key_id, pem, clock=clock)

    def __repr__(self) -> str:
        return f"KalshiSigner(key_id={self._key_id!r})"

    def _sign(self, message: bytes) -> bytes:
        if self._sign_fn is None:
            self._sign_fn = _rsa_pss_sha256_signer(self._private_key.reveal())
        return self._sign_fn(message)

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Auth headers for ``method`` on ``path`` (the query string is ignored)."""
        clean_path = urlsplit(path).path or "/"
        timestamp_ms = str(self._clock.now_ns() // NS_PER_MS)
        message = f"{timestamp_ms}{method.upper()}{clean_path}".encode()
        signature = base64.b64encode(self._sign(message)).decode("ascii")
        return {
            "KALSHI-ACCESS-KEY": self._key_id.reveal(),
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }


def ws_auth_headers(signer: KalshiSigner, ws_url: str = KALSHI_WS_URL) -> dict[str, str]:
    """Handshake headers for the WebSocket (signed ``GET`` on the WS path)."""
    return signer.headers("GET", urlsplit(ws_url).path)


class KalshiRestClient:
    """Typed Kalshi REST access (market data endpoints are public; signing is optional)."""

    def __init__(
        self,
        fetcher: HttpFetcher,
        *,
        base_url: str = KALSHI_REST_URL,
        recorder: RawRecorder | None = None,
        signer: KalshiSigner | None = None,
        page_limit: int = 1000,
        max_pages: int = 500,
    ) -> None:
        self._fetcher = fetcher
        self._base_url = base_url.rstrip("/")
        self._recorder = recorder
        self._signer = signer
        self._page_limit = page_limit
        self._max_pages = max_pages
        self._adapter = KalshiAdapter()

    @property
    def connection_id(self) -> str:
        return self._fetcher.connection_id

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def _get(self, path: str, params: Mapping[str, QueryValue] | None = None) -> FetchedPage:
        url = self._url(path)
        headers = self._signer.headers("GET", urlsplit(url).path) if self._signer else None
        return await self._fetcher.get(
            url, params=params, headers=headers, what=f"Kalshi GET {path}"
        )

    def _raw(self, stream: str, page: FetchedPage) -> RawMessage:
        return RawMessage(
            venue=Venue.KALSHI,
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

    async def _all_pages(
        self, path: str, params: Mapping[str, QueryValue]
    ) -> list[tuple[FetchedPage, Any]]:
        """Fetch every cursor page first; any failure raises before results escape."""
        pages: list[tuple[FetchedPage, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            query = dict(params)
            query["limit"] = self._page_limit
            if cursor:
                query["cursor"] = cursor
            page = await self._get(path, query)
            data = load_json(page.text)
            pages.append((page, data))
            cursor = opt_str(as_mapping(data, f"Kalshi {path} page"), "cursor", path)
            if not cursor:
                return pages
            if cursor in seen:
                raise MalformedPayloadError(f"Kalshi {path}: pagination cursor repeated")
            seen.add(cursor)
            if len(pages) >= self._max_pages:
                raise RestError(f"Kalshi {path}: more than {self._max_pages} pages")

    async def list_markets(
        self,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        tickers: Sequence[str] | None = None,
        series: KalshiSeries | None = None,
        fee_schedule_id: str | None = None,
    ) -> list[PredictionContract]:
        """All markets matching the filters (cursor pagination), as contracts."""
        params: dict[str, QueryValue] = {
            "series_ticker": series_ticker,
            "event_ticker": event_ticker,
            "status": status,
            "tickers": ",".join(tickers) if tickers else None,
        }
        pages = await self._all_pages("/markets", params)
        contracts: list[PredictionContract] = []
        for _, data in pages:
            batch, _ = parse_markets_page(
                data, series=series, series_ticker=series_ticker, fee_schedule_id=fee_schedule_id
            )
            contracts.extend(batch)
        self._record([self._raw(REST_MARKETS_STREAM, page) for page, _ in pages])
        return contracts

    async def get_market(
        self, ticker: str, *, series: KalshiSeries | None = None
    ) -> PredictionContract:
        page = await self._get(f"/markets/{ticker}")
        contract = parse_market_response(load_json(page.text), series=series)
        self._record([self._raw(REST_MARKETS_STREAM, page)])
        return contract

    async def get_series(self, series_ticker: str) -> KalshiSeries:
        """Series metadata, including ``fee_type``/``fee_multiplier`` -> fee schedule id."""
        page = await self._get(f"/series/{series_ticker}")
        series = parse_series(load_json(page.text))
        self._record([self._raw(REST_SERIES_STREAM, page)])
        return series

    async def get_events(
        self,
        *,
        series_ticker: str | None = None,
        status: str | None = None,
        with_nested_markets: bool = True,
        series: KalshiSeries | None = None,
    ) -> list[KalshiEvent]:
        params: dict[str, QueryValue] = {
            "series_ticker": series_ticker,
            "status": status,
            "with_nested_markets": "true" if with_nested_markets else None,
        }
        pages = await self._all_pages("/events", params)
        events: list[KalshiEvent] = []
        for _, data in pages:
            batch, _ = parse_events_page(data, series=series)
            events.extend(batch)
        self._record([self._raw(REST_EVENTS_STREAM, page) for page, _ in pages])
        return events

    async def get_orderbook_page(self, ticker: str, *, depth: int | None = None) -> FetchedPage:
        """Raw orderbook response (pollers record and parse it through their pipeline)."""
        return await self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    async def get_orderbook(self, ticker: str, *, depth: int | None = None) -> BookSnapshotEvent:
        page = await self.get_orderbook_page(ticker, depth=depth)
        raw = self._raw(orderbook_stream(ticker), page)
        events = self._adapter.parse(raw)
        self._record([raw])
        snapshot = events[0]
        assert isinstance(snapshot, BookSnapshotEvent)
        return snapshot

    async def get_trades(
        self,
        *,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> list[TradeEvent]:
        """Trades (``min_ts``/``max_ts`` are Unix seconds, as the endpoint expects)."""
        params: dict[str, QueryValue] = {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts}
        pages = await self._all_pages("/markets/trades", params)
        raws = [self._raw(REST_TRADES_STREAM, page) for page, _ in pages]
        trades: list[TradeEvent] = []
        for raw, (_, data) in zip(raws, pages, strict=True):
            batch, _ = parse_trades_page(data, factory=EventFactory.for_raw(raw))
            trades.extend(batch)
        self._record(raws)
        return trades
