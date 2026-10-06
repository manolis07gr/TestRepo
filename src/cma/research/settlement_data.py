"""Historical inputs for the hold-to-settlement study: settled Kalshi BTC markets with their
1-minute YES bid/ask candles, Coinbase BTC-USD 1-minute candles and Deribit DVOL.

Everything is fetched once into a data directory (``data/settlement/``, git-ignored) as
JSON lines, and fetching is resumable: a re-run skips markets and chunks already on disk.
Kalshi market data is public; when ``KALSHI_API_KEY_ID`` and ``KALSHI_PRIVATE_KEY`` are set
the requests are signed, which only raises the read limit. Nothing here can place an order.

Files (one JSON document per line):

* ``kalshi_markets_<SERIES>.jsonl``  settled markets: strike, open/close, result
* ``kalshi_candles_<SERIES>.jsonl``  ``{"ticker", "candles": [[end_ts, bid_o, bid_h, bid_l,
  bid_c, ask_o, ask_h, ask_l, ask_c, volume], ...]}`` (dollars, ``null`` when absent)
* ``coinbase_btcusd_1m.jsonl``       ``{"start", "rows": [[ts, o, h, l, c, v], ...]}``
* ``deribit_dvol_1h.jsonl``          ``{"start", "rows": [[ts, close_pct], ...]}`` (hourly:
  Deribit serves 1-minute DVOL only from about May 2026, hourly over the whole period)

Kalshi serves markets settled before its historical cutoff only from ``/historical/...``;
the two endpoints spell candle fields differently (``close_dollars`` / ``close``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from cma.adapters.kalshi import KalshiSigner
from cma.adapters.kalshi.client import KALSHI_REST_URL
from cma.domain.time import SystemClock
from cma.ingestion.rest import (
    HttpFetcher,
    HttpStatusError,
    RateLimiter,
    RetryExhaustedError,
    RetryPolicy,
)

log = logging.getLogger(__name__)

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
DVOL_HISTORY_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
COINBASE_CHUNK_MIN = 300  # Coinbase returns at most 300 candles per request
DVOL_RESOLUTION_S = 3600
DVOL_CHUNK_BARS = 1000
SECONDS_PER_MIN = 60

type Candle = tuple[
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float,
]


@dataclass(frozen=True)
class SettledMarket:
    ticker: str
    event_ticker: str
    series: str
    open_ts: int  # unix seconds
    close_ts: int
    strike_type: str
    floor_strike: float | None
    cap_strike: float | None
    result: str  # "yes" | "no"
    expiration_value: float | None
    archived: bool  # served from /historical


def _ts(text: str | None) -> int | None:
    if not text:
        return None
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def _float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def market_from_api(m: Mapping[str, Any], *, series: str, archived: bool) -> SettledMarket | None:
    """A settled YES/NO market, or None when it is not usable (voided, missing times)."""
    result = str(m.get("result") or "").lower()
    open_ts, close_ts = _ts(m.get("open_time")), _ts(m.get("close_time"))
    if result not in {"yes", "no"} or open_ts is None or close_ts is None:
        return None
    return SettledMarket(
        ticker=str(m["ticker"]),
        event_ticker=str(m.get("event_ticker") or ""),
        series=series,
        open_ts=open_ts,
        close_ts=close_ts,
        strike_type=str(m.get("strike_type") or ""),
        floor_strike=_float(m.get("floor_strike")),
        cap_strike=_float(m.get("cap_strike")),
        result=result,
        expiration_value=_float(m.get("expiration_value")),
        archived=archived,
    )


def _price(side: Mapping[str, Any] | None, key: str) -> float | None:
    """A dollar price from either spelling (``close_dollars`` live, ``close`` archived)."""
    if not side:
        return None
    v = side.get(f"{key}_dollars")
    if v is None:
        v = side.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            return v / 100.0  # legacy integer cents
    return _float(v)


def candle_from_api(c: Mapping[str, Any]) -> Candle:
    bid, ask = c.get("yes_bid"), c.get("yes_ask")
    vol = _float(c.get("volume_fp", c.get("volume"))) or 0.0
    return (
        int(c["end_period_ts"]),
        _price(bid, "open"),
        _price(bid, "high"),
        _price(bid, "low"),
        _price(bid, "close"),
        _price(ask, "open"),
        _price(ask, "high"),
        _price(ask, "low"),
        _price(ask, "close"),
        vol,
    )


# ----------------------------------------------------------------------------- files


def markets_path(data_dir: Path, series: str) -> Path:
    return data_dir / f"kalshi_markets_{series}.jsonl"


def candles_path(data_dir: Path, series: str) -> Path:
    return data_dir / f"kalshi_candles_{series}.jsonl"


def coinbase_path(data_dir: Path) -> Path:
    return data_dir / "coinbase_btcusd_1m.jsonl"


def dvol_path(data_dir: Path) -> Path:
    return data_dir / "deribit_dvol_1h.jsonl"


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:  # a line cut short by an interrupted run
                continue


def _open_append(path: Path) -> Any:
    """Append handle that starts on a fresh line: a run cut off mid-write leaves a partial
    last line, which the loaders skip, but a record glued onto it would be lost too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as tail:
            tail.seek(-1, 2)
            needs_newline = tail.read(1) != b"\n"
    fh = path.open("a", encoding="utf-8")
    if needs_newline:
        fh.write("\n")
    return fh


def load_markets(data_dir: Path, series: str) -> list[SettledMarket]:
    return [SettledMarket(**d) for d in _lines(markets_path(data_dir, series))]


def load_candles(data_dir: Path, series: str) -> dict[str, list[Candle]]:
    out: dict[str, list[Candle]] = {}
    for d in _lines(candles_path(data_dir, series)):
        out[str(d["ticker"])] = [tuple(row) for row in d["candles"]]
    return out


def load_chunked_rows(path: Path) -> list[list[float]]:
    """Rows of every chunk, de-duplicated on the timestamp and sorted."""
    rows: dict[float, list[float]] = {}
    for d in _lines(path):
        for r in d["rows"]:
            rows[r[0]] = r
    return [rows[k] for k in sorted(rows)]


# ----------------------------------------------------------------------------- fetching


class _Http:
    """One rate-limited, retrying fetcher per host, plus optional Kalshi signing."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        rate_per_s: float,
        signer: Any = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        clock = SystemClock()
        self._fetcher = HttpFetcher(
            client,
            clock=clock,
            limiter=RateLimiter(rate_per_s, clock=clock, burst=max(1.0, rate_per_s)),
            retry=retry or RetryPolicy(max_attempts=8),
        )
        self._signer = signer

    async def json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        headers = self._signer.headers("GET", urlsplit(url).path) if self._signer else None
        page = await self._fetcher.get(url, params=params, headers=headers)
        return json.loads(page.text)


def kalshi_signer_from_env() -> KalshiSigner | None:
    try:
        return KalshiSigner.from_env(
            key_id_env="KALSHI_API_KEY_ID",
            private_key_env="KALSHI_PRIVATE_KEY",
            private_key_path_env="KALSHI_PRIVATE_KEY_PATH",
            clock=SystemClock(),
        )
    except Exception:  # unloadable key: fall back to public, unsigned reads
        log.warning("Kalshi credentials present but unusable; fetching unsigned")
        return None


async def fetch_markets(
    http: _Http,
    series: str,
    *,
    since_ts: int,
    data_dir: Path,
    base_url: str = KALSHI_REST_URL,
) -> list[SettledMarket]:
    """Every settled market of ``series`` that closed at or after ``since_ts``.

    Both listings come newest first; paging stops once a page reaches back past
    ``since_ts``. The file is rewritten as a whole (listing is cheap)."""
    found: dict[str, SettledMarket] = {}
    for path, archived in (("/markets", False), ("/historical/markets", True)):
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"series_ticker": series, "limit": 1000, "cursor": cursor}
            if not archived:
                params |= {"status": "settled", "min_close_ts": since_ts}
            doc = await http.json(f"{base_url}{path}", params)
            page = doc.get("markets") or []
            oldest: int | None = None
            for raw in page:
                m = market_from_api(raw, series=series, archived=archived)
                close = _ts(raw.get("close_time"))
                if close is not None:
                    oldest = close if oldest is None else min(oldest, close)
                if m is not None and m.close_ts >= since_ts:
                    found.setdefault(m.ticker, m)
            cursor = doc.get("cursor") or None
            if not page or cursor is None or (oldest is not None and oldest < since_ts):
                break
    markets = sorted(found.values(), key=lambda m: (m.close_ts, m.ticker))
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = markets_path(data_dir, series).with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for m in markets:
            fh.write(json.dumps(asdict(m), separators=(",", ":")) + "\n")
    tmp.replace(markets_path(data_dir, series))
    return markets


async def fetch_candles(
    http: _Http,
    markets: Sequence[SettledMarket],
    *,
    data_dir: Path,
    concurrency: int = 8,
    base_url: str = KALSHI_REST_URL,
    progress_every: int = 500,
) -> int:
    """1-minute candles over each market's life; appends, skipping tickers on disk."""
    if not markets:
        return 0
    path = candles_path(data_dir, markets[0].series)
    done = {str(d["ticker"]) for d in _lines(path)}
    todo = [m for m in markets if m.ticker not in done]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    written = 0

    async def one(m: SettledMarket, fh: Any) -> None:
        nonlocal written
        if m.archived:
            url = f"{base_url}/historical/markets/{m.ticker}/candlesticks"
        else:
            url = f"{base_url}/series/{m.series}/markets/{m.ticker}/candlesticks"
        params = {"start_ts": m.open_ts, "end_ts": m.close_ts, "period_interval": 1}
        async with sem:
            try:
                doc = await http.json(url, params)
            except HttpStatusError as exc:  # e.g. 404 for a market without history
                log.warning("candles %s: %s", m.ticker, exc)
                doc = {"candlesticks": []}
            except RetryExhaustedError as exc:  # not written: retried on the next run
                log.warning("candles %s: %s", m.ticker, exc)
                return
        rows = [candle_from_api(c) for c in doc.get("candlesticks") or []]
        async with lock:
            fh.write(json.dumps({"ticker": m.ticker, "candles": rows}, separators=(",", ":")))
            fh.write("\n")
            written += 1
            if written % progress_every == 0:
                fh.flush()
                log.info("candles: %d / %d", written, len(todo))

    with _open_append(path) as fh:
        await asyncio.gather(*(one(m, fh) for m in todo))
    return written


def _chunk_starts(start_ts: int, end_ts: int, minutes: int) -> list[int]:
    step = minutes * SECONDS_PER_MIN
    first = start_ts - start_ts % SECONDS_PER_MIN
    return list(range(first, end_ts, step))


async def _fetch_chunks(
    path: Path,
    starts: Sequence[int],
    fetch_one: Any,
    *,
    concurrency: int,
) -> int:
    done = {int(d["start"]) for d in _lines(path)}
    todo = [s for s in starts if s not in done]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(start: int, fh: Any) -> None:
        async with sem:
            try:
                rows = await fetch_one(start)
            except (HttpStatusError, RetryExhaustedError) as exc:  # retried on the next run
                log.warning("%s chunk %d: %s", path.name, start, exc)
                return
        async with lock:
            fh.write(json.dumps({"start": start, "rows": rows}, separators=(",", ":")) + "\n")

    with _open_append(path) as fh:
        await asyncio.gather(*(one(s, fh) for s in todo))
    return len(todo)


async def fetch_coinbase(
    http: _Http, *, start_ts: int, end_ts: int, data_dir: Path, concurrency: int = 4
) -> int:
    async def one(start: int) -> list[list[float]]:
        end = start + (COINBASE_CHUNK_MIN - 1) * SECONDS_PER_MIN
        params = {
            "granularity": SECONDS_PER_MIN,
            "start": datetime.fromtimestamp(start, UTC).isoformat(),
            "end": datetime.fromtimestamp(end, UTC).isoformat(),
        }
        rows = await http.json(COINBASE_CANDLES_URL, params)
        # [time, low, high, open, close, volume], newest first -> [ts, o, h, l, c, v]
        return sorted(
            [
                [int(r[0]), float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])]
                for r in rows
                if start <= int(r[0]) <= end
            ]
        )

    starts = _chunk_starts(start_ts, end_ts, COINBASE_CHUNK_MIN)
    return await _fetch_chunks(coinbase_path(data_dir), starts, one, concurrency=concurrency)


async def fetch_dvol(
    http: _Http, *, start_ts: int, end_ts: int, data_dir: Path, concurrency: int = 4
) -> int:
    async def one(start: int) -> list[list[float]]:
        lo_ms = start * 1000
        hi_ms = (start + (DVOL_CHUNK_BARS - 1) * DVOL_RESOLUTION_S) * 1000
        rows: dict[int, float] = {}
        end_ms: int | None = hi_ms
        while end_ms is not None and end_ms >= lo_ms:
            params = {
                "currency": "BTC",
                "resolution": DVOL_RESOLUTION_S,
                "start_timestamp": lo_ms,
                "end_timestamp": end_ms,
            }
            doc = await http.json(DVOL_HISTORY_URL, params)
            result = doc.get("result") or {}
            data = result.get("data") or []
            for r in data:
                if lo_ms <= int(r[0]) <= hi_ms:
                    rows[int(r[0]) // 1000] = float(r[4])
            cont = result.get("continuation")
            if not data or cont is None or int(cont) >= end_ms:
                break
            end_ms = int(cont)
        return [[t, rows[t]] for t in sorted(rows)]

    first = start_ts - start_ts % DVOL_RESOLUTION_S
    starts = list(range(first, end_ts, DVOL_CHUNK_BARS * DVOL_RESOLUTION_S))
    return await _fetch_chunks(dvol_path(data_dir), starts, one, concurrency=concurrency)


async def fetch_all(
    *,
    series: Sequence[str],
    since_ts: int,
    until_ts: int,
    data_dir: Path,
    kalshi_rate_per_s: float = 10.0,
    signed: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Fetch (or complete) every input; returns counts for the log."""
    signer = kalshi_signer_from_env() if signed else None
    summary: dict[str, Any] = {"signed": signer is not None}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0), headers={"User-Agent": "cma-research"}, transport=transport
    ) as cl:
        kalshi = _Http(cl, rate_per_s=kalshi_rate_per_s, signer=signer)
        coinbase = _Http(cl, rate_per_s=4.0)
        deribit = _Http(cl, rate_per_s=4.0)
        for s in series:
            markets = await fetch_markets(kalshi, s, since_ts=since_ts, data_dir=data_dir)
            log.info("%s: %d settled markets since %s", s, len(markets), since_ts)
            summary[f"{s}_markets"] = len(markets)
            summary[f"{s}_candles_new"] = await fetch_candles(kalshi, markets, data_dir=data_dir)
        summary["coinbase_chunks_new"] = await fetch_coinbase(
            coinbase, start_ts=since_ts - 3 * 3600, end_ts=until_ts, data_dir=data_dir
        )
        summary["dvol_chunks_new"] = await fetch_dvol(
            deribit, start_ts=since_ts - 3 * 3600, end_ts=until_ts, data_dir=data_dir
        )
    return summary
