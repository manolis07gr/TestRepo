"""Kalshi REST client (mocked transport), request signing and the REST book poller."""

from __future__ import annotations

import asyncio
import base64
import logging
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from cma.adapters.kalshi import (
    KALSHI_REST_URL,
    KALSHI_WS_URL,
    KalshiAuthError,
    KalshiRestBookPoller,
    KalshiRestClient,
    KalshiSigner,
    signing_available,
    ws_auth_headers,
)
from cma.domain.enums import BookSide, QualityFlag, Venue
from cma.domain.time import ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.health import ConnectionState
from cma.ingestion.rest import HttpFetcher, RetryPolicy
from cma.security import Secret, redact
from cma.storage.raw import RawReader, RawRecorder
from tests.unit.adapters.helpers import (
    KALSHI_INST,
    KALSHI_TICKER,
    T0_NS,
    FakeSleep,
    fixture_text,
    mock_client,
)

pytestmark = pytest.mark.unit
D = Decimal


def make_client(
    handler: httpx.MockTransport, clock: ManualClock, recorder: RawRecorder | None = None
) -> KalshiRestClient:
    fetcher = HttpFetcher(
        mock_client(handler), clock=clock, sleep=FakeSleep(clock), connection_id="rest:test"
    )
    return KalshiRestClient(fetcher, recorder=recorder)


def test_default_urls_are_the_external_api_hosts() -> None:
    assert KALSHI_REST_URL == "https://external-api.kalshi.com/trade-api/v2"
    assert KALSHI_WS_URL == "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


def test_list_markets_follows_cursor_and_records_all_pages(tmp_path: Path) -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        assert request.url.path == "/trade-api/v2/markets"
        page = 2 if request.url.params.get("cursor") else 1
        return httpx.Response(200, text=fixture_text("kalshi", f"markets_page_{page}.json"))

    clock = ManualClock(T0_NS)
    recorder = RawRecorder(tmp_path)

    async def go() -> list[str]:
        client = make_client(httpx.MockTransport(handler), clock, recorder)
        contracts = await client.list_markets(series_ticker="KXBTCD", status="open")
        return [c.native_id for c in contracts]

    assert asyncio.run(go()) == [KALSHI_TICKER, "KXBTCD-26OCT0617-T111999.99"]
    assert seen[0].params["series_ticker"] == "KXBTCD"
    assert seen[0].params["status"] == "open"
    assert seen[1].params["cursor"] == "CgwI2e3JxgYQ0"
    recorder.close()
    recorded = list(RawReader(tmp_path).iter_messages())
    assert [r.stream for r in recorded] == ["rest:markets", "rest:markets"]
    assert [r.connection_seq for r in recorded] == [1, 2]


def test_get_orderbook_series_and_trades() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/trade-api/v2")
        if path == f"/markets/{KALSHI_TICKER}/orderbook":
            return httpx.Response(200, text=fixture_text("kalshi", "orderbook_rest_fp.json"))
        if path == "/series/KXBTCD":
            return httpx.Response(200, text=fixture_text("kalshi", "series.json"))
        if path == "/markets/trades":
            assert request.url.params["min_ts"] == "1791318600"
            return httpx.Response(200, text=fixture_text("kalshi", "trades_page.json"))
        return httpx.Response(404)

    clock = ManualClock(T0_NS)

    async def go() -> None:
        client = make_client(httpx.MockTransport(handler), clock)
        book = await client.get_orderbook(KALSHI_TICKER)
        assert book.instrument_id == KALSHI_INST
        assert book.bids[0].price == D("0.45")
        series = await client.get_series("KXBTCD")
        assert series.fee_schedule_id == "kalshi-standard"
        trades = await client.get_trades(ticker=KALSHI_TICKER, min_ts=1_791_318_600)
        assert len(trades) == 2
        assert trades[0].recv_ts_ns == T0_NS

    asyncio.run(go())


def test_signer_headers_sign_timestamp_method_and_path_without_query() -> None:
    clock = ManualClock(T0_NS + 123_456_789)
    signed: list[bytes] = []

    def fake_sign(message: bytes) -> bytes:
        signed.append(message)
        return b"signature-bytes"

    signer = KalshiSigner(
        Secret("KALSHI_API_KEY_ID", "key-id-123"),
        Secret("KALSHI_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"),
        clock=clock,
        sign_fn=fake_sign,
    )
    headers = signer.headers("get", "/trade-api/v2/portfolio/balance?limit=5")
    assert signed == [b"1791318600123GET/trade-api/v2/portfolio/balance"]
    assert headers == {
        "KALSHI-ACCESS-KEY": "key-id-123",
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(b"signature-bytes").decode(),
        "KALSHI-ACCESS-TIMESTAMP": "1791318600123",
    }
    ws = ws_auth_headers(signer)
    assert signed[-1] == b"1791318600123GET/trade-api/ws/v2"
    assert ws["KALSHI-ACCESS-TIMESTAMP"] == "1791318600123"
    assert "key-id-123" not in repr(signer)
    assert "PRIVATE KEY" not in repr(signer)
    assert "key-id-123" not in redact(f"connecting with {headers}")


@pytest.mark.skipif(signing_available(), reason="cryptography installed; real signing works")
def test_signing_without_cryptography_raises_a_clear_error() -> None:
    signer = KalshiSigner(Secret("K", "key-id"), Secret("P", "pem"), clock=ManualClock(T0_NS))
    with pytest.raises(KalshiAuthError, match="cryptography"):
        signer.headers("GET", "/trade-api/ws/v2")


@pytest.mark.skipif(not signing_available(), reason="cryptography not installed")
def test_rsa_pss_signature_verifies() -> None:  # pragma: no cover - environment dependent
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    signer = KalshiSigner(Secret("K", "kid"), Secret("P", pem), clock=ManualClock(T0_NS))
    headers = signer.headers("GET", "/trade-api/ws/v2")
    key.public_key().verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
        f"{headers['KALSHI-ACCESS-TIMESTAMP']}GET/trade-api/ws/v2".encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_signer_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clock = ManualClock(T0_NS)
    monkeypatch.delenv("CMA_TEST_KALSHI_KEY", raising=False)
    assert KalshiSigner.from_env(key_id_env="CMA_TEST_KALSHI_KEY", clock=clock) is None
    pem = tmp_path / "kalshi.pem"
    pem.write_text("-----BEGIN PRIVATE KEY-----\nxyz\n-----END PRIVATE KEY-----\n")
    monkeypatch.setenv("CMA_TEST_KALSHI_KEY", "kid-777")
    monkeypatch.setenv("CMA_TEST_KALSHI_PEM_PATH", str(pem))
    signer = KalshiSigner.from_env(
        key_id_env="CMA_TEST_KALSHI_KEY",
        private_key_path_env="CMA_TEST_KALSHI_PEM_PATH",
        clock=clock,
    )
    assert signer is not None
    assert "kid-777" not in repr(signer)
    monkeypatch.setenv("CMA_TEST_KALSHI_PEM_PATH", str(tmp_path / "missing.pem"))
    with pytest.raises(KalshiAuthError):
        KalshiSigner.from_env(
            key_id_env="CMA_TEST_KALSHI_KEY",
            private_key_path_env="CMA_TEST_KALSHI_PEM_PATH",
            clock=clock,
        )


def test_rest_book_poller_applies_snapshots_and_fails_closed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = {"fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            return httpx.Response(503)
        assert request.url.params.get("depth") == "10"
        return httpx.Response(200, text=fixture_text("kalshi", "orderbook_rest_fp.json"))

    clock = ManualClock(T0_NS)
    sleep = FakeSleep(clock)
    books = BookManager()
    recorder = RawRecorder(tmp_path)
    fetcher = HttpFetcher(
        mock_client(httpx.MockTransport(handler)),
        clock=clock,
        sleep=sleep,
        retry=RetryPolicy(max_attempts=2),
        connection_id="rest:poll",
    )
    poller = KalshiRestBookPoller(
        client=KalshiRestClient(fetcher),
        tickers=[KALSHI_TICKER],
        clock=clock,
        books=books,
        depth=10,
        recorder=recorder,
        sleep=sleep,
    )

    async def go() -> None:
        events = await poller.poll_once()
        assert len(events) == 1
        assert events[0].sequence is None
        assert QualityFlag.SOURCE_TS_MISSING in events[0].quality_flags
        assert events[0].process_ts_ns is not None
        assert poller.is_ready(KALSHI_INST)
        book = books.get(KALSHI_INST)
        assert book is not None
        assert book.levels(BookSide.ASK)[0] == (D("0.47"), D("80.00"))
        assert poller.health.state is ConnectionState.CONNECTED
        state["fail"] = True
        with caplog.at_level(logging.WARNING):
            assert await poller.poll_once() == []
        assert not poller.is_ready(KALSHI_INST)
        assert QualityFlag.DISCONNECTED in book.flags
        assert poller.health.poll_errors == 1
        assert poller.health.state is ConnectionState.DISCONNECTED
        state["fail"] = False
        await poller.poll_once()
        assert poller.is_ready(KALSHI_INST)

    asyncio.run(go())
    recorder.close()
    streams = {r.stream for r in RawReader(tmp_path).iter_messages()}
    assert streams == {f"rest:orderbook:{KALSHI_TICKER}"}
    assert sleep.calls  # the 503 was retried with backoff before failing


def test_poller_run_loop_stops(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=fixture_text("kalshi", "orderbook_rest_fp.json"))

    clock = ManualClock(T0_NS)
    books = BookManager()
    poller: KalshiRestBookPoller

    async def sleep(seconds: float) -> None:
        clock.advance(int(seconds * 1e9))
        if poller.polls >= 3:
            await poller.stop()

    fetcher = HttpFetcher(mock_client(httpx.MockTransport(handler)), clock=clock)
    poller = KalshiRestBookPoller(
        client=KalshiRestClient(fetcher),
        tickers=[KALSHI_TICKER],
        clock=clock,
        books=books,
        interval_s=2.0,
        sleep=sleep,
    )
    asyncio.run(poller.run())
    assert poller.polls == 3
    assert poller.health.state is ConnectionState.STOPPED
    assert not poller.is_ready(KALSHI_INST)  # stopping invalidates (fail closed)
    assert Venue.KALSHI is poller.venue
