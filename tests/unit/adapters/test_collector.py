"""Collector orchestration and the paper-mode smoke test (CI gate: collectors start
against mocks and produce a valid health status)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from cma.adapters.kalshi import KalshiRestBookPoller
from cma.config import AppConfig, VenueConfig, load_config
from cma.domain.time import ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.collector import Collector, build_collector
from cma.ingestion.health import ConnectionState, HealthStatus
from cma.ingestion.ws import FeedSession, ScriptedTransport
from cma.storage.db import open_database
from cma.storage.raw import RawReader
from tests.unit.adapters.helpers import (
    KALSHI_INST,
    KALSHI_TICKER,
    NO_TOKEN,
    T0_NS,
    YES_INST,
    YES_TOKEN,
    fixture_text,
    mock_client,
)

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]


def kalshi_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path.removeprefix("/trade-api/v2")
    if path.startswith("/series/"):
        series = json.loads(fixture_text("kalshi", "series.json"))
        series["series"]["ticker"] = path.rsplit("/", 1)[1]
        return httpx.Response(200, json=series)
    if path == "/markets":
        if request.url.params.get("series_ticker") != "KXBTCD":
            return httpx.Response(200, json={"markets": [], "cursor": ""})
        page = 2 if request.url.params.get("cursor") else 1
        return httpx.Response(200, text=fixture_text("kalshi", f"markets_page_{page}.json"))
    if path.endswith("/orderbook"):
        return httpx.Response(200, text=fixture_text("kalshi", "orderbook_rest_fp.json"))
    return httpx.Response(404)


async def wait_for(predicate: object, timeout_s: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():  # type: ignore[operator]
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def test_paper_mode_smoke_collectors_against_mocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"):
        monkeypatch.delenv(var, raising=False)
    config = load_config(
        REPO / "config" / "base.yaml",
        REPO / "config" / "paper.yaml",
        overrides={"venues": {"polymarket": {"instruments": [YES_TOKEN, NO_TOKEN]}}},
    )
    coinbase_url = config.venues["coinbase"].ws_url
    polymarket_url = config.venues["polymarket"].ws_url
    assert coinbase_url
    assert polymarket_url
    transport = ScriptedTransport(
        {
            coinbase_url: [[fixture_text("coinbase", "ws_ticker.json")]],
            polymarket_url: [[fixture_text("polymarket", "ws_book_array.json")]],
        }
    )
    db = open_database("sqlite:///:memory:")

    async def go() -> dict[str, object]:
        collector = build_collector(
            config,
            transport=transport,
            http_client=mock_client(httpx.MockTransport(kalshi_handler)),
            db=db,
            raw_root=tmp_path / "raw",
            quarantine_root=tmp_path / "quarantine",
            kalshi_poll_interval_s=0.01,
        )
        names = [f.name for f in collector.feeds]
        assert names == ["coinbase-ws", "kalshi-rest-orderbook", "polymarket-ws"]
        assert isinstance(collector.feeds[1], KalshiRestBookPoller)  # no credentials
        await collector.start()
        await wait_for(lambda: collector.health_status() is HealthStatus.OK)
        report = collector.health_report()
        assert collector.is_ready(KALSHI_INST)
        assert collector.is_ready(YES_INST)
        assert collector.is_ready("COINBASE:BTC-USD")
        await collector.aclose()
        assert collector.health_status() is HealthStatus.DOWN
        return report

    report = asyncio.run(go())
    assert report["status"] == "OK"
    assert {f["state"] for f in report["feeds"]} == {"CONNECTED"}  # type: ignore[index,union-attr]
    sent = json.loads(transport.connections[1].sent[0])
    assert sent["assets_ids"] == [YES_TOKEN, NO_TOKEN]
    assert sent["custom_feature_enabled"] is True
    discovered = db.query("SELECT native_id, tick_size FROM contracts ORDER BY native_id")
    assert [r["native_id"] for r in discovered] == [KALSHI_TICKER, "KXBTCD-26OCT0617-T111999.99"]
    streams = {m.stream for m in RawReader(tmp_path / "raw").iter_messages()}
    assert {"ws", "rest:markets", "rest:series", f"rest:orderbook:{KALSHI_TICKER}"} <= streams


def test_failed_feed_does_not_stop_other_feeds() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    from cma.adapters.crypto import CoinbaseAdapter
    from cma.ingestion.rest import BackoffPolicy
    from tests.unit.adapters.helpers import FakeSleep

    good_transport = ScriptedTransport([[fixture_text("coinbase", "ws_ticker.json")]])
    good = FeedSession(
        name="good",
        url="wss://good.invalid",
        adapter=CoinbaseAdapter(),
        transport=good_transport,
        clock=clock,
        books=books,
    )
    bad = FeedSession(
        name="bad",
        url="wss://bad.invalid",
        adapter=CoinbaseAdapter(),
        transport=ScriptedTransport([]),
        clock=clock,
        books=books,
        backoff=BackoffPolicy(max_attempts=1, jitter_ratio=0.0),
        sleep=FakeSleep(clock),
    )
    collector = Collector(clock=clock, books=books, feeds=[good, bad], stale_after_s=None)

    async def go() -> None:
        await collector.start()
        await wait_for(lambda: bad.health.state is ConnectionState.FAILED)
        await wait_for(lambda: good.is_ready("COINBASE:BTC-USD"))
        assert collector.health_status() is HealthStatus.DEGRADED
        await collector.stop()

    asyncio.run(go())
    assert good.health.state is ConnectionState.STOPPED


def test_unknown_or_unconfigured_venues_are_skipped(tmp_path: Path) -> None:
    config = AppConfig(
        venues={
            "kalshi": VenueConfig(enabled=True),  # nothing to collect
            "reference": VenueConfig(enabled=True),  # file-backed: no live feed
            "nowhere": VenueConfig(enabled=True),
            "binance": VenueConfig(enabled=True, instruments=["BTCUSDT"]),
            "coinbase": VenueConfig(enabled=False, instruments=["BTC-USD"]),
        }
    )

    async def go() -> list[str]:
        collector = build_collector(
            config,
            transport=ScriptedTransport([]),
            http_client=mock_client(httpx.MockTransport(lambda r: httpx.Response(404))),
            raw_root=tmp_path / "raw",
            quarantine_root=tmp_path / "q",
        )
        names = [f.name for f in collector.feeds]
        await collector.aclose()
        return names

    assert asyncio.run(go()) == ["binance-ws"]
