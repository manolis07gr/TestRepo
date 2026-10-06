"""T047 malformed payload: bad external payloads (invalid JSON, schema violations,
probabilities > 1) are quarantined and counted; the collector keeps processing good
messages and never trusts a book that may have missed an update."""

from __future__ import annotations

import asyncio
import json
import random
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from cma.adapters.kalshi import KalshiAdapter, KalshiRestBookPoller, KalshiRestClient
from cma.adapters.polymarket import PolymarketAdapter
from cma.domain.enums import BookSide, QualityFlag, Venue
from cma.domain.models import MarketEvent
from cma.domain.time import ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.quarantine import Quarantine
from cma.ingestion.rest import HttpFetcher
from cma.ingestion.ws import FeedSession, ScriptedTransport
from cma.storage.raw import RawReader, RawRecorder
from tests.unit.adapters.helpers import (
    KALSHI_INST,
    KALSHI_TICKER,
    NO_INST,
    T0_NS,
    YES_INST,
    YES_TOKEN,
    fixture_text,
    mock_client,
    stop_session,
)

pytestmark = pytest.mark.integration
D = Decimal


def book(*, version: int, bid_price: str | None = None) -> str:
    """The YES book fixture as re-sent later (new timestamp and content hash)."""
    msg = json.loads(fixture_text("polymarket", "ws_book.json"))
    msg["timestamp"] = str(int(msg["timestamp"]) + version)
    msg["hash"] = f"0xbook{version}"
    if bid_price is not None:
        msg["bids"][0]["price"] = bid_price
    return json.dumps(msg)


def test_T047_malformed_payloads_are_quarantined_and_feed_continues(tmp_path: Path) -> None:
    clock = ManualClock(T0_NS)
    books = BookManager(sequence_free_venues=frozenset({Venue.POLYMARKET}))
    quarantine = Quarantine(tmp_path / "quarantine", clock=clock)
    recorder = RawRecorder(tmp_path / "raw")
    delivered: list[MarketEvent] = []
    seen: dict[str, object] = {}
    session: FeedSession

    def observe(name: str) -> object:
        def check() -> None:
            builder = books.get(YES_INST)
            seen[name] = builder is not None and builder.is_valid

        return check

    missing_asset = json.loads(fixture_text("polymarket", "ws_book.json"))
    del missing_asset["asset_id"]
    script = [
        fixture_text("polymarket", "ws_book.json"),  # good: YES book valid
        observe("valid_initially"),
        "{not json",  # invalid JSON
        json.dumps(missing_asset),  # schema violation (no asset id)
        observe("after_unattributable"),
        book(version=1, bid_price="1.5"),  # probability > 1 for the YES token
        observe("after_bad_yes_book"),
        book(version=2),  # fresh snapshot restores the book
        fixture_text("polymarket", "ws_price_change_new.json"),  # later good deltas apply
        fixture_text("polymarket", "ws_last_trade_price.json"),
        observe("valid_at_end"),
        stop_session(lambda: session),
    ]
    session = FeedSession(
        name="polymarket-ws",
        url="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        adapter=PolymarketAdapter(),
        transport=ScriptedTransport([script], clock=clock, step_ns=1_000_000),
        clock=clock,
        books=books,
        recorder=recorder,
        quarantine=quarantine,
        on_events=delivered.extend,
        rng=random.Random(0),
    )
    asyncio.run(session.run())

    assert quarantine.total == 3
    assert session.health.quarantined == 3
    assert quarantine.counters() == {"POLYMARKET/MalformedPayloadError": 3}
    errors = [r["error"] for r in quarantine.iter_records()]
    assert "invalid JSON" in errors[0]
    assert "asset_id" in errors[1]
    assert "outside [0, 1]" in errors[2]
    records = list(quarantine.iter_records())
    assert records[2]["payload"] == book(version=1, bid_price="1.5")
    assert records[2]["recv_ts_ns"] > T0_NS
    # fail closed: a lost book update invalidates the affected book until a fresh snapshot
    assert seen == {
        "valid_initially": True,
        "after_unattributable": False,  # unattributable bad frame: all session books
        "after_bad_yes_book": False,
        "valid_at_end": True,
    }
    # the NO-token delta is not applied (that book never had a snapshot), so not forwarded
    assert [type(e).__name__ for e in delivered] == [
        "BookSnapshotEvent",
        "BookSnapshotEvent",
        "BookDeltaEvent",
        "TradeEvent",
    ]
    yes = books.get(YES_INST)
    assert yes is not None
    assert yes.levels(BookSide.BID)[0] == (D("0.5"), D("200"))
    no = books.get(NO_INST)
    assert no is not None
    assert not no.is_valid
    recorder.close()
    assert len(list(RawReader(tmp_path / "raw").iter_messages())) == 7  # all frames kept raw


def test_T047_malformed_kalshi_delta_invalidates_book_and_requests_snapshot() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    quarantine = Quarantine(clock=clock)
    flags: list[frozenset[QualityFlag]] = []
    session: FeedSession
    bad = json.loads(fixture_text("kalshi", "ws_orderbook_delta_yes.json"))
    bad["msg"]["price_dollars"] = "1.5000"

    def capture() -> None:
        builder = books.get(KALSHI_INST)
        assert builder is not None
        flags.append(builder.flags)

    transport = ScriptedTransport(
        [
            [
                fixture_text("kalshi", "ws_orderbook_snapshot.json"),
                json.dumps(bad),
                capture,
                fixture_text("kalshi", "ws_orderbook_snapshot.json").replace('"seq":1', '"seq":5'),
                stop_session(lambda: session),
            ]
        ],
        clock=clock,
    )
    session = FeedSession(
        name="kalshi-ws",
        url="wss://kalshi.invalid/ws",
        adapter=KalshiAdapter(),
        transport=transport,
        clock=clock,
        books=books,
        quarantine=quarantine,
    )
    asyncio.run(session.run())
    assert QualityFlag.SEQUENCE_GAP in flags[0]  # a lost delta: book unusable at once
    resync = json.loads(transport.connections[0].sent[-1])
    assert resync["cmd"] == "update_subscription"
    assert resync["params"]["action"] == "get_snapshot"
    assert quarantine.counters() == {"KALSHI/MalformedPayloadError": 1}
    assert len(transport.connections) == 1  # recovered in band


def test_T047_malformed_rest_response_quarantined_poller_continues(tmp_path: Path) -> None:
    responses = iter(
        ['{"orderbook_fp": {"yes_dollars": [["1.2000", "1.00"]]}}', "<html>oops</html>"]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        text = next(responses, None) or fixture_text("kalshi", "orderbook_rest_fp.json")
        return httpx.Response(200, text=text)

    clock = ManualClock(T0_NS)
    books = BookManager()
    quarantine = Quarantine(tmp_path, clock=clock)
    poller = KalshiRestBookPoller(
        client=KalshiRestClient(
            HttpFetcher(mock_client(httpx.MockTransport(handler)), clock=clock)
        ),
        tickers=[KALSHI_TICKER],
        clock=clock,
        books=books,
        quarantine=quarantine,
    )

    async def go() -> list[bool]:
        ready = []
        for _ in range(3):
            await poller.poll_once()
            ready.append(poller.is_ready(KALSHI_INST))
        return ready

    assert asyncio.run(go()) == [False, False, True]
    assert quarantine.total == 2
    assert poller.health.quarantined == 2
    assert YES_TOKEN not in json.dumps(quarantine.counters())
