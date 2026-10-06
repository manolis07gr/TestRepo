"""T044 API reconnect: a dropped WebSocket reconnects with bounded backoff, resubscribes,
and keeps signals suppressed (is_ready False, books invalid) until a fresh snapshot."""

from __future__ import annotations

import asyncio
import json
import random
from decimal import Decimal

import pytest

from cma.adapters.kalshi import KalshiAdapter, build_subscriptions
from cma.adapters.polymarket import PolymarketAdapter, build_market_subscription
from cma.domain.enums import BookSide, QualityFlag, Venue
from cma.domain.time import ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.rest import BackoffPolicy
from cma.ingestion.ws import FeedSession, ScriptedTransport, TransportClosed
from cma.storage.raw import RawReader, RawRecorder
from tests.unit.adapters.helpers import (
    KALSHI_INST,
    KALSHI_TICKER,
    NO_TOKEN,
    T0_NS,
    YES_INST,
    YES_TOKEN,
    fixture_text,
    stop_session,
)

pytestmark = pytest.mark.integration
D = Decimal
URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
POLICY = BackoffPolicy(
    initial_delay_s=0.5, max_delay_s=8.0, multiplier=2.0, jitter_ratio=0.2, max_attempts=5
)
SEED = 44


def snapshot(seq: int, yes_qty: str) -> str:
    return json.dumps(
        {
            "type": "orderbook_snapshot",
            "sid": 1,
            "seq": seq,
            "msg": {
                "market_ticker": KALSHI_TICKER,
                "yes_dollars_fp": [["0.4500", yes_qty]],
                "no_dollars_fp": [["0.5300", "10.00"]],
            },
        }
    )


def delta(seq: int, qty: str) -> str:
    return json.dumps(
        {
            "type": "orderbook_delta",
            "sid": 1,
            "seq": seq,
            "msg": {
                "market_ticker": KALSHI_TICKER,
                "price_dollars": "0.4500",
                "delta_fp": qty,
                "side": "yes",
            },
        }
    )


def test_T044_reconnect_resubscribes_and_waits_for_snapshot() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    observed: dict[str, object] = {}
    sleeps: list[float] = []
    session: FeedSession

    def book_state() -> tuple[bool, bool, bool]:
        builder = books.get(KALSHI_INST)
        assert builder is not None
        return (
            session.is_ready(KALSHI_INST),
            builder.is_valid,
            QualityFlag.DISCONNECTED in builder.flags,
        )

    async def backoff_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        observed.setdefault("during_backoff", []).append((*book_state(), session.connected))  # type: ignore[union-attr]
        clock.advance(round(seconds * 1e9))

    def mark(name: str) -> object:
        return lambda: observed.__setitem__(name, book_state())

    subscriptions = build_subscriptions([KALSHI_TICKER])
    transport = ScriptedTransport(
        [
            # connection 1: healthy, then the network drops mid-stream
            [
                snapshot(1, "10.00"),
                delta(2, "5.00"),
                mark("live_before_drop"),
                TransportClosed("network drop"),
            ],
            # first reconnect attempt is refused
            ConnectionRefusedError("connection refused"),
            # connection 2: a delta before the fresh snapshot must not resurrect the old book
            [
                mark("connected_before_snapshot"),
                delta(2, "1.00"),
                mark("after_stray_delta"),
                snapshot(1, "40.00"),
                mark("after_snapshot"),
                delta(2, "2.00"),
                stop_session(lambda: session),
            ],
        ],
        clock=clock,
        step_ns=1_000_000,
    )
    session = FeedSession(
        name="kalshi-ws",
        url=URL,
        adapter=KalshiAdapter(),
        transport=transport,
        clock=clock,
        books=books,
        subscriptions=subscriptions,
        backoff=POLICY,
        rng=random.Random(SEED),
        sleep=backoff_sleep,
    )
    asyncio.run(session.run())

    # reconnected (3 attempts, 2 connections) and resubscribed identically
    assert len(transport.attempts) == 3
    assert len(transport.connections) == 2
    assert transport.connections[0].sent == subscriptions == transport.connections[1].sent
    # bounded exponential backoff with seeded jitter: attempt 1 after the drop, 2 after the refusal
    rng = random.Random(SEED)
    assert sleeps == [POLICY.delay(1, rng), POLICY.delay(2, rng)]
    assert 0.5 <= sleeps[0] <= 0.6
    assert 1.0 <= sleeps[1] <= 1.2
    # fail closed from the moment of the drop until the fresh snapshot
    assert observed["live_before_drop"] == (True, True, False)
    assert observed["during_backoff"] == [(False, False, True, False), (False, False, True, False)]
    assert observed["connected_before_snapshot"][:2] == (False, False)  # type: ignore[index]
    assert observed["after_stray_delta"][:2] == (False, False)  # type: ignore[index]
    assert observed["after_snapshot"] == (True, True, False)
    builder = books.get(KALSHI_INST)
    assert builder is not None
    assert builder.levels(BookSide.BID) == [(D("0.4500"), D("42.00"))]  # new snapshot + its delta
    health = session.health
    assert (health.connects, health.reconnects, health.connect_failures) == (2, 1, 1)
    assert not session.is_ready(KALSHI_INST)  # stopped


def test_T044_identical_snapshot_after_reconnect_is_processed_not_deduplicated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Polymarket may re-send an unchanged book after a reconnect; it must revalidate the book."""
    clock = ManualClock(T0_NS)
    books = BookManager(sequence_free_venues=frozenset({Venue.POLYMARKET}))
    recorder = RawRecorder(tmp_path)
    book = fixture_text("polymarket", "ws_book.json")
    ready: list[bool] = []
    session: FeedSession
    transport = ScriptedTransport(
        [
            [book, TransportClosed("drop")],
            [
                lambda: ready.append(session.is_ready(YES_INST)),
                book,
                lambda: ready.append(session.is_ready(YES_INST)),
                stop_session(lambda: session),
            ],
        ],
        clock=clock,
        step_ns=1_000,
    )
    session = FeedSession(
        name="polymarket-ws",
        url="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        adapter=PolymarketAdapter(),
        transport=transport,
        clock=clock,
        books=books,
        recorder=recorder,
        subscriptions=[build_market_subscription([YES_TOKEN, NO_TOKEN])],
        rng=random.Random(1),
        sleep=lambda s: asyncio.sleep(0),
    )
    asyncio.run(session.run())
    assert ready == [False, True]
    assert session.health.duplicates == 0
    recorder.close()
    assert len(list(RawReader(tmp_path).iter_messages())) == 2
