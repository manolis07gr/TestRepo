"""T045 duplicate message idempotency: a replayed duplicate trade/book event never
double-applies, at the raw, trade and book levels (and replays are id-stable)."""

from __future__ import annotations

import asyncio
import json
import random
from decimal import Decimal
from pathlib import Path

import pytest

from cma.adapters.crypto import CoinbaseAdapter
from cma.adapters.kalshi import KalshiAdapter
from cma.domain.enums import BookSide, Venue
from cma.domain.models import MarketEvent, TradeEvent
from cma.domain.time import ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.dedup import TradeDeduplicator
from cma.ingestion.health import FeedHealth
from cma.ingestion.rest import BackoffPolicy
from cma.ingestion.ws import FeedSession, MessagePipeline, ScriptedTransport, TransportClosed
from cma.storage.raw import RawReader, RawRecorder
from tests.unit.adapters.helpers import KALSHI_INST, T0_NS, fixture_text, raw_msg, stop_session

pytestmark = pytest.mark.integration
D = Decimal


def snapshot() -> str:
    return fixture_text("kalshi", "ws_orderbook_snapshot.json")  # seq 1: YES bid 0.45 x 35.50


def delta(seq: int, qty: str = "-10.50") -> str:
    msg = json.loads(fixture_text("kalshi", "ws_orderbook_delta_yes.json"))
    msg["seq"] = seq
    msg["msg"]["delta_fp"] = qty
    return json.dumps(msg)


TRADE = fixture_text("kalshi", "ws_trade.json")


def run_session(
    script: list[object], books_holder: dict[str, BookManager] | None = None, **kwargs: object
) -> tuple[FeedSession, list[MarketEvent], BookManager]:
    clock = ManualClock(T0_NS)
    books = BookManager()
    if books_holder is not None:
        books_holder["books"] = books
    delivered: list[MarketEvent] = []
    session: FeedSession
    transport = ScriptedTransport(
        [[*script, stop_session(lambda: session)]],
        clock=clock,
        step_ns=1_000_000,  # type: ignore[list-item]
    )
    session = FeedSession(
        name="kalshi-ws",
        url="wss://kalshi.invalid/ws",
        adapter=KalshiAdapter(),
        transport=transport,
        clock=clock,
        books=books,
        on_events=delivered.extend,
        rng=random.Random(0),
        **kwargs,  # type: ignore[arg-type]
    )
    asyncio.run(session.run())
    return session, delivered, books


def yes_bid(books: BookManager) -> tuple[Decimal, Decimal]:
    builder = books.get(KALSHI_INST)
    assert builder is not None
    return builder.levels(BookSide.BID)[0]


def test_T045_replayed_duplicate_trade_and_book_messages_do_not_double_apply(
    tmp_path: Path,
) -> None:
    recorder = RawRecorder(tmp_path)
    session, delivered, books = run_session(
        [snapshot(), delta(2), TRADE, delta(2), TRADE, delta(3, "1.00")],
        recorder=recorder,
        trade_dedup=TradeDeduplicator(),
    )
    # 35.50 - 10.50 (once) + 1.00
    assert yes_bid(books) == (D("0.4500"), D("26.00"))
    trades = [e for e in delivered if isinstance(e, TradeEvent)]
    assert len(trades) == 1
    assert session.health.duplicates == 2  # dropped at the raw layer, never parsed again
    recorder.close()
    recorded = list(RawReader(tmp_path).iter_messages())
    assert len(recorded) == 4  # snapshot, delta 2, trade, delta 3
    keys = [(r.idempotency_key or "").removeprefix(f"{r.connection_id}|") for r in recorded]
    # sid/seq keys are connection-scoped; the trade id is a global key
    assert keys == ["ob|2|1", "ob|2|2", "trade|d91bc706-ee49-470d-82d8-11418bda6fed", "ob|2|3"]


def test_T045_stale_delta_sequence_and_duplicate_trade_are_ignored_without_recorder() -> None:
    valid_before_stop: list[bool] = []
    holder: dict[str, BookManager] = {}

    def check() -> None:
        builder = holder["books"].get(KALSHI_INST)
        valid_before_stop.append(builder is not None and builder.is_valid)

    session, delivered, books = run_session(
        [snapshot(), delta(2), TRADE, delta(2), TRADE, check],
        trade_dedup=TradeDeduplicator(),
        books_holder=holder,
    )
    assert yes_bid(books) == (D("0.4500"), D("25.00"))
    assert valid_before_stop == [True]  # a stale delta never invalidates or mutates the book
    builder = books.get(KALSHI_INST)
    assert builder is not None
    assert builder.counters.duplicates_ignored == 1
    assert session.health.duplicate_trades == 1
    assert [type(e).__name__ for e in delivered] == [
        "BookSnapshotEvent",
        "BookDeltaEvent",
        "TradeEvent",
    ]


def test_T045_same_trade_via_rest_backfill_and_websocket_counts_once() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    shared = TradeDeduplicator()
    delivered: list[MarketEvent] = []

    def pipeline(name: str) -> MessagePipeline:
        return MessagePipeline(
            adapter=KalshiAdapter(),
            clock=clock,
            books=books,
            health=FeedHealth(name=name, venue=Venue.KALSHI),
            trade_dedup=shared,
            on_events=delivered.extend,
        )

    ws, rest = pipeline("ws"), pipeline("rest")
    ws.handle(raw_msg(Venue.KALSHI, TRADE))
    rest.handle(
        raw_msg(Venue.KALSHI, fixture_text("kalshi", "trades_page.json"), stream="rest:trades")
    )
    trade_ids = [e.trade_id for e in delivered if isinstance(e, TradeEvent)]
    assert trade_ids == [
        "d91bc706-ee49-470d-82d8-11418bda6fed",
        "0c7e8f3a-91b2-4d55-8a1e-5c0f2b9d7e61",
    ]
    assert rest.health.duplicate_trades == 1


def test_T045_coinbase_last_match_after_reconnect_is_not_double_counted(tmp_path: Path) -> None:
    clock = ManualClock(T0_NS)
    delivered: list[MarketEvent] = []
    recorder = RawRecorder(tmp_path)
    session: FeedSession
    transport = ScriptedTransport(
        [
            [fixture_text("coinbase", "ws_match.json"), TransportClosed("drop")],
            [
                fixture_text("coinbase", "ws_last_match.json"),
                fixture_text("coinbase", "ws_ticker.json"),
                stop_session(lambda: session),
            ],
        ],
        clock=clock,
        step_ns=1_000,
    )
    session = FeedSession(
        name="coinbase-ws",
        url="wss://ws-feed.exchange.coinbase.com",
        adapter=CoinbaseAdapter(),
        transport=transport,
        clock=clock,
        books=BookManager(),
        recorder=recorder,
        trade_dedup=TradeDeduplicator(),
        on_events=delivered.extend,
        backoff=BackoffPolicy(jitter_ratio=0.0),
        sleep=lambda s: asyncio.sleep(0),
    )
    asyncio.run(session.run())
    recorder.close()
    assert [type(e).__name__ for e in delivered] == ["TradeEvent", "BookSnapshotEvent"]


def test_T045_replaying_recorded_raw_is_idempotent(tmp_path: Path) -> None:
    recorder = RawRecorder(tmp_path)
    run_session([snapshot(), delta(2), TRADE], recorder=recorder)
    recorder.flush()

    def replay() -> tuple[list[str], tuple[Decimal, Decimal]]:
        books = BookManager()
        pipe = MessagePipeline(
            adapter=KalshiAdapter(),
            clock=ManualClock(T0_NS),
            books=books,
            health=FeedHealth(name="replay", venue=Venue.KALSHI),
            trade_dedup=TradeDeduplicator(),
        )
        ids: list[str] = []
        for raw in RawReader(tmp_path).iter_messages():
            ids += [e.event_id for e in pipe.process(raw).events]
            assert not recorder.append(raw)  # re-recording the replay adds nothing
        return ids, yes_bid(books)

    first, second = replay(), replay()
    recorder.close()
    assert first == second
    assert len(first[0]) == 3
    assert first[1] == (D("0.4500"), D("25.00"))
