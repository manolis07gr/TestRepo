"""FeedSession behaviour against the scripted transport (no network)."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import pytest

from cma.adapters.crypto import CoinbaseAdapter
from cma.adapters.kalshi import KalshiAdapter, build_subscriptions
from cma.adapters.polymarket import POLYMARKET_KEEPALIVE, PolymarketAdapter
from cma.domain.enums import QualityFlag, Venue
from cma.domain.models import MarketEvent
from cma.domain.time import ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.health import ConnectionState, IncidentLog
from cma.ingestion.quarantine import Quarantine
from cma.ingestion.rest import BackoffPolicy
from cma.ingestion.ws import FeedExhaustedError, FeedSession, ScriptedTransport, TransportClosed
from cma.storage.db import open_database
from cma.storage.raw import RawReader, RawRecorder
from tests.unit.adapters.helpers import (
    KALSHI_INST,
    KALSHI_TICKER,
    T0_NS,
    YES_INST,
    FakeSleep,
    fixture_text,
    stop_session,
)

pytestmark = pytest.mark.unit
URL = "wss://example.invalid/ws"


def kalshi_frame(kind: str, seq: int, **msg: object) -> str:
    body: dict[str, object] = {"market_ticker": KALSHI_TICKER, **msg}
    if kind == "snapshot":
        body.setdefault("yes_dollars_fp", [["0.4500", "10.00"]])
        body.setdefault("no_dollars_fp", [["0.5300", "10.00"]])
        return json.dumps({"type": "orderbook_snapshot", "sid": 2, "seq": seq, "msg": body})
    body.setdefault("price_dollars", "0.4500")
    body.setdefault("delta_fp", "1.00")
    body.setdefault("side", "yes")
    return json.dumps({"type": "orderbook_delta", "sid": 2, "seq": seq, "msg": body})


def make_session(
    transport: ScriptedTransport,
    clock: ManualClock,
    books: BookManager,
    *,
    adapter: object | None = None,
    **kwargs: object,
) -> FeedSession:
    return FeedSession(
        name="test",
        url=URL,
        adapter=adapter or KalshiAdapter(),  # type: ignore[arg-type]
        transport=transport,
        clock=clock,
        books=books,
        rng=random.Random(1),
        sleep=FakeSleep(clock),
        **kwargs,  # type: ignore[arg-type]
    )


def test_coinbase_session_subscribes_streams_and_stops_fail_closed() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    observed: dict[str, object] = {}
    events: list[MarketEvent] = []
    session: FeedSession

    def check() -> None:
        observed["ready"] = session.is_ready("COINBASE:BTC-USD")
        session.request_stop()

    transport = ScriptedTransport(
        [
            [
                fixture_text("coinbase", "ws_ticker.json"),
                fixture_text("coinbase", "ws_match.json"),
                check,
            ]
        ],
        clock=clock,
        step_ns=1_000_000,
    )
    session = make_session(
        transport,
        clock,
        books,
        adapter=CoinbaseAdapter(),
        subscriptions=['{"type":"subscribe"}'],
        on_events=events.extend,
    )
    asyncio.run(session.run())
    assert transport.connections[0].sent == ['{"type":"subscribe"}']
    assert observed["ready"] is True
    assert not session.is_ready("COINBASE:BTC-USD")  # stopped -> not ready
    builder = books.get("COINBASE:BTC-USD")
    assert builder is not None
    assert QualityFlag.DISCONNECTED in builder.flags
    assert [type(e).__name__ for e in events] == ["BookSnapshotEvent", "TradeEvent"]
    assert all(e.process_ts_ns is not None and e.process_ts_ns >= e.recv_ts_ns for e in events)
    assert session.health.state is ConnectionState.STOPPED
    assert session.health.messages == 2
    assert session.health.latency.total == 2


def test_sequence_gap_requests_in_band_snapshot_without_reconnect() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    seen: dict[str, object] = {}
    session: FeedSession

    def after_gap() -> None:
        builder = books.get(KALSHI_INST)
        assert builder is not None
        seen["after_gap"] = (
            session.is_ready(KALSHI_INST),
            QualityFlag.SEQUENCE_GAP in builder.flags,
        )

    def finish() -> None:
        seen["final_ready"] = session.is_ready(KALSHI_INST)
        session.request_stop()

    script = [
        kalshi_frame("snapshot", 1),
        kalshi_frame("delta", 2),
        kalshi_frame("delta", 4),  # seq 3 lost
        after_gap,
        kalshi_frame("delta", 5),  # ignored: awaiting the requested snapshot
        kalshi_frame("snapshot", 6, yes_dollars_fp=[["0.4400", "7.00"]]),
        kalshi_frame("delta", 7),
        finish,
    ]
    transport = ScriptedTransport([script], clock=clock, step_ns=1_000)
    session = make_session(
        transport, clock, books, subscriptions=build_subscriptions([KALSHI_TICKER], trades=False)
    )
    asyncio.run(session.run())
    sent = [json.loads(f) for f in transport.connections[0].sent]
    assert sent[-1] == {
        "id": 2,
        "cmd": "update_subscription",
        "params": {"sids": [2], "market_tickers": [KALSHI_TICKER], "action": "get_snapshot"},
    }
    assert len(transport.attempts) == 1  # recovered in band
    assert seen == {"after_gap": (False, True), "final_ready": True}
    assert session.health.sequence_gaps == 1
    assert session.health.resyncs == 1


def test_gap_with_reconnect_policy_reconnects_and_resubscribes() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    session: FeedSession
    subs = build_subscriptions([KALSHI_TICKER], trades=False)
    transport = ScriptedTransport(
        [
            [kalshi_frame("snapshot", 1), kalshi_frame("delta", 3)],
            [kalshi_frame("snapshot", 1), stop_session(lambda: session)],
        ],
        clock=clock,
    )
    session = make_session(transport, clock, books, subscriptions=subs, on_sequence_gap="reconnect")
    asyncio.run(session.run())
    assert len(transport.connections) == 2
    assert [c.sent for c in transport.connections] == [subs, subs]
    assert session.health.reconnects == 1


def test_resync_timeout_falls_back_to_reconnect() -> None:
    clock = ManualClock(T0_NS)
    books = BookManager()
    session: FeedSession
    transport = ScriptedTransport(
        [
            [
                kalshi_frame("snapshot", 1),
                kalshi_frame("delta", 3),  # gap -> get_snapshot requested
                lambda: clock.advance(11 * 1_000_000_000),
                kalshi_frame("delta", 4),  # still no snapshot after 10 s -> reconnect
            ],
            [stop_session(lambda: session)],
        ],
        clock=clock,
    )
    session = make_session(transport, clock, books, resync_timeout_s=10.0)
    asyncio.run(session.run())
    assert len(transport.connections) == 2


def test_reconnect_attempts_are_bounded() -> None:
    clock = ManualClock(T0_NS)
    sleep = FakeSleep(clock)
    transport = ScriptedTransport([ConnectionRefusedError("down")] * 5)
    session = FeedSession(
        name="t",
        url=URL,
        adapter=KalshiAdapter(),
        transport=transport,
        clock=clock,
        books=BookManager(),
        backoff=BackoffPolicy(
            initial_delay_s=1.0, max_delay_s=3.0, jitter_ratio=0.0, max_attempts=3
        ),
        sleep=sleep,
    )
    with pytest.raises(FeedExhaustedError):
        asyncio.run(session.run())
    assert sleep.calls == [1.0, 2.0, 3.0]  # bounded by max_delay_s
    assert len(transport.attempts) == 4
    assert session.health.state is ConnectionState.FAILED
    assert session.health.connect_failures == 4


def test_headers_are_rebuilt_for_every_connection() -> None:
    clock = ManualClock(T0_NS)
    calls: list[int] = []
    session: FeedSession

    def headers() -> dict[str, str]:
        calls.append(clock.now_ns())
        return {"X-Signed-At": str(clock.now_ns())}

    transport = ScriptedTransport(
        [[TransportClosed("drop")], [stop_session(lambda: session)]], clock=clock
    )
    session = make_session(transport, clock, BookManager(), headers=headers)
    asyncio.run(session.run())
    assert len(calls) == 2
    assert calls[0] < calls[1]
    assert [a[1]["X-Signed-At"] for a in transport.attempts] == [str(c) for c in calls]


def test_keepalive_frames_are_sent_while_connected() -> None:
    clock = ManualClock(T0_NS)
    ticks: list[float] = []
    session: FeedSession

    async def keepalive_sleep(seconds: float) -> None:
        ticks.append(seconds)
        if len(ticks) > 2:
            session.request_stop()
            await asyncio.Event().wait()  # park the keepalive task until it is cancelled
        await asyncio.sleep(0)

    transport = ScriptedTransport([[fixture_text("polymarket", "ws_book.json")]], clock=clock)
    session = make_session(
        transport,
        clock,
        BookManager(sequence_free_venues=frozenset({Venue.POLYMARKET})),
        adapter=PolymarketAdapter(),
        subscriptions=['{"assets_ids":[],"type":"market"}'],
        keepalive=POLYMARKET_KEEPALIVE,
        keepalive_sleep=keepalive_sleep,
    )
    asyncio.run(session.run())
    assert transport.connections[0].sent[1:] == ["PING", "PING"]
    assert ticks[:2] == [10.0, 10.0]
    assert session.is_ready(YES_INST) is False


def test_callback_and_adapter_failures_do_not_stop_the_feed(tmp_path: Path) -> None:
    clock = ManualClock(T0_NS)
    quarantine = Quarantine(tmp_path / "q")
    session: FeedSession

    class ExplodingAdapter(KalshiAdapter):
        def parse(self, raw):  # type: ignore[no-untyped-def]
            if "boom" in raw.payload:
                raise RuntimeError("adapter bug")
            return super().parse(raw)

    def bad_callback(events: list[MarketEvent]) -> None:
        raise ValueError("downstream bug")

    transport = ScriptedTransport(
        [['{"type": "boom"}', kalshi_frame("snapshot", 1), stop_session(lambda: session)]],
        clock=clock,
    )
    session = make_session(
        transport,
        clock,
        BookManager(),
        adapter=ExplodingAdapter(),
        quarantine=quarantine,
        on_events=bad_callback,
    )
    asyncio.run(session.run())
    assert quarantine.counters() == {"KALSHI/RuntimeError": 1}
    assert session.health.callback_errors == 1
    assert session.health.quarantined == 1
    assert len(session.pipeline.last_snapshot_ns) == 1  # later frames still processed


def test_raw_messages_carry_connection_metadata_and_scoped_keys(tmp_path: Path) -> None:
    clock = ManualClock(T0_NS)
    recorder = RawRecorder(tmp_path)
    session: FeedSession
    same_snapshot = kalshi_frame("snapshot", 1)
    transport = ScriptedTransport(
        [[same_snapshot, TransportClosed("drop")], [same_snapshot, stop_session(lambda: session)]],
        clock=clock,
        step_ns=5,
    )
    session = make_session(transport, clock, BookManager(), recorder=recorder)
    asyncio.run(session.run())
    recorder.close()
    first, second = RawReader(tmp_path).iter_messages()
    assert first.connection_id != second.connection_id
    assert (first.connection_seq, second.connection_seq) == (1, 1)
    assert first.idempotency_key == f"{first.connection_id}|ob|2|1"
    # identical snapshot text after a reconnect is a NEW message, not a duplicate
    assert second.idempotency_key == f"{second.connection_id}|ob|2|1"
    assert session.health.duplicates == 0


def test_incidents_record_gap_and_outage_windows() -> None:
    db = open_database("sqlite:///:memory:")
    incidents = IncidentLog(db)
    clock = ManualClock(T0_NS)
    session: FeedSession
    transport = ScriptedTransport(
        [
            [
                kalshi_frame("snapshot", 1),
                kalshi_frame("delta", 3),  # gap -> BOOK_INTEGRITY opens
                kalshi_frame("snapshot", 4),  # fresh snapshot -> closes it
                TransportClosed("drop"),  # FEED_DISCONNECTED opens
            ],
            [kalshi_frame("snapshot", 1), stop_session(lambda: session)],  # reconnect closes it
        ],
        clock=clock,
        step_ns=1_000_000,
    )
    session = make_session(transport, clock, BookManager(), incidents=incidents)
    asyncio.run(session.run())
    rows = db.query(
        "SELECT kind, instrument_id, start_ns, end_ns, detail FROM data_quality_incidents "
        "ORDER BY start_ns"
    )
    assert [(r["kind"], r["instrument_id"]) for r in rows] == [
        ("BOOK_INTEGRITY", KALSHI_INST),
        ("FEED_DISCONNECTED", None),
    ]
    gap, outage = rows
    assert gap["end_ns"] - gap["start_ns"] == 1_000_000  # closed by the next snapshot
    assert outage["end_ns"] is not None
    assert outage["end_ns"] >= outage["start_ns"]
    assert "drop" in outage["detail"]
    assert incidents.open_incidents() == []  # the final stop is not an incident


def test_incident_log_extends_open_windows_and_is_idempotent() -> None:
    db = open_database("sqlite:///:memory:")
    log = IncidentLog(db)
    first = log.open(Venue.KALSHI, "POLL_FAILED", 100, instrument_id="KALSHI:X", detail="503")
    assert log.open(Venue.KALSHI, "POLL_FAILED", 150, instrument_id="KALSHI:X") is first
    assert [i.kind for i in log.open_incidents()] == ["POLL_FAILED"]
    (done,) = log.close_instrument(Venue.KALSHI, "KALSHI:X", 200)
    assert (done.start_ns, done.end_ns) == (100, 200)
    assert done.to_dict()["venue"] == "KALSHI"
    assert log.close(Venue.KALSHI, "POLL_FAILED", 300, instrument_id="KALSHI:X") is None
    rows = db.query("SELECT start_ns, end_ns FROM data_quality_incidents")
    assert rows == [{"start_ns": 100, "end_ns": 200}]


def test_session_subscribes_markets_listed_after_connect() -> None:
    """A long-running feed must pick up markets that list later (new hourly / 15-minute
    BTC contracts) on the live connection, with command ids that never collide."""
    clock = ManualClock(T0_NS)
    books = BookManager()
    pending: list[list[str]] = [["KXNEW-1", "KXNEW-2"]]
    session: FeedSession

    async def refresh() -> list[str]:
        return pending.pop(0) if pending else []

    async def wait_for_refresh() -> None:
        for _ in range(100):
            if session.subscription_refreshes:
                return
            await asyncio.sleep(0)

    async def no_wait(_seconds: float) -> None:
        await asyncio.sleep(0)

    initial = build_subscriptions([KALSHI_TICKER])
    transport = ScriptedTransport(
        [
            [
                kalshi_frame("snapshot", 1),
                wait_for_refresh,
                kalshi_frame("delta", 2),
                stop_session(lambda: session),
            ]
        ]
    )
    session = make_session(
        transport,
        clock,
        books,
        subscriptions=initial,
        subscription_refresh=refresh,
        subscription_frames=lambda tickers, first: build_subscriptions(tickers, start_id=first),
        subscription_refresh_s=0.0,
        refresh_sleep=no_wait,
    )
    asyncio.run(session.run())
    sent = [json.loads(f) for f in transport.connections[0].sent]
    assert [f["id"] for f in sent] == list(range(1, len(sent) + 1))
    added = [f for f in sent[len(initial) :] if f["params"]["channels"] == ["orderbook_delta"]]
    assert [f["params"]["market_tickers"] for f in added] == [["KXNEW-1"], ["KXNEW-2"]]
    assert session.subscription_refreshes == 1


OTHER_TICKER = "KXBTCD-26OCT0617-T111999.99"
OTHER_INST = f"KALSHI:{OTHER_TICKER}"


def shared_frame(kind: str, seq: int, ticker: str) -> str:
    """Kalshi folds every orderbook subscription into one: one sid, one seq counter."""
    body: dict[str, object] = {"market_ticker": ticker}
    if kind == "snapshot":
        body |= {"yes_dollars_fp": [["0.4500", "10.00"]], "no_dollars_fp": [["0.5300", "10.00"]]}
        return json.dumps({"type": "orderbook_snapshot", "sid": 1, "seq": seq, "msg": body})
    body |= {"price_dollars": "0.4500", "delta_fp": "1.00", "side": "yes"}
    return json.dumps({"type": "orderbook_delta", "sid": 1, "seq": seq, "msg": body})


def test_kalshi_sequence_is_checked_per_subscription_not_per_market() -> None:
    """Interleaved markets share one seq counter: no false gaps and no snapshot requests.
    A real gap in the subscription may hit any of its books, so all are re-snapshotted."""
    clock = ManualClock(T0_NS)
    books = BookManager()
    seen: dict[str, object] = {}
    session: FeedSession

    def ready() -> tuple[bool, bool]:
        return session.is_ready(KALSHI_INST), session.is_ready(OTHER_INST)

    def mid() -> None:
        seen["mid"] = (*ready(), session.health.sequence_gaps, session.health.resyncs)

    def after_gap() -> None:
        seen["after_gap"] = ready()

    def finish() -> None:
        seen["final"] = ready()
        session.request_stop()

    script = [
        shared_frame("snapshot", 1, KALSHI_TICKER),
        shared_frame("snapshot", 2, OTHER_TICKER),
        shared_frame("delta", 3, KALSHI_TICKER),
        shared_frame("delta", 4, OTHER_TICKER),
        shared_frame("delta", 5, KALSHI_TICKER),
        shared_frame("delta", 5, KALSHI_TICKER),  # duplicate: dropped
        mid,
        shared_frame("delta", 7, OTHER_TICKER),  # seq 6 lost
        after_gap,
        shared_frame("snapshot", 8, KALSHI_TICKER),
        shared_frame("snapshot", 9, OTHER_TICKER),
        finish,
    ]
    transport = ScriptedTransport([script], clock=clock, step_ns=1_000)
    session = make_session(
        transport,
        clock,
        books,
        subscriptions=build_subscriptions([KALSHI_TICKER, OTHER_TICKER], trades=False),
    )
    asyncio.run(session.run())
    assert seen == {"mid": (True, True, 0, 0), "after_gap": (False, False), "final": (True, True)}
    assert (session.health.sequence_gaps, session.health.resyncs) == (1, 2)
    resyncs = [json.loads(f) for f in transport.connections[0].sent][-2:]
    assert sorted(f["params"]["market_tickers"][0] for f in resyncs) == [
        KALSHI_TICKER,
        OTHER_TICKER,
    ]
    assert {f["params"]["sids"][0] for f in resyncs} == {1}
    builder = books.get(KALSHI_INST)
    assert builder is not None and builder.counters.duplicates_ignored == 1
