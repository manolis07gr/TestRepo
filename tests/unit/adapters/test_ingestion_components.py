"""Ingestion building blocks: rate limiter, backoff policies, dedup, quarantine, health."""

from __future__ import annotations

import asyncio
import json
import random
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from cma.adapters.base import MalformedPayloadError
from cma.domain.enums import BookSide, QualityFlag, Side, Venue
from cma.domain.models import BookLevel, BookSnapshotEvent, TradeEvent
from cma.domain.time import NS_PER_S, ManualClock
from cma.ingestion.book import BookManager
from cma.ingestion.dedup import BoundedLRUSet, EventDeduplicator, TradeDeduplicator
from cma.ingestion.health import (
    BookHealth,
    ConnectionState,
    FeedHealth,
    HealthStatus,
    LatencyStats,
    aggregate_status,
)
from cma.ingestion.quarantine import Quarantine
from cma.ingestion.rest import (
    BackoffPolicy,
    HttpStatusError,
    RateLimiter,
    RetryPolicy,
    fetch_with_backoff,
    parse_retry_after,
)
from cma.security import SECRETS, Secret
from tests.unit.adapters.helpers import T0_NS, FakeSleep, raw_msg

pytestmark = pytest.mark.unit
D = Decimal


# ------------------------------------------------------------------ backoff / rate limit


def test_backoff_is_exponential_bounded_and_seeded() -> None:
    policy = BackoffPolicy(initial_delay_s=0.5, max_delay_s=4.0, multiplier=2.0, jitter_ratio=0.2)
    assert [policy.base_delay(n) for n in range(1, 6)] == [0.5, 1.0, 2.0, 4.0, 4.0]
    a = [policy.delay(n, random.Random(7)) for n in range(1, 8)]
    b = [policy.delay(n, random.Random(7)) for n in range(1, 8)]
    assert a == b  # deterministic under a seed
    for n, delay in enumerate(a, start=1):
        assert policy.base_delay(n) <= delay <= min(4.0, policy.base_delay(n) * 1.2)
    with pytest.raises(ValueError, match="jitter"):
        BackoffPolicy(jitter_ratio=2.0)


def test_retry_after_parsing() -> None:
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after(" 0.5 ") == 0.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    now = 1_445_412_480 * NS_PER_S  # Wed, 21 Oct 2015 07:28:00 GMT
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:30 GMT", now_ns=now) == 30.0
    assert parse_retry_after("Wed, 21 Oct 2015 07:27:00 GMT", now_ns=now) == 0.0


def test_token_bucket_rate_limiter_waits_deterministically() -> None:
    clock = ManualClock(T0_NS)
    sleep = FakeSleep(clock)

    async def go() -> list[float]:
        limiter = RateLimiter(2.0, burst=2.0, clock=clock, sleep=sleep)
        return [await limiter.acquire() for _ in range(5)]

    waits = asyncio.run(go())
    assert waits == [0.0, 0.0, 0.5, 0.5, 0.5]
    assert sleep.calls == [0.5, 0.5, 0.5]
    assert clock.now_ns() == T0_NS + int(1.5 * NS_PER_S)


def test_rate_limiter_refills_over_time() -> None:
    clock = ManualClock(T0_NS)
    sleep = FakeSleep(clock)

    async def go() -> float:
        limiter = RateLimiter(10.0, burst=1.0, clock=clock, sleep=sleep)
        await limiter.acquire()
        clock.advance(NS_PER_S)  # plenty of time: bucket refilled (capped at burst)
        return await limiter.acquire()

    assert asyncio.run(go()) == 0.0
    assert sleep.calls == []


def test_non_retryable_status_fails_immediately() -> None:
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="not found")

    sleep = FakeSleep()
    with pytest.raises(HttpStatusError) as info:
        asyncio.run(fetch_with_backoff(send, sleep=sleep))
    assert info.value.status == 404
    assert calls == 1
    assert sleep.calls == []


def test_transport_errors_are_retried() -> None:
    attempts = 0

    async def send() -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, text="ok")

    sleep = FakeSleep()
    policy = RetryPolicy(
        max_attempts=4, backoff=BackoffPolicy(initial_delay_s=1.0, jitter_ratio=0.0)
    )
    response = asyncio.run(fetch_with_backoff(send, policy=policy, sleep=sleep))
    assert response.text == "ok"
    assert sleep.calls == [1.0, 2.0]


# ------------------------------------------------------------------ dedup


def test_bounded_lru_set_evicts_least_recent() -> None:
    lru: BoundedLRUSet[str] = BoundedLRUSet(2)
    assert lru.add("a")
    assert lru.add("b")
    assert not lru.add("a")  # refreshes "a"
    assert lru.add("c")  # evicts "b"
    assert "a" in lru
    assert "b" not in lru
    assert len(lru) == 2


def _trade(trade_id: str, payload_hash: str = "h") -> TradeEvent:
    return TradeEvent(
        venue=Venue.KALSHI,
        instrument_id="KALSHI:X",
        source_ts_ns=1,
        recv_ts_ns=2,
        payload_hash=payload_hash,
        price=D("0.5"),
        size=D(1),
        aggressor_side=Side.BUY,
        trade_id=trade_id,
    )


def test_trade_and_event_deduplicators() -> None:
    trades = TradeDeduplicator(capacity=10)
    assert not trades.is_duplicate(_trade("1", "a"))
    assert trades.is_duplicate(_trade("1", "b"))  # same trade id, different envelope
    assert not trades.is_duplicate(_trade("2"))
    assert trades.duplicates == 1
    events = EventDeduplicator()
    assert events.filter([_trade("1", "a"), _trade("1", "a"), _trade("1", "b")]) == [
        _trade("1", "a"),
        _trade("1", "b"),
    ]


# ------------------------------------------------------------------ quarantine


def test_quarantine_persists_counts_and_redacts(tmp_path: Path) -> None:
    secret = Secret("CMA_TEST_TOKEN", "sekret-token-123")
    try:
        quarantine = Quarantine(tmp_path, clock=ManualClock(T0_NS + 5))
        raw = raw_msg(Venue.KALSHI, '{"leak": "sekret-token-123"}', recv_ts_ns=T0_NS)
        quarantine.put(raw, MalformedPayloadError("bad price 1.5"))
        quarantine.put(raw_msg(Venue.KALSHI, "{"), MalformedPayloadError("invalid JSON"))
        quarantine.put(raw_msg(Venue.POLYMARKET, "x"), ValueError("boom"))
        assert quarantine.total == 3
        assert quarantine.counters() == {
            "KALSHI/MalformedPayloadError": 2,
            "POLYMARKET/ValueError": 1,
        }
        records = list(quarantine.iter_records())
        assert len(records) == 3
        first = records[0]
        assert first["error"] == "bad price 1.5"
        assert first["quarantined_at_ns"] == T0_NS + 5
        assert "sekret-token-123" not in json.dumps(records)
        assert (tmp_path / "venue=KALSHI" / "2026-10-06.jsonl").exists()
    finally:
        SECRETS.clear()
        assert secret is not None


# ------------------------------------------------------------------ health


def test_latency_stats_percentiles() -> None:
    stats = LatencyStats(capacity=100)
    for ms in range(1, 101):
        stats.add(ms * 1_000_000)
    stats.add(-2_000_000)
    report = stats.to_dict()
    # capacity 100: the 1 ms sample was evicted; sorted sample is [-2, 2, 3, ..., 100]
    assert report["p50_ms"] == 50.0
    assert report["p99_ms"] == 99.0
    assert report["max_ms"] == 100.0
    assert report["negative"] == 1
    assert report["total"] == 101


def test_feed_health_rate_age_and_reconnects() -> None:
    health = FeedHealth(name="f", venue=Venue.KALSHI, rate_window_ns=10 * NS_PER_S)
    health.on_connected(T0_NS)
    for i in range(20):
        health.record_message(T0_NS + i * NS_PER_S // 2)
    assert health.message_rate(T0_NS + 10 * NS_PER_S) == pytest.approx(1.9)
    health.on_disconnected(T0_NS + 11 * NS_PER_S, "closed")
    health.on_connected(T0_NS + 12 * NS_PER_S)
    report = health.to_dict(T0_NS + 12 * NS_PER_S)
    assert report["reconnects"] == 1
    assert report["disconnects"] == 1
    assert report["last_message_age_ms"] == pytest.approx(2500.0)
    assert report["state"] == "CONNECTED"


def test_book_health_and_aggregate_status() -> None:
    books = BookManager()
    snap = BookSnapshotEvent(
        venue=Venue.KALSHI,
        instrument_id="KALSHI:X",
        source_ts_ns=T0_NS,
        recv_ts_ns=T0_NS,
        payload_hash="h",
        sequence=1,
        bids=(BookLevel(D("0.4"), D(1)),),
        asks=(),
    )
    books.apply(snap)
    builder = books.get("KALSHI:X")
    assert builder is not None
    health = BookHealth.from_builder(builder, now_ns=T0_NS + NS_PER_S, last_snapshot_ns=T0_NS)
    assert health.valid
    assert health.one_sided
    assert not health.empty
    assert health.snapshot_age_ms == 1000.0
    assert health.bid_levels == 1
    feed = FeedHealth(name="f", venue=Venue.KALSHI)
    assert aggregate_status([], now_ns=T0_NS) is HealthStatus.DOWN
    assert aggregate_status([feed], now_ns=T0_NS) is HealthStatus.DOWN
    feed.on_connected(T0_NS)
    feed.record_message(T0_NS)
    assert aggregate_status([feed], [health], now_ns=T0_NS) is HealthStatus.OK
    builder.invalidate(QualityFlag.DISCONNECTED)
    bad = BookHealth.from_builder(builder, now_ns=T0_NS)
    assert not bad.valid
    assert "DISCONNECTED" in bad.flags
    assert aggregate_status([feed], [bad], now_ns=T0_NS) is HealthStatus.DEGRADED
    down = FeedHealth(name="g", venue=Venue.COINBASE, state=ConnectionState.DISCONNECTED)
    assert aggregate_status([feed, down], [health], now_ns=T0_NS) is HealthStatus.DEGRADED
    stale = aggregate_status(
        [feed], [health], now_ns=T0_NS + 61 * NS_PER_S, stale_after_ns=60 * NS_PER_S
    )
    assert stale is HealthStatus.DEGRADED
    assert health.to_dict()["instrument_id"] == "KALSHI:X"
    assert BookSide.BID.value == "BID"
