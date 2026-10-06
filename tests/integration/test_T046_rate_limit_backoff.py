"""T046 rate-limit/backoff: HTTP 429 (Retry-After) and 5xx trigger bounded backoff with
deterministic sleeps; the result equals a clean fetch; exhaustion raises and records
nothing (no partial pages escape)."""

from __future__ import annotations

import asyncio
import itertools
import random
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from cma.adapters.kalshi import KalshiRestClient
from cma.domain.models import PredictionContract
from cma.domain.time import NS_PER_S, ManualClock
from cma.ingestion.rest import (
    BackoffPolicy,
    HttpFetcher,
    RateLimiter,
    RetryExhaustedError,
    RetryPolicy,
    fetch_with_backoff,
)
from cma.storage.raw import RawReader, RawRecorder, discover_parts
from tests.unit.adapters.helpers import T0_NS, FakeSleep, fixture_text, mock_client

pytestmark = pytest.mark.integration
POLICY = RetryPolicy(
    max_attempts=4,
    backoff=BackoffPolicy(initial_delay_s=0.5, max_delay_s=8.0, multiplier=2.0, jitter_ratio=0.0),
)


def page(request: httpx.Request) -> httpx.Response:
    n = 2 if request.url.params.get("cursor") else 1
    return httpx.Response(200, text=fixture_text("kalshi", f"markets_page_{n}.json"))


def scripted(responses: list[Callable[[httpx.Request], httpx.Response]]) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return queue.pop(0)(request) if queue else page(request)

    return httpx.MockTransport(handler)


def throttled(
    retry_after: str | None = None, status: int = 429
) -> Callable[[httpx.Request], httpx.Response]:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return lambda _req: httpx.Response(status, headers=headers, text="slow down")


async def fetch_markets(
    transport: httpx.MockTransport, root: Path, sleep: FakeSleep, clock: ManualClock
) -> list[PredictionContract]:
    recorder = RawRecorder(root)
    fetcher = HttpFetcher(
        mock_client(transport), clock=clock, retry=POLICY, sleep=sleep, rng=random.Random(46)
    )
    try:
        return await KalshiRestClient(fetcher, recorder=recorder).list_markets(
            series_ticker="KXBTCD"
        )
    finally:
        recorder.close()


def test_T046_429_with_retry_after_backs_off_then_matches_clean_fetch(tmp_path: Path) -> None:
    clean_clock, clean_sleep = ManualClock(T0_NS), FakeSleep()
    clean = asyncio.run(fetch_markets(scripted([]), tmp_path / "clean", clean_sleep, clean_clock))

    clock = ManualClock(T0_NS)
    sleep = FakeSleep(clock)
    transport = scripted(
        [
            throttled("2"),  # page 1: 429, Retry-After 2 s
            throttled("1"),  # page 1: 429, Retry-After 1 s
            page,  # page 1 OK (cursor -> page 2)
            throttled(None, status=503),  # page 2: 503 without Retry-After -> exponential backoff
            page,  # page 2 OK
        ]
    )
    result = asyncio.run(fetch_markets(transport, tmp_path / "throttled", sleep, clock))

    assert sleep.calls == [2.0, 1.0, 0.5]  # Retry-After honoured, then backoff attempt 1
    assert clean_sleep.calls == []
    assert result == clean
    assert len(result) == 2
    payloads = [m.payload for m in RawReader(tmp_path / "throttled").iter_messages()]
    assert payloads == [m.payload for m in RawReader(tmp_path / "clean").iter_messages()]
    assert len(payloads) == 2  # exactly the two successful pages, never the 429 bodies


def test_T046_exhaustion_raises_and_records_nothing(tmp_path: Path) -> None:
    clock = ManualClock(T0_NS)
    sleep = FakeSleep(clock)
    # page 1 succeeds, page 2 is throttled on every attempt
    transport = scripted([page] + [throttled("1")] * POLICY.max_attempts)
    with pytest.raises(RetryExhaustedError) as info:
        asyncio.run(fetch_markets(transport, tmp_path, sleep, clock))
    assert info.value.attempts == 4
    assert info.value.last_status == 429
    assert sleep.calls == [1.0, 1.0, 1.0]  # bounded: max_attempts - 1 waits
    assert discover_parts(tmp_path) == []  # the successful page 1 was not recorded either


def test_T046_retry_after_is_clamped_and_backoff_bounded() -> None:
    sleep = FakeSleep()
    statuses = iter([429, 500, 502, 504, 200])
    retry_after = iter(["3600", None, None, None, None])

    async def send() -> httpx.Response:
        value = next(retry_after)
        return httpx.Response(next(statuses), headers={"Retry-After": value} if value else {})

    policy = RetryPolicy(
        max_attempts=5,
        backoff=BackoffPolicy(
            initial_delay_s=1.0, max_delay_s=5.0, multiplier=3.0, jitter_ratio=0.5
        ),
    )
    response = asyncio.run(
        fetch_with_backoff(send, policy=policy, sleep=sleep, rng=random.Random(7))
    )
    assert response.status_code == 200
    assert sleep.calls[0] == 5.0  # one-hour Retry-After clamped to max_delay_s
    rng = random.Random(7)
    assert sleep.calls[1:] == [policy.backoff.delay(n, rng) for n in (2, 3, 4)]
    assert all(delay <= 5.0 for delay in sleep.calls)


def test_T046_rate_limiter_spaces_requests(tmp_path: Path) -> None:
    clock = ManualClock(T0_NS)
    sleep = FakeSleep(clock)
    sent_at: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_at.append(clock.now_ns())
        return httpx.Response(200, text=fixture_text("kalshi", "orderbook_rest_fp.json"))

    async def go() -> None:
        limiter = RateLimiter(2.0, burst=1.0, clock=clock, sleep=sleep)
        fetcher = HttpFetcher(
            mock_client(httpx.MockTransport(handler)), clock=clock, limiter=limiter, sleep=sleep
        )
        client = KalshiRestClient(fetcher)
        for _ in range(4):
            await client.get_orderbook("KXBTCD-26OCT0617-T110999.99")

    asyncio.run(go())
    gaps = [b - a for a, b in itertools.pairwise(sent_at)]
    assert gaps == [NS_PER_S // 2] * 3  # 2 requests/s sustained
