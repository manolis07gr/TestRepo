"""REST plumbing: token-bucket rate limiting and bounded, Retry-After-aware backoff (T046).

* :class:`RateLimiter` is a token bucket driven by an injected clock and sleep, so it is
  deterministic in tests.
* :func:`fetch_with_backoff` retries HTTP 429, 5xx and transport errors with bounded
  exponential backoff. The jitter comes from a seeded RNG. A ``Retry-After`` header
  (delta-seconds or HTTP-date) replaces the computed delay, clamped to ``max_delay_s``.
  Other 4xx statuses fail immediately. After ``max_attempts`` the call raises
  :class:`RetryExhaustedError`.
* Clients built on :class:`HttpFetcher` fetch every page of a paginated listing before
  they parse, record or return anything. A failure part-way through leaves nothing
  behind: no partial page results reach callers or the raw store.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import math
import random
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC

import httpx

from cma.domain.errors import CMAError
from cma.domain.time import NS_PER_S, Clock, ns_from_datetime

log = logging.getLogger(__name__)

type SleepFn = Callable[[float], Awaitable[None]]
type QueryValue = str | int | float | bool | None

_DELTA_SECONDS_RE = re.compile(r"^\d+(?:\.\d+)?$")


class RestError(CMAError):
    """Base class for REST ingestion failures."""


class HttpStatusError(RestError):
    """Non-retryable HTTP status (e.g. 400/401/403/404)."""

    def __init__(self, status: int, what: str, body: str = "") -> None:
        super().__init__(f"{what}: HTTP {status} {body[:200]}".rstrip())
        self.status = status


class RetryExhaustedError(RestError):
    """Every attempt failed with a retryable error (429/5xx/transport)."""

    def __init__(
        self, what: str, *, attempts: int, last_status: int | None, last_error: str | None
    ) -> None:
        super().__init__(f"{what}: giving up after {attempts} attempts ({last_error})")
        self.attempts = attempts
        self.last_status = last_status
        self.last_error = last_error


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Bounded exponential backoff with additive jitter.

    The delay for attempt ``n`` (1-based) is
    ``min(max_delay_s, base_n + U[0, jitter_ratio) * base_n)``, where
    ``base_n = min(max_delay_s, initial_delay_s * multiplier ** (n - 1))``. After
    ``max_attempts`` consecutive failures the caller gives up; ``None`` means never.
    """

    initial_delay_s: float = 0.5
    max_delay_s: float = 30.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.2
    max_attempts: int | None = 8

    def __post_init__(self) -> None:
        if self.initial_delay_s < 0 or self.max_delay_s < self.initial_delay_s:
            raise ValueError("require 0 <= initial_delay_s <= max_delay_s")
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be >= 1")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be within [0, 1]")
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1 (or None)")

    def base_delay(self, attempt: int) -> float:
        if attempt < 1:
            raise ValueError("attempt numbers start at 1")
        exponent = min(attempt - 1, 64)
        return min(self.max_delay_s, self.initial_delay_s * self.multiplier**exponent)

    def delay(self, attempt: int, rng: random.Random) -> float:
        base = self.base_delay(attempt)
        return min(self.max_delay_s, base + base * self.jitter_ratio * rng.random())


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """HTTP retry policy: which failures retry, how many attempts, how long to wait."""

    max_attempts: int = 5
    backoff: BackoffPolicy = field(
        default_factory=lambda: BackoffPolicy(initial_delay_s=0.5, max_delay_s=30.0)
    )
    retry_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})
    honor_retry_after: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def delay(self, attempt: int, rng: random.Random, retry_after_s: float | None) -> float:
        """Wait before attempt ``attempt + 1``: Retry-After if given, else backoff."""
        if self.honor_retry_after and retry_after_s is not None:
            return min(max(retry_after_s, 0.0), self.backoff.max_delay_s)
        return self.backoff.delay(attempt, rng)


def parse_retry_after(value: str | None, *, now_ns: int | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    if _DELTA_SECONDS_RE.match(text):
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if now_ns is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (ns_from_datetime(when) - now_ns) / NS_PER_S)


class RateLimiter:
    """Token bucket: ``rate_per_s`` sustained, ``burst`` tokens of headroom.

    Acquisitions are serialized (FIFO under asyncio's lock), so concurrent callers
    cannot overdraw the bucket. Time comes only from the injected clock, and waiting only
    from the injected sleep.
    """

    _EPSILON = 1e-6

    def __init__(
        self,
        rate_per_s: float,
        *,
        clock: Clock,
        burst: float | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self.rate_per_s = float(rate_per_s)
        self.capacity = float(burst) if burst is not None else max(1.0, self.rate_per_s)
        if self.capacity < 1.0:
            raise ValueError("burst must allow at least one token")
        self._clock = clock
        self._sleep: SleepFn = sleep or asyncio.sleep
        self._tokens = self.capacity
        self._last_ns = clock.now_ns()
        self._lock = asyncio.Lock()
        self.waits = 0
        self.total_wait_s = 0.0

    def _refill(self) -> None:
        now = self._clock.now_ns()
        if now > self._last_ns:
            elapsed_s = (now - self._last_ns) / NS_PER_S
            self._tokens = min(self.capacity, self._tokens + elapsed_s * self.rate_per_s)
            self._last_ns = now

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    async def acquire(self, tokens: float = 1.0) -> float:
        """Take ``tokens``, waiting as needed; returns the seconds spent waiting."""
        if tokens <= 0 or tokens > self.capacity:
            raise ValueError(f"tokens must be in (0, {self.capacity}]")
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens + self._EPSILON >= tokens:
                    self._tokens = max(0.0, self._tokens - tokens)
                    return waited
                wait_s = math.ceil((tokens - self._tokens) / self.rate_per_s * NS_PER_S) / NS_PER_S
                self.waits += 1
                self.total_wait_s += wait_s
                waited += wait_s
                await self._sleep(wait_s)


async def fetch_with_backoff(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    policy: RetryPolicy | None = None,
    sleep: SleepFn | None = None,
    rng: random.Random | None = None,
    clock: Clock | None = None,
    limiter: RateLimiter | None = None,
    what: str = "request",
) -> httpx.Response:
    """Issue ``send()`` until it succeeds (status < 400) or retries are exhausted.

    Each attempt first takes a rate-limiter token. HTTP 429/5xx and transport errors are
    retried, honouring ``Retry-After``. Other 4xx statuses raise
    :class:`HttpStatusError` at once. Exhaustion raises :class:`RetryExhaustedError`.
    """
    policy = policy or RetryPolicy()
    sleep_fn: SleepFn = sleep or asyncio.sleep
    rng = rng or random.Random(0)
    last_status: int | None = None
    last_error: str | None = None
    for attempt in range(1, policy.max_attempts + 1):
        if limiter is not None:
            await limiter.acquire()
        retry_after: float | None = None
        try:
            response = await send()
        except httpx.TransportError as exc:  # connect/read errors and timeouts
            last_status, last_error = None, f"{type(exc).__name__}: {exc}"
        else:
            status = response.status_code
            if status < 400:
                return response
            if status not in policy.retry_statuses:
                raise HttpStatusError(status, what, response.text)
            last_status, last_error = status, f"HTTP {status}"
            retry_after = parse_retry_after(
                response.headers.get("Retry-After"),
                now_ns=clock.now_ns() if clock is not None else None,
            )
        if attempt == policy.max_attempts:
            break
        delay = policy.delay(attempt, rng, retry_after)
        log.info(
            "%s failed (%s), attempt %d/%d; retrying in %.3fs",
            what,
            last_error,
            attempt,
            policy.max_attempts,
            delay,
        )
        await sleep_fn(delay)
    raise RetryExhaustedError(
        what, attempts=policy.max_attempts, last_status=last_status, last_error=last_error
    )


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """A successful response body with its receive metadata."""

    url: str
    text: str
    recv_ts_ns: int
    seq: int


class HttpFetcher:
    """GET helper shared by venue REST clients: rate limit + backoff + receive stamps.

    ``connection_id``/``seq`` identify fetched pages in the raw store the way connection
    id/sequence identify WebSocket frames.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        clock: Clock,
        limiter: RateLimiter | None = None,
        retry: RetryPolicy | None = None,
        sleep: SleepFn | None = None,
        rng: random.Random | None = None,
        connection_id: str | None = None,
    ) -> None:
        self._client = client
        self._clock = clock
        self._limiter = limiter
        self._retry = retry or RetryPolicy()
        self._sleep: SleepFn = sleep or asyncio.sleep
        self._rng = rng or random.Random(0)
        self.connection_id = connection_id or f"rest:{clock.now_ns()}"
        self._seq = 0

    @property
    def clock(self) -> Clock:
        return self._clock

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, QueryValue] | None = None,
        headers: Mapping[str, str] | None = None,
        what: str | None = None,
    ) -> FetchedPage:
        query = {k: v for k, v in (params or {}).items() if v is not None}
        request_headers = dict(headers or {})

        async def send() -> httpx.Response:
            return await self._client.get(url, params=query, headers=request_headers)

        response = await fetch_with_backoff(
            send,
            policy=self._retry,
            sleep=self._sleep,
            rng=self._rng,
            clock=self._clock,
            limiter=self._limiter,
            what=what or f"GET {url}",
        )
        self._seq += 1
        return FetchedPage(
            url=str(response.url),
            text=response.text,
            recv_ts_ns=self._clock.now_ns(),
            seq=self._seq,
        )
