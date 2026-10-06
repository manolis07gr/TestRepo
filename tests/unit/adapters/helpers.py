"""Shared helpers for adapter/ingestion tests: fixtures, raw envelopes, fake sleeps."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx

from cma.domain.enums import Venue
from cma.domain.models import RawMessage
from cma.domain.time import ManualClock

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "adapters"
T0_NS = 1_791_318_600_000_000_000  # 2026-10-06T20:30:00Z
RECV_NS = T0_NS + 1_000_000_000

KALSHI_TICKER = "KXBTCD-26OCT0617-T110999.99"
KALSHI_INST = f"KALSHI:{KALSHI_TICKER}"
YES_TOKEN = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
NO_TOKEN = "52114319501245915516055106046884209969926127482827954674443846427813813222426"
YES_INST = f"POLYMARKET:{YES_TOKEN}"
NO_INST = f"POLYMARKET:{NO_TOKEN}"
CONDITION = "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1"


def fixture_text(venue: str, name: str) -> str:
    return (FIXTURES / venue / name).read_text(encoding="utf-8")


def raw_msg(
    venue: Venue,
    payload: str,
    *,
    stream: str = "ws",
    recv_ts_ns: int = RECV_NS,
    connection_id: str = "conn-1",
    connection_seq: int = 1,
    idempotency_key: str | None = None,
) -> RawMessage:
    return RawMessage(
        venue=venue,
        stream=stream,
        recv_ts_ns=recv_ts_ns,
        payload=payload,
        connection_id=connection_id,
        connection_seq=connection_seq,
        idempotency_key=idempotency_key,
    )


class FakeSleep:
    """Injected async sleep: records requested delays, advances a ManualClock, never waits."""

    def __init__(self, clock: ManualClock | None = None) -> None:
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.clock is not None:
            self.clock.advance(round(seconds * 1_000_000_000))
        await asyncio.sleep(0)


def mock_client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    """AsyncClient whose requests are answered by ``handler`` (no network)."""
    return httpx.AsyncClient(transport=handler)


class _Stoppable(Protocol):
    def request_stop(self) -> None: ...


def stop_session(get: Callable[[], _Stoppable]) -> Callable[[], None]:
    """Script item stopping a session that is created after the script (late binding)."""

    def stop() -> None:
        get().request_stop()

    return stop
