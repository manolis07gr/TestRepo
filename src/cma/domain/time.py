"""Time handling: integer UTC nanoseconds everywhere, injected clocks, DST-safe conversion.

Core logic never calls the wall clock directly; it receives a :class:`Clock`.
"""

from __future__ import annotations

import re
import time as _time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from cma.domain.errors import AmbiguousLocalTimeError, NonexistentLocalTimeError

NS_PER_US = 1_000
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_S
NS_PER_HOUR = 60 * NS_PER_MIN
NS_PER_DAY = 24 * NS_PER_HOUR

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ISO_RE = re.compile(
    r"^(?P<main>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d{1,9}))?"
    r"(?P<tz>Z|z|[+-]\d{2}:?\d{2})?$"
)


def ns_from_datetime(dt: datetime) -> int:
    """Exact integer nanoseconds since the Unix epoch for an aware datetime."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("naive datetime rejected: timestamps must be timezone-aware")
    delta = dt.astimezone(UTC) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * NS_PER_S + delta.microseconds * NS_PER_US


def datetime_from_ns(ns: int) -> datetime:
    """UTC datetime (microsecond precision; sub-microsecond digits are truncated)."""
    return _EPOCH + timedelta(microseconds=ns // NS_PER_US)


def ns_from_iso8601(text: str) -> int:
    """Parse ISO-8601 with an explicit offset/Z and up to 9 fractional digits."""
    m = _ISO_RE.match(text.strip())
    if m is None:
        raise ValueError(f"unsupported ISO-8601 timestamp: {text!r}")
    tz = m.group("tz")
    if tz is None:
        raise ValueError(f"naive timestamp rejected (no offset): {text!r}")
    tz = "+00:00" if tz in ("Z", "z") else tz
    if len(tz) == 5:  # +HHMM -> +HH:MM
        tz = f"{tz[:3]}:{tz[3:]}"
    base = datetime.fromisoformat(m.group("main").replace(" ", "T") + tz)
    frac = m.group("frac") or ""
    frac_ns = int(frac.ljust(9, "0")) if frac else 0
    return ns_from_datetime(base) + frac_ns


def iso_from_ns(ns: int) -> str:
    """Render integer nanoseconds as an ISO-8601 UTC string with 9 fractional digits."""
    seconds, rem = divmod(ns, NS_PER_S)
    dt = _EPOCH + timedelta(seconds=seconds)
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{rem:09d}Z"


def ns_from_ms(ms: int) -> int:
    return int(ms) * NS_PER_MS


def ns_from_s(seconds: int) -> int:
    return int(seconds) * NS_PER_S


def ms_from_ns(ns: int) -> float:
    return ns / NS_PER_MS


def local_to_utc_ns(local: datetime, tz_name: str, *, fold: int | None = None) -> int:
    """Convert a naive local wall-clock time in ``tz_name`` to UTC nanoseconds.

    Raises :class:`NonexistentLocalTimeError` inside a spring-forward gap and
    :class:`AmbiguousLocalTimeError` inside a fall-back overlap unless ``fold`` is given.
    """
    if local.tzinfo is not None:
        raise ValueError("local_to_utc_ns expects a naive wall-clock datetime")
    tz = ZoneInfo(tz_name)
    first = local.replace(tzinfo=tz, fold=0)
    second = local.replace(tzinfo=tz, fold=1)
    if first.utcoffset() != second.utcoffset():
        round_trip = first.astimezone(UTC).astimezone(tz).replace(tzinfo=None)
        if round_trip != local:
            raise NonexistentLocalTimeError(f"{local.isoformat()} does not exist in {tz_name}")
        if fold is None:
            raise AmbiguousLocalTimeError(
                f"{local.isoformat()} is ambiguous in {tz_name}; pass fold=0 or fold=1"
            )
        return ns_from_datetime(local.replace(tzinfo=tz, fold=fold))
    return ns_from_datetime(first)


def utc_ns_to_local(ns: int, tz_name: str) -> datetime:
    return datetime_from_ns(ns).astimezone(ZoneInfo(tz_name))


def utc_day_start_ns(ns: int) -> int:
    return ns - ns % NS_PER_DAY


class Clock(Protocol):
    """Source of 'now'. Inject this; never call the wall clock inside core logic."""

    def now_ns(self) -> int: ...


class SystemClock:
    """Wall-clock implementation, used only at process edges (collectors, CLI)."""

    def now_ns(self) -> int:
        return _time.time_ns()


@dataclass
class ManualClock:
    """Deterministic clock for tests and replay. It never moves backwards."""

    current_ns: int = 0

    def now_ns(self) -> int:
        return self.current_ns

    def set(self, ns: int) -> None:
        if ns < self.current_ns:
            raise ValueError(f"clock cannot move backwards ({ns} < {self.current_ns})")
        self.current_ns = ns

    def advance(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise ValueError("clock cannot move backwards")
        self.current_ns += delta_ns
        return self.current_ns
