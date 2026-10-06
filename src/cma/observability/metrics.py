"""Minimal in-process metrics registry (counters, gauges, histograms) with JSON export.

Deliberately dependency-free; exporters (Prometheus text format, JSON) read snapshots.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BUCKETS_MS = (1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)


@dataclass
class Histogram:
    buckets: Sequence[float] = DEFAULT_BUCKETS_MS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    n: int = 0
    _samples: list[float] = field(default_factory=list)
    max_samples: int = 10_000

    def __post_init__(self) -> None:
        self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, value)] += 1
        self.total += value
        self.n += 1
        if len(self._samples) < self.max_samples:
            self._samples.append(value)

    def quantile(self, q: float) -> float:
        if not self._samples:
            return math.nan
        s = sorted(self._samples)
        idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
        return s[idx]

    def snapshot(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.total / self.n if self.n else math.nan,
            "p50": self.quantile(0.5),
            "p99": self.quantile(0.99),
            "buckets": dict(zip([*map(str, self.buckets), "+Inf"], self.counts, strict=True)),
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = {}
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, Histogram] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            h = self.histograms.get(name)
            if h is None:
                h = Histogram()
                self.histograms[name] = h
            h.observe(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {k: h.snapshot() for k, h in self.histograms.items()},
            }

    def prometheus_text(self, prefix: str = "cma_") -> str:
        lines = []
        snap = self.snapshot()
        for k, v in sorted(snap["counters"].items()):
            lines.append(f"{prefix}{_safe(k)}_total {v}")
        for k, v in sorted(snap["gauges"].items()):
            lines.append(f"{prefix}{_safe(k)} {v}")
        for k, h in sorted(snap["histograms"].items()):
            lines.append(f"{prefix}{_safe(k)}_count {h['n']}")
            lines.append(f"{prefix}{_safe(k)}_p99 {h['p99']}")
        return "\n".join(lines) + "\n"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)
