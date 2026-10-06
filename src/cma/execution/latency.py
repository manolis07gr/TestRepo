"""Latency model: feed, compute, outbound, acknowledgement and cancel delays (scope s.12).

Deterministic by default; optional seeded log-normal jitter around each median.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from cma.config import SimulationConfig
from cma.domain.time import NS_PER_MS


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    outbound_ms: float
    compute_ms: float = 5.0
    ack_ms: float = 50.0
    cancel_ms: float | None = None  # defaults to outbound_ms
    extra_feed_ms: float = 0.0
    jitter: str = "none"  # "none" | "lognormal"
    jitter_sigma: float = 0.25

    def __post_init__(self) -> None:
        for name in ("outbound_ms", "compute_ms", "ack_ms", "extra_feed_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.cancel_ms is not None and self.cancel_ms < 0:
            raise ValueError("cancel_ms must be non-negative")
        if self.jitter not in ("none", "lognormal"):
            raise ValueError(f"unknown jitter model {self.jitter!r}")

    @property
    def name(self) -> str:
        return f"{self.outbound_ms:g}ms"

    @classmethod
    def from_config(cls, sim: SimulationConfig, outbound_ms: float) -> LatencyProfile:
        return cls(
            outbound_ms=outbound_ms,
            compute_ms=sim.compute_latency_ms,
            ack_ms=sim.ack_latency_ms,
            cancel_ms=sim.cancel_latency_ms,
            extra_feed_ms=sim.extra_feed_latency_ms,
            jitter=sim.latency_jitter,
            jitter_sigma=sim.latency_jitter_sigma,
        )

    def with_outbound(self, outbound_ms: float) -> LatencyProfile:
        return LatencyProfile(
            outbound_ms=outbound_ms,
            compute_ms=self.compute_ms,
            ack_ms=self.ack_ms,
            cancel_ms=self.cancel_ms,
            extra_feed_ms=self.extra_feed_ms,
            jitter=self.jitter,
            jitter_sigma=self.jitter_sigma,
        )


class LatencyModel:
    def __init__(self, profile: LatencyProfile, seed: int = 0) -> None:
        self.profile = profile
        self._rng = random.Random(seed)

    def _draw(self, median_ms: float) -> int:
        if median_ms <= 0:
            return 0
        if self.profile.jitter == "none":
            return round(median_ms * NS_PER_MS)
        factor = math.exp(self.profile.jitter_sigma * self._rng.gauss(0.0, 1.0))
        return round(median_ms * factor * NS_PER_MS)

    def compute_ns(self) -> int:
        return self._draw(self.profile.compute_ms)

    def outbound_ns(self) -> int:
        return self._draw(self.profile.outbound_ms)

    def ack_ns(self) -> int:
        return self._draw(self.profile.ack_ms)

    def cancel_ns(self) -> int:
        cancel = self.profile.cancel_ms
        return self._draw(self.profile.outbound_ms if cancel is None else cancel)

    def feed_ns(self) -> int:
        return self._draw(self.profile.extra_feed_ms)

    def schedule_order(self, decision_ts_ns: int) -> tuple[int, int]:
        """(submit_ts, arrival_ts): arrival = decision + compute + outbound."""
        submit = decision_ts_ns + self.compute_ns()
        return submit, submit + self.outbound_ns()
