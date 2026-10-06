"""Deterministic synthetic pairs for lead-lag tests and simulation studies (T037-T039).

* :func:`lead_lag_pair` - X is a Brownian log-price observed at Poisson times; Y(t) =
  X(t - lag) + independent observation noise, observed at its own Poisson times.
* :func:`independent_pair` - two independent random walks (null, T038).
* :func:`contemporaneous_pair` - common shocks hit X and Y at identical timestamps, with
  no lagged relation (T039).

The latent path is simulated exactly at the union of all times it is needed, so the
lag is exact to the timestamp resolution. Everything is driven by ``numpy`` generators
seeded explicitly; no wall clock is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

NS_PER_S = 1_000_000_000
DEFAULT_START_NS = 1_791_208_800_000_000_000  # 2026-10-05T14:00:00Z
DEFAULT_X0 = math.log(60_000.0)  # log of a BTC-like price

__all__ = [
    "DEFAULT_START_NS",
    "SyntheticPair",
    "brownian_at",
    "contemporaneous_pair",
    "independent_pair",
    "lead_lag_pair",
    "pair_from_frame",
    "poisson_times",
]


@dataclass(frozen=True, slots=True)
class SyntheticPair:
    x_ts_ns: IntArray
    x_values: FloatArray
    y_ts_ns: IntArray
    y_values: FloatArray

    def to_frame(self) -> pd.DataFrame:
        """Long format with columns ``series`` ("x"/"y"), ``ts_ns`` (int64), ``value``."""
        frame = pd.DataFrame(
            {
                "series": np.concatenate(
                    [np.full(self.x_ts_ns.size, "x"), np.full(self.y_ts_ns.size, "y")]
                ),
                "ts_ns": np.concatenate([self.x_ts_ns, self.y_ts_ns]).astype(np.int64),
                "value": np.concatenate([self.x_values, self.y_values]).astype(np.float64),
            }
        )
        return frame.sort_values(["ts_ns", "series"], kind="stable").reset_index(drop=True)


def pair_from_frame(
    frame: pd.DataFrame, *, x_label: str = "x", y_label: str = "y"
) -> SyntheticPair:
    """Split a long ``series/ts_ns/value`` frame into X and Y arrays (time-sorted)."""
    out: list[tuple[IntArray, FloatArray]] = []
    for label in (x_label, y_label):
        part = frame[frame["series"] == label].sort_values("ts_ns", kind="stable")
        out.append(
            (
                part["ts_ns"].to_numpy(dtype=np.int64),
                part["value"].to_numpy(dtype=np.float64),
            )
        )
    return SyntheticPair(out[0][0], out[0][1], out[1][0], out[1][1])


def poisson_times(
    rng: np.random.Generator,
    rate_hz: float,
    start_ns: int,
    end_ns: int,
    *,
    resolution_ns: int = 1_000,
) -> IntArray:
    """Homogeneous Poisson arrival times in ``[start_ns, end_ns)``, rounded and unique."""
    if rate_hz <= 0 or end_ns <= start_ns:
        raise ValueError("rate_hz must be positive and end_ns > start_ns")
    duration_s = (end_ns - start_ns) / NS_PER_S
    n = int(rng.poisson(rate_hz * duration_s))
    raw = np.sort(rng.random(n)) * (end_ns - start_ns)
    ticks = start_ns + (raw // resolution_ns).astype(np.int64) * resolution_ns
    return np.unique(ticks).astype(np.int64)


def brownian_at(
    rng: np.random.Generator, times_ns: IntArray, sigma_per_sqrt_s: float, x0: float
) -> FloatArray:
    """Exact Brownian motion ``x0 + sigma * B(t - t_0)`` sampled at sorted unique times."""
    if times_ns.size == 0:
        return np.empty(0, dtype=np.float64)
    dt_s = np.diff(times_ns).astype(np.float64) / NS_PER_S
    steps = sigma_per_sqrt_s * np.sqrt(dt_s) * rng.standard_normal(dt_s.size)
    return np.asarray(x0 + np.concatenate(([0.0], np.cumsum(steps))), dtype=np.float64)


def lead_lag_pair(
    *,
    seed: int,
    duration_s: float = 5_400.0,
    lag_ms: int = 2_000,
    x_rate_hz: float = 5.0,
    y_rate_hz: float = 2.0,
    sigma: float = 1e-4,
    noise_std: float = 1e-4,
    x0: float = DEFAULT_X0,
    start_ns: int = DEFAULT_START_NS,
) -> SyntheticPair:
    """X leads Y by exactly ``lag_ms``: ``Y(s) = X_latent(s - lag) + eps``."""
    rng = np.random.default_rng(seed)
    end_ns = start_ns + int(duration_s * NS_PER_S)
    lag_ns = int(lag_ms) * 1_000_000
    x_ts = poisson_times(rng, x_rate_hz, start_ns, end_ns)
    y_ts = poisson_times(rng, y_rate_hz, start_ns, end_ns)
    latent_t, inverse = np.unique(np.concatenate([x_ts, y_ts - lag_ns]), return_inverse=True)
    latent = brownian_at(rng, latent_t.astype(np.int64), sigma, x0)
    x_vals = latent[inverse[: x_ts.size]]
    y_vals = latent[inverse[x_ts.size :]] + noise_std * rng.standard_normal(y_ts.size)
    return SyntheticPair(x_ts, x_vals, y_ts, np.asarray(y_vals, dtype=np.float64))


def independent_pair(
    *,
    seed: int,
    duration_s: float = 5_400.0,
    x_rate_hz: float = 5.0,
    y_rate_hz: float = 2.0,
    sigma: float = 1e-4,
    noise_std: float = 1e-4,
    x0: float = DEFAULT_X0,
    start_ns: int = DEFAULT_START_NS,
) -> SyntheticPair:
    """Two independent random walks (Y with observation noise): the null hypothesis."""
    rng = np.random.default_rng(seed)
    end_ns = start_ns + int(duration_s * NS_PER_S)
    x_ts = poisson_times(rng, x_rate_hz, start_ns, end_ns)
    y_ts = poisson_times(rng, y_rate_hz, start_ns, end_ns)
    x_vals = brownian_at(rng, x_ts, sigma, x0)
    y_vals = brownian_at(rng, y_ts, sigma, x0) + noise_std * rng.standard_normal(y_ts.size)
    return SyntheticPair(x_ts, x_vals, y_ts, np.asarray(y_vals, dtype=np.float64))


def contemporaneous_pair(
    *,
    seed: int,
    duration_s: float = 3_600.0,
    rate_hz: float = 5.0,
    common_sigma: float = 1e-4,
    idio_sigma: float = 5e-5,
    x0: float = DEFAULT_X0,
    start_ns: int = DEFAULT_START_NS,
) -> SyntheticPair:
    """Common shocks move X and Y at identical timestamps; there is no lagged relation."""
    rng = np.random.default_rng(seed)
    end_ns = start_ns + int(duration_s * NS_PER_S)
    ts = poisson_times(rng, rate_hz, start_ns, end_ns)
    dt_s = np.diff(ts).astype(np.float64) / NS_PER_S
    scale = np.sqrt(dt_s)
    common = common_sigma * scale * rng.standard_normal(dt_s.size)
    x_steps = common + idio_sigma * scale * rng.standard_normal(dt_s.size)
    y_steps = common + idio_sigma * scale * rng.standard_normal(dt_s.size)
    x_vals = x0 + np.concatenate(([0.0], np.cumsum(x_steps)))
    y_vals = x0 + np.concatenate(([0.0], np.cumsum(y_steps)))
    return SyntheticPair(
        ts.copy(), np.asarray(x_vals, dtype=np.float64), ts.copy(), np.asarray(y_vals)
    )
