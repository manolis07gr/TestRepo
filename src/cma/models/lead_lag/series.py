"""Event-time series handling for lead-lag research.

Conventions
-----------
* Timestamps are int64 UTC nanoseconds; values are float64.
* An observation stamped ``ts`` is known at every decision time ``t >= ts`` (inclusive).
* Resampling is last-observation-carried-forward (LOCF) **only up to a maximum age**:
  beyond it the resampled value is NaN and flagged stale, so a gap in the data is never
  bridged by forward-filling (scope s.11.1).
* Changes ("returns") come in three kinds: ``"log"`` (log v1 - log v0), ``"arith"``
  (v1 / v0 - 1) and ``"diff"`` (v1 - v0, e.g. for probabilities or log-prices).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cma.domain.errors import LookaheadError

ReturnKind = Literal["log", "arith", "diff"]
RETURN_KINDS: tuple[ReturnKind, ...] = ("log", "arith", "diff")

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__ = [
    "RETURN_KINDS",
    "AlignedPair",
    "EventSeries",
    "ResampledSeries",
    "ReturnKind",
    "align_pair",
    "change",
    "forward_change",
    "grid_returns",
    "lagged_window_changes",
    "locf_at",
    "make_grid",
    "resample_locf",
]


@dataclass(frozen=True, slots=True)
class EventSeries:
    """A cleaned irregular series: strictly increasing timestamps, finite values."""

    ts_ns: IntArray
    values: FloatArray
    name: str = "series"

    @classmethod
    def from_arrays(
        cls, ts_ns: ArrayLike, values: ArrayLike, *, name: str = "series"
    ) -> EventSeries:
        """Validate and clean raw observations.

        Non-finite values are dropped; timestamps must be non-decreasing (raises
        ``ValueError`` otherwise); for duplicate timestamps the last observation wins.
        """
        ts = np.asarray(ts_ns)
        vals = np.asarray(values, dtype=np.float64)
        if ts.ndim != 1 or vals.ndim != 1 or ts.shape != vals.shape:
            raise ValueError(f"{name}: ts_ns and values must be 1-D arrays of equal length")
        if ts.size and not np.issubdtype(ts.dtype, np.integer):
            raise TypeError(f"{name}: ts_ns must be integer nanoseconds, got {ts.dtype}")
        ts64 = ts.astype(np.int64, copy=False)
        if ts64.size > 1 and bool((np.diff(ts64) < 0).any()):
            raise ValueError(f"{name}: timestamps must be non-decreasing")
        finite = np.isfinite(vals)
        ts64, vals = ts64[finite], vals[finite]
        if ts64.size > 1:
            last_of_run = np.ones(ts64.size, dtype=np.bool_)
            last_of_run[:-1] = ts64[1:] != ts64[:-1]
            ts64, vals = ts64[last_of_run], vals[last_of_run]
        return cls(np.ascontiguousarray(ts64), np.ascontiguousarray(vals), name)

    def __len__(self) -> int:
        return int(self.ts_ns.size)

    @property
    def start_ns(self) -> int:
        return int(self.ts_ns[0])

    @property
    def end_ns(self) -> int:
        return int(self.ts_ns[-1])


def _as_series(ts_ns: ArrayLike | EventSeries, values: ArrayLike | None, name: str) -> EventSeries:
    if isinstance(ts_ns, EventSeries):
        return ts_ns
    if values is None:
        raise ValueError(f"{name}: values are required with raw timestamps")
    return EventSeries.from_arrays(ts_ns, values, name=name)


@dataclass(frozen=True, slots=True)
class ResampledSeries:
    """LOCF resampling result on a set of query times."""

    at_ns: IntArray
    values: FloatArray  # NaN where stale (no observation within max age)
    age_ns: IntArray  # age of the carried observation; -1 where there is none
    stale: BoolArray
    source_ts_ns: IntArray  # timestamp of the carried observation; -1 where none

    @property
    def watermark_ns(self) -> int:
        """Latest source timestamp actually used (-1 if nothing was used)."""
        used = self.source_ts_ns[~self.stale]
        return int(used.max()) if used.size else -1


def make_grid(start_ns: int, end_ns: int, step_ns: int, *, align: bool = True) -> IntArray:
    """Regular grid of decision times in ``[start_ns, end_ns]``.

    With ``align`` the first point is the first multiple of ``step_ns`` at or after
    ``start_ns`` (so grids from different series line up).
    """
    if step_ns <= 0:
        raise ValueError("step_ns must be positive")
    first = -(-start_ns // step_ns) * step_ns if align else start_ns
    if end_ns < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, end_ns + 1, step_ns, dtype=np.int64)


def locf_at(series: EventSeries, at_ns: ArrayLike, max_age_ns: int) -> ResampledSeries:
    """Value of the last observation with ``ts <= t`` for each query time ``t``.

    Observations older than ``max_age_ns`` are not carried: the value is NaN and the
    point is flagged stale. Never reads observations stamped after ``t``.
    """
    if max_age_ns < 0:
        raise ValueError("max_age_ns must be non-negative")
    at = np.asarray(at_ns, dtype=np.int64)
    flat = at.reshape(-1)
    idx = np.searchsorted(series.ts_ns, flat, side="right") - 1
    has = idx >= 0
    safe = np.maximum(idx, 0)
    src_ts = np.where(has, series.ts_ns[safe] if series.ts_ns.size else -1, -1)
    age = np.where(has, flat - src_ts, -1)
    fresh = has & (age <= max_age_ns)
    if bool((fresh & (src_ts > flat)).any()):  # invariant of searchsorted(side="right")
        raise LookaheadError("LOCF used an observation stamped after the query time")
    vals = np.where(fresh, series.values[safe] if series.values.size else np.nan, np.nan)
    shape = at.shape
    return ResampledSeries(
        at_ns=at,
        values=np.asarray(vals, dtype=np.float64).reshape(shape),
        age_ns=np.asarray(age, dtype=np.int64).reshape(shape),
        stale=np.asarray(~fresh, dtype=np.bool_).reshape(shape),
        source_ts_ns=np.asarray(src_ts, dtype=np.int64).reshape(shape),
    )


def resample_locf(
    ts_ns: ArrayLike | EventSeries,
    values: ArrayLike | None,
    grid_ns: ArrayLike,
    max_age_ns: int,
) -> ResampledSeries:
    """Resample an irregular series onto ``grid_ns`` with max-age-limited LOCF."""
    return locf_at(_as_series(ts_ns, values, "series"), grid_ns, max_age_ns)


def change(end: ArrayLike, start: ArrayLike, kind: ReturnKind) -> FloatArray:
    """Change from ``start`` to ``end`` values; NaN propagates.

    ``"log"`` and ``"arith"`` require strictly positive finite values (ValueError).
    """
    e = np.asarray(end, dtype=np.float64)
    s = np.asarray(start, dtype=np.float64)
    if kind == "diff":
        return np.asarray(e - s, dtype=np.float64)
    if kind not in ("log", "arith"):
        raise ValueError(f"unknown return kind {kind!r}")
    for arr in (e, s):
        finite = arr[np.isfinite(arr)]
        if bool((finite <= 0).any()):
            raise ValueError(f"{kind} returns need strictly positive values")
    if kind == "log":
        return np.asarray(np.log(e) - np.log(s), dtype=np.float64)
    return np.asarray(e / s - 1.0, dtype=np.float64)


def grid_returns(values: ArrayLike, kind: ReturnKind) -> FloatArray:
    """One-step changes of a gridded series; element 0 is NaN."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if v.size > 1:
        out[1:] = change(v[1:], v[:-1], kind)
    return out


@dataclass(frozen=True, slots=True)
class AlignedPair:
    """Two series resampled onto one common grid."""

    grid_ns: IntArray
    step_ns: int
    x: ResampledSeries
    y: ResampledSeries
    x_series: EventSeries
    y_series: EventSeries

    @property
    def n(self) -> int:
        return int(self.grid_ns.size)

    @property
    def both_fresh(self) -> BoolArray:
        return np.asarray(~self.x.stale & ~self.y.stale, dtype=np.bool_)


def align_pair(
    x_ts_ns: ArrayLike | EventSeries,
    x_values: ArrayLike | None,
    y_ts_ns: ArrayLike | EventSeries,
    y_values: ArrayLike | None,
    *,
    step_ns: int,
    max_age_x_ns: int,
    max_age_y_ns: int,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> AlignedPair:
    """Resample X and Y onto a common aligned grid over their overlapping time range."""
    xs = _as_series(x_ts_ns, x_values, "x")
    ys = _as_series(y_ts_ns, y_values, "y")
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("both series need at least one finite observation")
    lo = max(xs.start_ns, ys.start_ns) if start_ns is None else start_ns
    hi = min(xs.end_ns, ys.end_ns) if end_ns is None else end_ns
    grid = make_grid(lo, hi, step_ns)
    return AlignedPair(
        grid_ns=grid,
        step_ns=step_ns,
        x=locf_at(xs, grid, max_age_x_ns),
        y=locf_at(ys, grid, max_age_y_ns),
        x_series=xs,
        y_series=ys,
    )


def lagged_window_changes(
    series: EventSeries,
    at_ns: ArrayLike,
    *,
    bucket_ns: int,
    n_buckets: int,
    max_age_ns: int,
    kind: ReturnKind,
) -> FloatArray:
    """Distributed-lag features: change over ``(t - (j+1) b, t - j b]`` for ``j < n_buckets``.

    Every bucket is the difference of two trailing-window changes ending at ``t``, so the
    features use only information stamped at or before the decision time ``t``. Shape
    ``(len(at_ns), n_buckets)``; NaN where any endpoint is stale.
    """
    if bucket_ns <= 0 or n_buckets <= 0:
        raise ValueError("bucket_ns and n_buckets must be positive")
    at = np.asarray(at_ns, dtype=np.int64).reshape(-1)
    offsets = np.arange(n_buckets + 1, dtype=np.int64) * bucket_ns
    levels = locf_at(series, at[:, None] - offsets[None, :], max_age_ns).values
    return change(levels[:, :-1], levels[:, 1:], kind)


def forward_change(
    series: EventSeries,
    at_ns: ArrayLike,
    *,
    horizon_ns: int,
    max_age_ns: int,
    kind: ReturnKind,
) -> FloatArray:
    """Target: change of the LOCF value over ``(t, t + horizon]``.

    This deliberately reads the future and must only be used as a training/evaluation
    label, never as a feature.
    """
    if horizon_ns <= 0:
        raise ValueError("horizon_ns must be positive")
    at = np.asarray(at_ns, dtype=np.int64).reshape(-1)
    now = locf_at(series, at, max_age_ns).values
    later = locf_at(series, at + horizon_ns, max_age_ns).values
    return change(later, now, kind)
