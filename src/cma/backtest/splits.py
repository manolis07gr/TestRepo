"""Leakage-safe data partitioning (scope s.11.1; tests T034 and T036).

* :class:`TimePartition` - chronological train / validation / final-test boundaries.
* :class:`LockedDataset` - wraps a :class:`pandas.DataFrame` with a ``ts_ns`` column. Its
  training views never contain final-test rows; the final test can only be read after an
  explicit, recorded :meth:`LockedDataset.unlock`, and only a limited number of times.
* :func:`purged_walk_forward_splits` - purged k-fold-style splits with an embargo, for
  samples whose labels span an interval ``[start, end]`` (Lopez de Prado, ch. 7).
* :func:`walk_forward_windows` - chronological expanding/rolling train -> test windows.
* :func:`purged_train_test_split` - a single chronological split with label purging.

All timestamps are integer UTC nanoseconds. Nothing here reads the wall clock; a
:class:`~cma.domain.time.Clock` can be injected to stamp access records.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from cma.config import ResearchConfig
from cma.domain.errors import FinalTestAccessError
from cma.domain.time import NS_PER_MS, Clock

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
Split = tuple[IntArray, IntArray]

__all__ = [
    "AccessCallback",
    "AccessRecord",
    "LockedDataset",
    "Segment",
    "Split",
    "TimePartition",
    "WalkForwardWindow",
    "purged_train_test_split",
    "purged_walk_forward_splits",
    "walk_forward_windows",
]


def _as_ts_array(values: ArrayLike, name: str) -> IntArray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must hold integer nanosecond timestamps, got {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _ts_and_reach(ts_ns: ArrayLike, label_end_ns: ArrayLike | None) -> tuple[IntArray, IntArray]:
    """Timestamps and how far each row reaches (``max(ts, label_end)``)."""
    ts = _as_ts_array(ts_ns, "ts_ns")
    if label_end_ns is None:
        return ts, ts
    label = _as_ts_array(label_end_ns, "label_end_ns")
    if label.shape != ts.shape:
        raise ValueError("label_end_ns must have the same length as ts_ns")
    return ts, np.maximum(ts, label)


def _tidy(x: float) -> float:
    """Remove binary floating-point noise before floor/ceil (10 * 0.3 -> 3, not 3.0000...4)."""
    return round(x, 9)


class Segment(IntEnum):
    """Partition membership of a row."""

    EXCLUDED = -1  # before the partition start, inside an embargo buffer, or label-purged
    TRAIN = 0
    VALIDATION = 1
    FINAL_TEST = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class TimePartition:
    """Chronological train -> validation -> final-test boundaries.

    * train:      ``start_ns <= ts`` and ``max(ts, label_end) < validation_start_ns - embargo``
    * validation: ``validation_start_ns <= ts`` and
      ``max(ts, label_end) < final_test_start_ns - embargo``
    * final test: ``ts >= final_test_start_ns`` (everything later is final test too, so data
      appended after ``end_ns`` can never leak into training)

    ``embargo_ns`` is a buffer *before* each boundary that is excluded from the earlier
    partition; it protects against labels/features that look forward in time. Rows whose
    label interval ends at or after a boundary (minus the embargo) are purged from the
    earlier partition when label end times are supplied.
    """

    start_ns: int
    validation_start_ns: int
    final_test_start_ns: int
    end_ns: int
    embargo_ns: int = 0

    def __post_init__(self) -> None:
        if not (
            self.start_ns <= self.validation_start_ns <= self.final_test_start_ns <= self.end_ns
        ):
            raise ValueError(
                "TimePartition requires start <= validation_start <= final_test_start <= end"
            )
        if self.embargo_ns < 0:
            raise ValueError("embargo_ns must be non-negative")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_fractions(
        cls,
        ts_ns: ArrayLike,
        *,
        validation_fraction: float = 0.2,
        final_test_fraction: float = 0.2,
        embargo_ns: int = 0,
        by: Literal["rows", "time"] = "rows",
    ) -> TimePartition:
        """Boundaries from fractions of the rows (``by="rows"``) or of the time span."""
        ts = np.sort(_as_ts_array(ts_ns, "ts_ns"))
        if ts.size < 3:
            raise ValueError("need at least three timestamps to build a partition")
        if not (0.0 < final_test_fraction < 1.0) or not (0.0 <= validation_fraction < 1.0):
            raise ValueError("fractions must satisfy 0 < final_test < 1 and 0 <= validation < 1")
        if validation_fraction + final_test_fraction >= 1.0:
            raise ValueError("validation_fraction + final_test_fraction must be < 1")
        start, end = int(ts[0]), int(ts[-1])
        if by == "rows":
            n = ts.size
            ft_pos = min(n - 1, max(1, math.floor(_tidy(n * (1.0 - final_test_fraction)))))
            va_pos = min(
                ft_pos,
                max(1, math.floor(_tidy(n * (1.0 - final_test_fraction - validation_fraction)))),
            )
            final_start, val_start = int(ts[ft_pos]), int(ts[va_pos])
        elif by == "time":
            span = end - start
            final_start = start + math.ceil(_tidy(span * (1.0 - final_test_fraction)))
            val_start = start + math.ceil(
                _tidy(span * (1.0 - final_test_fraction - validation_fraction))
            )
        else:  # pragma: no cover - guarded by typing
            raise ValueError(f"unknown partition basis {by!r}")
        return cls(
            start_ns=start,
            validation_start_ns=val_start,
            final_test_start_ns=final_start,
            end_ns=end,
            embargo_ns=int(embargo_ns),
        )

    @classmethod
    def from_research_config(
        cls, ts_ns: ArrayLike, cfg: ResearchConfig, *, by: Literal["rows", "time"] = "rows"
    ) -> TimePartition:
        """Boundaries from :class:`cma.config.ResearchConfig` fractions and ``embargo_ms``."""
        return cls.from_fractions(
            ts_ns,
            validation_fraction=cfg.validation_fraction,
            final_test_fraction=cfg.final_test_fraction,
            embargo_ns=cfg.embargo_ms * NS_PER_MS,
            by=by,
        )

    # ------------------------------------------------------------------ queries
    @property
    def train_end_ns(self) -> int:
        """Exclusive end of the training partition (after the embargo buffer)."""
        return self.validation_start_ns - self.embargo_ns

    @property
    def validation_end_ns(self) -> int:
        """Exclusive end of the validation partition (after the embargo buffer)."""
        return self.final_test_start_ns - self.embargo_ns

    @property
    def protected_from_ns(self) -> int:
        """Start of the region protected like the final test (final test + its embargo)."""
        return self.final_test_start_ns - self.embargo_ns

    def segment(self, ts_ns: ArrayLike, label_end_ns: ArrayLike | None = None) -> NDArray[np.int8]:
        """Per-row :class:`Segment` codes (as int8)."""
        ts, reach = _ts_and_reach(ts_ns, label_end_ns)
        out = np.full(ts.shape, Segment.EXCLUDED, dtype=np.int8)
        out[(ts >= self.start_ns) & (reach < self.train_end_ns)] = Segment.TRAIN
        out[(ts >= self.validation_start_ns) & (reach < self.validation_end_ns)] = (
            Segment.VALIDATION
        )
        out[ts >= self.final_test_start_ns] = Segment.FINAL_TEST
        return out

    def train_mask(self, ts_ns: ArrayLike, label_end_ns: ArrayLike | None = None) -> BoolArray:
        return np.asarray(self.segment(ts_ns, label_end_ns) == Segment.TRAIN, dtype=np.bool_)

    def validation_mask(self, ts_ns: ArrayLike, label_end_ns: ArrayLike | None = None) -> BoolArray:
        seg = self.segment(ts_ns, label_end_ns)
        return np.asarray(seg == Segment.VALIDATION, dtype=np.bool_)

    def final_test_mask(self, ts_ns: ArrayLike) -> BoolArray:
        return np.asarray(_as_ts_array(ts_ns, "ts_ns") >= self.final_test_start_ns)

    def touches_protected(
        self, ts_ns: ArrayLike, label_end_ns: ArrayLike | None = None
    ) -> BoolArray:
        """Rows that are final-test rows, inside its embargo, or whose label reaches it."""
        _, reach = _ts_and_reach(ts_ns, label_end_ns)
        return np.asarray(reach >= self.protected_from_ns, dtype=np.bool_)

    def overlaps_final_test(self, start_ns: int, end_ns: int) -> bool:
        """Whether the closed interval ``[start_ns, end_ns]`` reaches the protected region."""
        if end_ns < start_ns:
            raise ValueError("end_ns must be >= start_ns")
        return end_ns >= self.protected_from_ns

    def assert_outside_final_test(
        self,
        ts_ns: ArrayLike,
        label_end_ns: ArrayLike | None = None,
        *,
        context: str = "training",
    ) -> None:
        """Raise :class:`FinalTestAccessError` if any row touches the protected final test."""
        bad = self.touches_protected(ts_ns, label_end_ns)
        if bool(bad.any()):
            raise FinalTestAccessError(
                f"{context} attempted to use {int(bad.sum())} row(s) at/after the locked "
                f"final-test boundary (incl. embargo) {self.protected_from_ns}"
            )


# ======================================================================================
# Locked dataset
# ======================================================================================


@dataclass(frozen=True, slots=True)
class AccessRecord:
    """An auditable final-test unlock/read event (hand it to the experiment registry)."""

    event: Literal["unlock", "final_test_read"]
    reason: str
    sequence: int
    n_rows: int
    final_test_start_ns: int
    at_ns: int | None = None  # from the injected clock, when one is supplied


AccessCallback = Callable[[AccessRecord], None]


class _GuardedILoc:
    """Positional access over the full (time-sorted) frame that refuses hidden rows."""

    __slots__ = ("_owner",)

    def __init__(self, owner: LockedDataset) -> None:
        self._owner = owner

    def __getitem__(self, key: int | slice | ArrayLike) -> pd.DataFrame:
        return self._owner._positional(key)


class LockedDataset:
    """A time-indexed frame whose final-test partition is locked away from training code.

    * :meth:`train`, :meth:`validation`, :meth:`research`, :meth:`rows_between`,
      :attr:`iloc` and :attr:`timestamps` never expose rows that touch the protected
      region (final test plus its embargo buffer, or rows whose label reaches it).
      Attempts to reach it raise :class:`FinalTestAccessError`.
    * :meth:`final_test` requires a prior :meth:`unlock` with a non-empty reason; the
      unlock and every read are recorded in :attr:`access_log` and passed to the
      ``on_access`` callback. ``unlock`` may happen once, and the final test may be read
      at most ``max_final_test_reads`` times (default 1): do not iterate against it.

    Positions used with :attr:`iloc` refer to the full frame sorted by timestamp, so hidden
    positions exist and are refused rather than silently remapped.
    """

    __slots__ = (
        "_access_log",
        "_callback",
        "_clock",
        "_frame",
        "_hidden",
        "_locked",
        "_max_reads",
        "_partition",
        "_reads",
        "_segment",
        "_ts",
        "_ts_col",
        "_unlock_reason",
    )

    def __init__(
        self,
        frame: pd.DataFrame,
        partition: TimePartition,
        *,
        ts_col: str = "ts_ns",
        label_end_col: str | None = None,
        max_final_test_reads: int = 1,
        clock: Clock | None = None,
    ) -> None:
        if ts_col not in frame.columns:
            raise KeyError(f"frame has no timestamp column {ts_col!r}")
        if max_final_test_reads < 1:
            raise ValueError("max_final_test_reads must be >= 1")
        ts = _as_ts_array(frame[ts_col].to_numpy(), ts_col)
        order = np.argsort(ts, kind="stable")
        self._frame: pd.DataFrame = frame.iloc[order].copy()
        self._ts: IntArray = ts[order]
        label_end: IntArray | None = None
        if label_end_col is not None:
            if label_end_col not in frame.columns:
                raise KeyError(f"frame has no label-end column {label_end_col!r}")
            label_end = _as_ts_array(frame[label_end_col].to_numpy(), label_end_col)[order]
        self._partition = partition
        self._segment = partition.segment(self._ts, label_end)
        self._hidden: BoolArray = partition.touches_protected(self._ts, label_end)
        self._ts_col = ts_col
        self._locked = True
        self._max_reads = max_final_test_reads
        self._reads = 0
        self._callback: AccessCallback | None = None
        self._unlock_reason = ""
        self._access_log: list[AccessRecord] = []
        self._clock = clock

    @classmethod
    def from_fractions(
        cls,
        frame: pd.DataFrame,
        *,
        ts_col: str = "ts_ns",
        validation_fraction: float = 0.2,
        final_test_fraction: float = 0.2,
        embargo_ns: int = 0,
        by: Literal["rows", "time"] = "rows",
        label_end_col: str | None = None,
        clock: Clock | None = None,
    ) -> LockedDataset:
        partition = TimePartition.from_fractions(
            frame[ts_col].to_numpy(),
            validation_fraction=validation_fraction,
            final_test_fraction=final_test_fraction,
            embargo_ns=embargo_ns,
            by=by,
        )
        return cls(frame, partition, ts_col=ts_col, label_end_col=label_end_col, clock=clock)

    # ------------------------------------------------------------------ metadata
    @property
    def partition(self) -> TimePartition:
        return self._partition

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def access_log(self) -> tuple[AccessRecord, ...]:
        return tuple(self._access_log)

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self._frame.columns]

    @property
    def ts_col(self) -> str:
        return self._ts_col

    def __len__(self) -> int:
        """Number of rows visible to research code (never counts protected rows)."""
        return int((~self._hidden).sum())

    def __repr__(self) -> str:
        return (
            f"LockedDataset(visible_rows={len(self)}, locked={self._locked}, "
            f"final_test_start_ns={self._partition.final_test_start_ns})"
        )

    # ------------------------------------------------------------------ research views
    def _rows(self, mask: BoolArray) -> pd.DataFrame:
        if bool((mask & self._hidden).any()):  # defensive: views never include hidden rows
            raise FinalTestAccessError("internal view selected protected final-test rows")
        return self._frame.iloc[np.flatnonzero(mask)].copy()

    def train(self) -> pd.DataFrame:
        """Training rows (never final-test, embargo-buffer or label-purged rows)."""
        return self._rows(self._segment == Segment.TRAIN)

    def validation(self) -> pd.DataFrame:
        """Validation rows (never final-test rows)."""
        return self._rows(self._segment == Segment.VALIDATION)

    def research(self) -> pd.DataFrame:
        """Every row that does not touch the protected final-test region."""
        return self._rows(~self._hidden)

    @property
    def timestamps(self) -> IntArray:
        """Timestamps of the visible rows only."""
        return self._ts[~self._hidden].copy()

    def rows_between(self, start_ns: int, end_ns: int) -> pd.DataFrame:
        """Visible rows with ``start_ns <= ts <= end_ns``.

        Raises :class:`FinalTestAccessError` if the window reaches the protected region
        (even after :meth:`unlock`: evaluation code must go through :meth:`final_test`).
        """
        if self._partition.overlaps_final_test(int(start_ns), int(end_ns)):
            raise FinalTestAccessError(
                f"window [{start_ns}, {end_ns}] overlaps the locked final-test partition "
                f"(protected from {self._partition.protected_from_ns})"
            )
        mask = (self._ts >= start_ns) & (self._ts <= end_ns) & ~self._hidden
        return self._rows(np.asarray(mask, dtype=np.bool_))

    @property
    def iloc(self) -> _GuardedILoc:
        """Positional access over the full time-sorted frame; protected positions raise."""
        return _GuardedILoc(self)

    def _positional(self, key: int | slice | ArrayLike) -> pd.DataFrame:
        n = len(self._ts)
        if isinstance(key, slice):
            pos = np.arange(n, dtype=np.int64)[key]
        else:
            arr = np.asarray(key)
            if arr.dtype == np.bool_:
                if arr.shape != (n,):
                    raise IndexError("boolean iloc mask must cover the full frame")
                pos = np.flatnonzero(arr).astype(np.int64)
            elif np.issubdtype(arr.dtype, np.integer):
                pos = np.atleast_1d(arr).astype(np.int64)
                if bool(((pos < -n) | (pos >= n)).any()):
                    raise IndexError("iloc position out of bounds")
                pos = np.where(pos < 0, pos + n, pos)
            else:
                raise TypeError("iloc accepts integers, slices, integer or boolean arrays")
        if bool(self._hidden[pos].any()):
            raise FinalTestAccessError(
                "iloc requested position(s) belonging to the locked final-test partition"
            )
        return self._frame.iloc[pos].copy()

    # ------------------------------------------------------------------ final test
    def _record(self, event: Literal["unlock", "final_test_read"], reason: str) -> AccessRecord:
        record = AccessRecord(
            event=event,
            reason=reason,
            sequence=len(self._access_log) + 1,
            n_rows=int((self._segment == Segment.FINAL_TEST).sum()),
            final_test_start_ns=self._partition.final_test_start_ns,
            at_ns=None if self._clock is None else self._clock.now_ns(),
        )
        self._access_log.append(record)
        if self._callback is not None:
            self._callback(record)
        return record

    def unlock(self, reason: str, *, on_access: AccessCallback | None = None) -> AccessRecord:
        """Explicitly unlock the final test. Allowed once; recorded and reported."""
        if not reason or not reason.strip():
            raise ValueError("unlocking the final test requires a non-empty reason")
        if not self._locked:
            raise FinalTestAccessError("final test was already unlocked once for this dataset")
        self._locked = False
        self._callback = on_access
        self._unlock_reason = reason.strip()
        return self._record("unlock", self._unlock_reason)

    def final_test(self) -> pd.DataFrame:
        """The final-test rows. Requires :meth:`unlock`; limited number of reads."""
        if self._locked:
            raise FinalTestAccessError(
                "final test is locked; call unlock(reason, on_access=...) explicitly first"
            )
        if self._reads >= self._max_reads:
            raise FinalTestAccessError(
                f"final test already read {self._reads} time(s); iterating against the "
                "untouched test period is not allowed"
            )
        self._reads += 1
        self._record("final_test_read", self._unlock_reason)
        return self._frame.iloc[np.flatnonzero(self._segment == Segment.FINAL_TEST)].copy()


# ======================================================================================
# Purged / embargoed splits
# ======================================================================================


def _event_arrays(event_start_ns: ArrayLike, event_end_ns: ArrayLike) -> tuple[IntArray, IntArray]:
    start = _as_ts_array(event_start_ns, "event_start_ns")
    end = _as_ts_array(event_end_ns, "event_end_ns")
    if start.shape != end.shape:
        raise ValueError("event_start_ns and event_end_ns must have the same length")
    if bool((end < start).any()):
        raise ValueError("every label interval must satisfy end >= start")
    return start, end


def purged_walk_forward_splits(
    event_start_ns: ArrayLike,
    event_end_ns: ArrayLike,
    n_splits: int,
    embargo_ns: int = 0,
    *,
    train_side: Literal["both", "past"] = "both",
) -> list[Split]:
    """Purged, embargoed splits for samples with label intervals ``[start_i, end_i]``.

    Samples are ordered by label start and cut into ``n_splits`` contiguous test folds.
    For each test fold with span ``[t0, t1]`` (``t0`` = earliest test label start,
    ``t1`` = latest test label end):

    * **purge**: a training sample is dropped if its label interval overlaps the span,
      i.e. ``start_i <= t1 and end_i >= t0``;
    * **embargo** (``train_side="both"``): samples starting in ``(t1, t1 + embargo_ns]``
      (just after the test fold) are dropped as well;
    * ``train_side="past"`` keeps only samples entirely before the fold (walk-forward);
      the embargo then removes training samples whose label ends within ``embargo_ns``
      before ``t0``, and folds without any training sample are skipped.

    Returns ``(train_idx, test_idx)`` pairs of sorted positions into the input arrays.
    """
    start, end = _event_arrays(event_start_ns, event_end_ns)
    n = start.size
    if n_splits < 2 or n_splits > n:
        raise ValueError(f"n_splits must be in [2, {n}], got {n_splits}")
    if embargo_ns < 0:
        raise ValueError("embargo_ns must be non-negative")
    order = np.argsort(start, kind="stable").astype(np.int64)
    splits: list[Split] = []
    for fold in np.array_split(order, n_splits):
        test_idx = np.sort(fold)
        t0 = int(start[test_idx].min())
        t1 = int(end[test_idx].max())
        candidate = np.ones(n, dtype=np.bool_)
        candidate[test_idx] = False
        overlap = (start <= t1) & (end >= t0)
        keep = candidate & ~overlap
        if train_side == "both":
            keep &= ~((start > t1) & (start <= t1 + embargo_ns))
        elif train_side == "past":
            keep &= end < t0 - embargo_ns
        else:  # pragma: no cover - guarded by typing
            raise ValueError(f"unknown train_side {train_side!r}")
        train_idx = np.flatnonzero(keep).astype(np.int64)
        if train_side == "past" and train_idx.size == 0:
            continue
        splits.append((train_idx, test_idx.astype(np.int64)))
    return splits


def purged_train_test_split(
    event_start_ns: ArrayLike,
    event_end_ns: ArrayLike,
    *,
    test_fraction: float,
    embargo_ns: int = 0,
) -> Split:
    """One chronological split: the last ``test_fraction`` of samples (by label start) are
    the test set; earlier samples whose label reaches ``t0 - embargo_ns`` are purged."""
    start, end = _event_arrays(event_start_ns, event_end_ns)
    n = start.size
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0, 1)")
    if n < 2:
        raise ValueError("need at least two samples to split")
    order = np.argsort(start, kind="stable").astype(np.int64)
    n_test = min(n - 1, max(1, math.ceil(_tidy(n * test_fraction))))
    test_idx = np.sort(order[n - n_test :])
    t0 = int(start[test_idx].min())
    before = order[: n - n_test]
    train_idx = np.sort(before[end[before] < t0 - embargo_ns])
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    fold: int
    train_idx: IntArray
    test_idx: IntArray
    train_start_ns: int | None
    train_end_ns: int | None  # latest training timestamp (inclusive)
    test_start_ns: int
    test_end_ns: int


def walk_forward_windows(
    ts_ns: ArrayLike,
    *,
    n_folds: int,
    mode: Literal["expanding", "rolling"] = "expanding",
    rolling_blocks: int = 1,
    embargo_ns: int = 0,
    label_end_ns: ArrayLike | None = None,
) -> list[WalkForwardWindow]:
    """Chronological train -> test windows.

    The time-sorted samples are cut into ``n_folds + 1`` contiguous blocks. Fold ``k``
    (1-based) tests on block ``k`` and trains on blocks ``0..k-1`` (``"expanding"``) or on
    the last ``rolling_blocks`` blocks before it (``"rolling"``). Training samples whose
    label end (or timestamp, when no label ends are given) reaches
    ``test_start - embargo_ns`` are purged.
    """
    ts = _as_ts_array(ts_ns, "ts_ns")
    reach = ts if label_end_ns is None else _as_ts_array(label_end_ns, "label_end_ns")
    if reach.shape != ts.shape:
        raise ValueError("label_end_ns must have the same length as ts_ns")
    if n_folds < 1 or n_folds + 1 > ts.size:
        raise ValueError("n_folds must be >= 1 and leave at least one sample per block")
    if rolling_blocks < 1:
        raise ValueError("rolling_blocks must be >= 1")
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"unknown walk-forward mode {mode!r}")
    order = np.argsort(ts, kind="stable").astype(np.int64)
    blocks = np.array_split(order, n_folds + 1)
    windows: list[WalkForwardWindow] = []
    for k in range(1, n_folds + 1):
        test_idx = np.sort(blocks[k])
        first = 0 if mode == "expanding" else max(0, k - rolling_blocks)
        train_pool = np.concatenate(blocks[first:k])
        test_start = int(ts[test_idx].min())
        train_idx = np.sort(train_pool[reach[train_pool] < test_start - embargo_ns])
        windows.append(
            WalkForwardWindow(
                fold=k,
                train_idx=train_idx.astype(np.int64),
                test_idx=test_idx.astype(np.int64),
                train_start_ns=int(ts[train_idx].min()) if train_idx.size else None,
                train_end_ns=int(ts[train_idx].max()) if train_idx.size else None,
                test_start_ns=test_start,
                test_end_ns=int(ts[test_idx].max()),
            )
        )
    return windows
