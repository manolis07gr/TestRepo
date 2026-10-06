"""Lead-lag series utilities: max-age LOCF, returns, alignment, causal features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cma.models.lead_lag.series import (
    EventSeries,
    align_pair,
    change,
    forward_change,
    grid_returns,
    lagged_window_changes,
    locf_at,
    make_grid,
    resample_locf,
)

pytestmark = pytest.mark.unit

NAN = math.nan


def test_locf_carries_forward_only_up_to_max_age() -> None:
    ts = np.array([0, 100, 400], dtype=np.int64)
    grid = np.array([-10, 0, 50, 100, 150, 250, 350, 400, 450], dtype=np.int64)
    out = resample_locf(ts, [1.0, 2.0, 3.0], grid, max_age_ns=150)
    np.testing.assert_array_equal(out.values, [NAN, 1, 1, 2, 2, 2, NAN, 3, 3])
    assert out.stale.tolist() == [True, False, False, False, False, False, True, False, False]
    assert out.age_ns.tolist() == [-1, 0, 50, 0, 50, 150, 250, 0, 50]
    assert out.source_ts_ns.tolist() == [-1, 0, 0, 100, 100, 100, 100, 400, 400]
    assert out.watermark_ns == 400


def test_locf_never_bridges_a_gap_beyond_max_age() -> None:
    series = EventSeries.from_arrays([0, 10_000], [1.0, 2.0])
    grid = make_grid(0, 10_000, 1_000)
    out = locf_at(series, grid, max_age_ns=2_500)
    assert np.isnan(out.values[3:10]).all()  # 3_000 .. 9_000 are stale, not forward-filled
    assert out.values[2] == 1.0
    assert out.values[10] == 2.0


def test_locf_is_causal_inclusive_of_the_decision_time() -> None:
    series = EventSeries.from_arrays([0, 100, 101], [1.0, 2.0, 3.0])
    out = locf_at(series, np.array([100]), max_age_ns=1_000)
    assert out.values.tolist() == [2.0]  # the observation at t is known, t+1 is not


def test_event_series_cleaning() -> None:
    cleaned = EventSeries.from_arrays([0, 0, 10, 20], [1.0, 2.0, NAN, 4.0])
    assert cleaned.ts_ns.tolist() == [0, 20]  # NaN dropped, last duplicate wins
    assert cleaned.values.tolist() == [2.0, 4.0]
    with pytest.raises(ValueError, match="non-decreasing"):
        EventSeries.from_arrays([10, 0], [1.0, 2.0])
    with pytest.raises(TypeError, match="integer"):
        EventSeries.from_arrays([0.5, 1.0], [1.0, 2.0])


def test_make_grid_alignment() -> None:
    assert make_grid(105, 430, 100).tolist() == [200, 300, 400]
    assert make_grid(105, 430, 100, align=False).tolist() == [105, 205, 305, 405]
    assert make_grid(500, 400, 100).size == 0


def test_change_kinds_and_nan_propagation() -> None:
    np.testing.assert_allclose(change([2.0, 4.0], [1.0, 2.0], "log"), [math.log(2)] * 2)
    np.testing.assert_allclose(change([2.0, 4.0], [1.0, 2.0], "arith"), [1.0, 1.0])
    np.testing.assert_allclose(change([2.0, 4.0], [1.0, 2.0], "diff"), [1.0, 2.0])
    np.testing.assert_array_equal(grid_returns([1.0, 2.0, NAN, 4.0], "diff"), [NAN, 1, NAN, NAN])
    with pytest.raises(ValueError, match="positive"):
        change([1.0], [0.0], "log")


def test_align_pair_uses_the_common_window_on_an_aligned_grid() -> None:
    x_ts = np.arange(0, 1_001, 100, dtype=np.int64)
    y_ts = np.arange(250, 1_501, 50, dtype=np.int64)
    pair = align_pair(
        x_ts,
        np.arange(x_ts.size, dtype=float),
        y_ts,
        np.arange(y_ts.size, dtype=float),
        step_ns=100,
        max_age_x_ns=100,
        max_age_y_ns=100,
    )
    assert pair.grid_ns.tolist() == list(range(300, 1_001, 100))
    assert pair.both_fresh.all()
    assert pair.y.values[0] == 1.0  # y observation at 300 is index 1


def test_lagged_window_changes_hand_example_and_causality() -> None:
    ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    vals = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    series = EventSeries.from_arrays(ts, vals)
    feats = lagged_window_changes(
        series, [400], bucket_ns=100, n_buckets=3, max_age_ns=1_000, kind="diff"
    )
    assert feats.tolist() == [[4.0, 3.0, 2.0]]
    future = EventSeries.from_arrays(np.append(ts, 450), np.append(vals, 99.0))
    same = lagged_window_changes(
        future, [400], bucket_ns=100, n_buckets=3, max_age_ns=1_000, kind="diff"
    )
    np.testing.assert_array_equal(same, feats)  # later observations cannot leak in
    target = forward_change(series, [100], horizon_ns=200, max_age_ns=1_000, kind="diff")
    assert target.tolist() == [5.0]


def test_lagged_window_changes_are_nan_when_stale() -> None:
    series = EventSeries.from_arrays([0, 1_000], [0.0, 5.0])
    feats = lagged_window_changes(
        series, [900, 1_000], bucket_ns=100, n_buckets=1, max_age_ns=300, kind="diff"
    )
    assert np.isnan(feats).all()
