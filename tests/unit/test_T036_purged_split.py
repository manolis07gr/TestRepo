"""T036 - Purged split: overlapping label horizons are purged/embargoed at fold boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from cma.backtest.splits import (
    purged_train_test_split,
    purged_walk_forward_splits,
    walk_forward_windows,
)

pytestmark = pytest.mark.unit

# Ten samples; label of sample i spans [10 i, 10 i + 15], so each label overlaps the next.
START = np.arange(10, dtype=np.int64) * 10
END = START + 15


def _as_lists(splits: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[list[int], list[int]]]:
    return [(tr.tolist(), te.tolist()) for tr, te in splits]


def test_T036_hand_checked_purge_and_embargo() -> None:
    splits = _as_lists(purged_walk_forward_splits(START, END, n_splits=5, embargo_ns=10))
    assert splits == [
        # test {0,1} spans [0, 25]: 2 overlaps (purged), 3 starts at 30 in (25, 35] (embargo)
        ([4, 5, 6, 7, 8, 9], [0, 1]),
        # test {2,3} spans [20, 45]: 1 ([10,25]) and 4 ([40,55]) purged, 5 (50) embargoed
        ([0, 6, 7, 8, 9], [2, 3]),
        # test {4,5} spans [40, 65]: 3 and 6 purged, 7 (start 70) embargoed
        ([0, 1, 2, 8, 9], [4, 5]),
        ([0, 1, 2, 3, 4], [6, 7]),
        # last fold: nothing after it to embargo
        ([0, 1, 2, 3, 4, 5, 6], [8, 9]),
    ]


def test_T036_without_embargo_only_overlaps_are_purged() -> None:
    splits = _as_lists(purged_walk_forward_splits(START, END, n_splits=5, embargo_ns=0))
    assert splits[2] == ([0, 1, 2, 7, 8, 9], [4, 5])


def test_T036_past_only_walk_forward_variant() -> None:
    splits = _as_lists(
        purged_walk_forward_splits(START, END, n_splits=5, embargo_ns=10, train_side="past")
    )
    # train labels must end before t0 - embargo; folds {0,1} (t0=0) and {2,3} (t0=20,
    # every label ends at >= 15 > 10) have no admissible past sample and are skipped
    assert splits == [
        ([0, 1], [4, 5]),  # t0=40: label ends 15, 25 < 30
        ([0, 1, 2, 3], [6, 7]),  # t0=60: label ends <= 45 < 50
        ([0, 1, 2, 3, 4, 5], [8, 9]),  # t0=80: label ends <= 65 < 70
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_T036_invariants_on_random_label_intervals(seed: int) -> None:
    rng = np.random.default_rng(seed)
    start = np.sort(rng.integers(0, 10_000, size=300))
    end = start + rng.integers(0, 400, size=300)
    embargo = 150
    for train, test in purged_walk_forward_splits(start, end, n_splits=6, embargo_ns=embargo):
        assert np.intersect1d(train, test).size == 0
        t0, t1 = start[test].min(), end[test].max()
        overlaps = (start[train] <= t1) & (end[train] >= t0)
        assert not overlaps.any()
        in_embargo = (start[train] > t1) & (start[train] <= t1 + embargo)
        assert not in_embargo.any()


def test_T036_chronological_split_and_windows_purge_labels() -> None:
    train, test = purged_train_test_split(START, END, test_fraction=0.3)
    assert test.tolist() == [7, 8, 9]
    assert train.tolist() == [0, 1, 2, 3, 4, 5]  # 6 ends at 75 >= t0=70: purged
    windows = walk_forward_windows(START, n_folds=4, label_end_ns=END, embargo_ns=0)
    assert [w.test_idx.tolist() for w in windows] == [[2, 3], [4, 5], [6, 7], [8, 9]]
    assert windows[1].train_idx.tolist() == [0, 1, 2]  # 3 ends at 45 >= 40
    rolling = walk_forward_windows(START, n_folds=4, mode="rolling", label_end_ns=END)
    assert rolling[3].train_idx.tolist() == [6]  # only the previous block, 7 purged
