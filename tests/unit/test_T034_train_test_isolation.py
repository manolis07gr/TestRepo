"""T034 - Train/test isolation: training code cannot access final-test rows/timestamps."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cma.backtest.splits import AccessRecord, LockedDataset, Segment, TimePartition
from cma.config import ResearchConfig
from cma.domain.errors import FinalTestAccessError
from cma.domain.time import NS_PER_MS, NS_PER_S, ManualClock
from cma.models.lead_lag import LeadLagPredictor, LeadLagPredictorConfig
from cma.models.lead_lag.synthetic import lead_lag_pair

pytestmark = pytest.mark.unit


def _frame(n: int = 100, step: int = 10) -> pd.DataFrame:
    ts = np.arange(n, dtype=np.int64) * step
    rng = np.random.default_rng(0)
    order = rng.permutation(n)  # unsorted input must not matter
    return pd.DataFrame({"ts_ns": ts[order], "feature": ts[order] * 0.5, "row": order})


def _locked(ds: LockedDataset) -> bool:
    return ds.is_locked


def _dataset(clock: ManualClock | None = None) -> LockedDataset:
    partition = TimePartition(
        start_ns=0, validation_start_ns=600, final_test_start_ns=800, end_ns=990
    )
    return LockedDataset(_frame(), partition, clock=clock)


def test_T034_training_views_never_contain_final_test_rows() -> None:
    ds = _dataset()
    fts = ds.partition.final_test_start_ns
    for view in (ds.train(), ds.validation(), ds.research()):
        assert (view["ts_ns"] < fts).all()
    assert ds.train()["ts_ns"].max() < 600 <= ds.validation()["ts_ns"].min()
    assert (ds.timestamps < fts).all()
    assert len(ds) == 80  # 20 final-test rows are invisible


def test_T034_rows_between_overlapping_final_test_raises() -> None:
    ds = _dataset()
    assert len(ds.rows_between(100, 790)) == 70
    with pytest.raises(FinalTestAccessError):
        ds.rows_between(700, 800)
    with pytest.raises(FinalTestAccessError):
        ds.rows_between(0, 10**12)


def test_T034_iloc_on_hidden_positions_raises() -> None:
    ds = _dataset()
    assert int(ds.iloc[0]["ts_ns"].iloc[0]) == 0
    assert len(ds.iloc[10:20]) == 10
    for key in (80, 99, -1, slice(75, 85), [3, 85], np.arange(100) >= 50):
        with pytest.raises(FinalTestAccessError):
            ds.iloc[key]


def test_T034_final_test_requires_explicit_recorded_unlock_once() -> None:
    clock = ManualClock(123)
    ds = _dataset(clock=clock)
    log: list[AccessRecord] = []
    with pytest.raises(FinalTestAccessError):
        ds.final_test()
    with pytest.raises(ValueError, match="reason"):
        ds.unlock("   ")
    assert _locked(ds)
    record = ds.unlock("final evaluation of candidate v1", on_access=log.append)
    assert record.event == "unlock"
    assert record.at_ns == 123
    assert not _locked(ds)
    clock.advance(5)
    final = ds.final_test()
    assert (final["ts_ns"] >= 800).all()
    assert len(final) == 20
    assert [r.event for r in log] == ["unlock", "final_test_read"]
    assert log[1].reason == "final evaluation of candidate v1"
    assert log[1].at_ns == 128
    assert ds.access_log == tuple(log)
    with pytest.raises(FinalTestAccessError, match="already read"):
        ds.final_test()  # no iterating against the untouched test period
    with pytest.raises(FinalTestAccessError, match="already unlocked"):
        ds.unlock("again")
    with pytest.raises(FinalTestAccessError):
        ds.rows_between(0, 900)  # guarded views stay guarded after unlock


def test_T034_embargo_and_label_horizons_protect_the_boundary() -> None:
    frame = _frame()
    frame["label_end_ns"] = frame["ts_ns"] + 25  # labels look 25 ns ahead
    partition = TimePartition(
        start_ns=0, validation_start_ns=600, final_test_start_ns=800, end_ns=990, embargo_ns=10
    )
    ds = LockedDataset(frame, partition, label_end_col="label_end_ns")
    research = ds.research()
    assert (research["label_end_ns"] < 790).all()  # label + embargo never reach final test
    assert research["ts_ns"].max() == 760
    assert ds.train()["label_end_ns"].max() < 590
    with pytest.raises(FinalTestAccessError):
        ds.iloc[77]  # ts=770: its label (795) reaches into the embargo buffer
    seg = partition.segment(np.array([585, 595, 600, 785, 795, 800]))
    assert seg.tolist() == [
        Segment.TRAIN,
        Segment.EXCLUDED,
        Segment.VALIDATION,
        Segment.VALIDATION,
        Segment.EXCLUDED,
        Segment.FINAL_TEST,
    ]


def test_T034_partition_from_research_config_and_fractions() -> None:
    ts = np.arange(1_000, dtype=np.int64) * NS_PER_S
    cfg = ResearchConfig(final_test_fraction=0.2, validation_fraction=0.2, embargo_ms=1_000)
    part = TimePartition.from_research_config(ts, cfg)
    assert part.final_test_start_ns == ts[800]
    assert part.validation_start_ns == ts[600]
    assert part.embargo_ns == 1_000 * NS_PER_MS
    assert int(part.final_test_mask(ts).sum()) == 200
    with pytest.raises(FinalTestAccessError):
        part.assert_outside_final_test(ts)
    part.assert_outside_final_test(ts[:799])
    by_time = TimePartition.from_fractions(ts, final_test_fraction=0.5, by="time")
    assert by_time.final_test_start_ns == ts[0] + math.ceil((ts[-1] - ts[0]) * 0.5)


def test_T034_predictor_refuses_to_fit_on_final_test_rows() -> None:
    pair = lead_lag_pair(seed=11, duration_s=600)
    frame = pair.to_frame()
    ds = LockedDataset.from_fractions(frame, validation_fraction=0.2, final_test_fraction=0.2)
    cfg = LeadLagPredictorConfig(
        horizon_ms=1_000, lookback_ms=3_000, x_return_kind="diff", min_train_rows=100
    )
    with pytest.raises(FinalTestAccessError):
        LeadLagPredictor(cfg).fit(
            pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, partition=ds.partition
        )
    model = LeadLagPredictor(cfg).fit_dataset(ds)
    info = model.training_info
    assert info["last_label_end_ns"] < ds.partition.validation_start_ns
    assert info["final_test_start_ns"] == ds.partition.final_test_start_ns
