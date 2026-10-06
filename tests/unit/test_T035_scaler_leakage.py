"""T035 - Scaler leakage: transform statistics are fitted only on the training fold."""

from __future__ import annotations

import numpy as np
import pytest

from cma.backtest.scaling import FoldScaler, LeakageError
from cma.backtest.splits import walk_forward_windows
from cma.models.lead_lag.estimators import (
    PredictiveEvaluation,
    evaluate_predictive,
    predictive_split,
)

pytestmark = pytest.mark.unit


def _data(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(n, 3))
    x[n // 2 :] = rng.normal(10.0, 5.0, size=(n - n // 2, 3))  # regime shift in the test half
    return x, np.arange(n, dtype=np.int64)


def test_T035_statistics_come_only_from_training_rows() -> None:
    x, ids = _data()
    train, test = ids[:200], ids[200:]
    scaler = FoldScaler().fit(x[train], row_ids=train, forbidden_row_ids=test)
    np.testing.assert_allclose(scaler.mean_, x[train].mean(axis=0))
    np.testing.assert_allclose(scaler.std_, x[train].std(axis=0))
    assert not np.allclose(scaler.mean_, x.mean(axis=0))  # full-sample stats differ
    before = scaler.mean_.copy()
    z_test = scaler.transform(x[test])
    np.testing.assert_array_equal(scaler.mean_, before)  # transform never refits
    assert z_test.mean() > 5  # the test regime shift stays visible: nothing was re-centred
    assert scaler.n_fit == 200


def test_T035_fitting_with_leaked_rows_raises() -> None:
    x, ids = _data()
    train, test = ids[:210], ids[200:]  # 10 test rows leak into the fit set
    with pytest.raises(LeakageError, match="10 held-out row"):
        FoldScaler().fit(x[train], row_ids=train, forbidden_row_ids=test)
    with pytest.raises(ValueError, match="row_ids"):
        FoldScaler().fit(x[:10], forbidden_row_ids=test)  # cannot verify without ids
    scaler = FoldScaler().fit(x[:200], row_ids=ids[:200])
    with pytest.raises(LeakageError):
        scaler.check_disjoint(ids[150:250])  # evaluating on rows used for fitting
    scaler.check_disjoint(ids[200:])


def test_T035_walk_forward_scalers_are_fold_local() -> None:
    x, ids = _data()
    for window in walk_forward_windows(ids * 1_000, n_folds=3):
        scaler = FoldScaler().fit(
            x[window.train_idx], row_ids=window.train_idx, forbidden_row_ids=window.test_idx
        )
        np.testing.assert_allclose(scaler.mean_, x[window.train_idx].mean(axis=0))
        assert window.train_idx.max() < window.test_idx.min()


def test_T035_predictive_test_is_invariant_to_test_period_features() -> None:
    """Fitted model (scaler + ridge) depends on training rows only."""
    rng = np.random.default_rng(3)
    n = 2_000
    lead = rng.normal(size=(n, 4))
    ar = rng.normal(size=(n, 2))
    target = 0.5 * lead[:, 0] + rng.normal(size=n)
    ts = np.arange(n, dtype=np.int64) * 100
    train, test = predictive_split(ts, 300, train_fraction=0.7)

    def evaluate(
        lead_x: np.ndarray, ar_x: np.ndarray, test_idx: np.ndarray
    ) -> PredictiveEvaluation:
        return evaluate_predictive(
            lead_x,
            ar_x,
            target,
            row_ts_ns=ts,
            train_idx=train,
            test_idx=test_idx,
            ridge_alpha=1e-3,
            cost_hurdle=0.0,
            nw_lags=3,
        )

    base = evaluate(lead, ar, test)
    lead2, ar2 = lead.copy(), ar.copy()
    lead2[test] = lead2[test] * 50 + 7  # wildly different test-period features
    ar2[test] = -ar2[test] + 3
    moved = evaluate(lead2, ar2, test)
    assert base.lead_coefficients == moved.lead_coefficients
    assert base.oos_r2 != moved.oos_r2  # ...while the evaluation does see the new features
    with pytest.raises(LeakageError):
        evaluate(lead, ar, train[-5:])
