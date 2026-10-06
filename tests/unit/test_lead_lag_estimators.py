"""Lead-lag estimators: CCF sign convention, Hayashi-Yoshida, scan significance, OOS test."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cma.models.lead_lag.estimators import (
    Increments,
    NullMethod,
    contemporaneous_check,
    cross_correlation,
    evaluate_predictive,
    fit_ridge,
    hayashi_yoshida,
    hy_lead_lag,
    increments,
    predictive_split,
    scan_significance,
)
from cma.models.lead_lag.series import EventSeries
from cma.models.lead_lag.synthetic import lead_lag_pair

pytestmark = pytest.mark.unit


def _lagged_noise(n: int, lag: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rx = rng.standard_normal(n)
    ry = np.full(n, np.nan)
    ry[lag:] = rx[:-lag] + 0.5 * rng.standard_normal(n - lag)  # ry_t = rx_{t-lag} + noise
    return rx, ry


def test_ccf_positive_lag_means_x_leads_y() -> None:
    rx, ry = _lagged_noise(5_000, 3)
    ccf = cross_correlation(rx, ry, 10)
    lag, corr = ccf.best()
    assert lag == 3
    assert corr > 0.8
    assert abs(ccf.at(-3)) < 0.1
    assert ccf.n_pairs[ccf.lags == 0][0] == 4_997


def test_ccf_is_nan_safe() -> None:
    rx, ry = _lagged_noise(5_000, 3)
    rng = np.random.default_rng(1)
    rx[rng.random(5_000) < 0.2] = np.nan
    ry[rng.random(5_000) < 0.2] = np.nan
    ccf = cross_correlation(rx, ry, 10)
    assert ccf.best()[0] == 3
    assert np.isfinite(ccf.corr).all()


def _brute_force_hy(x: Increments, y: Increments, lags: np.ndarray) -> np.ndarray:
    out = []
    for theta in lags:
        total = 0.0
        for i, j in itertools.product(range(len(x)), range(len(y))):
            lo = max(x.start_ns[i], y.start_ns[j] - theta)
            hi = min(x.end_ns[i], y.end_ns[j] - theta)
            if lo < hi:
                total += x.dx[i] * y.dx[j]
        out.append(total)
    return np.asarray(out)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hayashi_yoshida_matches_the_quadratic_definition(seed: int) -> None:
    rng = np.random.default_rng(seed)
    xt = np.unique(rng.integers(0, 1_000, 40))
    yt = np.unique(rng.integers(0, 1_000, 25))
    xs = EventSeries.from_arrays(xt, rng.standard_normal(xt.size).cumsum())
    ys = EventSeries.from_arrays(yt, rng.standard_normal(yt.size).cumsum())
    xi, yi = increments(xs, "diff"), increments(ys, "diff")
    lags = np.arange(-200, 201, 7)
    fast = hayashi_yoshida(xi, yi, lags).cov
    np.testing.assert_allclose(fast, _brute_force_hy(xi, yi, lags), atol=1e-12)


def test_hayashi_yoshida_recovers_exact_asynchronous_lag() -> None:
    pair = lead_lag_pair(seed=4, duration_s=600, lag_ms=500, noise_std=0.0)
    lags = np.arange(-20, 21) * 100_000_000
    res = hy_lead_lag(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, lags)
    assert res.best_lag_ns == 500_000_000
    assert res.best_corr > 0.9
    flipped = hy_lead_lag(pair.y_ts_ns, pair.y_values, pair.x_ts_ns, pair.x_values, lags)
    assert flipped.best_lag_ns == -500_000_000  # Y lags X  <=>  negative when roles swap


def test_hayashi_yoshida_weights() -> None:
    pair = lead_lag_pair(seed=5, duration_s=300)
    xi = increments(EventSeries.from_arrays(pair.x_ts_ns, pair.x_values), "diff")
    yi = increments(EventSeries.from_arrays(pair.y_ts_ns, pair.y_values), "diff")
    lags = np.arange(-5, 6) * 500_000_000
    base = hayashi_yoshida(xi, yi, lags)
    ones = hayashi_yoshida(xi, yi, lags, x_weight=np.ones(len(xi)), y_weight=np.ones(len(yi)))
    np.testing.assert_allclose(base.cov, ones.cov)
    zero = hayashi_yoshida(xi, yi, lags, x_weight=np.zeros(len(xi)))
    assert zero.best_lag_ns is None


@pytest.mark.parametrize("method", ["circular_shift", "block_permutation"])
def test_scan_significance_detects_dependence_and_respects_the_null(method: NullMethod) -> None:
    rx, ry = _lagged_noise(4_000, 3)
    lags = np.arange(1, 11)
    strong = scan_significance(
        rx, ry, lags, n_perm=99, block_len=20, rng=np.random.default_rng(0), method=method
    )
    assert strong.p_value == pytest.approx(1 / 100)
    assert strong.statistic > strong.null_q95
    rng = np.random.default_rng(9)
    null = scan_significance(
        rng.standard_normal(4_000),
        rng.standard_normal(4_000),
        lags,
        n_perm=99,
        block_len=20,
        rng=np.random.default_rng(0),
        method=method,
    )
    assert 0.05 < null.p_value <= 1.0
    again = scan_significance(
        rx, ry, lags, n_perm=99, block_len=20, rng=np.random.default_rng(0), method=method
    )
    assert again.p_value == strong.p_value


def test_scan_significance_short_series_is_not_testable() -> None:
    rx, ry = _lagged_noise(50, 3)
    res = scan_significance(
        rx, ry, np.arange(1, 11), n_perm=99, block_len=20, rng=np.random.default_rng(0)
    )
    assert np.isnan(res.p_value)


def test_evaluate_predictive_finds_signal_and_reports_baselines() -> None:
    rng = np.random.default_rng(3)
    n = 4_000
    lead = rng.standard_normal((n, 3))
    ar = rng.standard_normal((n, 2))
    target = 0.8 * lead[:, 1] + rng.standard_normal(n)
    ts = np.arange(n, dtype=np.int64) * 100
    train, test = predictive_split(ts, 100, train_fraction=0.7)
    assert train.max() < test.min()
    ev = evaluate_predictive(
        lead,
        ar,
        target,
        row_ts_ns=ts,
        train_idx=train,
        test_idx=test,
        ridge_alpha=1e-3,
        cost_hurdle=0.5,
        nw_lags=1,
    )
    assert ev.ok
    assert ev.oos_r2 > 0.25
    assert abs(ev.baseline_oos_r2) < 0.02
    assert ev.incremental_oos_r2 > 0.25
    assert ev.cw_p_value < 1e-6
    assert ev.hit_rate > 0.65
    assert 0 < ev.trade_fraction < 1
    assert ev.economic_edge_estimate > 0
    assert int(np.argmax(np.abs(ev.lead_coefficients))) == 1


def test_fit_ridge_without_penalty_is_least_squares() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 4))
    y = z @ np.array([1.0, -2.0, 0.0, 0.5]) + 0.1 * rng.standard_normal(200)
    np.testing.assert_allclose(fit_ridge(z, y, 0.0), np.linalg.lstsq(z, y, rcond=None)[0])
    shrunk = fit_ridge(z, y, 1_000.0)
    assert np.linalg.norm(shrunk) < np.linalg.norm(fit_ridge(z, y, 0.0))


def test_contemporaneous_check_flags_zero_lag_dominance() -> None:
    rng = np.random.default_rng(2)
    common = rng.standard_normal(3_000)
    rx = common + 0.3 * rng.standard_normal(3_000)
    ry = common + 0.3 * rng.standard_normal(3_000)
    check = contemporaneous_check(cross_correlation(rx, ry, 10))
    assert check.zero_dominates
    assert check.corr_at_zero > 0.8
    assert check.ratio > 10
