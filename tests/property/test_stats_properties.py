"""Validation statistics: Hypothesis invariants plus hand-checked reference values."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats as sps

from cma.research.stats import (
    BootstrapMethod,
    aggregate_by_period,
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    bootstrap_indices,
    bootstrap_mean_ci,
    bootstrap_total_ci,
    brier_score,
    clark_west_test,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    expected_shortfall,
    hit_rate,
    log_loss,
    max_drawdown,
    probabilistic_sharpe_ratio,
    profit_factor,
    reliability_table,
    sharpe_ratio,
    sortino_ratio,
)

FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
RETURNS = arrays(np.float64, st.integers(2, 120), elements=FINITE)
PVALUES = arrays(np.float64, st.integers(1, 60), elements=st.floats(0.0, 1.0))
PROBS = st.floats(0.0, 1.0, allow_nan=False)
METHODS = st.sampled_from(["iid", "block", "stationary"])
PROPERTY = settings(max_examples=60, deadline=None)


def _tol(values: np.ndarray) -> float:
    return 1e-9 * (1.0 + float(np.max(np.abs(values))))


# ======================================================================================
# Hypothesis invariants
# ======================================================================================


@pytest.mark.property
@PROPERTY
@given(values=RETURNS, method=METHODS, seed=st.integers(0, 2**32 - 1))
def test_bootstrap_ci_brackets_the_sample_statistic(
    values: np.ndarray, method: BootstrapMethod, seed: int
) -> None:
    tol = _tol(values) * values.size
    for ci in (
        bootstrap_ci(values, "mean", n_boot=200, method=method, seed=seed),
        bootstrap_ci(values, "sum", n_boot=200, method=method, seed=seed),
    ):
        assert ci.lower - tol <= ci.estimate <= ci.upper + tol
        assert ci.lower <= ci.upper


@pytest.mark.property
@PROPERTY
@given(p=PVALUES, alpha=st.floats(0.001, 0.5))
def test_bh_adjusted_pvalues_are_monotone_and_not_below_raw(p: np.ndarray, alpha: float) -> None:
    res = benjamini_hochberg(p, alpha)
    assert (res.adjusted >= p).all()
    assert (res.adjusted <= 1.0).all()
    order = np.argsort(p, kind="stable")
    assert (np.diff(res.adjusted[order]) >= 0).all()
    assert (res.adjusted <= bonferroni(p, alpha).adjusted + 1e-15).all()
    if res.reject.any():  # step-up: everything at least as small as a rejection is rejected
        assert res.reject[p <= p[res.reject].max()].all()


@pytest.mark.property
@PROPERTY
@given(pnl=RETURNS)
def test_max_drawdown_is_non_negative_and_bounded(pnl: np.ndarray) -> None:
    path = np.cumsum(pnl)
    mdd = max_drawdown(path)
    with_origin = np.concatenate(([0.0], path))
    assert mdd >= 0.0
    assert mdd <= with_origin.max() - with_origin.min() + _tol(with_origin)
    assert max_drawdown(np.sort(path), include_origin=False) == 0.0  # never draws down


@pytest.mark.property
@PROPERTY
@given(data=st.lists(st.tuples(PROBS, st.sampled_from([0.0, 1.0])), min_size=1, max_size=80))
def test_brier_in_unit_interval_and_log_loss_non_negative(
    data: list[tuple[float, float]],
) -> None:
    probs = np.array([d[0] for d in data])
    outcomes = np.array([d[1] for d in data])
    assert 0.0 <= brier_score(probs, outcomes) <= 1.0
    assert log_loss(probs, outcomes) >= 0.0


@pytest.mark.property
@PROPERTY
@given(r=RETURNS, level=st.floats(0.5, 0.999))
def test_expected_shortfall_not_above_mean_and_monotone_in_level(
    r: np.ndarray, level: float
) -> None:
    es = expected_shortfall(r, level)
    assert es <= float(np.mean(r)) + _tol(r)
    assert es >= float(np.min(r)) - _tol(r)
    assert expected_shortfall(r, min(0.999, level + 0.0005)) <= es + _tol(r)


@pytest.mark.property
@PROPERTY
@given(r=arrays(np.float64, st.integers(5, 120), elements=st.floats(-10, 10)))
def test_psr_is_a_probability_and_deflation_never_helps(r: np.ndarray) -> None:
    assume(np.std(r) > 1e-6)
    psr = probabilistic_sharpe_ratio(r)
    assert 0.0 <= psr <= 1.0
    dsr = deflated_sharpe_ratio(r, n_trials=10)
    assert dsr.sr_benchmark >= 0.0
    assert dsr.deflated_sharpe_ratio <= psr + 1e-12


@pytest.mark.property
@PROPERTY
@given(pnl=RETURNS)
def test_trade_metric_ranges(pnl: np.ndarray) -> None:
    hr = hit_rate(pnl)
    assert 0.0 <= hr <= 1.0
    pf = profit_factor(pnl)
    assert math.isnan(pf) or pf >= 0.0


# ======================================================================================
# Hand-checked reference values
# ======================================================================================


@pytest.mark.unit
def test_bh_and_bonferroni_reference_values() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    bh = benjamini_hochberg(p, alpha=0.05)
    np.testing.assert_allclose(bh.adjusted, [0.02, 0.04, 0.04, 0.02])
    assert bh.reject.tolist() == [True, True, True, True]
    bonf = bonferroni(p, alpha=0.05)
    np.testing.assert_allclose(bonf.adjusted, [0.04, 0.16, 0.12, 0.02])
    assert bonf.reject.tolist() == [True, False, False, True]
    assert benjamini_hochberg([], 0.05).adjusted.size == 0


@pytest.mark.unit
def test_calibration_reference_values() -> None:
    assert brier_score([0.9, 0.2], [1, 0]) == pytest.approx(0.025)
    assert log_loss([0.9, 0.2], [1, 0]) == pytest.approx(-(math.log(0.9) + math.log(0.8)) / 2)
    assert math.isfinite(log_loss([0.0, 1.0], [1, 0]))  # clipped, not infinite
    table = reliability_table([0.05, 0.15, 0.15, 0.95], [0, 0, 1, 1], n_bins=10)
    assert table.counts.tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    assert table.observed_frequency[1] == pytest.approx(0.5)
    assert np.isnan(table.mean_predicted[2])
    assert table.expected_calibration_error == pytest.approx(0.2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.2], [1])


@pytest.mark.unit
def test_trading_metric_reference_values() -> None:
    assert max_drawdown([1, 3, 2, 5, 1, 4]) == pytest.approx(4.0)
    assert max_drawdown([-2.0, -1.0]) == pytest.approx(2.0)  # from the zero origin
    assert profit_factor([1.0, -2.0, 3.0]) == pytest.approx(2.0)
    assert hit_rate([1.0, -2.0, 3.0, 0.0]) == pytest.approx(0.5)
    assert hit_rate([1.0, -2.0, 3.0, 0.0], exclude_zero=True) == pytest.approx(2 / 3)
    assert sharpe_ratio([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert sharpe_ratio([1.0, 2.0, 3.0], annualization=4) == pytest.approx(4.0)
    assert sortino_ratio([2.0, -1.0, 3.0]) == pytest.approx((4 / 3) / math.sqrt(1 / 3))
    assert expected_shortfall(np.arange(100) - 50.0, 0.95) == pytest.approx(-48.0)
    np.testing.assert_allclose(aggregate_by_period([1, 2, 3, 4], ["b", "a", "b", "a"]), [6, 4])


@pytest.mark.unit
def test_psr_and_dsr_reference_values() -> None:
    r = np.random.default_rng(0).normal(0.05, 1.0, 500)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = sps.skew(r), sps.kurtosis(r, fisher=False)
    expected = sps.norm.cdf(sr * math.sqrt(499) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    assert probabilistic_sharpe_ratio(r) == pytest.approx(expected)
    g = 0.5772156649015329
    emax = (1 - g) * sps.norm.ppf(1 - 1 / 100) + g * sps.norm.ppf(1 - 1 / (100 * math.e))
    assert expected_max_sharpe(100, 1.0) == pytest.approx(emax)
    assert expected_max_sharpe(1, 1.0) == 0.0
    trials = np.random.default_rng(1).normal(0, 0.05, 50)
    dsr = deflated_sharpe_ratio(r, n_trials=50, trial_sharpes=trials)
    assert dsr.sr_variance == pytest.approx(trials.var(ddof=1))
    assert dsr.deflated_sharpe_ratio < probabilistic_sharpe_ratio(r)


@pytest.mark.unit
def test_bootstrap_is_seeded_and_block_schemes_wrap() -> None:
    x = np.random.default_rng(3).normal(0.1, 1.0, 300)
    a = bootstrap_mean_ci(x, seed=11, n_boot=500)
    assert a == bootstrap_mean_ci(x, seed=11, n_boot=500)
    assert a != bootstrap_mean_ci(x, seed=12, n_boot=500)
    assert a.lower < x.mean() < a.upper
    total = bootstrap_total_ci(x, seed=1, n_boot=300, block_len=10)
    assert total.method == "block"
    assert total.lower < x.sum() < total.upper
    idx = bootstrap_indices(10, 4, method="block", block_len=4, rng=np.random.default_rng(0))
    assert idx.shape == (4, 10)
    assert ((np.diff(idx[:, :4], axis=1) % 10) == 1).all()  # contiguous circular blocks
    stat = bootstrap_indices(
        1_000, 50, method="stationary", block_len=20, rng=np.random.default_rng(0)
    )
    breaks = np.mean((np.diff(stat, axis=1) % 1_000) != 1)
    assert 0.03 < breaks < 0.07  # mean block length ~ 20


@pytest.mark.unit
def test_clark_west_detects_a_better_nested_forecast() -> None:
    rng = np.random.default_rng(5)
    signal = rng.standard_normal(2_000)
    y = 0.5 * signal + rng.standard_normal(2_000)
    restricted = np.zeros(2_000)
    better = clark_west_test(y, restricted, 0.5 * signal, nw_lags=2)
    assert better.p_value < 1e-6
    same = clark_west_test(y, restricted, restricted)
    assert same.p_value == 1.0
