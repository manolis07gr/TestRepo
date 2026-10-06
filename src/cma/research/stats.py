"""Validation statistics (scope s.13 and s.24).

* Seeded bootstrap confidence intervals - iid, circular moving-block and stationary
  (Politis & Romano 1992, 1994) - for mean trade return and aggregate P&L.
* Multiple-testing control: Benjamini-Hochberg FDR and Bonferroni.
* Probability calibration: Brier score, (clipped) log loss, reliability table.
* Trading metrics: Sharpe-like and Sortino ratios (per trade or per period, with an
  annualization factor), max drawdown, profit factor, hit rate, expected shortfall.
* Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2012, 2014).
* Out-of-sample forecast comparison for nested models (Clark & West 2007) with a
  Newey-West long-run variance.

Everything is pure numpy/scipy and deterministic given its ``seed``. Sharpe-like ratios on
overlapping, autocorrelated or few observations are descriptive only - report them with
their sample size and confidence intervals, not as decisive evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntpArray = NDArray[np.intp]
BootstrapMethod = Literal["iid", "block", "stationary"]
NamedStatistic = Literal["mean", "sum", "median"]

EULER_MASCHERONI = 0.5772156649015329
_MAX_CHUNK_CELLS = 2_000_000

__all__ = [
    "BootstrapCI",
    "DeflatedSharpe",
    "ForecastComparison",
    "MultipleTestingResult",
    "ReliabilityTable",
    "aggregate_by_period",
    "benjamini_hochberg",
    "bonferroni",
    "bootstrap_ci",
    "bootstrap_indices",
    "bootstrap_mean_ci",
    "bootstrap_total_ci",
    "brier_score",
    "clark_west_test",
    "deflated_sharpe_ratio",
    "drawdown_path",
    "expected_max_sharpe",
    "expected_shortfall",
    "hit_rate",
    "log_loss",
    "max_drawdown",
    "newey_west_long_run_variance",
    "probabilistic_sharpe_ratio",
    "profit_factor",
    "reliability_table",
    "sharpe_ratio",
    "sortino_ratio",
]


def _norm_cdf(x: float) -> float:
    return float(ndtr(x))


def _norm_ppf(q: float) -> float:
    return float(ndtri(q))


def _vector(values: ArrayLike, name: str, *, allow_empty: bool = False) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{name} must contain only finite values")
    return arr


# ======================================================================================
# Bootstrap
# ======================================================================================


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Percentile bootstrap confidence interval for a statistic."""

    estimate: float
    lower: float
    upper: float
    std_error: float
    alpha: float
    n_boot: int
    method: str
    block_len: int | None
    seed: int


def default_block_length(n: int) -> int:
    """Rule-of-thumb block length ``ceil(n ** (1/3))`` (at least 1)."""
    return max(1, math.ceil(math.pow(n, 1.0 / 3.0)))


def bootstrap_indices(
    n: int,
    n_resamples: int,
    *,
    method: BootstrapMethod = "iid",
    block_len: int | None = None,
    rng: np.random.Generator,
) -> IntpArray:
    """Resampling index matrix of shape ``(n_resamples, n)``.

    ``"block"`` is the circular moving-block bootstrap with fixed block length,
    ``"stationary"`` uses geometric block lengths with mean ``block_len``. Both wrap
    around the end of the sample, so every observation is equally likely to be drawn and
    the bootstrap mean is unbiased for the sample mean.
    """
    if n < 1 or n_resamples < 1:
        raise ValueError("n and n_resamples must be positive")
    if method == "iid":
        return rng.integers(0, n, size=(n_resamples, n)).astype(np.intp)
    length = default_block_length(n) if block_len is None else int(block_len)
    if length < 1:
        raise ValueError("block_len must be >= 1")
    if method == "block":
        n_blocks = math.ceil(n / length)
        starts = rng.integers(0, n, size=(n_resamples, n_blocks))
        offsets = np.arange(length)
        idx = (starts[:, :, None] + offsets[None, None, :]) % n
        return idx.reshape(n_resamples, n_blocks * length)[:, :n].astype(np.intp)
    if method == "stationary":
        new_block = rng.random((n_resamples, n)) < 1.0 / length
        new_block[:, 0] = True
        starts = rng.integers(0, n, size=(n_resamples, n))
        pos = np.arange(n)
        last_start = np.maximum.accumulate(np.where(new_block, pos[None, :], 0), axis=1)
        rows = np.arange(n_resamples)[:, None]
        idx = (starts[rows, last_start] + (pos[None, :] - last_start)) % n
        return idx.astype(np.intp)
    raise ValueError(f"unknown bootstrap method {method!r}")


def _apply_statistic(
    sample: FloatArray, statistic: NamedStatistic | Callable[[FloatArray], float]
) -> FloatArray:
    """Evaluate the statistic on each row of a 2-D sample."""
    if statistic == "mean":
        return np.asarray(sample.mean(axis=1), dtype=np.float64)
    if statistic == "sum":
        return np.asarray(sample.sum(axis=1), dtype=np.float64)
    if statistic == "median":
        return np.asarray(np.median(sample, axis=1), dtype=np.float64)
    if callable(statistic):
        return np.asarray([float(statistic(row)) for row in sample], dtype=np.float64)
    raise ValueError(f"unknown statistic {statistic!r}")


def bootstrap_ci(
    values: ArrayLike,
    statistic: NamedStatistic | Callable[[FloatArray], float] = "mean",
    *,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    method: BootstrapMethod = "iid",
    block_len: int | None = None,
    seed: int = 0,
) -> BootstrapCI:
    """Seeded percentile bootstrap CI for ``statistic`` of ``values``.

    Use ``method="block"`` or ``"stationary"`` for serially dependent observations
    (e.g. per-period P&L or overlapping trades).
    """
    data = _vector(values, "values")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    estimate = float(_apply_statistic(data.reshape(1, -1), statistic)[0])
    rng = np.random.default_rng(seed)
    n = data.size
    chunk = max(1, min(n_boot, _MAX_CHUNK_CELLS // n))
    boots: list[FloatArray] = []
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = bootstrap_indices(n, m, method=method, block_len=block_len, rng=rng)
        boots.append(_apply_statistic(data[idx], statistic))
        done += m
    dist = np.concatenate(boots)
    lower, upper = np.quantile(dist, [alpha / 2.0, 1.0 - alpha / 2.0])
    resolved_block = None if method == "iid" else (block_len or default_block_length(n))
    return BootstrapCI(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        std_error=float(dist.std(ddof=1)) if dist.size > 1 else 0.0,
        alpha=alpha,
        n_boot=n_boot,
        method=method,
        block_len=resolved_block,
        seed=seed,
    )


def bootstrap_mean_ci(
    trade_returns: ArrayLike,
    *,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    method: BootstrapMethod = "iid",
    block_len: int | None = None,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap CI for the mean trade return."""
    return bootstrap_ci(
        trade_returns,
        "mean",
        n_boot=n_boot,
        alpha=alpha,
        method=method,
        block_len=block_len,
        seed=seed,
    )


def bootstrap_total_ci(
    pnl: ArrayLike,
    *,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    method: BootstrapMethod = "block",
    block_len: int | None = None,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap CI for aggregate P&L (sum); block resampling by default."""
    return bootstrap_ci(
        pnl, "sum", n_boot=n_boot, alpha=alpha, method=method, block_len=block_len, seed=seed
    )


# ======================================================================================
# Multiple testing
# ======================================================================================


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    adjusted: FloatArray  # adjusted p-values, in the input order
    reject: BoolArray
    alpha: float
    method: str

    @property
    def n_rejected(self) -> int:
        return int(self.reject.sum())


def _pvalues(pvalues: ArrayLike) -> FloatArray:
    p = _vector(pvalues, "pvalues", allow_empty=True)
    if bool(((p < 0.0) | (p > 1.0)).any()):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def benjamini_hochberg(pvalues: ArrayLike, alpha: float = 0.05) -> MultipleTestingResult:
    """Benjamini-Hochberg step-up FDR control (independent / PRDS tests).

    Adjusted p-values ``p_(i) * m / i`` are made monotone from the largest p down and
    capped at 1; ``reject = adjusted <= alpha``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    p = _pvalues(pvalues)
    m = p.size
    if m == 0:
        return MultipleTestingResult(np.empty(0), np.empty(0, dtype=np.bool_), alpha, "BH")
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1, dtype=np.float64)
    monotone = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(monotone, 1.0)
    adjusted = np.maximum(adjusted, p)  # guard against floating-point round-down
    return MultipleTestingResult(adjusted, np.asarray(adjusted <= alpha), alpha, "BH")


def bonferroni(pvalues: ArrayLike, alpha: float = 0.05) -> MultipleTestingResult:
    """Bonferroni family-wise error control: ``min(1, m * p)``."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    p = _pvalues(pvalues)
    adjusted = np.maximum(np.minimum(p * p.size, 1.0), p)
    return MultipleTestingResult(adjusted, np.asarray(adjusted <= alpha), alpha, "bonferroni")


# ======================================================================================
# Calibration
# ======================================================================================


def _probs_outcomes(probs: ArrayLike, outcomes: ArrayLike) -> tuple[FloatArray, FloatArray]:
    p = _vector(probs, "probs")
    o = _vector(outcomes, "outcomes")
    if p.shape != o.shape:
        raise ValueError("probs and outcomes must have the same length")
    if bool(((p < 0.0) | (p > 1.0)).any()):
        raise ValueError("probabilities must lie in [0, 1]")
    if not bool(np.isin(o, (0.0, 1.0)).all()):
        raise ValueError("outcomes must be 0 or 1")
    return p, o


def brier_score(probs: ArrayLike, outcomes: ArrayLike) -> float:
    """Mean squared error of probability forecasts; in [0, 1]."""
    p, o = _probs_outcomes(probs, outcomes)
    return float(np.mean((p - o) ** 2))


def log_loss(probs: ArrayLike, outcomes: ArrayLike, *, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood with probabilities clipped to ``[eps, 1 - eps]``."""
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")
    p, o = _probs_outcomes(probs, outcomes)
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(o * np.log(q) + (1.0 - o) * np.log1p(-q)))


@dataclass(frozen=True, slots=True)
class ReliabilityTable:
    bin_edges: FloatArray  # n_bins + 1 edges
    counts: NDArray[np.int64]
    mean_predicted: FloatArray  # NaN for empty bins
    observed_frequency: FloatArray  # NaN for empty bins

    @property
    def expected_calibration_error(self) -> float:
        total = int(self.counts.sum())
        if total == 0:
            return float("nan")
        filled = self.counts > 0
        gaps = np.abs(self.mean_predicted[filled] - self.observed_frequency[filled])
        return float((self.counts[filled] * gaps).sum() / total)


def reliability_table(
    probs: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> ReliabilityTable:
    """Calibration table: per probability bin, mean forecast vs observed frequency."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, o = _probs_outcomes(probs, outcomes)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.maximum.accumulate(edges)
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins).astype(np.int64)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    sum_o = np.bincount(bins, weights=o, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(counts > 0, sum_p / np.maximum(counts, 1), np.nan)
        freq = np.where(counts > 0, sum_o / np.maximum(counts, 1), np.nan)
    return ReliabilityTable(
        bin_edges=np.asarray(edges, dtype=np.float64),
        counts=counts,
        mean_predicted=np.asarray(mean_p, dtype=np.float64),
        observed_frequency=np.asarray(freq, dtype=np.float64),
    )


# ======================================================================================
# Trading metrics
# ======================================================================================


def aggregate_by_period(values: ArrayLike, period_labels: ArrayLike) -> FloatArray:
    """Sum per period (e.g. daily P&L from per-trade P&L), ordered by sorted label."""
    v = _vector(values, "values", allow_empty=True)
    labels = np.asarray(period_labels).reshape(-1)
    if labels.shape != v.shape:
        raise ValueError("period_labels must align with values")
    _, inverse = np.unique(labels, return_inverse=True)
    return np.asarray(np.bincount(inverse.reshape(-1), weights=v), dtype=np.float64)


def sharpe_ratio(returns: ArrayLike, *, annualization: float = 1.0, ddof: int = 1) -> float:
    """Sharpe-like ratio ``mean / std * sqrt(annualization)``.

    Pass per-trade returns with ``annualization=1`` for a per-trade ratio, or per-period
    returns (see :func:`aggregate_by_period`) with periods-per-year to annualize.
    NaN when fewer than two observations or zero dispersion.
    """
    r = _vector(returns, "returns", allow_empty=True)
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if r.size < 2:
        return float("nan")
    sd = float(r.std(ddof=ddof))
    if sd == 0.0 or not math.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * math.sqrt(annualization))


def sortino_ratio(returns: ArrayLike, *, target: float = 0.0, annualization: float = 1.0) -> float:
    """``mean(r - target) / downside deviation * sqrt(annualization)``.

    Downside deviation is ``sqrt(mean(min(r - target, 0)^2))``; +inf if there is no
    downside and the mean excess is positive, NaN if undefined.
    """
    r = _vector(returns, "returns", allow_empty=True)
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if r.size == 0:
        return float("nan")
    excess = r - target
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    mean_excess = float(excess.mean())
    if downside == 0.0:
        return float("inf") if mean_excess > 0 else float("nan")
    return float(mean_excess / downside * math.sqrt(annualization))


def drawdown_path(cumulative_pnl: ArrayLike, *, include_origin: bool = True) -> FloatArray:
    """Drawdown (running peak minus value, >= 0) at each point of a cumulative P&L path.

    With ``include_origin`` the path is assumed to start from 0 (the initial equity).
    """
    path = _vector(cumulative_pnl, "cumulative_pnl", allow_empty=True)
    if include_origin:
        path = np.concatenate(([0.0], path))
    if path.size == 0:
        return np.empty(0, dtype=np.float64)
    dd = np.maximum.accumulate(path) - path
    return np.asarray(dd[1:] if include_origin else dd, dtype=np.float64)


def max_drawdown(cumulative_pnl: ArrayLike, *, include_origin: bool = True) -> float:
    """Maximum peak-to-trough decline of a cumulative P&L path (in P&L units, >= 0).

    For per-trade/per-period P&L increments pass ``np.cumsum(pnl)``.
    """
    dd = drawdown_path(cumulative_pnl, include_origin=include_origin)
    return float(dd.max()) if dd.size else 0.0


def profit_factor(pnl: ArrayLike) -> float:
    """Gross profit / gross loss; +inf without losses (and positive profit), NaN if empty."""
    p = _vector(pnl, "pnl", allow_empty=True)
    gains = float(p[p > 0].sum())
    losses = float(-p[p < 0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def hit_rate(pnl: ArrayLike, *, exclude_zero: bool = False) -> float:
    """Fraction of winning trades (``pnl > 0``); optionally ignoring flat trades."""
    p = _vector(pnl, "pnl", allow_empty=True)
    if exclude_zero:
        p = p[p != 0.0]
    if p.size == 0:
        return float("nan")
    return float(np.mean(p > 0.0))


def expected_shortfall(returns: ArrayLike, level: float = 0.95) -> float:
    """Expected shortfall / CVaR at ``level``: the mean of the worst ``1 - level`` tail.

    Returned in return units (a loss is negative), using the exact discrete
    (Acerbi-Tasche) weighting, so ``expected_shortfall(r) <= mean(r)`` always holds.
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")
    r = np.sort(_vector(returns, "returns"))
    k = (1.0 - level) * r.size
    whole = math.floor(k)
    tail = float(r[:whole].sum())
    if k > whole:
        tail += (k - whole) * float(r[whole])
    return tail / k


# ======================================================================================
# Probabilistic / Deflated Sharpe ratio
# ======================================================================================


def _sharpe_moments(returns: ArrayLike) -> tuple[int, float, float, float]:
    """(n, per-period Sharpe, skewness, Pearson kurtosis) of a return series."""
    r = _vector(returns, "returns")
    n = r.size
    if n < 3:
        raise ValueError("need at least three returns")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        raise ValueError("returns have zero dispersion")
    centred = r - r.mean()
    m2 = float(np.mean(centred**2))
    skew = float(np.mean(centred**3)) / m2**1.5
    kurt = float(np.mean(centred**4)) / m2**2
    return n, float(r.mean()) / sd, skew, kurt


def _psr(sr: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    variance_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    variance_term = max(variance_term, 1e-12)
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(variance_term)
    return _norm_cdf(z)


def probabilistic_sharpe_ratio(returns: ArrayLike, *, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > ``sr_benchmark``) accounting for sample length, skew and kurtosis.

    ``sr_benchmark`` is per period (same frequency as ``returns``, not annualized).
    """
    n, sr, skew, kurt = _sharpe_moments(returns)
    return _psr(sr, sr_benchmark, n, skew, kurt)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe among ``n_trials`` unskilled trials (False Strategy theorem).

    ``sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))``, g = Euler-Mascheroni.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    return math.sqrt(sr_variance) * (
        (1.0 - g) * _norm_ppf(1.0 - 1.0 / n_trials) + g * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


@dataclass(frozen=True, slots=True)
class DeflatedSharpe:
    sharpe: float  # per-period, non-annualized
    sr_benchmark: float  # expected max Sharpe of n_trials null strategies
    deflated_sharpe_ratio: float  # PSR evaluated at sr_benchmark
    n_obs: int
    n_trials: int
    skewness: float
    kurtosis: float
    sr_variance: float


def deflated_sharpe_ratio(
    returns: ArrayLike,
    n_trials: int,
    *,
    trial_sharpes: ArrayLike | None = None,
    sr_variance: float | None = None,
) -> DeflatedSharpe:
    """Deflated Sharpe Ratio: PSR against the expected maximum Sharpe of ``n_trials``.

    The variance of Sharpe ratios across trials comes from ``trial_sharpes`` (per-period
    Sharpes of every configuration tried), else ``sr_variance``, else the asymptotic null
    variance ``1 / (n - 1)`` of a single Sharpe estimate.
    """
    n, sr, skew, kurt = _sharpe_moments(returns)
    if trial_sharpes is not None:
        trials = _vector(trial_sharpes, "trial_sharpes")
        variance = float(trials.var(ddof=1)) if trials.size > 1 else 1.0 / (n - 1)
    elif sr_variance is not None:
        variance = float(sr_variance)
    else:
        variance = 1.0 / (n - 1)
    benchmark = expected_max_sharpe(n_trials, variance)
    return DeflatedSharpe(
        sharpe=sr,
        sr_benchmark=benchmark,
        deflated_sharpe_ratio=_psr(sr, benchmark, n, skew, kurt),
        n_obs=n,
        n_trials=n_trials,
        skewness=skew,
        kurtosis=kurt,
        sr_variance=variance,
    )


# ======================================================================================
# Out-of-sample forecast comparison
# ======================================================================================


def newey_west_long_run_variance(values: ArrayLike, lags: int) -> float:
    """Bartlett-kernel (Newey-West) long-run variance of a series."""
    x = _vector(values, "values")
    if lags < 0:
        raise ValueError("lags must be non-negative")
    centred = x - x.mean()
    n = centred.size
    lrv = float(np.einsum("i,i->", centred, centred)) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * weight * float(np.einsum("i,i->", centred[lag:], centred[:-lag])) / n
    return max(lrv, 0.0)


@dataclass(frozen=True, slots=True)
class ForecastComparison:
    statistic: float  # Clark-West t-statistic (HAC)
    p_value: float  # one-sided: the larger model forecasts better
    mean_adjusted_loss_diff: float
    n: int


def clark_west_test(
    y: ArrayLike,
    pred_restricted: ArrayLike,
    pred_unrestricted: ArrayLike,
    *,
    nw_lags: int = 0,
) -> ForecastComparison:
    """Clark & West (2007) MSPE-adjusted test for nested out-of-sample forecasts.

    ``f_t = (y - p_r)^2 - [(y - p_u)^2 - (p_r - p_u)^2]``; one-sided test of
    ``E[f] > 0`` with a Newey-West standard error (use ``nw_lags >= horizon - 1`` for
    overlapping multi-step targets). Degenerate (zero-variance) cases return p = 1.
    """
    yy = _vector(y, "y")
    pr = _vector(pred_restricted, "pred_restricted")
    pu = _vector(pred_unrestricted, "pred_unrestricted")
    if not (yy.shape == pr.shape == pu.shape):
        raise ValueError("y and predictions must have the same length")
    f = (yy - pr) ** 2 - ((yy - pu) ** 2 - (pr - pu) ** 2)
    n = f.size
    mean_f = float(f.mean())
    if n < 2:
        return ForecastComparison(float("nan"), 1.0, mean_f, n)
    lrv = newey_west_long_run_variance(f, nw_lags)
    if lrv <= 0.0:
        return ForecastComparison(0.0, 1.0, mean_f, n)
    stat = mean_f / math.sqrt(lrv / n)
    return ForecastComparison(stat, 1.0 - _norm_cdf(stat), mean_f, n)
