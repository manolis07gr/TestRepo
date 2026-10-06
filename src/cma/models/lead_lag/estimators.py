"""Lead-lag estimators.

Sign convention everywhere: a **positive lag k means X leads Y**, i.e. the statistic
compares X at ``t`` with Y at ``t + k``.

(a) :func:`cross_correlation` - NaN-safe grid cross-correlation ``corr(rx_t, ry_{t+k})``.
(b) :func:`hayashi_yoshida` - lagged Hayashi-Yoshida covariance/correlation for
    asynchronous observations (Hoffmann, Rosenbaum & Yoshida 2013), computed with
    ``searchsorted`` + cumulative sums in O((n + m) log(n + m)) per lag.
(c) :func:`scan_significance` - family-wise p-value of ``max |corr|`` over the scanned
    lags against a circular-shift (or block-permutation) null of X relative to Y.
(d) :func:`evaluate_predictive` - out-of-sample test: lagged X features predict the
    future change of Y beyond the zero and own-history (AR) baselines.
(e) :func:`contemporaneous_check` - zero-lag correlation versus the best positive lag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cma.backtest.scaling import FoldScaler, LeakageError
from cma.backtest.splits import Split, purged_train_test_split
from cma.models.lead_lag.series import EventSeries, ReturnKind, change
from cma.research.stats import clark_west_test

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
NullMethod = Literal["circular_shift", "block_permutation"]

__all__ = [
    "ContemporaneousCheck",
    "CrossCorrelation",
    "HYResult",
    "Increments",
    "NullMethod",
    "PredictiveEvaluation",
    "ScanSignificance",
    "contemporaneous_check",
    "cross_correlation",
    "evaluate_predictive",
    "fit_ridge",
    "hayashi_yoshida",
    "hy_lead_lag",
    "increments",
    "predictive_split",
    "scan_significance",
]

_NAN = float("nan")


def _vdot(a: FloatArray, b: FloatArray) -> float:
    """Inner product via einsum (avoids multi-threaded BLAS ``ddot`` overhead on long
    vectors, which is pathological on some container CPUs)."""
    return float(np.einsum("i,i->", a, b))


def _matvec(m: FloatArray, v: FloatArray) -> FloatArray:
    return np.asarray(np.einsum("ij,j->i", m, v), dtype=np.float64)


# ======================================================================================
# (a) Grid cross-correlation
# ======================================================================================


@dataclass(frozen=True, slots=True)
class CrossCorrelation:
    lags: IntArray  # grid steps; positive => X leads Y
    corr: FloatArray
    n_pairs: IntArray

    def at(self, lag: int) -> float:
        hit = np.flatnonzero(self.lags == lag)
        return float(self.corr[hit[0]]) if hit.size else _NAN

    def best(self, *, positive_only: bool = False) -> tuple[int | None, float]:
        """Lag maximising ``|corr|`` (ties -> smallest |lag|) and the corr there."""
        mask = np.isfinite(self.corr)
        if positive_only:
            mask &= self.lags > 0
        if not bool(mask.any()):
            return None, _NAN
        cand = np.flatnonzero(mask)
        score = np.abs(self.corr[cand])
        top = cand[score == score.max()]
        pick = int(top[np.argmin(np.abs(self.lags[top]))])
        return int(self.lags[pick]), float(self.corr[pick])


def cross_correlation(
    rx: ArrayLike,
    ry: ArrayLike,
    max_lag: int,
    *,
    min_pairs: int = 30,
    x_mask: ArrayLike | None = None,
) -> CrossCorrelation:
    """``corr(rx_t, ry_{t+k})`` for ``k = -max_lag..max_lag`` over finite pairs only.

    ``x_mask`` restricts the pairs to anchor times ``t`` where the mask is true (used
    for regime strata). Lags with fewer than ``min_pairs`` pairs or zero variance give NaN.
    """
    x = np.asarray(rx, dtype=np.float64)
    y = np.asarray(ry, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("rx and ry must be 1-D arrays of equal length")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    fx = np.isfinite(x)
    if x_mask is not None:
        fx &= np.asarray(x_mask, dtype=np.bool_)
    fy = np.isfinite(y)
    n = x.size
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    corr = np.full(lags.size, np.nan, dtype=np.float64)
    pairs = np.zeros(lags.size, dtype=np.int64)
    # Zero-filled, globally centred copies: per-lag pair sums become plain inner products
    # (no copies), and centring keeps the moment formulas numerically stable.
    x0 = np.where(fx, x - (x[fx].mean() if fx.any() else 0.0), 0.0)
    y0 = np.where(fy, y - (y[fy].mean() if fy.any() else 0.0), 0.0)
    mx, my = fx.astype(np.float64), fy.astype(np.float64)
    x2, y2 = x0 * x0, y0 * y0
    for i, k in enumerate(lags.tolist()):
        if abs(k) >= n:
            continue
        sx_, sy_ = (slice(0, n - k), slice(k, n)) if k >= 0 else (slice(-k, n), slice(0, n + k))
        cnt = _vdot(mx[sx_], my[sy_])
        pairs[i] = round(cnt)
        if cnt < max(min_pairs, 2):
            continue
        sum_x, sum_y = _vdot(x0[sx_], my[sy_]), _vdot(mx[sx_], y0[sy_])
        var_x = _vdot(x2[sx_], my[sy_]) - sum_x * sum_x / cnt
        var_y = _vdot(mx[sx_], y2[sy_]) - sum_y * sum_y / cnt
        cov = _vdot(x0[sx_], y0[sy_]) - sum_x * sum_y / cnt
        if var_x > 0.0 and var_y > 0.0:
            corr[i] = cov / math.sqrt(var_x * var_y)
    return CrossCorrelation(lags, corr, pairs)


# ======================================================================================
# (b) Hayashi-Yoshida lagged covariance
# ======================================================================================


@dataclass(frozen=True, slots=True)
class Increments:
    """Changes over consecutive observation intervals ``(start_ns, end_ns]``."""

    start_ns: IntArray
    end_ns: IntArray
    dx: FloatArray

    def __len__(self) -> int:
        return int(self.dx.size)


def increments(series: EventSeries, kind: ReturnKind) -> Increments:
    """Interval changes of a cleaned event series (non-finite changes set to 0)."""
    if len(series) < 2:
        empty_i = np.empty(0, dtype=np.int64)
        return Increments(empty_i, empty_i.copy(), np.empty(0, dtype=np.float64))
    dx = change(series.values[1:], series.values[:-1], kind)
    dx = np.where(np.isfinite(dx), dx, 0.0)
    return Increments(series.ts_ns[:-1].copy(), series.ts_ns[1:].copy(), dx)


@dataclass(frozen=True, slots=True)
class HYResult:
    lags_ns: IntArray  # positive => X leads Y
    cov: FloatArray
    corr: FloatArray
    best_lag_ns: int | None
    best_corr: float


def hayashi_yoshida(
    x: Increments,
    y: Increments,
    lags_ns: ArrayLike,
    *,
    x_weight: ArrayLike | None = None,
    y_weight: ArrayLike | None = None,
) -> HYResult:
    """Lagged Hayashi-Yoshida estimator (Hoffmann, Rosenbaum & Yoshida 2013).

    ``U(theta) = sum_{i,j} dX(I_i) dY(J_j) 1{I_i intersects J_j - theta}`` with half-open
    intervals ``(a, b]``. If ``Y(t) = X(t - theta0)`` the overlap structure is exact at
    ``theta = theta0``, so ``theta* = argmax |U|`` estimates the lead of X over Y.

    For each Y interval the overlapping X intervals form a contiguous index range found
    with two ``searchsorted`` calls; their summed changes come from a cumulative sum, so
    each lag costs O((n + m) log n) instead of O(n m). Optional 0/1 weights restrict the
    estimator to sub-samples (strata).
    """
    lags = np.asarray(lags_ns, dtype=np.int64).reshape(-1)
    wx = np.ones(len(x)) if x_weight is None else np.asarray(x_weight, dtype=np.float64)
    wy = np.ones(len(y)) if y_weight is None else np.asarray(y_weight, dtype=np.float64)
    if wx.shape != x.dx.shape or wy.shape != y.dx.shape:
        raise ValueError("HY weights must have one entry per increment")
    dx = x.dx * wx
    dy = y.dx * wy
    cov = np.full(lags.size, np.nan, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return HYResult(lags, cov, cov.copy(), None, _NAN)
    # X intervals are (t_k, t_{k+1}] with t = observation times of X.
    t = np.concatenate((x.start_ns[:1], x.end_ns))
    contiguous = bool(np.array_equal(x.start_ns[1:], x.end_ns[:-1]))
    if not contiguous:
        raise ValueError("X increments must come from consecutive observations")
    csum = np.concatenate(([0.0], np.cumsum(dx)))
    n_dx = dx.size
    for i, theta in enumerate(lags.tolist()):
        a = y.start_ns - theta
        b = y.end_ns - theta
        # overlap of (t_k, t_{k+1}] and (a, b]  <=>  t_k < b  and  a < t_{k+1}
        hi = np.minimum(np.searchsorted(t, b, side="left") - 1, n_dx - 1)
        lo = np.maximum(np.searchsorted(t, a, side="right") - 1, 0)
        valid = hi >= lo
        sums = np.where(valid, csum[np.maximum(hi, 0) + 1] - csum[lo], 0.0)
        cov[i] = _vdot(dy, sums)
    norm = math.sqrt(_vdot(dx, dx) * _vdot(dy, dy))
    corr = cov / norm if norm > 0 else np.full_like(cov, np.nan)
    finite = np.isfinite(cov)
    if not bool(finite.any()) or norm == 0.0:
        return HYResult(lags, cov, corr, None, _NAN)
    score = np.where(finite, np.abs(cov), -np.inf)
    top = np.flatnonzero(score == score.max())
    pick = int(top[np.argmin(np.abs(lags[top]))])
    return HYResult(lags, cov, corr, int(lags[pick]), float(corr[pick]))


def hy_lead_lag(
    x_ts_ns: ArrayLike,
    x_values: ArrayLike,
    y_ts_ns: ArrayLike,
    y_values: ArrayLike,
    lags_ns: ArrayLike,
    *,
    x_kind: ReturnKind = "diff",
    y_kind: ReturnKind = "diff",
) -> HYResult:
    """Convenience wrapper: lagged HY on raw asynchronous observations."""
    xs = EventSeries.from_arrays(x_ts_ns, x_values, name="x")
    ys = EventSeries.from_arrays(y_ts_ns, y_values, name="y")
    return hayashi_yoshida(increments(xs, x_kind), increments(ys, y_kind), lags_ns)


# ======================================================================================
# (c) Family-wise significance of the scan
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ScanSignificance:
    statistic: float  # max |corr| over the scanned lags (circular estimator)
    p_value: float  # (1 + #{null >= statistic}) / (1 + n_perm)
    null_q95: float
    n_perm: int
    method: str
    scanned_lags: IntArray = field(repr=False)


def _centred_zero_filled(values: FloatArray, mask: BoolArray) -> FloatArray:
    out = np.zeros(values.size, dtype=np.float64)
    if bool(mask.any()):
        out[mask] = values[mask] - values[mask].mean()
    return out


def scan_significance(
    rx: ArrayLike,
    ry: ArrayLike,
    lags: ArrayLike,
    *,
    n_perm: int,
    block_len: int,
    rng: np.random.Generator,
    method: NullMethod = "circular_shift",
    x_mask: ArrayLike | None = None,
) -> ScanSignificance:
    """Family-wise p-value for ``S = max_{k in lags} |corr(rx_t, ry_{t+k})|``.

    The null keeps each series' own dependence and destroys the cross dependence:

    * ``"circular_shift"``: X is rotated by a random offset ``s`` with
      ``2 max|lag| + block_len <= s <= n - 2 max|lag| - block_len``, so no rotated scan
      lag lands where genuine cross-dependence lives (``|k| <= max|lag|``); all offsets
      come from a single FFT circular cross-covariance (``c_s(k) = c(k + s)``).
    * ``"block_permutation"``: X is cut into blocks of ``block_len`` steps that are
      randomly permuted; each resample is scored at the scanned lags only.

    The observed statistic uses the same (circular, zero-filled) estimator as the null,
    and the maximum over all scanned lags makes the p-value family-wise for the scan.
    """
    x = np.asarray(rx, dtype=np.float64)
    y = np.asarray(ry, dtype=np.float64)
    scan = np.asarray(lags, dtype=np.int64).reshape(-1)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("rx and ry must be 1-D arrays of equal length")
    if n_perm < 1 or block_len < 1 or scan.size == 0:
        raise ValueError("n_perm, block_len and the lag set must be non-empty/positive")
    mx = np.isfinite(x)
    if x_mask is not None:
        mx &= np.asarray(x_mask, dtype=np.bool_)
    my = np.isfinite(y)
    a = _centred_zero_filled(x, mx)
    b = _centred_zero_filled(y, my)
    n = a.size
    norm = math.sqrt(_vdot(a, a) * _vdot(b, b))
    max_abs_lag = int(np.abs(scan).max())
    min_shift = 2 * max_abs_lag + block_len
    if norm == 0.0 or n - 2 * min_shift < 1:
        return ScanSignificance(_NAN, _NAN, _NAN, n_perm, method, scan)
    fb = np.fft.rfft(b)
    circ = np.fft.irfft(np.conj(np.fft.rfft(a)) * fb, n=n) / norm
    statistic = float(np.abs(circ[scan % n]).max())
    if method == "circular_shift":
        shifts = rng.integers(min_shift, n - min_shift + 1, size=n_perm)
        idx = (scan[None, :] + shifts[:, None]) % n
        null = np.abs(circ[idx]).max(axis=1)
    elif method == "block_permutation":
        n_blocks = n // block_len
        if n_blocks < 2:
            return ScanSignificance(statistic, _NAN, _NAN, n_perm, method, scan)
        # Y rotated by every scanned lag: c(k) = <a, roll(b, -k)>, evaluated for many
        # permuted copies of X at once with one matrix product per chunk.
        shifted_b = np.stack([np.roll(b, -int(k)) for k in scan.tolist()])
        block_ids = np.arange(n_blocks * block_len).reshape(n_blocks, block_len)
        tail_ids = np.arange(n_blocks * block_len, n)
        null = np.empty(n_perm, dtype=np.float64)
        chunk = max(1, min(n_perm, 2_000_000 // max(n, 1)))
        for lo in range(0, n_perm, chunk):
            hi = min(n_perm, lo + chunk)
            cols = [
                a[np.concatenate((block_ids[rng.permutation(n_blocks)].reshape(-1), tail_ids))]
                for _ in range(hi - lo)
            ]
            # einsum avoids BLAS-threading pathologies for this long, thin product
            cov = np.einsum("ln,pn->lp", shifted_b, np.stack(cols)) / norm
            null[lo:hi] = np.abs(cov).max(axis=0)
    else:
        raise ValueError(f"unknown null method {method!r}")
    exceed = int((null >= statistic - 1e-15).sum())
    return ScanSignificance(
        statistic=statistic,
        p_value=(1.0 + exceed) / (1.0 + n_perm),
        null_q95=float(np.quantile(null, 0.95)),
        n_perm=n_perm,
        method=method,
        scanned_lags=scan,
    )


# ======================================================================================
# (d) Out-of-sample predictive test
# ======================================================================================


def fit_ridge(z: FloatArray, y: FloatArray, lam: float) -> FloatArray:
    """Ridge coefficients (no intercept) for standardized features ``z``."""
    gram = z.T @ z
    if lam > 0:
        gram = gram + lam * np.eye(gram.shape[0])
        return np.asarray(np.linalg.solve(gram, np.einsum("ij,i->j", z, y)), dtype=np.float64)
    sol, *_ = np.linalg.lstsq(z, y, rcond=None)
    return np.asarray(sol, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PredictiveEvaluation:
    """Out-of-sample metrics; R^2 values are relative to the zero forecast."""

    n_train: int
    n_test: int
    oos_r2: float  # lead model (X lags + Y own lags) vs zero prediction
    baseline_oos_r2: float  # Y own-lag (AR) model vs zero prediction
    x_only_oos_r2: float  # X lags only vs zero prediction
    incremental_oos_r2: float  # lead model vs the better of zero / AR baselines
    cw_statistic: float  # Clark-West, lead model vs AR baseline (nested)
    cw_p_value: float
    hit_rate: float  # sign agreement on test rows where both are non-zero
    trade_fraction: float  # share of test rows with |prediction| > cost hurdle
    economic_edge_estimate: float  # mean sign(pred) * dY - hurdle over those rows
    baseline_hit_rate: float = _NAN  # same metrics for the own-history (AR) baseline
    baseline_trade_fraction: float = _NAN
    baseline_economic_edge: float = _NAN
    lead_coefficients: tuple[float, ...] = ()
    note: str = ""

    @property
    def economic_value_per_row(self) -> float:
        """Average net edge per test row of trading the lead model (0 when no trade)."""
        return _value_per_row(self.trade_fraction, self.economic_edge_estimate)

    @property
    def baseline_economic_value_per_row(self) -> float:
        return _value_per_row(self.baseline_trade_fraction, self.baseline_economic_edge)

    @property
    def ok(self) -> bool:
        return not self.note


def _value_per_row(trade_fraction: float, edge: float) -> float:
    if not math.isfinite(trade_fraction):
        return _NAN
    return 0.0 if trade_fraction == 0.0 else trade_fraction * edge


def _directional(pred: FloatArray, y: FloatArray, cost_hurdle: float) -> tuple[float, float, float]:
    """(hit rate, trade fraction, mean edge after hurdle) of trading ``sign(pred)``."""
    nonzero = (y != 0.0) & (pred != 0.0)
    hit = float(np.mean(np.sign(pred[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else _NAN
    trade = np.abs(pred) > cost_hurdle
    edge = float(np.mean(np.sign(pred[trade]) * y[trade] - cost_hurdle)) if trade.any() else _NAN
    return hit, float(trade.mean()), edge


def _failed_evaluation(n_train: int, n_test: int, note: str) -> PredictiveEvaluation:
    return PredictiveEvaluation(
        n_train=n_train,
        n_test=n_test,
        oos_r2=_NAN,
        baseline_oos_r2=_NAN,
        x_only_oos_r2=_NAN,
        incremental_oos_r2=_NAN,
        cw_statistic=_NAN,
        cw_p_value=_NAN,
        hit_rate=_NAN,
        trade_fraction=_NAN,
        economic_edge_estimate=_NAN,
        note=note,
    )


def predictive_split(
    row_ts_ns: ArrayLike, horizon_ns: int, *, train_fraction: float, embargo_ns: int = 0
) -> Split:
    """Chronological train/test split whose training labels ``(t, t + h]`` never overlap
    the test period (purged), plus an optional embargo."""
    ts = np.asarray(row_ts_ns, dtype=np.int64)
    return purged_train_test_split(
        ts, ts + int(horizon_ns), test_fraction=1.0 - train_fraction, embargo_ns=embargo_ns
    )


def evaluate_predictive(
    lead: FloatArray,
    ar: FloatArray | None,
    target: FloatArray,
    *,
    row_ts_ns: IntArray,
    train_idx: IntArray,
    test_idx: IntArray,
    ridge_alpha: float,
    cost_hurdle: float,
    nw_lags: int,
    min_rows: int = 50,
) -> PredictiveEvaluation:
    """Fit on ``train_idx`` only and evaluate on the later ``test_idx`` rows.

    The scaler is a :class:`FoldScaler` fitted on training rows (test rows forbidden);
    ridge (penalty ``ridge_alpha * n_train`` on standardized features, no intercept)
    is fitted on training rows. Models: lead = [X lags, Y lags], baseline AR = Y lags,
    X-only = X lags; plus the zero forecast.
    """
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size:
        raise LeakageError(f"{overlap.size} row(s) are in both the train and the test fold")
    n_tr, n_te = int(train_idx.size), int(test_idx.size)
    p_x = lead.shape[1]
    full = lead if ar is None else np.hstack((lead, ar))
    p = full.shape[1]
    if n_tr < max(min_rows, 2 * p) or n_te < min_rows:
        return _failed_evaluation(n_tr, n_te, f"too few rows (train={n_tr}, test={n_te})")
    scaler = FoldScaler().fit(
        full[train_idx], row_ids=row_ts_ns[train_idx], forbidden_row_ids=row_ts_ns[test_idx]
    )
    z = scaler.transform(full)
    z_tr, z_te = z[train_idx], z[test_idx]
    y_tr, y_te = target[train_idx], target[test_idx]
    lam = ridge_alpha * n_tr
    beta_full = fit_ridge(z_tr, y_tr, lam)
    beta_x = fit_ridge(z_tr[:, :p_x], y_tr, lam)
    pred_full = _matvec(z_te, beta_full)
    pred_x = _matvec(z_te[:, :p_x], beta_x)
    if ar is not None:
        beta_ar = fit_ridge(z_tr[:, p_x:], y_tr, lam)
        pred_ar = _matvec(z_te[:, p_x:], beta_ar)
    else:
        pred_ar = np.zeros(n_te, dtype=np.float64)
    sse_zero = _vdot(y_te, y_te)
    if sse_zero <= 0.0:
        return _failed_evaluation(n_tr, n_te, "target never changes in the test period")

    def sse(pred: FloatArray) -> float:
        resid = y_te - pred
        return _vdot(resid, resid)

    sse_full, sse_ar, sse_x = sse(pred_full), sse(pred_ar), sse(pred_x)
    cw = clark_west_test(y_te, pred_ar, pred_full, nw_lags=nw_lags)
    hit, trade_fraction, edge = _directional(pred_full, y_te, cost_hurdle)
    base_hit, base_fraction, base_edge = _directional(pred_ar, y_te, cost_hurdle)
    return PredictiveEvaluation(
        n_train=n_tr,
        n_test=n_te,
        oos_r2=1.0 - sse_full / sse_zero,
        baseline_oos_r2=1.0 - sse_ar / sse_zero,
        x_only_oos_r2=1.0 - sse_x / sse_zero,
        incremental_oos_r2=1.0 - sse_full / min(sse_zero, sse_ar),
        cw_statistic=cw.statistic,
        cw_p_value=cw.p_value,
        hit_rate=hit,
        trade_fraction=trade_fraction,
        economic_edge_estimate=edge,
        baseline_hit_rate=base_hit,
        baseline_trade_fraction=base_fraction,
        baseline_economic_edge=base_edge,
        lead_coefficients=tuple(float(v) for v in beta_full[:p_x]),
    )


# ======================================================================================
# (e) Contemporaneous check
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ContemporaneousCheck:
    corr_at_zero: float
    best_positive_lag: int | None  # grid steps
    corr_at_best_positive: float
    zero_dominates: bool  # |corr(0)| >= |corr(best positive lag)|

    @property
    def ratio(self) -> float:
        """``|corr(0)| / |corr(best positive)|`` (inf when the latter is 0)."""
        denom = abs(self.corr_at_best_positive)
        if not math.isfinite(denom) or not math.isfinite(self.corr_at_zero):
            return _NAN
        return abs(self.corr_at_zero) / denom if denom > 0 else math.inf


def contemporaneous_check(ccf: CrossCorrelation) -> ContemporaneousCheck:
    """Compare the zero-lag correlation with the strongest X-leads-Y correlation."""
    zero = ccf.at(0)
    lag, corr = ccf.best(positive_only=True)
    dominates = math.isfinite(zero) and (not math.isfinite(corr) or abs(zero) >= abs(corr))
    return ContemporaneousCheck(zero, lag, corr, bool(dominates))
