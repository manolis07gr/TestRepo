"""Lead-lag discovery (scope s.4 H1, s.10.1, s.11.1, s.13; tests T037-T039).

The question is not whether X and Y are correlated, but whether a timestamp-causal,
repeatable X -> Y relationship exists that survives out of sample and a cost hurdle.
:func:`discover_lead_lag` therefore combines:

1. a grid cross-correlation scan (``corr(rx_t, ry_{t+k})``, positive k = X leads Y) and
   a lagged Hayashi-Yoshida scan on the raw asynchronous observations;
2. a family-wise significance test of ``max |corr|`` over the scanned lags against a
   circular-shift (or block-permutation) null;
3. an out-of-sample predictive test: ridge on distributed-lag X features (plus Y's own
   lags) fitted on the earlier part of the sample only, evaluated on the later part
   against the zero and own-history (AR) baselines, with a Clark-West test; the
   improvement must also hold in an inner validation window inside the training
   period (stability across two disjoint out-of-sample windows);
4. an economic filter in Y units (``cost_hurdle``);
5. a contemporaneous check: zero-lag correlation alone never qualifies.

The predictive horizon is chosen *inside the training period* (inner chronological
validation), so the reported out-of-sample numbers are not selected on the test period.
Every :class:`LeadLagResult` lists the reasons it did or did not qualify.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Hashable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from cma.domain.errors import ConfigError
from cma.domain.time import NS_PER_MS
from cma.models.lead_lag.estimators import (
    ContemporaneousCheck,
    CrossCorrelation,
    HYResult,
    Increments,
    NullMethod,
    PredictiveEvaluation,
    contemporaneous_check,
    cross_correlation,
    evaluate_predictive,
    hayashi_yoshida,
    increments,
    predictive_split,
    scan_significance,
)
from cma.models.lead_lag.series import (
    RETURN_KINDS,
    AlignedPair,
    EventSeries,
    ReturnKind,
    align_pair,
    forward_change,
    grid_returns,
    lagged_window_changes,
)

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
StrataInput = ArrayLike | Callable[[IntArray], ArrayLike]

__all__ = [
    "HorizonEvaluation",
    "LeadLagConfig",
    "LeadLagResult",
    "discover_lead_lag",
    "lead_lag_grid",
    "stratified_lead_lag",
]

_NAN = float("nan")


# ======================================================================================
# Configuration
# ======================================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadLagConfig:
    """Lead-lag discovery parameters. Times are milliseconds unless stated otherwise.

    ``block_len`` is in grid steps: the guard band added to the minimum circular shift
    (``2 * max_lag + block_len``) and the block size of the block-permutation null.
    ``cost_hurdle`` is in the units of the Y target (``y_return_kind``; ``"diff"`` = Y
    units such as probability points): a prediction is "tradeable" only if its magnitude
    exceeds the hurdle, and the edge of trading the tradeable predictions must stay
    positive after paying it.
    """

    grid_ms: int = 100
    max_lag_ms: int = 5_000
    horizons_ms: tuple[int, ...] = (500, 1_000, 2_000)
    max_age_ms: int = 2_000  # X (reference) LOCF limit, cf. DataQualityConfig.max_forward_fill_ms
    max_age_ms_y: int | None = 5_000  # Y (prediction feed) limit; None -> max_age_ms
    alpha: float = 0.05
    permutation_samples: int = 999
    block_len: int = 50
    null_method: NullMethod = "circular_shift"
    significance_lags: Literal["positive", "all"] = "positive"
    min_oos_r2: float = 0.005
    cost_hurdle: float = 0.0
    min_trade_fraction: float = 0.01
    min_hit_rate: float = 0.5
    seed: int = 7
    x_return_kind: ReturnKind = "log"
    y_return_kind: ReturnKind = "diff"
    feature_bucket_ms: int | None = None  # None -> grid_ms * max(1, 250 // grid_ms)
    feature_lookback_ms: int | None = None  # None -> max_lag_ms
    ar_lookback_ms: int | None = None  # None -> feature lookback; 0 disables the AR terms
    ridge_alpha: float = 1e-2  # penalty = ridge_alpha * n_train on standardized features
    train_fraction: float = 0.7
    inner_validation_fraction: float = 0.3
    oos_embargo_ms: int = 0
    hy_lag_step_ms: int | None = None  # None -> grid_ms
    min_obs: int = 1_000
    min_pairs_per_lag: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons_ms", tuple(int(h) for h in self.horizons_ms))
        errors: list[str] = []
        if self.grid_ms <= 0:
            errors.append("grid_ms must be positive")
        else:
            if self.max_lag_ms <= 0 or self.max_lag_ms % self.grid_ms:
                errors.append("max_lag_ms must be a positive multiple of grid_ms")
            if not self.horizons_ms or any(h <= 0 or h % self.grid_ms for h in self.horizons_ms):
                errors.append("horizons_ms must be non-empty positive multiples of grid_ms")
            if self.feature_bucket_ms is not None and (
                self.feature_bucket_ms <= 0 or self.feature_bucket_ms % self.grid_ms
            ):
                errors.append("feature_bucket_ms must be a positive multiple of grid_ms")
        if len(set(self.horizons_ms)) != len(self.horizons_ms):
            errors.append("horizons_ms must be unique")
        if self.max_age_ms < 0 or (self.max_age_ms_y is not None and self.max_age_ms_y < 0):
            errors.append("max ages must be non-negative")
        if not (0.0 < self.alpha < 1.0):
            errors.append("alpha must be in (0, 1)")
        elif self.permutation_samples < math.ceil(1.0 / self.alpha) - 1:
            errors.append("permutation_samples too small for a p-value <= alpha")
        if self.block_len < 1:
            errors.append("block_len must be >= 1")
        if self.null_method not in ("circular_shift", "block_permutation"):
            errors.append(f"unknown null_method {self.null_method!r}")
        if self.significance_lags not in ("positive", "all"):
            errors.append("significance_lags must be 'positive' or 'all'")
        if self.cost_hurdle < 0 or not (0.0 <= self.min_trade_fraction <= 1.0):
            errors.append("cost_hurdle must be >= 0 and min_trade_fraction in [0, 1]")
        if not (0.0 <= self.min_hit_rate < 1.0):
            errors.append("min_hit_rate must be in [0, 1)")
        if self.x_return_kind not in RETURN_KINDS or self.y_return_kind not in RETURN_KINDS:
            errors.append(f"return kinds must be one of {RETURN_KINDS}")
        for name in ("feature_lookback_ms", "hy_lag_step_ms"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                errors.append(f"{name} must be positive")
        if self.ar_lookback_ms is not None and self.ar_lookback_ms < 0:
            errors.append("ar_lookback_ms must be >= 0")
        if self.ridge_alpha < 0:
            errors.append("ridge_alpha must be >= 0")
        if not (0.0 < self.train_fraction < 1.0) or not (
            0.0 < self.inner_validation_fraction < 1.0
        ):
            errors.append("train_fraction and inner_validation_fraction must be in (0, 1)")
        if self.oos_embargo_ms < 0 or self.min_obs < 1 or self.min_pairs_per_lag < 2:
            errors.append("oos_embargo_ms >= 0, min_obs >= 1 and min_pairs_per_lag >= 2")
        if errors:
            raise ConfigError("invalid LeadLagConfig: " + "; ".join(errors))

    # ------------------------------------------------------------------ derived values
    @property
    def grid_ns(self) -> int:
        return self.grid_ms * NS_PER_MS

    @property
    def max_lag_steps(self) -> int:
        return self.max_lag_ms // self.grid_ms

    @property
    def resolved_max_age_y_ms(self) -> int:
        return self.max_age_ms if self.max_age_ms_y is None else self.max_age_ms_y

    @property
    def resolved_bucket_ms(self) -> int:
        if self.feature_bucket_ms is not None:
            return self.feature_bucket_ms
        return self.grid_ms * max(1, 250 // self.grid_ms)

    @property
    def n_lead_buckets(self) -> int:
        lookback = self.feature_lookback_ms or self.max_lag_ms
        return max(1, math.ceil(lookback / self.resolved_bucket_ms))

    @property
    def n_ar_buckets(self) -> int:
        if self.ar_lookback_ms == 0:
            return 0
        lookback = self.ar_lookback_ms or self.feature_lookback_ms or self.max_lag_ms
        return max(1, math.ceil(lookback / self.resolved_bucket_ms))

    @property
    def resolved_hy_step_ms(self) -> int:
        return self.hy_lag_step_ms or self.grid_ms

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["horizons_ms"] = list(self.horizons_ms)
        return data

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# ======================================================================================
# Results
# ======================================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonEvaluation:
    horizon_ms: int
    n_rows: int
    inner_incremental_oos_r2: float  # selection criterion, computed inside train only
    evaluation: PredictiveEvaluation  # fit on train, evaluated on the later test rows


def _clean_float(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadLagResult:
    """Outcome of a lead-lag discovery run (lags in ms; positive = X leads Y)."""

    best_lag_ms: int | None  # argmax |ccf| over the full scanned range [-max_lag, max_lag]
    corr_at_best: float
    corr_at_zero: float
    p_value: float  # family-wise over the significance scan (see config)
    hy_best_lag_ms: int | None
    oos_r2: float  # lead model vs zero forecast, at the selected horizon
    baseline_oos_r2: float  # own-history (AR) baseline vs zero forecast
    hit_rate: float
    n_obs: int  # grid points with fresh X and Y returns (in the stratum, if any)
    economic_edge_estimate: float  # mean edge per tradeable prediction after the hurdle
    qualifies: bool
    reasons: list[str]
    ccf: dict[int, float]  # lag_ms -> grid cross-correlation
    best_positive_lag_ms: int | None = None
    corr_at_best_positive: float = _NAN
    hy_corr_at_best: float = _NAN
    hy_ccf: dict[int, float] = field(default_factory=dict)  # lag_ms -> HY correlation
    scan_statistic: float = _NAN
    horizon_ms: int | None = None
    incremental_oos_r2: float = _NAN  # lead model vs the better of zero / AR baselines
    x_only_oos_r2: float = _NAN
    oos_p_value: float = _NAN  # Clark-West, lead model vs AR baseline
    trade_fraction: float = _NAN
    n_train: int = 0
    n_test: int = 0
    horizon_results: dict[int, HorizonEvaluation] = field(default_factory=dict)
    stratum: Hashable | None = None
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (non-finite floats become ``None``)."""
        out: dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if key in ("ccf", "hy_ccf"):
                value = {int(k): _clean_float(v) for k, v in value.items()}
            elif key == "horizon_results":
                value = {
                    int(h): {
                        "horizon_ms": ev.horizon_ms,
                        "n_rows": ev.n_rows,
                        "inner_incremental_oos_r2": _clean_float(ev.inner_incremental_oos_r2),
                        **{
                            k: _clean_float(v) if not isinstance(v, tuple) else list(v)
                            for k, v in asdict(ev.evaluation).items()
                        },
                    }
                    for h, ev in value.items()
                }
            elif key == "reasons":
                value = list(value)
            elif key == "stratum" and value is not None:
                value = str(value)
            out[key] = _clean_float(value)
        return out

    def summary(self) -> str:
        verdict = "QUALIFIES" if self.qualifies else "DOES NOT QUALIFY"
        return (
            f"{verdict}: ccf lag={self.best_lag_ms}ms corr={self.corr_at_best:.4f} "
            f"(zero-lag {self.corr_at_zero:.4f}), p={self.p_value:.4g}, "
            f"HY lag={self.hy_best_lag_ms}ms, h={self.horizon_ms}ms "
            f"oos_r2={self.oos_r2:.4f} (AR {self.baseline_oos_r2:.4f}, "
            f"incr {self.incremental_oos_r2:.4f}), hit={self.hit_rate:.3f}, "
            f"edge={self.economic_edge_estimate:.3g}, n={self.n_obs}"
        )


# ======================================================================================
# Preparation (shared by the pooled and the stratified analyses)
# ======================================================================================


@dataclass(frozen=True, slots=True)
class _Prepared:
    aligned: AlignedPair
    rx: FloatArray
    ry: FloatArray
    lead: FloatArray
    ar: FloatArray | None
    features_ok: BoolArray
    targets: dict[int, FloatArray]
    x_inc: Increments
    y_inc: Increments


def _prepare(
    x_ts_ns: ArrayLike,
    x_values: ArrayLike,
    y_ts_ns: ArrayLike,
    y_values: ArrayLike,
    cfg: LeadLagConfig,
) -> _Prepared:
    xs = EventSeries.from_arrays(x_ts_ns, x_values, name="x")
    ys = EventSeries.from_arrays(y_ts_ns, y_values, name="y")
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("each series needs at least two finite observations")
    age_x = cfg.max_age_ms * NS_PER_MS
    age_y = cfg.resolved_max_age_y_ms * NS_PER_MS
    aligned = align_pair(
        xs, None, ys, None, step_ns=cfg.grid_ns, max_age_x_ns=age_x, max_age_y_ns=age_y
    )
    grid = aligned.grid_ns
    bucket_ns = cfg.resolved_bucket_ms * NS_PER_MS
    lead = lagged_window_changes(
        xs,
        grid,
        bucket_ns=bucket_ns,
        n_buckets=cfg.n_lead_buckets,
        max_age_ns=age_x,
        kind=cfg.x_return_kind,
    )
    ar: FloatArray | None = None
    ok = np.isfinite(lead).all(axis=1)
    if cfg.n_ar_buckets > 0:
        ar = lagged_window_changes(
            ys,
            grid,
            bucket_ns=bucket_ns,
            n_buckets=cfg.n_ar_buckets,
            max_age_ns=age_y,
            kind=cfg.y_return_kind,
        )
        ok &= np.isfinite(ar).all(axis=1)
    targets = {
        h: forward_change(
            ys, grid, horizon_ns=h * NS_PER_MS, max_age_ns=age_y, kind=cfg.y_return_kind
        )
        for h in cfg.horizons_ms
    }
    return _Prepared(
        aligned=aligned,
        rx=grid_returns(aligned.x.values, cfg.x_return_kind),
        ry=grid_returns(aligned.y.values, cfg.y_return_kind),
        lead=lead,
        ar=ar,
        features_ok=np.asarray(ok, dtype=np.bool_),
        targets=targets,
        x_inc=increments(xs, cfg.x_return_kind),
        y_inc=increments(ys, cfg.y_return_kind),
    )


def lead_lag_grid(
    x_ts_ns: ArrayLike,
    x_values: ArrayLike,
    y_ts_ns: ArrayLike,
    y_values: ArrayLike,
    cfg: LeadLagConfig,
) -> IntArray:
    """The decision grid :func:`discover_lead_lag` uses (for per-grid-point strata)."""
    xs = EventSeries.from_arrays(x_ts_ns, x_values, name="x")
    ys = EventSeries.from_arrays(y_ts_ns, y_values, name="y")
    age_x = cfg.max_age_ms * NS_PER_MS
    age_y = cfg.resolved_max_age_y_ms * NS_PER_MS
    return align_pair(
        xs, None, ys, None, step_ns=cfg.grid_ns, max_age_x_ns=age_x, max_age_y_ns=age_y
    ).grid_ns


# ======================================================================================
# Analysis
# ======================================================================================


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}g}"


def _gate(name: str, ok: bool, detail: str) -> str:
    return f"{'PASS' if ok else 'FAIL'} {name}: {detail}"


def _steps_to_ms(steps: int | None, cfg: LeadLagConfig) -> int | None:
    return None if steps is None else int(steps) * cfg.grid_ms


def _ns_to_ms(value_ns: int | None) -> int | None:
    return None if value_ns is None else int(value_ns // NS_PER_MS)


def _insufficient(
    reason: str, *, n_obs: int, cfg: LeadLagConfig, stratum: Hashable | None
) -> LeadLagResult:
    return LeadLagResult(
        best_lag_ms=None,
        corr_at_best=_NAN,
        corr_at_zero=_NAN,
        p_value=_NAN,
        hy_best_lag_ms=None,
        oos_r2=_NAN,
        baseline_oos_r2=_NAN,
        hit_rate=_NAN,
        n_obs=n_obs,
        economic_edge_estimate=_NAN,
        qualifies=False,
        reasons=[_gate("data", False, reason)],
        ccf={},
        stratum=stratum,
        config_hash=cfg.config_hash(),
    )


def _evaluate_horizon(
    prep: _Prepared, cfg: LeadLagConfig, row_mask: BoolArray, horizon_ms: int
) -> HorizonEvaluation:
    target = prep.targets[horizon_ms]
    ok = row_mask & prep.features_ok & np.isfinite(target)
    rows = np.flatnonzero(ok)
    h_ns = horizon_ms * NS_PER_MS
    nw_lags = 2 * (horizon_ms // cfg.grid_ms)
    common: dict[str, Any] = {
        "ridge_alpha": cfg.ridge_alpha,
        "cost_hurdle": cfg.cost_hurdle,
        "nw_lags": nw_lags,
    }
    if rows.size < 4:
        failed = PredictiveEvaluation(
            n_train=0,
            n_test=0,
            oos_r2=_NAN,
            baseline_oos_r2=_NAN,
            x_only_oos_r2=_NAN,
            incremental_oos_r2=_NAN,
            cw_statistic=_NAN,
            cw_p_value=_NAN,
            hit_rate=_NAN,
            trade_fraction=_NAN,
            economic_edge_estimate=_NAN,
            note=f"only {rows.size} usable rows",
        )
        return HorizonEvaluation(
            horizon_ms=horizon_ms,
            n_rows=int(rows.size),
            inner_incremental_oos_r2=_NAN,
            evaluation=failed,
        )
    ts = prep.aligned.grid_ns[rows]
    lead = prep.lead[rows]
    ar = None if prep.ar is None else prep.ar[rows]
    y = target[rows]
    embargo = cfg.oos_embargo_ms * NS_PER_MS
    train_idx, test_idx = predictive_split(
        ts, h_ns, train_fraction=cfg.train_fraction, embargo_ns=embargo
    )
    inner_val = _NAN
    if train_idx.size >= 4:
        sub_ts = ts[train_idx]
        in_tr, in_va = predictive_split(
            sub_ts, h_ns, train_fraction=1.0 - cfg.inner_validation_fraction, embargo_ns=embargo
        )
        inner = evaluate_predictive(
            lead[train_idx],
            None if ar is None else ar[train_idx],
            y[train_idx],
            row_ts_ns=sub_ts,
            train_idx=in_tr,
            test_idx=in_va,
            **common,
        )
        inner_val = inner.incremental_oos_r2
    outer = evaluate_predictive(
        lead, ar, y, row_ts_ns=ts, train_idx=train_idx, test_idx=test_idx, **common
    )
    return HorizonEvaluation(
        horizon_ms=horizon_ms,
        n_rows=int(rows.size),
        inner_incremental_oos_r2=inner_val,
        evaluation=outer,
    )


def _mask_at(grid: IntArray, row_mask: BoolArray, ts_ns: IntArray) -> FloatArray:
    """0/1 stratum weight of increments ending at ``ts_ns``.

    Consistent with the grid returns: an increment ending in ``(g[t-1], g[t]]`` belongs to
    the return anchored at ``g[t]`` and takes that grid point's stratum.
    """
    idx = np.searchsorted(grid, ts_ns, side="left")
    inside = (idx >= 1) & (idx < grid.size) & row_mask[np.clip(idx, 0, grid.size - 1)]
    return np.asarray(inside, dtype=np.float64)


def _analyse(
    prep: _Prepared,
    cfg: LeadLagConfig,
    *,
    row_mask: BoolArray | None,
    rng: np.random.Generator,
    stratum: Hashable | None,
) -> LeadLagResult:
    grid = prep.aligned.grid_ns
    mask = np.ones(grid.size, dtype=np.bool_) if row_mask is None else row_mask
    anchors = mask & np.isfinite(prep.rx) & np.isfinite(prep.ry)
    n_obs = int(anchors.sum())
    if n_obs < cfg.min_obs:
        return _insufficient(
            f"only {n_obs} fresh grid observations (< min_obs={cfg.min_obs})",
            n_obs=n_obs,
            cfg=cfg,
            stratum=stratum,
        )
    reasons: list[str] = []

    # (a) grid cross-correlation over [-L, L]
    lag_steps = cfg.max_lag_steps
    ccf: CrossCorrelation = cross_correlation(
        prep.rx, prep.ry, lag_steps, min_pairs=cfg.min_pairs_per_lag, x_mask=mask
    )
    best_steps, corr_best = ccf.best()
    cont: ContemporaneousCheck = contemporaneous_check(ccf)

    # (c) family-wise significance of the scan
    scan = (
        np.arange(1, lag_steps + 1, dtype=np.int64)
        if cfg.significance_lags == "positive"
        else np.arange(-lag_steps, lag_steps + 1, dtype=np.int64)
    )
    sig = scan_significance(
        prep.rx,
        prep.ry,
        scan,
        n_perm=cfg.permutation_samples,
        block_len=cfg.block_len,
        rng=rng,
        method=cfg.null_method,
        x_mask=mask,
    )

    # (b) Hayashi-Yoshida on the raw asynchronous observations
    hy_step = cfg.resolved_hy_step_ms
    k = cfg.max_lag_ms // hy_step
    hy_lags = np.arange(-k, k + 1, dtype=np.int64) * hy_step * NS_PER_MS
    x_w = None if row_mask is None else _mask_at(grid, mask, prep.x_inc.end_ns)
    y_w = None if row_mask is None else _mask_at(grid, mask, prep.y_inc.end_ns)
    hy: HYResult = hayashi_yoshida(prep.x_inc, prep.y_inc, hy_lags, x_weight=x_w, y_weight=y_w)

    # (d) out-of-sample predictive test per horizon; horizon chosen inside train only
    horizons = {h: _evaluate_horizon(prep, cfg, mask, h) for h in cfg.horizons_ms}
    scored = [
        (ev.inner_incremental_oos_r2, -h)
        for h, ev in horizons.items()
        if math.isfinite(ev.inner_incremental_oos_r2)
    ]
    chosen = -max(scored)[1] if scored else cfg.horizons_ms[0]
    ev = horizons[chosen].evaluation

    # ------------------------------------------------------------------ gates
    reasons.append(
        _gate(
            "data",
            ev.ok,
            f"n_obs={n_obs}, n_train={ev.n_train}, n_test={ev.n_test}"
            + (f" ({ev.note})" if ev.note else ""),
        )
    )
    scan_desc = "positive lags" if cfg.significance_lags == "positive" else "all lags"
    sig_ok = math.isfinite(sig.p_value) and sig.p_value <= cfg.alpha
    reasons.append(
        _gate(
            "significance",
            sig_ok,
            f"max|corr| over {scan_desc} = {_fmt(sig.statistic)}, family-wise "
            f"{sig.method} p={_fmt(sig.p_value)} vs alpha={cfg.alpha} "
            f"(null q95={_fmt(sig.null_q95)}, {sig.n_perm} draws)",
        )
    )
    r2_ok = math.isfinite(ev.incremental_oos_r2) and ev.incremental_oos_r2 >= cfg.min_oos_r2
    reasons.append(
        _gate(
            "oos_r2",
            r2_ok,
            f"h={chosen}ms lead-model OOS R2={_fmt(ev.oos_r2)} vs AR baseline "
            f"{_fmt(ev.baseline_oos_r2)} and zero forecast 0; incremental "
            f"{_fmt(ev.incremental_oos_r2)} vs min {cfg.min_oos_r2} "
            f"(X-only {_fmt(ev.x_only_oos_r2)})",
        )
    )
    inner_r2 = horizons[chosen].inner_incremental_oos_r2
    stable_ok = math.isfinite(inner_r2) and inner_r2 >= cfg.min_oos_r2
    reasons.append(
        _gate(
            "oos_stability",
            stable_ok,
            f"incremental R2 in the inner validation window (inside train, disjoint from "
            f"the test window) {_fmt(inner_r2)} vs min {cfg.min_oos_r2}",
        )
    )
    cw_ok = math.isfinite(ev.cw_p_value) and ev.cw_p_value <= cfg.alpha
    reasons.append(
        _gate(
            "oos_significance",
            cw_ok,
            f"Clark-West lead vs AR t={_fmt(ev.cw_statistic)}, p={_fmt(ev.cw_p_value)} "
            f"vs alpha={cfg.alpha}",
        )
    )
    base_hit = ev.baseline_hit_rate if math.isfinite(ev.baseline_hit_rate) else 0.0
    hit_bar = max(cfg.min_hit_rate, base_hit)
    hit_ok = math.isfinite(ev.hit_rate) and ev.hit_rate > hit_bar
    reasons.append(
        _gate(
            "hit_rate",
            hit_ok,
            f"OOS hit rate {_fmt(ev.hit_rate)} vs > max({cfg.min_hit_rate}, AR baseline "
            f"{_fmt(ev.baseline_hit_rate)})",
        )
    )
    value, base_value = ev.economic_value_per_row, ev.baseline_economic_value_per_row
    econ_ok = (
        math.isfinite(ev.trade_fraction)
        and ev.trade_fraction > 0
        and ev.trade_fraction >= cfg.min_trade_fraction
        and math.isfinite(ev.economic_edge_estimate)
        and ev.economic_edge_estimate > 0
        and math.isfinite(value)
        and value > (base_value if math.isfinite(base_value) else 0.0)
    )
    reasons.append(
        _gate(
            "economic",
            econ_ok,
            f"|pred| > hurdle {cfg.cost_hurdle:g} on {_fmt(ev.trade_fraction)} of OOS rows "
            f"(min {cfg.min_trade_fraction}); mean edge after hurdle "
            f"{_fmt(ev.economic_edge_estimate)} (must be > 0); net value per row "
            f"{_fmt(value)} vs AR baseline {_fmt(base_value)} (X must add value)",
        )
    )
    qualifies = bool(ev.ok and sig_ok and r2_ok and stable_ok and cw_ok and hit_ok and econ_ok)

    # ------------------------------------------------------------------ diagnostics
    best_ms = _steps_to_ms(best_steps, cfg)
    direction = (
        "none"
        if best_ms is None
        else "X leads Y"
        if best_ms > 0
        else "contemporaneous"
        if best_ms == 0
        else "Y leads X"
    )
    reasons.append(
        f"INFO direction: argmax |corr| over [-{cfg.max_lag_ms}, {cfg.max_lag_ms}]ms at "
        f"{best_ms}ms ({direction}), corr={_fmt(corr_best)}"
    )
    pos_ms = _steps_to_ms(cont.best_positive_lag, cfg)
    reasons.append(
        f"INFO contemporaneous: |corr(0)|={_fmt(abs(cont.corr_at_zero))} vs best positive "
        f"lag {pos_ms}ms |corr|={_fmt(abs(cont.corr_at_best_positive))}"
        + (
            "; zero-lag correlation dominates - only strictly-past X predicting future Y "
            "out of sample can qualify"
            if cont.zero_dominates
            else ""
        )
    )
    hy_ms = _ns_to_ms(hy.best_lag_ns)
    agree = hy_ms is not None and best_ms is not None and abs(hy_ms - best_ms) <= cfg.grid_ms
    reasons.append(
        f"INFO hy: Hayashi-Yoshida argmax |HY| at {hy_ms}ms (corr={_fmt(hy.best_corr)}); "
        f"{'agrees' if agree else 'disagrees'} with the grid CCF within one grid step"
    )
    reasons.append(
        f"INFO horizon: h={chosen}ms selected by inner-train validation incremental R2 "
        + ", ".join(f"{h}ms={_fmt(v.inner_incremental_oos_r2)}" for h, v in horizons.items())
    )
    if qualifies and best_ms is not None and best_ms <= 0:
        reasons.append(
            "INFO note: the strongest correlation is not at a positive lag; the edge rests "
            "on the out-of-sample predictive test, not on the correlation peak"
        )

    return LeadLagResult(
        best_lag_ms=best_ms,
        corr_at_best=corr_best,
        corr_at_zero=cont.corr_at_zero,
        p_value=sig.p_value,
        hy_best_lag_ms=hy_ms,
        oos_r2=ev.oos_r2,
        baseline_oos_r2=ev.baseline_oos_r2,
        hit_rate=ev.hit_rate,
        n_obs=n_obs,
        economic_edge_estimate=ev.economic_edge_estimate,
        qualifies=qualifies,
        reasons=reasons,
        ccf={
            int(lag) * cfg.grid_ms: float(c)
            for lag, c in zip(ccf.lags.tolist(), ccf.corr.tolist(), strict=True)
        },
        best_positive_lag_ms=pos_ms,
        corr_at_best_positive=cont.corr_at_best_positive,
        hy_corr_at_best=hy.best_corr,
        hy_ccf={
            int(lag // NS_PER_MS): float(c)
            for lag, c in zip(hy.lags_ns.tolist(), hy.corr.tolist(), strict=True)
        },
        scan_statistic=sig.statistic,
        horizon_ms=chosen,
        incremental_oos_r2=ev.incremental_oos_r2,
        x_only_oos_r2=ev.x_only_oos_r2,
        oos_p_value=ev.cw_p_value,
        trade_fraction=ev.trade_fraction,
        n_train=ev.n_train,
        n_test=ev.n_test,
        horizon_results=horizons,
        stratum=stratum,
        config_hash=cfg.config_hash(),
    )


# ======================================================================================
# Public entry points
# ======================================================================================


def discover_lead_lag(
    x_ts_ns: ArrayLike,
    x_values: ArrayLike,
    y_ts_ns: ArrayLike,
    y_values: ArrayLike,
    cfg: LeadLagConfig | None = None,
) -> LeadLagResult:
    """Test whether X leads Y (see module docstring). Deterministic given ``cfg.seed``.

    ``x_*``/``y_*`` are irregular observations: int64 UTC-ns timestamps (non-decreasing)
    and float values. The data passed in must already exclude any locked final-test
    period; the internal train/test split is a research validation split.
    """
    config = cfg or LeadLagConfig()
    prep = _prepare(x_ts_ns, x_values, y_ts_ns, y_values, config)
    if prep.aligned.n == 0:
        return _insufficient("series do not overlap in time", n_obs=0, cfg=config, stratum=None)
    return _analyse(
        prep, config, row_mask=None, rng=np.random.default_rng(config.seed), stratum=None
    )


def _python_label(value: Any) -> Hashable:
    item = getattr(value, "item", None)
    out = item() if callable(item) else value
    return out if isinstance(out, Hashable) else str(out)


def stratified_lead_lag(
    x_ts_ns: ArrayLike,
    x_values: ArrayLike,
    y_ts_ns: ArrayLike,
    y_values: ArrayLike,
    cfg: LeadLagConfig | None = None,
    *,
    strata: StrataInput,
    strata_on: Literal["auto", "y_obs", "grid"] = "auto",
    min_stratum_obs: int | None = None,
) -> list[tuple[Hashable, LeadLagResult]]:
    """Lead-lag stability by regime (time-to-expiry, liquidity, volatility, ...).

    ``strata`` gives a label per Y observation (carried forward to the grid), a label per
    grid point (see :func:`lead_lag_grid`), or a callable mapping grid timestamps to
    labels. Missing labels (None/NaN) are excluded. Each stratum is analysed with the
    full discovery protocol on the decision times carrying its label; strata with fewer
    than ``min_stratum_obs`` (default ``cfg.min_obs``) fresh observations are reported as
    non-qualifying with the reason. Results are ordered by label.
    """
    config = cfg or LeadLagConfig()
    prep = _prepare(x_ts_ns, x_values, y_ts_ns, y_values, config)
    grid = prep.aligned.grid_ns
    if callable(strata):
        labels = np.asarray(strata(grid), dtype=object)
        if labels.shape != grid.shape:
            raise ValueError("strata callable must return one label per grid point")
    else:
        raw = np.asarray(strata, dtype=object).reshape(-1)
        y_ts = np.asarray(y_ts_ns, dtype=np.int64).reshape(-1)
        mode = strata_on
        if mode == "auto":
            fits_y, fits_grid = raw.size == y_ts.size, raw.size == grid.size
            if fits_y and fits_grid:
                raise ValueError("ambiguous strata length; pass strata_on='y_obs' or 'grid'")
            if not (fits_y or fits_grid):
                raise ValueError("strata must have one label per Y observation or grid point")
            mode = "y_obs" if fits_y else "grid"
        if mode == "y_obs":
            if raw.size != y_ts.size:
                raise ValueError("strata_on='y_obs' needs one label per Y observation")
            idx = np.searchsorted(y_ts, grid, side="right") - 1
            labels = np.full(grid.size, None, dtype=object)
            seen = idx >= 0
            labels[seen] = raw[idx[seen]]
        else:
            if raw.size != grid.size:
                raise ValueError("strata_on='grid' needs one label per grid point")
            labels = raw
    try:
        codes, uniques = pd.factorize(pd.Series(labels, dtype=object), sort=True)
    except TypeError:  # labels of mutually unorderable types
        codes, uniques = pd.factorize(pd.Series(labels, dtype=object), sort=False)
    threshold = config.min_obs if min_stratum_obs is None else min_stratum_obs
    results: list[tuple[Hashable, LeadLagResult]] = []
    for i, raw_label in enumerate(list(uniques)):
        label = _python_label(raw_label)
        mask = np.asarray(codes == i, dtype=np.bool_)
        fresh = int((mask & np.isfinite(prep.rx) & np.isfinite(prep.ry)).sum())
        if fresh < threshold:
            results.append(
                (
                    label,
                    _insufficient(
                        f"stratum has only {fresh} fresh grid observations (< {threshold})",
                        n_obs=fresh,
                        cfg=config,
                        stratum=label,
                    ),
                )
            )
            continue
        rng = np.random.default_rng([config.seed, i])
        results.append((label, _analyse(prep, config, row_mask=mask, rng=rng, stratum=label)))
    return results
