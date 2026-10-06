"""Lead-lag predictor: expected change of Y over ``(t, t + h]`` from lagged X changes.

* Features at decision time ``t`` are distributed-lag changes of X over
  ``(t - (j+1) b, t - j b]`` (only observations stamped ``<= t``; max-age LOCF), optionally
  multiplied by a supplied sensitivity such as a digital-option delta ``dP/dlogS`` so one
  set of coefficients maps underlying moves into probability moves; plus optional own
  lags of Y.
* :meth:`LeadLagPredictor.fit` uses a training fold only: given a
  :class:`~cma.backtest.splits.TimePartition` it refuses any input stamped in the locked
  final test (:class:`~cma.domain.errors.FinalTestAccessError`), trains only on decision
  times whose label window ends inside the training partition, and fits its
  :class:`~cma.backtest.scaling.FoldScaler` with every other decision time forbidden.
* :meth:`params` is a JSON-serializable dict; its SHA-256 content hash is the model
  version, and :meth:`from_params` rebuilds an identical predictor.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cma.backtest.scaling import FoldScaler
from cma.backtest.splits import LockedDataset, TimePartition
from cma.domain.errors import ConfigError, StrategyConfigMismatchError
from cma.domain.time import NS_PER_MS
from cma.models.lead_lag.estimators import fit_ridge
from cma.models.lead_lag.series import (
    RETURN_KINDS,
    EventSeries,
    ReturnKind,
    forward_change,
    lagged_window_changes,
    make_grid,
)

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
Sensitivity = Callable[[IntArray], ArrayLike]

__all__ = ["LeadLagPredictor", "LeadLagPredictorConfig"]

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadLagPredictorConfig:
    horizon_ms: int = 1_000
    grid_ms: int = 100  # spacing of training decision times
    bucket_ms: int = 200
    lookback_ms: int = 5_000
    ar_lookback_ms: int = 0  # own-lag Y terms; 0 disables them
    max_age_ms: int = 2_000
    max_age_ms_y: int | None = 5_000  # None -> max_age_ms
    ridge_alpha: float = 1e-2  # penalty = ridge_alpha * n_train on standardized features
    x_return_kind: ReturnKind = "log"
    y_return_kind: ReturnKind = "diff"
    min_train_rows: int = 200

    def __post_init__(self) -> None:
        problems = []
        if min(self.horizon_ms, self.grid_ms, self.bucket_ms, self.lookback_ms) <= 0:
            problems.append("horizon/grid/bucket/lookback must be positive")
        if self.ar_lookback_ms < 0 or self.max_age_ms < 0:
            problems.append("ar_lookback_ms and max_age_ms must be >= 0")
        if self.max_age_ms_y is not None and self.max_age_ms_y < 0:
            problems.append("max_age_ms_y must be >= 0")
        if self.ridge_alpha < 0 or self.min_train_rows < 1:
            problems.append("ridge_alpha >= 0 and min_train_rows >= 1 required")
        if self.x_return_kind not in RETURN_KINDS or self.y_return_kind not in RETURN_KINDS:
            problems.append(f"return kinds must be one of {RETURN_KINDS}")
        if problems:
            raise ConfigError("invalid LeadLagPredictorConfig: " + "; ".join(problems))

    @property
    def n_lead_buckets(self) -> int:
        return max(1, math.ceil(self.lookback_ms / self.bucket_ms))

    @property
    def n_ar_buckets(self) -> int:
        return 0 if self.ar_lookback_ms == 0 else math.ceil(self.ar_lookback_ms / self.bucket_ms)

    @property
    def resolved_max_age_y_ms(self) -> int:
        return self.max_age_ms if self.max_age_ms_y is None else self.max_age_ms_y

    def feature_names(self) -> list[str]:
        b = self.bucket_ms
        names = [f"x_chg_{j * b}_{(j + 1) * b}ms" for j in range(self.n_lead_buckets)]
        names += [f"y_chg_{j * b}_{(j + 1) * b}ms" for j in range(self.n_ar_buckets)]
        return names

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


class LeadLagPredictor:
    """Ridge lead-lag model; see the module docstring for the leakage guarantees."""

    def __init__(self, config: LeadLagPredictorConfig | None = None) -> None:
        self.config = config or LeadLagPredictorConfig()
        self._scaler: FoldScaler | None = None
        self._coef: FloatArray | None = None
        self._uses_sensitivity = False
        self._training: dict[str, Any] = {}

    # ------------------------------------------------------------------ state
    @property
    def is_fitted(self) -> bool:
        return self._coef is not None

    @property
    def uses_sensitivity(self) -> bool:
        return self._uses_sensitivity

    @property
    def coefficients(self) -> FloatArray:
        if self._coef is None:
            raise RuntimeError("LeadLagPredictor is not fitted")
        return self._coef.copy()

    @property
    def training_info(self) -> dict[str, Any]:
        return dict(self._training)

    # ------------------------------------------------------------------ features
    def _feature_matrix(
        self,
        xs: EventSeries,
        ys: EventSeries | None,
        at_ns: IntArray,
        sensitivity: FloatArray | None,
    ) -> FloatArray:
        cfg = self.config
        lead = lagged_window_changes(
            xs,
            at_ns,
            bucket_ns=cfg.bucket_ms * NS_PER_MS,
            n_buckets=cfg.n_lead_buckets,
            max_age_ns=cfg.max_age_ms * NS_PER_MS,
            kind=cfg.x_return_kind,
        )
        if sensitivity is not None:
            lead = lead * sensitivity[:, None]
        if cfg.n_ar_buckets == 0:
            return lead
        if ys is None:
            raise ValueError("this predictor uses own lags of Y; pass y observations")
        ar = lagged_window_changes(
            ys,
            at_ns,
            bucket_ns=cfg.bucket_ms * NS_PER_MS,
            n_buckets=cfg.n_ar_buckets,
            max_age_ns=cfg.resolved_max_age_y_ms * NS_PER_MS,
            kind=cfg.y_return_kind,
        )
        return np.hstack((lead, ar))

    @staticmethod
    def _sensitivity(
        sensitivity: Sensitivity | ArrayLike | None, at_ns: IntArray
    ) -> FloatArray | None:
        if sensitivity is None:
            return None
        raw = sensitivity(at_ns) if callable(sensitivity) else sensitivity
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.shape != at_ns.shape:
            raise ValueError("sensitivity must provide one value per decision time")
        return values

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        x_ts_ns: ArrayLike,
        x_values: ArrayLike,
        y_ts_ns: ArrayLike,
        y_values: ArrayLike,
        *,
        partition: TimePartition | None = None,
        train_end_ns: int | None = None,
        sensitivity: Sensitivity | None = None,
    ) -> LeadLagPredictor:
        """Fit on the training fold only.

        With ``partition``: any X/Y observation stamped in the locked final test (or its
        embargo) raises :class:`FinalTestAccessError`; training decision times must lie in
        the training partition with their label window ``(t, t + h]`` ending before the
        validation boundary. ``train_end_ns`` additionally caps label ends. All other
        decision times are passed to the scaler as forbidden rows.
        """
        cfg = self.config
        xs = EventSeries.from_arrays(x_ts_ns, x_values, name="x")  # validates dtype/order
        ys = EventSeries.from_arrays(y_ts_ns, y_values, name="y")
        if partition is not None:  # every supplied row counts, even ones cleaning drops
            partition.assert_outside_final_test(x_ts_ns, context="LeadLagPredictor.fit (X)")
            partition.assert_outside_final_test(y_ts_ns, context="LeadLagPredictor.fit (Y)")
        if len(xs) < 2 or len(ys) < 2:
            raise ValueError("need at least two observations of each series")
        h_ns = cfg.horizon_ms * NS_PER_MS
        lookback_ns = max(cfg.lookback_ms, cfg.ar_lookback_ms) * NS_PER_MS
        grid = make_grid(
            max(xs.start_ns, ys.start_ns) + lookback_ns,
            min(xs.end_ns, ys.end_ns) - h_ns,
            cfg.grid_ms * NS_PER_MS,
        )
        label_end = grid + h_ns
        train = np.ones(grid.size, dtype=np.bool_)
        if partition is not None:
            train &= partition.train_mask(grid, label_end)
        if train_end_ns is not None:
            train &= label_end < train_end_ns
        sens = self._sensitivity(sensitivity, grid)
        features = self._feature_matrix(xs, ys, grid, sens)
        target = forward_change(
            ys,
            grid,
            horizon_ns=h_ns,
            max_age_ns=cfg.resolved_max_age_y_ms * NS_PER_MS,
            kind=cfg.y_return_kind,
        )
        usable = train & np.isfinite(features).all(axis=1) & np.isfinite(target)
        rows = np.flatnonzero(usable)
        if rows.size < max(cfg.min_train_rows, 2 * features.shape[1]):
            raise ValueError(f"only {rows.size} usable training rows")
        scaler = FoldScaler().fit(
            features[rows], row_ids=grid[rows], forbidden_row_ids=grid[~train]
        )
        z = scaler.transform(features[rows])
        coef = fit_ridge(z, target[rows], cfg.ridge_alpha * rows.size)
        resid = target[rows] - np.einsum("ij,j->i", z, coef)
        sst = float(np.einsum("i,i->", target[rows], target[rows]))
        self._scaler = scaler
        self._coef = coef
        self._uses_sensitivity = sensitivity is not None
        self._training = {
            "n_rows": int(rows.size),
            "first_decision_ns": int(grid[rows[0]]),
            "last_decision_ns": int(grid[rows[-1]]),
            "last_label_end_ns": int(label_end[rows[-1]]),
            "final_test_start_ns": None if partition is None else partition.final_test_start_ns,
            "train_end_ns": None if train_end_ns is None else int(train_end_ns),
            "in_sample_r2": float(1.0 - float(np.einsum("i,i->", resid, resid)) / sst)
            if sst > 0
            else None,
        }
        return self

    def fit_dataset(
        self,
        dataset: LockedDataset,
        *,
        series_col: str = "series",
        value_col: str = "value",
        x_label: str = "x",
        y_label: str = "y",
        sensitivity: Sensitivity | None = None,
    ) -> LeadLagPredictor:
        """Fit from a long-format :class:`LockedDataset` using its training view only."""
        frame = dataset.train()
        ts_col = dataset.ts_col
        parts = []
        for label in (x_label, y_label):
            part = frame[frame[series_col] == label].sort_values(ts_col, kind="stable")
            parts.append(
                (
                    part[ts_col].to_numpy(dtype=np.int64),
                    part[value_col].to_numpy(dtype=np.float64),
                )
            )
        return self.fit(
            parts[0][0],
            parts[0][1],
            parts[1][0],
            parts[1][1],
            partition=dataset.partition,
            sensitivity=sensitivity,
        )

    # ------------------------------------------------------------------ predicting
    def predict(
        self,
        x_ts_ns: ArrayLike,
        x_values: ArrayLike,
        y_ts_ns: ArrayLike | None,
        y_values: ArrayLike | None,
        at_ns: ArrayLike,
        *,
        sensitivity: Sensitivity | ArrayLike | None = None,
    ) -> FloatArray:
        """Expected change of Y over ``(t, t + horizon]`` for each decision time ``t``.

        Uses only observations stamped at or before ``t`` (later observations in the
        inputs are ignored); NaN where a required input is stale.
        """
        if self._coef is None or self._scaler is None:
            raise RuntimeError("LeadLagPredictor is not fitted")
        at = np.asarray(at_ns, dtype=np.int64).reshape(-1)
        sens = self._sensitivity(sensitivity, at)
        if self._uses_sensitivity and sens is None:
            raise ValueError("model was fitted with a sensitivity; supply one to predict")
        if not self._uses_sensitivity and sens is not None:
            raise ValueError("model was fitted without a sensitivity")
        xs = EventSeries.from_arrays(x_ts_ns, x_values, name="x")
        ys = (
            None
            if y_ts_ns is None or y_values is None
            else EventSeries.from_arrays(y_ts_ns, y_values, name="y")
        )
        features = self._feature_matrix(xs, ys, at, sens)
        pred = np.einsum("ij,j->i", self._scaler.transform(features), self._coef)
        return np.asarray(np.where(np.isfinite(features).all(axis=1), pred, np.nan))

    # ------------------------------------------------------------------ persistence
    def params(self) -> dict[str, Any]:
        """JSON-serializable parameters (configuration, scaler, coefficients, provenance)."""
        if self._coef is None or self._scaler is None:
            raise RuntimeError("LeadLagPredictor is not fitted")
        return {
            "model": "LeadLagPredictor",
            "schema_version": SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "feature_names": self.config.feature_names(),
            "coefficients": [float(v) for v in self._coef],
            "scaler": self._scaler.to_dict(),
            "uses_sensitivity": self._uses_sensitivity,
            "training": dict(self._training),
        }

    @property
    def model_version(self) -> str:
        """SHA-256 content hash of :meth:`params` (canonical JSON)."""
        return _canonical_hash(self.params())

    @classmethod
    def from_params(
        cls, params: dict[str, Any], *, expected_version: str | None = None
    ) -> LeadLagPredictor:
        """Rebuild a fitted predictor; optionally verify its content hash."""
        if params.get("model") != "LeadLagPredictor":
            raise ValueError("not LeadLagPredictor parameters")
        if params.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {params.get('schema_version')}")
        if expected_version is not None and _canonical_hash(params) != expected_version:
            raise StrategyConfigMismatchError(
                "lead-lag model parameters do not match the expected model version"
            )
        model = cls(LeadLagPredictorConfig(**params["config"]))
        if list(params["feature_names"]) != model.config.feature_names():
            raise StrategyConfigMismatchError("feature layout does not match the configuration")
        model._scaler = FoldScaler.from_dict(params["scaler"])
        model._coef = np.asarray(params["coefficients"], dtype=np.float64)
        model._uses_sensitivity = bool(params["uses_sensitivity"])
        model._training = dict(params.get("training", {}))
        return model
