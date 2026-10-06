"""Fold-aware feature scaling (scope s.11.1: fit scalers on training data only; T035).

:class:`FoldScaler` is a z-score scaler that remembers *which rows* it was fitted on and
refuses to fit when those rows intersect a held-out (test / validation / final-test) fold.
Statistics are therefore provably train-only, and :meth:`FoldScaler.transform` never
refits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cma.domain.errors import LeakageError

FloatArray = NDArray[np.float64]

__all__ = ["FoldScaler", "LeakageError"]


def _as_2d(x: ArrayLike) -> tuple[FloatArray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    if arr.ndim != 2:
        raise ValueError("FoldScaler expects a 1-D or 2-D array")
    return arr, False


class FoldScaler:
    """Z-score scaler whose statistics come only from explicitly identified training rows.

    ``fit(X, row_ids, forbidden_row_ids=...)`` raises :class:`LeakageError` when any
    training row id is also a forbidden (held-out) row id. NaNs are ignored per column;
    columns with (near) zero spread are scaled by 1 so they transform to their centred
    value instead of exploding.
    """

    __slots__ = ("_fit_row_ids", "_mean", "_n_fit", "_std", "ddof", "min_std")

    def __init__(self, *, ddof: int = 0, min_std: float = 1e-12) -> None:
        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        if min_std <= 0:
            raise ValueError("min_std must be positive")
        self.ddof = ddof
        self.min_std = min_std
        self._mean: FloatArray | None = None
        self._std: FloatArray | None = None
        self._n_fit = 0
        self._fit_row_ids: NDArray[Any] | None = None

    # ------------------------------------------------------------------ state
    @property
    def fitted(self) -> bool:
        return self._mean is not None

    def _require_fitted(self) -> tuple[FloatArray, FloatArray]:
        if self._mean is None or self._std is None:
            raise RuntimeError("FoldScaler is not fitted")
        return self._mean, self._std

    @property
    def mean_(self) -> FloatArray:
        return self._require_fitted()[0].copy()

    @property
    def std_(self) -> FloatArray:
        return self._require_fitted()[1].copy()

    @property
    def n_fit(self) -> int:
        return self._n_fit

    @property
    def fit_row_ids(self) -> NDArray[Any] | None:
        return None if self._fit_row_ids is None else self._fit_row_ids.copy()

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        x: ArrayLike,
        row_ids: ArrayLike | None = None,
        *,
        forbidden_row_ids: ArrayLike | None = None,
    ) -> FoldScaler:
        """Estimate mean/std from ``x`` (rows identified by ``row_ids``).

        Raises :class:`LeakageError` if ``row_ids`` intersects ``forbidden_row_ids``.
        """
        arr, _ = _as_2d(x)
        ids: NDArray[Any] | None = None
        if row_ids is not None:
            ids = np.asarray(row_ids).reshape(-1)
            if ids.shape[0] != arr.shape[0]:
                raise ValueError("row_ids must have one id per row of x")
            if np.unique(ids).size != ids.size:
                raise ValueError("row_ids must be unique")
        if forbidden_row_ids is not None:
            if ids is None:
                raise ValueError("forbidden_row_ids requires row_ids to check against")
            forbidden = np.asarray(forbidden_row_ids).reshape(-1)
            leaked = np.intersect1d(ids, forbidden)
            if leaked.size:
                raise LeakageError(
                    f"scaler fit rows include {leaked.size} held-out row(s), e.g. "
                    f"{leaked[:3].tolist()}"
                )
        if arr.shape[0] == 0:
            raise ValueError("cannot fit a scaler on zero rows")
        finite = np.isfinite(arr)
        if bool(finite.all()):  # fast path
            mean = arr.mean(axis=0)
            if arr.shape[0] > self.ddof:
                std = arr.std(axis=0, ddof=self.ddof)
            else:
                std = np.zeros(arr.shape[1], dtype=np.float64)
        else:
            counts = finite.sum(axis=0)
            if bool((counts == 0).any()):
                raise ValueError("every column needs at least one finite training value")
            filled = np.where(finite, arr, 0.0)
            mean = filled.sum(axis=0) / counts
            dev = np.where(finite, arr - mean, 0.0)
            denom = np.maximum(counts - self.ddof, 1)
            std = np.sqrt((dev * dev).sum(axis=0) / denom)
        std = np.where(std > self.min_std, std, 1.0)
        self._mean = np.asarray(mean, dtype=np.float64)
        self._std = np.asarray(std, dtype=np.float64)
        self._n_fit = int(arr.shape[0])
        self._fit_row_ids = None if ids is None else ids.copy()
        return self

    def check_disjoint(self, row_ids: ArrayLike) -> None:
        """Raise :class:`LeakageError` if ``row_ids`` (e.g. a test fold) were fit rows."""
        if self._fit_row_ids is None:
            raise LeakageError("scaler was fitted without row ids; disjointness unknown")
        leaked = np.intersect1d(self._fit_row_ids, np.asarray(row_ids).reshape(-1))
        if leaked.size:
            raise LeakageError(f"{leaked.size} evaluation row(s) were used to fit the scaler")

    # ------------------------------------------------------------------ transforming
    def transform(self, x: ArrayLike) -> FloatArray:
        mean, std = self._require_fitted()
        arr, was_1d = _as_2d(x)
        if arr.shape[1] != mean.shape[0]:
            raise ValueError(f"expected {mean.shape[0]} columns, got {arr.shape[1]}")
        out = np.asarray((arr - mean) / std, dtype=np.float64)
        return out.reshape(-1) if was_1d else out

    def fit_transform(
        self,
        x: ArrayLike,
        row_ids: ArrayLike | None = None,
        *,
        forbidden_row_ids: ArrayLike | None = None,
    ) -> FloatArray:
        return self.fit(x, row_ids, forbidden_row_ids=forbidden_row_ids).transform(x)

    def inverse_transform(self, z: ArrayLike) -> FloatArray:
        mean, std = self._require_fitted()
        arr, was_1d = _as_2d(z)
        out = np.asarray(arr * std + mean, dtype=np.float64)
        return out.reshape(-1) if was_1d else out

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        mean, std = self._require_fitted()
        return {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "n_fit": self._n_fit,
            "ddof": self.ddof,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FoldScaler:
        scaler = cls(ddof=int(data.get("ddof", 0)))
        scaler._mean = np.asarray(data["mean"], dtype=np.float64)
        scaler._std = np.asarray(data["std"], dtype=np.float64)
        scaler._n_fit = int(data.get("n_fit", 0))
        return scaler
