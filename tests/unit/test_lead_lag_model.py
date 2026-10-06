"""LeadLagPredictor: causal features, train-fold-only fitting, versioned parameters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cma.domain.errors import StrategyConfigMismatchError
from cma.domain.time import NS_PER_MS, NS_PER_S
from cma.models.lead_lag import LeadLagPredictor, LeadLagPredictorConfig
from cma.models.lead_lag.series import EventSeries, forward_change, make_grid
from cma.models.lead_lag.synthetic import SyntheticPair, lead_lag_pair

pytestmark = pytest.mark.unit

CFG = LeadLagPredictorConfig(
    horizon_ms=1_000,
    bucket_ms=200,
    lookback_ms=4_000,
    ar_lookback_ms=1_000,
    max_age_ms_y=5_000,
    x_return_kind="diff",
    y_return_kind="diff",
)


@pytest.fixture(scope="module")
def pair() -> SyntheticPair:
    return lead_lag_pair(seed=21, duration_s=1_800)


@pytest.fixture(scope="module")
def split_ns(pair: SyntheticPair) -> int:
    return int(pair.x_ts_ns[0] + 0.7 * (pair.x_ts_ns[-1] - pair.x_ts_ns[0]))


@pytest.fixture(scope="module")
def model(pair: SyntheticPair, split_ns: int) -> LeadLagPredictor:
    return LeadLagPredictor(CFG).fit(
        pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, train_end_ns=split_ns
    )


def _oos_r2(model: LeadLagPredictor, pair: SyntheticPair, start_ns: int) -> float:
    at = make_grid(start_ns, int(pair.y_ts_ns[-1]) - 2 * NS_PER_S, 100 * NS_PER_MS)
    pred = model.predict(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, at)
    target = forward_change(
        EventSeries.from_arrays(pair.y_ts_ns, pair.y_values),
        at,
        horizon_ns=1_000 * NS_PER_MS,
        max_age_ns=5_000 * NS_PER_MS,
        kind="diff",
    )
    ok = np.isfinite(pred) & np.isfinite(target)
    resid = target[ok] - pred[ok]
    return float(1 - (resid @ resid) / (target[ok] @ target[ok]))


def test_predictor_learns_the_lead_out_of_sample(
    model: LeadLagPredictor, pair: SyntheticPair, split_ns: int
) -> None:
    assert model.training_info["last_label_end_ns"] < split_ns
    assert _oos_r2(model, pair, split_ns + 5 * NS_PER_S) > 0.25
    # Y(t+1s) - Y(t) ~ X(t-1s) - X(t-2s): buckets (t-2s, t-1s] carry the weight
    names = CFG.feature_names()
    x_coef = model.coefficients[: CFG.n_lead_buckets]
    top = {names[i] for i in np.argsort(-np.abs(x_coef))[:3]}
    assert top <= {f"x_chg_{lo}_{lo + 200}ms" for lo in range(800, 2_200, 200)}


def test_predictions_use_only_information_up_to_the_decision_time(
    model: LeadLagPredictor, pair: SyntheticPair, split_ns: int
) -> None:
    at = make_grid(split_ns, split_ns + 30 * NS_PER_S, 500 * NS_PER_MS)
    full = model.predict(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, at)
    for i, t in enumerate(at.tolist()):
        xk, yk = pair.x_ts_ns <= t, pair.y_ts_ns <= t
        truncated = model.predict(
            pair.x_ts_ns[xk], pair.x_values[xk], pair.y_ts_ns[yk], pair.y_values[yk], [t]
        )
        assert truncated[0] == pytest.approx(full[i], rel=1e-12, abs=1e-15)


def test_params_round_trip_and_content_hash(
    model: LeadLagPredictor, pair: SyntheticPair, split_ns: int
) -> None:
    params = model.params()
    restored = LeadLagPredictor.from_params(
        json.loads(json.dumps(params)), expected_version=model.model_version
    )
    assert restored.model_version == model.model_version
    assert len(model.model_version) == 64
    at = make_grid(split_ns, split_ns + 60 * NS_PER_S, NS_PER_S)
    args = (pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, at)
    np.testing.assert_array_equal(restored.predict(*args), model.predict(*args))
    tampered = json.loads(json.dumps(params))
    tampered["coefficients"][0] += 1e-9
    with pytest.raises(StrategyConfigMismatchError):
        LeadLagPredictor.from_params(tampered, expected_version=model.model_version)
    assert LeadLagPredictor.from_params(tampered).model_version != model.model_version
    assert params["scaler"]["n_fit"] == model.training_info["n_rows"]  # train rows only


def test_sensitivity_scaling_of_lead_features() -> None:
    """Y moves by delta(t) x the lagged X move; supplying delta(t) explains Y better."""
    rng = np.random.default_rng(8)
    step = 100 * NS_PER_MS
    ts = np.arange(18_000, dtype=np.int64) * step  # 30 minutes at 10 Hz
    dx = 1e-4 * np.sqrt(0.1) * rng.standard_normal(ts.size)

    def delta(t: np.ndarray) -> np.ndarray:
        return np.where((np.asarray(t) // (60 * NS_PER_S)) % 2 == 0, 0.3, 3.0)

    dy = np.zeros(ts.size)
    dy[20:] = delta(ts[20:]) * dx[:-20]  # 2 s lag, regime-dependent sensitivity
    x, y = np.cumsum(dx), np.cumsum(dy) + 2e-5 * rng.standard_normal(ts.size)
    cfg = LeadLagPredictorConfig(
        horizon_ms=1_000, bucket_ms=200, lookback_ms=3_000, x_return_kind="diff"
    )
    split = int(ts[int(0.7 * ts.size)])
    plain = LeadLagPredictor(cfg).fit(ts, x, ts, y, train_end_ns=split)
    scaled = LeadLagPredictor(cfg).fit(ts, x, ts, y, train_end_ns=split, sensitivity=delta)
    at = ts[(ts > split + 5 * NS_PER_S) & (ts < ts[-1] - 2 * NS_PER_S)]
    target = forward_change(
        EventSeries.from_arrays(ts, y), at, horizon_ns=NS_PER_S, max_age_ns=NS_PER_S, kind="diff"
    )

    def r2(pred: np.ndarray) -> float:
        return float(1 - np.sum((target - pred) ** 2) / np.sum(target**2))

    assert scaled.uses_sensitivity
    with pytest.raises(ValueError, match="sensitivity"):
        scaled.predict(ts, x, ts, y, at)
    r2_scaled = r2(scaled.predict(ts, x, ts, y, at, sensitivity=delta(at)))
    r2_plain = r2(plain.predict(ts, x, ts, y, at))
    assert r2_scaled > 0.8
    assert r2_scaled > r2_plain + 0.2


def test_fit_requires_enough_training_rows(pair: SyntheticPair) -> None:
    with pytest.raises(ValueError, match="usable training rows"):
        LeadLagPredictor(CFG).fit(
            pair.x_ts_ns,
            pair.x_values,
            pair.y_ts_ns,
            pair.y_values,
            train_end_ns=int(pair.x_ts_ns[0]) + 10 * NS_PER_S,
        )
    with pytest.raises(RuntimeError):
        LeadLagPredictor(CFG).params()
