"""T037 - Lead-lag synthetic positive: X leading Y by a known lag is recovered."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from cma.models.lead_lag import LeadLagConfig, LeadLagResult, discover_lead_lag
from cma.models.lead_lag.synthetic import SyntheticPair, lead_lag_pair, pair_from_frame

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TRUE_LAG_MS = 2_000
# Fixture values are log-prices, hence "diff" changes; the hurdle (~half a typical 1 s
# |dY|) is in Y units.
CFG = LeadLagConfig(
    grid_ms=100,
    max_lag_ms=5_000,
    horizons_ms=(500, 1_000, 2_000),
    max_age_ms=2_000,
    max_age_ms_y=5_000,
    x_return_kind="diff",
    y_return_kind="diff",
    cost_hurdle=5e-5,
    seed=7,
)


@pytest.fixture(scope="module")
def fixture_pair() -> SyntheticPair:
    return pair_from_frame(pd.read_parquet(FIXTURES / "lead_lag_2s.parquet"))


@pytest.fixture(scope="module")
def result(fixture_pair: SyntheticPair) -> LeadLagResult:
    p = fixture_pair
    return discover_lead_lag(p.x_ts_ns, p.x_values, p.y_ts_ns, p.y_values, CFG)


def test_T037_fixture_shape() -> None:
    frame = pd.read_parquet(FIXTURES / "lead_lag_2s.parquet")
    assert list(frame.columns) == ["series", "ts_ns", "value"]
    assert set(frame["series"]) == {"x", "y"}
    assert frame["ts_ns"].dtype == np.int64
    span_s = (frame["ts_ns"].max() - frame["ts_ns"].min()) / 1e9
    assert 3_600 <= span_s <= 7_300
    assert (FIXTURES / "lead_lag_2s.parquet").stat().st_size < 2_000_000


def test_T037_fixture_matches_its_generator(fixture_pair: SyntheticPair) -> None:
    meta = json.loads(pq.read_schema(FIXTURES / "lead_lag_2s.parquet").metadata[b"cma_fixture"])
    assert meta["kind"] == "lead_lag"
    assert meta["lag_ms"] == TRUE_LAG_MS
    regenerated = lead_lag_pair(
        seed=meta["seed"],
        duration_s=meta["duration_s"],
        lag_ms=meta["lag_ms"],
        x_rate_hz=meta["x_rate_hz"],
        y_rate_hz=meta["y_rate_hz"],
        sigma=meta["sigma"],
        noise_std=meta["noise_std"],
    )
    np.testing.assert_array_equal(regenerated.x_ts_ns, fixture_pair.x_ts_ns)
    np.testing.assert_array_equal(regenerated.y_values, fixture_pair.y_values)


def test_T037_recovers_2s_lead_lag(result: LeadLagResult) -> None:
    assert result.best_lag_ms is not None
    assert result.hy_best_lag_ms is not None
    assert abs(result.best_lag_ms - TRUE_LAG_MS) <= CFG.grid_ms  # grid CCF
    assert abs(result.hy_best_lag_ms - TRUE_LAG_MS) <= CFG.grid_ms  # Hayashi-Yoshida
    assert result.best_positive_lag_ms == result.best_lag_ms
    assert result.corr_at_best > 0
    assert abs(result.corr_at_best) > 10 * abs(result.corr_at_zero)
    assert max(result.ccf, key=lambda k: abs(result.ccf[k])) == result.best_lag_ms
    assert max(result.hy_ccf, key=lambda k: abs(result.hy_ccf[k])) == result.hy_best_lag_ms


def test_T037_result_qualifies_with_reasons(result: LeadLagResult) -> None:
    assert result.qualifies, result.reasons
    gates = [r for r in result.reasons if r.startswith(("PASS", "FAIL"))]
    assert {g.split(":")[0] for g in gates} == {
        "PASS data",
        "PASS significance",
        "PASS oos_r2",
        "PASS oos_stability",
        "PASS oos_significance",
        "PASS hit_rate",
        "PASS economic",
    }
    assert result.p_value <= CFG.alpha
    assert result.oos_r2 > result.baseline_oos_r2 + CFG.min_oos_r2
    assert result.incremental_oos_r2 >= CFG.min_oos_r2
    assert result.hit_rate > 0.5
    assert result.economic_edge_estimate > 0
    assert result.n_train > 0
    assert result.n_test > 0
    assert result.horizon_ms in CFG.horizons_ms
    assert result.n_obs > 50_000


def test_T037_deterministic_and_serializable(
    fixture_pair: SyntheticPair, result: LeadLagResult
) -> None:
    p = fixture_pair
    again = discover_lead_lag(p.x_ts_ns, p.x_values, p.y_ts_ns, p.y_values, CFG)
    assert again.to_dict() == result.to_dict()
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["qualifies"] is True
    assert payload["config_hash"] == CFG.config_hash()


@pytest.mark.parametrize("lag_ms", [700, 3_000])
def test_T037_recovers_other_lags(lag_ms: int) -> None:
    pair = lead_lag_pair(seed=lag_ms, duration_s=1_800, lag_ms=lag_ms)
    cfg = dataclasses.replace(CFG, permutation_samples=199)
    res = discover_lead_lag(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, cfg)
    assert res.hy_best_lag_ms is not None
    assert res.best_lag_ms is not None
    assert abs(res.hy_best_lag_ms - lag_ms) <= cfg.grid_ms
    assert abs(res.best_lag_ms - lag_ms) <= 2 * cfg.grid_ms  # LOCF grid: slight right skew
    assert res.qualifies, res.reasons
