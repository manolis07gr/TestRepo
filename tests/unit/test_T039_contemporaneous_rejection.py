"""T039 - Zero-lag correlation without future predictive power is not a lead-lag edge."""

from __future__ import annotations

import dataclasses

import pytest

from cma.models.lead_lag import LeadLagConfig, discover_lead_lag
from cma.models.lead_lag.synthetic import contemporaneous_pair

pytestmark = pytest.mark.unit

CFG = LeadLagConfig(
    grid_ms=100,
    max_lag_ms=5_000,
    horizons_ms=(500, 1_000, 2_000),
    x_return_kind="diff",
    y_return_kind="diff",
    permutation_samples=199,
    seed=7,
)


def test_T039_contemporaneous_correlation_does_not_qualify() -> None:
    pair = contemporaneous_pair(seed=39, duration_s=3_600)
    res = discover_lead_lag(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, CFG)
    assert res.best_lag_ms == 0
    assert res.corr_at_zero > 0.5  # strong common shocks at identical timestamps
    assert abs(res.corr_at_zero) > 10 * abs(res.corr_at_best_positive)
    assert not res.qualifies
    assert any("zero-lag correlation dominates" in r for r in res.reasons)
    assert any(r.startswith("FAIL oos_r2") for r in res.reasons)
    assert res.incremental_oos_r2 < CFG.min_oos_r2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_T039_rejected_even_when_lag_zero_is_in_the_significance_scan(seed: int) -> None:
    """Including lag 0 makes the correlation 'significant'; the OOS gate still rejects."""
    cfg = dataclasses.replace(CFG, significance_lags="all")
    pair = contemporaneous_pair(seed=seed, duration_s=1_800)
    res = discover_lead_lag(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, cfg)
    assert res.p_value <= cfg.alpha
    assert any(r.startswith("PASS significance") for r in res.reasons)
    assert not res.qualifies
    oos_gates = ("FAIL oos_r2", "FAIL oos_stability", "FAIL oos_significance")
    assert any(r.startswith(oos_gates) for r in res.reasons)
