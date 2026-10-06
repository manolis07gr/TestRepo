"""Lead-lag discovery: regime strata, configuration validation, degenerate inputs."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from cma.domain.errors import ConfigError
from cma.domain.time import NS_PER_S
from cma.models.lead_lag import (
    LeadLagConfig,
    discover_lead_lag,
    lead_lag_grid,
    stratified_lead_lag,
)
from cma.models.lead_lag.synthetic import (
    DEFAULT_START_NS,
    SyntheticPair,
    independent_pair,
    lead_lag_pair,
)

pytestmark = pytest.mark.unit

CFG = LeadLagConfig(
    horizons_ms=(1_000, 2_000),
    max_age_ms_y=5_000,
    x_return_kind="diff",
    y_return_kind="diff",
    permutation_samples=199,
)
SEGMENT_S = 1_200
BOUNDARY_NS = DEFAULT_START_NS + SEGMENT_S * NS_PER_S


def _regime_pair() -> SyntheticPair:
    """No relation for the first 20 minutes, X leads Y by 2 s for the next 20 minutes."""
    early = independent_pair(seed=31, duration_s=SEGMENT_S)
    late = lead_lag_pair(seed=32, duration_s=SEGMENT_S, start_ns=BOUNDARY_NS)
    x_late = late.x_values - late.x_values[0] + early.x_values[-1]  # continuous levels
    y_late = late.y_values - late.y_values[0] + early.y_values[-1]
    return SyntheticPair(
        np.concatenate([early.x_ts_ns, late.x_ts_ns]),
        np.concatenate([early.x_values, x_late]),
        np.concatenate([early.y_ts_ns, late.y_ts_ns]),
        np.concatenate([early.y_values, y_late]),
    )


@pytest.fixture(scope="module")
def regime_pair() -> SyntheticPair:
    return _regime_pair()


def test_stratified_lead_lag_isolates_the_regime_with_a_lead(regime_pair: SyntheticPair) -> None:
    p = regime_pair
    labels = np.where(p.y_ts_ns < BOUNDARY_NS, "early", "late")  # one label per Y observation
    results = dict(
        stratified_lead_lag(p.x_ts_ns, p.x_values, p.y_ts_ns, p.y_values, CFG, strata=labels)
    )
    assert list(results) == ["early", "late"]
    assert results["late"].qualifies, results["late"].reasons
    assert results["late"].hy_best_lag_ms is not None
    assert abs(results["late"].hy_best_lag_ms - 2_000) <= CFG.grid_ms
    assert results["late"].stratum == "late"
    assert not results["early"].qualifies
    assert any(r.startswith("FAIL") for r in results["early"].reasons)


def test_stratified_lead_lag_accepts_grid_labels_and_callables(
    regime_pair: SyntheticPair,
) -> None:
    p = regime_pair
    grid = lead_lag_grid(p.x_ts_ns, p.x_values, p.y_ts_ns, p.y_values, CFG)
    per_grid = np.where(grid < BOUNDARY_NS, 0, 1)
    by_grid = stratified_lead_lag(
        p.x_ts_ns, p.x_values, p.y_ts_ns, p.y_values, CFG, strata=per_grid, strata_on="grid"
    )
    by_callable = stratified_lead_lag(
        p.x_ts_ns,
        p.x_values,
        p.y_ts_ns,
        p.y_values,
        CFG,
        strata=lambda g: np.where(g < BOUNDARY_NS, 0, 1),
    )
    assert [label for label, _ in by_grid] == [0, 1]
    assert [r.to_dict() for _, r in by_grid] == [r.to_dict() for _, r in by_callable]
    assert [r.qualifies for _, r in by_grid] == [False, True]


def test_small_strata_are_reported_not_analysed(regime_pair: SyntheticPair) -> None:
    p = regime_pair
    labels = np.where(p.y_ts_ns < DEFAULT_START_NS + 60 * NS_PER_S, "first-minute", "rest")
    res = dict(
        stratified_lead_lag(
            p.x_ts_ns,
            p.x_values,
            p.y_ts_ns,
            p.y_values,
            CFG,
            strata=labels,
            min_stratum_obs=1_000,
        )
    )
    small = res["first-minute"]
    assert not small.qualifies
    assert small.reasons[0].startswith("FAIL data")
    with pytest.raises(ValueError, match="one label per"):
        stratified_lead_lag(p.x_ts_ns, p.x_values, p.y_ts_ns, p.y_values, CFG, strata=labels[:-1])


def test_config_validation_and_hash() -> None:
    for bad in (
        {"horizons_ms": (150,)},  # not a multiple of the 100 ms grid
        {"max_lag_ms": 0},
        {"alpha": 1.5},
        {"permutation_samples": 5},  # p-value could never reach alpha=0.05
        {"x_return_kind": "pct"},
        {"train_fraction": 1.0},
    ):
        with pytest.raises(ConfigError):
            LeadLagConfig(**bad)
    cfg = LeadLagConfig()
    assert cfg.config_hash() == LeadLagConfig().config_hash()
    assert cfg.config_hash() != dataclasses.replace(cfg, seed=8).config_hash()
    assert cfg.resolved_bucket_ms == 200
    assert cfg.n_lead_buckets == 25
    assert cfg.max_lag_steps == 50


def test_non_overlapping_series_do_not_qualify() -> None:
    a = independent_pair(seed=1, duration_s=60)
    later = DEFAULT_START_NS + 3_600 * NS_PER_S
    res = discover_lead_lag(a.x_ts_ns, a.x_values, a.y_ts_ns + later, a.y_values, CFG)
    assert not res.qualifies
    assert res.reasons[0].startswith("FAIL data")
    json.dumps(res.to_dict(), allow_nan=False)
