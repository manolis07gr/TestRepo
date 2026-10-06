"""T038 - Lead-lag synthetic null: independent series do not pass the configured filters."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from cma.models.lead_lag import LeadLagConfig, discover_lead_lag
from cma.models.lead_lag.synthetic import independent_pair, pair_from_frame

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
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
N_SEEDS = 20
FPR_MARGIN = 0.05


def test_T038_null_fixture_does_not_qualify() -> None:
    pair = pair_from_frame(pd.read_parquet(FIXTURES / "lead_lag_null.parquet"))
    res = discover_lead_lag(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, CFG)
    assert not res.qualifies
    failed = {r.split(":")[0] for r in res.reasons if r.startswith("FAIL")}
    assert "FAIL oos_r2" in failed  # no out-of-sample predictive value beyond baselines
    assert res.incremental_oos_r2 < CFG.min_oos_r2
    assert len(res.reasons) >= 6  # every gate reports PASS/FAIL with its numbers


def test_T038_null_fixture_matches_its_generator() -> None:
    path = FIXTURES / "lead_lag_null.parquet"
    meta = json.loads(pq.read_schema(path).metadata[b"cma_fixture"])
    assert meta["kind"] == "null"
    regenerated = independent_pair(
        seed=meta["seed"],
        duration_s=meta["duration_s"],
        x_rate_hz=meta["x_rate_hz"],
        y_rate_hz=meta["y_rate_hz"],
        sigma=meta["sigma"],
        noise_std=meta["noise_std"],
    )
    stored = pair_from_frame(pd.read_parquet(path))
    np.testing.assert_array_equal(regenerated.y_ts_ns, stored.y_ts_ns)
    np.testing.assert_array_equal(regenerated.x_values, stored.x_values)
    assert path.stat().st_size < 2_000_000


def test_T038_false_positive_rate_over_independent_null_seeds() -> None:
    # A weaker economic filter (no hurdle) so the statistical gates carry the burden.
    cfg = dataclasses.replace(CFG, cost_hurdle=0.0, permutation_samples=199)
    qualified, pvalues = 0, []
    for seed in range(N_SEEDS):
        pair = independent_pair(seed=50_000 + seed, duration_s=900)
        res = discover_lead_lag(pair.x_ts_ns, pair.x_values, pair.y_ts_ns, pair.y_values, cfg)
        qualified += int(res.qualifies)
        pvalues.append(res.p_value)
    assert qualified / N_SEEDS <= cfg.alpha + FPR_MARGIN
    p = np.asarray(pvalues)
    assert np.isfinite(p).all()
    assert ((p > 0) & (p <= 1)).all()
    assert np.mean(p <= cfg.alpha) <= cfg.alpha + 0.15  # scan test roughly calibrated
