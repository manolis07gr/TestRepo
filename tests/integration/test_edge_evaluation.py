"""Edge-evaluation diagnostics on a small synthetic market through the full replay core."""

from __future__ import annotations

import pytest

from cma.backtest.experiment import run_single
from cma.config import CostScenario
from cma.research.evaluation import (
    _SHARED,
    BASE_STRATEGY,
    _cfg,
    edge_attribution,
    ex_ante_edge,
    synthetic_dataset,
)
from cma.research.report import positive_maker_lags

pytestmark = pytest.mark.integration


def test_attribution_reconciles_with_ex_ante_edge() -> None:
    params = {"hours": 1, "seed": 11, "n_strikes": 5, "mm_lag_ms": 3000.0}
    ds, market = synthetic_dataset(params)
    try:
        _r, core = run_single(
            ds, BASE_STRATEGY, _cfg(100), latency_ms=100, cost=CostScenario(name="base"), seed=1
        )
    finally:
        _SHARED.clear()
    assert core.records.fills, "slow makers must leave something to take"
    ex = ex_ante_edge(core, market)
    attr = edge_attribution(core, market)
    filled = float(sum(f.quantity for f in core.records.fills))
    for table in ("by_perceived_edge", "by_time_to_expiry"):
        rows = attr[table]
        assert sum(r["contracts"] for r in rows) == pytest.approx(filled)
        weighted = sum(r["contracts"] * r["true_c_per_contract"] for r in rows) / filled
        assert weighted == pytest.approx(ex["expected_c_per_contract"], abs=1e-9)
    # every fill came from a signal that cleared the 1 cent net-edge threshold
    assert all(r["perceived_c_per_contract"] >= 1.0 - 1e-9 for r in attr["by_perceived_edge"])


def test_positive_maker_lags_filters_by_latency_and_competition() -> None:
    def row(mm: float, comp: float | None, lat: int, v: float) -> dict[str, object]:
        return {
            "mm_lag_ms": mm,
            "competitor_ms": comp,
            "latency_ms": lat,
            "model": "implied_vol",
            "expected_c_per_contract": v,
        }

    frontier = [
        row(350, None, 250, 0.1),
        row(350, 120, 250, -0.5),
        row(1000, 120, 250, 0.2),
        row(3000, 120, 250, 0.8),
        row(3000, 120, 500, -0.1),
        {**row(350, 120, 250, 5.0), "model": "realized_vol"},
    ]
    assert positive_maker_lags(frontier, 250, "any") == [1000.0, 3000.0]
    assert positive_maker_lags(frontier, 250, None) == [350.0]
    assert positive_maker_lags(frontier, 500, "any") == []
