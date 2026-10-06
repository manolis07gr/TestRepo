"""T049-T052, T055: publication completeness, cost stress, OOS/paper gates, manifests."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from cma.backtest.experiment import (
    DatasetSpec,
    RunResult,
    StrategySpec,
    StressGrid,
    build_manifest,
    reproduce_from_manifest,
    run_single,
    run_stress_grid,
    validate_publication,
)
from cma.backtest.metrics import TradingMetrics
from cma.config import MANDATORY_LATENCY_GRID_MS, AppConfig, CostScenario
from cma.domain.enums import Decision, Side, TimeInForce
from cma.domain.errors import ExperimentPublicationError
from cma.research.gates import (
    NetPnlCriterion,
    PromotionError,
    PromotionPolicy,
    assert_paper_complete,
    assert_promotable_to_paper,
    decide,
)
from tests.factories import D, ScriptedStrategy, contract, ms, order, run_replay, snapshot, trade

pytestmark = pytest.mark.acceptance

TINY = {"hours": 1, "n_strikes": 3, "seed": 5, "mm_lag_ms": 1500.0, "competitor_latency_ms": None}


def _metrics(net: float, n_positions: int = 50, **conc: float) -> TradingMetrics:
    return TradingMetrics(
        gross_pnl=net + 10,
        fees=10,
        net_pnl=net,
        slippage_cost=0,
        n_signals=100,
        n_orders=100,
        n_fills=80,
        n_positions=n_positions,
        filled_contracts=800,
        fill_rate=0.8,
        order_fill_ratio=0.8,
        hit_rate=0.55,
        profit_factor=1.2,
        mean_position_pnl=net / max(n_positions, 1),
        median_position_pnl=0.1,
        max_drawdown=5,
        sharpe_daily=1.0,
        turnover=400,
        avg_net_edge_bps=150,
        median_net_edge_bps=140,
        maker_share=0,
        time_in_market_frac=0.5,
        tail_loss_cvar5=-3,
        concentration={
            "top_contract_share_of_gains": conc.get("c", 0.1),
            "top_family_share_of_gains": conc.get("f", 0.2),
            "top_day_share_of_gains": conc.get("d", 0.3),
        },
    )


def _grid(net_by_cell: dict[tuple[int, str], float], **kw: float) -> StressGrid:
    return StressGrid(
        [
            RunResult(lat, cost, _metrics(v, **kw), f"h{lat}{cost}", 0)
            for (lat, cost), v in net_by_cell.items()
        ]
    )


def _full_grid(net: float) -> StressGrid:
    cells = {}
    for cost in ("base", "fees_x1.5", "slip_+1tick"):
        for lat in MANDATORY_LATENCY_GRID_MS:
            cells[(lat, cost)] = net
    return _grid(cells)


MANIFEST_OK = {
    "git_commit": "abc",
    "config_hash": "c",
    "dataset": {"kind": "synthetic"},
    "dataset_hash": "d",
    "strategy": {"name": "fv_taker"},
    "seed": 1,
}


def test_T049_missing_latency_scenario_blocks_publication() -> None:
    validate_publication(_full_grid(5.0), MANIFEST_OK)
    partial = _grid({(lat, "base"): 1.0 for lat in MANDATORY_LATENCY_GRID_MS if lat != 2000})
    with pytest.raises(ExperimentPublicationError, match="2000"):
        validate_publication(partial, MANIFEST_OK)


def test_T050_cost_stress_requires_base_plus_two_adverse_scenarios() -> None:
    only_one = _grid(
        {(lat, c): 1.0 for lat in MANDATORY_LATENCY_GRID_MS for c in ("base", "fees_x1.5")}
    )
    with pytest.raises(ExperimentPublicationError, match="cost stress"):
        validate_publication(only_one, MANIFEST_OK)
    cfg = AppConfig()
    adverse = [c for c in cfg.research.cost_scenarios if c.name != "base"]
    assert len(adverse) >= 2
    assert all(c.fee_multiplier >= 1 and c.extra_slippage_ticks >= 0 for c in adverse)


def test_T050_higher_fees_with_identical_fills_never_improve_net_pnl() -> None:
    inst = "KALSHI:FIXTURE-BOOK"
    events = [snapshot(inst, 1, ms(0), [("0.40", "100")], [("0.45", "100")])]
    events.append(trade(inst, ms(5000), "0.44", "1", Side.BUY))
    script = [(ms(0), lambda ctx: [order(inst, Side.BUY, "20", "0.45", tif=TimeInForce.IOC)])]
    nets = []
    for mult in ("1", "1.5", "3"):
        core = run_replay(
            events,
            [ScriptedStrategy(list(script))],
            contracts=[contract(inst)],
            fee_multiplier=Decimal(mult),
        )
        fills = [(f.price, f.quantity) for f in core.records.fills]
        marks = {inst: D("0.42")}
        nets.append((fills, core.portfolio.nav(marks)))
    assert nets[0][0] == nets[1][0] == nets[2][0]
    assert nets[0][1] > nets[1][1] > nets[2][1]


def test_T051_candidate_requires_positive_untouched_test() -> None:
    policy = PromotionPolicy(criterion=NetPnlCriterion(latency_ms=250, cost="base"))
    neg = decide(
        final_test=_full_grid(-12.0),
        policy=policy,
        sensitivity_net_pnls=[1, 2, 3],
        mappings_approved_paper=True,
    )
    assert neg.decision is not Decision.FORWARD_PAPER_CANDIDATE
    assert neg.decision is Decision.REJECT
    assert not neg.gate("oos_net_pnl").passed
    pos = decide(
        final_test=_full_grid(25.0),
        policy=policy,
        sensitivity_net_pnls=[1, 2, 3],
        mappings_approved_paper=True,
    )
    assert pos.decision is Decision.FORWARD_PAPER_CANDIDATE
    # positive at base but not under worse latency -> not viable
    cells = {
        (lat, c): (25.0 if lat <= 250 else -5.0)
        for lat in MANDATORY_LATENCY_GRID_MS
        for c in ("base", "fees_x1.5", "slip_+1tick")
    }
    fragile = decide(final_test=_grid(cells), policy=policy, mappings_approved_paper=True)
    assert fragile.decision is Decision.REJECT
    assert not fragile.gate("stress_latency_viable").passed
    # too few positions -> collect more data rather than a verdict
    thin = decide(
        final_test=_grid(dict.fromkeys(cells, 5.0), n_positions=5),
        policy=policy,
        mappings_approved_paper=True,
    )
    assert thin.decision is Decision.COLLECT_MORE_DATA


def test_T052_paper_window_must_complete_before_promotion() -> None:
    policy = PromotionPolicy(min_forward_paper_days=14)
    kw = {"final_test": _full_grid(25.0), "policy": policy, "mappings_approved_paper": True}
    fresh = decide(**kw)
    assert fresh.decision is Decision.FORWARD_PAPER_CANDIDATE
    assert_promotable_to_paper(fresh)
    with pytest.raises(PromotionError):
        assert_paper_complete(fresh)
    midway = decide(**kw, paper_days_completed=6.5)
    assert midway.decision is Decision.CONTINUE_PAPER
    with pytest.raises(PromotionError, match=r"6\.5 of 14"):
        assert_paper_complete(midway)
    done = decide(**kw, paper_days_completed=14.2)
    assert_paper_complete(done)
    rejected = decide(final_test=_full_grid(-1.0), policy=policy, mappings_approved_paper=True)
    with pytest.raises(PromotionError):
        assert_promotable_to_paper(rejected)


def test_T055_manifest_identifies_inputs_and_reruns_identically(tmp_path: Path) -> None:
    cfg = AppConfig.model_validate({"mode": "BACKTEST", "signal": {"min_net_edge_bps": 50}})
    ds_spec = DatasetSpec("synthetic", TINY)
    strat = StrategySpec("fv_taker", "1.0", {})
    ds = ds_spec.load()
    grid = run_stress_grid(
        ds, strat, cfg, latencies_ms=[0, 500], costs=[CostScenario(name="base")], seed=3
    )
    manifest = build_manifest(
        experiment_id="exp-test",
        created_ns=1,
        cfg=cfg,
        dataset=ds_spec,
        dataset_hash=ds.dataset_hash,
        strategy=strat,
        seed=3,
        grid=grid,
    )
    for key in ("git_commit", "config_hash", "dataset_hash", "strategy", "seed", "environment"):
        assert manifest[key]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, default=str))
    reloaded = json.loads(path.read_text())
    assert reproduce_from_manifest(reloaded) == {"base@0": True, "base@500": True}
    assert any(r.metrics.n_fills for r in grid.results)


def test_T055_single_run_hash_matches_grid_cell() -> None:
    cfg = AppConfig.model_validate({"mode": "BACKTEST", "signal": {"min_net_edge_bps": 50}})
    ds = DatasetSpec("synthetic", TINY).load()
    strat = StrategySpec("fv_taker", "1.0", {})
    a, _ = run_single(ds, strat, cfg, latency_ms=500, cost=CostScenario(name="base"), seed=3)
    b, _ = run_single(ds, strat, cfg, latency_ms=500, cost=CostScenario(name="base"), seed=3)
    assert a.ledger_hash == b.ledger_hash
