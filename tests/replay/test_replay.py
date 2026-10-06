"""T032 replay determinism and T033 historical/production parity."""

from __future__ import annotations

import dataclasses

import pytest

from cma.backtest.core import StaticMappings
from cma.backtest.engine import ledger_bytes
from cma.backtest.experiment import DatasetSpec, StrategySpec, run_single
from cma.config import AppConfig, CostScenario
from cma.domain.enums import MappingStatus
from cma.domain.models import MarketEvent
from cma.domain.time import ManualClock
from cma.execution.latency import LatencyModel, LatencyProfile
from cma.execution.paper.executor import PaperTradingSession
from cma.research.synthetic import SyntheticMarketConfig, generate_market
from cma.signals.strategies.fair_value_taker import FairValueTakerConfig, FairValueTakerStrategy

pytestmark = pytest.mark.replay

SMALL = {"hours": 1, "n_strikes": 5, "seed": 11, "mm_lag_ms": 1500.0, "competitor_latency_ms": None}


def _cfg(mode: str = "BACKTEST", **sim: object) -> AppConfig:
    return AppConfig.model_validate(
        {"mode": mode, "signal": {"min_net_edge_bps": 50}, "simulation": dict(sim)}
    )


def test_T032_same_dataset_config_seed_gives_byte_identical_ledger() -> None:
    ds_spec = DatasetSpec("synthetic", SMALL)
    strat = StrategySpec("fv_taker", "1.0", {})
    cfg = _cfg(latency_jitter="lognormal", latency_jitter_sigma=0.4)
    outputs = []
    for _ in range(2):
        ds = ds_spec.load()
        r, core = run_single(
            ds, strat, cfg, latency_ms=250, cost=CostScenario(name="base"), seed=99
        )
        outputs.append(
            (ledger_bytes(core.records.fills), r.metrics.canonical_bytes(), r.ledger_hash)
        )
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]
    assert outputs[0][2] == outputs[1][2]
    assert outputs[0][0].count(b"\n") > 0, "scenario should trade so the comparison is meaningful"


def test_T032_different_seed_changes_jittered_latency_path() -> None:
    ds = DatasetSpec("synthetic", SMALL).load()
    strat = StrategySpec("fv_taker", "1.0", {})
    cfg = _cfg(latency_jitter="lognormal", latency_jitter_sigma=0.8)
    _a, core_a = run_single(ds, strat, cfg, latency_ms=250, cost=CostScenario(name="base"), seed=1)
    _b, core_b = run_single(ds, strat, cfg, latency_ms=250, cost=CostScenario(name="base"), seed=2)
    arrivals_a = [o.arrival_ts_ns - o.decision_ts_ns for o in core_a.records.orders]
    arrivals_b = [o.arrival_ts_ns - o.decision_ts_ns for o in core_b.records.orders]
    assert arrivals_a != arrivals_b


def _zero_delay(events: list[MarketEvent]) -> list[MarketEvent]:
    """Same information timing for both drivers: recv == venue time, strictly increasing."""
    out: list[MarketEvent] = []
    last = -1
    for ev in sorted(events, key=lambda e: (e.venue_ts_ns, e.recv_ts_ns)):
        t = max(ev.venue_ts_ns, last + 1)
        last = t
        out.append(dataclasses.replace(ev, source_ts_ns=t, recv_ts_ns=t))
    return out


def test_T033_replay_and_paper_paths_share_code_and_agree() -> None:
    market = generate_market(
        SyntheticMarketConfig(**SMALL, mapping_status=MappingStatus.APPROVED_PAPER)
    )
    market.events = _zero_delay(market.events)
    pairs = list(zip(market.contracts, market.mappings, strict=True))
    contracts = {c.contract_id: c for c in market.contracts}
    maps = StaticMappings({m.contract_id: m for m in market.mappings})
    refs = frozenset({market.reference_instrument})
    profile = LatencyProfile(outbound_ms=200, compute_ms=5, ack_ms=50)

    # historical replay
    from cma.backtest.core import TradingCore
    from cma.backtest.engine import ReplayEngine

    replay_strat = FairValueTakerStrategy(
        FairValueTakerConfig(reference_instrument=market.reference_instrument), pairs
    )
    replay_cfg = _cfg("BACKTEST")

    def factory(engine: ReplayEngine) -> TradingCore:
        return TradingCore(
            config=replay_cfg,
            strategies=[replay_strat],
            contracts=contracts,
            mappings=maps,
            latency=LatencyModel(profile, seed=5),
            scheduler=engine,
            reference_instruments=refs,
        )

    engine = ReplayEngine(factory, settlements=market.settlements, closes=market.closes)
    engine.run(market.events)
    replay_core = engine.core

    # forward paper path, fed event by event on a manual clock at receive time
    paper_strat = FairValueTakerStrategy(
        FairValueTakerConfig(reference_instrument=market.reference_instrument), pairs
    )
    clock = ManualClock()
    session = PaperTradingSession(
        config=_cfg("PAPER"),
        strategies=[paper_strat],
        contracts=contracts,
        mappings=maps,
        latency=LatencyModel(profile, seed=5),
        clock=clock,
        reference_instruments=refs,
        feed_allowance_ms=0,
        settlements=market.settlements,
        closes=market.closes,
    )
    for ev in market.events:
        clock.set(max(clock.now_ns(), ev.recv_ts_ns))
        session.on_event(ev)
    session.drain()
    paper_core = session.core

    assert type(paper_core) is type(replay_core)  # one implementation, two drivers
    assert [s.signal_id for s in paper_core.records.signals] == [
        s.signal_id for s in replay_core.records.signals
    ]
    assert ledger_bytes(paper_core.records.fills) == ledger_bytes(replay_core.records.fills)
    assert len(replay_core.records.signals) > 0
    assert paper_core.portfolio.cash == replay_core.portfolio.cash
