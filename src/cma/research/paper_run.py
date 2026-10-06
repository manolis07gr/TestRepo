"""Continuous forward paper trading against live venue data (requires network access).

Wiring: live collectors -> normalized events -> PaperTradingSession (same TradingCore as
replay) -> persisted fills/state + health. Only contracts whose mappings are
APPROVED_PAPER are tradeable; the session refuses any live configuration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from cma.config import AppConfig
from cma.domain.enums import ExecutionMode
from cma.domain.time import SystemClock
from cma.execution.latency import LatencyModel, LatencyProfile
from cma.execution.paper.executor import PaperTradingSession
from cma.ingestion.collector import build_collector
from cma.mapping.registry import MappingRegistry
from cma.signals.strategies.fair_value_taker import FairValueTakerConfig, FairValueTakerStrategy
from cma.storage.contracts import ContractStore
from cma.storage.db import open_database


def run_paper(
    cfg: AppConfig,
    *,
    duration_s: float,
    mappings_dir: Path,
    reference_instrument: str = "COINBASE:BTC-USD",
    run_id: str | None = None,
    status_every_s: float = 30.0,
) -> dict[str, Any]:
    clock = SystemClock()
    registry = MappingRegistry.from_yaml(mappings_dir, clock=clock)
    db = open_database(cfg.storage.db_url)
    contracts = {
        c.contract_id: c
        for c in ContractStore(db).load()
        if registry.can_trade(c.contract_id, ExecutionMode.PAPER)
    }
    pairs = [(c, m) for cid, c in contracts.items() if (m := registry.get(cid)) is not None]
    if not pairs:
        return {
            "status": "NO_APPROVED_PAPER_MAPPINGS",
            "detail": "run `cma collect` to discover contracts, then review/approve mappings",
        }
    strategy = FairValueTakerStrategy(
        FairValueTakerConfig(reference_instrument=reference_instrument), pairs
    )
    latency_ms = cfg.simulation.latency_ms[0] if cfg.simulation.latency_ms else 250
    session = PaperTradingSession(
        config=cfg,
        strategies=[strategy],
        contracts=contracts,
        mappings=registry,
        latency=LatencyModel(LatencyProfile.from_config(cfg.simulation, latency_ms)),
        clock=clock,
        reference_instruments=frozenset({reference_instrument}),
        db=db,
        run_id=run_id or f"paper-{clock.now_ns()}",
    )
    collector = build_collector(cfg, db=db, on_events=session.on_events, clock=clock)
    snapshots: list[dict[str, Any]] = []

    async def main() -> None:
        await collector.start()
        try:
            elapsed = 0.0
            next_status = status_every_s
            while elapsed < duration_s:
                await asyncio.sleep(1.0)
                elapsed += 1.0
                session.tick()
                if elapsed >= next_status:
                    session.persist()
                    snapshots.append(
                        {"collector": collector.health_status().value, **session.health()}
                    )
                    next_status += status_every_s
        finally:
            await collector.stop()
            await collector.aclose()
            session.persist()

    asyncio.run(main())
    return {"final": session.health(), "snapshots": snapshots[-5:]}
