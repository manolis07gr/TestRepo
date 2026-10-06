"""Forward paper trading: live events -> the same TradingCore as historical replay.

Ordering mirrors the replay engine so identical inputs give identical decisions (T033):
for each incoming event, scheduled actions strictly before its venue time run first, the
event is applied to the simulated venue, actions that precede an observation at this
receive time run, then strategies observe it. ``tick(now)`` releases remaining due actions
(e.g. order arrivals when the market is quiet) once ``now`` passes them plus an allowance
for feed delay, so an order never meets a venue book we have not seen yet.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from cma.backtest.core import ActionKind, MappingProvider, Strategy, TradingCore
from cma.backtest.engine import fill_to_record
from cma.config import AppConfig
from cma.domain.enums import ExecutionMode, Venue
from cma.domain.errors import ConfigError, LiveTradingDisabledError
from cma.domain.models import Fill, MarketEvent, PredictionContract, Settlement
from cma.domain.time import NS_PER_MS, NS_PER_S, Clock
from cma.execution.latency import LatencyModel
from cma.storage.db import Database


@dataclass(order=True)
class _Item:
    ts_ns: int
    kind: int
    seq: int
    payload: Any = field(compare=False)


class PaperScheduler:
    def __init__(self) -> None:
        self._heap: list[_Item] = []
        self._seq = 0

    def schedule(self, ts_ns: int, kind: ActionKind, payload: object) -> None:
        heapq.heappush(self._heap, _Item(ts_ns, int(kind), self._seq, payload))
        self._seq += 1

    def pop_until(self, ts_ns: int, kind_limit: int) -> _Item | None:
        """Pop the next item strictly before (ts_ns, kind_limit), else None."""
        if self._heap and (self._heap[0].ts_ns, self._heap[0].kind) < (ts_ns, kind_limit):
            return heapq.heappop(self._heap)
        return None

    def __len__(self) -> int:
        return len(self._heap)


class PaperTradingSession:
    def __init__(
        self,
        *,
        config: AppConfig,
        strategies: Sequence[Strategy],
        contracts: Mapping[str, PredictionContract],
        mappings: MappingProvider,
        latency: LatencyModel,
        clock: Clock,
        reference_instruments: frozenset[str] = frozenset(),
        run_id: str = "paper",
        db: Database | None = None,
        feed_allowance_ms: int = 1_000,
        timer_interval_ms: int = 60_000,
        settlements: Sequence[Settlement] = (),
        closes: Sequence[tuple[int, str]] = (),
        venue_taker_delay_ms: Mapping[Venue, int] | None = None,
    ) -> None:
        if config.live_execution_enabled or config.mode is ExecutionMode.LIVE:
            raise LiveTradingDisabledError("paper sessions refuse live configuration")
        if config.mode is not ExecutionMode.PAPER:
            raise ConfigError(f"paper session requires mode PAPER, got {config.mode}")
        self.clock = clock
        self.run_id = run_id
        self.db = db
        self.feed_allowance_ns = feed_allowance_ms * NS_PER_MS
        self.timer_interval_ns = timer_interval_ms * NS_PER_MS
        self.scheduler = PaperScheduler()
        self.core = TradingCore(
            config=config,
            strategies=strategies,
            contracts=contracts,
            mappings=mappings,
            latency=latency,
            scheduler=self.scheduler,
            mode=ExecutionMode.PAPER,
            reference_instruments=reference_instruments,
            venue_taker_delay_ms=venue_taker_delay_ms,
        )
        for ts, cid in closes:
            self.scheduler.schedule(ts, ActionKind.CLOSE, cid)
        for s in settlements:
            self.scheduler.schedule(s.settled_ts_ns, ActionKind.SETTLE, s)
        self._timer_armed = False
        self._persisted_fills = 0
        self._last_event_ts = 0
        self.events_seen = 0

    # ------------------------------------------------------------------ driving

    def _run(self, until_ts: int, kind_limit: int) -> None:
        while True:
            item = self.scheduler.pop_until(until_ts, kind_limit)
            if item is None:
                return
            kind = ActionKind(item.kind)
            self.core.on_action(kind, item.payload, item.ts_ns)
            if kind is ActionKind.TIMER and item.ts_ns < max(
                self._last_event_ts, self.clock.now_ns()
            ):
                self.scheduler.schedule(item.ts_ns + self.timer_interval_ns, ActionKind.TIMER, None)

    def on_event(self, event: MarketEvent, now_ns: int | None = None) -> None:
        now = self.clock.now_ns() if now_ns is None else now_ns
        if not self._timer_armed and self.timer_interval_ns > 0:
            self.scheduler.schedule(
                event.recv_ts_ns + self.timer_interval_ns, ActionKind.TIMER, None
            )
            self._timer_armed = True
        self._run(event.venue_ts_ns, int(ActionKind.VENUE))
        self.core.on_venue_event(event, event.venue_ts_ns)
        self._run(now, int(ActionKind.OBSERVATION))
        self.core.on_observation(event, now)
        self._last_event_ts = max(self._last_event_ts, event.recv_ts_ns)
        self.events_seen += 1

    def on_events(self, events: Iterable[MarketEvent]) -> None:
        for ev in events:
            self.on_event(ev)
        self.persist()

    def tick(self, now_ns: int | None = None) -> None:
        """Release due actions; venue-facing ones only after the feed allowance."""
        now = self.clock.now_ns() if now_ns is None else now_ns
        self._run(now - self.feed_allowance_ns + 1, 1 << 30)

    def drain(self) -> None:
        """Process everything still scheduled (end of session / tests)."""
        self._run(1 << 62, 1 << 30)

    # ------------------------------------------------------------------ persistence

    def persist(self) -> int:
        """Append new fills and a state snapshot to the metadata DB (restart recovery)."""
        if self.db is None:
            return 0
        fills: list[Fill] = self.core.records.fills[self._persisted_fills :]
        with self.db.transaction():
            for f in fills:
                rec = fill_to_record(f)
                self.db.execute(
                    "INSERT OR IGNORE INTO paper_fills (run_id, fill_id, order_id, venue, "
                    "contract_id, side, fill_ts_ns, price, quantity, fee, fee_schedule_version, "
                    "liquidity_role) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        self.run_id,
                        rec["fill_id"],
                        rec["order_id"],
                        rec["venue"],
                        rec["contract_id"],
                        rec["side"],
                        rec["ts_ns"],
                        rec["price"],
                        rec["quantity"],
                        rec["fee"],
                        rec["fee_schedule"],
                        rec["role"],
                    ),
                )
            state = self.state_snapshot()
            self.db.execute(
                "INSERT INTO paper_state (run_id, state_key, state_json, updated_at_ns) "
                "VALUES (?, 'portfolio', ?, ?) ON CONFLICT (run_id, state_key) DO UPDATE SET "
                "state_json = excluded.state_json, updated_at_ns = excluded.updated_at_ns",
                (self.run_id, json.dumps(state, sort_keys=True), self.clock.now_ns()),
            )
        self._persisted_fills += len(fills)
        return len(fills)

    def state_snapshot(self) -> dict[str, Any]:
        pf = self.core.portfolio
        return {
            "cash": str(pf.cash),
            "realized": str(pf.realized_total),
            "fees": str(pf.fees_total),
            "positions": {
                cid: {"quantity": str(p.quantity), "cost_basis": str(p.cost_basis)}
                for cid, p in pf.positions.items()
                if p.quantity != 0
            },
            "kill_switch": self.core.risk.kill_switch_active,
            "events_seen": self.events_seen,
        }

    def health(self) -> dict[str, Any]:
        core = self.core
        return {
            "run_id": self.run_id,
            "status": "KILLED" if core.risk.kill_switch_active else "OK",
            "events_seen": self.events_seen,
            "pending_actions": len(self.scheduler),
            "signals": len(core.records.signals),
            "orders": len(core.records.orders),
            "fills": len(core.records.fills),
            "suppressions": {k.value: v for k, v in core.signal_engine.suppressions.items()},
            "risk_rejections": {k.value: v for k, v in core.records.risk_rejections.items()},
            "nav": str(core.nav_estimate()),
            "daily_stop": core.risk.daily_stop_active,
        }


def restore_portfolio_state(db: Database, run_id: str) -> dict[str, Any] | None:
    """Load the last persisted paper state for ``run_id`` (None when absent)."""
    rows = db.query(
        "SELECT state_json FROM paper_state WHERE run_id = ? AND state_key = 'portfolio'",
        (run_id,),
    )
    if not rows:
        return None
    state: dict[str, Any] = json.loads(rows[0]["state_json"])
    state["persisted_fills"] = int(
        db.scalar("SELECT COUNT(*) FROM paper_fills WHERE run_id = ?", (run_id,)) or 0
    )
    state["cash"] = Decimal(state["cash"])
    return state


def paper_days(first_ts_ns: int, last_ts_ns: int) -> float:
    return max(0.0, (last_ts_ns - first_ts_ns) / (24 * 3600 * NS_PER_S))
