"""Discrete-event historical replay (scope s.11).

Input: normalized events in receive-time order (exactly what was recorded). Each event is
scheduled twice - at its venue time for the execution simulator and at its receive time
(+ optional extra feed delay) for the strategy - and merged with the core's own scheduled
actions in a single priority queue ordered by (time, ActionKind, sequence).

Because venue time <= receive time, an event can only be released once the input stream
has advanced past ``recv_ts - max_lateness``; events whose source timestamp lags by more
than ``max_lateness`` are applied late (counted) rather than reordering the past.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from cma.backtest.core import ActionKind, CoreRecords, TradingCore
from cma.domain.models import Fill, MarketEvent, Settlement
from cma.domain.time import NS_PER_S


@dataclass(order=True)
class _Item:
    ts_ns: int
    kind: int
    seq: int
    payload: Any = field(compare=False)


class ReplayEngine:
    def __init__(
        self,
        core_factory: Callable[[ReplayEngine], TradingCore],
        *,
        max_lateness_ns: int = 10 * NS_PER_S,
        extra_feed_ns: int = 0,
        timer_interval_ns: int = 60 * NS_PER_S,
        settlements: Sequence[Settlement] = (),
        closes: Sequence[tuple[int, str]] = (),
    ) -> None:
        self._heap: list[_Item] = []
        self._seq = 0
        self.now_ns = 0
        self.max_lateness_ns = max_lateness_ns
        self.extra_feed_ns = extra_feed_ns
        self.timer_interval_ns = timer_interval_ns
        self._last_input_ts: int | None = None
        self._timer_armed = False
        self.core = core_factory(self)
        for ts, contract_id in closes:
            self.schedule(ts, ActionKind.CLOSE, contract_id)
        for s in settlements:
            self.schedule(s.settled_ts_ns, ActionKind.SETTLE, s)

    # Scheduler protocol -------------------------------------------------------------
    def schedule(self, ts_ns: int, kind: ActionKind, payload: object) -> None:
        if ts_ns < self.now_ns:
            if kind is ActionKind.VENUE:
                self.core.records.late_venue_events += 1
            ts_ns = self.now_ns
        heapq.heappush(self._heap, _Item(ts_ns, int(kind), self._seq, payload))
        self._seq += 1

    # ---------------------------------------------------------------------------------
    def run(self, events: Iterable[MarketEvent]) -> CoreRecords:
        for ev in events:
            if self._last_input_ts is not None and ev.recv_ts_ns < self._last_input_ts:
                raise ValueError("replay input must be sorted by recv_ts_ns")
            self._last_input_ts = ev.recv_ts_ns
            if not self._timer_armed and self.timer_interval_ns > 0:
                self.schedule(ev.recv_ts_ns + self.timer_interval_ns, ActionKind.TIMER, None)
                self._timer_armed = True
            self._drain(ev.recv_ts_ns - self.max_lateness_ns)
            self.schedule(ev.venue_ts_ns, ActionKind.VENUE, ev)
            self.schedule(ev.recv_ts_ns + self.extra_feed_ns, ActionKind.OBSERVATION, ev)
        self._drain(None)
        if self._last_input_ts is not None:
            self.core.mark_to_market(max(self.now_ns, self._last_input_ts))
        return self.core.records

    def _drain(self, until_ns: int | None) -> None:
        heap = self._heap
        core = self.core
        while heap and (until_ns is None or heap[0].ts_ns <= until_ns):
            item = heapq.heappop(heap)
            self.now_ns = item.ts_ns
            kind = ActionKind(item.kind)
            if kind is ActionKind.VENUE:
                core.on_venue_event(item.payload, item.ts_ns)
            elif kind is ActionKind.OBSERVATION:
                core.on_observation(item.payload, item.ts_ns)
            elif kind is ActionKind.TIMER:
                core.on_action(kind, None, item.ts_ns)
                last = self._last_input_ts
                if last is not None and item.ts_ns < last:
                    self.schedule(item.ts_ns + self.timer_interval_ns, ActionKind.TIMER, None)
            else:
                core.on_action(kind, item.payload, item.ts_ns)


def fill_to_record(f: Fill) -> dict[str, Any]:
    return {
        "fill_id": f.fill_id,
        "order_id": f.order_id,
        "venue": f.venue.value,
        "contract_id": f.contract_id,
        "side": f.side.value,
        "ts_ns": f.fill_ts_ns,
        "price": str(f.price),
        "quantity": str(f.quantity),
        "fee": str(f.fee),
        "fee_schedule": f.fee_schedule_version,
        "role": f.liquidity_role.value,
        "strategy_id": f.strategy_id,
        "signal_id": f.signal_id,
    }


def ledger_bytes(fills: Sequence[Fill]) -> bytes:
    """Canonical serialization of a trade ledger (byte-comparable across runs)."""
    lines = [json.dumps(fill_to_record(f), sort_keys=True, separators=(",", ":")) for f in fills]
    return ("\n".join(lines) + "\n").encode()


def ledger_hash(fills: Sequence[Fill]) -> str:
    return hashlib.sha256(ledger_bytes(fills)).hexdigest()
