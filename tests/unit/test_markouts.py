"""Mark-outs stop at the contract close: a horizon ending after close has no market mid."""

from __future__ import annotations

import pytest

from cma.backtest.metrics import compute_metrics
from cma.domain.enums import SettlementOutcome, Venue
from cma.domain.models import Settlement
from tests.factories import D, ScriptedStrategy, buy_ioc, contract, ms, run_replay, snapshot

pytestmark = pytest.mark.unit
CID = "KALSHI:MARKOUT-CLOSE"


def test_markout_horizons_after_close_are_not_marked() -> None:
    close = ms(10_000)
    strat = ScriptedStrategy([(0, lambda ctx: [buy_ioc(CID, "5", "0.45")])])
    events = [
        snapshot(CID, 1, ms(0), [("0.40", "50")], [("0.42", "50")]),
        snapshot(CID, 2, ms(3_000), [("0.48", "50")], [("0.50", "50")]),
    ]
    core = run_replay(
        events,
        [strat],
        contracts=[contract(CID, resolve_ts_ns=close)],
        settlements=[
            Settlement(
                venue=Venue.KALSHI,
                contract_id=CID,
                outcome=SettlementOutcome.YES,
                yes_value=D(1),
                settled_ts_ns=close,
            )
        ],
        outbound_ms=50,
    )
    (fill,) = core.records.fills
    marks = {
        h: core.records.markouts.get((fill.fill_id, h)) for h in (1_000, 5_000, 30_000, 60_000)
    }
    assert marks[1_000] == D("0.41")  # mid of the first book
    assert marks[5_000] == D("0.49")
    # 30 s / 60 s end after the 10 s close: the outcome is known, so no mark-out
    assert marks[30_000] is None and marks[60_000] is None
    m = compute_metrics(core)
    assert m.markout_coverage["5000ms"] == 1.0 and m.markout_coverage["60000ms"] == 0.0
