"""T031: the kill switch stops new paper orders and cancels outstanding simulated makers."""

from __future__ import annotations

import pytest

from cma.domain.enums import BookSide, OrderState, ReasonCode, Side, TimeInForce
from tests.factories import D, ScriptedStrategy, delta, ms, order, run_replay, snapshot

pytestmark = pytest.mark.integration
INST = "KALSHI:FIXTURE-BOOK"


def test_T031_kill_switch_stops_new_orders_and_cancels_resting_makers() -> None:
    events = [snapshot(INST, 1, ms(0), [("0.40", "100")], [("0.50", "100")])]
    events += [
        delta(INST, i + 2, ms(10 * (i + 1)), [(BookSide.BID, "0.40", "1")]) for i in range(60)
    ]

    def make_quotes(ctx):  # type: ignore[no-untyped-def]
        return [
            order(INST, Side.BUY, "10", "0.41", tif=TimeInForce.GTC),
            order(INST, Side.SELL, "10", "0.49", tif=TimeInForce.GTC),
        ]

    def kill(ctx):  # type: ignore[no-untyped-def]
        ctx.core.risk.activate_kill_switch(ctx.now_ns, "test")
        return []

    def try_again(ctx):  # type: ignore[no-untyped-def]
        return [order(INST, Side.BUY, "1", "0.50")]

    strat = ScriptedStrategy([(ms(0), make_quotes), (ms(200), kill), (ms(300), try_again)])
    core = run_replay(events, [strat], outbound_ms=20, cancel_ms=30)

    makers = [o for o in core.sim.orders.values() if o.tif is TimeInForce.GTC]
    assert len(makers) == 2
    assert all(o.state is OrderState.CANCELED for o in makers)
    assert core.risk.kill_switch_active
    assert core.records.risk_rejections[ReasonCode.KILL_SWITCH] == 1
    assert len(core.sim.orders) == 2  # nothing new reached the simulator after the switch
    assert not core.records.fills
    assert D(0) == sum((f.quantity for f in core.records.fills), D(0))
