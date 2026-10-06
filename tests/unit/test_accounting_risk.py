"""T026-T030: settlement P&L (YES/NO), position accounting, exposure limits, daily stop."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from cma.config import RiskConfig
from cma.domain.binary import to_yes_order
from cma.domain.enums import (
    LiquidityRole,
    OrderType,
    Outcome,
    ReasonCode,
    SettlementOutcome,
    Side,
    TimeInForce,
    Venue,
)
from cma.domain.models import Fill, Settlement, SimOrder
from cma.domain.time import NS_PER_DAY
from cma.portfolio.ledger import Portfolio
from cma.risk.engine import OrderRequest, PositionView, RiskEngine
from tests.factories import FIXTURES, D

pytestmark = pytest.mark.unit
CID = "KALSHI:X"


def _fill(side: Side, price: str, qty: str, fee: str = "0", cid: str = CID, n: int = 0) -> Fill:
    return Fill(
        fill_id=f"f{n}-{side}-{price}-{qty}",
        order_id="o",
        venue=Venue.KALSHI,
        contract_id=cid,
        instrument_id=cid,
        side=side,
        fill_ts_ns=n,
        price=D(price),
        quantity=D(qty),
        fee=D(fee),
        fee_schedule_version="test@1",
        liquidity_role=LiquidityRole.TAKER,
    )


def _settlement(outcome: str, value: str, cid: str = CID, ts: int = 10) -> Settlement:
    return Settlement(
        venue=Venue.KALSHI,
        contract_id=cid,
        outcome=SettlementOutcome(outcome),
        yes_value=D(value),
        settled_ts_ns=ts,
    )


_CASES = json.loads((FIXTURES / "settlement_cases.json").read_text())


@pytest.mark.parametrize("case", [c for c in _CASES if c["side"] == "BUY"], ids=lambda c: c["name"])
def test_T026_settlement_pnl_yes_positions(case: dict) -> None:  # type: ignore[type-arg]
    pf = Portfolio(initial_cash=D(1000), void_policy=case.get("void_policy", "refund_cost"))
    pf.apply_fill(_fill(Side.BUY, case["price"], case["quantity"]))
    pnl = pf.settle(_settlement(case["outcome"], case["yes_value"]))
    assert pnl == D(case["expected_pnl"])
    assert pf.cash == D(1000) + D(case["expected_pnl"])
    assert pf.positions[CID].quantity == 0 and pf.positions[CID].settled


@pytest.mark.parametrize(
    "case", [c for c in _CASES if c["side"] == "BUY_NO"], ids=lambda c: c["name"]
)
def test_T027_settlement_pnl_no_positions_after_normalization(case: dict) -> None:  # type: ignore[type-arg]
    side, yes_price = to_yes_order(Outcome.NO, Side.BUY, D(case["price"]))
    pf = Portfolio(initial_cash=D(1000))
    pf.apply_fill(_fill(side, str(yes_price), case["quantity"]))
    pnl = pf.settle(_settlement(case["outcome"], case["yes_value"]))
    assert pnl == D(case["expected_pnl"])
    assert pf.cash == D(1000) + D(case["expected_pnl"])


def test_T026_settlement_is_idempotent_and_blocks_further_fills() -> None:
    pf = Portfolio(initial_cash=D(100))
    pf.apply_fill(_fill(Side.BUY, "0.4", "10"))
    assert pf.settle(_settlement("YES", "1")) == D(6)
    assert pf.settle(_settlement("YES", "1")) == D(0)
    with pytest.raises(ValueError, match="settled"):
        pf.apply_fill(_fill(Side.BUY, "0.4", "1"))


def test_T028_average_cost_realized_unrealized_and_fees_reconcile() -> None:
    pf = Portfolio(initial_cash=D(1000))
    pf.apply_fill(_fill(Side.BUY, "0.40", "10", "0.17", n=1))
    pf.apply_fill(_fill(Side.BUY, "0.50", "10", "0.18", n=2))
    pos = pf.positions[CID]
    assert pos.quantity == D(20) and pos.avg_cost == D("0.45")
    pf.apply_fill(_fill(Side.SELL, "0.60", "5", "0.08", n=3))  # reduce
    assert pf.realized_total == D("0.75")  # 5 * (0.60 - 0.45)
    assert pos.quantity == D(15) and pos.avg_cost == D("0.45")
    pf.apply_fill(_fill(Side.SELL, "0.55", "25", "0.20", n=4))  # close 15, flip short 10
    assert pf.realized_total == D("0.75") + D(15) * D("0.10")
    assert pos.quantity == D(-10) and pos.avg_cost == D("0.55")
    assert pf.fees_total == D("0.63")
    marks = {CID: D("0.52")}
    assert pf.unrealized(marks) == D(-10) * D("0.52") - D(-10) * D("0.55")
    assert abs(pf.reconcile(marks)) < D("1e-12")
    assert pf.nav(marks) == D(1000) + pf.realized_total - pf.fees_total + pf.unrealized(marks)
    # short worst case: pays 1 per contract above sale price
    assert pos.worst_case_loss() == D(10) * (1 - D("0.55"))
    assert pf.buying_power() == pf.cash - D(10)


def test_T028_reconciliation_after_settlement_of_mixed_book() -> None:
    pf = Portfolio(initial_cash=D(500))
    pf.apply_fill(_fill(Side.BUY, "0.30", "7", "0.15", cid="KALSHI:A", n=1))
    pf.apply_fill(_fill(Side.SELL, "0.80", "3", "0.05", cid="KALSHI:B", n=2))
    pf.apply_fill(_fill(Side.SELL, "0.35", "2", "0.02", cid="KALSHI:A", n=3))
    pf.settle(_settlement("NO", "0", cid="KALSHI:B"))
    marks = {"KALSHI:A": D("0.33")}
    assert abs(pf.reconcile(marks)) < D("1e-12")


def _risk(nav_pct: dict[str, str] | None = None) -> RiskEngine:
    return RiskEngine(RiskConfig(**(nav_pct or {})))  # type: ignore[arg-type]


def _req(side: Side, qty: str, price: str, cid: str = CID, fam: str = "F") -> OrderRequest:
    return OrderRequest(
        contract_id=cid, event_family=fam, side=side, quantity=D(qty), price=D(price), ts_ns=1
    )


def test_T029_contract_family_and_portfolio_limits() -> None:
    risk = _risk()
    nav = D(10_000)  # contract 2% = 200, family 5% = 500, total 10% = 1000
    assert risk.check_order(_req(Side.BUY, "500", "0.40"), nav=nav, positions={},
                            working_orders=[]).approved
    d = risk.check_order(_req(Side.BUY, "501", "0.40"), nav=nav, positions={}, working_orders=[])
    assert not d.approved and d.reasons == (ReasonCode.CONTRACT_LIMIT,)
    # selling YES (buying NO) risks (1 - p) per contract
    d = risk.check_order(_req(Side.SELL, "334", "0.40"), nav=nav, positions={}, working_orders=[])
    assert not d.approved and d.reasons == (ReasonCode.CONTRACT_LIMIT,)

    positions = {
        "KALSHI:A": PositionView(D(400), D(160), "F"),
        "KALSHI:B": PositionView(D(400), D(160), "F"),
    }
    d = risk.check_order(_req(Side.BUY, "500", "0.40", cid="KALSHI:C"), nav=nav,
                         positions=positions, working_orders=[])
    assert not d.approved and d.reasons == (ReasonCode.EVENT_LIMIT,)

    positions = {f"KALSHI:{i}": PositionView(D(450), D(180), f"F{i}") for i in range(5)}
    d = risk.check_order(_req(Side.BUY, "300", "0.40", cid="KALSHI:Z", fam="FZ"), nav=nav,
                         positions=positions, working_orders=[])
    assert not d.approved and d.reasons == (ReasonCode.PORTFOLIO_LIMIT,)


def test_T029_working_orders_count_toward_exposure() -> None:
    risk = _risk()
    working = [
        SimOrder(
            order_id="w1",
            venue=Venue.KALSHI,
            contract_id=CID,
            instrument_id=CID,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=D(400),
            tif=TimeInForce.GTC,
            submit_ts_ns=0,
            arrival_ts_ns=0,
            limit_price=D("0.40"),
        )
    ]
    d = risk.check_order(_req(Side.BUY, "101", "0.40"), nav=D(10_000), positions={},
                         working_orders=working)
    assert not d.approved and d.reasons == (ReasonCode.CONTRACT_LIMIT,)


def test_T030_daily_stop_blocks_new_risk_but_allows_reduction() -> None:
    risk = _risk()
    day = 20_000 * NS_PER_DAY
    risk.on_nav(day + 1, D(10_000))
    risk.on_nav(day + 2, D("9801"))
    assert not risk.daily_stop_active
    risk.on_nav(day + 3, D("9800"))  # -2.00% -> stop
    assert risk.daily_stop_active
    positions = {CID: PositionView(D(100), D(40), "F")}
    d = risk.check_order(_req(Side.BUY, "10", "0.40"), nav=D(9800), positions=positions,
                         working_orders=[])
    assert not d.approved and d.reasons == (ReasonCode.DAILY_STOP,)
    d = risk.check_order(_req(Side.SELL, "50", "0.38"), nav=D(9800), positions=positions,
                         working_orders=[])
    assert d.approved and d.detail == "risk-reducing"
    risk.on_nav(day + NS_PER_DAY + 1, D(9800))  # next UTC day resets
    assert not risk.daily_stop_active


def test_T030_kill_switch_blocks_everything_and_calls_cancel_all() -> None:
    risk = _risk()
    calls: list[int] = []
    risk.register_cancel_all(calls.append)
    risk.activate_kill_switch(42, "manual")
    assert calls == [42]
    positions = {CID: PositionView(D(100), D(40), "F")}
    d = risk.check_order(_req(Side.SELL, "50", "0.38"), nav=D(9800), positions=positions,
                         working_orders=[])
    assert not d.approved and d.reasons == (ReasonCode.KILL_SWITCH,)
    with pytest.raises(ValueError, match="operator"):
        risk.reset_kill_switch(43, "")
    risk.reset_kill_switch(43, "ops")
    assert not risk.kill_switch_active
    assert Decimal(0) == Decimal(0)
