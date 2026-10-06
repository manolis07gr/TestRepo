"""H3 (implied probability) and H4 (nested strikes) strategies through the full replay core."""

from __future__ import annotations

from decimal import Decimal

import pytest

from cma.domain.enums import MappingStatus, Operator, SettlementOutcome, Venue
from cma.domain.models import ContractMapping, Settlement
from cma.domain.time import NS_PER_S
from cma.models.implied_probability.surface import ExpiryAlignment, OptionSlice, OptionSurface
from cma.signals.strategies.implied_prob import ImpliedProbabilityStrategy
from cma.signals.strategies.structural_arb import StructuralArbStrategy
from tests.factories import D, config, contract, ms, run_replay, snapshot

pytestmark = pytest.mark.integration
LOW, HIGH = "KALSHI:NEST-T100", "KALSHI:NEST-T101"
EXPIRY = ms(3_600_000)


def _gt(cid: str, strike: str) -> ContractMapping:
    return ContractMapping(
        venue=Venue.KALSHI,
        contract_id=cid,
        underlyings=("BTC-USD",),
        operator=Operator.GT,
        strikes=(D(strike),),
        observation_start_ns=None,
        observation_end_ns=EXPIRY,
        observation_method="POINT",
        timezone="America/New_York",
        resolution_source="TEST",
        event_family="BTC|POINT|test",
        review_status=MappingStatus.REVIEWED,
    )


def test_nested_strike_violation_traded_as_two_leg_package() -> None:
    contracts = [contract(LOW, resolve_ts_ns=EXPIRY), contract(HIGH, resolve_ts_ns=EXPIRY)]
    mappings = [_gt(LOW, "100000"), _gt(HIGH, "101000")]
    # P(X > 101k) bid 0.60 exceeds P(X > 100k) ask 0.55 by more than both taker fees
    events = [
        snapshot(LOW, 1, ms(0), [("0.53", "50")], [("0.55", "50")]),
        snapshot(HIGH, 1, ms(1), [("0.60", "50")], [("0.62", "50")]),
    ]
    strat = StructuralArbStrategy(list(zip(contracts, mappings, strict=True)))
    settlements = [
        Settlement(
            venue=Venue.KALSHI,
            contract_id=LOW,
            outcome=SettlementOutcome.YES,
            yes_value=D(1),
            settled_ts_ns=EXPIRY + NS_PER_S,
        ),
        Settlement(
            venue=Venue.KALSHI,
            contract_id=HIGH,
            outcome=SettlementOutcome.NO,
            yes_value=D(0),
            settled_ts_ns=EXPIRY + NS_PER_S,
        ),
    ]
    core = run_replay(
        events,
        [strat],
        contracts=contracts,
        mappings=mappings,
        settlements=settlements,
        outbound_ms=50,
    )
    assert strat.packages_sent == 1
    legs = {(f.contract_id, f.side.value, f.price) for f in core.records.fills}
    assert legs == {(LOW, "BUY", D("0.55")), (HIGH, "SELL", D("0.60"))}
    # the package pays >= 0 in every state; here the middle state paid 1 per unit
    assert core.portfolio.realized_total > 0
    assert core.portfolio.realized_total - core.portfolio.fees_total > 0


def test_implied_probability_strategy_signals_against_mispriced_book() -> None:
    cid = "KALSHI:IMPL-T100"
    c = contract(cid, resolve_ts_ns=EXPIRY)
    m = _gt(cid, "100000")
    surface = OptionSurface(
        [
            OptionSlice(
                expiry_ns=EXPIRY,
                forward=101_000.0,
                strikes=(95_000.0, 100_000.0, 105_000.0),
                ivs=(0.5, 0.5, 0.5),
                asof_ns=ms(0),
            )
        ]
    )
    strat = ImpliedProbabilityStrategy(
        [(c, m)],
        surface_provider=lambda now: (surface, ms(0)),
        alignment=ExpiryAlignment.exact(),
        uncertainty_bps=Decimal(100),
    )
    # implied P(S_T > 100k) is ~0.65 at F=101k, 1h, 50% vol; the book offers YES at 0.40
    events = [snapshot(cid, 1, ms(10), [("0.38", "100")], [("0.40", "100")])]
    core = run_replay(
        events,
        [strat],
        contracts=[c],
        mappings=[m],
        cfg=config(signal={"min_net_edge_bps": 100}),
        outbound_ms=50,
    )
    assert core.records.signals and core.records.signals[0].side.value == "BUY"
    assert core.records.fills and core.records.fills[0].price == D("0.40")
    # mismatched expiry without an explicit rule is skipped, never silently substituted
    strict = ImpliedProbabilityStrategy(
        [(c, m)],
        surface_provider=lambda now: (
            OptionSurface(
                [
                    OptionSlice(
                        expiry_ns=EXPIRY + 3600 * NS_PER_S,
                        forward=101_000.0,
                        strikes=(100_000.0,),
                        ivs=(0.5,),
                        asof_ns=ms(0),
                    )
                ]
            ),
            ms(0),
        ),
    )
    core2 = run_replay(events, [strict], contracts=[c], mappings=[m])
    assert strict.skipped_expiry == 1 and not core2.records.signals
