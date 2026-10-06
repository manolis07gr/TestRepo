"""YES/NO complementarity and cross-venue arbitrage on equivalence-verified pairs (H2/H4).

All expected numbers are hand-computed; fees use the fixtures' quadratic model
ceil_to_cent(0.07 x C x P x (1 - P)) at the canonical YES price, or zero fees.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from cma.domain.enums import MappingStatus, Operator, Outcome, QualityFlag, Side, Venue
from cma.domain.errors import MappingNotApprovedError
from cma.domain.models import BookSnapshot, ContractMapping
from cma.mapping import MappingRegistry
from cma.models.structural import (
    ContractQuote,
    CrossVenueCosts,
    OpportunityKind,
    StructuralConfig,
    StructuralInputError,
    detect_complement_mispricing,
    detect_cross_venue_arbitrage,
)
from cma.models.structural.types import YEAR_NS
from tests.unit.test_mapping_fixtures import (
    D,
    FixtureQuadraticFee,
    contract_from_dict,
    fixture_mapping,
    load_fixture,
    make_book,
    review_through_registry,
)

pytestmark = pytest.mark.unit

FEE = FixtureQuadraticFee(D("0.07"))
NO_FEE = FixtureQuadraticFee(D("0"))
FX = load_fixture("equivalence_good_bad.json")
# 8.76 h = 365 d / 1000, so the lock-up year fraction is exactly 0.001
COSTS = CrossVenueCosts(
    settlement_ts_a_ns=0,
    settlement_ts_b_ns=YEAR_NS // 1000,
    annual_capital_rate=D("0.05"),
    resolution_risk_buffer=D("0.005"),
)


def reviewed(pair: str) -> tuple[ContractMapping, ContractMapping]:
    registry = MappingRegistry()
    a, b = (
        review_through_registry(
            registry,
            fixture_mapping(FX[pair][side]["mapping"]),
            contract_from_dict(FX[pair][side]["contract"]),
        )
        for side in ("a", "b")
    )
    return a, b


GOOD_A, GOOD_B = reviewed("good_pair")
BAD_A, BAD_B = reviewed("bad_pair")


def quote(
    mapping: ContractMapping, bids: list[tuple[str, str]], asks: list[tuple[str, str]]
) -> ContractQuote:
    return ContractQuote(
        mapping=mapping,
        book=make_book(mapping.contract_id, bids, asks, venue=mapping.venue),
        fee_schedule=FEE,
    )


# ---------------------------------------------------------------------- cross venue


def test_cross_venue_same_relation_exact() -> None:
    a = quote(GOOD_A, [("0.47", "100")], [("0.50", "60")])  # Kalshi
    b = quote(GOOD_B, [("0.56", "40")], [("0.58", "100")])  # Polymarket (synthetic twin)
    (opp,) = detect_cross_venue_arbitrage(a, b, COSTS)
    assert opp.kind is OpportunityKind.CROSS_VENUE
    # buy YES on Kalshi at 0.50, buy NO on Polymarket at 1 - 0.56 = 0.44; Q = min(60, 40)
    assert [(leg.venue, leg.side, leg.price, leg.quantity) for leg in opp.legs] == [
        (Venue.KALSHI, Side.BUY, D("0.50"), D("40")),
        (Venue.POLYMARKET, Side.SELL, D("0.56"), D("40")),
    ]
    # fees: ceil(0.07*40*0.25 = 0.70) ; ceil(0.07*40*0.56*0.44 = 0.68992) = 0.69
    assert [leg.fee for leg in opp.legs] == [D("0.70"), D("0.69")]
    assert opp.gross_profit == D("2.40")  # 40 * (0.56 - 0.50)
    assert opp.total_fees == D("1.39")
    # other costs per unit: buffer 0.005 + 0.05 * 0.001 * outlay (0.50 + 0.44) = 0.005047
    assert opp.other_costs_per_unit == D("0.005047")
    assert opp.other_costs == D("0.20188")
    assert opp.net_profit == D("0.80812")  # 2.40 - 1.39 - 0.20188
    assert opp.net_edge_per_unit == D("0.020203")
    # canonical-book leg: a YES sale at 0.56, economically a NO purchase costing 0.44
    assert opp.legs[1].token_side is Side.SELL
    assert opp.legs[1].token_price == D("0.56")
    assert sum(leg.cash_outlay for leg in opp.legs) == D("0.50") * 40 + D("0.44") * 40 + D("1.39")


def test_cross_venue_costs_can_kill_the_edge() -> None:
    a = quote(GOOD_A, [("0.47", "100")], [("0.50", "60")])
    b = quote(GOOD_B, [("0.56", "40")], [("0.58", "100")])
    big_buffer = dataclasses.replace(COSTS, resolution_risk_buffer=D("0.03"))
    assert detect_cross_venue_arbitrage(a, b, big_buffer) == []
    assert (
        detect_cross_venue_arbitrage(a, b, COSTS, config=StructuralConfig(tolerance=D("0.0203")))
        == []
    )
    assert (
        len(
            detect_cross_venue_arbitrage(
                a, b, COSTS, config=StructuralConfig(tolerance=D("0.0202"))
            )
        )
        == 1
    )


def test_cross_venue_reverse_direction() -> None:
    a = quote(GOOD_A, [("0.60", "25")], [("0.62", "60")])
    b = quote(GOOD_B, [("0.50", "40")], [("0.52", "30")])
    (opp,) = detect_cross_venue_arbitrage(a, b, COSTS)
    assert [(leg.contract_id, leg.side) for leg in opp.legs] == [
        (GOOD_B.contract_id, Side.BUY),
        (GOOD_A.contract_id, Side.SELL),
    ]
    assert opp.quantity == D("25")


def test_cross_venue_complement_relation_buys_yes_on_both() -> None:
    complement_b = dataclasses.replace(GOOD_B, operator=Operator.LE)  # YES_b == NO_a
    a = quote(GOOD_A, [("0.47", "100")], [("0.50", "60")])
    b = quote(complement_b, [("0.40", "30")], [("0.45", "30")])
    (opp,) = detect_cross_venue_arbitrage(a, b, COSTS)
    assert all(leg.side is Side.BUY for leg in opp.legs)
    assert opp.quantity == D("30")
    # gross 1 - 0.50 - 0.45 = 0.05 ; fees ceil(0.525) = 0.53, ceil(0.51975) = 0.52
    assert [leg.fee for leg in opp.legs] == [D("0.53"), D("0.52")]
    assert opp.gross_profit == D("1.50")
    # other: 0.005 + 0.05 * 0.001 * 0.95 = 0.0050475 per unit
    assert opp.other_costs == D("0.151425")
    assert opp.net_profit == D("0.298575")


def test_cross_venue_refuses_non_equivalent_pairs_even_with_huge_apparent_edge() -> None:
    a = quote(BAD_A, [("0.10", "100")], [("0.12", "100")])
    b = quote(BAD_B, [("0.90", "100")], [("0.92", "100")])
    assert detect_cross_venue_arbitrage(a, b, COSTS) == []
    unreviewed = quote(
        dataclasses.replace(GOOD_B, review_status=MappingStatus.DRAFT),
        [("0.90", "100")],
        [("0.92", "100")],
    )
    assert (
        detect_cross_venue_arbitrage(
            quote(GOOD_A, [("0.10", "100")], [("0.12", "100")]), unreviewed, COSTS
        )
        == []
    )


def test_cross_venue_input_and_book_checks() -> None:
    a = quote(GOOD_A, [("0.47", "100")], [("0.50", "60")])
    b = quote(GOOD_B, [("0.56", "40")], [("0.58", "100")])
    with pytest.raises(StructuralInputError):
        detect_cross_venue_arbitrage(a, a, COSTS)
    crossed = dataclasses.replace(
        b, book=dataclasses.replace(b.book, quality_flags=frozenset({QualityFlag.CROSSED}))
    )
    assert detect_cross_venue_arbitrage(a, crossed, COSTS) == []
    with pytest.raises(ValueError, match="non-negative"):
        CrossVenueCosts(settlement_ts_a_ns=0, settlement_ts_b_ns=0, annual_capital_rate=D("-0.01"))


# ---------------------------------------------------------------------- complement


def _yes_no_books(
    yes: tuple[list[tuple[str, str]], list[tuple[str, str]]],
    no: tuple[list[tuple[str, str]], list[tuple[str, str]]],
) -> tuple[ContractMapping, BookSnapshot, BookSnapshot]:
    mapping = GOOD_B
    yes_book = make_book("POLYMARKET:token-yes", *yes, venue=Venue.POLYMARKET)
    no_book = make_book("POLYMARKET:token-no", *no, venue=Venue.POLYMARKET)
    return mapping, yes_book, no_book


def test_complement_buy_both_exact() -> None:
    mapping, yes_book, no_book = _yes_no_books(
        ([("0.52", "100")], [("0.55", "50")]), ([("0.40", "70")], [("0.42", "30")])
    )
    (opp,) = detect_complement_mispricing(mapping, yes_book, no_book, yes_fee_schedule=NO_FEE)
    assert opp.kind is OpportunityKind.COMPLEMENT_BUY_BOTH
    yes_leg, no_leg = opp.legs
    assert (yes_leg.outcome, yes_leg.side, yes_leg.price) == (Outcome.YES, Side.BUY, D("0.55"))
    # buying NO at 0.42 is a canonical SELL of YES at 0.58
    assert (no_leg.outcome, no_leg.side, no_leg.price) == (Outcome.NO, Side.SELL, D("0.58"))
    assert (no_leg.token_side, no_leg.token_price) == (Side.BUY, D("0.42"))
    assert opp.quantity == D("30")
    assert opp.gross_edge_per_unit == D("0.03")  # 1 - 0.55 - 0.42
    assert opp.net_profit == D("0.90")


def test_complement_fees_absorb_small_gap() -> None:
    mapping, yes_book, no_book = _yes_no_books(
        ([("0.52", "100")], [("0.55", "50")]), ([("0.40", "70")], [("0.42", "30")])
    )
    # with 0.07 p(1-p) taker fees: ceil(0.51975) = 0.52 and ceil(0.07*30*0.58*0.42 = 0.51156) = 0.52
    assert detect_complement_mispricing(mapping, yes_book, no_book, yes_fee_schedule=FEE) == []


def test_complement_sell_both_exact() -> None:
    mapping, yes_book, no_book = _yes_no_books(
        ([("0.60", "20")], [("0.62", "50")]), ([("0.45", "50")], [("0.47", "30")])
    )
    (opp,) = detect_complement_mispricing(mapping, yes_book, no_book, yes_fee_schedule=NO_FEE)
    assert opp.kind is OpportunityKind.COMPLEMENT_SELL_BOTH
    yes_leg, no_leg = opp.legs
    assert (yes_leg.side, yes_leg.price, yes_leg.token_side) == (Side.SELL, D("0.60"), Side.SELL)
    assert (no_leg.side, no_leg.price) == (Side.BUY, D("0.55"))  # selling NO at 0.45
    assert (no_leg.token_side, no_leg.token_price) == (Side.SELL, D("0.45"))
    assert opp.quantity == D("20")
    assert opp.net_profit == D("1.00")  # 20 * (0.60 + 0.45 - 1)
    assert any("minted" in note for note in opp.notes)


def test_complement_checks() -> None:
    mapping, yes_book, no_book = _yes_no_books(
        ([("0.52", "100")], [("0.55", "50")]), ([("0.40", "70")], [("0.42", "30")])
    )
    with pytest.raises(MappingNotApprovedError):
        detect_complement_mispricing(
            dataclasses.replace(mapping, review_status=MappingStatus.DRAFT),
            yes_book,
            no_book,
            yes_fee_schedule=NO_FEE,
        )
    with pytest.raises(StructuralInputError, match="different token"):
        detect_complement_mispricing(mapping, yes_book, yes_book, yes_fee_schedule=NO_FEE)
    wrong_venue = make_book("KALSHI:x", [("0.4", "1")], [("0.5", "1")])
    with pytest.raises(StructuralInputError, match="is not on"):
        detect_complement_mispricing(mapping, yes_book, wrong_venue, yes_fee_schedule=NO_FEE)
    consistent = make_book(
        "POLYMARKET:token-no", [("0.40", "70")], [("0.47", "30")], venue=Venue.POLYMARKET
    )
    assert (
        detect_complement_mispricing(mapping, yes_book, consistent, yes_fee_schedule=NO_FEE) == []
    )


def test_opportunity_group_id_is_deterministic() -> None:
    a = quote(GOOD_A, [("0.47", "100")], [("0.50", "60")])
    b = quote(GOOD_B, [("0.56", "40")], [("0.58", "100")])
    first = detect_cross_venue_arbitrage(a, b, COSTS)[0]
    second = detect_cross_venue_arbitrage(a, b, COSTS)[0]
    assert first == second
    assert first.group_id == second.group_id
    assert isinstance(first.net_profit, Decimal)
