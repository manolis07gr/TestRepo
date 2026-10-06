"""T014 - exhaustive outcome constraint: exact over-/under-round mispricing with costs.

Set: a KXBTC-style partition of the BRTI 60 s average before 2026-10-06 17:00 ET:
  A: X < 110000   B: 110000 <= X <= 111999.99 (Kalshi 'between' is inclusive)   C: X > 111999.99
Fees: ceil_to_cent(0.07 x C x P x (1 - P)) per leg (single-fill order), all hand-computed.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from cma.domain.enums import MappingStatus, Operator, QualityFlag, Side
from cma.domain.errors import MappingNotApprovedError
from cma.domain.models import ContractMapping
from cma.domain.time import NS_PER_HOUR
from cma.models.structural import (
    ContractQuote,
    ExhaustiveSet,
    OpportunityKind,
    StructuralConfig,
    StructuralInputError,
    detect_exhaustive_mispricing,
)
from tests.unit.test_mapping_fixtures import (
    D,
    FixtureQuadraticFee,
    as_reviewed,
    fixture_mapping,
    load_fixture,
    make_book,
)

pytestmark = pytest.mark.unit

FEE = FixtureQuadraticFee(D("0.07"))
BASE = fixture_mapping(load_fixture("nested_contracts.json")["mappings"][0])


def member(code: str, operator: Operator, strikes: tuple[str, ...]) -> ContractMapping:
    return as_reviewed(
        dataclasses.replace(
            BASE,
            contract_id=f"KALSHI:KXBTC-26OCT0617-{code}",
            operator=operator,
            strikes=tuple(D(s) for s in strikes),
            outcome_semantics=f"bucket {code}",
        )
    )


A = member("A", Operator.LT, ("110000",))
B = member("B", Operator.BETWEEN, ("110000", "111999.99"))
C = member("C", Operator.GT, ("111999.99",))
DECLARED = ExhaustiveSet(
    set_id="KXBTC-26OCT0617",
    contract_ids=frozenset({A.contract_id, B.contract_id, C.contract_id}),
    mutually_exclusive=True,
    exhaustive=True,
    declared_by="alice.reviewer",
    basis="range buckets + both tails partition the BRTI average",
)


def quotes(
    books: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]],
) -> list[ContractQuote]:
    return [
        ContractQuote(
            mapping=m, book=make_book(m.contract_id, *books[m.contract_id]), fee_schedule=FEE
        )
        for m in (A, B, C)
    ]


UNDER = {
    A.contract_id: ([("0.28", "100")], [("0.30", "100")]),
    B.contract_id: ([("0.31", "100")], [("0.33", "150")]),
    C.contract_id: ([("0.28", "100")], [("0.30", "120"), ("0.31", "500")]),
}
OVER = {
    A.contract_id: ([("0.40", "50")], [("0.42", "50")]),
    B.contract_id: ([("0.38", "80")], [("0.40", "80")]),
    C.contract_id: ([("0.35", "60")], [("0.37", "60")]),
}


def test_t014_under_round_exact() -> None:
    (opp,) = detect_exhaustive_mispricing(quotes(UNDER), DECLARED)
    assert opp.kind is OpportunityKind.EXHAUSTIVE_UNDER_ROUND
    # executable quantity = min top-of-book ask depth across legs = min(100, 150, 120)
    assert opp.quantity == D("100")
    assert all(leg.side is Side.BUY for leg in opp.legs)
    # fees: A ceil(0.07*100*0.30*0.70 = 1.47) = 1.47 ; B ceil(7*0.33*0.67 = 1.5477) = 1.55 ;
    #       C 1.47  -> 4.49 total
    assert [leg.fee for leg in opp.legs] == [D("1.47"), D("1.55"), D("1.47")]
    assert opp.gross_edge_per_unit == D("0.07")  # 1 - (0.30 + 0.33 + 0.30)
    assert opp.gross_profit == D("7.00")
    assert opp.total_fees == D("4.49")
    assert opp.fees_per_unit == D("0.0449")
    assert opp.net_profit == D("2.51")  # 7.00 - 4.49
    assert opp.net_edge_per_unit == D("0.0251")
    # sum(asks) + sum(fees)/unit = 0.93 + 0.0449 = 0.9749 < 1 - tolerance
    assert sum(leg.price for leg in opp.legs) + opp.fees_per_unit == D("0.9749")


def test_t014_over_round_exact() -> None:
    (opp,) = detect_exhaustive_mispricing(quotes(OVER), DECLARED)
    assert opp.kind is OpportunityKind.EXHAUSTIVE_OVER_ROUND
    assert opp.quantity == D("50")  # min bid depth (50, 80, 60)
    assert all(leg.side is Side.SELL for leg in opp.legs)
    # fees: ceil(3.5*0.40*0.60 = 0.84) ; ceil(3.5*0.38*0.62 = 0.8246) = 0.83 ;
    #       ceil(3.5*0.35*0.65 = 0.79625) = 0.80  -> 2.47
    assert [leg.fee for leg in opp.legs] == [D("0.84"), D("0.83"), D("0.80")]
    assert opp.gross_edge_per_unit == D("0.13")  # (0.40 + 0.38 + 0.35) - 1
    assert opp.gross_profit == D("6.50")
    assert opp.total_fees == D("2.47")
    assert opp.net_profit == D("4.03")
    assert opp.net_edge_per_unit == D("0.0806")
    assert sum(leg.price for leg in opp.legs) - opp.fees_per_unit == D("1.0806")


def test_t014_costs_absorb_small_mispricing() -> None:
    near = {
        A.contract_id: ([("0.30", "100")], [("0.32", "100")]),
        B.contract_id: ([("0.33", "100")], [("0.35", "100")]),
        C.contract_id: ([("0.29", "100")], [("0.31", "100")]),
    }
    # sum(asks) = 0.98 < 1, but fees ~0.0466 per unit exceed the 0.02 gross edge
    assert detect_exhaustive_mispricing(quotes(near), DECLARED) == []


def test_t014_tolerance_is_applied() -> None:
    # under-round net edge is 0.0251 per unit: tolerance 0.0251 suppresses, 0.025 does not
    assert (
        detect_exhaustive_mispricing(
            quotes(UNDER), DECLARED, config=StructuralConfig(tolerance=D("0.0251"))
        )
        == []
    )
    assert (
        len(
            detect_exhaustive_mispricing(
                quotes(UNDER), DECLARED, config=StructuralConfig(tolerance=D("0.025"))
            )
        )
        == 1
    )


def test_t014_walk_levels_increases_size_only_while_profitable() -> None:
    deep = dict(UNDER)
    deep[A.contract_id] = ([("0.28", "100")], [("0.30", "100"), ("0.31", "50")])
    # step 1: 100 @ (0.30, 0.33, 0.30)  gross 0.07
    # step 2:  20 @ (0.31, 0.33, 0.30)  gross 0.06, marginal fees 0.04515 -> take
    # step 3:  30 @ (0.31, 0.33, 0.31)  gross 0.05, marginal fees 0.045423 -> take
    # then A has no depth left -> stop. Per (leg, level) single-fill fees:
    #   A 0.30x100 1.47, A 0.31x50 ceil(0.74865)=0.75, B 0.33x150 ceil(2.32155)=2.33,
    #   C 0.30x120 ceil(1.764)=1.77, C 0.31x30 ceil(0.44919)=0.45  -> 6.77
    # gross 7.00 + 1.20 + 1.50 = 9.70 -> net 2.93 (> 2.80 for two steps, > 2.51 for one)
    (opp,) = detect_exhaustive_mispricing(
        quotes(deep), DECLARED, config=StructuralConfig(walk_levels=True)
    )
    assert opp.quantity == D("150")
    assert [(leg.contract_id[-1], leg.price, leg.quantity, leg.fee) for leg in opp.legs] == [
        ("A", D("0.30"), D("100"), D("1.47")),
        ("A", D("0.31"), D("50"), D("0.75")),
        ("B", D("0.33"), D("150"), D("2.33")),
        ("C", D("0.30"), D("120"), D("1.77")),
        ("C", D("0.31"), D("30"), D("0.45")),
    ]
    assert opp.gross_profit == D("9.70")
    assert opp.total_fees == D("6.77")
    assert opp.net_profit == D("2.93")
    capped = detect_exhaustive_mispricing(
        quotes(deep), DECLARED, config=StructuralConfig(walk_levels=True, max_levels=1)
    )
    assert capped[0].quantity == D("100")
    top_only = detect_exhaustive_mispricing(quotes(deep), DECLARED)
    assert top_only[0].quantity == D("100")
    assert top_only[0].net_profit == D("2.51")


@pytest.mark.parametrize(
    "declaration",
    [
        dataclasses.replace(DECLARED, exhaustive=False),
        dataclasses.replace(DECLARED, mutually_exclusive=False),
        dataclasses.replace(DECLARED, declared_by="llm:assistant"),
        dataclasses.replace(DECLARED, contract_ids=frozenset({A.contract_id, B.contract_id})),
    ],
)
def test_t014_refuses_undeclared_or_mismatched_sets(declaration: ExhaustiveSet) -> None:
    with pytest.raises(StructuralInputError):
        detect_exhaustive_mispricing(quotes(UNDER), declaration)


def test_t014_members_must_share_observation_family() -> None:
    qs = quotes(UNDER)
    moved = dataclasses.replace(
        qs[2].mapping,
        observation_end_ns=qs[2].mapping.observation_end_ns + NS_PER_HOUR,
        observation_start_ns=(qs[2].mapping.observation_start_ns or 0) + NS_PER_HOUR,
    )
    with pytest.raises(StructuralInputError, match="observation"):
        detect_exhaustive_mispricing(
            [qs[0], qs[1], dataclasses.replace(qs[2], mapping=moved)], DECLARED
        )


def test_t014_requires_reviewed_mappings() -> None:
    qs = quotes(UNDER)
    draft = dataclasses.replace(qs[0].mapping, review_status=MappingStatus.DRAFT)
    with pytest.raises(MappingNotApprovedError):
        detect_exhaustive_mispricing([dataclasses.replace(qs[0], mapping=draft), *qs[1:]], DECLARED)


def test_t014_any_unusable_leg_suppresses() -> None:
    qs = quotes(UNDER)
    stale = dataclasses.replace(qs[1].book, quality_flags=frozenset({QualityFlag.STALE}))
    assert (
        detect_exhaustive_mispricing(
            [qs[0], dataclasses.replace(qs[1], book=stale), qs[2]], DECLARED
        )
        == []
    )
    empty = dataclasses.replace(qs[1].book, asks=())
    assert (
        detect_exhaustive_mispricing(
            [qs[0], dataclasses.replace(qs[1], book=empty), qs[2]], DECLARED
        )
        == []
    )


def test_t014_quantities_are_exact_decimals() -> None:
    (opp,) = detect_exhaustive_mispricing(quotes(UNDER), DECLARED)
    for value in (opp.gross_profit, opp.total_fees, opp.net_profit, opp.quantity):
        assert isinstance(value, Decimal)
        assert value == value.quantize(D("0.01"))
