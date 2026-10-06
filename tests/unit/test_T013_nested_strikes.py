"""T013 - nested strike constraint: detect a violation only when the higher-strike executable
probability (its bid) exceeds the lower-strike executable probability (its ask) after
tolerance and costs.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import pytest

from cma.domain.enums import MappingStatus, Operator, QualityFlag, Side, Venue
from cma.domain.errors import LookaheadError, MappingNotApprovedError
from cma.domain.models import BookSnapshot, ContractMapping
from cma.domain.time import NS_PER_HOUR
from cma.mapping import MappingRegistry
from cma.models.structural import (
    ContractQuote,
    OpportunityKind,
    StructuralConfig,
    StructuralInputError,
    detect_nested_violations,
)
from tests.unit.test_mapping_fixtures import (
    D,
    FixtureQuadraticFee,
    as_reviewed,
    book_from_dict,
    contract_from_dict,
    fixture_mapping,
    load_fixture,
    make_book,
    review_through_registry,
)

pytestmark = pytest.mark.unit

FX = load_fixture("nested_contracts.json")
FEE = FixtureQuadraticFee(D(FX["fee_model"]["taker_rate"]))


@pytest.fixture(scope="module")
def reviewed_family() -> dict[str, ContractMapping]:
    registry = MappingRegistry()
    out = {}
    for contract_data, mapping_data in zip(FX["contracts"], FX["mappings"], strict=True):
        contract = contract_from_dict(contract_data)
        out[contract.contract_id] = review_through_registry(
            registry, fixture_mapping(mapping_data), contract
        )
    return out


def quotes(
    family: dict[str, ContractMapping], books: dict[str, dict[str, Any]]
) -> list[ContractQuote]:
    return [
        ContractQuote(
            mapping=family[cid], book=book_from_dict(cid, data, Venue.KALSHI), fee_schedule=FEE
        )
        for cid, data in sorted(books.items())
    ]


def scenario(name: str) -> dict[str, Any]:
    return next(s for s in FX["scenarios"] if s["name"] == name)


@pytest.mark.parametrize("name", [s["name"] for s in FX["scenarios"]])
def test_t013_fixture_scenarios_match_hand_computed_expectations(
    reviewed_family: dict[str, ContractMapping], name: str
) -> None:
    sc = scenario(name)
    config = StructuralConfig(
        tolerance=D(sc["config"]["tolerance"]), walk_levels=sc["config"]["walk_levels"]
    )
    found = detect_nested_violations(quotes(reviewed_family, sc["books"]), config=config)
    assert len(found) == len(sc["expected"])
    for opp, exp in zip(found, sc["expected"], strict=True):
        assert opp.kind is OpportunityKind(exp["kind"])
        assert opp.quantity == D(exp["quantity"])
        assert [
            (leg.contract_id, leg.side.value, leg.price, leg.quantity, leg.fee) for leg in opp.legs
        ] == [
            (leg["contract_id"], leg["side"], D(leg["price"]), D(leg["quantity"]), D(leg["fee"]))
            for leg in exp["legs"]
        ]
        assert opp.gross_profit == D(exp["gross_profit"])
        assert opp.total_fees == D(exp["total_fees"])
        assert opp.net_profit == D(exp["net_profit"])
        assert opp.net_profit == opp.gross_profit - opp.total_fees - opp.other_costs
        assert opp.gross_edge_per_unit == opp.gross_profit / opp.quantity
        assert opp.fees_per_unit == opp.total_fees / opp.quantity
        assert opp.net_edge_per_unit == opp.net_profit / opp.quantity
        for key in ("gross_edge_per_unit", "fees_per_unit", "net_edge_per_unit"):
            if key in exp:
                assert getattr(opp, key) == D(exp[key])


def test_t013_exactly_one_violation_among_three_strikes(
    reviewed_family: dict[str, ContractMapping],
) -> None:
    low, mid, _ = sorted(reviewed_family)
    found = detect_nested_violations(quotes(reviewed_family, scenario("violation")["books"]))
    assert len(found) == 1
    assert found[0].contract_ids == (low, mid)
    buy, sell = found[0].legs
    assert (buy.side, sell.side) == (Side.BUY, Side.SELL)  # buy lower strike, sell higher


def test_t013_tolerance_boundary_is_strict(reviewed_family: dict[str, ContractMapping]) -> None:
    books = scenario("violation")["books"]
    # net edge per unit is exactly 0.043375 (3.47 / 80): equality is NOT a violation
    at_boundary = StructuralConfig(tolerance=D("0.043375"))
    assert detect_nested_violations(quotes(reviewed_family, books), config=at_boundary) == []
    just_inside = StructuralConfig(tolerance=D("0.043374"))
    assert len(detect_nested_violations(quotes(reviewed_family, books), config=just_inside)) == 1


def test_t013_monotone_books_have_no_violation(
    reviewed_family: dict[str, ContractMapping],
) -> None:
    low, mid, high = sorted(reviewed_family)
    ts = scenario("violation")["books"][low]["recv_ts"]
    books = {
        low: {"recv_ts": ts, "source_ts": ts, "bids": [["0.69", "100"]], "asks": [["0.71", "100"]]},
        mid: {"recv_ts": ts, "source_ts": ts, "bids": [["0.49", "100"]], "asks": [["0.51", "100"]]},
        high: {
            "recv_ts": ts,
            "source_ts": ts,
            "bids": [["0.29", "100"]],
            "asks": [["0.31", "100"]],
        },
    }
    assert detect_nested_violations(quotes(reviewed_family, books)) == []


def test_t013_invalid_book_suppresses_detection(
    reviewed_family: dict[str, ContractMapping],
) -> None:
    qs = quotes(reviewed_family, scenario("violation")["books"])
    flagged = [
        dataclasses.replace(
            q, book=dataclasses.replace(q.book, quality_flags=frozenset({QualityFlag.SEQUENCE_GAP}))
        )
        if q.contract_id == sorted(reviewed_family)[1]
        else q
        for q in qs
    ]
    assert detect_nested_violations(flagged) == []


def test_t013_lookahead_and_staleness(reviewed_family: dict[str, ContractMapping]) -> None:
    qs = quotes(reviewed_family, scenario("violation")["books"])
    book_ts = qs[0].book.recv_ts_ns
    with pytest.raises(LookaheadError):
        detect_nested_violations(qs, config=StructuralConfig(asof_ns=book_ts - 1))
    fresh = StructuralConfig(asof_ns=book_ts + 1, max_book_age_ns=NS_PER_HOUR)
    assert len(detect_nested_violations(qs, config=fresh)) == 1
    stale = StructuralConfig(asof_ns=book_ts + 2 * NS_PER_HOUR, max_book_age_ns=NS_PER_HOUR)
    assert detect_nested_violations(qs, config=stale) == []


def test_t013_requires_reviewed_mappings(reviewed_family: dict[str, ContractMapping]) -> None:
    qs = quotes(reviewed_family, scenario("violation")["books"])
    drafted = [
        dataclasses.replace(
            qs[0], mapping=dataclasses.replace(qs[0].mapping, review_status=MappingStatus.DRAFT)
        ),
        *qs[1:],
    ]
    with pytest.raises(MappingNotApprovedError):
        detect_nested_violations(drafted)


def test_t013_family_must_share_observation_semantics(
    reviewed_family: dict[str, ContractMapping],
) -> None:
    qs = quotes(reviewed_family, scenario("violation")["books"])
    shifted = dataclasses.replace(
        qs[1].mapping,
        observation_end_ns=qs[1].mapping.observation_end_ns + NS_PER_HOUR,
        observation_start_ns=(qs[1].mapping.observation_start_ns or 0) + NS_PER_HOUR,
    )
    with pytest.raises(StructuralInputError, match="observation_end_ns"):
        detect_nested_violations([qs[0], dataclasses.replace(qs[1], mapping=shifted), qs[2]])
    other_source = dataclasses.replace(qs[1].mapping, resolution_source="BINANCE_BTCUSDT_1M_CLOSE")
    with pytest.raises(StructuralInputError, match="resolution_source"):
        detect_nested_violations([qs[0], dataclasses.replace(qs[1], mapping=other_source)])
    mixed = dataclasses.replace(qs[1].mapping, operator=Operator.LT)
    with pytest.raises(StructuralInputError, match="all X>K"):
        detect_nested_violations([qs[0], dataclasses.replace(qs[1], mapping=mixed)])


def _lt_mapping(base: ContractMapping, strike: str, operator: Operator) -> ContractMapping:
    native = f"KXBTC-26OCT0617-T{strike}"
    return as_reviewed(
        dataclasses.replace(
            base,
            contract_id=f"KALSHI:{native}",
            operator=operator,
            strikes=(D(strike),),
            outcome_semantics=f"YES iff X {operator} {strike}",
        )
    )


def test_t013_mirrored_less_than_family(reviewed_family: dict[str, ContractMapping]) -> None:
    base = next(iter(reviewed_family.values()))
    lower = _lt_mapping(base, "110000", Operator.LT)  # {X < 110000} is the SUBSET event
    upper = _lt_mapping(base, "111000", Operator.LT)
    # P(X < 110000) bid 0.40 exceeds P(X < 111000) ask 0.30 by 0.10 per unit
    books: dict[str, BookSnapshot] = {
        lower.contract_id: make_book(lower.contract_id, [("0.40", "50")], [("0.42", "50")]),
        upper.contract_id: make_book(upper.contract_id, [("0.28", "70")], [("0.30", "30")]),
    }
    qs = [
        ContractQuote(mapping=m, book=books[m.contract_id], fee_schedule=FEE)
        for m in (lower, upper)
    ]
    (opp,) = detect_nested_violations(qs)
    assert [(leg.contract_id, leg.side) for leg in opp.legs] == [
        (upper.contract_id, Side.BUY),
        (lower.contract_id, Side.SELL),
    ]
    assert opp.quantity == D("30")  # min(ask depth 30, bid depth 50)
    # fees: ceil(0.07*30*0.30*0.70 = 0.441) = 0.45 ; ceil(0.07*30*0.40*0.60 = 0.504) = 0.51
    assert [leg.fee for leg in opp.legs] == [D("0.45"), D("0.51")]
    assert opp.gross_profit == D("3.00")
    assert opp.net_profit == D("2.04")


def test_t013_same_strike_ge_contains_gt(reviewed_family: dict[str, ContractMapping]) -> None:
    gt = next(iter(reviewed_family.values()))
    ge = as_reviewed(
        dataclasses.replace(gt, contract_id="KALSHI:KXBTCD-26OCT0617-GE", operator=Operator.GE)
    )
    # P(X > K) bid above P(X >= K) ask: the GE contract is the superset
    qs = [
        ContractQuote(
            mapping=gt,
            book=make_book(gt.contract_id, [("0.60", "10")], [("0.62", "10")]),
            fee_schedule=FEE,
        ),
        ContractQuote(
            mapping=ge,
            book=make_book(ge.contract_id, [("0.50", "10")], [("0.52", "10")]),
            fee_schedule=FEE,
        ),
    ]
    (opp,) = detect_nested_violations(qs)
    assert [(leg.contract_id, leg.side) for leg in opp.legs] == [
        (ge.contract_id, Side.BUY),
        (gt.contract_id, Side.SELL),
    ]
    assert opp.gross_edge_per_unit == Decimal("0.08")
