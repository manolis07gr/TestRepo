"""T043 - contracts differing in cutoff/timezone/resolution source fail the equivalence check.

The good fixture pair (identical semantics on two venues) passes; the deceptively similar
pair (BRTI 60 s average before 5 PM ET vs Binance BTC/USDT 1-minute close at 16:00 UTC =
12 PM ET) fails with every mismatch listed. Both pairs are reviewed through the registry,
so deterministic validation runs on the fixture contracts as well.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import pytest

from cma.domain.enums import MappingStatus, Operator
from cma.domain.models import ContractMapping
from cma.domain.time import NS_PER_HOUR
from cma.mapping import EquivalenceRelation, MappingRegistry, check_equivalence
from tests.unit.test_mapping_fixtures import (
    contract_from_dict,
    fixture_mapping,
    load_fixture,
    review_through_registry,
)

pytestmark = pytest.mark.unit

FX = load_fixture("equivalence_good_bad.json")


def reviewed_pair(pair: dict[str, Any]) -> tuple[ContractMapping, ContractMapping]:
    registry = MappingRegistry()
    out = []
    for side in ("a", "b"):
        entry = pair[side]
        out.append(
            review_through_registry(
                registry, fixture_mapping(entry["mapping"]), contract_from_dict(entry["contract"])
            )
        )
    return out[0], out[1]


@pytest.fixture(scope="module")
def good() -> tuple[ContractMapping, ContractMapping]:
    return reviewed_pair(FX["good_pair"])


@pytest.fixture(scope="module")
def bad() -> tuple[ContractMapping, ContractMapping]:
    return reviewed_pair(FX["bad_pair"])


def test_t043_good_pair_is_equivalent(good: tuple[ContractMapping, ContractMapping]) -> None:
    a, b = good
    assert a.venue != b.venue
    assert a.review_status is b.review_status is MappingStatus.REVIEWED
    result = check_equivalence(a, b)
    expected = FX["good_pair"]["expected"]
    assert result.relation is EquivalenceRelation(expected["relation"])
    assert result.equivalent is expected["equivalent"] is True
    assert result.reasons == []
    assert check_equivalence(b, a).equivalent  # symmetric


def test_t043_deceptive_pair_fails_on_cutoff_timezone_and_source(
    bad: tuple[ContractMapping, ContractMapping],
) -> None:
    a, b = bad
    assert a.strikes == b.strikes
    assert a.operator is b.operator
    result = check_equivalence(a, b)
    expected = FX["bad_pair"]["expected"]
    assert result.relation is EquivalenceRelation.NONE
    assert result.equivalent is False
    assert set(result.mismatched_fields) == set(expected["mismatched_fields"])
    joined = " | ".join(result.reasons)
    for needle in ("observation_end", "timezone", "resolution_source", "BTCUSDT@BINANCE"):
        assert needle in joined
    assert len(result.reasons) == len(expected["mismatched_fields"])  # one reason per field


@pytest.mark.parametrize(
    ("field", "change"),
    [
        (
            "observation_end",
            lambda m: {
                "observation_end_ns": m.observation_end_ns + NS_PER_HOUR,
                "observation_start_ns": (m.observation_start_ns or 0) + NS_PER_HOUR,
            },
        ),
        ("timezone", lambda m: {"timezone": "UTC"}),
        ("resolution_source", lambda m: {"resolution_source": "BINANCE_BTCUSDT_1M_CLOSE"}),
        ("underlyings", lambda m: {"underlyings": ("BTCUSDT",)}),
        ("observation_method", lambda m: {"observation_method": "POINT"}),
        ("strikes", lambda m: {"strikes": (Decimal("112000"),)}),
        ("rounding_rule", lambda m: {"rounding_rule": "ROUND_HALF_UP_2DP"}),
        ("early_close_rule", lambda m: {"early_close_rule": "NO_EARLY_CLOSE;FALLBACK_50_50"}),
        ("operator", lambda m: {"operator": Operator.GE}),
    ],
)
def test_t043_any_single_semantic_difference_breaks_equivalence(
    good: tuple[ContractMapping, ContractMapping], field: str, change: Any
) -> None:
    a, b = good
    mutated = dataclasses.replace(b, **change(b))
    result = check_equivalence(a, mutated)
    assert not result.equivalent
    assert result.relation is EquivalenceRelation.NONE
    assert field in result.mismatched_fields
    assert any(reason.startswith(f"{field}:") for reason in result.reasons)


def test_t043_unreviewed_mappings_are_never_equivalent(
    good: tuple[ContractMapping, ContractMapping],
) -> None:
    a, b = good
    draft_b = dataclasses.replace(b, review_status=MappingStatus.DRAFT, reviewer=None)
    result = check_equivalence(a, draft_b)
    assert result.relation is EquivalenceRelation.SAME  # semantics match ...
    assert not result.equivalent  # ... but review is missing
    assert result.mismatched_fields == ("review_status",)


def test_t043_unspecified_fields_never_match(good: tuple[ContractMapping, ContractMapping]) -> None:
    a, b = good
    ua = dataclasses.replace(a, early_close_rule="UNSPECIFIED")
    ub = dataclasses.replace(b, early_close_rule="UNSPECIFIED")
    result = check_equivalence(ua, ub)
    assert not result.equivalent
    assert "early_close_rule" in result.mismatched_fields


@pytest.mark.parametrize(
    ("op_a", "op_b", "relation"),
    [
        (Operator.GT, Operator.LE, EquivalenceRelation.COMPLEMENT),
        (Operator.LE, Operator.GT, EquivalenceRelation.COMPLEMENT),
        (Operator.GE, Operator.LT, EquivalenceRelation.COMPLEMENT),
        (Operator.GT, Operator.LT, EquivalenceRelation.NONE),  # X == K belongs to neither
        (Operator.GT, Operator.GE, EquivalenceRelation.NONE),
    ],
)
def test_t043_exact_complements(
    good: tuple[ContractMapping, ContractMapping],
    op_a: Operator,
    op_b: Operator,
    relation: EquivalenceRelation,
) -> None:
    a, b = good
    result = check_equivalence(
        dataclasses.replace(a, operator=op_a), dataclasses.replace(b, operator=op_b)
    )
    assert result.relation is relation
    assert result.equivalent is (relation is not EquivalenceRelation.NONE)


def test_t043_up_down_are_not_exact_complements(
    good: tuple[ContractMapping, ContractMapping],
) -> None:
    a, b = good
    up = dataclasses.replace(
        a, operator=Operator.UP, strikes=(), observation_method="CANDLE_OPEN_CLOSE_1H"
    )
    down = dataclasses.replace(
        b, operator=Operator.DOWN, strikes=(), observation_method="CANDLE_OPEN_CLOSE_1H"
    )
    result = check_equivalence(up, down)
    assert result.relation is EquivalenceRelation.NONE
    assert any("tie" in r for r in result.reasons)
