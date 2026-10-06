"""Fixture loaders shared by the mapping/structural tests, plus fixture integrity checks.

Other test modules import the helpers (never the tests) from here.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cma.domain.enums import ContractStatus, LiquidityRole, MappingStatus, Venue
from cma.domain.fees import FeeSchedule
from cma.domain.models import BookLevel, BookSnapshot, ContractMapping, PredictionContract
from cma.domain.numbers import CENT, ONE, ZERO, ceil_to, validate_probability
from cma.domain.time import ns_from_iso8601
from cma.mapping import (
    MappingRegistry,
    ReviewChecklist,
    is_machine_actor,
    mapping_from_dict,
    propose_mapping,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
GENERATOR = REPO / "scripts" / "fixtures" / "gen_mapping_structural.py"
SAMPLE_REGISTRY_DIR = REPO / "config" / "mappings"

# Semantic fields compared between hand-written fixture mappings and parser output.
SEMANTIC_FIELDS = (
    "venue",
    "contract_id",
    "underlyings",
    "operator",
    "strikes",
    "observation_start_ns",
    "observation_end_ns",
    "observation_method",
    "timezone",
    "resolution_source",
    "rounding_rule",
    "early_close_rule",
    "event_family",
)


def D(text: str | int) -> Decimal:
    return Decimal(str(text))


def load_fixture(name: str) -> dict[str, Any]:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _opt_ns(value: str | None) -> int | None:
    return None if value is None else ns_from_iso8601(value)


def contract_from_dict(data: dict[str, Any]) -> PredictionContract:
    return PredictionContract(
        venue=Venue(data["venue"]),
        contract_id=data["contract_id"],
        native_id=data["native_id"],
        event_id=data["event_id"],
        title=data["title"],
        yes_semantics=data["yes_semantics"],
        no_semantics=data["no_semantics"],
        open_ts_ns=_opt_ns(data["open_ts"]),
        close_ts_ns=_opt_ns(data["close_ts"]),
        resolve_ts_ns=_opt_ns(data["resolve_ts"]),
        status=ContractStatus(data["status"]),
        tick_size=D(data["tick_size"]),
        series_id=data["series_id"],
        rules_text=data["rules_text"],
        can_close_early=data["can_close_early"],
        settlement_metadata=data["settlement_metadata"],
    )


def fixture_mapping(data: dict[str, Any]) -> ContractMapping:
    return mapping_from_dict(data, version=1, review_status=MappingStatus.DRAFT)


def book_from_dict(instrument_id: str, data: dict[str, Any], venue: Venue) -> BookSnapshot:
    return BookSnapshot(
        venue=venue,
        instrument_id=instrument_id,
        source_ts_ns=ns_from_iso8601(data["source_ts"]),
        recv_ts_ns=ns_from_iso8601(data["recv_ts"]),
        sequence=1,
        bids=tuple(BookLevel(D(p), D(q)) for p, q in data["bids"]),
        asks=tuple(BookLevel(D(p), D(q)) for p, q in data["asks"]),
    )


def make_book(
    instrument_id: str,
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    *,
    venue: Venue = Venue.KALSHI,
    recv_ts_ns: int = 0,
) -> BookSnapshot:
    return BookSnapshot(
        venue=venue,
        instrument_id=instrument_id,
        source_ts_ns=recv_ts_ns,
        recv_ts_ns=recv_ts_ns,
        sequence=1,
        bids=tuple(BookLevel(D(p), D(q)) for p, q in bids),
        asks=tuple(BookLevel(D(p), D(q)) for p, q in asks),
    )


DEFAULT_TAKER_RATE = Decimal("0.07")


class FixtureQuadraticFee(FeeSchedule):
    """The fixtures' fee model: ceil_to_cent(rate x C x P x (1 - P)) per single-fill order.

    Defined here (not taken from ``cma.domain.fees``) so hand-computed expectations stay
    valid when the shared venue schedules are re-verified or re-parameterised.
    """

    def __init__(self, taker_rate: Decimal = DEFAULT_TAKER_RATE) -> None:
        self.schedule_id = f"fixture-quadratic-{taker_rate}"
        self.version = "fixture"
        self.taker_rate = taker_rate

    def raw_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        p = validate_probability(price)
        rate = self.taker_rate if role is LiquidityRole.TAKER else ZERO
        return rate * quantity * p * (ONE - p)

    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        raw = self.raw_fee(price=price, quantity=quantity, role=role)
        return raw if raw == ZERO else ceil_to(raw, CENT)

    def fee_per_contract(self, *, price: Decimal, role: LiquidityRole) -> Decimal:
        return self.raw_fee(price=price, quantity=ONE, role=role)


def review_through_registry(
    registry: MappingRegistry,
    mapping: ContractMapping,
    contract: PredictionContract,
    *,
    proposer: str = "parser:fixture",
    reviewer: str = "alice.reviewer",
) -> ContractMapping:
    """Propose + human review with a complete checklist (deterministic validation runs)."""
    registry.propose(mapping, proposer, contract=contract)
    return registry.review(
        mapping.contract_id,
        reviewer,
        ReviewChecklist.all_affirmed(evidence="fixture rules text"),
        contract=contract,
    )


def as_reviewed(mapping: ContractMapping) -> ContractMapping:
    """Test data shortcut for arithmetic-only tests (no registry involved)."""
    return dataclasses.replace(
        mapping, review_status=MappingStatus.REVIEWED, reviewer="test.reviewer", reviewed_at_ns=0
    )


def semantic_view(mapping: ContractMapping) -> dict[str, object]:
    return {name: getattr(mapping, name) for name in SEMANTIC_FIELDS}


# --------------------------------------------------------------------------------------
# Fixture integrity
# --------------------------------------------------------------------------------------


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("gen_mapping_structural", GENERATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_files_match_deterministic_generator() -> None:
    generator = _load_generator()
    for name, document in generator.documents().items():
        assert (FIXTURES / name).read_text(encoding="utf-8") == generator.render(document), name


def test_kalshi_parser_reproduces_nested_fixture_mappings() -> None:
    fx = load_fixture("nested_contracts.json")
    for contract_data, mapping_data in zip(fx["contracts"], fx["mappings"], strict=True):
        result = propose_mapping(contract_from_dict(contract_data))
        assert result.mapping is not None, result.reason
        assert result.mapping.review_status is MappingStatus.DRAFT
        assert semantic_view(result.mapping) == semantic_view(fixture_mapping(mapping_data))


def test_polymarket_parser_reproduces_deceptive_fixture_mapping_except_undetermined_rule() -> None:
    fx = load_fixture("equivalence_good_bad.json")
    entry = fx["bad_pair"]["b"]
    result = propose_mapping(contract_from_dict(entry["contract"]))
    assert result.mapping is not None, result.reason
    expected = semantic_view(fixture_mapping(entry["mapping"]))
    parsed = semantic_view(result.mapping)
    # The parser cannot determine the data-outage fallback and leaves it UNSPECIFIED.
    assert parsed.pop("early_close_rule") == "UNSPECIFIED"
    expected.pop("early_close_rule")
    assert parsed == expected
    assert any("UTC" in w for w in result.warnings)


def test_synthetic_good_pair_counterpart_is_not_auto_parsed() -> None:
    fx = load_fixture("equivalence_good_bad.json")
    result = propose_mapping(contract_from_dict(fx["good_pair"]["b"]["contract"]))
    assert result.mapping is None
    assert result.reason is not None


def test_sample_family_yaml_is_draft_only_and_matches_parser_output() -> None:
    registry = MappingRegistry.from_yaml(SAMPLE_REGISTRY_DIR)
    fx = load_fixture("nested_contracts.json")
    ids = sorted(c["contract_id"] for c in fx["contracts"])
    assert registry.contract_ids() == ids
    family_key = fx["mappings"][0]["event_family"]
    assert [m.contract_id for m in registry.family(family_key)] == ids
    for contract_data in fx["contracts"]:
        cid = contract_data["contract_id"]
        record = registry.record(cid)
        assert record is not None
        assert record.status is MappingStatus.DRAFT
        assert record.review is None
        assert record.approval is None
        assert is_machine_actor(record.proposed_by)
        parsed = propose_mapping(contract_from_dict(contract_data)).mapping
        assert parsed is not None
        assert semantic_view(record.mapping) == semantic_view(parsed)
