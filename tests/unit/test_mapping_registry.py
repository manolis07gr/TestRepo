"""Mapping registry: deterministic validation, version history and YAML/DB persistence."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from cma.domain.enums import ExecutionMode, MappingStatus
from cma.domain.models import ContractMapping, PredictionContract
from cma.domain.time import NS_PER_MIN, ManualClock
from cma.mapping import (
    MappingRegistry,
    MappingReviewError,
    ReviewChecklist,
    canonical_event_family,
    validate_mapping,
)
from cma.storage.db import Database, migrate
from tests.unit.test_mapping_fixtures import (
    D,
    contract_from_dict,
    fixture_mapping,
    load_fixture,
)

pytestmark = pytest.mark.unit

FX = load_fixture("nested_contracts.json")
CHECKLIST = ReviewChecklist.all_affirmed(evidence="contract terms BTC.pdf, 2026-10-06")


@pytest.fixture
def family() -> list[tuple[ContractMapping, PredictionContract]]:
    return [
        (fixture_mapping(m), contract_from_dict(c))
        for m, c in zip(FX["mappings"], FX["contracts"], strict=True)
    ]


def approved_registry(
    family: list[tuple[ContractMapping, PredictionContract]], **kwargs: object
) -> MappingRegistry:
    clock = ManualClock(1_000)
    registry = MappingRegistry(clock=clock, **kwargs)  # type: ignore[arg-type]
    (m0, _), (m1, _), (m2, _) = family
    for m, c in family:
        registry.propose(m, "parser:kalshi-series/v1", contract=c)
    clock.advance(5)
    registry.review(m0.contract_id, "alice.reviewer", CHECKLIST)
    registry.review(m1.contract_id, "alice.reviewer", CHECKLIST)
    clock.advance(5)
    registry.approve_paper(m0.contract_id, "bob.approver")
    clock.advance(5)
    registry.propose(dataclasses.replace(m2, strikes=(D("111999.98"),), event_family=""), "carol")
    return registry


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"timezone": "EST"}, "IANA"),
        ({"timezone": "Mars/Olympus"}, "IANA"),
        ({"strikes": (D("110000"),)}, "disagree with venue metadata"),
        ({"observation_start_ns": None}, "requires observation_start"),
        ({"observation_method": "AVG_30S_BEFORE"}, "implies"),
        ({"observation_method": "CLOSE_ISH"}, "not a known method"),
        ({"early_close_rule": "UNSPECIFIED"}, "early_close_rule is unspecified"),
        ({"resolution_source": "UNKNOWN"}, "resolution_source is unspecified"),
        ({"event_family": "BTC-USD|whatever"}, "event_family"),
        (
            {"observation_end_ns": "SHIFT", "observation_start_ns": "SHIFT"},
            "not within",
        ),
    ],
)
def test_validation_problems_are_reported(
    family: list[tuple[ContractMapping, PredictionContract]],
    change: dict[str, object],
    message: str,
) -> None:
    mapping, contract = family[0]
    if change.get("observation_end_ns") == "SHIFT":  # 10 minutes off the venue close time
        end = mapping.observation_end_ns + 10 * NS_PER_MIN
        change = {
            "observation_end_ns": end,
            "observation_start_ns": end - NS_PER_MIN,
            "event_family": canonical_event_family(
                mapping.underlyings, mapping.observation_method, end
            ),
        }
    candidate = dataclasses.replace(mapping, **change)  # type: ignore[arg-type]
    problems = validate_mapping(candidate, contract)
    assert any(message in p for p in problems), problems
    registry = MappingRegistry()
    registry.propose(candidate, "carol.analyst", contract=contract)
    with pytest.raises(MappingReviewError):
        registry.review(candidate.contract_id, "alice.reviewer", CHECKLIST)
    assert registry.status(candidate.contract_id) is MappingStatus.DRAFT


def test_fixture_mappings_validate_cleanly(
    family: list[tuple[ContractMapping, PredictionContract]],
) -> None:
    for mapping, contract in family:
        assert validate_mapping(mapping, contract) == []


def test_review_requires_contract_metadata(
    family: list[tuple[ContractMapping, PredictionContract]],
) -> None:
    mapping, contract = family[0]
    registry = MappingRegistry()
    registry.propose(mapping, "parser:kalshi-series/v1")  # no contract registered
    with pytest.raises(MappingReviewError, match="no contract listing metadata"):
        registry.review(mapping.contract_id, "alice.reviewer", CHECKLIST)
    other = contract_from_dict(FX["contracts"][1])
    with pytest.raises(MappingReviewError, match="does not belong"):
        registry.review(mapping.contract_id, "alice.reviewer", CHECKLIST, contract=other)
    reviewed = registry.review(mapping.contract_id, "alice.reviewer", CHECKLIST, contract=contract)
    assert reviewed.review_status is MappingStatus.REVIEWED


def test_proposals_are_validated_for_identity(
    family: list[tuple[ContractMapping, PredictionContract]],
) -> None:
    mapping, contract = family[0]
    registry = MappingRegistry()
    with pytest.raises(ValueError, match="VENUE"):
        registry.propose(dataclasses.replace(mapping, contract_id="KXBTCD-no-prefix"), "carol")
    with pytest.raises(MappingReviewError):
        registry.propose(mapping, "  ")
    with pytest.raises(ValueError, match="does not belong"):
        registry.propose(mapping, "carol", contract=contract_from_dict(FX["contracts"][1]))
    registry.propose(mapping, "carol", contract=contract, at_ns=100)
    changed = dataclasses.replace(mapping, rounding_rule="ROUND_2DP")
    with pytest.raises(ValueError, match="precedes"):
        registry.propose(changed, "carol", at_ns=50)


def test_registry_queries_and_family(
    family: list[tuple[ContractMapping, PredictionContract]],
) -> None:
    registry = approved_registry(family)
    ids = [m.contract_id for m, _ in family]
    assert registry.contract_ids() == sorted(ids)
    assert len(registry) == 3
    assert ids[0] in registry
    assert "KALSHI:nope" not in registry
    assert [registry.status(cid) for cid in ids] == [
        MappingStatus.APPROVED_PAPER,
        MappingStatus.REVIEWED,
        MappingStatus.DRAFT,
    ]
    family_key = family[0][0].event_family
    assert [m.contract_id for m in registry.family(family_key)] == sorted(ids)
    assert [
        m.contract_id for m in registry.family(family_key, min_status=MappingStatus.REVIEWED)
    ] == ids[:2]
    assert registry.family("nothing") == []
    assert [m.version for m in registry.history(ids[2])] == [1, 2]
    assert registry.get(ids[2], version=3) is None
    assert registry.get("KALSHI:nope") is None
    assert registry.records("KALSHI:nope") == []
    record = registry.record(ids[0])
    assert record is not None
    assert record.machine_proposed
    assert record.review is not None
    assert record.review.reviewed_at_ns == 1_005
    assert record.approval is not None
    assert record.approval.approved_at_ns == 1_010


def test_yaml_round_trip_preserves_history_and_status(
    family: list[tuple[ContractMapping, PredictionContract]], tmp_path: Path
) -> None:
    registry = approved_registry(family)
    path = registry.save_yaml(tmp_path / "mappings" / "family.yaml")
    loaded = MappingRegistry.from_yaml(tmp_path / "mappings")
    assert loaded.contract_ids() == registry.contract_ids()
    for cid in registry.contract_ids():
        assert loaded.records(cid) == registry.records(cid)
    assert loaded.can_trade(family[0][0].contract_id, ExecutionMode.PAPER)
    assert not loaded.can_trade(family[2][0].contract_id, ExecutionMode.BACKTEST)
    text = path.read_text(encoding="utf-8")
    assert "schema: cma.contract_mappings/v1" in text
    with pytest.raises(ValueError, match="already loaded"):
        loaded.load_yaml(path)


def _tamper(path: Path, edit: object) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)  # type: ignore[operator]
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["records"][0]["versions"][0].update(approval=None), "no approval record"),
        (
            lambda d: d["records"][0]["versions"][0]["approval"].update(approver="alice.reviewer"),
            "four-eyes",
        ),
        (
            lambda d: d["records"][0]["versions"][0]["review"]["checklist"].update(rounding=False),
            "checklist incomplete",
        ),
        (
            lambda d: d["records"][0]["versions"][0].update(review_status="LIVE_ELIGIBLE"),
            "cannot be stored",
        ),
        (lambda d: d["records"][1]["versions"][0].update(review=None), "no review record"),
        (lambda d: d["records"][2]["versions"][1].update(version=3), "versions must be 1..n"),
        (
            lambda d: d["records"][2]["versions"][0].update(
                review={"reviewer": "x", "reviewed_at": "2026-10-06T00:00:00Z", "checklist": {}}
            ),
            "DRAFT mapping must not carry",
        ),
        (
            lambda d: d["records"][0]["versions"][0]["review"].update(reviewer="llm:assistant"),
            "not a human actor",
        ),
    ],
)
def test_yaml_load_fails_closed_on_tampered_status(
    family: list[tuple[ContractMapping, PredictionContract]],
    tmp_path: Path,
    edit: object,
    message: str,
) -> None:
    path = approved_registry(family).save_yaml(tmp_path / "family.yaml")
    _tamper(path, edit)
    with pytest.raises(ValueError, match=message):
        MappingRegistry.from_yaml(path)


def test_yaml_rejects_float_strikes(
    family: list[tuple[ContractMapping, PredictionContract]], tmp_path: Path
) -> None:
    path = approved_registry(family).save_yaml(tmp_path / "family.yaml")
    _tamper(path, lambda d: d["records"][2]["versions"][0]["mapping"].update(strikes=[111999.99]))
    with pytest.raises(ValueError, match="quoted decimal"):
        MappingRegistry.from_yaml(path)


def test_database_write_through_and_reload(
    family: list[tuple[ContractMapping, PredictionContract]],
) -> None:
    db = Database("sqlite:///:memory:")
    migrate(db)
    registry = approved_registry(family, db=db)
    rows = db.query(
        "SELECT contract_id, version, review_status, reviewer FROM contract_mappings "
        "ORDER BY contract_id, version"
    )
    # 3 contracts, 4 versions; status changes upsert in place (no duplicate rows)
    assert [(r["version"], r["review_status"]) for r in rows] == [
        (1, "APPROVED_PAPER"),
        (1, "REVIEWED"),
        (1, "DRAFT"),
        (2, "DRAFT"),
    ]
    assert rows[0]["reviewer"] == "alice.reviewer"
    reloaded = MappingRegistry(db=db)
    assert reloaded.load_db() == 3
    for cid in registry.contract_ids():
        assert reloaded.records(cid) == registry.records(cid)
    assert reloaded.can_trade(family[0][0].contract_id, ExecutionMode.PAPER)
    with pytest.raises(ValueError, match="no database"):
        MappingRegistry().load_db()
    db.close()


def test_four_eyes_can_be_disabled_explicitly(
    family: list[tuple[ContractMapping, PredictionContract]],
) -> None:
    mapping, contract = family[0]
    registry = MappingRegistry(require_four_eyes=False)
    registry.propose(mapping, "parser:kalshi-series/v1", contract=contract)
    registry.review(mapping.contract_id, "alice.reviewer", CHECKLIST)
    approved = registry.approve_paper(mapping.contract_id, "alice.reviewer")
    assert approved.review_status is MappingStatus.APPROVED_PAPER
    with pytest.raises(MappingReviewError, match="only REVIEWED"):
        registry.approve_paper(mapping.contract_id, "bob.approver")
    with pytest.raises(MappingReviewError, match="only DRAFT"):
        registry.review(mapping.contract_id, "bob.approver", CHECKLIST)


def test_timestamps_must_be_ordered(
    family: list[tuple[ContractMapping, PredictionContract]],
) -> None:
    mapping, contract = family[0]
    registry = MappingRegistry()
    registry.propose(mapping, "parser:kalshi-series/v1", contract=contract, at_ns=100)
    with pytest.raises(MappingReviewError, match="precedes"):
        registry.review(mapping.contract_id, "alice.reviewer", CHECKLIST, at_ns=99)
    registry.review(mapping.contract_id, "alice.reviewer", CHECKLIST, at_ns=200)
    with pytest.raises(MappingReviewError, match="precedes"):
        registry.approve_paper(mapping.contract_id, "bob.approver", at_ns=150)
