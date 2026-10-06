"""T012 - UNMAPPED/DRAFT/REVIEWED contracts cannot forward-paper trade; APPROVED_PAPER can.

LIVE is never tradeable in v1 and LIVE_ELIGIBLE cannot be granted.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from cma.domain.enums import ExecutionMode, MappingStatus
from cma.domain.errors import LiveTradingDisabledError, MappingNotApprovedError
from cma.domain.models import ContractMapping, PredictionContract
from cma.domain.time import ManualClock
from cma.mapping import MappingRegistry, MappingReviewError, ReviewChecklist
from tests.unit.test_mapping_fixtures import contract_from_dict, fixture_mapping, load_fixture

pytestmark = pytest.mark.unit

PAPER, BACKTEST, LIVE = ExecutionMode.PAPER, ExecutionMode.BACKTEST, ExecutionMode.LIVE


@pytest.fixture
def sample() -> tuple[ContractMapping, PredictionContract]:
    fx = load_fixture("nested_contracts.json")
    return fixture_mapping(fx["mappings"][0]), contract_from_dict(fx["contracts"][0])


def gates(registry: MappingRegistry, cid: str) -> tuple[bool, bool, bool]:
    return (
        registry.can_trade(cid, BACKTEST),
        registry.can_trade(cid, PAPER),
        registry.can_trade(cid, LIVE),
    )


def test_t012_full_lifecycle_gates(sample: tuple[ContractMapping, PredictionContract]) -> None:
    mapping, contract = sample
    cid = mapping.contract_id
    clock = ManualClock(1_000)
    registry = MappingRegistry(clock=clock)

    assert registry.status(cid) is MappingStatus.UNMAPPED
    assert gates(registry, cid) == (False, False, False)
    with pytest.raises(MappingNotApprovedError):
        registry.require_tradeable(cid, PAPER)

    registry.propose(mapping, "parser:kalshi-series/v1", contract=contract)
    assert registry.status(cid) is MappingStatus.DRAFT
    assert gates(registry, cid) == (False, False, False)
    with pytest.raises(MappingNotApprovedError):
        registry.require_tradeable(cid, BACKTEST)

    clock.advance(10)
    registry.review(cid, "alice.reviewer", ReviewChecklist.all_affirmed("rules text v1"))
    assert registry.status(cid) is MappingStatus.REVIEWED
    assert gates(registry, cid) == (True, False, False)
    assert registry.require_tradeable(cid, BACKTEST).review_status is MappingStatus.REVIEWED
    with pytest.raises(MappingNotApprovedError):
        registry.require_tradeable(cid, PAPER)

    clock.advance(10)
    approved = registry.approve_paper(cid, "bob.approver")
    assert approved.review_status is MappingStatus.APPROVED_PAPER
    assert approved.reviewer == "alice.reviewer"
    assert approved.reviewed_at_ns == 1_010
    assert gates(registry, cid) == (True, True, False)
    assert registry.require_tradeable(cid, PAPER) == approved
    record = registry.record(cid)
    assert record is not None
    assert record.approval is not None
    assert record.approval.approver == "bob.approver"
    assert record.approval.approved_at_ns == 1_020

    with pytest.raises(MappingNotApprovedError):
        registry.require_tradeable(cid, LIVE)
    with pytest.raises(LiveTradingDisabledError):
        registry.mark_live_eligible(cid, "carol.approver")
    assert registry.status(cid) is MappingStatus.APPROVED_PAPER


def test_t012_machine_proposal_cannot_claim_approval(
    sample: tuple[ContractMapping, PredictionContract],
) -> None:
    mapping, contract = sample
    claimed = dataclasses.replace(
        mapping, review_status=MappingStatus.APPROVED_PAPER, reviewer="llm:assistant", version=7
    )
    registry = MappingRegistry()
    stored = registry.propose(claimed, "llm:assistant", contract=contract)
    assert stored.review_status is MappingStatus.DRAFT
    assert stored.reviewer is None
    assert stored.version == 1
    assert not registry.can_trade(mapping.contract_id, PAPER)
    assert not registry.can_trade(mapping.contract_id, BACKTEST)


@pytest.mark.parametrize("actor", ["llm:assistant", "parser:kalshi-series/v1", "bot:x", "  "])
def test_t012_machine_or_blank_actor_cannot_review(
    sample: tuple[ContractMapping, PredictionContract], actor: str
) -> None:
    mapping, contract = sample
    registry = MappingRegistry()
    registry.propose(mapping, "parser:kalshi-series/v1", contract=contract)
    with pytest.raises(MappingReviewError):
        registry.review(mapping.contract_id, actor, ReviewChecklist.all_affirmed())
    assert registry.status(mapping.contract_id) is MappingStatus.DRAFT


def test_t012_incomplete_checklist_keeps_draft(
    sample: tuple[ContractMapping, PredictionContract],
) -> None:
    mapping, contract = sample
    registry = MappingRegistry()
    registry.propose(mapping, "parser:kalshi-series/v1", contract=contract)
    partial = dataclasses.replace(ReviewChecklist.all_affirmed(), early_close_rule=False)
    with pytest.raises(MappingReviewError, match="early_close_rule"):
        registry.review(mapping.contract_id, "alice.reviewer", partial)
    assert registry.status(mapping.contract_id) is MappingStatus.DRAFT


def test_t012_four_eyes_rule(sample: tuple[ContractMapping, PredictionContract]) -> None:
    mapping, contract = sample
    cid = mapping.contract_id
    registry = MappingRegistry()
    registry.propose(mapping, "parser:kalshi-series/v1", contract=contract)
    with pytest.raises(MappingReviewError, match="only REVIEWED"):
        registry.approve_paper(cid, "bob.approver")  # DRAFT cannot be approved
    registry.review(cid, "Alice.Reviewer", ReviewChecklist.all_affirmed())
    with pytest.raises(MappingReviewError, match="four-eyes"):
        registry.approve_paper(cid, "alice.reviewer")  # case-insensitive identity
    with pytest.raises(MappingReviewError):
        registry.approve_paper(cid, "llm:assistant")
    assert not registry.can_trade(cid, PAPER)
    registry.approve_paper(cid, "bob.approver")
    assert registry.can_trade(cid, PAPER)


def test_t012_semantic_change_creates_new_draft_version_and_suspends_paper(
    sample: tuple[ContractMapping, PredictionContract],
) -> None:
    mapping, contract = sample
    cid = mapping.contract_id
    registry = MappingRegistry()
    registry.propose(mapping, "parser:kalshi-series/v1", contract=contract)
    registry.review(cid, "alice.reviewer", ReviewChecklist.all_affirmed())
    registry.approve_paper(cid, "bob.approver")
    assert registry.can_trade(cid, PAPER)

    # Identical semantics (only notes differ): idempotent, approval stands.
    same = registry.propose(
        dataclasses.replace(mapping, notes="re-parsed"), "parser:kalshi-series/v1"
    )
    assert same.version == 1
    assert same.review_status is MappingStatus.APPROVED_PAPER

    changed = registry.propose(
        dataclasses.replace(mapping, strikes=(Decimal("110000"),), event_family=""),
        "parser:kalshi-series/v1",
    )
    assert changed.version == 2
    assert changed.review_status is MappingStatus.DRAFT
    assert registry.status(cid) is MappingStatus.DRAFT
    assert not registry.can_trade(cid, PAPER)
    assert not registry.can_trade(cid, BACKTEST)
    history = registry.history(cid)
    assert [m.version for m in history] == [1, 2]
    assert history[0].review_status is MappingStatus.APPROVED_PAPER  # audit trail kept
    assert registry.get(cid, version=1) == history[0]
    with pytest.raises(MappingReviewError, match="superseded"):
        registry.approve_paper(cid, "bob.approver", version=1)
