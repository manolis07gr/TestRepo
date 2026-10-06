"""Human-reviewable contract-mapping registry (scope s.9, test T012).

Lifecycle of one mapping *version*::

    propose  ->  DRAFT  --review(checklist + deterministic validation)-->  REVIEWED
             --approve_paper(four-eyes)-->  APPROVED_PAPER

* Every proposal enters as DRAFT, whatever status the submitted object claims; machine/LLM
  proposals can therefore never skip review.
* Any semantic change creates a new version, which starts again at DRAFT. The latest
  version governs tradeability, so a semantic change immediately suspends trading.
* ``LIVE_ELIGIBLE`` is reserved: :meth:`MappingRegistry.mark_live_eligible` always raises.
* Gates: BACKTEST needs >= REVIEWED, PAPER needs >= APPROVED_PAPER, LIVE is always refused.

Persistence: YAML documents (human review / version control, ``config/mappings``) and the
``contract_mappings`` table of :class:`cma.storage.db.Database`. Both are validated on load
(fail closed: a document that claims a status without the matching review/approval record
is rejected).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

import yaml

from cma.domain.enums import ExecutionMode, MappingStatus
from cma.domain.errors import LiveTradingDisabledError, MappingNotApprovedError
from cma.domain.models import ContractMapping, PredictionContract
from cma.domain.time import Clock, ManualClock, iso_from_ns, ns_from_iso8601
from cma.mapping.review import (
    DEFAULT_OBSERVATION_TOLERANCE_NS,
    MappingReviewError,
    ReviewChecklist,
    is_machine_actor,
    require_actor,
    same_actor,
    validate_mapping,
)
from cma.mapping.semantics import (
    mapping_family_key,
    mapping_from_dict,
    mapping_to_dict,
    semantic_key,
)
from cma.storage.db import Database

YAML_SCHEMA: Final = "cma.contract_mappings/v1"

# Minimum status per execution mode. LIVE is absent on purpose: never tradeable in v1.
REQUIRED_STATUS: Final[Mapping[ExecutionMode, MappingStatus]] = {
    ExecutionMode.BACKTEST: MappingStatus.REVIEWED,
    ExecutionMode.PAPER: MappingStatus.APPROVED_PAPER,
}

_UPSERT_SQL: Final = (
    "INSERT INTO contract_mappings (venue, contract_id, version, review_status, reviewer, "
    "reviewed_at_ns, mapping_json, created_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (venue, contract_id, version) DO UPDATE SET "
    "review_status = excluded.review_status, reviewer = excluded.reviewer, "
    "reviewed_at_ns = excluded.reviewed_at_ns, mapping_json = excluded.mapping_json"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewRecord:
    reviewer: str
    reviewed_at_ns: int
    checklist: ReviewChecklist


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalRecord:
    approver: str
    approved_at_ns: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MappingRecord:
    """One mapping version plus its audit trail (who proposed/reviewed/approved, when)."""

    mapping: ContractMapping
    proposed_by: str
    proposed_at_ns: int
    review: ReviewRecord | None = None
    approval: ApprovalRecord | None = None

    @property
    def contract_id(self) -> str:
        return self.mapping.contract_id

    @property
    def version(self) -> int:
        return self.mapping.version

    @property
    def status(self) -> MappingStatus:
        return self.mapping.review_status

    @property
    def machine_proposed(self) -> bool:
        return is_machine_actor(self.proposed_by)

    def problems(self, *, require_four_eyes: bool = True) -> list[str]:
        """Integrity of the status claim against the audit trail (used when loading)."""
        status, m = self.mapping.review_status, self.mapping
        out: list[str] = []
        if not self.proposed_by.strip():
            out.append("proposed_by missing")
        if status in (MappingStatus.UNMAPPED, MappingStatus.LIVE_ELIGIBLE):
            out.append(f"status {status} cannot be stored in the v1 registry")
            return out
        if status is MappingStatus.DRAFT:
            if self.review is not None or self.approval is not None or m.reviewer is not None:
                out.append("DRAFT mapping must not carry review/approval records")
            return out
        review = self.review
        if review is None:
            return [f"{status} mapping has no review record"]
        if not review.checklist.complete:
            out.append(f"review checklist incomplete: {list(review.checklist.missing())}")
        if not review.reviewer.strip() or is_machine_actor(review.reviewer):
            out.append(f"reviewer {review.reviewer!r} is not a human actor")
        if m.reviewer != review.reviewer or m.reviewed_at_ns != review.reviewed_at_ns:
            out.append("mapping reviewer fields disagree with the review record")
        if review.reviewed_at_ns < self.proposed_at_ns:
            out.append("review predates the proposal")
        if status is MappingStatus.REVIEWED:
            if self.approval is not None:
                out.append("REVIEWED mapping must not carry an approval record")
            return out
        approval = self.approval
        if approval is None:
            return [*out, "APPROVED_PAPER mapping has no approval record"]
        if not approval.approver.strip() or is_machine_actor(approval.approver):
            out.append(f"approver {approval.approver!r} is not a human actor")
        if require_four_eyes and same_actor(approval.approver, review.reviewer):
            out.append("four-eyes rule: approver must differ from reviewer")
        if approval.approved_at_ns < review.reviewed_at_ns:
            out.append("approval predates the review")
        return out


# --------------------------------------------------------------------------------------
# Record (de)serialisation shared by YAML and DB persistence
# --------------------------------------------------------------------------------------


def record_to_dict(record: MappingRecord) -> dict[str, Any]:
    review = record.review
    approval = record.approval
    return {
        "version": record.mapping.version,
        "review_status": record.mapping.review_status.value,
        "proposed_by": record.proposed_by,
        "proposed_at": iso_from_ns(record.proposed_at_ns),
        "mapping": mapping_to_dict(record.mapping, include_status=False),
        "review": None
        if review is None
        else {
            "reviewer": review.reviewer,
            "reviewed_at": iso_from_ns(review.reviewed_at_ns),
            "checklist": review.checklist.to_dict(),
        },
        "approval": None
        if approval is None
        else {"approver": approval.approver, "approved_at": iso_from_ns(approval.approved_at_ns)},
    }


def _iso_field(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an ISO-8601 string, got {value!r}")
    return ns_from_iso8601(value)


def record_from_dict(data: Mapping[str, Any]) -> MappingRecord:
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"version must be an integer, got {version!r}")
    status = MappingStatus(str(data.get("review_status")))
    mapping_data = data.get("mapping")
    if not isinstance(mapping_data, Mapping):
        raise ValueError("record.mapping must be a mapping")
    review_data = data.get("review")
    review: ReviewRecord | None = None
    if review_data is not None:
        if not isinstance(review_data, Mapping):
            raise ValueError("record.review must be a mapping or null")
        checklist_data = review_data.get("checklist", {})
        if not isinstance(checklist_data, Mapping):
            raise ValueError("review.checklist must be a mapping")
        review = ReviewRecord(
            reviewer=str(review_data.get("reviewer", "")),
            reviewed_at_ns=_iso_field(review_data, "reviewed_at"),
            checklist=ReviewChecklist.from_dict(checklist_data),
        )
    approval_data = data.get("approval")
    approval: ApprovalRecord | None = None
    if approval_data is not None:
        if not isinstance(approval_data, Mapping):
            raise ValueError("record.approval must be a mapping or null")
        approval = ApprovalRecord(
            approver=str(approval_data.get("approver", "")),
            approved_at_ns=_iso_field(approval_data, "approved_at"),
        )
    mapping = mapping_from_dict(
        mapping_data,
        version=version,
        review_status=status,
        reviewer=review.reviewer if review is not None else None,
        reviewed_at_ns=review.reviewed_at_ns if review is not None else None,
    )
    if review is None and mapping.reviewer is not None:
        mapping = dataclasses.replace(mapping, reviewer=None, reviewed_at_ns=None)
    return MappingRecord(
        mapping=mapping,
        proposed_by=str(data.get("proposed_by", "")),
        proposed_at_ns=_iso_field(data, "proposed_at"),
        review=review,
        approval=approval,
    )


class MappingRegistry:
    """In-memory registry of versioned contract mappings with optional DB write-through."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        db: Database | None = None,
        observation_tolerance_ns: int = DEFAULT_OBSERVATION_TOLERANCE_NS,
        require_four_eyes: bool = True,
    ) -> None:
        if observation_tolerance_ns < 0:
            raise ValueError("observation_tolerance_ns must be non-negative")
        self._clock: Clock = clock if clock is not None else ManualClock()
        self._db = db
        self.observation_tolerance_ns = observation_tolerance_ns
        self.require_four_eyes = require_four_eyes
        self._records: dict[str, list[MappingRecord]] = {}
        self._contracts: dict[str, PredictionContract] = {}

    # ------------------------------------------------------------------ contracts
    def register_contract(self, contract: PredictionContract) -> None:
        """Remember listing metadata used for deterministic validation at review time."""
        self._contracts[contract.contract_id] = contract

    def contract(self, contract_id: str) -> PredictionContract | None:
        return self._contracts.get(contract_id)

    # ------------------------------------------------------------------ workflow
    def propose(
        self,
        mapping: ContractMapping,
        proposed_by: str,
        *,
        contract: PredictionContract | None = None,
        at_ns: int | None = None,
    ) -> ContractMapping:
        """Register a proposal. Always returns a DRAFT unless semantics are unchanged.

        Re-proposing semantics identical to the latest version is a no-op that returns the
        latest version (with its current status). Any semantic difference creates version
        ``latest + 1`` in DRAFT; the submitted ``version``/status/reviewer are ignored.
        """
        actor = require_actor(proposed_by, "proposed_by", human=False)
        if not mapping.contract_id.startswith(f"{mapping.venue.value}:"):
            raise ValueError(
                f"contract_id {mapping.contract_id!r} must be '<VENUE>:<native id>' for "
                f"venue {mapping.venue}"
            )
        if contract is not None:
            if contract.contract_id != mapping.contract_id:
                raise ValueError("contract metadata does not belong to the proposed mapping")
            self.register_contract(contract)
        candidate = dataclasses.replace(
            mapping,
            event_family=mapping.event_family or mapping_family_key(mapping),
            review_status=MappingStatus.DRAFT,
            reviewer=None,
            reviewed_at_ns=None,
        )
        now = self._now(at_ns)
        versions = self._records.setdefault(mapping.contract_id, [])
        if versions:
            latest = versions[-1]
            if semantic_key(latest.mapping) == semantic_key(candidate):
                return latest.mapping
            if now < latest.proposed_at_ns:
                raise ValueError("proposal timestamp precedes the latest version's proposal")
            version = latest.mapping.version + 1
        else:
            version = 1
        record = MappingRecord(
            mapping=dataclasses.replace(candidate, version=version),
            proposed_by=actor,
            proposed_at_ns=now,
        )
        versions.append(record)
        self._persist(record)
        return record.mapping

    def review(
        self,
        contract_id: str,
        reviewer: str,
        checklist: ReviewChecklist,
        *,
        contract: PredictionContract | None = None,
        version: int | None = None,
        at_ns: int | None = None,
    ) -> ContractMapping:
        """DRAFT -> REVIEWED, only if the checklist is complete and validation passes."""
        record = self._latest_or_raise(contract_id, version)
        if record.status is not MappingStatus.DRAFT:
            raise MappingReviewError(
                f"{contract_id} v{record.version} is {record.status}; only DRAFT can be reviewed"
            )
        actor = require_actor(reviewer, "reviewer", human=True)
        missing = checklist.missing()
        if missing:
            raise MappingReviewError(
                f"{contract_id}: review checklist items not affirmed: {list(missing)}"
            )
        if contract is not None:
            if contract.contract_id != contract_id:
                raise MappingReviewError("contract metadata does not belong to this mapping")
            self.register_contract(contract)
        problems = validate_mapping(
            record.mapping,
            self._contracts.get(contract_id),
            observation_tolerance_ns=self.observation_tolerance_ns,
        )
        if problems:
            raise MappingReviewError(
                f"{contract_id} v{record.version}: deterministic validation failed: "
                + "; ".join(problems)
            )
        now = self._now(at_ns)
        if now < record.proposed_at_ns:
            raise MappingReviewError("review timestamp precedes the proposal")
        mapping = dataclasses.replace(
            record.mapping,
            review_status=MappingStatus.REVIEWED,
            reviewer=actor,
            reviewed_at_ns=now,
        )
        updated = dataclasses.replace(
            record,
            mapping=mapping,
            review=ReviewRecord(reviewer=actor, reviewed_at_ns=now, checklist=checklist),
        )
        self._replace_latest(updated)
        return mapping

    def approve_paper(
        self,
        contract_id: str,
        approver: str,
        *,
        version: int | None = None,
        at_ns: int | None = None,
    ) -> ContractMapping:
        """REVIEWED -> APPROVED_PAPER (four-eyes: approver must differ from the reviewer)."""
        record = self._latest_or_raise(contract_id, version)
        if record.status is not MappingStatus.REVIEWED or record.review is None:
            raise MappingReviewError(
                f"{contract_id} v{record.version} is {record.status}; only REVIEWED mappings "
                "can be approved for paper trading"
            )
        actor = require_actor(approver, "approver", human=True)
        if self.require_four_eyes and same_actor(actor, record.review.reviewer):
            raise MappingReviewError(
                f"four-eyes rule: approver {actor!r} reviewed {contract_id} v{record.version}"
            )
        now = self._now(at_ns)
        if now < record.review.reviewed_at_ns:
            raise MappingReviewError("approval timestamp precedes the review")
        mapping = dataclasses.replace(record.mapping, review_status=MappingStatus.APPROVED_PAPER)
        updated = dataclasses.replace(
            record,
            mapping=mapping,
            approval=ApprovalRecord(approver=actor, approved_at_ns=now),
        )
        self._replace_latest(updated)
        return mapping

    def mark_live_eligible(self, contract_id: str, *args: object, **kwargs: object) -> NoReturn:
        """LIVE_ELIGIBLE is reserved for a future, separately scoped approval process."""
        raise LiveTradingDisabledError(
            f"{contract_id}: LIVE_ELIGIBLE is reserved in v1; live trading requires a separate "
            "approval process that does not exist yet"
        )

    # ------------------------------------------------------------------ queries
    def get(self, contract_id: str, version: int | None = None) -> ContractMapping | None:
        record = self.record(contract_id, version)
        return record.mapping if record is not None else None

    def record(self, contract_id: str, version: int | None = None) -> MappingRecord | None:
        versions = self._records.get(contract_id)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        for rec in versions:
            if rec.version == version:
                return rec
        return None

    def records(self, contract_id: str) -> list[MappingRecord]:
        return list(self._records.get(contract_id, ()))

    def history(self, contract_id: str) -> list[ContractMapping]:
        """All versions, oldest first."""
        return [rec.mapping for rec in self._records.get(contract_id, ())]

    def status(self, contract_id: str) -> MappingStatus:
        latest = self.get(contract_id)
        return latest.review_status if latest is not None else MappingStatus.UNMAPPED

    def can_trade(self, contract_id: str, mode: ExecutionMode) -> bool:
        required = REQUIRED_STATUS.get(mode)
        if required is None:  # LIVE (or any future mode) is never tradeable in v1
            return False
        return self.status(contract_id).at_least(required)

    def require_tradeable(self, contract_id: str, mode: ExecutionMode) -> ContractMapping:
        """Return the governing mapping or raise :class:`MappingNotApprovedError`."""
        mapping = self.get(contract_id)
        if mapping is None or not self.can_trade(contract_id, mode):
            required = REQUIRED_STATUS.get(mode)
            need = f">= {required}" if required is not None else "LIVE (disabled in v1)"
            raise MappingNotApprovedError(
                f"{contract_id}: mapping status {self.status(contract_id)} does not permit "
                f"{mode} trading (requires {need})"
            )
        return mapping

    def family(
        self, event_family: str, *, min_status: MappingStatus | None = None
    ) -> list[ContractMapping]:
        """Latest versions of all contracts in ``event_family`` (sorted by contract id)."""
        out = [
            versions[-1].mapping
            for _, versions in sorted(self._records.items())
            if versions and versions[-1].mapping.event_family == event_family
        ]
        if min_status is not None:
            out = [m for m in out if m.review_status.at_least(min_status)]
        return out

    def contract_ids(self) -> list[str]:
        return sorted(self._records)

    def __contains__(self, contract_id: object) -> bool:
        return contract_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------ YAML persistence
    def to_yaml_document(self, contract_ids: Iterable[str] | None = None) -> dict[str, Any]:
        ids = sorted(contract_ids) if contract_ids is not None else self.contract_ids()
        records = []
        for cid in ids:
            versions = self._records.get(cid)
            if not versions:
                raise KeyError(f"unknown contract {cid!r}")
            records.append(
                {
                    "contract_id": cid,
                    "venue": versions[0].mapping.venue.value,
                    "versions": [record_to_dict(r) for r in versions],
                }
            )
        return {"schema": YAML_SCHEMA, "records": records}

    def save_yaml(self, path: Path, contract_ids: Iterable[str] | None = None) -> Path:
        """Write the selected contracts' full version histories to one YAML document."""
        document = self.to_yaml_document(contract_ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)
        path.write_text(text, encoding="utf-8")
        return path

    def load_yaml(self, path: Path) -> int:
        """Load a YAML document, or every ``*.yaml`` under a directory. Returns #contracts."""
        files = sorted(path.rglob("*.yaml")) if path.is_dir() else [path]
        loaded = 0
        for file in files:
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping) or data.get("schema") != YAML_SCHEMA:
                continue  # other documents (e.g. series_semantics.yaml) live alongside
            raw_records = data.get("records") or []
            if not isinstance(raw_records, list):
                raise ValueError(f"{file}: records must be a list")
            for raw in raw_records:
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{file}: each record must be a mapping")
                versions = raw.get("versions")
                if not isinstance(versions, list) or not versions:
                    raise ValueError(f"{file}: record {raw.get('contract_id')!r} has no versions")
                records = [record_from_dict(v) for v in versions if isinstance(v, Mapping)]
                if len(records) != len(versions):
                    raise ValueError(f"{file}: malformed version entry")
                self._install(str(raw.get("contract_id")), records, source=str(file))
                loaded += 1
        return loaded

    @classmethod
    def from_yaml(cls, path: Path, **kwargs: Any) -> MappingRegistry:
        registry = cls(**kwargs)
        registry.load_yaml(path)
        return registry

    # ------------------------------------------------------------------ DB persistence
    def load_db(self) -> int:
        """Load all versions from the ``contract_mappings`` table. Returns #contracts."""
        if self._db is None:
            raise ValueError("registry has no database")
        rows = self._db.query(
            "SELECT contract_id, version, mapping_json FROM contract_mappings "
            "ORDER BY contract_id, version"
        )
        grouped: dict[str, list[MappingRecord]] = {}
        for row in rows:
            record = record_from_dict(json.loads(str(row["mapping_json"])))
            if record.contract_id != row["contract_id"] or record.version != row["version"]:
                raise ValueError(f"contract_mappings row {row['contract_id']} is inconsistent")
            grouped.setdefault(record.contract_id, []).append(record)
        for cid, records in grouped.items():
            self._install(cid, records, source="database", persist=False)
        return len(grouped)

    # ------------------------------------------------------------------ internals
    def _now(self, at_ns: int | None) -> int:
        return self._clock.now_ns() if at_ns is None else at_ns

    def _latest_or_raise(self, contract_id: str, version: int | None) -> MappingRecord:
        versions = self._records.get(contract_id)
        if not versions:
            raise MappingReviewError(f"{contract_id}: no mapping proposed")
        latest = versions[-1]
        if version is not None and version != latest.version:
            raise MappingReviewError(
                f"{contract_id}: v{version} is superseded by v{latest.version}; only the latest "
                "version can change status"
            )
        return latest

    def _replace_latest(self, record: MappingRecord) -> None:
        self._records[record.contract_id][-1] = record
        self._persist(record)

    def _install(
        self,
        contract_id: str,
        records: list[MappingRecord],
        *,
        source: str,
        persist: bool = True,
    ) -> None:
        if contract_id in self._records:
            raise ValueError(f"{source}: contract {contract_id!r} is already loaded")
        for expected, record in enumerate(records, start=1):
            if record.contract_id != contract_id:
                raise ValueError(
                    f"{source}: version entry for {record.contract_id!r} under {contract_id!r}"
                )
            if record.version != expected:
                raise ValueError(f"{source}: {contract_id} versions must be 1..n in order")
            problems = record.problems(require_four_eyes=self.require_four_eyes)
            if problems:
                raise ValueError(
                    f"{source}: {contract_id} v{record.version} rejected: " + "; ".join(problems)
                )
        self._records[contract_id] = list(records)
        if persist:
            for record in records:
                self._persist(record)

    def _persist(self, record: MappingRecord) -> None:
        if self._db is None:
            return
        m = record.mapping
        self._db.execute(
            _UPSERT_SQL,
            (
                m.venue.value,
                m.contract_id,
                m.version,
                m.review_status.value,
                m.reviewer,
                m.reviewed_at_ns,
                json.dumps(record_to_dict(record), sort_keys=True),
                record.proposed_at_ns,
            ),
        )
