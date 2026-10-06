"""Persistence of :class:`PredictionContract` metadata (``contracts`` table).

Upserts are idempotent: re-upserting an identical contract changes nothing, and
``first_seen_ns`` is preserved across updates. The full contract is stored as canonical
JSON (exact decimal strings), so a load reproduces an equal ``PredictionContract``.

Only the latest listing state is kept. Backtests that need point-in-time contract
metadata (scope s.11.1) must snapshot the table or the raw listing pages
(``rest:markets`` streams in the raw store).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, Final

from cma.domain.enums import ContractStatus, Venue
from cma.domain.models import PredictionContract
from cma.domain.time import Clock, SystemClock
from cma.storage.db import Database

SCHEMA_VERSION: Final = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def contract_to_json(contract: PredictionContract) -> str:
    """Canonical JSON for a contract (sorted keys, exact decimals)."""
    data = {
        "schema": SCHEMA_VERSION,
        "venue": contract.venue.value,
        "contract_id": contract.contract_id,
        "native_id": contract.native_id,
        "event_id": contract.event_id,
        "title": contract.title,
        "yes_semantics": contract.yes_semantics,
        "no_semantics": contract.no_semantics,
        "open_ts_ns": contract.open_ts_ns,
        "close_ts_ns": contract.close_ts_ns,
        "resolve_ts_ns": contract.resolve_ts_ns,
        "status": contract.status.value,
        "tick_size": str(contract.tick_size),
        "series_id": contract.series_id,
        "rules_text": contract.rules_text,
        "can_close_early": contract.can_close_early,
        "fee_schedule_id": contract.fee_schedule_id,
        "settlement_metadata": _jsonable(contract.settlement_metadata),
        "outcome_instruments": _jsonable(contract.outcome_instruments),
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def contract_from_json(text: str) -> PredictionContract:
    data = json.loads(text)
    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported contract schema {data.get('schema')!r}")
    return PredictionContract(
        venue=Venue(data["venue"]),
        contract_id=data["contract_id"],
        native_id=data["native_id"],
        event_id=data["event_id"],
        title=data["title"],
        yes_semantics=data["yes_semantics"],
        no_semantics=data["no_semantics"],
        open_ts_ns=data["open_ts_ns"],
        close_ts_ns=data["close_ts_ns"],
        resolve_ts_ns=data["resolve_ts_ns"],
        status=ContractStatus(data["status"]),
        tick_size=Decimal(data["tick_size"]),
        series_id=data["series_id"],
        rules_text=data["rules_text"],
        can_close_early=bool(data["can_close_early"]),
        fee_schedule_id=data["fee_schedule_id"],
        settlement_metadata=data["settlement_metadata"],
        outcome_instruments=data["outcome_instruments"],
    )


class ContractStore:
    """Upsert/load prediction contracts in the metadata database."""

    def __init__(self, db: Database, *, clock: Clock | None = None) -> None:
        self._db = db
        self._clock: Clock = clock or SystemClock()

    def upsert(self, contract: PredictionContract, *, seen_ns: int | None = None) -> bool:
        """Insert or update ``contract``; returns True when the stored row changed."""
        payload = contract_to_json(contract)
        current = self._db.scalar(
            "SELECT contract_json FROM contracts WHERE venue = ? AND contract_id = ?",
            (contract.venue.value, contract.contract_id),
        )
        if current == payload:
            return False
        now = self._clock.now_ns() if seen_ns is None else seen_ns
        self._db.execute(
            "INSERT INTO contracts (venue, contract_id, native_id, event_id, series_id, title, "
            "status, open_ts_ns, close_ts_ns, resolve_ts_ns, tick_size, contract_json, "
            "first_seen_ns, updated_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (venue, contract_id) DO UPDATE SET native_id = excluded.native_id, "
            "event_id = excluded.event_id, series_id = excluded.series_id, "
            "title = excluded.title, status = excluded.status, "
            "open_ts_ns = excluded.open_ts_ns, close_ts_ns = excluded.close_ts_ns, "
            "resolve_ts_ns = excluded.resolve_ts_ns, tick_size = excluded.tick_size, "
            "contract_json = excluded.contract_json, updated_ns = excluded.updated_ns",
            (
                contract.venue.value,
                contract.contract_id,
                contract.native_id,
                contract.event_id,
                contract.series_id,
                contract.title,
                contract.status.value,
                contract.open_ts_ns,
                contract.close_ts_ns,
                contract.resolve_ts_ns,
                str(contract.tick_size),
                payload,
                now,
                now,
            ),
        )
        return True

    def upsert_many(
        self, contracts: Iterable[PredictionContract], *, seen_ns: int | None = None
    ) -> int:
        """Upsert all ``contracts`` in one transaction; returns how many rows changed."""
        changed = 0
        with self._db.transaction():
            for contract in contracts:
                changed += self.upsert(contract, seen_ns=seen_ns)
        return changed

    def get(self, venue: Venue, contract_id: str) -> PredictionContract | None:
        text = self._db.scalar(
            "SELECT contract_json FROM contracts WHERE venue = ? AND contract_id = ?",
            (venue.value, contract_id),
        )
        return None if text is None else contract_from_json(str(text))

    def load(
        self, *, venue: Venue | None = None, status: ContractStatus | None = None
    ) -> list[PredictionContract]:
        """All stored contracts (optionally filtered), ordered by venue and contract id."""
        sql = "SELECT contract_json FROM contracts"
        clauses: list[str] = []
        params: list[Any] = []
        if venue is not None:
            clauses.append("venue = ?")
            params.append(venue.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY venue, contract_id"
        return [contract_from_json(str(r["contract_json"])) for r in self._db.query(sql, params)]

    def first_seen_ns(self, venue: Venue, contract_id: str) -> int | None:
        value = self._db.scalar(
            "SELECT first_seen_ns FROM contracts WHERE venue = ? AND contract_id = ?",
            (venue.value, contract_id),
        )
        return None if value is None else int(value)

    def count(self) -> int:
        return int(self._db.scalar("SELECT COUNT(*) FROM contracts") or 0)
