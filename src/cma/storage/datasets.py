"""Normalized dataset directories (derived, regenerable from the raw store).

Layout::

    <dir>/manifest.json       provenance: sources + hashes, collection version, gaps, transforms
    <dir>/events.jsonl.gz     normalized MarketEvents sorted by recv_ts
    <dir>/contracts.json      PredictionContract list
    <dir>/mappings.json       ContractMapping list (review status included)
    <dir>/settlements.json    Settlement list
    <dir>/reference.json      reference instrument ids
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cma.domain.enums import ContractStatus, MappingStatus, Operator, SettlementOutcome, Venue
from cma.domain.models import ContractMapping, PredictionContract, Settlement
from cma.storage.events_io import read_events_jsonl, write_events_jsonl

if TYPE_CHECKING:
    from cma.backtest.experiment import LoadedDataset


def contract_to_dict(c: PredictionContract) -> dict[str, Any]:
    return {
        "venue": c.venue.value,
        "contract_id": c.contract_id,
        "native_id": c.native_id,
        "event_id": c.event_id,
        "title": c.title,
        "yes_semantics": c.yes_semantics,
        "no_semantics": c.no_semantics,
        "open_ts_ns": c.open_ts_ns,
        "close_ts_ns": c.close_ts_ns,
        "resolve_ts_ns": c.resolve_ts_ns,
        "status": c.status.value,
        "tick_size": str(c.tick_size),
        "series_id": c.series_id,
        "rules_text": c.rules_text,
        "can_close_early": c.can_close_early,
        "fee_schedule_id": c.fee_schedule_id,
        "settlement_metadata": dict(c.settlement_metadata),
        "outcome_instruments": dict(c.outcome_instruments),
    }


def contract_from_dict(d: dict[str, Any]) -> PredictionContract:
    return PredictionContract(
        venue=Venue(d["venue"]),
        contract_id=d["contract_id"],
        native_id=d["native_id"],
        event_id=d["event_id"],
        title=d["title"],
        yes_semantics=d.get("yes_semantics", ""),
        no_semantics=d.get("no_semantics", ""),
        open_ts_ns=d.get("open_ts_ns"),
        close_ts_ns=d.get("close_ts_ns"),
        resolve_ts_ns=d.get("resolve_ts_ns"),
        status=ContractStatus(d.get("status", "UNKNOWN")),
        tick_size=Decimal(d["tick_size"]),
        series_id=d.get("series_id"),
        rules_text=d.get("rules_text", ""),
        can_close_early=bool(d.get("can_close_early", False)),
        fee_schedule_id=d.get("fee_schedule_id", ""),
        settlement_metadata=d.get("settlement_metadata", {}),
        outcome_instruments=d.get("outcome_instruments", {}),
    )


def mapping_to_dict(m: ContractMapping) -> dict[str, Any]:
    return {
        "venue": m.venue.value,
        "contract_id": m.contract_id,
        "underlyings": list(m.underlyings),
        "operator": m.operator.value,
        "strikes": [str(k) for k in m.strikes],
        "observation_start_ns": m.observation_start_ns,
        "observation_end_ns": m.observation_end_ns,
        "observation_method": m.observation_method,
        "timezone": m.timezone,
        "resolution_source": m.resolution_source,
        "rounding_rule": m.rounding_rule,
        "early_close_rule": m.early_close_rule,
        "outcome_semantics": m.outcome_semantics,
        "event_family": m.event_family,
        "version": m.version,
        "review_status": m.review_status.value,
        "reviewer": m.reviewer,
        "reviewed_at_ns": m.reviewed_at_ns,
        "notes": m.notes,
    }


def mapping_from_dict(d: dict[str, Any]) -> ContractMapping:
    return ContractMapping(
        venue=Venue(d["venue"]),
        contract_id=d["contract_id"],
        underlyings=tuple(d["underlyings"]),
        operator=Operator(d["operator"]),
        strikes=tuple(Decimal(k) for k in d["strikes"]),
        observation_start_ns=d.get("observation_start_ns"),
        observation_end_ns=int(d["observation_end_ns"]),
        observation_method=d["observation_method"],
        timezone=d["timezone"],
        resolution_source=d["resolution_source"],
        rounding_rule=d.get("rounding_rule", "NONE"),
        early_close_rule=d.get("early_close_rule", "NONE"),
        outcome_semantics=d.get("outcome_semantics", ""),
        event_family=d.get("event_family", ""),
        version=int(d.get("version", 1)),
        review_status=MappingStatus(d.get("review_status", "DRAFT")),
        reviewer=d.get("reviewer"),
        reviewed_at_ns=d.get("reviewed_at_ns"),
        notes=d.get("notes", ""),
    )


def settlement_to_dict(s: Settlement) -> dict[str, Any]:
    return {
        "venue": s.venue.value,
        "contract_id": s.contract_id,
        "outcome": s.outcome.value,
        "yes_value": str(s.yes_value),
        "settled_ts_ns": s.settled_ts_ns,
        "source": s.source,
    }


def settlement_from_dict(d: dict[str, Any]) -> Settlement:
    return Settlement(
        venue=Venue(d["venue"]),
        contract_id=d["contract_id"],
        outcome=SettlementOutcome(d["outcome"]),
        yes_value=Decimal(d["yes_value"]),
        settled_ts_ns=int(d["settled_ts_ns"]),
        source=d.get("source", ""),
    )


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(json.dumps(list(rows), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def save_dataset_dir(
    ds: LoadedDataset, path: Path, provenance: dict[str, Any] | None = None
) -> str:
    path.mkdir(parents=True, exist_ok=True)
    write_events_jsonl(path / "events.jsonl.gz", ds.events)
    _dump(path / "contracts.json", (contract_to_dict(c) for c in ds.contracts))
    _dump(path / "mappings.json", (mapping_to_dict(m) for m in ds.mappings))
    _dump(path / "settlements.json", (settlement_to_dict(s) for s in ds.settlements))
    (path / "reference.json").write_text(
        json.dumps(sorted(ds.reference_instruments)) + "\n", encoding="utf-8"
    )
    files = {
        name: _file_sha(path / name)
        for name in (
            "events.jsonl.gz",
            "contracts.json",
            "mappings.json",
            "settlements.json",
            "reference.json",
        )
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {
        "name": ds.name,
        "files": files,
        "dataset_hash": digest,
        "n_events": len(ds.events),
        "provenance": {**ds.provenance, **(provenance or {})},
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return digest


def load_dataset_dir(path: Path) -> LoadedDataset:
    from cma.backtest.experiment import LoadedDataset

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["files"].items():
        if _file_sha(path / name) != sha:
            raise ValueError(f"dataset file {name} does not match its manifest hash")
    contracts = [contract_from_dict(d) for d in json.loads((path / "contracts.json").read_text())]
    mappings = [mapping_from_dict(d) for d in json.loads((path / "mappings.json").read_text())]
    settlements = [
        settlement_from_dict(d) for d in json.loads((path / "settlements.json").read_text())
    ]
    refs = frozenset(json.loads((path / "reference.json").read_text()))
    closes = [(c.close_ts_ns, c.contract_id) for c in contracts if c.close_ts_ns is not None]
    return LoadedDataset(
        name=manifest["name"],
        events=list(read_events_jsonl(path / "events.jsonl.gz")),
        contracts=contracts,
        mappings=mappings,
        settlements=settlements,
        closes=closes,
        reference_instruments=refs,
        dataset_hash=manifest["dataset_hash"],
        provenance=manifest.get("provenance", {}),
    )
