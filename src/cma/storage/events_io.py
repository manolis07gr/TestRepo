"""Canonical (normalized) event serialization: JSONL and Parquet.

Normalized events are *derived* data (regenerable from the raw store); this format is used
for replay fixtures and for caching normalized datasets. Decimals are written as strings.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

from cma.domain.enums import BookSide, ContractStatus, DeltaMode, QualityFlag, Side, Venue
from cma.domain.models import (
    BookDeltaEvent,
    BookLevel,
    BookSnapshotEvent,
    LevelChange,
    MarketEvent,
    StatusEvent,
    TradeEvent,
)


def _levels(levels: Iterable[BookLevel]) -> list[list[str]]:
    return [[str(lvl.price), str(lvl.quantity)] for lvl in levels]


def event_to_dict(ev: MarketEvent) -> dict[str, Any]:
    d: dict[str, Any] = {
        "type": ev.event_type.value,
        "venue": ev.venue.value,
        "instrument_id": ev.instrument_id,
        "source_ts_ns": ev.source_ts_ns,
        "recv_ts_ns": ev.recv_ts_ns,
        "sequence": ev.sequence,
        "payload_hash": ev.payload_hash,
        "sub_index": ev.sub_index,
    }
    if ev.quality_flags:
        d["quality_flags"] = sorted(f.value for f in ev.quality_flags)
    if isinstance(ev, BookSnapshotEvent):
        d["bids"] = _levels(ev.bids)
        d["asks"] = _levels(ev.asks)
        if ev.checksum:
            d["checksum"] = ev.checksum
    elif isinstance(ev, BookDeltaEvent):
        d["mode"] = ev.mode.value
        d["changes"] = [[c.side.value, str(c.price), str(c.quantity)] for c in ev.changes]
        if ev.checksum:
            d["checksum"] = ev.checksum
    elif isinstance(ev, TradeEvent):
        d["price"] = str(ev.price)
        d["size"] = str(ev.size)
        d["aggressor_side"] = ev.aggressor_side.value if ev.aggressor_side else None
        d["trade_id"] = ev.trade_id
    elif isinstance(ev, StatusEvent):
        d["status"] = ev.status.value
        d["detail"] = ev.detail
    else:  # pragma: no cover
        raise TypeError(f"unsupported event {type(ev).__name__}")
    return d


def event_from_dict(d: dict[str, Any]) -> MarketEvent:
    common: dict[str, Any] = {
        "venue": Venue(d["venue"]),
        "instrument_id": d["instrument_id"],
        "source_ts_ns": d.get("source_ts_ns"),
        "recv_ts_ns": int(d["recv_ts_ns"]),
        "sequence": d.get("sequence"),
        "payload_hash": d.get("payload_hash") or "",
        "sub_index": int(d.get("sub_index", 0)),
        "quality_flags": frozenset(QualityFlag(f) for f in d.get("quality_flags", [])),
    }
    kind = d["type"]
    if kind == "BOOK_SNAPSHOT":
        return BookSnapshotEvent(
            **common,
            bids=tuple(BookLevel(Decimal(p), Decimal(q)) for p, q in d.get("bids", [])),
            asks=tuple(BookLevel(Decimal(p), Decimal(q)) for p, q in d.get("asks", [])),
            checksum=d.get("checksum"),
        )
    if kind == "BOOK_DELTA":
        return BookDeltaEvent(
            **common,
            mode=DeltaMode(d["mode"]),
            changes=tuple(
                LevelChange(BookSide(s), Decimal(p), Decimal(q)) for s, p, q in d["changes"]
            ),
            checksum=d.get("checksum"),
        )
    if kind == "TRADE":
        agg = d.get("aggressor_side")
        return TradeEvent(
            **common,
            price=Decimal(d["price"]),
            size=Decimal(d["size"]),
            aggressor_side=Side(agg) if agg else None,
            trade_id=str(d["trade_id"]),
        )
    if kind == "STATUS":
        return StatusEvent(**common, status=ContractStatus(d["status"]), detail=d.get("detail", ""))
    raise ValueError(f"unknown event type {kind!r}")


def _open(path: Path, mode: str) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def write_events_jsonl(path: Path, events: Iterable[MarketEvent]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as fh:
        for ev in events:
            fh.write(json.dumps(event_to_dict(ev), sort_keys=True, separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n


def read_events_jsonl(path: Path) -> Iterator[MarketEvent]:
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                yield event_from_dict(json.loads(line))


def sort_by_recv(events: Iterable[MarketEvent]) -> list[MarketEvent]:
    return sorted(events, key=lambda e: (e.recv_ts_ns, e.venue_ts_ns, e.instrument_id))
