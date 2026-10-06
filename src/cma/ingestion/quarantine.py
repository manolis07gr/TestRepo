"""Quarantine for bad external payloads (T047): persisted, counted, never fatal.

A payload that an adapter cannot parse, or that violates its schema (including a
probability outside [0, 1]), is written to a JSONL quarantine file. The record keeps
the receive metadata and the error, and the collector continues. Quarantine files are
evidence for schema-drift review. They are not replayed, though the same messages are
also in the raw store, because recording happens before parsing.

Layout: ``<root>/venue=<VENUE>/<YYYY-MM-DD>.jsonl`` (UTC day of ``recv_ts_ns``).
Registered secrets are redacted from the stored error text and payload.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cma.domain.models import RawMessage
from cma.domain.time import Clock, datetime_from_ns
from cma.security import redact

log = logging.getLogger(__name__)


class Quarantine:
    """Append-only store of rejected payloads with per-(venue, error type) counters."""

    def __init__(
        self, root: Path | str | None = None, *, clock: Clock | None = None, keep_last: int = 1_000
    ) -> None:
        self.root = None if root is None else Path(root)
        self._clock = clock
        self.counts: Counter[str] = Counter()
        self.recent: deque[dict[str, Any]] = deque(maxlen=keep_last)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def put(self, raw: RawMessage, error: BaseException | str) -> dict[str, Any]:
        """Quarantine ``raw`` with ``error``; returns the stored record."""
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Rejected"
        record: dict[str, Any] = {
            "venue": raw.venue.value,
            "stream": raw.stream,
            "recv_ts_ns": raw.recv_ts_ns,
            "connection_id": raw.connection_id,
            "connection_seq": raw.connection_seq,
            "idempotency_key": raw.idempotency_key,
            "payload_hash": raw.payload_hash,
            "error_type": error_type,
            "error": redact(str(error)),
            "quarantined_at_ns": (
                self._clock.now_ns() if self._clock is not None else raw.recv_ts_ns
            ),
            "payload": redact(raw.payload),
        }
        self.counts[f"{raw.venue.value}/{error_type}"] += 1
        self.recent.append(record)
        log.warning(
            "quarantined %s/%s message (%s): %s",
            raw.venue.value,
            raw.stream,
            error_type,
            record["error"][:300],
        )
        if self.root is not None:
            day = datetime_from_ns(raw.recv_ts_ns).strftime("%Y-%m-%d")
            directory = self.root / f"venue={raw.venue.value}"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{day}.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
                fh.write("\n")
        return record

    def counters(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Persisted records (all files, in path order); in-memory ones without a root."""
        if self.root is None:
            yield from list(self.recent)
            return
        for path in sorted(self.root.glob("venue=*/*.jsonl")):
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
