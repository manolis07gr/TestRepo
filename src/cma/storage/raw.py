"""Append-only raw message store (scope s.15): immutable, replayable, content-hashed.

Layout::

    <root>/venue=<VENUE>/stream=<sanitized stream>/day=<YYYY-MM-DD>/part-<n>.jsonl[.gz]

* Each line is one JSON object written verbatim. Its fields are ``venue``, ``stream``,
  ``recv_ts_ns``, ``connection_id``, ``connection_seq``, ``idempotency_key``,
  ``payload_hash`` and ``payload`` (the exact text received). The day is the UTC day of
  ``recv_ts_ns``.
* Files are append-only, and a recorder never re-opens an existing part. After a
  restart it always starts a new part, so a crash cannot turn a truncated line into a
  corrupted one. Readers skip a final line that has no newline; such a line is a partial
  write and was never acknowledged.
* Every part is sorted by ``(recv_ts_ns, connection_id, connection_seq)``. If an append
  would break that order (for example after a wall-clock step), the recorder rotates to a
  new part. :class:`RawReader` can therefore k-way merge parts lazily (``heapq.merge``).
* :meth:`RawRecorder.seal` gzips a part deterministically (mtime 0) and records its
  SHA-256 in ``raw_partitions``. Sealed parts are never modified again.
* Dedup: :meth:`RawRecorder.append` returns False for a message whose
  :attr:`~cma.domain.models.RawMessage.dedup_key` was already recorded. With a
  database, keys live in ``raw_event_keys`` so dedup survives restarts. Keys are
  committed in :meth:`~RawRecorder.flush` only *after* the data reaches the file. On
  start-up the keys of unsealed parts are reconciled against the files: missing keys are
  re-derived and keys whose line never reached the file are removed. A crash between the
  two writes can therefore neither duplicate nor lose a message. Without a database,
  dedup uses a bounded in-memory LRU for the life of the process.
* Single writer per root: two recorders must not write the same root concurrently.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final, Self

from cma.domain.enums import Venue
from cma.domain.errors import CMAError
from cma.domain.models import RawMessage
from cma.domain.time import NS_PER_DAY, datetime_from_ns, ns_from_datetime
from cma.storage.db import Database

log = logging.getLogger(__name__)

FORMAT_NAME: Final = "cma.raw.jsonl"
FORMAT_VERSION: Final = 1
_PART_RE = re.compile(r"^part-(\d+)\.jsonl(\.gz)?$")
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_KEY_BATCH = 5_000

type OrderKey = tuple[int, str, int]


class RawStoreError(CMAError):
    """Corrupt, unsorted or otherwise inconsistent raw-store content."""


def sanitize_stream(stream: str) -> str:
    """Directory-safe stream name (``ws:orderbook_delta`` -> ``ws_orderbook_delta``)."""
    return _SANITIZE_RE.sub("_", stream) or "_"


def utc_day(ns: int) -> str:
    return datetime_from_ns(ns).strftime("%Y-%m-%d")


def order_key(raw: RawMessage) -> OrderKey:
    return (raw.recv_ts_ns, raw.connection_id, raw.connection_seq)


def encode_record(raw: RawMessage) -> str:
    """One JSONL record (ASCII-escaped so any payload text round-trips exactly)."""
    return json.dumps(
        {
            "venue": raw.venue.value,
            "stream": raw.stream,
            "recv_ts_ns": raw.recv_ts_ns,
            "connection_id": raw.connection_id,
            "connection_seq": raw.connection_seq,
            "idempotency_key": raw.idempotency_key,
            "payload_hash": raw.payload_hash,
            "payload": raw.payload,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def decode_record(line: str, *, verify: bool = True) -> RawMessage:
    """Inverse of :func:`encode_record`; ``verify`` re-checks the payload hash."""
    try:
        obj = json.loads(line)
        key = obj["idempotency_key"]
        raw = RawMessage(
            venue=Venue(obj["venue"]),
            stream=str(obj["stream"]),
            recv_ts_ns=int(obj["recv_ts_ns"]),
            payload=str(obj["payload"]),
            connection_id=str(obj["connection_id"]),
            connection_seq=int(obj["connection_seq"]),
            idempotency_key=None if key is None else str(key),
        )
        stored_hash = obj.get("payload_hash")
    except (ValueError, KeyError, TypeError) as exc:
        raise RawStoreError(f"undecodable raw record: {exc}") from exc
    if verify and stored_hash != raw.payload_hash:
        raise RawStoreError(f"payload hash mismatch for {raw.venue}/{raw.stream}@{raw.recv_ts_ns}")
    return raw


@dataclass(frozen=True, slots=True)
class PartFile:
    """One part file on disk (plain while open/unsealed, ``.gz`` once sealed)."""

    path: Path
    venue: str
    stream_dir: str
    day: str
    number: int
    sealed: bool

    @property
    def partition_id(self) -> str:
        return f"venue={self.venue}/stream={self.stream_dir}/day={self.day}/part-{self.number:05d}"

    @property
    def day_start_ns(self) -> int:
        y, m, d = (int(x) for x in self.day.split("-"))
        return ns_from_datetime(datetime(y, m, d, tzinfo=UTC))

    def relative_path(self, root: Path) -> str:
        return self.path.relative_to(root).as_posix()


def _dir_value(name: str, prefix: str) -> str | None:
    return name[len(prefix) :] if name.startswith(prefix) else None


def discover_parts(root: Path) -> list[PartFile]:
    """All part files under ``root``, sorted by partition id.

    A part that exists both plain and gzipped (a seal interrupted after the atomic rename)
    is reported once, as the complete ``.gz`` file.
    """
    found: dict[tuple[str, str, str, int], PartFile] = {}
    if not root.exists():
        return []
    for venue_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        venue = _dir_value(venue_dir.name, "venue=")
        if venue is None:
            continue
        for stream_dir in sorted(p for p in venue_dir.iterdir() if p.is_dir()):
            stream = _dir_value(stream_dir.name, "stream=")
            if stream is None:
                continue
            for day_dir in sorted(p for p in stream_dir.iterdir() if p.is_dir()):
                day = _dir_value(day_dir.name, "day=")
                if day is None or not _DAY_RE.match(day):
                    continue
                for path in sorted(day_dir.iterdir()):
                    m = _PART_RE.match(path.name)
                    if m is None or not path.is_file():
                        continue
                    number, sealed = int(m.group(1)), m.group(2) is not None
                    key = (venue, stream, day, number)
                    existing = found.get(key)
                    if existing is not None and existing.sealed:
                        continue
                    found[key] = PartFile(path, venue, stream, day, number, sealed)
    return sorted(found.values(), key=lambda p: p.partition_id)


def _open_text(part: PartFile) -> IO[str]:
    # newline="\n": records are terminated by LF only (payload text is JSON-escaped)
    if part.sealed:
        return gzip.open(part.path, "rt", encoding="utf-8", newline="\n")
    return part.path.open("r", encoding="utf-8", newline="\n")


def iter_part_lines(part: PartFile) -> Iterator[str]:
    """Complete lines of a part (without newline); a partial final line is skipped."""
    with _open_text(part) as fh:
        for line in fh:
            if line.endswith("\n"):
                yield line[:-1]
                continue
            if part.sealed:
                raise RawStoreError(f"sealed part {part.path} ends with a partial record")
            log.warning("skipping partial final record in unsealed part %s", part.path)


def iter_part(part: PartFile, *, verify: bool = True) -> Iterator[RawMessage]:
    for line in iter_part_lines(part):
        yield decode_record(line, verify=verify)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _KeyLru:
    """Bounded LRU set of dedup keys."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._data: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        if key in self._data:
            self._data.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        self._data[key] = None
        self._data.move_to_end(key)
        if len(self._data) > self._capacity:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class _OpenPart:
    part: PartFile
    fh: IO[str]
    stream: str
    created_at_ns: int
    records: int = 0
    last_key: OrderKey | None = None
    dirty: bool = True


@dataclass(frozen=True, slots=True)
class SealedPart:
    partition_id: str
    path: Path
    records: int
    sha256: str


@dataclass
class RecorderStats:
    appended: int = 0
    duplicates: int = 0
    parts_opened: int = 0
    rotations: int = 0
    recovered_keys: int = 0
    removed_stale_keys: int = 0


class RawRecorder:
    """Append-only, deduplicating raw-message recorder (see module docstring).

    ``append`` buffers lines in the open part files. ``flush`` pushes them to the OS (and
    to disk with ``fsync=True``) and then commits their dedup keys. ``flush`` also runs
    automatically every ``flush_every`` appends. ``seal`` gzips every unsealed part,
    including parts left by an earlier, crashed process.
    """

    def __init__(
        self,
        root: Path | str,
        db: Database | None = None,
        *,
        max_records_per_part: int = 250_000,
        flush_every: int = 1_000,
        memory_dedup_capacity: int = 1_000_000,
        fsync: bool = False,
    ) -> None:
        if max_records_per_part <= 0 or flush_every <= 0:
            raise ValueError("max_records_per_part and flush_every must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = db
        self._max_records = max_records_per_part
        self._flush_every = flush_every
        self._fsync = fsync
        self._open: dict[tuple[str, str, str], _OpenPart] = {}
        self._known = _KeyLru(memory_dedup_capacity)
        self._pending: dict[str, tuple[str, int]] = {}
        self._unflushed = 0
        self._closed = False
        self.stats = RecorderStats()
        self._recover()

    # ------------------------------------------------------------------ public API

    def append(self, raw: RawMessage) -> bool:
        """Record ``raw``; False (and nothing written) if it is a duplicate."""
        if self._closed:
            raise RawStoreError("recorder is closed")
        key = raw.dedup_key
        if self._seen(key):
            self.stats.duplicates += 1
            return False
        op = self._part_for(raw)
        op.fh.write(encode_record(raw))
        op.fh.write("\n")
        op.records += 1
        op.last_key = order_key(raw)
        op.dirty = True
        self._known.add(key)
        if self._db is not None:
            self._pending[key] = (op.part.partition_id, raw.recv_ts_ns)
        self.stats.appended += 1
        self._unflushed += 1
        if self._unflushed >= self._flush_every:
            self.flush()
        return True

    def contains(self, raw: RawMessage) -> bool:
        """Whether ``raw`` (by dedup key) has already been recorded."""
        return self._seen(raw.dedup_key)

    def flush(self) -> None:
        """Write buffered lines to the files, then commit their dedup keys."""
        for op in self._open.values():
            op.fh.flush()
            if self._fsync:
                os.fsync(op.fh.fileno())
        if self._db is not None:
            rows = [(k, pid, ts) for k, (pid, ts) in self._pending.items()]
            dirty = [op for op in self._open.values() if op.dirty]
            if rows or dirty:
                with self._db.transaction():
                    for i in range(0, len(rows), _KEY_BATCH):
                        self._db.executemany(
                            "INSERT INTO raw_event_keys (dedup_key, partition_id, recv_ts_ns) "
                            "VALUES (?, ?, ?) ON CONFLICT (dedup_key) DO NOTHING",
                            rows[i : i + _KEY_BATCH],
                        )
                    for op in dirty:
                        self._upsert_partition(
                            op.part, op.stream, op.records, op.created_at_ns, None, sealed=False
                        )
            self._pending.clear()
        for op in self._open.values():
            op.dirty = False
        self._unflushed = 0

    def seal(self) -> list[SealedPart]:
        """Close open parts and gzip + hash every unsealed part under the root."""
        self.flush()
        for op in self._open.values():
            op.fh.close()
        self._open.clear()
        return [self._seal_part(part) for part in discover_parts(self.root) if not part.sealed]

    def close(self) -> None:
        """Flush and close file handles (parts stay unsealed; ``seal`` them later)."""
        if self._closed:
            return
        self.flush()
        for op in self._open.values():
            op.fh.close()
        self._open.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ internals

    def _seen(self, key: str) -> bool:
        if key in self._known:
            return True
        if self._db is None:
            return False
        if key in self._pending:
            return True
        hit = self._db.scalar("SELECT 1 FROM raw_event_keys WHERE dedup_key = ?", (key,))
        if hit is not None:
            self._known.add(key)
            return True
        return False

    def _part_for(self, raw: RawMessage) -> _OpenPart:
        venue = raw.venue.value
        stream_dir = sanitize_stream(raw.stream)
        day = utc_day(raw.recv_ts_ns)
        slot = (venue, stream_dir, day)
        op = self._open.get(slot)
        if op is not None:
            out_of_order = op.last_key is not None and order_key(raw) < op.last_key
            if op.records >= self._max_records or out_of_order:
                if out_of_order:
                    log.info("raw part %s: out-of-order append, rotating", op.part.partition_id)
                self._retire(op)
                del self._open[slot]
                self.stats.rotations += 1
                op = None
        if op is None:
            op = self._open_new_part(venue, stream_dir, day, raw)
            self._open[slot] = op
        return op

    def _retire(self, op: _OpenPart) -> None:
        op.fh.flush()
        if self._fsync:
            os.fsync(op.fh.fileno())
        if self._db is not None:
            # keys of the retired part must be committed together with its final count
            self.flush()
        op.fh.close()

    def _open_new_part(self, venue: str, stream_dir: str, day: str, raw: RawMessage) -> _OpenPart:
        directory = self.root / f"venue={venue}" / f"stream={stream_dir}" / f"day={day}"
        directory.mkdir(parents=True, exist_ok=True)
        numbers = [
            int(m.group(1))
            for p in directory.iterdir()
            if (m := _PART_RE.match(p.name)) is not None
        ]
        number = max(numbers, default=-1) + 1
        path = directory / f"part-{number:05d}.jsonl"
        fh = path.open("x", encoding="utf-8", newline="\n")
        self.stats.parts_opened += 1
        part = PartFile(path, venue, stream_dir, day, number, sealed=False)
        return _OpenPart(part=part, fh=fh, stream=raw.stream, created_at_ns=raw.recv_ts_ns)

    def _upsert_partition(
        self,
        part: PartFile,
        stream: str,
        records: int,
        created_at_ns: int,
        sha256: str | None,
        *,
        sealed: bool,
    ) -> None:
        assert self._db is not None
        self._db.execute(
            "INSERT INTO raw_partitions "
            "(partition_id, venue, stream, day, path, n_records, sha256, sealed, created_at_ns) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (partition_id) DO UPDATE SET path = excluded.path, "
            "n_records = excluded.n_records, sha256 = excluded.sha256, sealed = excluded.sealed",
            (
                part.partition_id,
                part.venue,
                stream,
                part.day,
                part.relative_path(self.root),
                records,
                sha256,
                1 if sealed else 0,
                created_at_ns,
            ),
        )

    def _seal_part(self, part: PartFile) -> SealedPart:
        gz_path = part.path.with_name(part.path.name + ".gz")
        tmp_path = part.path.with_name(part.path.name + ".gz.tmp")
        records = 0
        first: RawMessage | None = None
        with tmp_path.open("wb") as raw_fh:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz:
                for line in iter_part_lines(part):
                    if first is None:
                        first = decode_record(line, verify=False)
                    gz.write(line.encode("utf-8"))
                    gz.write(b"\n")
                    records += 1
            raw_fh.flush()
            os.fsync(raw_fh.fileno())
        os.replace(tmp_path, gz_path)
        part.path.unlink()
        sealed = PartFile(gz_path, part.venue, part.stream_dir, part.day, part.number, True)
        digest = _sha256_file(gz_path)
        if self._db is not None:
            stream = first.stream if first is not None else part.stream_dir
            created = first.recv_ts_ns if first is not None else part.day_start_ns
            with self._db.transaction():
                self._upsert_partition(sealed, stream, records, created, digest, sealed=True)
        return SealedPart(sealed.partition_id, gz_path, records, digest)

    def _recover(self) -> None:
        """Finish interrupted seals and reconcile dedup keys of unsealed parts."""
        for tmp in self.root.glob("venue=*/stream=*/day=*/*.gz.tmp"):
            tmp.unlink()
        for gz in self.root.glob("venue=*/stream=*/day=*/part-*.jsonl.gz"):
            plain = gz.with_suffix("")
            if plain.exists():  # rename completed, plain copy not yet deleted
                plain.unlink()
        if self._db is None:
            return
        for part in discover_parts(self.root):
            if part.sealed:
                continue
            rows: list[tuple[str, str, int]] = []
            stream = part.stream_dir
            created = part.day_start_ns
            for raw in iter_part(part, verify=False):
                if not rows:
                    stream, created = raw.stream, raw.recv_ts_ns
                rows.append((raw.dedup_key, part.partition_id, raw.recv_ts_ns))
            file_keys = {r[0] for r in rows}
            db_keys = {
                str(r["dedup_key"])
                for r in self._db.query(
                    "SELECT dedup_key FROM raw_event_keys WHERE partition_id = ?",
                    (part.partition_id,),
                )
            }
            stale = sorted(db_keys - file_keys)
            missing = [r for r in rows if r[0] not in db_keys]
            with self._db.transaction():
                for key in stale:
                    self._db.execute("DELETE FROM raw_event_keys WHERE dedup_key = ?", (key,))
                for i in range(0, len(missing), _KEY_BATCH):
                    self._db.executemany(
                        "INSERT INTO raw_event_keys (dedup_key, partition_id, recv_ts_ns) "
                        "VALUES (?, ?, ?) ON CONFLICT (dedup_key) DO NOTHING",
                        missing[i : i + _KEY_BATCH],
                    )
                self._upsert_partition(part, stream, len(rows), created, None, sealed=False)
            self.stats.recovered_keys += len(missing)
            self.stats.removed_stale_keys += len(stale)
            if missing or stale:
                log.warning(
                    "raw recovery %s: %d keys re-derived, %d stale keys removed",
                    part.partition_id,
                    len(missing),
                    len(stale),
                )


class RawReader:
    """Replays recorded messages in ``(recv_ts_ns, connection_id, connection_seq)`` order."""

    def __init__(self, root: Path | str, *, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        self._verify = verify_hashes

    def parts(
        self,
        *,
        venues: Iterable[Venue | str] | None = None,
        streams: Iterable[str] | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> list[PartFile]:
        venue_set = None if venues is None else {Venue(v).value for v in venues}
        stream_dirs = None if streams is None else {sanitize_stream(s) for s in streams}
        out = []
        for part in discover_parts(self.root):
            if venue_set is not None and part.venue not in venue_set:
                continue
            if stream_dirs is not None and part.stream_dir not in stream_dirs:
                continue
            day_start = part.day_start_ns
            if end_ns is not None and day_start >= end_ns:
                continue
            if start_ns is not None and day_start + NS_PER_DAY <= start_ns:
                continue
            out.append(part)
        return out

    def iter_messages(
        self,
        venues: Iterable[Venue | str] | None = None,
        streams: Iterable[str] | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> Iterator[RawMessage]:
        """Messages with ``start_ns <= recv_ts_ns < end_ns`` merged across partitions."""
        stream_set = None if streams is None else set(streams)
        parts = self.parts(venues=venues, streams=stream_set, start_ns=start_ns, end_ns=end_ns)
        iterators = [self._iter_filtered(p, stream_set, start_ns, end_ns) for p in parts]
        last: OrderKey | None = None
        for raw in heapq.merge(*iterators, key=order_key):
            key = order_key(raw)
            if last is not None and key < last:
                raise RawStoreError(f"raw parts are not sorted at {key} (after {last})")
            last = key
            yield raw

    def _iter_filtered(
        self,
        part: PartFile,
        streams: set[str] | None,
        start_ns: int | None,
        end_ns: int | None,
    ) -> Iterator[RawMessage]:
        for raw in iter_part(part, verify=self._verify):
            if start_ns is not None and raw.recv_ts_ns < start_ns:
                continue
            if end_ns is not None and raw.recv_ts_ns >= end_ns:
                break  # parts are sorted by recv time
            if streams is not None and raw.stream not in streams:
                continue
            yield raw


# --------------------------------------------------------------------------------------
# Dataset manifests (scope s.8 provenance, s.15 content hashes)
# --------------------------------------------------------------------------------------

DEFAULT_PROVENANCE: Final[Mapping[str, Any]] = {
    "generator": "cma.storage.raw",
    "payloads": "verbatim text as received; recv_ts from the collector clock",
    "transformations": [],
}


@dataclass
class _PartSummary:
    records: int = 0
    first_recv_ts_ns: int | None = None
    last_recv_ts_ns: int | None = None
    connections: set[str] = field(default_factory=set)
    streams: set[str] = field(default_factory=set)
    content: Any = field(default_factory=hashlib.sha256)


def _summarize(part: PartFile) -> _PartSummary:
    s = _PartSummary()
    for line in iter_part_lines(part):
        s.content.update(line.encode("utf-8"))
        s.content.update(b"\n")
        raw = decode_record(line, verify=False)
        s.records += 1
        if s.first_recv_ts_ns is None:
            s.first_recv_ts_ns = raw.recv_ts_ns
        s.last_recv_ts_ns = raw.recv_ts_ns
        s.connections.add(raw.connection_id)
        s.streams.add(raw.stream)
    return s


def build_manifest(
    root: Path | str, *, provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Machine-readable dataset manifest: files, sizes, record counts, hashes, provenance.

    ``content_sha256`` hashes the complete records (identical before and after sealing).
    ``sha256``/``bytes`` describe the file exactly as stored. The manifest contains no
    generation timestamp, so building it twice gives identical output.
    """
    root_path = Path(root)
    files = []
    total_records = 0
    total_bytes = 0
    for part in discover_parts(root_path):
        summary = _summarize(part)
        size = part.path.stat().st_size
        files.append(
            {
                "partition_id": part.partition_id,
                "path": part.relative_path(root_path),
                "venue": part.venue,
                "streams": sorted(summary.streams),
                "day": part.day,
                "sealed": part.sealed,
                "bytes": size,
                "records": summary.records,
                "sha256": _sha256_file(part.path),
                "content_sha256": summary.content.hexdigest(),
                "first_recv_ts_ns": summary.first_recv_ts_ns,
                "last_recv_ts_ns": summary.last_recv_ts_ns,
                "connections": len(summary.connections),
            }
        )
        total_records += summary.records
        total_bytes += size
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "provenance": {**DEFAULT_PROVENANCE, **(provenance or {})},
        "files": files,
        "totals": {"files": len(files), "records": total_records, "bytes": total_bytes},
    }


def dataset_hash(manifest: Mapping[str, Any]) -> str:
    """Content identity of a dataset: format + per-partition record content hashes.

    Independent of the storage location, of whether parts are sealed (gzip) and of the
    provenance labels.
    """
    canonical = {
        "format": manifest.get("format"),
        "format_version": manifest.get("format_version"),
        "partitions": sorted(
            (
                {
                    "partition_id": f["partition_id"],
                    "records": f["records"],
                    "content_sha256": f["content_sha256"],
                }
                for f in manifest.get("files", [])
            ),
            key=lambda f: str(f["partition_id"]),
        ),
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
