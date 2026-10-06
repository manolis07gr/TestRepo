"""Append-only raw store: layout, dedup, ordering across partitions, sealing, manifests."""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path

import pytest

from cma.domain.enums import Venue
from cma.domain.models import RawMessage
from cma.domain.time import NS_PER_DAY
from cma.storage.db import open_database
from cma.storage.raw import (
    RawReader,
    RawRecorder,
    RawStoreError,
    build_manifest,
    dataset_hash,
    discover_parts,
    sanitize_stream,
)

pytestmark = pytest.mark.unit
T0 = 1_791_318_600_000_000_000  # 2026-10-06T20:30:00Z


def msg(
    i: int,
    *,
    venue: Venue = Venue.KALSHI,
    stream: str = "ws",
    ts: int | None = None,
    conn: str = "c1",
    key: str | None = None,
    payload: str | None = None,
) -> RawMessage:
    return RawMessage(
        venue=venue,
        stream=stream,
        recv_ts_ns=T0 + i if ts is None else ts,
        payload=payload if payload is not None else f'{{"n": {i}}}',
        connection_id=conn,
        connection_seq=i,
        idempotency_key=key,
    )


def test_layout_one_json_line_per_message(tmp_path: Path) -> None:
    rec = RawRecorder(tmp_path)
    assert rec.append(msg(1, stream="ws:orderbook_delta", key="conn|ob|2|1"))
    rec.close()
    (part,) = discover_parts(tmp_path)
    assert part.relative_path(tmp_path) == (
        "venue=KALSHI/stream=ws_orderbook_delta/day=2026-10-06/part-00000.jsonl"
    )
    line = part.path.read_text(encoding="utf-8")
    assert line.endswith("\n")
    assert line.count("\n") == 1
    for field in ("venue", "stream", "recv_ts_ns", "connection_id", "connection_seq"):
        assert f'"{field}"' in line
    assert '"idempotency_key":"conn|ob|2|1"' in line
    assert '"payload_hash"' in line
    assert sanitize_stream("rest:orderbook:KXBTCD-26OCT0617-T110999.99") == (
        "rest_orderbook_KXBTCD-26OCT0617-T110999.99"
    )


def test_duplicates_are_rejected_by_dedup_key(tmp_path: Path) -> None:
    rec = RawRecorder(tmp_path)
    assert rec.append(msg(1, key="trade|1"))
    assert not rec.append(msg(2, key="trade|1"))  # same natural key, different envelope
    assert rec.append(msg(3))
    assert not rec.append(msg(3))  # keyless: same payload hash + recv time
    assert rec.append(msg(3, ts=T0 + 99))  # same payload received at another time
    rec.close()
    assert rec.stats.appended == 3
    assert rec.stats.duplicates == 2


def test_reader_merges_partitions_in_receive_order(tmp_path: Path) -> None:
    rec = RawRecorder(tmp_path)
    messages = [
        msg(5, venue=Venue.POLYMARKET, conn="p"),
        msg(1, venue=Venue.KALSHI, stream="ws", conn="k"),
        msg(3, venue=Venue.KALSHI, stream="rest:trades", conn="r"),
        msg(2, venue=Venue.COINBASE, conn="cb"),
        msg(7, venue=Venue.KALSHI, stream="ws", conn="k"),
        msg(4, venue=Venue.KALSHI, ts=T0 + NS_PER_DAY, conn="k"),  # next UTC day
    ]
    for m in messages:
        assert rec.append(m)
    rec.close()
    assert len(discover_parts(tmp_path)) == 5
    out = list(RawReader(tmp_path).iter_messages())
    keys = [(m.recv_ts_ns, m.connection_id, m.connection_seq) for m in out]
    assert keys == sorted(keys)
    assert len(out) == 6
    assert out == sorted(messages, key=lambda m: (m.recv_ts_ns, m.connection_id, m.connection_seq))
    kalshi_ws = list(RawReader(tmp_path).iter_messages(venues=[Venue.KALSHI], streams=["ws"]))
    assert [m.connection_seq for m in kalshi_ws] == [1, 7, 4]
    window = list(RawReader(tmp_path).iter_messages(start_ns=T0 + 2, end_ns=T0 + 6))
    assert [m.recv_ts_ns - T0 for m in window] == [2, 3, 5]


def test_ties_are_broken_by_connection_then_sequence(tmp_path: Path) -> None:
    rec = RawRecorder(tmp_path)
    for m in (msg(2, ts=T0, conn="b"), msg(1, ts=T0, conn="b"), msg(9, ts=T0, conn="a")):
        rec.append(m)
    rec.close()
    out = [(m.connection_id, m.connection_seq) for m in RawReader(tmp_path).iter_messages()]
    assert out == [("a", 9), ("b", 1), ("b", 2)]
    assert rec.stats.rotations >= 1  # out-of-order appends rotate to keep parts sorted


def test_rotation_by_size(tmp_path: Path) -> None:
    rec = RawRecorder(tmp_path, max_records_per_part=2)
    for i in range(5):
        rec.append(msg(i))
    rec.close()
    assert [p.number for p in discover_parts(tmp_path)] == [0, 1, 2]
    assert [m.connection_seq for m in RawReader(tmp_path).iter_messages()] == [0, 1, 2, 3, 4]


def test_payload_text_round_trips_exactly(tmp_path: Path) -> None:
    tricky = '{"t":"line1\\nline2","u":"é😀","raw":"\r\n\t"}\n tail'
    rec = RawRecorder(tmp_path)
    rec.append(msg(1, payload=tricky))
    rec.close()
    (out,) = RawReader(tmp_path).iter_messages()
    assert out.payload == tricky
    assert out.payload_hash == hashlib.sha256(tricky.encode()).hexdigest()


def test_seal_gzips_hashes_and_registers_partitions(tmp_path: Path) -> None:
    db = open_database(f"sqlite:///{tmp_path / 'meta.sqlite'}")
    root = tmp_path / "raw"
    rec = RawRecorder(root, db)
    for i in range(4):
        rec.append(msg(i, venue=Venue.KALSHI if i % 2 else Venue.POLYMARKET))
    sealed = rec.seal()
    assert len(sealed) == 2
    for part in sealed:
        assert part.path.name.endswith(".jsonl.gz")
        assert hashlib.sha256(part.path.read_bytes()).hexdigest() == part.sha256
        assert not part.path.with_suffix("").exists()  # plain file removed
        with gzip.open(part.path, "rt", encoding="utf-8") as fh:
            assert sum(1 for _ in fh) == part.records == 2
    rows = db.query("SELECT partition_id, sealed, sha256, n_records FROM raw_partitions ORDER BY 1")
    assert [(r["sealed"], r["n_records"]) for r in rows] == [(1, 2), (1, 2)]
    assert {r["sha256"] for r in rows} == {p.sha256 for p in sealed}
    assert [m.connection_seq for m in RawReader(root).iter_messages()] == [0, 1, 2, 3]
    # appending after a seal opens a new part instead of touching the sealed one
    rec.append(msg(10))
    rec.close()
    assert any(not p.sealed and p.number == 1 for p in discover_parts(root))
    db.close()


def test_sealing_is_deterministic(tmp_path: Path) -> None:
    digests = []
    for name in ("a", "b"):
        rec = RawRecorder(tmp_path / name)
        for i in range(3):
            rec.append(msg(i))
        digests.append(rec.seal()[0].sha256)
    assert digests[0] == digests[1]


def test_manifest_and_dataset_hash_are_stable(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    rec = RawRecorder(root)
    for i in range(6):
        rec.append(msg(i, venue=[Venue.KALSHI, Venue.COINBASE][i % 2], conn=f"c{i % 2}"))
    rec.close()
    manifest = build_manifest(root, provenance={"collection_version": "test-1"})
    assert manifest == build_manifest(root, provenance={"collection_version": "test-1"})
    assert manifest["totals"]["records"] == 6
    assert manifest["totals"]["files"] == 2
    entry = manifest["files"][0]
    assert {"path", "bytes", "records", "sha256", "content_sha256", "connections"} <= entry.keys()
    assert manifest["provenance"]["collection_version"] == "test-1"
    digest = dataset_hash(manifest)
    rec.seal()  # sealing changes file bytes, not dataset content
    sealed_manifest = build_manifest(root)
    assert all(f["sealed"] for f in sealed_manifest["files"])
    assert dataset_hash(sealed_manifest) == digest
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    assert dataset_hash(build_manifest(moved)) == digest  # location independent
    rec2 = RawRecorder(root)
    rec2.append(msg(100))
    rec2.close()
    assert dataset_hash(build_manifest(root)) != digest


def test_partial_final_line_is_skipped_but_corruption_is_detected(tmp_path: Path) -> None:
    rec = RawRecorder(tmp_path)
    rec.append(msg(1))
    rec.append(msg(2))
    rec.close()
    (part,) = discover_parts(tmp_path)
    with part.path.open("a", encoding="utf-8") as fh:
        fh.write('{"venue":"KALSHI","stream":"ws","recv_ts_ns":')  # crash mid-write
    assert [m.connection_seq for m in RawReader(tmp_path).iter_messages()] == [1, 2]
    text = part.path.read_text(encoding="utf-8").replace('\\"n\\": 1', '\\"n\\": 9')
    part.path.write_text(text, encoding="utf-8")
    with pytest.raises(RawStoreError, match="hash mismatch"):
        list(RawReader(tmp_path).iter_messages())


def test_closed_recorder_refuses_appends(tmp_path: Path) -> None:
    with RawRecorder(tmp_path) as rec:
        rec.append(msg(1))
    with pytest.raises(RawStoreError):
        rec.append(msg(2))
    assert len(list(RawReader(tmp_path).iter_messages())) == 1
