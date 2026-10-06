"""T048 database restart recovery: after a crash (no seal, no clean close) a restarted
recorder/contract store on the same root and SQLite file restores metadata and never
duplicates (or loses) immutable raw events."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from cma.adapters.base import load_json
from cma.adapters.kalshi import parse_market_response, parse_series
from cma.adapters.polymarket import parse_gamma_market
from cma.domain.enums import Venue
from cma.domain.models import PredictionContract, RawMessage
from cma.domain.time import ManualClock
from cma.storage.contracts import ContractStore
from cma.storage.db import open_database
from cma.storage.raw import RawReader, RawRecorder, build_manifest, dataset_hash, discover_parts
from tests.unit.adapters.helpers import T0_NS, fixture_text

# crash simulations deliberately abandon open recorders (no close, no seal)
pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore::ResourceWarning")]


def contracts() -> list[PredictionContract]:
    series = parse_series(load_json(fixture_text("kalshi", "series.json")))
    return [
        parse_market_response(load_json(fixture_text("kalshi", "market.json")), series=series),
        parse_gamma_market(load_json(fixture_text("polymarket", "gamma_market_yes_no.json"))),
        parse_gamma_market(load_json(fixture_text("polymarket", "gamma_market_updown.json"))),
    ]


def messages() -> list[RawMessage]:
    frames = [
        (Venue.KALSHI, "ws", fixture_text("kalshi", "ws_orderbook_snapshot.json"), "k1|ob|2|1"),
        (Venue.KALSHI, "ws", fixture_text("kalshi", "ws_orderbook_delta_yes.json"), "k1|ob|2|2"),
        (Venue.KALSHI, "ws", fixture_text("kalshi", "ws_trade.json"), "trade|d91bc706"),
        (Venue.POLYMARKET, "ws", fixture_text("polymarket", "ws_book.json"), None),
        (Venue.POLYMARKET, "ws", fixture_text("polymarket", "ws_last_trade_price.json"), None),
        (
            Venue.COINBASE,
            "ws",
            fixture_text("coinbase", "ws_match.json"),
            "trade|BTC-USD|370843402",
        ),
    ]
    return [
        RawMessage(
            venue=venue,
            stream=stream,
            recv_ts_ns=T0_NS + i * 1_000,
            payload=payload,
            connection_id=f"{venue.value.lower()}-conn",
            connection_seq=i,
            idempotency_key=key,
        )
        for i, (venue, stream, payload, key) in enumerate(frames, start=1)
    ]


def keys(root: Path) -> list[str]:
    return [m.dedup_key for m in RawReader(root).iter_messages()]


def test_T048_restart_restores_contracts_and_does_not_duplicate_raw_events(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'meta.sqlite'}"
    raw_root = tmp_path / "raw"
    msgs = messages()

    # ---- process 1: record + upsert, then crash (never sealed, never closed)
    db = open_database(db_url)
    store = ContractStore(db, clock=ManualClock(T0_NS))
    assert store.upsert_many(contracts()) == 3
    recorder = RawRecorder(raw_root, db)
    assert all(recorder.append(m) for m in msgs[:4])
    recorder.flush()  # file data, then dedup keys, committed for the first four
    assert all(recorder.append(m) for m in msgs[4:])  # buffered: keys not committed yet
    del recorder  # crash: buffers reach the OS when the process dies; keys never commit
    gc.collect()
    db.close()

    # ---- process 2: same root, same SQLite file
    db = open_database(db_url)  # migrations are idempotent
    store = ContractStore(db, clock=ManualClock(T0_NS + 10))
    assert store.load() == sorted(contracts(), key=lambda c: (c.venue.value, c.contract_id))
    assert store.first_seen_ns(Venue.KALSHI, contracts()[0].contract_id) == T0_NS
    assert not store.upsert_many(contracts())  # unchanged metadata: no rewrites

    recorder = RawRecorder(raw_root, db)
    assert recorder.stats.recovered_keys == 2  # keys of the uncommitted tail re-derived from files
    assert [recorder.append(m) for m in msgs] == [False] * 6  # replay after restart: no dupes
    late = RawMessage(
        venue=Venue.KALSHI,
        stream="ws",
        recv_ts_ns=T0_NS + 99_000,
        payload=fixture_text("kalshi", "ws_orderbook_delta_no.json"),
        connection_id="kalshi-conn-2",
        connection_seq=1,
        idempotency_key="k2|ob|2|3",
    )
    assert recorder.append(late)
    recorder.flush()
    recorded = keys(raw_root)
    assert len(recorded) == 7
    assert len(set(recorded)) == 7
    assert recorded == [m.dedup_key for m in [*msgs, late]]  # receive order preserved

    # sealing after the restart covers the crashed process's parts too
    before = dataset_hash(build_manifest(raw_root))
    sealed = recorder.seal()
    assert sum(p.records for p in sealed) == 7
    rows = db.query("SELECT sealed, n_records, sha256 FROM raw_partitions")
    assert all(r["sealed"] == 1 and r["sha256"] for r in rows)
    assert sum(r["n_records"] for r in rows) == 7
    assert db.scalar("SELECT COUNT(*) FROM raw_event_keys") == 7
    assert dataset_hash(build_manifest(raw_root)) == before

    # ---- process 3: dedup still holds against sealed partitions
    recorder.close()
    db.close()
    db = open_database(db_url)
    recorder = RawRecorder(raw_root, db)
    assert not any(recorder.append(m) for m in [*msgs, late])
    db.close()


def test_T048_torn_write_is_neither_lost_nor_duplicated(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'meta.sqlite'}"
    raw_root = tmp_path / "raw"
    msgs = [m for m in messages() if m.venue is Venue.KALSHI]
    db = open_database(db_url)
    recorder = RawRecorder(raw_root, db)
    for m in msgs[:2]:
        recorder.append(m)
    recorder.flush()
    (torn,) = discover_parts(raw_root)
    del recorder
    gc.collect()
    with torn.path.open("a", encoding="utf-8") as fh:  # crash in the middle of writing msgs[2]
        fh.write('{"venue":"KALSHI","stream":"ws","recv_ts_ns":17913186')
    db.close()

    db = open_database(db_url)
    recorder = RawRecorder(raw_root, db)
    assert [recorder.append(m) for m in msgs] == [False, False, True]
    recorder.flush()
    assert keys(raw_root) == [m.dedup_key for m in msgs]  # each message exactly once
    parts = discover_parts(raw_root)
    assert [p.number for p in parts] == [0, 1]  # the torn file is never appended to
    assert torn.path.read_text(encoding="utf-8").endswith('"recv_ts_ns":17913186')
    db.close()


def test_T048_keys_whose_data_never_reached_disk_are_released(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'meta.sqlite'}"
    raw_root = tmp_path / "raw"
    msgs = [m for m in messages() if m.venue is Venue.KALSHI]
    db = open_database(db_url)
    recorder = RawRecorder(raw_root, db)
    for m in msgs:
        recorder.append(m)
    recorder.close()
    (part,) = discover_parts(raw_root)
    lines = part.path.read_text(encoding="utf-8").splitlines(keepends=True)
    part.path.write_text("".join(lines[:2]), encoding="utf-8")  # OS lost the tail (no fsync)
    db.close()

    db = open_database(db_url)
    recorder = RawRecorder(raw_root, db)
    assert recorder.stats.removed_stale_keys == 1
    assert [recorder.append(m) for m in msgs] == [False, False, True]  # the lost one is re-recorded
    recorder.flush()
    assert keys(raw_root) == [m.dedup_key for m in msgs]
    db.close()
