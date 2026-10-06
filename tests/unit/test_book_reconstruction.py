"""T007-T010: snapshot+delta reconstruction, gap detection, out-of-order deltas,
crossed-book validation."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from cma.domain.enums import BookSide, DeltaMode, ExecutionMode, QualityFlag, ReasonCode, Venue
from cma.domain.fees import KALSHI_STANDARD
from cma.ingestion.book import BookState, L2BookBuilder
from cma.signals.engine import FairValueEstimate, SignalEngine, Suppression
from tests.factories import FIXTURES, D, config, delta, fixture_events, snapshot

pytestmark = pytest.mark.unit
INST = "KALSHI:FIXTURE-BOOK"


def _levels(snap_levels: tuple) -> list[list[str]]:  # type: ignore[type-arg]
    return [[str(lvl.price), str(lvl.quantity)] for lvl in snap_levels]


def _expected(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((FIXTURES / f"{name}.expected.json").read_text())


def test_T007_snapshot_plus_deltas_reconstructs_expected_levels() -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    for ev in fixture_events("book_normal.jsonl"):
        b.apply(ev)
    exp = _expected("book_normal")
    snap = b.snapshot()
    assert _levels(snap.bids) == exp["bids"]
    assert _levels(snap.asks) == exp["asks"]
    assert snap.sequence == exp["last_sequence"]
    assert b.is_valid and snap.is_valid


def test_T007_absolute_mode_deltas_and_level_removal() -> None:
    b = L2BookBuilder(venue=Venue.POLYMARKET, instrument_id="P", require_sequence=False)
    b.apply(snapshot("P", None, 1, [("0.40", "10")], [("0.45", "5")], venue=Venue.POLYMARKET))
    b.apply(
        delta(
            "P",
            None,
            2,
            [(BookSide.BID, "0.40", "7"), (BookSide.ASK, "0.45", "0"), (BookSide.ASK, "0.46", "9")],
            mode=DeltaMode.ABSOLUTE,
            venue=Venue.POLYMARKET,
        )
    )
    snap = b.snapshot()
    assert _levels(snap.bids) == [["0.40", "7"]]
    assert _levels(snap.asks) == [["0.46", "9"]]


def test_T008_sequence_gap_invalidates_until_resnapshot() -> None:
    events = fixture_events("book_gap.jsonl")
    exp = _expected("book_gap")
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    states = []
    for ev in events:
        res = b.apply(ev)
        states.append((b.state, res.needs_snapshot))
    assert states[exp["invalid_after_index"]] == (BookState.INVALID, True)
    # deltas while invalid are ignored and keep requesting a snapshot
    assert states[3] == (BookState.INVALID, True)
    assert states[exp["valid_after_index"]][0] is BookState.VALID
    snap = b.snapshot()
    assert _levels(snap.bids) == exp["bids"]
    assert _levels(snap.asks) == exp["asks"]
    assert snap.sequence == exp["last_sequence"]
    assert b.counters.gaps == exp["gaps"]
    assert b.is_valid


def test_T008_gap_flag_blocks_snapshot_validity() -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    for ev in fixture_events("book_gap.jsonl")[:3]:
        b.apply(ev)
    snap = b.snapshot()
    assert QualityFlag.SEQUENCE_GAP in snap.quality_flags
    assert not snap.is_valid


def test_T009_old_and_duplicate_deltas_cannot_mutate_book() -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    for ev in fixture_events("book_out_of_order.jsonl"):
        b.apply(ev)
    exp = _expected("book_out_of_order")
    snap = b.snapshot()
    assert _levels(snap.bids) == exp["bids"]
    assert _levels(snap.asks) == exp["asks"]
    assert snap.sequence == exp["last_sequence"]
    assert b.counters.duplicates_ignored == exp["duplicates_ignored"]
    assert b.is_valid


def test_T009_stale_snapshot_ignored_when_book_valid() -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    b.apply(snapshot(INST, 5, 1, [("0.40", "10")], [("0.45", "5")]))
    res = b.apply(snapshot(INST, 4, 2, [("0.10", "1")], [("0.90", "1")]))
    assert not res.applied
    assert b.snapshot().best_bid is not None and b.snapshot().best_bid.price == D("0.40")


def test_T010_crossed_book_flagged_invalid_and_cannot_signal() -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    b.apply(snapshot(INST, 1, 1, [("0.45", "10")], [("0.47", "10")]))
    res = b.apply(delta(INST, 2, 2, [(BookSide.BID, "0.48", "5")]))
    assert res.needs_snapshot
    snap = b.snapshot()
    assert snap.is_crossed
    assert QualityFlag.CROSSED in snap.quality_flags
    assert not snap.is_valid and not b.is_valid

    eng = SignalEngine(
        config=config().signal,
        mode=ExecutionMode.BACKTEST,
        can_trade=lambda _c, _m: True,
        fee_resolver=lambda _c: KALSHI_STANDARD,
    )
    est = FairValueEstimate(
        strategy_id="s",
        strategy_version="1",
        venue=Venue.KALSHI,
        contract_id=INST,
        instrument_id=INST,
        asof_ts_ns=10,
        fair_probability=D("0.99"),  # enormous apparent edge
        feature_watermark_ns=10,
    )
    out = eng.evaluate(est, snap, decision_ts_ns=10)
    assert isinstance(out, Suppression) and out.reason is ReasonCode.BOOK_INVALID


def test_T010_locked_book_is_not_valid_and_negative_size_invalidates() -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    b.apply(snapshot(INST, 1, 1, [("0.45", "10")], [("0.46", "10")]))
    b.apply(delta(INST, 2, 2, [(BookSide.BID, "0.46", "1")]))
    assert b.snapshot().is_locked and not b.snapshot().is_valid
    b2 = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    b2.apply(snapshot(INST, 1, 1, [("0.45", "10")], [("0.47", "10")]))
    res = b2.apply(delta(INST, 2, 2, [(BookSide.BID, "0.45", "-11")]))
    assert res.needs_snapshot and not b2.is_valid
    assert QualityFlag.NEGATIVE_QUANTITY in b2.flags


def test_book_requires_snapshot_before_use() -> None:
    b = L2BookBuilder(venue=Venue.KALSHI, instrument_id=INST)
    res = b.apply(delta(INST, 1, 1, [(BookSide.BID, "0.45", "1")]))
    assert not res.applied and res.needs_snapshot
    assert QualityFlag.AWAITING_SNAPSHOT in b.snapshot().quality_flags
    assert b.snapshot().best_bid is None
    assert Decimal(0) == b.quantity_at(BookSide.BID, D("0.45"))
