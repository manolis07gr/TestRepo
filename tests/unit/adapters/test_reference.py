"""File-backed reference series (CSV and Parquet) -> trades / top-of-book snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cma.adapters.base import MalformedPayloadError
from cma.adapters.reference import ReferenceSeriesReader
from cma.domain.enums import QualityFlag, Venue
from cma.domain.models import BookSnapshotEvent, TradeEvent
from cma.domain.time import ns_from_iso8601
from tests.unit.adapters.helpers import FIXTURES

pytestmark = pytest.mark.unit
D = Decimal


def test_csv_trades_are_sorted_with_exact_decimals_and_default_size() -> None:
    events = ReferenceSeriesReader(FIXTURES / "reference" / "spy_trades.csv", symbol="SPY").read()
    assert all(isinstance(e, TradeEvent) for e in events)
    assert [e.venue for e in events] == [Venue.REFERENCE] * 3
    assert [e.price for e in events if isinstance(e, TradeEvent)] == [
        D("671.20"),
        D("671.25"),
        D("671.30"),
    ]
    assert events[0].source_ts_ns == ns_from_iso8601("2026-10-06T20:30:00Z") == events[0].recv_ts_ns
    last = events[2]
    assert isinstance(last, TradeEvent)
    assert last.size == D(1)
    assert last.source_ts_ns == 1_791_318_601_000_000_000  # integer ns column value
    assert events[0].instrument_id == "REFERENCE:SPY"


def test_csv_quotes_become_top_of_book_snapshots() -> None:
    events = ReferenceSeriesReader(FIXTURES / "reference" / "vix_quotes.csv", symbol="VIX").read()
    first, second, third = events
    assert isinstance(first, BookSnapshotEvent)
    assert first.bids[0].price == D("16.40")
    assert first.asks[0].quantity == D(12)
    assert QualityFlag.TOP_OF_BOOK_ONLY in first.quality_flags
    assert isinstance(second, BookSnapshotEvent)
    assert second.bids[0].quantity == D(0)
    assert isinstance(third, TradeEvent)
    assert third.price == D("16.47")


def test_reader_is_deterministic() -> None:
    reader = ReferenceSeriesReader(FIXTURES / "reference" / "vix_quotes.csv", symbol="VIX")
    assert [e.event_id for e in reader.read()] == [e.event_id for e in reader.read()]


def test_parquet_with_timestamp_and_float_columns(tmp_path: Path) -> None:
    path = tmp_path / "rates.parquet"
    table = pa.table(
        {
            "ts": pa.array(
                [
                    datetime(2026, 10, 6, 20, 30, tzinfo=UTC),
                    datetime(2026, 10, 6, 20, 31, tzinfo=UTC),
                ],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "price": pa.array([4.125, 4.1375], type=pa.float64()),
            "size": pa.array([D("10.5"), D("2")], type=pa.decimal128(10, 2)),
        }
    )
    pq.write_table(table, path)
    events = ReferenceSeriesReader(path, symbol="US10Y", float_quantum=D("0.0001")).read()
    assert [e.source_ts_ns for e in events] == [
        ns_from_iso8601("2026-10-06T20:30:00Z"),
        ns_from_iso8601("2026-10-06T20:31:00Z"),
    ]
    trades = [e for e in events if isinstance(e, TradeEvent)]
    assert [t.price for t in trades] == [D("4.1250"), D("4.1375")]  # explicit float quantum
    assert trades[0].size == D("10.50")


def test_naive_timestamps_and_bad_rows_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "naive.csv"
    path.write_text("ts,price\n2026-10-06T20:30:00,1.0\n", encoding="utf-8")
    with pytest.raises(MalformedPayloadError):
        ReferenceSeriesReader(path, symbol="X").read()
    assert ReferenceSeriesReader(path, symbol="X", assume_utc=True).read()[0].source_ts_ns == (
        ns_from_iso8601("2026-10-06T20:30:00Z")
    )
    bad = tmp_path / "bad.csv"
    bad.write_text("ts,price\n2026-10-06T20:30:00Z,abc\n", encoding="utf-8")
    with pytest.raises(MalformedPayloadError):
        ReferenceSeriesReader(bad, symbol="X").read()
    with pytest.raises(ValueError, match="unsupported"):
        ReferenceSeriesReader(tmp_path / "x.txt", symbol="X").read()
