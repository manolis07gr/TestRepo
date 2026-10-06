"""File-backed reference series (ETF / rates / VIX proxies) -> canonical events.

The reader handles CSV and Parquet files with these columns:

* ``ts`` (required): ISO-8601 with an explicit offset, integer UTC nanoseconds, or a
  Parquet timestamp column. Naive timestamps are refused unless ``assume_utc=True``.
* ``price``: a last/close/index level. Rows without a quote become trades.
* ``bid`` / ``ask`` (optional): rows with both become top-of-book snapshots, flagged
  ``TOP_OF_BOOK_ONLY``. ``bid_size`` / ``ask_size`` are optional, and missing sizes are
  recorded as 0, meaning unknown. Such snapshots are price references, not executable
  liquidity; the book builder drops zero-size levels.
* ``size`` / ``volume`` (optional): the trade size. Without it, ``default_trade_size``
  (1) marks the row as a unit observation.

Events use ``Venue.REFERENCE`` and ``instrument_key(REFERENCE, symbol)``. A historical
file has no separate receive time, so ``recv_ts_ns`` equals the source timestamp.
Decimals are parsed from text. Parquet float columns cross into Decimal only through
:func:`cma.domain.numbers.from_float` with an explicit ``float_quantum``. Prices must be
non-negative, because the canonical book and trade types reject negatives; offset
series that can go negative, such as rate spreads, before loading them.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from cma.adapters.base import EventFactory, MalformedPayloadError, iso_to_ns, to_decimal
from cma.domain.enums import QualityFlag, Venue
from cma.domain.models import BookLevel, MarketEvent, instrument_key, payload_digest
from cma.domain.numbers import from_float
from cma.domain.time import ns_from_datetime

VENUE: Final = Venue.REFERENCE
_TS_UNIT_NS: Final = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def reference_instrument(symbol: str) -> str:
    return instrument_key(VENUE, symbol)


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    row: int
    ts_ns: int
    price: Decimal | None
    bid: Decimal | None
    ask: Decimal | None
    bid_size: Decimal | None
    ask_size: Decimal | None
    size: Decimal | None


class ReferenceSeriesReader:
    """Reads one symbol's series from a ``.csv`` or ``.parquet`` file."""

    def __init__(
        self,
        path: Path | str,
        *,
        symbol: str,
        float_quantum: Decimal = Decimal("0.00000001"),
        default_trade_size: Decimal | None = Decimal(1),
        assume_utc: bool = False,
        sort: bool = True,
    ) -> None:
        self.path = Path(path)
        self.symbol = symbol
        self.instrument_id = reference_instrument(symbol)
        self._quantum = float_quantum
        self._default_size = default_trade_size
        self._assume_utc = assume_utc
        self._sort = sort

    # ------------------------------------------------------------------ cells

    def _decimal(self, value: Any, what: str) -> Decimal | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, float):
            if math.isnan(value):  # NaN marks a missing Parquet value
                return None
            return from_float(value, self._quantum)
        return to_decimal(value, what)

    def _timestamp(self, value: Any, what: str) -> int:
        if isinstance(value, bool) or value is None:
            raise MalformedPayloadError(f"{what}: missing timestamp")
        if isinstance(value, int):
            return value
        if isinstance(value, datetime):
            if value.tzinfo is None:
                if not self._assume_utc:
                    raise MalformedPayloadError(f"{what}: naive timestamp (pass assume_utc=True)")
                value = value.replace(tzinfo=UTC)
            return ns_from_datetime(value)
        text = str(value).strip()
        if text.lstrip("-").isdigit():
            return int(text)
        has_offset = text.endswith(("Z", "z")) or "+" in text[10:] or "-" in text[10:]
        if self._assume_utc and not has_offset:
            text += "Z"
        return iso_to_ns(text, what)

    # ------------------------------------------------------------------ files

    def _records(self) -> Iterator[Mapping[str, Any]]:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            with self.path.open("r", encoding="utf-8", newline="") as fh:
                yield from csv.DictReader(fh)
        elif suffix in (".parquet", ".pq"):
            yield from self._parquet_records()
        else:
            raise ValueError(f"unsupported reference file type: {self.path.name}")

    def _parquet_records(self) -> Iterator[Mapping[str, Any]]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(self.path)
        columns: dict[str, list[Any]] = {}
        for name in table.column_names:
            column = table.column(name)
            if name == "ts" and pa.types.is_timestamp(column.type):
                if column.type.tz is None and not self._assume_utc:
                    raise MalformedPayloadError(
                        f"{self.path.name}: naive Parquet timestamps (pass assume_utc=True)"
                    )
                factor = _TS_UNIT_NS[column.type.unit]
                columns[name] = [
                    None if v is None else int(v) * factor
                    for v in column.cast(pa.int64()).to_pylist()
                ]
            else:
                columns[name] = column.to_pylist()
        for i in range(table.num_rows):
            yield {name: values[i] for name, values in columns.items()}

    def rows(self) -> list[ReferenceRow]:
        out = []
        for i, record in enumerate(self._records()):
            what = f"{self.path.name} row {i + 1}"
            if "ts" not in record:
                raise MalformedPayloadError(f"{what}: missing 'ts' column")

            def cell(
                *names: str, record: Mapping[str, Any] = record, what: str = what
            ) -> Decimal | None:
                for name in names:
                    if name in record:
                        return self._decimal(record[name], f"{what}.{name}")
                return None

            out.append(
                ReferenceRow(
                    row=i + 1,
                    ts_ns=self._timestamp(record["ts"], f"{what}.ts"),
                    price=cell("price"),
                    bid=cell("bid"),
                    ask=cell("ask"),
                    bid_size=cell("bid_size"),
                    ask_size=cell("ask_size"),
                    size=cell("size", "volume"),
                )
            )
        if self._sort:
            out.sort(key=lambda r: r.ts_ns)
        return out

    def iter_events(self) -> Iterator[MarketEvent]:
        for row in self.rows():
            yield self._event(row)

    def read(self) -> list[MarketEvent]:
        return list(self.iter_events())

    def _event(self, row: ReferenceRow) -> MarketEvent:
        what = f"{self.path.name} row {row.row}"
        canonical = json.dumps(
            [
                self.symbol,
                row.ts_ns,
                *(
                    None if v is None else str(v)
                    for v in (row.price, row.bid, row.ask, row.bid_size, row.ask_size, row.size)
                ),
            ],
            separators=(",", ":"),
        )
        factory = EventFactory(VENUE, row.ts_ns, payload_digest(canonical))
        if row.bid is not None and row.ask is not None:
            bid_qty = row.bid_size if row.bid_size is not None else Decimal(0)
            ask_qty = row.ask_size if row.ask_size is not None else Decimal(0)
            if row.bid < 0 or row.ask < 0 or bid_qty < 0 or ask_qty < 0:
                raise MalformedPayloadError(f"{what}: negative quote or size")
            return factory.snapshot(
                instrument_id=self.instrument_id,
                source_ts_ns=row.ts_ns,
                bids=(BookLevel(row.bid, bid_qty),),
                asks=(BookLevel(row.ask, ask_qty),),
                flags=(QualityFlag.TOP_OF_BOOK_ONLY,),
            )
        if row.price is None:
            raise MalformedPayloadError(f"{what}: neither price nor bid/ask")
        size = row.size if row.size is not None else self._default_size
        if size is None:
            raise MalformedPayloadError(f"{what}: no size column and no default_trade_size")
        return factory.trade(
            instrument_id=self.instrument_id,
            source_ts_ns=row.ts_ns,
            price=row.price,
            size=size,
            aggressor_side=None,
            trade_id=f"{self.symbol}:{row.row}",
        )
