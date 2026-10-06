"""Generate deterministic book/execution/settlement fixtures (scope s.20).

Writes canonical-event JSONL files plus expected-outcome sidecars under tests/fixtures:
book_normal, book_gap, book_out_of_order, latency_disappearing_liquidity, partial_depth
(.jsonl) and settlement_cases.json.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

from cma.domain.enums import BookSide, DeltaMode, Side, Venue
from cma.domain.models import (
    BookDeltaEvent,
    BookLevel,
    BookSnapshotEvent,
    LevelChange,
    MarketEvent,
    TradeEvent,
)
from cma.domain.time import NS_PER_MS, NS_PER_S
from cma.storage.events_io import write_events_jsonl

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
INST = "KALSHI:FIXTURE-BOOK"
FEED_DELAY = 20 * NS_PER_MS
T0 = 1_790_000_000 * NS_PER_S  # 2026-09-21T...Z, arbitrary fixed epoch


def D(x: str) -> Decimal:
    return Decimal(x)


def snap(seq: int, t: int, bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> MarketEvent:
    return BookSnapshotEvent(
        venue=Venue.KALSHI,
        instrument_id=INST,
        source_ts_ns=t,
        recv_ts_ns=t + FEED_DELAY,
        sequence=seq,
        payload_hash=f"snap-{seq}",
        bids=tuple(BookLevel(D(p), D(q)) for p, q in bids),
        asks=tuple(BookLevel(D(p), D(q)) for p, q in asks),
    )


def delta(seq: int, t: int, side: BookSide, price: str, qty: str, tag: str = "") -> MarketEvent:
    return BookDeltaEvent(
        venue=Venue.KALSHI,
        instrument_id=INST,
        source_ts_ns=t,
        recv_ts_ns=t + FEED_DELAY,
        sequence=seq,
        payload_hash=f"delta-{seq}{tag}",
        changes=(LevelChange(side, D(price), D(qty)),),
        mode=DeltaMode.INCREMENT,
    )


def trade(tid: str, t: int, price: str, size: str, aggressor: Side) -> MarketEvent:
    return TradeEvent(
        venue=Venue.KALSHI,
        instrument_id=INST,
        source_ts_ns=t,
        recv_ts_ns=t + FEED_DELAY,
        payload_hash=f"trade-{tid}",
        price=D(price),
        size=D(size),
        aggressor_side=aggressor,
        trade_id=tid,
    )


def ms(k: int) -> int:
    return T0 + k * NS_PER_MS


BASE_BIDS = [("0.45", "100"), ("0.44", "200"), ("0.43", "300")]
BASE_ASKS = [("0.47", "150"), ("0.48", "250"), ("0.49", "100")]


def book_normal() -> tuple[list[MarketEvent], dict[str, object]]:
    events = [
        snap(1, ms(0), BASE_BIDS, BASE_ASKS),
        delta(2, ms(10), BookSide.BID, "0.45", "50"),
        delta(3, ms(20), BookSide.ASK, "0.47", "-150"),
        delta(4, ms(30), BookSide.ASK, "0.46", "80"),
        trade("t1", ms(35), "0.46", "30", Side.BUY),
        delta(5, ms(36), BookSide.ASK, "0.46", "-30"),
        delta(6, ms(40), BookSide.BID, "0.44", "-200"),
    ]
    expected = {
        "bids": [["0.45", "150"], ["0.43", "300"]],
        "asks": [["0.46", "50"], ["0.48", "250"], ["0.49", "100"]],
        "last_sequence": 6,
        "valid": True,
    }
    return events, expected


def book_gap() -> tuple[list[MarketEvent], dict[str, object]]:
    events = [
        snap(1, ms(0), BASE_BIDS, BASE_ASKS),
        delta(2, ms(10), BookSide.BID, "0.45", "50"),
        delta(4, ms(20), BookSide.ASK, "0.47", "-50"),  # seq 3 missing -> gap
        delta(5, ms(30), BookSide.ASK, "0.48", "-250"),  # ignored until re-snapshot
        snap(10, ms(40), [("0.45", "120"), ("0.44", "200")], [("0.47", "90"), ("0.49", "100")]),
        delta(11, ms(50), BookSide.BID, "0.46", "10"),
    ]
    expected = {
        "invalid_after_index": 2,
        "valid_after_index": 4,
        "bids": [["0.46", "10"], ["0.45", "120"], ["0.44", "200"]],
        "asks": [["0.47", "90"], ["0.49", "100"]],
        "last_sequence": 11,
        "gaps": 1,
    }
    return events, expected


def book_out_of_order() -> tuple[list[MarketEvent], dict[str, object]]:
    events = [
        snap(1, ms(0), BASE_BIDS, BASE_ASKS),
        delta(2, ms(10), BookSide.BID, "0.45", "50"),
        delta(3, ms(20), BookSide.ASK, "0.47", "-50"),
        delta(3, ms(20), BookSide.ASK, "0.47", "-50", tag="-dup"),  # exact duplicate seq
        delta(2, ms(25), BookSide.BID, "0.45", "999", tag="-old"),  # old seq, new content
        delta(4, ms(30), BookSide.BID, "0.43", "-100"),
    ]
    expected = {
        "bids": [["0.45", "150"], ["0.44", "200"], ["0.43", "200"]],
        "asks": [["0.47", "100"], ["0.48", "250"], ["0.49", "100"]],
        "last_sequence": 4,
        "duplicates_ignored": 2,
    }
    return events, expected


def latency_disappearing_liquidity() -> tuple[list[MarketEvent], dict[str, object]]:
    events = [
        snap(1, ms(0), [("0.38", "100")], [("0.40", "50"), ("0.45", "100")]),
        delta(2, ms(50), BookSide.ASK, "0.40", "-50"),  # attractive ask pulled at +50 ms
        delta(3, ms(500), BookSide.BID, "0.38", "10"),
    ]
    expected = {
        "decision_ts_ns": ms(0) + FEED_DELAY,
        "order": {"side": "BUY", "limit": "0.40", "quantity": "50", "tif": "IOC"},
        "fills_at_0ms": "50",
        "fills_at_100ms": "0",
    }
    return events, expected


def partial_depth() -> tuple[list[MarketEvent], dict[str, object]]:
    events = [
        snap(
            1,
            ms(0),
            [("0.48", "40"), ("0.47", "40")],
            [("0.50", "10"), ("0.51", "20"), ("0.53", "30")],
        )
    ]
    expected = {
        "limit_order": {"side": "BUY", "limit": "0.52", "quantity": "50"},
        "limit_filled": "30",
        "limit_vwap": str((D("0.50") * 10 + D("0.51") * 20) / 30),
        "market_order": {"side": "BUY", "quantity": "100"},
        "market_filled": "60",
        "market_vwap": str((D("0.50") * 10 + D("0.51") * 20 + D("0.53") * 30) / 60),
    }
    return events, expected


def settlement_cases() -> list[dict[str, object]]:
    return [
        {"name": "yes_long_wins", "outcome": "YES", "yes_value": "1", "side": "BUY",
         "price": "0.40", "quantity": "10", "expected_pnl": "6.00"},
        {"name": "yes_long_loses", "outcome": "NO", "yes_value": "0", "side": "BUY",
         "price": "0.40", "quantity": "10", "expected_pnl": "-4.00"},
        {"name": "no_long_wins", "outcome": "NO", "yes_value": "0", "side": "BUY_NO",
         "price": "0.35", "quantity": "10", "expected_pnl": "6.50"},
        {"name": "no_long_loses", "outcome": "YES", "yes_value": "1", "side": "BUY_NO",
         "price": "0.35", "quantity": "10", "expected_pnl": "-3.50"},
        {"name": "early_close_yes", "outcome": "YES", "yes_value": "1", "side": "BUY",
         "price": "0.70", "quantity": "5", "expected_pnl": "1.50", "early_close": True},
        {"name": "void_refund", "outcome": "VOID", "yes_value": "0", "side": "BUY",
         "price": "0.40", "quantity": "10", "expected_pnl": "0", "void_policy": "refund_cost"},
        {"name": "scalar_half", "outcome": "SCALAR", "yes_value": "0.5", "side": "BUY",
         "price": "0.40", "quantity": "10", "expected_pnl": "1.00"},
        {"name": "no_long_scalar_half", "outcome": "SCALAR", "yes_value": "0.5",
         "side": "BUY_NO", "price": "0.35", "quantity": "10", "expected_pnl": "1.50"},
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    builders = {
        "book_normal": book_normal,
        "book_gap": book_gap,
        "book_out_of_order": book_out_of_order,
        "latency_disappearing_liquidity": latency_disappearing_liquidity,
        "partial_depth": partial_depth,
    }
    for name, fn in builders.items():
        events, expected = fn()
        write_events_jsonl(OUT / f"{name}.jsonl", events)
        (OUT / f"{name}.expected.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (OUT / "settlement_cases.json").write_text(
        json.dumps(settlement_cases(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote fixtures to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
