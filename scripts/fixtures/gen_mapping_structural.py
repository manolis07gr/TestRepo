"""Generate the mapping/structural fixtures (scope s.20; tests T013 and T043).

Writes two small deterministic JSON documents under ``tests/fixtures``:

* ``nested_contracts.json`` - a Kalshi-style KXBTCD family (BRTI 60-second average before
  17:00 ET on 2026-10-06) with three "X > K" strikes, canonical YES books per scenario and the
  expected detection. Scenario ``violation`` contains exactly one structural violation:
  bid(K=110999.99) = 0.78 exceeds ask(K=109999.99) = 0.71 by more than fees.
* ``equivalence_good_bad.json`` - one truly equivalent venue pair and one deceptively
  similar but semantically different pair (BRTI 60 s average before 5 PM ET vs Binance
  BTC/USDT 1-minute close at 16:00 UTC = 12 PM ET: different cutoff instant, timezone and
  resolution source).

Expected values are HAND-COMPUTED below (the arithmetic is spelled out next to each number)
and are never produced by the detector under test. The Polymarket side of the "good" pair is
SYNTHETIC: a hypothetical market constructed with BRTI semantics so that a truly equivalent
pair exists in the fixture; it does not claim that such a market is listed.

Usage: ``python scripts/fixtures/gen_mapping_structural.py [--out-dir tests/fixtures]``
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from cma.domain.time import NS_PER_MIN, iso_from_ns, local_to_utc_ns

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
GENERATOR = "scripts/fixtures/gen_mapping_structural.py"
ET = "America/New_York"


def _local(tz: str, day: date, hh: int, mm: int = 0) -> int:
    return local_to_utc_ns(datetime.combine(day, time(hh, mm)), tz)


# 17:00 ET on 2026-10-06 is 21:00Z (EDT, UTC-4); the BRTI window is the 60 s before it.
OBS_END = _local(ET, date(2026, 10, 6), 17)
OBS_START = OBS_END - NS_PER_MIN
BRTI_FAMILY = f"BTC-USD|AVG_60S_BEFORE|{iso_from_ns(OBS_END)}"
KALSHI_EARLY_CLOSE = "NO_EARLY_CLOSE;MISSING_INDEX_DATA_RESOLVES_NO"
BOOK_TS = iso_from_ns(OBS_END - 60 * NS_PER_MIN)  # books observed at 16:00 ET

FEE_MODEL = {
    "venue_schedule_id": "kalshi-standard",
    "model": "quadratic",
    "taker_rate": "0.07",
    "rounding": "ceil to 0.01 per single-fill order",
    "formula": "fee = ceil_to_cent(0.07 * C * P * (1 - P)), P = canonical YES price",
}


def _kalshi_contract(strike: str) -> dict[str, Any]:
    native = f"KXBTCD-26OCT0617-T{strike}"
    return {
        "venue": "KALSHI",
        "contract_id": f"KALSHI:{native}",
        "native_id": native,
        "event_id": "KXBTCD-26OCT0617",
        "series_id": "KXBTCD",
        "title": "Bitcoin price today at 5pm EDT?",
        "yes_semantics": f"Above {strike}",
        "no_semantics": f"At or below {strike}",
        "open_ts": iso_from_ns(_local(ET, date(2026, 10, 6), 13)),
        "close_ts": iso_from_ns(OBS_END),
        "resolve_ts": None,
        "status": "OPEN",
        "tick_size": "0.01",
        "can_close_early": False,
        "rules_text": (
            "If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time "
            f"Index (BRTI) before 5 PM EDT is above {strike} at 5 PM EDT on Oct 6, 2026, then "
            "the market resolves to Yes."
        ),
        # venue payloads carry JSON numbers; parsers convert them explicitly
        "settlement_metadata": {"strike_type": "greater", "floor_strike": float(strike)},
    }


def _brti_mapping(venue: str, contract_id: str, strike: str) -> dict[str, Any]:
    return {
        "venue": venue,
        "contract_id": contract_id,
        "underlyings": ["BTC-USD"],
        "operator": "GT",
        "strikes": [strike],
        "observation_start": iso_from_ns(OBS_START),
        "observation_end": iso_from_ns(OBS_END),
        "observation_method": "AVG_60S_BEFORE",
        "timezone": ET,
        "resolution_source": "CF_BENCHMARKS_BRTI",
        "rounding_rule": "NONE",
        "early_close_rule": KALSHI_EARLY_CLOSE,
        "outcome_semantics": (
            f"YES iff the 60 s simple average of BRTI before 2026-10-06 17:00 ET > {strike}"
        ),
        "event_family": BRTI_FAMILY,
        "notes": "fixture mapping (hand-written)",
    }


def _book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "recv_ts": BOOK_TS,
        "source_ts": BOOK_TS,
        "bids": [list(level) for level in bids],
        "asks": [list(level) for level in asks],
    }


def build_nested_contracts() -> dict[str, Any]:
    strikes = ["109999.99", "110999.99", "111999.99"]
    contracts = [_kalshi_contract(k) for k in strikes]
    low, mid, high = (c["contract_id"] for c in contracts)
    violation_books = {
        # P(X > 109999.99): bid 0.69 / ask 0.71
        low: _book([("0.69", "200"), ("0.68", "300")], [("0.71", "120"), ("0.72", "200")]),
        # P(X > 110999.99) mispriced ABOVE the lower strike: bid 0.78 > ask(low) 0.71
        mid: _book([("0.78", "80"), ("0.74", "100")], [("0.80", "90"), ("0.81", "50")]),
        # P(X > 111999.99): consistent with both others
        high: _book([("0.29", "100"), ("0.28", "150")], [("0.31", "100"), ("0.32", "120")]),
    }
    fee_absorbed_books = {
        low: violation_books[low],
        # bid(mid) 0.72 > ask(low) 0.71, but 0.01 < fees ~0.0144 + 0.0141 per contract
        mid: _book([("0.72", "80"), ("0.70", "100")], [("0.80", "90")]),
        high: violation_books[high],
    }
    return {
        "schema": "cma.fixture.nested_contracts/v1",
        "generator": GENERATOR,
        "description": (
            "KXBTCD-style 'X > K' family (BRTI 60 s average before 2026-10-06 17:00 ET = "
            "21:00Z) with three strikes; one known structural violation in scenario "
            "'violation' (buy YES 109999.99 at its ask 0.71, sell YES 110999.99 at its bid 0.78)."
        ),
        "fee_model": FEE_MODEL,
        "contracts": contracts,
        "mappings": [
            _brti_mapping("KALSHI", c["contract_id"], k)
            for c, k in zip(contracts, strikes, strict=True)
        ],
        "scenarios": [
            {
                "name": "violation",
                "config": {"tolerance": "0.005", "walk_levels": False},
                "books": violation_books,
                "expected": [
                    {
                        "kind": "NESTED_THRESHOLD",
                        # Q = min(ask(low) depth 120, bid(mid) depth 80) = 80
                        "quantity": "80",
                        "legs": [
                            # fee = ceil(0.07 * 80 * 0.71 * 0.29 = 1.15304) = 1.16
                            {
                                "contract_id": low,
                                "side": "BUY",
                                "price": "0.71",
                                "quantity": "80",
                                "fee": "1.16",
                            },
                            # fee = ceil(0.07 * 80 * 0.78 * 0.22 = 0.96096) = 0.97
                            {
                                "contract_id": mid,
                                "side": "SELL",
                                "price": "0.78",
                                "quantity": "80",
                                "fee": "0.97",
                            },
                        ],
                        "gross_profit": "5.60",  # 80 * (0.78 - 0.71)
                        "total_fees": "2.13",  # 1.16 + 0.97
                        "net_profit": "3.47",  # 5.60 - 2.13
                        "gross_edge_per_unit": "0.07",
                        "fees_per_unit": "0.026625",  # 2.13 / 80
                        "net_edge_per_unit": "0.043375",  # 3.47 / 80
                    }
                ],
            },
            {
                "name": "violation_walk_levels",
                "config": {"tolerance": "0", "walk_levels": True},
                "books": violation_books,
                "expected": [
                    {
                        "kind": "NESTED_THRESHOLD",
                        # step 1: 80 @ (0.71, 0.78); marginal 0.07 - 0.014413 - 0.012012 > 0
                        # step 2: 40 @ (0.71, 0.74); marginal 0.03 - 0.014413 - 0.013468 > 0
                        # step 3: (0.72, 0.74): 0.02 - 0.014112 - 0.013468 < 0 -> stop
                        "quantity": "120",
                        "legs": [
                            # ceil(0.07 * 120 * 0.71 * 0.29 = 1.72956) = 1.73
                            {
                                "contract_id": low,
                                "side": "BUY",
                                "price": "0.71",
                                "quantity": "120",
                                "fee": "1.73",
                            },
                            {
                                "contract_id": mid,
                                "side": "SELL",
                                "price": "0.78",
                                "quantity": "80",
                                "fee": "0.97",
                            },
                            # ceil(0.07 * 40 * 0.74 * 0.26 = 0.53872) = 0.54
                            {
                                "contract_id": mid,
                                "side": "SELL",
                                "price": "0.74",
                                "quantity": "40",
                                "fee": "0.54",
                            },
                        ],
                        "gross_profit": "6.80",  # 80 * 0.07 + 40 * 0.03
                        "total_fees": "3.24",  # 1.73 + 0.97 + 0.54
                        "net_profit": "3.56",  # 6.80 - 3.24 (> 3.47 for the first step alone)
                    }
                ],
            },
            {
                "name": "fee_absorbed",
                "config": {"tolerance": "0", "walk_levels": False},
                "books": fee_absorbed_books,
                # gross 0.72 - 0.71 = 0.01 per unit < fees (1.16 + 1.13 for 80 units)
                "expected": [],
            },
        ],
    }


def _polymarket_contract(
    native: str, title: str, rules: str, close_ns: int, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "venue": "POLYMARKET",
        "contract_id": f"POLYMARKET:{native}",
        "native_id": native,
        "event_id": f"event-{native}",
        "series_id": None,
        "title": title,
        "yes_semantics": "Yes",
        "no_semantics": "No",
        "open_ts": iso_from_ns(close_ns - 3 * 24 * 60 * NS_PER_MIN),
        "close_ts": iso_from_ns(close_ns),
        "resolve_ts": None,
        "status": "OPEN",
        "tick_size": "0.01",
        "can_close_early": False,
        "rules_text": rules,
        "settlement_metadata": metadata,
    }


def build_equivalence_good_bad() -> dict[str, Any]:
    strike = "111999.99"
    kalshi = _kalshi_contract(strike)
    good_native = "0xsynthetic-brti-5pm-et-111999.99"
    good = _polymarket_contract(
        good_native,
        "Will the BRTI 60-second average be above $111,999.99 at 5 PM ET on October 6, 2026?",
        "SYNTHETIC FIXTURE MARKET. Resolves Yes if the simple average of the sixty seconds of CF "
        "Benchmarks' Bitcoin Real-Time Index (BRTI) before 5 PM ET on October 6, 2026 is above "
        "111,999.99. If index data is missing or incomplete the market resolves No.",
        OBS_END,
        {"group_item_title": "111,999.99"},
    )
    bad_native = "0xbtc-above-111999.99-oct-6"
    bad_open = _local("UTC", date(2026, 10, 6), 16)  # 16:00 UTC == 12:00 PM EDT
    bad = _polymarket_contract(
        bad_native,
        "Bitcoin above $111,999.99 on October 6?",
        'This market will resolve to "Yes" if the Binance 1 minute candle for BTCUSDT 16:00 in '
        'the UTC timezone on the date specified in the title has a final "Close" price higher '
        'than the price specified in the title. Otherwise, this market will resolve to "No".',
        bad_open,
        {},
    )
    good_mapping = _brti_mapping("POLYMARKET", good["contract_id"], strike)
    good_mapping["notes"] = "SYNTHETIC equivalent of the Kalshi contract (fixture only)"
    bad_mapping = {
        "venue": "POLYMARKET",
        "contract_id": bad["contract_id"],
        "underlyings": ["BTCUSDT@BINANCE"],
        "operator": "GT",
        "strikes": [strike],
        "observation_start": iso_from_ns(bad_open),
        "observation_end": iso_from_ns(bad_open + NS_PER_MIN),  # 1-minute candle close
        "observation_method": "CANDLE_CLOSE_1M",
        "timezone": "UTC",
        "resolution_source": "BINANCE_BTCUSDT_1M_CLOSE",
        "rounding_rule": "NONE",
        # deliberately identical to the Kalshi rule so only cutoff/timezone/source differ
        "early_close_rule": KALSHI_EARLY_CLOSE,
        "outcome_semantics": (
            "YES iff Binance BTCUSDT 1m candle opening 16:00 UTC closes > 111999.99"
        ),
        "event_family": f"BTCUSDT@BINANCE|CANDLE_CLOSE_1M|{iso_from_ns(bad_open + NS_PER_MIN)}",
        "notes": "fixture mapping (hand-written); deceptively similar title, different semantics",
    }
    kalshi_entry = {
        "contract": kalshi,
        "mapping": _brti_mapping("KALSHI", kalshi["contract_id"], strike),
    }
    return {
        "schema": "cma.fixture.equivalence_good_bad/v1",
        "generator": GENERATOR,
        "description": (
            "Good pair: Kalshi KXBTCD 5 PM ET BRTI 60 s average > 111999.99 vs a SYNTHETIC "
            "Polymarket market with identical semantics. Bad pair: the same Kalshi contract vs "
            "'Bitcoin above $111,999.99 on October 6?' resolved on the Binance BTC/USDT 1-minute "
            "candle at 16:00 UTC (12 PM ET): different cutoff instant, timezone and source."
        ),
        "good_pair": {
            "a": kalshi_entry,
            "b": {"contract": good, "mapping": good_mapping},
            "expected": {"relation": "SAME", "equivalent": True, "mismatched_fields": []},
        },
        "bad_pair": {
            "a": kalshi_entry,
            "b": {"contract": bad, "mapping": bad_mapping},
            "expected": {
                "relation": "NONE",
                "equivalent": False,
                "mismatched_fields": [
                    "underlyings",
                    "observation_start",
                    "observation_end",
                    "observation_method",
                    "timezone",
                    "resolution_source",
                ],
            },
        },
    }


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def documents() -> dict[str, dict[str, Any]]:
    return {
        "nested_contracts.json": build_nested_contracts(),
        "equivalence_good_bad.json": build_equivalence_good_bad(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, document in documents().items():
        (out_dir / name).write_text(render(document), encoding="utf-8")
        print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
