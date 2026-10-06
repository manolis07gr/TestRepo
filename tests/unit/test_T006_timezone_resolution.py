"""T006 - contract observation/resolution instants across the 2026 US DST transitions.

US DST 2026: spring forward Sunday 2026-03-08 02:00 EST -> 03:00 EDT; fall back Sunday
2026-11-01 02:00 EDT -> 01:00 EST. "5:00 PM ET" is 22:00Z under EST and 21:00Z under EDT.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, time
from decimal import Decimal

import pytest

from cma.domain.enums import ContractStatus, Venue
from cma.domain.errors import AmbiguousLocalTimeError, NonexistentLocalTimeError
from cma.domain.models import PredictionContract
from cma.domain.time import NS_PER_HOUR, NS_PER_MIN, local_to_utc_ns, ns_from_iso8601
from cma.mapping import MappingRegistry, MappingReviewError, ReviewChecklist, propose_mapping

pytestmark = pytest.mark.unit

ET = "America/New_York"
BRTI_RULES = (
    "If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index "
    "(BRTI) before 5 PM ET is above 111999.99, then the market resolves to Yes."
)
POLY_ABOVE_RULES = (
    'This market will resolve to "Yes" if the Binance 1 minute candle for BTCUSDT 12:00 in the '
    'ET timezone (noon) on the date specified in the title has a final "Close" price higher '
    'than the price specified in the title. Otherwise, this market will resolve to "No".'
)
POLY_UPDOWN_RULES = (
    'This market will resolve to "Up" if the close price is greater than or equal to the open '
    "price for the BTC/USDT 1 Hour candle that begins on the time and date specified in the "
    'title. Otherwise, this market will resolve to "Down". The resolution source for this '
    "market is information from Binance, specifically the BTC/USDT pair."
)


def ns(text: str) -> int:
    return ns_from_iso8601(text)


def kalshi_contract(event_code: str, close_iso: str | None) -> PredictionContract:
    native = f"KXBTCD-{event_code}-T111999.99"
    return PredictionContract(
        venue=Venue.KALSHI,
        contract_id=f"KALSHI:{native}",
        native_id=native,
        event_id=f"KXBTCD-{event_code}",
        title="Bitcoin price at 5pm ET?",
        yes_semantics="Above 111999.99",
        no_semantics="At or below 111999.99",
        open_ts_ns=None,
        close_ts_ns=None if close_iso is None else ns(close_iso),
        resolve_ts_ns=None,
        status=ContractStatus.OPEN,
        tick_size=Decimal("0.01"),
        series_id="KXBTCD",
        rules_text=BRTI_RULES,
        settlement_metadata={"strike_type": "greater", "floor_strike": 111999.99},
    )


def polymarket_contract(
    title: str, rules: str, close_iso: str, yes: str = "Yes"
) -> PredictionContract:
    return PredictionContract(
        venue=Venue.POLYMARKET,
        contract_id="POLYMARKET:0xt006",
        native_id="0xt006",
        event_id="ev-t006",
        title=title,
        yes_semantics=yes,
        no_semantics="No" if yes == "Yes" else "Down",
        open_ts_ns=None,
        close_ts_ns=ns(close_iso),
        resolve_ts_ns=None,
        status=ContractStatus.OPEN,
        tick_size=Decimal("0.01"),
        rules_text=rules,
    )


@pytest.mark.parametrize(
    ("day", "expected_utc"),
    [
        (date(2026, 3, 7), "2026-03-07T22:00:00Z"),  # EST, day before spring-forward
        (date(2026, 3, 8), "2026-03-08T21:00:00Z"),  # EDT since 02:00 local that day
        (date(2026, 3, 9), "2026-03-09T21:00:00Z"),
        (date(2026, 10, 31), "2026-10-31T21:00:00Z"),  # still EDT
        (date(2026, 11, 1), "2026-11-01T22:00:00Z"),  # EST since 02:00 local that day
        (date(2026, 11, 2), "2026-11-02T22:00:00Z"),
    ],
)
def test_t006_five_pm_et_maps_to_correct_utc_across_dst(day: date, expected_utc: str) -> None:
    assert local_to_utc_ns(datetime.combine(day, time(17)), ET) == ns(expected_utc)


def test_t006_fall_back_overlap_requires_explicit_fold() -> None:
    local = datetime(2026, 11, 1, 1, 30)
    with pytest.raises(AmbiguousLocalTimeError):
        local_to_utc_ns(local, ET)
    assert local_to_utc_ns(local, ET, fold=0) == ns("2026-11-01T05:30:00Z")  # first pass, EDT
    assert local_to_utc_ns(local, ET, fold=1) == ns("2026-11-01T06:30:00Z")  # second, EST


def test_t006_spring_forward_gap_raises() -> None:
    with pytest.raises(NonexistentLocalTimeError):
        local_to_utc_ns(datetime(2026, 3, 8, 2, 30), ET)


@pytest.mark.parametrize(
    ("event_code", "expected_end"),
    [
        ("26MAR0617", "2026-03-06T22:00:00Z"),
        ("26MAR0817", "2026-03-08T21:00:00Z"),
        ("26MAR0917", "2026-03-09T21:00:00Z"),
        ("26OCT3017", "2026-10-30T21:00:00Z"),
        ("26NOV0117", "2026-11-01T22:00:00Z"),
        ("26NOV0217", "2026-11-02T22:00:00Z"),
    ],
)
def test_t006_kalshi_parser_observation_instant_across_dst(
    event_code: str, expected_end: str
) -> None:
    result = propose_mapping(kalshi_contract(event_code, expected_end))
    assert result.mapping is not None, result.reason
    mapping = result.mapping
    assert mapping.observation_end_ns == ns(expected_end)
    assert mapping.observation_start_ns == ns(expected_end) - NS_PER_MIN  # 60 s BRTI average
    assert mapping.timezone == ET


def test_t006_kalshi_ticker_in_fall_back_overlap_is_refused() -> None:
    result = propose_mapping(kalshi_contract("26NOV0101", "2026-11-01T05:00:00Z"))
    assert result.mapping is None
    assert result.reason is not None
    assert "ambiguous" in result.reason


def test_t006_kalshi_ticker_in_spring_forward_gap_is_refused() -> None:
    result = propose_mapping(kalshi_contract("26MAR0802", None))
    assert result.mapping is None
    assert result.reason is not None
    assert "does not exist" in result.reason


def test_t006_kalshi_close_time_off_by_one_dst_hour_is_refused() -> None:
    # Nov 2 is EST: 5 PM ET = 22:00Z. A close of 21:00Z is the classic stale-EDT bug.
    result = propose_mapping(kalshi_contract("26NOV0217", "2026-11-02T21:00:00Z"))
    assert result.mapping is None
    assert result.reason is not None
    assert "disagrees" in result.reason


@pytest.mark.parametrize(
    ("title", "close", "expected_start"),
    [
        ("Bitcoin Up or Down - October 6, 12PM ET", "2026-10-06T17:00:00Z", "2026-10-06T16:00:00Z"),
        (
            "Bitcoin Up or Down - November 2, 12PM ET",
            "2026-11-02T18:00:00Z",
            "2026-11-02T17:00:00Z",
        ),
        ("Bitcoin Up or Down - March 8, 12PM ET", "2026-03-08T17:00:00Z", "2026-03-08T16:00:00Z"),
        ("Bitcoin Up or Down - March 7, 12PM ET", "2026-03-07T18:00:00Z", "2026-03-07T17:00:00Z"),
    ],
)
def test_t006_polymarket_12pm_et_title_gives_correct_utc_instant(
    title: str, close: str, expected_start: str
) -> None:
    result = propose_mapping(polymarket_contract(title, POLY_UPDOWN_RULES, close, yes="Up"))
    assert result.mapping is not None, result.reason
    mapping = result.mapping
    assert mapping.observation_start_ns == ns(expected_start)  # candle opens at 12:00 ET
    assert mapping.observation_end_ns == ns(expected_start) + NS_PER_HOUR
    assert mapping.timezone == ET


@pytest.mark.parametrize(
    ("title", "close", "expected_open"),
    [
        (
            "Will the price of Bitcoin be above $110,000 on March 6?",
            "2026-03-06T17:00:00Z",
            "2026-03-06T17:00:00Z",
        ),
        (
            "Will the price of Bitcoin be above $110,000 on March 9?",
            "2026-03-09T16:00:00Z",
            "2026-03-09T16:00:00Z",
        ),
        ("Bitcoin above 110,000 on October 30?", "2026-10-30T16:00:00Z", "2026-10-30T16:00:00Z"),
        ("Bitcoin above 110,000 on November 2?", "2026-11-02T17:00:00Z", "2026-11-02T17:00:00Z"),
    ],
)
def test_t006_polymarket_noon_et_candle_rule_across_dst(
    title: str, close: str, expected_open: str
) -> None:
    result = propose_mapping(polymarket_contract(title, POLY_ABOVE_RULES, close))
    assert result.mapping is not None, result.reason
    mapping = result.mapping
    assert mapping.observation_start_ns == ns(expected_open)  # 12:00 ET candle opens
    assert mapping.observation_end_ns == ns(expected_open) + NS_PER_MIN  # and closes at 12:01


def test_t006_review_rejects_mapping_with_dst_shifted_observation() -> None:
    contract = kalshi_contract("26NOV0217", "2026-11-02T22:00:00Z")
    parsed = propose_mapping(contract).mapping
    assert parsed is not None
    # A hand-made mapping that used the EDT offset after fall-back (one hour early).
    wrong_end = parsed.observation_end_ns - NS_PER_HOUR
    wrong = dataclasses.replace(
        parsed,
        observation_end_ns=wrong_end,
        observation_start_ns=wrong_end - NS_PER_MIN,
        event_family="",
    )
    registry = MappingRegistry()
    registry.propose(wrong, "carol.analyst", contract=contract)
    with pytest.raises(MappingReviewError, match="observation_end"):
        registry.review(contract.contract_id, "alice.reviewer", ReviewChecklist.all_affirmed())
    registry.propose(parsed, "parser:kalshi-series/v1")
    reviewed = registry.review(
        contract.contract_id, "alice.reviewer", ReviewChecklist.all_affirmed()
    )
    assert reviewed.observation_end_ns == ns("2026-11-02T22:00:00Z")
    assert reviewed.version == 2
