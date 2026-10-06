"""Deterministic DRAFT proposals and fail-closed ambiguity handling of the contract parsers."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import pytest

from cma.domain.enums import ContractStatus, MappingStatus, Operator, Venue
from cma.domain.models import PredictionContract
from cma.domain.time import NS_PER_HOUR, NS_PER_MIN, ns_from_iso8601
from cma.mapping import (
    MappingRegistry,
    MappingReviewError,
    ReviewChecklist,
    load_series_semantics,
    parse_kalshi_contract,
    parse_polymarket_contract,
    propose_mapping,
)

pytestmark = pytest.mark.unit

BRTI_RULES = (
    "If the simple average of the sixty seconds of CF Benchmarks' Real-Time Index before "
    "5 PM EDT is above the strike, the market resolves to Yes."
)
ABOVE_RULES = (
    'This market will resolve to "Yes" if the Binance 1 minute candle for BTCUSDT 12:00 in the '
    'ET timezone (noon) on the date specified in the title has a final "Close" price higher '
    'than the price specified in the title. Otherwise, this market will resolve to "No".'
)
HOURLY_RULES = (
    'This market will resolve to "Up" if the close price is greater than or equal to the open '
    "price for the BTC/USDT 1 Hour candle that begins on the time and date specified in the "
    'title. Otherwise, this market will resolve to "Down". The resolution source for this '
    "market is information from Binance, specifically the BTC/USDT pair."
)
TWAP_RULES = (
    'This market will resolve to "Up" if the Bitcoin price at the end of the time range '
    "specified in the title is greater than or equal to the price at the beginning of that "
    "range. Prices are 60-second TWAPs computed from the Chainlink BTC/USD data stream."
)


def ns(text: str) -> int:
    return ns_from_iso8601(text)


def kalshi(native: str, metadata: dict[str, Any], **kw: Any) -> PredictionContract:
    fields: dict[str, Any] = {
        "venue": Venue.KALSHI,
        "contract_id": f"KALSHI:{native}",
        "native_id": native,
        "event_id": native.rsplit("-", 1)[0],
        "title": "Bitcoin price at 5pm EDT?",
        "yes_semantics": "Yes",
        "no_semantics": "No",
        "open_ts_ns": None,
        "close_ts_ns": ns("2026-10-06T21:00:00Z"),
        "resolve_ts_ns": None,
        "status": ContractStatus.OPEN,
        "tick_size": Decimal("0.01"),
        "series_id": native.split("-", maxsplit=1)[0],
        "rules_text": BRTI_RULES,
        "settlement_metadata": metadata,
    }
    fields.update(kw)
    return PredictionContract(**fields)


def poly(title: str, rules: str, close: str, **kw: Any) -> PredictionContract:
    fields: dict[str, Any] = {
        "venue": Venue.POLYMARKET,
        "contract_id": "POLYMARKET:0xparser",
        "native_id": "0xparser",
        "event_id": "ev",
        "title": title,
        "yes_semantics": "Yes",
        "no_semantics": "No",
        "open_ts_ns": None,
        "close_ts_ns": ns(close),
        "resolve_ts_ns": None,
        "status": ContractStatus.OPEN,
        "tick_size": Decimal("0.01"),
        "rules_text": rules,
    }
    fields.update(kw)
    return PredictionContract(**fields)


# ----------------------------------------------------------------------------- Kalshi


@pytest.mark.parametrize(
    ("strike_type", "metadata", "operator", "strikes"),
    [
        ("greater", {"floor_strike": 111999.99}, Operator.GT, ("111999.99",)),
        ("greater_or_equal", {"floor_strike": "112000"}, Operator.GE, ("112000",)),
        ("less", {"cap_strike": 108000}, Operator.LT, ("108000",)),
        ("less_or_equal", {"cap_strike": 108000.5}, Operator.LE, ("108000.5",)),
        (
            "between",
            {"floor_strike": 111000, "cap_strike": 111499.99},
            Operator.BETWEEN,
            ("111000", "111499.99"),
        ),
    ],
)
def test_kalshi_strike_types(
    strike_type: str, metadata: dict[str, Any], operator: Operator, strikes: tuple[str, ...]
) -> None:
    contract = kalshi("KXBTC-26OCT0617-B111250", {"strike_type": strike_type, **metadata})
    result = parse_kalshi_contract(contract)
    assert result.mapping is not None, result.reason
    m = result.mapping
    assert m.operator is operator
    assert m.strikes == tuple(Decimal(s) for s in strikes)
    assert m.review_status is MappingStatus.DRAFT
    assert m.version == 1
    assert m.underlyings == ("BTC-USD",)
    assert m.resolution_source == "CF_BENCHMARKS_BRTI"
    assert m.observation_method == "AVG_60S_BEFORE"
    assert m.early_close_rule == "NO_EARLY_CLOSE;MISSING_INDEX_DATA_RESOLVES_NO"
    assert m.event_family == "BTC-USD|AVG_60S_BEFORE|2026-10-06T21:00:00.000000000Z"
    assert result.proposed_by.startswith("parser:")
    if operator is Operator.BETWEEN:
        assert "inclusive" in m.notes
        assert "<=" in m.outcome_semantics


def test_kalshi_eth_series_and_unverified_warning() -> None:
    meta = {"strike_type": "greater", "floor_strike": 4000}
    ethd = parse_kalshi_contract(kalshi("KXETHD-26OCT0617-T4000", meta))
    assert ethd.mapping is not None
    assert ethd.mapping.underlyings == ("ETH-USD",)
    assert ethd.mapping.resolution_source == "CF_BENCHMARKS_ETHUSD_RTI"
    assert ethd.warnings == ()
    eth = parse_kalshi_contract(
        kalshi(
            "KXETH-26OCT0617-B4000",
            {"strike_type": "between", "floor_strike": 3950, "cap_strike": 4049.99},
        )
    )
    assert eth.mapping is not None
    assert any("UNVERIFIED" in w for w in eth.warnings)


@pytest.mark.parametrize(
    ("contract", "reason"),
    [
        (kalshi("KXBTCD-26OCT0617-T1", {"strike_type": "functional"}), "deterministic"),
        (kalshi("KXBTCD-26OCT0617-T1", {"strike_type": "custom"}), "deterministic"),
        (kalshi("KXBTCD-26OCT0617-T1", {"strike_type": "structured"}), "deterministic"),
        (kalshi("KXBTCD-26OCT0617-T1", {"strike_type": "weird"}), "unknown strike_type"),
        (kalshi("KXBTCD-26OCT0617-T1", {}), "strike_type missing"),
        (kalshi("KXBTCD-26OCT0617-T111999.99", {"strike_type": "greater"}), "floor_strike only"),
        (
            kalshi(
                "KXBTCD-26OCT0617-T111999.99",
                {"strike_type": "greater", "floor_strike": 1, "cap_strike": 2},
            ),
            "floor_strike only",
        ),
        (
            kalshi(
                "KXBTC-26OCT0617-B1", {"strike_type": "between", "floor_strike": 5, "cap_strike": 5}
            ),
            "floor_strike < cap_strike",
        ),
        (
            kalshi(
                "KXBTCD-26OCT0617-T111999.99", {"strike_type": "greater", "floor_strike": 110999.99}
            ),
            "ticker strike",
        ),
        (kalshi("KXDOGE-26OCT0617-T1", {"strike_type": "greater", "floor_strike": 1}), "no entry"),
        (
            kalshi("KXBTC15M-26OCT061715-T1", {"strike_type": "greater", "floor_strike": 1}),
            "not enabled",
        ),
        (
            kalshi("KXBTCD-26XYZ0617-T1", {"strike_type": "greater", "floor_strike": 1}),
            "unknown month",
        ),
        (
            kalshi("KXBTCD-26FEB3017-T1", {"strike_type": "greater", "floor_strike": 1}),
            "invalid ticker date",
        ),
        (
            kalshi("KXBTCD-26OCT0625-T1", {"strike_type": "greater", "floor_strike": 1}),
            "invalid ticker hour",
        ),
        (kalshi("KXBTCD-BAD", {"strike_type": "greater", "floor_strike": 1}), "is not <SERIES>"),
        (
            kalshi(
                "KXBTCD-26OCT0617-T111999.99",
                {"strike_type": "greater", "floor_strike": 111999.99},
                rules_text="Coinbase spot at 5pm",
            ),
            "does not mention",
        ),
        (
            kalshi(
                "KXBTCD-26OCT0617-T111999.99",
                {"strike_type": "greater", "floor_strike": 111999.99},
                can_close_early=True,
            ),
            "can_close_early",
        ),
        (
            kalshi(
                "KXBTCD-26OCT0617-T111999.99",
                {"strike_type": "greater", "floor_strike": 111999.99},
                series_id="KXBTC",
            ),
            "is not <SERIES>",
        ),
    ],
)
def test_kalshi_ambiguity_returns_none_with_reason(
    contract: PredictionContract, reason: str
) -> None:
    result = parse_kalshi_contract(contract)
    assert result.mapping is None
    assert result.reason is not None
    assert reason in result.reason, result.reason


def test_kalshi_missing_rules_or_close_only_warns() -> None:
    meta = {"strike_type": "greater", "floor_strike": 111999.99}
    result = parse_kalshi_contract(
        kalshi("KXBTCD-26OCT0617-T111999.99", meta, rules_text="", close_ts_ns=None)
    )
    assert result.mapping is not None
    assert len(result.warnings) == 2


def test_kalshi_parser_rejects_other_venues() -> None:
    contract = poly("Bitcoin above 1 on October 6?", ABOVE_RULES, "2026-10-06T16:00:00Z")
    assert parse_kalshi_contract(contract).mapping is None
    assert parse_polymarket_contract(kalshi("KXBTCD-26OCT0617-T1", {})).mapping is None
    other = dataclasses.replace(contract, venue=Venue.COINBASE, contract_id="COINBASE:x")
    assert propose_mapping(other).reason is not None


# ------------------------------------------------------------------------- Polymarket


@pytest.mark.parametrize(
    ("title", "metadata", "strike"),
    [
        ("Bitcoin above ___ on October 6?", {"group_item_title": "110,000"}, "110000"),
        ("Bitcoin above ___ on October 6?", {"group_item_title": "↑ $112k"}, "112000"),
        ("Will the price of Bitcoin be above $110,000 on October 6?", {}, "110000"),
        ("Will the price of Bitcoin be above $110k on October 6?", {}, "110000"),
        ("Bitcoin above 109,500.5 on Oct. 6?", {}, "109500.5"),
    ],
)
def test_polymarket_threshold_templates(title: str, metadata: dict[str, Any], strike: str) -> None:
    result = parse_polymarket_contract(
        poly(title, ABOVE_RULES, "2026-10-06T16:00:00Z", settlement_metadata=metadata)
    )
    assert result.mapping is not None, result.reason
    m = result.mapping
    assert m.operator is Operator.GT  # "higher than": strictly greater
    assert m.strikes == (Decimal(strike),)
    assert m.underlyings == ("BTCUSDT@BINANCE",)
    assert m.resolution_source == "BINANCE_BTCUSDT_1M_CLOSE"
    assert m.observation_method == "CANDLE_CLOSE_1M"
    assert m.observation_start_ns == ns("2026-10-06T16:00:00Z")
    assert m.observation_end_ns == ns("2026-10-06T16:01:00Z")
    assert m.early_close_rule == "UNSPECIFIED"  # reviewer must determine the fallback rule
    assert m.review_status is MappingStatus.DRAFT


def test_polymarket_ge_and_below_comparisons() -> None:
    ge_rules = ABOVE_RULES.replace("higher than", "higher than or equal to")
    ge = parse_polymarket_contract(
        poly("Bitcoin above 110,000 on October 6?", ge_rules, "2026-10-06T16:00:00Z")
    )
    assert ge.mapping is not None
    assert ge.mapping.operator is Operator.GE
    lt_rules = ABOVE_RULES.replace("higher than", "lower than")
    lt = parse_polymarket_contract(
        poly("Bitcoin below 110,000 on October 6?", lt_rules, "2026-10-06T16:00:00Z")
    )
    assert lt.mapping is not None
    assert lt.mapping.operator is Operator.LT
    eth = parse_polymarket_contract(
        poly(
            "Ethereum above 4,000 on October 6?",
            ABOVE_RULES.replace("BTCUSDT", "ETHUSDT"),
            "2026-10-06T16:00:00Z",
        )
    )
    assert eth.mapping is not None
    assert eth.mapping.underlyings == ("ETHUSDT@BINANCE",)


def test_polymarket_hourly_up_down() -> None:
    result = parse_polymarket_contract(
        poly(
            "Bitcoin Up or Down - October 6, 5PM ET",
            HOURLY_RULES,
            "2026-10-06T22:00:00Z",
            yes_semantics="Up",
        )
    )
    assert result.mapping is not None, result.reason
    m = result.mapping
    assert m.operator is Operator.UP
    assert m.strikes == ()
    assert m.observation_start_ns == ns("2026-10-06T21:00:00Z")
    assert m.observation_end_ns == ns("2026-10-06T22:00:00Z")
    assert m.observation_method == "CANDLE_OPEN_CLOSE_1H"
    assert m.resolution_source == "BINANCE_BTCUSDT_1H_CANDLE"
    assert m.rounding_rule == "TIES_RESOLVE_YES"
    assert ">=" in m.outcome_semantics
    strict = parse_polymarket_contract(
        poly(
            "Bitcoin Up or Down - October 6, 5PM ET",
            HOURLY_RULES.replace("greater than or equal to", "greater than"),
            "2026-10-06T22:00:00Z",
            yes_semantics="Up",
        )
    )
    assert strict.mapping is not None
    assert strict.mapping.rounding_rule == "TIES_RESOLVE_NO"


def test_polymarket_chainlink_twap_slug_market() -> None:
    start = ns("2026-10-06T21:00:00Z")
    slug = f"btc-updown-15m-{start // 1_000_000_000}"
    contract = poly(
        "Bitcoin Up or Down - October 6, 5:00PM-5:15PM ET",
        TWAP_RULES,
        "2026-10-06T21:15:00Z",
        yes_semantics="Up",
        native_id=slug,
        contract_id=f"POLYMARKET:{slug}",
    )
    result = parse_polymarket_contract(contract)
    assert result.mapping is not None, result.reason
    m = result.mapping
    assert (m.observation_start_ns, m.observation_end_ns) == (start, start + 15 * NS_PER_MIN)
    assert m.observation_method == "TWAP_60S"
    assert m.resolution_source == "CHAINLINK_DATA_STREAMS_TWAP60"
    assert m.rounding_rule == "TIES_RESOLVE_YES"
    # title disagreeing with the slug start, and pre-2026-08-07 rule versions, are refused
    wrong_title = dataclasses.replace(
        contract, title="Bitcoin Up or Down - October 6, 6:00PM-6:15PM ET"
    )
    assert "disagrees" in (parse_polymarket_contract(wrong_title).reason or "")
    old_start = ns("2026-08-01T00:00:00Z") // 1_000_000_000
    old = dataclasses.replace(contract, native_id=f"btc-updown-15m-{old_start}", title="x")
    assert "before the TWAP rule" in (parse_polymarket_contract(old).reason or "")
    # an unreadable title start (no AM/PM) only skips the cross-check: the slug is authoritative
    vague_title = dataclasses.replace(contract, title="Bitcoin Up or Down - October 6, 5-5:15PM ET")
    vague = parse_polymarket_contract(vague_title)
    assert vague.mapping is not None
    assert any("not cross-checked" in w for w in vague.warnings)
    no_twap = dataclasses.replace(
        contract, rules_text=TWAP_RULES.replace("60-second TWAPs", "prices")
    )
    assert "TWAP" in (parse_polymarket_contract(no_twap).reason or "")


@pytest.mark.parametrize(
    ("contract", "reason"),
    [
        (
            poly(
                "Bitcoin Up or Down on October 6?",
                HOURLY_RULES,
                "2026-10-06T16:00:00Z",
                yes_semantics="Up",
            ),
            "daily Up/Down",
        ),
        (
            poly("What price will Bitcoin hit in October?", ABOVE_RULES, "2026-10-31T16:00:00Z"),
            "no supported template",
        ),
        (
            poly("Bitcoin above 110,000 on October 6?", "", "2026-10-06T16:00:00Z"),
            "rules text is empty",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES.replace("Binance", "Kraken"),
                "2026-10-06T16:00:00Z",
            ),
            "does not name the binance",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES.replace("BTCUSDT", "BTCUSD"),
                "2026-10-06T16:00:00Z",
            ),
            "does not name the pair",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES + " Coinbase may be used.",
                "2026-10-06T16:00:00Z",
            ),
            "other sources",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES.replace("1 minute", "5 minute"),
                "2026-10-06T16:00:00Z",
            ),
            "1M candle",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES + " See the 1 hour chart.",
                "2026-10-06T16:00:00Z",
            ),
            "conflicting candle",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES.replace("higher than", "near"),
                "2026-10-06T16:00:00Z",
            ),
            "comparison",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES + " Or 13:00 UTC.",
                "2026-10-06T16:00:00Z",
            ),
            "exactly one observation time",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES.replace("12:00", "5:00").replace(" (noon)", ""),
                "2026-10-06T16:00:00Z",
            ),
            "morning or evening",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES.replace("12:00 in the ET", "12:00 PM EST in the"),
                "2026-10-06T17:00:00Z",
            ),
            "EST",
        ),
        (
            poly("Bitcoin above 110,000 on October 6?", ABOVE_RULES, "2026-12-20T16:00:00Z"),
            "far from the close date",
        ),
        (
            poly("Bitcoin above 110,000 on October 6?", ABOVE_RULES, "2026-10-06T19:00:00Z"),
            "disagrees with contract close",
        ),
        (
            poly("Bitcoin above 110,000 on October 6, 2027?", ABOVE_RULES, "2026-10-06T16:00:00Z"),
            "far from the close date",
        ),
        (
            poly("Bitcoin above 110,000 on Smarch 6?", ABOVE_RULES, "2026-10-06T16:00:00Z"),
            "unknown month",
        ),
        (
            poly("Bitcoin above ___ on October 6?", ABOVE_RULES, "2026-10-06T16:00:00Z"),
            "group_item_title is missing",
        ),
        (
            poly(
                "Bitcoin above ___ on October 6?",
                ABOVE_RULES,
                "2026-10-06T16:00:00Z",
                settlement_metadata={"group_item_title": "110k-112k"},
            ),
            "exactly one number",
        ),
        (
            poly(
                "Bitcoin above 110,000 on October 6?",
                ABOVE_RULES,
                "2026-10-06T16:00:00Z",
                yes_semantics="No",
            ),
            "orientation",
        ),
        (
            poly(
                "Bitcoin Up or Down - November 1, 1AM ET",
                HOURLY_RULES,
                "2026-11-01T07:00:00Z",
                yes_semantics="Up",
            ),
            "ambiguous",
        ),
        (
            poly(
                "Bitcoin Up or Down - March 8, 2AM ET",
                HOURLY_RULES,
                "2026-03-08T08:00:00Z",
                yes_semantics="Up",
            ),
            "does not exist",
        ),
        (
            poly(
                "Bitcoin Up or Down - October 6, 5PM ET",
                HOURLY_RULES.replace("greater than or equal to", "versus"),
                "2026-10-06T22:00:00Z",
                yes_semantics="Up",
            ),
            "compares",
        ),
    ],
)
def test_polymarket_ambiguity_returns_none_with_reason(
    contract: PredictionContract, reason: str
) -> None:
    result = parse_polymarket_contract(contract)
    assert result.mapping is None
    assert result.reason is not None
    assert reason in result.reason, result.reason


def test_polymarket_year_inference_uses_close_time() -> None:
    result = parse_polymarket_contract(
        poly("Bitcoin above 110,000 on January 2?", ABOVE_RULES, "2027-01-02T17:00:00Z")
    )
    assert result.mapping is not None
    assert result.mapping.observation_start_ns == ns("2027-01-02T17:00:00Z")  # EST in January
    no_anchor = poly(
        "Bitcoin above 110,000 on January 2?", ABOVE_RULES, "2027-01-02T17:00:00Z", close_ts_ns=None
    )
    assert "no year" in (parse_polymarket_contract(no_anchor).reason or "")


def test_series_semantics_table_flags() -> None:
    table = load_series_semantics()
    assert table.kalshi["KXBTCD"].verified
    assert table.kalshi["KXBTCD"].auto_parse
    assert not table.kalshi["KXETH"].verified
    assert not table.kalshi["KXBTC15M"].auto_parse
    assert table.kalshi["KXBTCD"].observation_window_ns == 60 * 1_000_000_000
    assert table.template("THRESHOLD") is not None
    assert table.template("NOPE") is None
    for row in table.kalshi.values():
        assert row.verified or "UNVERIFIED" in row.verification_note


def test_parsed_kalshi_mapping_passes_review_but_polymarket_needs_human_amendment() -> None:
    registry = MappingRegistry()
    checklist = ReviewChecklist.all_affirmed("rules checked")
    k = kalshi("KXBTCD-26OCT0617-T111999.99", {"strike_type": "greater", "floor_strike": 111999.99})
    km = propose_mapping(k)
    assert km.mapping is not None
    registry.propose(km.mapping, km.proposed_by, contract=k)
    assert registry.review(k.contract_id, "alice.reviewer", checklist).version == 1

    p = poly("Bitcoin above 110,000 on October 6?", ABOVE_RULES, "2026-10-06T16:00:00Z")
    pm = propose_mapping(p)
    assert pm.mapping is not None
    registry.propose(pm.mapping, pm.proposed_by, contract=p)
    with pytest.raises(MappingReviewError, match="early_close_rule is unspecified"):
        registry.review(p.contract_id, "alice.reviewer", checklist)
    amended = dataclasses.replace(pm.mapping, early_close_rule="NO_EARLY_CLOSE;FALLBACK_PER_RULES")
    registry.propose(amended, "alice.reviewer")
    reviewed = registry.review(p.contract_id, "alice.reviewer", checklist)
    assert reviewed.version == 2
    assert reviewed.review_status is MappingStatus.REVIEWED


def test_parse_is_deterministic() -> None:
    contract = poly(
        "Bitcoin Up or Down - October 6, 5PM ET",
        HOURLY_RULES,
        "2026-10-06T22:00:00Z",
        yes_semantics="Up",
    )
    assert propose_mapping(contract) == propose_mapping(contract)
    assert propose_mapping(contract).mapping is not None
    assert NS_PER_HOUR == 60 * NS_PER_MIN
