"""Polymarket parsing: Gamma contracts, token books, price_change formats, trades."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from cma.adapters.base import MalformedPayloadError, load_json
from cma.adapters.polymarket import (
    POLYMARKET_UNMAPPED_FEE_SCHEDULE,
    PolymarketAdapter,
    PolymarketRestClient,
    build_market_subscription,
    parse_gamma_event,
    parse_gamma_market,
    parse_tick_size_change,
    select_fee_schedule,
)
from cma.domain.enums import BookSide, ContractStatus, DeltaMode, QualityFlag, Side, Venue
from cma.domain.fees import get_fee_schedule
from cma.domain.models import BookDeltaEvent, BookLevel, BookSnapshotEvent, StatusEvent, TradeEvent
from cma.domain.time import ManualClock, ns_from_iso8601
from cma.ingestion.book import BookManager
from cma.ingestion.rest import HttpFetcher
from tests.unit.adapters.helpers import (
    CONDITION,
    NO_INST,
    NO_TOKEN,
    T0_NS,
    YES_INST,
    YES_TOKEN,
    fixture_text,
    mock_client,
    raw_msg,
)

pytestmark = pytest.mark.unit
D = Decimal
ADAPTER = PolymarketAdapter()


def poly(name: str, *, stream: str = "ws") -> list:  # type: ignore[type-arg]
    return ADAPTER.parse(raw_msg(Venue.POLYMARKET, fixture_text("polymarket", name), stream=stream))


def gamma(name: str = "gamma_market_yes_no.json") -> dict:  # type: ignore[type-arg]
    data = load_json(fixture_text("polymarket", name))
    assert isinstance(data, dict)
    return data


# ------------------------------------------------------------------ Gamma contracts


def test_gamma_yes_no_market_to_contract() -> None:
    c = parse_gamma_market(gamma())
    assert c.venue is Venue.POLYMARKET
    assert c.contract_id == f"POLYMARKET:{CONDITION}"
    assert c.native_id == CONDITION
    assert dict(c.outcome_instruments) == {"YES": YES_INST, "NO": NO_INST}
    assert (c.yes_semantics, c.no_semantics) == ("Yes", "No")
    assert c.tick_size == D("0.01")
    assert c.resolve_ts_ns == ns_from_iso8601("2026-10-31T16:00:00Z") == c.close_ts_ns
    assert c.open_ts_ns == ns_from_iso8601("2026-10-01T15:20:00Z")  # acceptingOrdersTimestamp
    assert c.status is ContractStatus.OPEN
    assert c.event_id == "61234"
    assert c.series_id == "bitcoin-above-on-october-31"
    assert c.rules_text.startswith('This market will resolve to "Yes"')
    assert c.settlement_metadata["resolution_source"].startswith("https://www.binance.com")
    assert c.settlement_metadata["neg_risk"] is False
    assert c.settlement_metadata["fee_schedule"]["rate"] == "0.07"
    assert c.fee_schedule_id == "polymarket-crypto-taker"
    assert c.settlement_metadata["fee_schedule_basis"] == "category:crypto"


def test_updown_market_maps_up_to_yes_and_window_from_slug() -> None:
    c = parse_gamma_market(gamma("gamma_market_updown.json"))
    assert c.yes_semantics == "Up"
    assert c.settlement_metadata["yes_mapping"] == "up_label"
    assert c.outcome_instruments["YES"].endswith(
        "11015470973684177829729219287262166995141465048508201953575582100565462316088"
    )
    meta = c.settlement_metadata
    assert meta["observation_window_start_ns"] == ns_from_iso8601("2026-10-06T20:15:00Z")
    assert meta["observation_window_end_ns"] == ns_from_iso8601("2026-10-06T20:30:00Z")
    assert meta["updown"] == {"coin": "btc", "interval": "15m"}
    # Gamma startDate is object creation (~24h early), recorded separately
    assert meta["created_ts_ns"] == ns_from_iso8601("2026-10-05T20:16:02.511Z")
    assert c.fee_schedule_id == "polymarket-crypto-taker"  # crypto by slug pattern


def test_first_outcome_is_yes_for_named_outcomes() -> None:
    m = gamma()
    m["outcomes"] = json.dumps(["Lakers", "Celtics"])
    c = parse_gamma_market(m)
    assert c.yes_semantics == "Lakers"
    assert c.settlement_metadata["yes_mapping"] == "first_outcome"


def test_fee_schedule_selection_rules() -> None:
    m = gamma()
    m["feesEnabled"] = False
    assert select_fee_schedule(m) == ("polymarket-no-fee", "fees_disabled")
    m = gamma()
    m["events"][0]["category"] = "Sports"
    m["events"][0]["tags"] = [{"slug": "sports"}]
    m["slug"] = "lakers-vs-celtics"
    assert select_fee_schedule(m)[0] == "polymarket-sports-taker"
    m = gamma()
    m.pop("events")
    m["slug"] = "some-market"
    m["feeSchedule"] = {"rate": Decimal("0.04"), "exponent": 1}
    assert select_fee_schedule(m) == ("polymarket-finance-taker", "fee_schedule_rate")
    m["feeSchedule"] = {"rate": Decimal("0.031"), "exponent": 1}
    schedule, basis = select_fee_schedule(m)
    assert (schedule, basis) == (POLYMARKET_UNMAPPED_FEE_SCHEDULE, "unmapped")
    with pytest.raises(KeyError):  # unmapped fee schedules fail closed downstream
        get_fee_schedule(schedule)
    for registered in ("polymarket-no-fee", "polymarket-crypto-taker", "polymarket-sports-taker"):
        assert get_fee_schedule(registered).schedule_id == registered
    assert parse_gamma_market(gamma(), fee_schedule_id="zero").fee_schedule_id == "zero"


def test_gamma_status_mapping() -> None:
    m = gamma()
    m["closed"] = True
    m["umaResolutionStatus"] = "resolved"
    assert parse_gamma_market(m).status is ContractStatus.SETTLED
    m["umaResolutionStatus"] = "proposed"
    assert parse_gamma_market(m).status is ContractStatus.CLOSED


def test_non_binary_gamma_market_is_malformed() -> None:
    m = gamma()
    m["outcomes"] = json.dumps(["A", "B", "C"])
    with pytest.raises(MalformedPayloadError, match="2 outcomes"):
        parse_gamma_market(m)


def test_gamma_event_with_nested_markets() -> None:
    (raw_event,) = load_json(fixture_text("polymarket", "gamma_events_page.json"))
    event = parse_gamma_event(raw_event)
    assert event.event_id == "61234"
    assert len(event.markets) == 1
    assert event.markets[0].event_id == "61234"


# ------------------------------------------------------------------ WS market channel


def test_book_message_is_a_token_snapshot_with_hash_and_no_sequence() -> None:
    (snap,) = poly("ws_book.json")
    assert isinstance(snap, BookSnapshotEvent)
    assert snap.instrument_id == YES_INST  # keyed by token, not by condition id
    assert snap.bids == (
        BookLevel(D("0.50"), D(15)),
        BookLevel(D("0.49"), D(20)),
        BookLevel(D("0.48"), D(30)),
    )
    assert snap.asks == (
        BookLevel(D("0.52"), D(25)),
        BookLevel(D("0.53"), D(60)),
        BookLevel(D("0.54"), D(10)),
    )
    assert snap.checksum == "0x0a1b2c3d4e5f60718293a4b5c6d7e8f901234567"
    assert snap.sequence is None
    assert snap.source_ts_ns == T0_NS + 123_000_000


def test_array_frames_produce_one_event_per_item_without_mixing_yes_and_no() -> None:
    yes, no = poly("ws_book_array.json")
    assert (yes.instrument_id, no.instrument_id) == (YES_INST, NO_INST)
    assert (yes.sub_index, no.sub_index) == (0, 1)
    assert isinstance(no, BookSnapshotEvent)
    assert no.bids[0].price == D("0.48")


def test_new_price_change_format_groups_absolute_changes_by_asset() -> None:
    yes, no = poly("ws_price_change_new.json")
    assert isinstance(yes, BookDeltaEvent)
    assert isinstance(no, BookDeltaEvent)
    assert yes.instrument_id == YES_INST
    assert no.instrument_id == NO_INST
    assert yes.mode is DeltaMode.ABSOLUTE
    assert [(c.side, c.price, c.quantity) for c in yes.changes] == [
        (BookSide.BID, D("0.5"), D(200)),
        (BookSide.BID, D("0.49"), D(0)),
    ]
    assert [(c.side, c.price) for c in no.changes] == [(BookSide.ASK, D("0.5"))]
    assert yes.checksum == "77ab1c5d6e7f8091a2b3c4d5e6f708192a3b4c5d"


def test_legacy_price_change_format() -> None:
    (delta,) = poly("ws_price_change_legacy.json")
    assert isinstance(delta, BookDeltaEvent)
    assert delta.instrument_id == YES_INST
    assert [(c.side, c.price, c.quantity) for c in delta.changes] == [
        (BookSide.ASK, D("0.53"), D(0)),
        (BookSide.ASK, D("0.51"), D(40)),
    ]


def test_book_then_price_changes_reconstruct_token_book() -> None:
    books = BookManager(sequence_free_venues=frozenset({Venue.POLYMARKET}))
    for name in ("ws_book.json", "ws_price_change_new.json", "ws_price_change_legacy.json"):
        for event in poly(name):
            if event.instrument_id == YES_INST:
                assert books.apply(event).applied
    book = books.get(YES_INST)
    assert book is not None
    assert book.is_valid
    assert book.levels(BookSide.BID)[:2] == [(D("0.5"), D(200)), (D("0.48"), D(30))]
    assert book.levels(BookSide.ASK)[:2] == [(D("0.51"), D(40)), (D("0.52"), D(25))]


def test_last_trade_price_is_a_trade_with_synthetic_stable_id() -> None:
    (trade,) = poly("ws_last_trade_price.json")
    assert isinstance(trade, TradeEvent)
    assert trade.instrument_id == YES_INST
    assert trade.price == D("0.52")
    assert trade.size == D("219.217767")
    assert trade.aggressor_side is Side.BUY
    (again,) = poly("ws_last_trade_price.json")
    assert trade.trade_id == again.trade_id
    assert len(trade.trade_id) == 24


def test_tick_size_change_best_bid_ask_and_keepalives_emit_nothing() -> None:
    assert poly("ws_tick_size_change.json") == []
    tick = parse_tick_size_change(
        json.loads(fixture_text("polymarket", "ws_tick_size_change.json"))
    )
    assert tick.new_tick_size == D("0.001")
    assert tick.old_tick_size == D("0.01")
    assert poly("ws_best_bid_ask.json") == []
    assert ADAPTER.parse(raw_msg(Venue.POLYMARKET, "PONG")) == []
    assert ADAPTER.parse(raw_msg(Venue.POLYMARKET, "[]")) == []
    assert ADAPTER.parse(raw_msg(Venue.POLYMARKET, '{"event_type": "new_market"}')) == []


def test_market_resolved_emits_settled_status_events() -> None:
    events = poly("ws_market_resolved.json")
    assert all(isinstance(e, StatusEvent) and e.status is ContractStatus.SETTLED for e in events)
    assert [e.instrument_id for e in events] == [f"POLYMARKET:{CONDITION}", YES_INST, NO_INST]
    assert "winning_outcome=Yes" in events[0].detail


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ('{"event_type": "book", "bids": [], "asks": []}', "asset_id"),
        (
            '{"event_type": "book", "asset_id": "1", "asks": [],'
            ' "bids": [{"price": "1.5", "size": "1"}]}',
            "outside",
        ),
        ('{"event_type": "price_change", "market": "m"}', "price_changes"),
        (
            '{"event_type": "last_trade_price", "asset_id": "1", "price": "0.5", "size": "0"}',
            "positive",
        ),
        ('{"no_event_type": true}', "event_type"),
    ],
)
def test_schema_violations_are_malformed(payload: str, match: str) -> None:
    with pytest.raises(MalformedPayloadError, match=match):
        ADAPTER.parse(raw_msg(Venue.POLYMARKET, payload))


def test_idempotency_keys_scope_snapshots_to_the_connection() -> None:
    book_key = ADAPTER.idempotency_key("ws", fixture_text("polymarket", "ws_book.json"))
    assert book_key is not None
    assert book_key.startswith("conn:book|")
    pc_key = ADAPTER.idempotency_key("ws", fixture_text("polymarket", "ws_price_change_new.json"))
    assert pc_key is not None
    assert pc_key.startswith("pc|")
    trade_key = ADAPTER.idempotency_key(
        "ws", fixture_text("polymarket", "ws_last_trade_price.json")
    )
    assert trade_key is not None
    assert trade_key.startswith("trade|")
    multi = ADAPTER.idempotency_key("ws", fixture_text("polymarket", "ws_book_array.json"))
    assert multi is not None
    assert multi.startswith("conn:multi|")
    assert ADAPTER.idempotency_key("ws", "PONG") is None


def test_subscription_frame() -> None:
    assert json.loads(build_market_subscription([YES_TOKEN, NO_TOKEN])) == {
        "assets_ids": [YES_TOKEN, NO_TOKEN],
        "type": "market",
        "custom_feature_enabled": True,
    }


# ------------------------------------------------------------------ REST


def test_rest_clients_book_history_trades_and_gamma() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.host == "clob.polymarket.com" and url.path == "/book":
            assert url.params["token_id"] == YES_TOKEN
            return httpx.Response(200, text=fixture_text("polymarket", "clob_book.json"))
        if url.path == "/prices-history":
            return httpx.Response(200, text=fixture_text("polymarket", "prices_history.json"))
        if url.host == "data-api.polymarket.com" and url.path == "/v2/trades":
            return httpx.Response(200, text=fixture_text("polymarket", "data_trades_v2.json"))
        if url.host == "gamma-api.polymarket.com" and url.path == "/markets":
            return httpx.Response(
                200, text="[" + fixture_text("polymarket", "gamma_market_yes_no.json") + "]"
            )
        if url.host == "gamma-api.polymarket.com" and url.path == "/events":
            return httpx.Response(200, text=fixture_text("polymarket", "gamma_events_page.json"))
        return httpx.Response(404)

    clock = ManualClock(T0_NS)

    async def go() -> None:
        client = PolymarketRestClient(
            HttpFetcher(mock_client(httpx.MockTransport(handler)), clock=clock)
        )
        book = await client.get_book(YES_TOKEN)
        assert book.instrument_id == YES_INST
        assert book.bids[0].price == D("0.5")
        assert book.asks[0].price == D("0.52")  # descending asks re-sorted
        history = await client.get_prices_history(YES_TOKEN, start_ts=1_791_318_000)
        assert [p.price for p in history] == [D("0.505"), D("0.51"), D("0.5125")]
        trades = await client.get_trades(condition_id=CONDITION)
        assert [t.instrument_id for t in trades] == [YES_INST, NO_INST]
        assert all(t.aggressor_side is None for t in trades)
        assert trades[0].price == D("0.52")
        assert trades[0].size == D("12.5")
        markets = await client.list_markets(closed=False)
        assert [m.native_id for m in markets] == [CONDITION]
        events = await client.list_events(slug="bitcoin-above-on-october-31")
        assert events[0].markets[0].outcome_instruments["YES"] == YES_INST

    asyncio.run(go())


def test_rest_book_stream_is_parsed_by_adapter() -> None:
    (snap,) = poly("clob_book.json", stream="rest:book")
    assert isinstance(snap, BookSnapshotEvent)
    assert snap.instrument_id == YES_INST
    assert QualityFlag.SOURCE_TS_MISSING not in snap.quality_flags
    assert poly("gamma_market_yes_no.json", stream="rest:gamma-markets") == []
