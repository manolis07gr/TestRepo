"""ContractStore: idempotent upserts, exact round trips, filters."""

from __future__ import annotations

import dataclasses

import pytest

from cma.adapters.base import load_json
from cma.adapters.kalshi import parse_market_response, parse_series
from cma.adapters.polymarket import parse_gamma_market
from cma.domain.enums import ContractStatus, Venue
from cma.domain.time import ManualClock
from cma.storage.contracts import ContractStore, contract_from_json, contract_to_json
from cma.storage.db import open_database
from tests.unit.adapters.helpers import T0_NS, fixture_text

pytestmark = pytest.mark.unit


def contracts() -> tuple:  # type: ignore[type-arg]
    series = parse_series(load_json(fixture_text("kalshi", "series.json")))
    kalshi = parse_market_response(load_json(fixture_text("kalshi", "market.json")), series=series)
    poly = parse_gamma_market(load_json(fixture_text("polymarket", "gamma_market_updown.json")))
    return kalshi, poly


def test_json_round_trip_is_exact() -> None:
    for contract in contracts():
        assert contract_from_json(contract_to_json(contract)) == contract


def test_upsert_is_idempotent_and_preserves_first_seen() -> None:
    db = open_database("sqlite:///:memory:")
    clock = ManualClock(T0_NS)
    store = ContractStore(db, clock=clock)
    kalshi, poly = contracts()
    assert store.upsert_many([kalshi, poly]) == 2
    clock.advance(1_000)
    assert not store.upsert(kalshi)  # identical content: no write
    closed = dataclasses.replace(kalshi, status=ContractStatus.CLOSED)
    assert store.upsert(closed)
    assert store.get(Venue.KALSHI, kalshi.contract_id) == closed
    assert store.first_seen_ns(Venue.KALSHI, kalshi.contract_id) == T0_NS
    row = db.query("SELECT updated_ns FROM contracts WHERE venue = 'KALSHI'")[0]
    assert row["updated_ns"] == T0_NS + 1_000
    assert store.count() == 2
    assert store.load(venue=Venue.POLYMARKET) == [poly]
    assert store.load(status=ContractStatus.CLOSED) == [closed]
    assert store.get(Venue.KALSHI, "KALSHI:missing") is None
