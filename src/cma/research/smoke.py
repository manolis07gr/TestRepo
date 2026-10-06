"""Paper-mode smoke test (scope s.21): collectors on scripted mock feeds -> health status.

No network. Builds two scripted WebSocket feeds (Coinbase ticker and a Kalshi order book),
runs them through the real FeedSession/Collector/BookManager stack, forwards every
normalized event into a PaperTradingSession and returns both health reports.
"""

from __future__ import annotations

import asyncio
import json
import random
from decimal import Decimal
from typing import Any

from cma.adapters.crypto.coinbase import CoinbaseAdapter
from cma.adapters.kalshi.parser import KalshiAdapter
from cma.backtest.core import StaticMappings
from cma.config import AppConfig
from cma.domain.enums import (
    ContractStatus,
    InstrumentKind,
    MappingStatus,
    Operator,
    Venue,
)
from cma.domain.models import ContractMapping, MarketEvent, PredictionContract, instrument_key
from cma.domain.time import NS_PER_MS, NS_PER_S, ManualClock, iso_from_ns
from cma.execution.latency import LatencyModel, LatencyProfile
from cma.execution.paper.executor import PaperTradingSession
from cma.ingestion.book import BookManager
from cma.ingestion.collector import Collector
from cma.ingestion.ws import FeedSession, ScriptedTransport
from cma.signals.strategies.fair_value_taker import FairValueTakerConfig, FairValueTakerStrategy

T0 = 1_790_000_000 * NS_PER_S
TICKER = "KXBTCD-SMOKE-T110000"


def _coinbase_ticker(i: int, price: float) -> str:
    return json.dumps(
        {
            "type": "ticker",
            "sequence": 1000 + i,
            "product_id": "BTC-USD",
            "price": f"{price:.2f}",
            "best_bid": f"{price - 0.5:.2f}",
            "best_bid_size": "0.5",
            "best_ask": f"{price + 0.5:.2f}",
            "best_ask_size": "0.5",
            "side": "buy",
            "time": iso_from_ns(T0 + i * 200 * NS_PER_MS),
            "trade_id": 5000 + i,
            "last_size": "0.01",
        }
    )


def _kalshi_snapshot() -> str:
    return json.dumps(
        {
            "type": "orderbook_snapshot",
            "sid": 1,
            "seq": 1,
            "msg": {
                "market_ticker": TICKER,
                "yes_dollars_fp": [["0.4500", "100.00"], ["0.4400", "200.00"]],
                "no_dollars_fp": [["0.5300", "100.00"], ["0.5200", "150.00"]],
            },
        }
    )


def _kalshi_delta(seq: int, qty: str) -> str:
    return json.dumps(
        {
            "type": "orderbook_delta",
            "sid": 1,
            "seq": seq,
            "msg": {
                "market_ticker": TICKER,
                "price_dollars": "0.4500",
                "delta_fp": qty,
                "side": "yes",
                "ts_ms": (T0 + seq * 100 * NS_PER_MS) // NS_PER_MS,
            },
        }
    )


def run_smoke() -> dict[str, Any]:
    clock = ManualClock(T0)
    books = BookManager()
    contract_id = instrument_key(Venue.KALSHI, TICKER)
    contract = PredictionContract(
        venue=Venue.KALSHI,
        contract_id=contract_id,
        native_id=TICKER,
        event_id="KXBTCD-SMOKE",
        title="smoke: BTC above 110000",
        yes_semantics="above",
        no_semantics="not above",
        open_ts_ns=T0 - 3600 * NS_PER_S,
        close_ts_ns=T0 + 3600 * NS_PER_S,
        resolve_ts_ns=T0 + 3600 * NS_PER_S,
        status=ContractStatus.OPEN,
        tick_size=Decimal("0.01"),
        fee_schedule_id="kalshi-standard",
        outcome_instruments={"YES": contract_id},
    )
    mapping = ContractMapping(
        venue=Venue.KALSHI,
        contract_id=contract_id,
        underlyings=("BTC-USD",),
        operator=Operator.GT,
        strikes=(Decimal(110000),),
        observation_start_ns=T0 + 3540 * NS_PER_S,
        observation_end_ns=T0 + 3600 * NS_PER_S,
        observation_method="AVG_60S_BEFORE",
        timezone="America/New_York",
        resolution_source="CF_BENCHMARKS_BRTI",
        event_family="smoke",
        review_status=MappingStatus.APPROVED_PAPER,
        reviewer="smoke-test",
    )
    ref = instrument_key(Venue.COINBASE, "BTC-USD")
    cfg = AppConfig.model_validate({"mode": "PAPER"})
    strategy = FairValueTakerStrategy(
        FairValueTakerConfig(reference_instrument=ref, fixed_vol=0.45), [(contract, mapping)]
    )
    session = PaperTradingSession(
        config=cfg,
        strategies=[strategy],
        contracts={contract_id: contract},
        mappings=StaticMappings({contract_id: mapping}),
        latency=LatencyModel(LatencyProfile(outbound_ms=250)),
        clock=clock,
        reference_instruments=frozenset({ref}),
        feed_allowance_ms=0,
    )
    received: list[MarketEvent] = []

    def sink(events: list[MarketEvent]) -> None:
        received.extend(events)
        session.on_events(events)

    feeds = []
    for name, url, adapter, script in (
        (
            "coinbase-ws",
            "wss://ws-feed.exchange.coinbase.com",
            CoinbaseAdapter(),
            [_coinbase_ticker(i, 110_000 + i) for i in range(10)],
        ),
        (
            "kalshi-ws",
            "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
            KalshiAdapter(),
            [_kalshi_snapshot(), _kalshi_delta(2, "5.00"), _kalshi_delta(3, "-5.00")],
        ),
    ):
        transport = ScriptedTransport([script], on_exhausted="block", clock=clock, step_ns=1_000_000)
        feeds.append(
            FeedSession(
                name=name,
                url=url,
                adapter=adapter,
                transport=transport,
                clock=clock,
                books=books,
                on_events=sink,
                rng=random.Random(1),
                sleep=lambda s: asyncio.sleep(0),
                backoff=None,
            )
        )
    collector = Collector(clock=clock, books=books, feeds=feeds, stale_after_s=None)

    live_report: dict[str, Any] = {}

    async def main() -> None:
        tasks = [asyncio.create_task(f.run()) for f in feeds]
        for _ in range(200):  # wait until every scripted message has been consumed
            await asyncio.sleep(0.005)
            if len(received) >= 13:
                break
        live_report.update(collector.health_report())  # health while feeds are live
        for f in feeds:
            f.request_stop()
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(main())
    session.drain()
    report = live_report
    return {
        "collector": {"status": report["status"], "feeds": report["feeds"]},
        "events_received": len(received),
        "paper": session.health(),
        "instrument_kind": InstrumentKind.BINARY_CONTRACT.value,
    }
