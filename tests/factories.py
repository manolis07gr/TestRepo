"""Shared deterministic builders for tests (no network, no wall clock)."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from cma.backtest.core import (
    CancelIntent,
    OrderIntent,
    StaticMappings,
    StrategyContext,
    StrategyOutput,
    TradingCore,
)
from cma.backtest.engine import ReplayEngine
from cma.config import AppConfig
from cma.domain.enums import (
    BookSide,
    ContractStatus,
    DeltaMode,
    ExecutionMode,
    MappingStatus,
    Operator,
    Side,
    TimeInForce,
    Venue,
)
from cma.domain.models import (
    BookDeltaEvent,
    BookLevel,
    BookSnapshotEvent,
    ContractMapping,
    Fill,
    LevelChange,
    MarketEvent,
    PredictionContract,
    Settlement,
    TradeEvent,
)
from cma.domain.time import NS_PER_MS, NS_PER_S
from cma.execution.latency import LatencyModel, LatencyProfile
from cma.features.fair_value import prob_above
from cma.research.settlement_data import Candle, SettledMarket
from cma.storage.events_io import read_events_jsonl

FIXTURES = Path(__file__).resolve().parent / "fixtures"
T0 = 1_790_000_000 * NS_PER_S


def D(x: str | int) -> Decimal:
    return Decimal(str(x))


def fixture_events(name: str) -> list[MarketEvent]:
    return list(read_events_jsonl(FIXTURES / name))


def config(**overrides: Any) -> AppConfig:
    base: dict[str, Any] = {"mode": "BACKTEST"}
    for key, value in overrides.items():
        base[key] = value
    return AppConfig.model_validate(base)


def contract(
    cid: str = "KALSHI:FIXTURE-BOOK",
    *,
    fee: str = "kalshi-standard",
    tick: str = "0.01",
    venue: Venue = Venue.KALSHI,
    resolve_ts_ns: int | None = None,
    yes_instrument: str | None = None,
) -> PredictionContract:
    native = cid.split(":", 1)[1]
    return PredictionContract(
        venue=venue,
        contract_id=cid,
        native_id=native,
        event_id="EVT",
        title=f"test contract {native}",
        yes_semantics="yes",
        no_semantics="no",
        open_ts_ns=0,
        close_ts_ns=resolve_ts_ns,
        resolve_ts_ns=resolve_ts_ns,
        status=ContractStatus.OPEN,
        tick_size=D(tick),
        fee_schedule_id=fee,
        outcome_instruments={"YES": yes_instrument or cid},
    )


def mapping(
    cid: str = "KALSHI:FIXTURE-BOOK",
    *,
    status: MappingStatus = MappingStatus.REVIEWED,
    strike: str = "100000",
    family: str = "FAM",
    end_ns: int = T0 + 3600 * NS_PER_S,
    method: str = "POINT",
    venue: Venue = Venue.KALSHI,
    underlying: str = "BTC-USD",
) -> ContractMapping:
    return ContractMapping(
        venue=venue,
        contract_id=cid,
        underlyings=(underlying,),
        operator=Operator.GT,
        strikes=(D(strike),),
        observation_start_ns=None,
        observation_end_ns=end_ns,
        observation_method=method,
        timezone="America/New_York",
        resolution_source="TEST",
        event_family=family,
        review_status=status,
    )


def snapshot(
    inst: str,
    seq: int | None,
    t: int,
    bids: Sequence[tuple[str, str]],
    asks: Sequence[tuple[str, str]],
    *,
    venue: Venue = Venue.KALSHI,
    delay_ns: int = 0,
) -> BookSnapshotEvent:
    return BookSnapshotEvent(
        venue=venue,
        instrument_id=inst,
        source_ts_ns=t,
        recv_ts_ns=t + delay_ns,
        sequence=seq,
        payload_hash=f"s-{inst}-{seq}-{t}",
        bids=tuple(BookLevel(D(p), D(q)) for p, q in bids),
        asks=tuple(BookLevel(D(p), D(q)) for p, q in asks),
    )


def delta(
    inst: str,
    seq: int | None,
    t: int,
    changes: Sequence[tuple[BookSide, str, str]],
    *,
    mode: DeltaMode = DeltaMode.INCREMENT,
    venue: Venue = Venue.KALSHI,
    delay_ns: int = 0,
) -> BookDeltaEvent:
    return BookDeltaEvent(
        venue=venue,
        instrument_id=inst,
        source_ts_ns=t,
        recv_ts_ns=t + delay_ns,
        sequence=seq,
        payload_hash=f"d-{inst}-{seq}-{t}",
        changes=tuple(LevelChange(s, D(p), D(q)) for s, p, q in changes),
        mode=mode,
    )


def trade(
    inst: str,
    t: int,
    price: str,
    size: str,
    aggressor: Side | None,
    *,
    tid: str | None = None,
    venue: Venue = Venue.KALSHI,
    delay_ns: int = 0,
) -> TradeEvent:
    return TradeEvent(
        venue=venue,
        instrument_id=inst,
        source_ts_ns=t,
        recv_ts_ns=t + delay_ns,
        payload_hash=f"t-{inst}-{tid or t}",
        price=D(price),
        size=D(size),
        aggressor_side=aggressor,
        trade_id=tid or f"t{t}",
    )


@dataclass
class ScriptedStrategy:
    """Emits pre-scripted outputs at the first observation at/after each time."""

    script: list[tuple[int, Callable[[StrategyContext], Iterable[StrategyOutput]]]]
    strategy_id: str = "scripted"
    version: str = "1"
    fills: list[Fill] = field(default_factory=list)
    _i: int = 0

    def on_observation(self, event: MarketEvent, ctx: StrategyContext) -> list[StrategyOutput]:
        out: list[StrategyOutput] = []
        while self._i < len(self.script) and self.script[self._i][0] <= ctx.now_ns:
            out.extend(self.script[self._i][1](ctx))
            self._i += 1
        return out

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> list[StrategyOutput]:
        self.fills.append(fill)
        return []

    def on_timer(self, ctx: StrategyContext) -> list[StrategyOutput]:
        return []


def buy_ioc(
    cid: str, qty: str, limit: str | None, *, tif: TimeInForce = TimeInForce.IOC
) -> OrderIntent:
    return OrderIntent(
        venue=Venue.KALSHI,
        contract_id=cid,
        instrument_id=cid,
        side=Side.BUY,
        quantity=D(qty),
        limit_price=None if limit is None else D(limit),
        tif=tif,
        strategy_id="scripted",
    )


def order(
    cid: str,
    side: Side,
    qty: str,
    limit: str | None,
    *,
    tif: TimeInForce = TimeInForce.IOC,
    strategy_id: str = "scripted",
) -> OrderIntent:
    from cma.domain.enums import OrderType

    return OrderIntent(
        venue=Venue.KALSHI,
        contract_id=cid,
        instrument_id=cid,
        side=side,
        quantity=D(qty),
        limit_price=None if limit is None else D(limit),
        tif=tif,
        order_type=OrderType.MARKET if limit is None else OrderType.LIMIT,
        strategy_id=strategy_id,
    )


def cancel(order_id: str) -> CancelIntent:
    return CancelIntent(order_id)


def run_replay(
    events: Sequence[MarketEvent],
    strategies: Sequence[Any],
    *,
    contracts: Sequence[PredictionContract] = (),
    mappings: Sequence[ContractMapping] = (),
    outbound_ms: float = 100.0,
    compute_ms: float = 0.0,
    cancel_ms: float | None = None,
    ack_ms: float = 0.0,
    cfg: AppConfig | None = None,
    settlements: Sequence[Settlement] = (),
    reference_instruments: frozenset[str] = frozenset(),
    seed: int = 7,
    **core_kwargs: Any,
) -> TradingCore:
    cfg = cfg or config()
    contracts = list(contracts) or [contract()]
    maps = {
        m.contract_id: m for m in (list(mappings) or [mapping(c.contract_id) for c in contracts])
    }
    profile = LatencyProfile(
        outbound_ms=outbound_ms, compute_ms=compute_ms, ack_ms=ack_ms, cancel_ms=cancel_ms
    )

    def factory(engine: ReplayEngine) -> TradingCore:
        return TradingCore(
            config=cfg,
            strategies=strategies,
            contracts={c.contract_id: c for c in contracts},
            mappings=StaticMappings(maps),
            latency=LatencyModel(profile, seed=seed),
            scheduler=engine,
            mode=ExecutionMode.BACKTEST,
            reference_instruments=reference_instruments,
            **core_kwargs,
        )

    engine = ReplayEngine(factory, settlements=settlements, max_lateness_ns=NS_PER_S)
    engine.run(sorted(events, key=lambda e: e.recv_ts_ns))
    return engine.core


def ms(k: float) -> int:
    return T0 + round(k * NS_PER_MS)


# ----------------------------------------------------------------------------- settlement study

SETTLEMENT_T0 = 1_767_225_600  # 2026-01-01T00:00:00Z
MINUTES_PER_YEAR = 365.25 * 24 * 60


def settlement_market(i: int, open_ts: int, strike: float, settle: float) -> SettledMarket:
    tk = f"KXBTC15M-T{i:05d}"
    return SettledMarket(
        ticker=tk,
        event_ticker=tk,
        series="KXBTC15M",
        open_ts=open_ts,
        close_ts=open_ts + 900,
        strike_type="greater_or_equal",
        floor_strike=strike,
        cap_strike=None,
        result="yes" if settle >= strike else "no",
        expiration_value=settle,
        archived=False,
    )


def settlement_world(
    n_markets: int, *, true_vol: float, kalshi_vol: float, dvol_pct: float, seed: int = 7
) -> tuple[list[SettledMarket], dict[str, list[Candle]], list[list[float]], list[list[float]]]:
    """A BTC path with ``true_vol``; Kalshi quotes the log-normal digital with ``kalshi_vol``
    (1c spread); DVOL reads ``dvol_pct``. The candle starting at t0 + 60 j closes at
    price[j + 1], the price at the end of that minute."""
    t0 = SETTLEMENT_T0
    rng = random.Random(seed)
    minutes = n_markets * 15 + 240
    sig = true_vol / math.sqrt(MINUTES_PER_YEAR)
    price = [60_000.0]
    for _ in range(minutes):
        price.append(price[-1] * math.exp(rng.gauss(0.0, sig)))
    coinbase = [[t0 + 60 * j, *([price[j + 1]] * 4), 1.0] for j in range(minutes)]
    dvol = [[t0 + 3600 * h, dvol_pct] for h in range(minutes // 60 + 2)]  # hourly closes

    def at(t: int) -> float:
        return price[(t - t0) // 60]

    markets: list[SettledMarket] = []
    candles: dict[str, list[Candle]] = {}
    first_open = t0 + 240 * 60
    for i in range(n_markets):
        o = first_open + 900 * i
        m = settlement_market(i, o, at(o), at(o + 900))
        markets.append(m)
        rows: list[Candle] = []
        for k in range(1, 16):
            e = o + 60 * k
            if k < 15:
                f = prob_above(
                    spot=at(e),
                    strike=m.floor_strike or 0.0,
                    now_ns=e * NS_PER_S,
                    observation_end_ns=m.close_ts * NS_PER_S,
                    sigma=kalshi_vol,
                    observation_method="AVG_60S_BEFORE",
                )
            else:
                f = 1.0 if m.result == "yes" else 0.0
            mid = min(max(f, 0.02), 0.98)
            bid, ask = round(mid - 0.005, 3), round(mid + 0.005, 3)
            rows.append((e, bid, bid, bid, bid, ask, ask, ask, ask, 100.0))
        candles[m.ticker] = rows
    return markets, candles, coinbase, dvol
