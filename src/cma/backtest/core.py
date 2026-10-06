"""Trading core shared by historical replay and forward paper trading (T033 parity).

The core is driven by three kinds of calls, all carrying an explicit timestamp:

* ``on_venue_event(event, t)`` - the event happens *at the venue* (venue time); only the
  execution simulator sees it.
* ``on_observation(event, t)``  - the event reaches *us* (receive time + simulated feed
  delay); market state, strategies, signals and risk see it.
* ``on_action(kind, payload, t)`` - scheduled consequences: order arrival at the venue,
  cancel arrival, acknowledgements, timers, mark-outs, contract close and settlement.

Who decides *when* those calls happen is the driver: ``ReplayEngine`` (historical,
discrete-event priority queue) or the paper executor (wall clock). Both use this class.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from typing import Protocol

from cma.config import AppConfig
from cma.domain.enums import (
    ExecutionMode,
    InstrumentKind,
    MarkMethod,
    OrderType,
    ReasonCode,
    Side,
    TimeInForce,
    Venue,
)
from cma.domain.fees import FeeSchedule, ScaledFeeSchedule, get_fee_schedule
from cma.domain.models import (
    ContractMapping,
    Fill,
    MarketEvent,
    PredictionContract,
    Settlement,
    Signal,
    SimOrder,
)
from cma.domain.numbers import ONE, ZERO
from cma.domain.time import NS_PER_MS
from cma.execution.latency import LatencyModel
from cma.execution.simulator.core import ExecutionSimulator
from cma.features.state import MarketState
from cma.portfolio.ledger import Portfolio, mark_from_book
from cma.risk.engine import OrderRequest, PositionView, RiskEngine
from cma.signals.engine import FairValueEstimate, GateInputs, SignalEngine, Suppression


class ActionKind(IntEnum):
    """Scheduling priority at equal timestamps (lower runs first).

    Venue events precede our order arrivals (liquidity that disappears "at the same time"
    is gone before we arrive - conservative), and both precede observations.
    """

    VENUE = 0
    CLOSE = 1
    ARRIVAL = 2
    CANCEL_ARRIVAL = 3
    GTD_EXPIRY = 4
    OBSERVATION = 5
    FILL_ACK = 6
    MARKOUT = 7
    TIMER = 8
    SETTLE = 9


class Scheduler(Protocol):
    def schedule(self, ts_ns: int, kind: ActionKind, payload: object) -> None: ...


class MappingProvider(Protocol):
    def get(self, contract_id: str) -> ContractMapping | None: ...

    def can_trade(self, contract_id: str, mode: ExecutionMode) -> bool: ...


@dataclass
class StaticMappings:
    """Simple MappingProvider over a dict (tests, synthetic studies)."""

    mappings: dict[str, ContractMapping]

    def get(self, contract_id: str) -> ContractMapping | None:
        return self.mappings.get(contract_id)

    def can_trade(self, contract_id: str, mode: ExecutionMode) -> bool:
        from cma.domain.enums import MappingStatus

        m = self.mappings.get(contract_id)
        if m is None or mode is ExecutionMode.LIVE:
            return False
        need = (
            MappingStatus.REVIEWED
            if mode is ExecutionMode.BACKTEST
            else (MappingStatus.APPROVED_PAPER)
        )
        return m.review_status.at_least(need)


@dataclass(frozen=True, slots=True)
class OrderIntent:
    venue: Venue
    contract_id: str
    instrument_id: str
    side: Side
    quantity: Decimal
    limit_price: Decimal | None
    tif: TimeInForce = TimeInForce.IOC
    order_type: OrderType = OrderType.LIMIT
    strategy_id: str = ""
    signal: Signal | None = None
    group_id: str | None = None
    gtd_expiry_ns: int | None = None


@dataclass(frozen=True, slots=True)
class CancelIntent:
    order_id: str


type StrategyOutput = FairValueEstimate | OrderIntent | CancelIntent


class Strategy(Protocol):
    strategy_id: str
    version: str

    def on_observation(
        self, event: MarketEvent, ctx: StrategyContext
    ) -> Iterable[StrategyOutput]: ...

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> Iterable[StrategyOutput]: ...

    def on_timer(self, ctx: StrategyContext) -> Iterable[StrategyOutput]: ...


@dataclass
class StrategyContext:
    now_ns: int
    core: TradingCore

    @property
    def state(self) -> MarketState:
        return self.core.state

    def mapping(self, contract_id: str) -> ContractMapping | None:
        return self.core.mappings.get(contract_id)

    def contract(self, contract_id: str) -> PredictionContract | None:
        return self.core.contracts.get(contract_id)

    def known_position(self, contract_id: str) -> Decimal:
        pv = self.core.known_positions.get(contract_id)
        return pv.quantity if pv else ZERO

    def working_orders(self, contract_id: str | None = None) -> list[SimOrder]:
        """Non-terminal orders, including those still travelling to the venue."""
        return [
            o
            for o in self.core.sim.working_orders()
            if contract_id is None or o.contract_id == contract_id
        ]


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts_ns: int
    nav: Decimal
    cash: Decimal
    open_risk: Decimal
    realized: Decimal
    fees: Decimal


@dataclass(frozen=True, slots=True)
class OrderRecord:
    order_id: str
    strategy_id: str
    contract_id: str
    side: Side
    quantity: Decimal
    limit_price: Decimal | None
    tif: TimeInForce
    decision_ts_ns: int
    submit_ts_ns: int
    arrival_ts_ns: int
    signal_id: str | None
    signal_executable_price: Decimal | None
    signal_fair: Decimal | None
    signal_net_edge: Decimal | None


@dataclass
class CoreRecords:
    fills: list[Fill] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    orders: list[OrderRecord] = field(default_factory=list)
    risk_rejections: Counter[ReasonCode] = field(default_factory=Counter)
    equity: list[EquityPoint] = field(default_factory=list)
    markouts: dict[tuple[str, int], Decimal] = field(default_factory=dict)  # (fill, ms) -> mid
    settlement_pnl: dict[str, Decimal] = field(default_factory=dict)
    late_venue_events: int = 0
    observations: int = 0
    venue_events: int = 0


class TradingCore:
    def __init__(
        self,
        *,
        config: AppConfig,
        strategies: Sequence[Strategy],
        contracts: Mapping[str, PredictionContract],
        mappings: MappingProvider,
        latency: LatencyModel,
        scheduler: Scheduler,
        mode: ExecutionMode = ExecutionMode.BACKTEST,
        reference_instruments: frozenset[str] = frozenset(),
        fee_multiplier: Decimal = ONE,
        stress_slippage_ticks: int = 0,
        venue_taker_delay_ms: Mapping[Venue, int] | None = None,
        sequence_free_venues: frozenset[Venue] = frozenset({Venue.POLYMARKET}),
        mark_method: MarkMethod = MarkMethod.MID,
        markout_horizons_ms: Sequence[int] = (1_000, 5_000, 30_000, 60_000),
        strict_lookahead: bool = True,
        family_of: Callable[[str], str] | None = None,
    ) -> None:
        if mode is ExecutionMode.LIVE:
            from cma.domain.errors import LiveTradingDisabledError

            raise LiveTradingDisabledError("the trading core never routes live orders in v1")
        self.config = config
        self.strategies = list(strategies)
        self.contracts = dict(contracts)
        self.mappings = mappings
        self.latency = latency
        self.scheduler = scheduler
        self.mode = mode
        self.mark_method = mark_method
        self.markout_horizons_ms = tuple(markout_horizons_ms)
        self.venue_taker_delay_ms = dict(venue_taker_delay_ms or {})
        self._fee_multiplier = fee_multiplier
        self._fee_cache: dict[str, FeeSchedule] = {}
        self._family_of = family_of or self._default_family
        self.state = MarketState(
            reference_instruments=reference_instruments,
            sequence_free_venues=sequence_free_venues,
        )
        self.sim = ExecutionSimulator(
            fee_resolver=self.fee_schedule,
            tick_resolver=self._tick,
            queue_model=config.simulation.queue_model,
            allow_partial_fills=config.simulation.allow_partial_fills,
            fill_on_cross=config.simulation.fill_on_cross,
            stress_slippage_ticks=stress_slippage_ticks,
            sequence_free_venues=sequence_free_venues,
        )
        self.portfolio = Portfolio(initial_cash=config.risk.initial_nav)
        self.risk = RiskEngine(config.risk)
        self.risk.register_cancel_all(self.cancel_all)
        self.signal_engine = SignalEngine(
            config=config.signal,
            mode=mode,
            can_trade=mappings.can_trade,
            fee_resolver=self.fee_schedule,
            strict_lookahead=strict_lookahead,
        )
        self.known_positions: dict[str, PositionView] = {}
        self.records = CoreRecords()
        self._order_meta: dict[str, OrderRecord] = {}
        self._last_mid: dict[str, Decimal] = {}
        self._closed: set[str] = set()
        self._nav_cache: Decimal = config.risk.initial_nav
        self.instrument_to_contract: dict[str, str] = {}
        for c in self.contracts.values():
            yes = c.outcome_instruments.get("YES", c.contract_id)
            self.instrument_to_contract[yes] = c.contract_id
            self.instrument_to_contract[c.contract_id] = c.contract_id

    # ------------------------------------------------------------------ resolvers

    def fee_schedule(self, contract_id: str) -> FeeSchedule:
        fs = self._fee_cache.get(contract_id)
        if fs is None:
            contract = self.contracts.get(contract_id)
            base = get_fee_schedule(contract.fee_schedule_id if contract else "zero")
            fs = (
                base
                if self._fee_multiplier == ONE
                else ScaledFeeSchedule(base=base, factor=self._fee_multiplier)
            )
            self._fee_cache[contract_id] = fs
        return fs

    def _tick(self, contract_id: str) -> Decimal:
        c = self.contracts.get(contract_id)
        return c.tick_size if c else Decimal("0.01")

    def _default_family(self, contract_id: str) -> str:
        m = self.mappings.get(contract_id)
        return m.event_family if m else ""

    # ------------------------------------------------------------------ driver calls

    def on_venue_event(self, event: MarketEvent, now_ns: int) -> None:
        self.records.venue_events += 1
        fills = self.sim.on_venue_event(event, now_ns)
        self._book_fills(fills, now_ns)

    def on_observation(self, event: MarketEvent, now_ns: int) -> None:
        self.records.observations += 1
        self.state.on_event(event)
        cid = self.instrument_to_contract.get(event.instrument_id)
        if cid is not None:
            b = self.state.books.get(event.instrument_id)
            if b is not None and b.is_valid:
                bb, ba = b.best_bid(), b.best_ask()
                if bb is not None and ba is not None:
                    self._last_mid[cid] = (bb[0] + ba[0]) / 2
        ctx = StrategyContext(now_ns=now_ns, core=self)
        for strat in self.strategies:
            for out in strat.on_observation(event, ctx):
                self.handle_output(strat, out, now_ns)

    def on_action(self, kind: ActionKind, payload: object, now_ns: int) -> None:
        if kind is ActionKind.ARRIVAL:
            assert isinstance(payload, str)
            self._book_fills(self.sim.process_arrival(payload, now_ns), now_ns)
        elif kind is ActionKind.CANCEL_ARRIVAL:
            assert isinstance(payload, str)
            self.sim.process_cancel_arrival(payload, now_ns)
        elif kind is ActionKind.GTD_EXPIRY:
            assert isinstance(payload, str)
            self.sim.expire_order(payload, now_ns)
        elif kind is ActionKind.FILL_ACK:
            assert isinstance(payload, Fill)
            self._ack_fill(payload, now_ns)
        elif kind is ActionKind.MARKOUT:
            assert isinstance(payload, tuple)
            fill_id, horizon_ms, contract_id = payload
            # a horizon that ends after the contract closed has no market mid: the last
            # pre-close quote is ~the settlement outcome, i.e. luck, not a mark-out
            mid = None if contract_id in self._closed else self._last_mid.get(contract_id)
            if mid is not None:
                self.records.markouts[(fill_id, horizon_ms)] = mid
        elif kind is ActionKind.TIMER:
            self.mark_to_market(now_ns)
            ctx = StrategyContext(now_ns=now_ns, core=self)
            for strat in self.strategies:
                for out in strat.on_timer(ctx):
                    self.handle_output(strat, out, now_ns)
        elif kind is ActionKind.CLOSE:
            assert isinstance(payload, str)
            self.close_contract(payload, now_ns)
        elif kind is ActionKind.SETTLE:
            assert isinstance(payload, Settlement)
            self.settle(payload, now_ns)
        else:  # pragma: no cover - VENUE/OBSERVATION are dispatched directly
            raise ValueError(f"unexpected action {kind}")

    # ------------------------------------------------------------------ strategy outputs

    def handle_output(self, strat: Strategy, out: StrategyOutput, now_ns: int) -> None:
        if isinstance(out, FairValueEstimate):
            self._handle_estimate(out, now_ns)
        elif isinstance(out, OrderIntent):
            self.submit_intent(out, now_ns)
        elif isinstance(out, CancelIntent):
            self.cancel(out.order_id, now_ns)

    def _gates(self, est: FairValueEstimate, now_ns: int) -> GateInputs:
        dq = self.config.data_quality
        if est.contract_id in self._closed:
            return GateInputs(data_fresh=False, freshness_detail="contract closed")
        ref_age = now_ns - est.feature_watermark_ns
        if ref_age > dq.max_reference_age_ms * NS_PER_MS:
            return GateInputs(data_fresh=False, freshness_detail=f"reference age {ref_age}ns")
        venue_age = self.state.venue_age_ns(est.venue, now_ns)
        if venue_age is None or venue_age > dq.max_prediction_feed_age_ms * NS_PER_MS:
            return GateInputs(data_fresh=False, freshness_detail=f"venue feed age {venue_age}")
        drift_ts = self.state.clock_drift_flagged.get(est.instrument_id)
        clock_ok = drift_ts is None or now_ns - drift_ts > 60_000 * NS_PER_MS
        return GateInputs(data_fresh=True, clock_ok=clock_ok)

    def _handle_estimate(self, est: FairValueEstimate, now_ns: int) -> Signal | Suppression:
        gates = self._gates(est, now_ns)
        builder = self.state.books.get(est.instrument_id)
        if gates.data_fresh and gates.clock_ok and builder is not None and builder.is_valid:
            pre = self.signal_engine.touch_precheck(
                est, builder.best_bid(), builder.best_ask(), now_ns
            )
            if pre is not None:
                return pre
        book = None
        if builder is not None:
            book = builder.snapshot(self.config.signal.max_levels_to_walk)
        res = self.signal_engine.evaluate(est, book, decision_ts_ns=now_ns, gates=gates)
        if isinstance(res, Signal):
            self.records.signals.append(res)
            self.order_from_signal(res, now_ns)
        return res

    def order_from_signal(self, signal: Signal, now_ns: int) -> SimOrder | None:
        if signal.is_expired(now_ns):
            self.records.risk_rejections[ReasonCode.SIGNAL_EXPIRED] += 1
            return None
        intent = OrderIntent(
            venue=signal.venue,
            contract_id=signal.contract_id,
            instrument_id=signal.instrument_id,
            side=signal.side,
            quantity=signal.quantity,
            limit_price=signal.limit_price,
            tif=TimeInForce.IOC,
            strategy_id=signal.strategy_id,
            signal=signal,
            group_id=signal.group_id,
        )
        return self.submit_intent(intent, now_ns)

    def submit_intent(self, intent: OrderIntent, now_ns: int) -> SimOrder | None:
        if intent.contract_id in self._closed:
            self.records.risk_rejections[ReasonCode.CONTRACT_NOT_OPEN] += 1
            return None
        price = intent.limit_price
        if price is None:
            price = ONE if intent.side is Side.BUY else ZERO
        request = OrderRequest(
            contract_id=intent.contract_id,
            event_family=self._family_of(intent.contract_id),
            side=intent.side,
            quantity=intent.quantity,
            price=price,
            ts_ns=now_ns,
        )
        working = self.sim.working_orders()
        decision = self.risk.check_order(
            request,
            nav=self.nav_estimate(),
            positions=self.known_positions,
            working_orders=working,
            buying_power=self.portfolio.buying_power() - self._pending_capital(working),
            family_of=self._family_of,
        )
        if not decision.approved:
            for r in decision.reasons:
                self.records.risk_rejections[r] += 1
            return None
        submit_ts, arrival_ts = self.latency.schedule_order(now_ns)
        if intent.tif in (TimeInForce.IOC, TimeInForce.FOK) or intent.order_type is (
            OrderType.MARKET
        ):
            arrival_ts += self.venue_taker_delay_ms.get(intent.venue, 0) * NS_PER_MS
        order = SimOrder(
            order_id=self.sim.next_order_id(),
            venue=intent.venue,
            contract_id=intent.contract_id,
            instrument_id=intent.instrument_id,
            side=intent.side,
            order_type=intent.order_type,
            quantity=intent.quantity,
            tif=intent.tif,
            submit_ts_ns=submit_ts,
            arrival_ts_ns=arrival_ts,
            limit_price=intent.limit_price,
            signal_id=intent.signal.signal_id if intent.signal else None,
            strategy_id=intent.strategy_id,
            expiry_ts_ns=intent.signal.expiry_ts_ns if intent.signal else None,
            gtd_expiry_ts_ns=intent.gtd_expiry_ns,
            group_id=intent.group_id,
        )
        self.sim.submit(order)
        sig = intent.signal
        rec = OrderRecord(
            order_id=order.order_id,
            strategy_id=intent.strategy_id,
            contract_id=intent.contract_id,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            tif=intent.tif,
            decision_ts_ns=now_ns,
            submit_ts_ns=submit_ts,
            arrival_ts_ns=arrival_ts,
            signal_id=sig.signal_id if sig else None,
            signal_executable_price=sig.executable_price if sig else None,
            signal_fair=sig.fair_probability if sig else None,
            signal_net_edge=sig.net_edge if sig else None,
        )
        self.records.orders.append(rec)
        self._order_meta[order.order_id] = rec
        self.scheduler.schedule(arrival_ts, ActionKind.ARRIVAL, order.order_id)
        if intent.tif is TimeInForce.GTD and intent.gtd_expiry_ns is not None:
            self.scheduler.schedule(intent.gtd_expiry_ns, ActionKind.GTD_EXPIRY, order.order_id)
        return order

    @staticmethod
    def _pending_capital(working: Iterable[SimOrder]) -> Decimal:
        total = ZERO
        for o in working:
            px = (
                o.limit_price
                if o.limit_price is not None
                else (ONE if o.side is Side.BUY else ZERO)
            )
            total += o.remaining * (px if o.side is Side.BUY else ONE - px)
        return total

    def cancel(self, order_id: str, now_ns: int) -> None:
        order = self.sim.orders.get(order_id)
        if order is None:
            return
        arrival = self.sim.request_cancel(order_id, now_ns, self.latency.cancel_ns())
        if arrival is not None:
            self.scheduler.schedule(
                max(arrival, order.arrival_ts_ns), ActionKind.CANCEL_ARRIVAL, order_id
            )

    def cancel_all(self, now_ns: int) -> None:
        for order in self.sim.working_orders():
            self.cancel(order.order_id, now_ns)

    # ------------------------------------------------------------------ fills & accounting

    def _book_fills(self, fills: Sequence[Fill], now_ns: int) -> None:
        if fills:
            self._nav_cache = ZERO  # refreshed below once all fills are booked
        for f in fills:
            self.portfolio.apply_fill(f, event_family=self._family_of(f.contract_id))
            self.records.fills.append(f)
            self.scheduler.schedule(now_ns + self.latency.ack_ns(), ActionKind.FILL_ACK, f)
            for h in self.markout_horizons_ms:
                self.scheduler.schedule(
                    now_ns + h * NS_PER_MS, ActionKind.MARKOUT, (f.fill_id, h, f.contract_id)
                )
        if fills:
            self.current_nav()

    def _ack_fill(self, fill: Fill, now_ns: int) -> None:
        pos = self.portfolio.positions.get(fill.contract_id)
        if pos is not None:
            self.known_positions[fill.contract_id] = PositionView(
                quantity=pos.quantity,
                cost_basis=pos.cost_basis,
                event_family=pos.event_family,
            )
        ctx = StrategyContext(now_ns=now_ns, core=self)
        for strat in self.strategies:
            if strat.strategy_id == fill.strategy_id:
                for out in strat.on_fill(fill, ctx):
                    self.handle_output(strat, out, now_ns)

    def marks(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for cid, pos in self.portfolio.positions.items():
            if pos.quantity == ZERO:
                continue
            contract = self.contracts.get(cid)
            inst = contract.outcome_instruments.get("YES", cid) if contract else cid
            book = self.state.book(inst)
            fallback = self._last_mid.get(cid, pos.avg_cost)
            out[cid] = mark_from_book(
                book if book is not None and book.is_valid else None,
                pos.quantity,
                self.mark_method,
                fallback,
            )
        return out

    def current_nav(self) -> Decimal:
        self._nav_cache = self.portfolio.nav(self.marks())
        return self._nav_cache

    def nav_estimate(self) -> Decimal:
        """NAV as of the last mark (fills, timers, settlements refresh it)."""
        return self._nav_cache

    def mark_to_market(self, now_ns: int) -> EquityPoint:
        nav = self.current_nav()
        self.risk.on_nav(now_ns, nav)
        point = EquityPoint(
            ts_ns=now_ns,
            nav=nav,
            cash=self.portfolio.cash,
            open_risk=self.portfolio.total_open_risk(),
            realized=self.portfolio.realized_total,
            fees=self.portfolio.fees_total,
        )
        self.records.equity.append(point)
        return point

    def close_contract(self, contract_id: str, now_ns: int) -> None:
        self._closed.add(contract_id)
        contract = self.contracts.get(contract_id)
        inst = contract.outcome_instruments.get("YES", contract_id) if contract else contract_id
        self.sim.close_instrument(inst, now_ns)

    def settle(self, settlement: Settlement, now_ns: int) -> None:
        if settlement.contract_id not in self._closed:
            self.close_contract(settlement.contract_id, now_ns)
        pnl = self.portfolio.settle(settlement)
        self.records.settlement_pnl[settlement.contract_id] = pnl
        self.known_positions.pop(settlement.contract_id, None)
        self.current_nav()


def is_reference_instrument(kind: InstrumentKind) -> bool:
    return kind is not InstrumentKind.BINARY_CONTRACT
