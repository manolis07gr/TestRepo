"""Latency-aware execution simulator (scope s.12).

The simulator owns the *venue-time* view of each book: events are applied at their venue
timestamp, our orders interact with the book only once they have *arrived*, and liquidity
we consume is remembered (an overlay on the recorded book) so the same displayed size can
never be filled twice.

Fill rules
----------
* Takers (market/limit, IOC/FOK/GTC/GTD) walk available levels at arrival, never through
  their limit, applying optional stress slippage that also respects the limit.
* Resting makers fill only on (a) a venue trade at our price after the queue ahead of us
  is consumed (FIFO proxy, not in ``trade_through`` mode), (b) a trade *through* our price,
  or (c) opposite-side liquidity crossing our price (``fill_on_cross``). A touch alone never
  fills (T023). Cancellations reach the venue only after the cancel latency (T024).
* Orders arriving after their signal TTL expire unfilled (T017); orders arriving on an
  invalid venue book are rejected (fail closed).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal

from cma.domain.enums import (
    BookSide,
    ContractStatus,
    LiquidityRole,
    OrderState,
    OrderType,
    ReasonCode,
    Side,
    TimeInForce,
    Venue,
)
from cma.domain.fees import FeeSchedule
from cma.domain.models import (
    BookDeltaEvent,
    BookSnapshotEvent,
    Fill,
    MarketEvent,
    SimOrder,
    StatusEvent,
    TradeEvent,
)
from cma.domain.numbers import ONE, ZERO
from cma.ingestion.book import ApplyResult, L2BookBuilder

QUEUE_MODELS = ("conservative", "fifo_proxy", "trade_through")


@dataclass
class VenueBook:
    """Venue-time book for one instrument plus the overlay of liquidity we consumed."""

    builder: L2BookBuilder
    consumed: dict[tuple[BookSide, Decimal], Decimal] = field(default_factory=dict)
    recent_trade_qty: dict[tuple[BookSide, Decimal], Decimal] = field(default_factory=dict)
    status: ContractStatus = ContractStatus.OPEN

    def apply(self, event: MarketEvent) -> ApplyResult:
        res = self.builder.apply(event)
        if not res.applied:
            return res
        if isinstance(event, BookSnapshotEvent):
            self.consumed.clear()
            self.recent_trade_qty.clear()
        elif isinstance(event, BookDeltaEvent):
            for ch in event.changes:
                key = (ch.side, ch.price)
                if key in self.consumed:
                    recorded = self.builder.quantity_at(ch.side, ch.price)
                    if recorded <= ZERO:
                        del self.consumed[key]
                    elif recorded < self.consumed[key]:
                        self.consumed[key] = recorded
        return res

    def available(self, side: BookSide, price: Decimal) -> Decimal:
        recorded = self.builder.quantity_at(side, price)
        return max(ZERO, recorded - self.consumed.get((side, price), ZERO))

    def available_levels(self, side: BookSide) -> list[tuple[Decimal, Decimal]]:
        out = []
        for price, qty in self.builder.levels(side):
            avail = qty - self.consumed.get((side, price), ZERO)
            if avail > ZERO:
                out.append((price, avail))
        return out

    def consume(self, side: BookSide, price: Decimal, qty: Decimal) -> None:
        key = (side, price)
        self.consumed[key] = self.consumed.get(key, ZERO) + qty


@dataclass
class SimulatorStats:
    orders_submitted: int = 0
    orders_arrived: int = 0
    taker_fills: int = 0
    maker_fills: int = 0
    rejected: int = 0
    expired: int = 0
    canceled: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def count_reject(self, reason: str) -> None:
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1


class ExecutionSimulator:
    def __init__(
        self,
        *,
        fee_resolver: Callable[[str], FeeSchedule],
        tick_resolver: Callable[[str], Decimal] | None = None,
        queue_model: str = "conservative",
        allow_partial_fills: bool = True,
        fill_on_cross: bool = True,
        stress_slippage_ticks: int = 0,
        sequence_free_venues: frozenset[Venue] = frozenset(),
        id_prefix: str = "sim",
        binary_price_bounds: bool = True,
    ) -> None:
        if queue_model not in QUEUE_MODELS:
            raise ValueError(f"queue_model must be one of {QUEUE_MODELS}")
        self.fee_resolver = fee_resolver
        self.tick_resolver = tick_resolver or (lambda _cid: Decimal("0.01"))
        self.queue_model = queue_model
        self.allow_partial_fills = allow_partial_fills
        self.fill_on_cross = fill_on_cross
        self.stress_slippage_ticks = stress_slippage_ticks
        self.sequence_free_venues = sequence_free_venues
        self.binary_price_bounds = binary_price_bounds
        self.books: dict[str, VenueBook] = {}
        self.orders: dict[str, SimOrder] = {}
        self._resting: dict[str, list[str]] = {}  # instrument -> resting order ids
        self._live: dict[str, None] = {}  # non-terminal order ids (insertion ordered)
        self._id_prefix = id_prefix
        self._order_seq = 0
        self._fill_seq = 0
        self._fee_acc: dict[str, Decimal] = {}  # per-order accumulated fill-rounded fee
        self.stats = SimulatorStats()

    # ------------------------------------------------------------------ ids & books

    def next_order_id(self) -> str:
        self._order_seq += 1
        return f"{self._id_prefix}-o{self._order_seq:08d}"

    def _next_fill_id(self, order_id: str) -> str:
        self._fill_seq += 1
        return f"{order_id}-f{self._fill_seq:08d}"

    def venue_book(self, venue: Venue, instrument_id: str) -> VenueBook:
        vb = self.books.get(instrument_id)
        if vb is None:
            vb = VenueBook(
                builder=L2BookBuilder(
                    venue=venue,
                    instrument_id=instrument_id,
                    require_sequence=venue not in self.sequence_free_venues,
                )
            )
            self.books[instrument_id] = vb
        return vb

    # ------------------------------------------------------------------ venue events

    def on_venue_event(self, event: MarketEvent, now_ns: int) -> list[Fill]:
        """Apply a venue-time market event; return maker fills it triggers."""
        if isinstance(event, BookSnapshotEvent | BookDeltaEvent):
            vb = self.venue_book(event.venue, event.instrument_id)
            before = self._level_sizes_for_resting(event.instrument_id, vb)
            vb.apply(event)
            fills = self._update_queues_after_book_change(event.instrument_id, vb, before, now_ns)
            if self.fill_on_cross:
                fills.extend(self._cross_fills(event.instrument_id, vb, now_ns))
            return fills
        if isinstance(event, TradeEvent):
            vb = self.venue_book(event.venue, event.instrument_id)
            return self._trade_fills(event, vb, now_ns)
        if isinstance(event, StatusEvent):
            vb = self.venue_book(event.venue, event.instrument_id)
            vb.status = event.status
        return []

    # ------------------------------------------------------------------ order lifecycle

    def submit(self, order: SimOrder) -> None:
        if order.order_id in self.orders:
            raise ValueError(f"duplicate order id {order.order_id}")
        if order.state is not OrderState.PENDING_ARRIVAL:
            raise ValueError("orders must be submitted in PENDING_ARRIVAL state")
        self.orders[order.order_id] = order
        self._live[order.order_id] = None
        self.stats.orders_submitted += 1

    def process_arrival(self, order_id: str, now_ns: int) -> list[Fill]:
        order = self.orders[order_id]
        if order.state is not OrderState.PENDING_ARRIVAL:
            return []
        if now_ns < order.arrival_ts_ns:
            raise ValueError("order processed before its arrival time")
        self.stats.orders_arrived += 1
        if order.expiry_ts_ns is not None and now_ns > order.expiry_ts_ns:
            order.transition(OrderState.EXPIRED, now_ns, ReasonCode.SIGNAL_EXPIRED.value)
            self.stats.expired += 1
            self.stats.count_reject(ReasonCode.SIGNAL_EXPIRED.value)
            return []
        vb = self.venue_book(order.venue, order.instrument_id)
        if vb.status is not ContractStatus.OPEN:
            return self._reject(order, now_ns, ReasonCode.CONTRACT_NOT_OPEN)
        if not vb.builder.is_valid:
            return self._reject(order, now_ns, ReasonCode.BOOK_INVALID)

        limit = self._effective_limit(order)
        book_side = BookSide.ASK if order.side is Side.BUY else BookSide.BID
        rests_if_unfilled = order.order_type is OrderType.LIMIT and order.tif in (
            TimeInForce.GTC,
            TimeInForce.GTD,
        )
        take_liquidity = True
        if order.tif is TimeInForce.FOK or not self.allow_partial_fills:
            fillable = self._fillable_quantity(vb, book_side, order.side, limit)
            if fillable < order.remaining:
                if order.tif is TimeInForce.FOK or not rests_if_unfilled:
                    return self._reject(order, now_ns, ReasonCode.FOK_UNFILLABLE)
                take_liquidity = False  # all-or-none: rest the whole order instead
        fills = self._match_taker(order, vb, book_side, limit, now_ns) if take_liquidity else []

        if order.remaining == ZERO:
            return fills
        if rests_if_unfilled:
            # A cancel sent before arrival only takes effect when *it* reaches the venue,
            # so the order rests (and can fill) until then.
            if order.cancel_request_ts_ns is not None:
                order.state = OrderState.CANCEL_PENDING
            else:
                order.state = (
                    OrderState.PARTIALLY_FILLED if order.filled_quantity > ZERO else OrderState.OPEN
                )
            order.last_update_ts_ns = now_ns
            maker_side = BookSide.BID if order.side is Side.BUY else BookSide.ASK
            assert order.limit_price is not None
            order.queue_ahead = vb.builder.quantity_at(maker_side, order.limit_price)
            self._resting.setdefault(order.instrument_id, []).append(order.order_id)
            return fills
        if order.filled_quantity == ZERO:
            return self._reject(order, now_ns, ReasonCode.NO_LIQUIDITY)
        order.transition(OrderState.CANCELED, now_ns, "ioc_remainder")
        self.stats.canceled += 1
        return fills

    def request_cancel(self, order_id: str, now_ns: int, cancel_latency_ns: int) -> int | None:
        """Send a cancel; returns its venue arrival time (None if nothing to cancel)."""
        order = self.orders.get(order_id)
        if order is None or order.state.is_terminal:
            return None
        if order.cancel_request_ts_ns is not None:
            return order.cancel_arrival_ts_ns
        order.cancel_request_ts_ns = now_ns
        order.cancel_arrival_ts_ns = now_ns + cancel_latency_ns
        if order.state.is_working:
            order.state = OrderState.CANCEL_PENDING
        return order.cancel_arrival_ts_ns

    def process_cancel_arrival(self, order_id: str, now_ns: int) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.state.is_terminal:
            return False
        if order.state is OrderState.PENDING_ARRIVAL:
            return False  # arrival will be processed first; cancel re-applied by caller
        order.transition(OrderState.CANCELED, now_ns, "canceled")
        self.stats.canceled += 1
        self._remove_resting(order)
        return True

    def expire_order(self, order_id: str, now_ns: int) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.state.is_terminal or order.state is OrderState.PENDING_ARRIVAL:
            return False
        order.transition(OrderState.EXPIRED, now_ns, "gtd_expired")
        self.stats.expired += 1
        self._remove_resting(order)
        return True

    def close_instrument(self, instrument_id: str, now_ns: int) -> list[str]:
        """Contract closed for trading: cancel all working orders on it immediately."""
        vb = self.books.get(instrument_id)
        if vb is not None:
            vb.status = ContractStatus.CLOSED
        canceled = []
        for oid in list(self._resting.get(instrument_id, [])):
            order = self.orders[oid]
            if not order.state.is_terminal:
                order.transition(OrderState.CANCELED, now_ns, "contract_closed")
                self.stats.canceled += 1
                canceled.append(oid)
        self._resting.pop(instrument_id, None)
        return canceled

    def working_orders(self, instrument_id: str | None = None) -> list[SimOrder]:
        """Non-terminal orders (pending arrival, resting or cancel-pending)."""
        out = []
        dead = []
        for oid in self._live:
            o = self.orders[oid]
            if o.state.is_terminal:
                dead.append(oid)
            elif instrument_id is None or o.instrument_id == instrument_id:
                out.append(o)
        for oid in dead:
            del self._live[oid]
        return out

    def resting_orders(self, instrument_id: str) -> list[SimOrder]:
        return [self.orders[oid] for oid in self._resting.get(instrument_id, [])]

    # ------------------------------------------------------------------ matching

    def _effective_limit(self, order: SimOrder) -> Decimal | None:
        if order.order_type is OrderType.LIMIT:
            return order.limit_price
        if self.binary_price_bounds:
            return ONE if order.side is Side.BUY else ZERO
        return None

    def _slipped(self, price: Decimal, side: Side, tick: Decimal) -> Decimal:
        if self.stress_slippage_ticks == 0:
            return price
        slip = tick * self.stress_slippage_ticks
        p = price + slip if side is Side.BUY else price - slip
        if self.binary_price_bounds:
            p = min(ONE, max(ZERO, p))
        return p

    def _fillable_quantity(
        self, vb: VenueBook, book_side: BookSide, side: Side, limit: Decimal | None
    ) -> Decimal:
        tick = self.tick_resolver(vb.builder.instrument_id)
        total = ZERO
        for price, qty in vb.available_levels(book_side):
            p = self._slipped(price, side, tick)
            if limit is not None and (
                (side is Side.BUY and p > limit) or (side is Side.SELL and p < limit)
            ):
                break
            total += qty
        return total

    def _match_taker(
        self,
        order: SimOrder,
        vb: VenueBook,
        book_side: BookSide,
        limit: Decimal | None,
        now_ns: int,
    ) -> list[Fill]:
        fills: list[Fill] = []
        tick = self.tick_resolver(order.contract_id)
        fee_schedule = self.fee_resolver(order.contract_id)
        for price, qty in vb.available_levels(book_side):
            if order.remaining == ZERO:
                break
            fill_price = self._slipped(price, order.side, tick)
            if limit is not None and (
                (order.side is Side.BUY and fill_price > limit)
                or (order.side is Side.SELL and fill_price < limit)
            ):
                break
            take = min(qty, order.remaining)
            fee = self._charge_fee(order, fee_schedule, fill_price, take, LiquidityRole.TAKER)
            order.record_fill(take, fill_price, now_ns)
            vb.consume(book_side, price, take)
            fills.append(
                self._make_fill(
                    order=order,
                    price=fill_price,
                    qty=take,
                    fee=fee,
                    schedule=fee_schedule,
                    now_ns=now_ns,
                    taker=True,
                )
            )
            self.stats.taker_fills += 1
        return fills

    def _charge_fee(
        self,
        order: SimOrder,
        schedule: FeeSchedule,
        price: Decimal,
        qty: Decimal,
        role: LiquidityRole,
    ) -> Decimal:
        charged, acc = schedule.order_fee_increment(
            self._fee_acc.get(order.order_id, ZERO), price=price, quantity=qty, role=role
        )
        self._fee_acc[order.order_id] = acc
        return charged

    def _make_fill(
        self,
        *,
        order: SimOrder,
        price: Decimal,
        qty: Decimal,
        fee: Decimal,
        schedule: FeeSchedule,
        now_ns: int,
        taker: bool,
    ) -> Fill:
        return Fill(
            fill_id=self._next_fill_id(order.order_id),
            order_id=order.order_id,
            venue=order.venue,
            contract_id=order.contract_id,
            instrument_id=order.instrument_id,
            side=order.side,
            fill_ts_ns=now_ns,
            price=price,
            quantity=qty,
            fee=fee,
            fee_schedule_version=schedule.tag,
            liquidity_role=LiquidityRole.TAKER if taker else LiquidityRole.MAKER,
            simulated=True,
            strategy_id=order.strategy_id,
            signal_id=order.signal_id,
        )

    def _maker_fill(self, order: SimOrder, qty: Decimal, now_ns: int) -> Fill | None:
        qty = min(qty, order.remaining)
        if qty <= ZERO:
            return None
        assert order.limit_price is not None
        schedule = self.fee_resolver(order.contract_id)
        fee = self._charge_fee(order, schedule, order.limit_price, qty, LiquidityRole.MAKER)
        order.record_fill(qty, order.limit_price, now_ns)
        self.stats.maker_fills += 1
        fill = self._make_fill(
            order=order,
            price=order.limit_price,
            qty=qty,
            fee=fee,
            schedule=schedule,
            now_ns=now_ns,
            taker=False,
        )
        if order.remaining == ZERO:
            self._remove_resting(order)
        return fill

    def _trade_fills(self, trade: TradeEvent, vb: VenueBook, now_ns: int) -> list[Fill]:
        fills: list[Fill] = []
        resting = self.resting_orders(trade.instrument_id)
        remaining_trade = trade.size
        for order in sorted(resting, key=lambda o: (o.arrival_ts_ns, o.order_id)):
            if remaining_trade <= ZERO:
                break
            if order.state.is_terminal or order.limit_price is None:
                continue
            limit = order.limit_price
            if order.side is Side.BUY:
                hits_us = trade.aggressor_side in (Side.SELL, None)
                through = trade.price < limit
                at_price = trade.price == limit and trade.aggressor_side is Side.SELL
            else:
                hits_us = trade.aggressor_side in (Side.BUY, None)
                through = trade.price > limit
                at_price = trade.price == limit and trade.aggressor_side is Side.BUY
            if not hits_us:
                continue
            if through:
                fill = self._maker_fill(order, remaining_trade, now_ns)
                if fill is not None:
                    remaining_trade -= fill.quantity
                    fills.append(fill)
                continue
            if at_price and self.queue_model != "trade_through":
                ahead = order.queue_ahead or ZERO
                if remaining_trade <= ahead:
                    order.queue_ahead = ahead - remaining_trade
                    side_key = (BookSide.BID if order.side is Side.BUY else BookSide.ASK, limit)
                    vb.recent_trade_qty[side_key] = (
                        vb.recent_trade_qty.get(side_key, ZERO) + remaining_trade
                    )
                    remaining_trade = ZERO
                    continue
                order.queue_ahead = ZERO
                excess = remaining_trade - ahead
                fill = self._maker_fill(order, excess, now_ns)
                if fill is not None:
                    remaining_trade = excess - fill.quantity
                    fills.append(fill)
        return fills

    def _level_sizes_for_resting(self, instrument_id: str, vb: VenueBook) -> dict[str, Decimal]:
        sizes = {}
        for order in self.resting_orders(instrument_id):
            if order.limit_price is None:
                continue
            side = BookSide.BID if order.side is Side.BUY else BookSide.ASK
            sizes[order.order_id] = vb.builder.quantity_at(side, order.limit_price)
        return sizes

    def _update_queues_after_book_change(
        self,
        instrument_id: str,
        vb: VenueBook,
        before: dict[str, Decimal],
        now_ns: int,
    ) -> list[Fill]:
        for order in self.resting_orders(instrument_id):
            if order.limit_price is None or order.queue_ahead is None:
                continue
            side = BookSide.BID if order.side is Side.BUY else BookSide.ASK
            displayed = vb.builder.quantity_at(side, order.limit_price)
            prev = before.get(order.order_id, displayed)
            if self.queue_model == "fifo_proxy" and displayed < prev and prev > ZERO:
                key = (side, order.limit_price)
                explained = vb.recent_trade_qty.get(key, ZERO)
                decrease = prev - displayed
                unexplained = max(ZERO, decrease - explained)
                vb.recent_trade_qty[key] = max(ZERO, explained - decrease)
                order.queue_ahead -= unexplained * order.queue_ahead / prev
            # never more queue ahead of us than is displayed at our price
            order.queue_ahead = max(ZERO, min(order.queue_ahead, displayed))
        return []

    def _cross_fills(self, instrument_id: str, vb: VenueBook, now_ns: int) -> list[Fill]:
        fills: list[Fill] = []
        for order in sorted(
            self.resting_orders(instrument_id), key=lambda o: (o.arrival_ts_ns, o.order_id)
        ):
            if order.state.is_terminal or order.limit_price is None:
                continue
            opposite = BookSide.ASK if order.side is Side.BUY else BookSide.BID
            for price, qty in vb.available_levels(opposite):
                crosses = (
                    price <= order.limit_price
                    if order.side is Side.BUY
                    else price >= order.limit_price
                )
                if not crosses or order.remaining == ZERO:
                    break
                fill = self._maker_fill(order, qty, now_ns)
                if fill is None:
                    break
                vb.consume(opposite, price, fill.quantity)
                fills.append(fill)
        return fills

    def _reject(self, order: SimOrder, now_ns: int, reason: ReasonCode) -> list[Fill]:
        order.transition(OrderState.REJECTED, now_ns, reason.value)
        self.stats.rejected += 1
        self.stats.count_reject(reason.value)
        return []

    def _remove_resting(self, order: SimOrder) -> None:
        ids = self._resting.get(order.instrument_id)
        if ids and order.order_id in ids:
            ids.remove(order.order_id)


def fills_total_quantity(fills: Iterable[Fill]) -> Decimal:
    return sum((f.quantity for f in fills), ZERO)
