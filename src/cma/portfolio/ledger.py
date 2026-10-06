"""Position and P&L accounting for binary contracts (canonical YES units).

* Positions are signed YES quantities. Long YES q at p costs q*p; a short YES position
  (economically a long NO) receives q*p and must hold $1 collateral per contract.
* Cash moves only by exact Decimal products/sums, so fills and fees reconcile exactly.
* Realized P&L uses the average-cost method (gross of fees); fees are tracked separately.
* Invariant (checked by :meth:`Portfolio.reconcile`):
  ``NAV == initial_cash + realized - fees + unrealized`` within fixed-point tolerance.
"""

from __future__ import annotations

import decimal
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal

from cma.domain.enums import MarkMethod, SettlementOutcome, Side
from cma.domain.models import BookSnapshot, Fill, Settlement
from cma.domain.numbers import ONE, ZERO

# High-precision context for average-cost division; cash itself never divides.
_ACCOUNTING_CONTEXT = decimal.Context(prec=50, rounding=decimal.ROUND_HALF_EVEN)
RECONCILIATION_TOLERANCE = Decimal("1e-12")


def _sign(x: Decimal) -> int:
    return (x > ZERO) - (x < ZERO)


@dataclass
class PositionState:
    contract_id: str
    quantity: Decimal = ZERO  # signed YES contracts
    cost_basis: Decimal = ZERO  # signed cash outlay of the open quantity (q * avg_cost)
    realized_pnl: Decimal = ZERO  # gross of fees
    fees: Decimal = ZERO
    event_family: str = ""
    n_fills: int = 0
    first_fill_ts_ns: int | None = None
    last_fill_ts_ns: int | None = None
    settled: bool = False

    @property
    def avg_cost(self) -> Decimal:
        if self.quantity == ZERO:
            return ZERO
        return _ACCOUNTING_CONTEXT.divide(self.cost_basis, self.quantity)

    def apply(self, side: Side, quantity: Decimal, price: Decimal, fee: Decimal) -> Decimal:
        """Apply a fill; return the realized (gross) P&L it generated."""
        with decimal.localcontext(_ACCOUNTING_CONTEXT):
            return self._apply(side, quantity, price, fee)

    def _apply(self, side: Side, quantity: Decimal, price: Decimal, fee: Decimal) -> Decimal:
        if quantity <= ZERO:
            raise ValueError("fill quantity must be positive")
        signed = quantity if side is Side.BUY else -quantity
        self.fees += fee
        self.n_fills += 1
        if self.quantity == ZERO or _sign(self.quantity) == _sign(signed):
            self.quantity += signed
            self.cost_basis += signed * price
            return ZERO
        closing = min(abs(self.quantity), quantity)
        # cost of the closed slice, pro-rata from the open cost basis
        closed_cost = self.cost_basis * closing / abs(self.quantity)
        direction = Decimal(_sign(self.quantity))
        proceeds = closing * price * direction  # value received for the closed slice
        realized = proceeds - closed_cost
        self.realized_pnl += realized
        self.quantity -= closing * direction
        self.cost_basis -= closed_cost
        if self.quantity == ZERO:
            self.cost_basis = ZERO
        remainder = quantity - closing
        if remainder > ZERO:  # position flips
            flip = remainder if side is Side.BUY else -remainder
            self.quantity = flip
            self.cost_basis = flip * price
        return realized

    def unrealized(self, mark: Decimal) -> Decimal:
        return self.quantity * mark - self.cost_basis

    def worst_case_loss(self) -> Decimal:
        """Max loss at settlement of the open position: max(C - Q, C), floored at 0."""
        c, q = self.cost_basis, self.quantity
        return max(ZERO, c - q, c)


def worst_case_loss(quantity: Decimal, cost_basis: Decimal) -> Decimal:
    return max(ZERO, cost_basis - quantity, cost_basis)


@dataclass
class Portfolio:
    initial_cash: Decimal
    cash: Decimal = field(init=False)
    positions: dict[str, PositionState] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    realized_total: Decimal = ZERO
    fees_total: Decimal = ZERO
    settlements: list[tuple[Settlement, Decimal]] = field(default_factory=list)
    void_policy: str = "refund_cost"  # or "settle_value"

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    # ------------------------------------------------------------------ mutation

    def position(self, contract_id: str) -> PositionState:
        pos = self.positions.get(contract_id)
        if pos is None:
            pos = PositionState(contract_id=contract_id)
            self.positions[contract_id] = pos
        return pos

    def apply_fill(self, fill: Fill, event_family: str = "") -> Decimal:
        with decimal.localcontext(_ACCOUNTING_CONTEXT):
            return self._apply_fill(fill, event_family)

    def _apply_fill(self, fill: Fill, event_family: str) -> Decimal:
        pos = self.position(fill.contract_id)
        if pos.settled:
            raise ValueError(f"fill on settled contract {fill.contract_id}")
        if event_family:
            pos.event_family = event_family
        signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
        self.cash -= signed * fill.price
        self.cash -= fill.fee
        realized = pos.apply(fill.side, fill.quantity, fill.price, fill.fee)
        if pos.first_fill_ts_ns is None:
            pos.first_fill_ts_ns = fill.fill_ts_ns
        pos.last_fill_ts_ns = fill.fill_ts_ns
        self.realized_total += realized
        self.fees_total += fill.fee
        self.fills.append(fill)
        return realized

    def settle(self, settlement: Settlement) -> Decimal:
        """Settle a contract; returns the realized P&L booked at settlement."""
        with decimal.localcontext(_ACCOUNTING_CONTEXT):
            return self._settle(settlement)

    def _settle(self, settlement: Settlement) -> Decimal:
        pos = self.position(settlement.contract_id)
        if pos.settled:
            return ZERO
        if settlement.outcome is SettlementOutcome.VOID and self.void_policy == "refund_cost":
            value_per_contract = None
        else:
            value_per_contract = settlement.yes_value
        if value_per_contract is None:
            # Unwind at cost: cash returns the open cost basis, no P&L.
            self.cash += pos.cost_basis
            realized = ZERO
        else:
            self.cash += pos.quantity * value_per_contract
            realized = pos.quantity * value_per_contract - pos.cost_basis
        pos.realized_pnl += realized
        self.realized_total += realized
        pos.quantity = ZERO
        pos.cost_basis = ZERO
        pos.settled = True
        self.settlements.append((settlement, realized))
        return realized

    # ------------------------------------------------------------------ valuation

    def open_positions(self) -> dict[str, PositionState]:
        return {k: p for k, p in self.positions.items() if p.quantity != ZERO}

    def unrealized(self, marks: Mapping[str, Decimal]) -> Decimal:
        total = ZERO
        for cid, pos in self.positions.items():
            if pos.quantity == ZERO:
                continue
            total += pos.unrealized(self._mark(cid, marks))
        return total

    def nav(self, marks: Mapping[str, Decimal]) -> Decimal:
        total = self.cash
        for cid, pos in self.positions.items():
            if pos.quantity != ZERO:
                total += pos.quantity * self._mark(cid, marks)
        return total

    def reconcile(self, marks: Mapping[str, Decimal]) -> Decimal:
        """Return NAV - (initial + realized - fees + unrealized); ~0 when consistent."""
        expected = self.initial_cash + self.realized_total - self.fees_total
        expected += self.unrealized(marks)
        return self.nav(marks) - expected

    def collateral_required(self) -> Decimal:
        """$1 per short YES contract (sale proceeds are already in cash)."""
        return sum((-p.quantity for p in self.positions.values() if p.quantity < ZERO), ZERO)

    def buying_power(self) -> Decimal:
        return self.cash - self.collateral_required()

    def exposure(self, contract_id: str) -> Decimal:
        pos = self.positions.get(contract_id)
        return pos.worst_case_loss() if pos else ZERO

    def family_exposure(self, event_family: str) -> Decimal:
        return sum(
            (
                p.worst_case_loss()
                for p in self.positions.values()
                if p.event_family == event_family
            ),
            ZERO,
        )

    def total_open_risk(self) -> Decimal:
        return sum((p.worst_case_loss() for p in self.positions.values()), ZERO)

    @staticmethod
    def _mark(contract_id: str, marks: Mapping[str, Decimal]) -> Decimal:
        try:
            return marks[contract_id]
        except KeyError as exc:
            raise KeyError(f"no mark for open position {contract_id}") from exc


def mark_from_book(
    book: BookSnapshot | None, quantity: Decimal, method: MarkMethod, fallback: Decimal
) -> Decimal:
    """Mark price for a position from a canonical YES book."""
    if book is None:
        return fallback
    if method is MarkMethod.MID:
        return book.mid if book.mid is not None else fallback
    if method is MarkMethod.CONSERVATIVE:
        level = book.best_bid if quantity > ZERO else book.best_ask
        return level.price if level is not None else fallback
    return fallback


def clamp_probability(value: Decimal) -> Decimal:
    return min(ONE, max(ZERO, value))


def total_fees(fills: Iterable[Fill]) -> Decimal:
    return sum((f.fee for f in fills), ZERO)
