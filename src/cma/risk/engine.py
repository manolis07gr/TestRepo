"""Pre-trade risk engine (scope s.14). Every limit is configuration-driven.

Exposure is the worst-case settlement loss of (known position + all working/pending orders
on the same contract assumed filled at their limits + the candidate order):
``max(0, C - Q, C)`` where Q is the signed YES quantity and C the signed cash outlay.
Family and portfolio exposures are conservative sums of contract exposures.

* Kill switch: synchronously blocks every new order and requests cancellation of every
  working simulated order through the registered callback (T031).
* Daily loss stop: once NAV falls ``daily_loss_stop_pct`` below the UTC-day start NAV, only
  risk-reducing orders (and cancels) are allowed until the next UTC day (T030).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal

from cma.config import RiskConfig
from cma.domain.enums import ReasonCode, Side
from cma.domain.models import SimOrder
from cma.domain.numbers import ZERO
from cma.domain.time import utc_day_start_ns
from cma.portfolio.ledger import worst_case_loss

HUNDRED = Decimal(100)


@dataclass(frozen=True, slots=True)
class PositionView:
    quantity: Decimal  # signed YES
    cost_basis: Decimal  # signed cash outlay of the open quantity
    event_family: str = ""


@dataclass(frozen=True, slots=True)
class OrderRequest:
    contract_id: str
    event_family: str
    side: Side
    quantity: Decimal
    price: Decimal  # limit (or worst acceptable) YES price
    ts_ns: int


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reasons: tuple[ReasonCode, ...] = ()
    detail: str = ""


@dataclass
class RiskEngine:
    config: RiskConfig
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    day_start_ns: int | None = None
    day_start_nav: Decimal | None = None
    daily_stop_active: bool = False
    breaches: list[tuple[int, ReasonCode, str]] = field(default_factory=list)
    _cancel_all: Callable[[int], None] | None = None

    # ------------------------------------------------------------------ controls

    def register_cancel_all(self, callback: Callable[[int], None]) -> None:
        self._cancel_all = callback

    def activate_kill_switch(self, ts_ns: int, reason: str) -> None:
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.breaches.append((ts_ns, ReasonCode.KILL_SWITCH, reason))
        if self._cancel_all is not None:
            self._cancel_all(ts_ns)

    def reset_kill_switch(self, ts_ns: int, operator: str) -> None:
        if not operator:
            raise ValueError("kill switch reset requires an operator identity")
        self.kill_switch_active = False
        self.breaches.append((ts_ns, ReasonCode.KILL_SWITCH, f"reset by {operator}"))

    def on_nav(self, ts_ns: int, nav: Decimal) -> None:
        day = utc_day_start_ns(ts_ns)
        if self.day_start_ns != day:
            self.day_start_ns = day
            self.day_start_nav = nav
            self.daily_stop_active = False
        assert self.day_start_nav is not None
        threshold = self.day_start_nav * (1 - self.config.daily_loss_stop_pct / HUNDRED)
        if not self.daily_stop_active and nav <= threshold:
            self.daily_stop_active = True
            self.breaches.append(
                (ts_ns, ReasonCode.DAILY_STOP, f"nav {nav} <= stop level {threshold}")
            )

    # ------------------------------------------------------------------ checks

    @staticmethod
    def _apply(
        q: Decimal, c: Decimal, side: Side, qty: Decimal, price: Decimal
    ) -> tuple[Decimal, Decimal]:
        signed = qty if side is Side.BUY else -qty
        return q + signed, c + signed * price

    def contract_exposure(
        self,
        position: PositionView | None,
        working: Iterable[SimOrder],
        extra: OrderRequest | None = None,
    ) -> Decimal:
        """Worst-case loss with every working order (and ``extra``) assumed filled.

        Buys and sells are evaluated separately so that offsetting working orders cannot
        hide risk: the result is the max over the all-buys-fill and all-sells-fill cases.
        """
        q0 = position.quantity if position else ZERO
        c0 = position.cost_basis if position else ZERO
        cases = []
        for fill_side in (Side.BUY, Side.SELL):
            q, c = q0, c0
            for o in working:
                if o.side is fill_side and o.remaining > ZERO:
                    px = (
                        o.limit_price
                        if o.limit_price is not None
                        else (Decimal(1) if o.side is Side.BUY else ZERO)
                    )
                    q, c = self._apply(q, c, o.side, o.remaining, px)
            if extra is not None and extra.side is fill_side:
                q, c = self._apply(q, c, extra.side, extra.quantity, extra.price)
            cases.append(worst_case_loss(q, c))
        return max(cases)

    def check_order(
        self,
        request: OrderRequest,
        *,
        nav: Decimal,
        positions: Mapping[str, PositionView],
        working_orders: Iterable[SimOrder],
        buying_power: Decimal | None = None,
        family_of: Callable[[str], str] | None = None,
    ) -> RiskDecision:
        if self.kill_switch_active:
            return self._deny(request, ReasonCode.KILL_SWITCH, self.kill_switch_reason)

        working = list(working_orders)
        pos = positions.get(request.contract_id)
        on_contract = [o for o in working if o.contract_id == request.contract_id]
        before = self.contract_exposure(pos, on_contract)
        after = self.contract_exposure(pos, on_contract, request)
        reduces = self._reduces(pos, request) and after <= before

        if self.daily_stop_active and not reduces:
            return self._deny(request, ReasonCode.DAILY_STOP, "daily loss stop active")
        if reduces:
            return RiskDecision(approved=True, detail="risk-reducing")

        limit_contract = nav * self.config.max_contract_nav_pct / HUNDRED
        if after > limit_contract:
            return self._deny(
                request, ReasonCode.CONTRACT_LIMIT, f"exposure {after} > {limit_contract}"
            )

        family_total = after
        portfolio_total = after
        seen = {request.contract_id}
        by_contract: dict[str, list[SimOrder]] = {}
        for o in working:
            by_contract.setdefault(o.contract_id, []).append(o)
        for cid in set(positions) | set(by_contract):
            if cid in seen:
                continue
            seen.add(cid)
            exp = self.contract_exposure(positions.get(cid), by_contract.get(cid, []))
            portfolio_total += exp
            p = positions.get(cid)
            fam = p.event_family if p and p.event_family else ""
            if not fam and family_of is not None:
                fam = family_of(cid)
            if request.event_family and fam == request.event_family:
                family_total += exp

        limit_family = nav * self.config.max_event_nav_pct / HUNDRED
        if request.event_family and family_total > limit_family:
            return self._deny(
                request, ReasonCode.EVENT_LIMIT, f"family exposure {family_total} > {limit_family}"
            )
        limit_total = nav * self.config.max_total_open_nav_pct / HUNDRED
        if portfolio_total > limit_total:
            return self._deny(
                request,
                ReasonCode.PORTFOLIO_LIMIT,
                f"open risk {portfolio_total} > {limit_total}",
            )
        if buying_power is not None:
            need = self._capital_needed(request)
            if need > buying_power:
                return self._deny(
                    request, ReasonCode.INSUFFICIENT_CAPITAL, f"needs {need} > {buying_power}"
                )
        return RiskDecision(approved=True)

    @staticmethod
    def _capital_needed(req: OrderRequest) -> Decimal:
        if req.side is Side.BUY:
            return req.quantity * req.price
        return req.quantity * (1 - req.price)

    @staticmethod
    def _reduces(pos: PositionView | None, req: OrderRequest) -> bool:
        if pos is None or pos.quantity == ZERO:
            return False
        if pos.quantity > ZERO:
            return req.side is Side.SELL and req.quantity <= pos.quantity
        return req.side is Side.BUY and req.quantity <= -pos.quantity

    def _deny(self, req: OrderRequest, reason: ReasonCode, detail: str) -> RiskDecision:
        self.breaches.append((req.ts_ns, reason, f"{req.contract_id}: {detail}"))
        return RiskDecision(approved=False, reasons=(reason,), detail=detail)
