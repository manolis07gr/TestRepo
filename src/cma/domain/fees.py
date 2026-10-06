"""Versioned venue fee schedules.

Fee rules change; every fill persists the ``tag`` (id@version) of the schedule that priced
it. Schedules are pure functions of (price, quantity, liquidity role); ``price`` is the
canonical YES price. The venue formulas used here are symmetric in p <-> 1-p, so pricing a
NO trade at q gives the same fee as the equivalent YES trade at 1-q.

Rounding is two-stage, mirroring the venues' published mechanics:

* ``fill_rounding_quantum`` - each fill's raw fee is rounded *up* to this quantum;
* ``order_rounding_quantum`` - cumulative per-order rounding: the amount charged for a
  fill is ``ceil(acc_after) - ceil(acc_before)``, so an order's total converges to the
  rounded-up raw fee of the whole order (Kalshi's per-order fee accumulator).

Sources (verified 2026-10-06 from official fee pages/terms; re-verify before relying on
results, and prefer per-series values from the venue API at runtime):

* Kalshi: taker = roundup(M x 0.07 x C x P x (1-P)); maker 0.0175 only on series with
  ``fee_type=quadratic_with_maker_fees``; S&P/Nasdaq index series 0.035; crypto series
  multiplier 1 (fee schedule "July 2026 - 7.7.26 update"; docs fee_rounding page).
* Polymarket: taker = C x rate x p x (1-p) rounded to 1e-5; crypto rate 0.07 (0.072 from
  2026-03-30 to ~July 2026), sports 0.05, finance/politics 0.04, makers 0 (+ rebates, not
  credited here). Q1-2026 pilot on 15-minute crypto markets: C x p x 0.25 x (p(1-p))^2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from cma.domain.enums import LiquidityRole
from cma.domain.numbers import CENT, ONE, ZERO, ceil_to, validate_probability


class FeeSchedule(ABC):
    """Fee in USD for ``quantity`` $1-face contracts traded at YES ``price``."""

    schedule_id: str
    version: str
    fill_rounding_quantum: Decimal | None = None
    order_rounding_quantum: Decimal | None = None

    @abstractmethod
    def raw_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        """Un-rounded fee."""

    def fill_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        raw = self.raw_fee(price=price, quantity=quantity, role=role)
        q = self.fill_rounding_quantum
        return raw if q is None or raw == ZERO else ceil_to(raw, q)

    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        """Fee of a single-fill order (both rounding stages applied)."""
        charged, _ = self.order_fee_increment(ZERO, price=price, quantity=quantity, role=role)
        return charged

    def order_fee_increment(
        self, accumulated: Decimal, *, price: Decimal, quantity: Decimal, role: LiquidityRole
    ) -> tuple[Decimal, Decimal]:
        """(amount charged for this fill, new accumulated fill-rounded fee of the order)."""
        f = self.fill_fee(price=price, quantity=quantity, role=role)
        new_acc = accumulated + f
        q = self.order_rounding_quantum
        if q is None:
            return f, new_acc
        before = ceil_to(accumulated, q) if accumulated > ZERO else ZERO
        after = ceil_to(new_acc, q) if new_acc > ZERO else ZERO
        return after - before, new_acc

    def fee_per_contract(self, *, price: Decimal, role: LiquidityRole) -> Decimal:
        """Marginal un-rounded fee per contract (edge estimation)."""
        return self.raw_fee(price=price, quantity=ONE, role=role)

    @property
    def tag(self) -> str:
        return f"{self.schedule_id}@{self.version}"


@dataclass(frozen=True)
class QuadraticFeeSchedule(FeeSchedule):
    """raw = multiplier x rate x C x P x (1 - P)  (Kalshi's trading-fee formula)."""

    schedule_id: str
    version: str
    taker_rate: Decimal
    maker_rate: Decimal = ZERO
    multiplier: Decimal = ONE
    fill_rounding_quantum: Decimal | None = Decimal("0.0001")
    order_rounding_quantum: Decimal | None = CENT

    def raw_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        p = validate_probability(price)
        if quantity < ZERO:
            raise ValueError("quantity must be non-negative")
        rate = self.taker_rate if role is LiquidityRole.TAKER else self.maker_rate
        return self.multiplier * rate * quantity * p * (ONE - p)


@dataclass(frozen=True)
class PowerFeeSchedule(FeeSchedule):
    """raw = C x rate x (p (1 - p)) ** exponent  [x p if ``price_factor``].

    Polymarket's fee curve family: exponent 1 without price factor is the current
    schedule; (0.25, exponent 2, price_factor) was the Q1-2026 15-minute crypto pilot.
    """

    schedule_id: str
    version: str
    taker_rate: Decimal
    exponent: int = 1
    price_factor: bool = False
    maker_rate: Decimal = ZERO
    fill_rounding_quantum: Decimal | None = Decimal("0.00001")
    order_rounding_quantum: Decimal | None = None

    def raw_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        p = validate_probability(price)
        if quantity < ZERO:
            raise ValueError("quantity must be non-negative")
        rate = self.taker_rate if role is LiquidityRole.TAKER else self.maker_rate
        raw = quantity * rate * (p * (ONE - p)) ** self.exponent
        return raw * p if self.price_factor else raw


@dataclass(frozen=True)
class NotionalBpsFeeSchedule(FeeSchedule):
    """raw = notional x bps (reference venues or generic stress)."""

    schedule_id: str
    version: str
    taker_bps: Decimal
    maker_bps: Decimal = ZERO

    def raw_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        if quantity < ZERO or price < ZERO:
            raise ValueError("price and quantity must be non-negative")
        bps = self.taker_bps if role is LiquidityRole.TAKER else self.maker_bps
        return price * quantity * bps / Decimal(10_000)


@dataclass(frozen=True)
class ZeroFeeSchedule(FeeSchedule):
    schedule_id: str = "zero"
    version: str = "1"

    def raw_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        validate_probability(price)
        return ZERO


@dataclass(frozen=True)
class ScaledFeeSchedule(FeeSchedule):
    """Cost-stress wrapper: multiplies another schedule's raw fee by ``factor``."""

    base: FeeSchedule
    factor: Decimal

    def __post_init__(self) -> None:
        if self.factor < ZERO:
            raise ValueError("fee scale factor must be non-negative")

    @property
    def schedule_id(self) -> str:  # type: ignore[override]
        return f"{self.base.schedule_id}*{self.factor}"

    @property
    def version(self) -> str:  # type: ignore[override]
        return self.base.version

    @property
    def fill_rounding_quantum(self) -> Decimal | None:  # type: ignore[override]
        return self.base.fill_rounding_quantum

    @property
    def order_rounding_quantum(self) -> Decimal | None:  # type: ignore[override]
        return self.base.order_rounding_quantum

    def raw_fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        return self.base.raw_fee(price=price, quantity=quantity, role=role) * self.factor


KALSHI_STANDARD = QuadraticFeeSchedule(
    schedule_id="kalshi-standard", version="2026-07-07", taker_rate=Decimal("0.07")
)
KALSHI_MAKER_FEE = QuadraticFeeSchedule(
    schedule_id="kalshi-maker-fee-series",
    version="2026-07-07",
    taker_rate=Decimal("0.07"),
    maker_rate=Decimal("0.0175"),
)
KALSHI_INDEX = QuadraticFeeSchedule(
    schedule_id="kalshi-index-series", version="2026-07-07", taker_rate=Decimal("0.035")
)
POLYMARKET_ZERO = ZeroFeeSchedule(schedule_id="polymarket-no-fee", version="2026-10")
POLYMARKET_CRYPTO_TAKER = PowerFeeSchedule(
    schedule_id="polymarket-crypto-taker", version="2026-07", taker_rate=Decimal("0.07")
)
POLYMARKET_CRYPTO_TAKER_0330 = PowerFeeSchedule(
    schedule_id="polymarket-crypto-taker-2026-03-30",
    version="2026-03-30",
    taker_rate=Decimal("0.072"),
)
POLYMARKET_CRYPTO_PILOT = PowerFeeSchedule(
    schedule_id="polymarket-crypto-15m-pilot",
    version="2026-01-05",
    taker_rate=Decimal("0.25"),
    exponent=2,
    price_factor=True,
)
POLYMARKET_SPORTS_TAKER = PowerFeeSchedule(
    schedule_id="polymarket-sports-taker", version="2026-07", taker_rate=Decimal("0.05")
)
POLYMARKET_FINANCE_TAKER = PowerFeeSchedule(
    schedule_id="polymarket-finance-taker", version="2026-03-30", taker_rate=Decimal("0.04")
)

_REGISTRY: dict[str, FeeSchedule] = {
    s.schedule_id: s
    for s in (
        KALSHI_STANDARD,
        KALSHI_MAKER_FEE,
        KALSHI_INDEX,
        POLYMARKET_ZERO,
        POLYMARKET_CRYPTO_TAKER,
        POLYMARKET_CRYPTO_TAKER_0330,
        POLYMARKET_CRYPTO_PILOT,
        POLYMARKET_SPORTS_TAKER,
        POLYMARKET_FINANCE_TAKER,
    )
}
_REGISTRY["zero"] = ZeroFeeSchedule()


def register_fee_schedule(schedule: FeeSchedule, *, replace: bool = False) -> None:
    if schedule.schedule_id in _REGISTRY and not replace:
        raise ValueError(f"fee schedule {schedule.schedule_id!r} already registered")
    _REGISTRY[schedule.schedule_id] = schedule


def get_fee_schedule(schedule_id: str) -> FeeSchedule:
    try:
        return _REGISTRY[schedule_id]
    except KeyError as exc:
        raise KeyError(f"unknown fee schedule {schedule_id!r}; known: {sorted(_REGISTRY)}") from exc


def known_fee_schedules() -> dict[str, FeeSchedule]:
    return dict(_REGISTRY)
