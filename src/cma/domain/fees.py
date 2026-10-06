"""Versioned venue fee schedules.

Fee rules change; every fill persists the ``version`` of the schedule that priced it.
Schedules are pure functions of (price, quantity, liquidity role); ``price`` is the
canonical YES price. Kalshi/Polymarket formulas are symmetric in p <-> 1-p, so pricing a
NO trade at q gives the same fee as the equivalent YES trade at 1-q.

The defaults below encode the published formulas as understood at implementation time.
They MUST be re-verified against the venues' current fee schedules before any result is
relied upon (scope s.28); they are data, selectable by id from configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from cma.domain.enums import LiquidityRole
from cma.domain.numbers import CENT, ONE, ZERO, ceil_to, validate_probability


class FeeSchedule(ABC):
    """Fee in USD for a fill of ``quantity`` $1-face contracts at YES ``price``."""

    schedule_id: str
    version: str

    @abstractmethod
    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal: ...

    def fee_per_contract(self, *, price: Decimal, role: LiquidityRole) -> Decimal:
        """Marginal (un-rounded) fee per contract, for edge estimation."""
        big = Decimal(1_000_000)
        return self.fee(price=price, quantity=big, role=role) / big

    @property
    def tag(self) -> str:
        return f"{self.schedule_id}@{self.version}"


@dataclass(frozen=True)
class QuadraticFeeSchedule(FeeSchedule):
    """fee = round_up(rate x C x P x (1 - P), rounding_quantum) per fill.

    Kalshi's published trading-fee formula has this form (taker rate 0.07; maker fees,
    where charged, 0.0175), rounded up to the next cent. Rounding per simulated fill is
    conservative relative to per-order rounding.
    """

    schedule_id: str
    version: str
    taker_rate: Decimal
    maker_rate: Decimal = ZERO
    rounding_quantum: Decimal | None = CENT

    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        p = validate_probability(price)
        if quantity < ZERO:
            raise ValueError("quantity must be non-negative")
        rate = self.taker_rate if role is LiquidityRole.TAKER else self.maker_rate
        raw = rate * quantity * p * (ONE - p)
        if self.rounding_quantum is None or raw == ZERO:
            return raw
        return ceil_to(raw, self.rounding_quantum)


@dataclass(frozen=True)
class PowerFeeSchedule(FeeSchedule):
    """fee = C x p x rate x (p x (1 - p)) ** exponent, rounded to ``rounding_quantum``.

    Shape used for Polymarket's fee-enabled (short-duration crypto) markets: the fee
    vanishes near 0/1 and peaks at p = 0.5. Makers pay nothing by default.
    """

    schedule_id: str
    version: str
    taker_rate: Decimal
    exponent: int = 2
    maker_rate: Decimal = ZERO
    rounding_quantum: Decimal | None = Decimal("0.0001")

    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        p = validate_probability(price)
        if quantity < ZERO:
            raise ValueError("quantity must be non-negative")
        rate = self.taker_rate if role is LiquidityRole.TAKER else self.maker_rate
        raw = quantity * p * rate * (p * (ONE - p)) ** self.exponent
        if self.rounding_quantum is None or raw == ZERO:
            return raw
        return ceil_to(raw, self.rounding_quantum)


@dataclass(frozen=True)
class NotionalBpsFeeSchedule(FeeSchedule):
    """fee = notional x bps (reference venues or generic stress)."""

    schedule_id: str
    version: str
    taker_bps: Decimal
    maker_bps: Decimal = ZERO

    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        if quantity < ZERO or price < ZERO:
            raise ValueError("price and quantity must be non-negative")
        bps = self.taker_bps if role is LiquidityRole.TAKER else self.maker_bps
        return price * quantity * bps / Decimal(10_000)


@dataclass(frozen=True)
class ZeroFeeSchedule(FeeSchedule):
    schedule_id: str = "zero"
    version: str = "1"

    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        validate_probability(price)
        return ZERO


@dataclass(frozen=True)
class ScaledFeeSchedule(FeeSchedule):
    """Cost-stress wrapper: multiplies another schedule's fee by ``factor`` (>= 1)."""

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

    def fee(self, *, price: Decimal, quantity: Decimal, role: LiquidityRole) -> Decimal:
        return self.base.fee(price=price, quantity=quantity, role=role) * self.factor


KALSHI_STANDARD = QuadraticFeeSchedule(
    schedule_id="kalshi-standard",
    version="2026-10-unverified",
    taker_rate=Decimal("0.07"),
    maker_rate=ZERO,
)
KALSHI_MAKER_FEE = QuadraticFeeSchedule(
    schedule_id="kalshi-maker-fee-series",
    version="2026-10-unverified",
    taker_rate=Decimal("0.07"),
    maker_rate=Decimal("0.0175"),
)
POLYMARKET_ZERO = ZeroFeeSchedule(schedule_id="polymarket-no-fee", version="2026-10-unverified")
POLYMARKET_CRYPTO_TAKER = PowerFeeSchedule(
    schedule_id="polymarket-crypto-taker",
    version="2026-10-unverified",
    taker_rate=Decimal("0.25"),
    exponent=2,
)

_REGISTRY: dict[str, FeeSchedule] = {
    s.schedule_id: s
    for s in (KALSHI_STANDARD, KALSHI_MAKER_FEE, POLYMARKET_ZERO, POLYMARKET_CRYPTO_TAKER)
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
