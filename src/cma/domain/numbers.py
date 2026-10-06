"""Exact numeric helpers.

Accounting and execution paths use :class:`decimal.Decimal`. Floats are accepted only
through :func:`from_float`, which forces the caller to choose a quantum, so binary
floating-point error can never leak silently into money or probability values.
"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, InvalidOperation

from cma.domain.errors import InvalidPriceError, InvalidProbabilityError

ZERO = Decimal(0)
ONE = Decimal(1)
CENT = Decimal("0.01")
BPS = Decimal("0.0001")

type DecimalLike = Decimal | int | str


def to_decimal(value: DecimalLike) -> Decimal:
    """Convert ``value`` to a finite Decimal without going through binary floats."""
    if isinstance(value, bool):
        raise TypeError("bool is not a numeric value")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        try:
            result = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"not a decimal number: {value!r}") from exc
    else:
        raise TypeError(
            f"refusing implicit {type(value).__name__}->Decimal conversion; "
            "use from_float() with an explicit quantum"
        )
    if not result.is_finite():
        raise ValueError(f"non-finite decimal: {value!r}")
    return result


def from_float(value: float, quantum: Decimal) -> Decimal:
    """Explicit float boundary crossing: round ``value`` half-even to ``quantum``."""
    if not math.isfinite(value):
        raise ValueError(f"non-finite float: {value!r}")
    return Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_EVEN)


def to_float(value: Decimal) -> float:
    """Explicit Decimal->float boundary crossing for statistical code."""
    return float(value)


def quantize(value: Decimal, quantum: Decimal, rounding: str = ROUND_HALF_EVEN) -> Decimal:
    return value.quantize(quantum, rounding=rounding)


def ceil_to(value: Decimal, quantum: Decimal) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_CEILING)


def floor_to(value: Decimal, quantum: Decimal) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_FLOOR)


def validate_probability(value: DecimalLike, *, name: str = "probability") -> Decimal:
    """Return ``value`` as a Decimal in [0, 1]; raise InvalidProbabilityError otherwise."""
    try:
        d = to_decimal(value)
    except (TypeError, ValueError) as exc:
        raise InvalidProbabilityError(f"{name} is not a finite decimal: {value!r}") from exc
    if d < ZERO or d > ONE:
        raise InvalidProbabilityError(f"{name} {d} outside [0, 1]")
    return d


def is_on_tick(price: Decimal, tick: Decimal) -> bool:
    if tick <= ZERO:
        raise InvalidPriceError(f"tick must be positive, got {tick}")
    return price % tick == ZERO


def snap_to_tick(price: Decimal, tick: Decimal, rounding: str = ROUND_HALF_EVEN) -> Decimal:
    """Round ``price`` to the nearest multiple of ``tick`` using ``rounding``."""
    if tick <= ZERO:
        raise InvalidPriceError(f"tick must be positive, got {tick}")
    units = (price / tick).quantize(ONE, rounding=rounding)
    return units * tick


def bps_to_decimal(bps: DecimalLike) -> Decimal:
    """Convert basis points of a $1 face value into a probability/price amount."""
    return to_decimal(bps) * BPS


def decimal_to_bps(value: Decimal) -> Decimal:
    return value / BPS
