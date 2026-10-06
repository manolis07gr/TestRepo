"""YES/NO normalization for binary contracts.

A binary contract pays $1 per YES contract if the event occurs, $1 per NO contract
otherwise. Holding one YES and one NO is a riskless $1, so:

* buying NO at q  ==  selling YES at 1 - q
* selling NO at q ==  buying YES at 1 - q
* a NO bid at q   ==  a YES ask at 1 - q (and vice versa)

Everything downstream of the adapters works in canonical YES terms.
"""

from __future__ import annotations

from decimal import Decimal

from cma.domain.enums import BookSide, Outcome, Side
from cma.domain.models import BookLevel
from cma.domain.numbers import ONE, validate_probability


def complement(price: Decimal) -> Decimal:
    """1 - p, validating both sides are probabilities."""
    return ONE - validate_probability(price)


def to_yes_order(outcome: Outcome, side: Side, price: Decimal) -> tuple[Side, Decimal]:
    """Map an (outcome, side, price) order/trade to its canonical YES equivalent."""
    p = validate_probability(price)
    if outcome is Outcome.YES:
        return side, p
    return side.opposite, ONE - p


def from_yes_order(outcome: Outcome, side: Side, yes_price: Decimal) -> tuple[Side, Decimal]:
    """Inverse of :func:`to_yes_order`: express a YES order in ``outcome`` terms."""
    p = validate_probability(yes_price)
    if outcome is Outcome.YES:
        return side, p
    return side.opposite, ONE - p


def yes_book_side(outcome: Outcome, side: BookSide) -> BookSide:
    """Which side of the canonical YES book a quote on ``outcome``'s book lands on."""
    return side if outcome is Outcome.YES else side.opposite


def no_levels_to_yes(levels: tuple[BookLevel, ...] | list[BookLevel]) -> tuple[BookLevel, ...]:
    """Transform NO-book levels (any side) to YES levels on the opposite side.

    The result is re-sorted best-first for the destination side: NO bids sorted
    descending become YES asks sorted ascending automatically because 1 - p reverses order.
    """
    return tuple(BookLevel(complement(lvl.price), lvl.quantity) for lvl in levels)


def yes_payoff(position_yes: Decimal, settlement_yes_value: Decimal) -> Decimal:
    """Settlement cash flow of a signed canonical-YES position."""
    return position_yes * validate_probability(settlement_yes_value)
