"""Common types and the executable-package walker for structural constraints (s.10.4, H4).

Conventions
-----------
* Books are canonical YES books (prices = YES probabilities in [0, 1]); a leg's ``side`` and
  ``price`` are canonical YES terms: BUY = buy YES at ``price``; SELL = sell YES at ``price``
  (economically: buy NO at ``1 - price``). ``outcome`` records which token book a quote came
  from when a venue lists separate YES/NO tokens.
* Arithmetic is exact ``Decimal`` on executable prices and *displayed* depth, never last
  prices. Fees are venue fee schedules evaluated at the canonical YES price (the shared
  ``FeeSchedule`` convention), one single-fill order per (leg, price level) - conservative
  relative to per-order fee accumulation.
* A package is one unit of every leg. Its guaranteed worst-case settlement value per unit is
  ``constant_payoff`` (e.g. +1 for buying every outcome of an exhaustive set), so
  ``gross per unit = sum(SELL prices) - sum(BUY prices) + constant_payoff``.
* Totals (gross, fees, other costs, net profit) are exact; per-unit fields are totals divided
  by the package quantity (Decimal context precision).
* Detection is strict: an opportunity is reported only if
  ``net_profit - tolerance * quantity > 0``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from cma.domain.binary import from_yes_order
from cma.domain.enums import LiquidityRole, MappingStatus, Outcome, Side, Venue
from cma.domain.errors import CMAError, LookaheadError, MappingNotApprovedError
from cma.domain.fees import FeeSchedule
from cma.domain.models import BookLevel, BookSnapshot, ContractMapping, stable_id
from cma.domain.numbers import ONE, ZERO
from cma.domain.time import NS_PER_DAY

YEAR_NS = 365 * NS_PER_DAY


class OpportunityKind(StrEnum):
    NESTED_THRESHOLD = "NESTED_THRESHOLD"
    EXHAUSTIVE_UNDER_ROUND = "EXHAUSTIVE_UNDER_ROUND"
    EXHAUSTIVE_OVER_ROUND = "EXHAUSTIVE_OVER_ROUND"
    COMPLEMENT_BUY_BOTH = "COMPLEMENT_BUY_BOTH"
    COMPLEMENT_SELL_BOTH = "COMPLEMENT_SELL_BOTH"
    CROSS_VENUE = "CROSS_VENUE"


class StructuralInputError(CMAError, ValueError):
    """Inputs do not form a family/set for which the constraint is semantically valid."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Leg:
    contract_id: str
    instrument_id: str
    venue: Venue
    side: Side  # canonical YES side
    price: Decimal  # canonical YES price
    quantity: Decimal
    fee: Decimal  # total fee for this leg (all of ``quantity``)
    outcome: Outcome = Outcome.YES  # token book the quote came from

    @property
    def token_side(self) -> Side:
        """Side in the traded token's own terms (e.g. BUY NO for a canonical SELL on NO)."""
        return from_yes_order(self.outcome, self.side, self.price)[0]

    @property
    def token_price(self) -> Decimal:
        return from_yes_order(self.outcome, self.side, self.price)[1]

    @property
    def cash_outlay(self) -> Decimal:
        """Capital committed: YES cost for BUY, NO cost (1 - p) for SELL, plus fee."""
        unit = self.price if self.side is Side.BUY else ONE - self.price
        return unit * self.quantity + self.fee


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuralOpportunity:
    kind: OpportunityKind
    legs: tuple[Leg, ...]
    gross_edge_per_unit: Decimal
    fees_per_unit: Decimal
    net_edge_per_unit: Decimal
    quantity: Decimal
    net_profit: Decimal
    notes: tuple[str, ...] = ()
    other_costs_per_unit: Decimal = ZERO  # e.g. capital lock-up and resolution-risk buffer
    gross_profit: Decimal = ZERO
    total_fees: Decimal = ZERO
    other_costs: Decimal = ZERO

    @property
    def group_id(self) -> str:
        """Deterministic package id (usable as ``Signal.group_id``)."""
        return stable_id(
            self.kind.value,
            *(
                f"{leg.instrument_id}:{leg.side.value}:{leg.price}:{leg.quantity}"
                for leg in self.legs
            ),
        )

    @property
    def contract_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(leg.contract_id for leg in self.legs))


@dataclass(frozen=True, slots=True, kw_only=True)
class ContractQuote:
    """A mapped contract with its canonical YES book and fee schedule."""

    mapping: ContractMapping
    book: BookSnapshot
    fee_schedule: FeeSchedule

    @property
    def contract_id(self) -> str:
        return self.mapping.contract_id

    @property
    def venue(self) -> Venue:
        return self.mapping.venue


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuralConfig:
    tolerance: Decimal = ZERO  # required net edge per unit beyond all costs
    walk_levels: bool = False  # False: top-of-book depth only
    max_levels: int | None = None  # per leg, when walking
    max_quantity: Decimal | None = None
    min_status: MappingStatus = MappingStatus.REVIEWED
    asof_ns: int | None = None  # decision time: books received later raise LookaheadError
    max_book_age_ns: int | None = None  # books older than this (vs asof) are skipped

    def __post_init__(self) -> None:
        if self.tolerance < ZERO:
            raise ValueError("tolerance must be non-negative")
        if self.max_levels is not None and self.max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        if self.max_quantity is not None and self.max_quantity <= ZERO:
            raise ValueError("max_quantity must be positive")
        if self.max_book_age_ns is not None and self.asof_ns is None:
            raise ValueError("max_book_age_ns requires asof_ns")


DEFAULT_CONFIG = StructuralConfig()


# --------------------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------------------


def require_status(mappings: Sequence[ContractMapping], min_status: MappingStatus) -> None:
    for m in mappings:
        if not m.review_status.at_least(min_status):
            raise MappingNotApprovedError(
                f"{m.contract_id}: mapping status {m.review_status} < {min_status}; structural "
                "constraints require reviewed mappings"
            )


def book_problems(book: BookSnapshot, config: StructuralConfig) -> list[str]:
    """Reasons a book cannot be used (fail closed). Lookahead is a hard error."""
    problems: list[str] = []
    if config.asof_ns is not None:
        if book.recv_ts_ns > config.asof_ns:
            raise LookaheadError(
                f"{book.instrument_id}: book received at {book.recv_ts_ns} after decision time "
                f"{config.asof_ns}"
            )
        if (
            config.max_book_age_ns is not None
            and config.asof_ns - book.source_or_recv_ts_ns > config.max_book_age_ns
        ):
            problems.append(f"{book.instrument_id}: book is stale")
    if not book.is_valid:
        problems.append(f"{book.instrument_id}: book invalid (crossed/locked/flagged)")
    for lvl in (*book.bids, *book.asks):
        if lvl.price > ONE:
            problems.append(f"{book.instrument_id}: price {lvl.price} outside [0, 1]")
            break
    return problems


# --------------------------------------------------------------------------------------
# Package walker
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class LegSpec:
    """One leg of a package: executable levels best-first in canonical YES prices."""

    contract_id: str
    instrument_id: str
    venue: Venue
    side: Side
    levels: tuple[BookLevel, ...]
    fee_schedule: FeeSchedule
    outcome: Outcome = Outcome.YES


type ExtraCost = Callable[[tuple[Decimal, ...]], Decimal]


def _unit_gross(
    specs: Sequence[LegSpec], prices: tuple[Decimal, ...], constant: Decimal
) -> Decimal:
    total = constant
    for spec, price in zip(specs, prices, strict=True):
        total += price if spec.side is Side.SELL else -price
    return total


def unit_outlay(specs: Sequence[LegSpec], prices: tuple[Decimal, ...]) -> Decimal:
    """Capital per package unit: YES cost for BUY legs, NO cost (1 - p) for SELL legs."""
    return sum(
        (p if s.side is Side.BUY else ONE - p for s, p in zip(specs, prices, strict=True)), ZERO
    )


def walk_package(
    kind: OpportunityKind,
    specs: Sequence[LegSpec],
    *,
    constant_payoff: Decimal,
    config: StructuralConfig,
    extra_cost_per_unit: ExtraCost | None = None,
    notes: Sequence[str] = (),
) -> StructuralOpportunity | None:
    """Size a package against displayed depth and price it exactly; ``None`` if unprofitable.

    Top-of-book only unless ``config.walk_levels``; when walking, a further step is taken only
    while its marginal edge (un-rounded marginal fees, extra costs and tolerance included)
    stays positive. The most profitable prefix of steps that passes the strict exact test
    ``net_profit - tolerance * quantity > 0`` is returned.
    """
    levels = [tuple(lvl for lvl in s.levels if lvl.quantity > ZERO) for s in specs]
    if not specs or any(not lv for lv in levels):
        return None
    idx = [0] * len(specs)
    remaining = [lv[0].quantity for lv in levels]
    steps: list[tuple[Decimal, tuple[Decimal, ...]]] = []
    total = ZERO
    while True:
        prices = tuple(lv[i].price for lv, i in zip(levels, idx, strict=True))
        qty = min(remaining)
        if config.max_quantity is not None:
            qty = min(qty, config.max_quantity - total)
        if qty <= ZERO:
            break
        marginal = _unit_gross(specs, prices, constant_payoff) - config.tolerance
        marginal -= sum(
            (
                s.fee_schedule.fee_per_contract(price=p, role=LiquidityRole.TAKER)
                for s, p in zip(specs, prices, strict=True)
            ),
            ZERO,
        )
        if extra_cost_per_unit is not None:
            marginal -= extra_cost_per_unit(prices)
        if marginal <= ZERO:
            break
        steps.append((qty, prices))
        total += qty
        if not config.walk_levels:
            break
        exhausted = False
        for j in range(len(specs)):
            remaining[j] -= qty
            if remaining[j] == ZERO:
                idx[j] += 1
                limit = len(levels[j]) if config.max_levels is None else config.max_levels
                if idx[j] >= min(limit, len(levels[j])):
                    exhausted = True
                else:
                    remaining[j] = levels[j][idx[j]].quantity
        if exhausted:
            break

    best: StructuralOpportunity | None = None
    for n in range(1, len(steps) + 1):
        candidate = _price_steps(
            kind,
            specs,
            steps[:n],
            constant_payoff=constant_payoff,
            extra_cost_per_unit=extra_cost_per_unit,
            notes=notes,
        )
        if candidate.net_profit - config.tolerance * candidate.quantity <= ZERO:
            continue
        if best is None or candidate.net_profit > best.net_profit:
            best = candidate
    return best


def _price_steps(
    kind: OpportunityKind,
    specs: Sequence[LegSpec],
    steps: Sequence[tuple[Decimal, tuple[Decimal, ...]]],
    *,
    constant_payoff: Decimal,
    extra_cost_per_unit: ExtraCost | None,
    notes: Sequence[str],
) -> StructuralOpportunity:
    quantity = sum((q for q, _ in steps), ZERO)
    gross = sum((q * _unit_gross(specs, prices, constant_payoff) for q, prices in steps), ZERO)
    other = ZERO
    if extra_cost_per_unit is not None:
        other = sum((q * extra_cost_per_unit(prices) for q, prices in steps), ZERO)
    legs: list[Leg] = []
    fees = ZERO
    for j, spec in enumerate(specs):
        per_level: dict[Decimal, Decimal] = {}
        for q, prices in steps:
            per_level[prices[j]] = per_level.get(prices[j], ZERO) + q
        for price, qty in per_level.items():
            fee = spec.fee_schedule.fee(price=price, quantity=qty, role=LiquidityRole.TAKER)
            fees += fee
            legs.append(
                Leg(
                    contract_id=spec.contract_id,
                    instrument_id=spec.instrument_id,
                    venue=spec.venue,
                    side=spec.side,
                    price=price,
                    quantity=qty,
                    fee=fee,
                    outcome=spec.outcome,
                )
            )
    net = gross - fees - other
    return StructuralOpportunity(
        kind=kind,
        legs=tuple(legs),
        gross_edge_per_unit=gross / quantity,
        fees_per_unit=fees / quantity,
        net_edge_per_unit=net / quantity,
        quantity=quantity,
        net_profit=net,
        notes=tuple(notes),
        other_costs_per_unit=other / quantity,
        gross_profit=gross,
        total_fees=fees,
        other_costs=other,
    )


def sort_opportunities(opps: list[StructuralOpportunity]) -> list[StructuralOpportunity]:
    return sorted(opps, key=lambda o: (-o.net_profit, o.kind.value, o.contract_ids))
