"""Nested-threshold constraint (scope s.10.4, test T013).

For one observation family (same underlying, observation window/method, timezone, source,
rounding and early-close rule) of "X > K" contracts, ``P(X > K_high) <= P(X > K_low)``.
Executably, the constraint is violated iff

    bid(K_high) - ask(K_low) - fee_low - fee_high - tolerance > 0

i.e. buying YES on the lower strike at its ask and selling YES on the higher strike at its
bid locks in a non-negative settlement (``1{X > K_low} - 1{X > K_high} >= 0``) plus a positive
net credit. "X < K" families are mirrored: the *lower* strike is the subset event.

Every ordered (superset, subset) pair is checked, not only adjacent strikes. Opportunities
for different pairs may share displayed depth; they are alternatives, not additive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from cma.domain.enums import Operator, Side
from cma.domain.models import ContractMapping
from cma.domain.numbers import ZERO
from cma.mapping.equivalence import family_mismatches
from cma.models.structural.types import (
    DEFAULT_CONFIG,
    ContractQuote,
    LegSpec,
    OpportunityKind,
    StructuralConfig,
    StructuralInputError,
    StructuralOpportunity,
    book_problems,
    require_status,
    sort_opportunities,
    walk_package,
)

UPPER_OPERATORS: Final = frozenset({Operator.GT, Operator.GE})
LOWER_OPERATORS: Final = frozenset({Operator.LT, Operator.LE})


def is_subset_event(sub: ContractMapping, sup: ContractMapping) -> bool:
    """True iff ``sub`` paying YES implies ``sup`` pays YES (same family assumed)."""
    if sub.contract_id == sup.contract_id:
        return False
    k_sub, k_sup = sub.strike, sup.strike
    if sub.operator in UPPER_OPERATORS and sup.operator in UPPER_OPERATORS:
        if k_sub != k_sup:
            return k_sub > k_sup
        # same strike: {X > K} is inside {X >= K}; identical operators are the same event
        return sub.operator is sup.operator or (
            sub.operator is Operator.GT and sup.operator is Operator.GE
        )
    if sub.operator in LOWER_OPERATORS and sup.operator in LOWER_OPERATORS:
        if k_sub != k_sup:
            return k_sub < k_sup
        return sub.operator is sup.operator or (
            sub.operator is Operator.LT and sup.operator is Operator.LE
        )
    return False


def _validate_family(quotes: Sequence[ContractQuote], config: StructuralConfig) -> None:
    mappings = [q.mapping for q in quotes]
    require_status(mappings, config.min_status)
    ids = [m.contract_id for m in mappings]
    if len(set(ids)) != len(ids):
        raise StructuralInputError(f"duplicate contracts in nested family: {ids}")
    operators = {m.operator for m in mappings}
    if not (operators <= UPPER_OPERATORS or operators <= LOWER_OPERATORS):
        raise StructuralInputError(
            f"nested family must be all X>K (GT/GE) or all X<K (LT/LE); got {sorted(operators)}"
        )
    mismatches = family_mismatches(mappings)
    if mismatches:
        raise StructuralInputError(
            "contracts do not share one observation family: " + "; ".join(mismatches)
        )
    for q in quotes:
        if q.book.venue is not q.venue:
            raise StructuralInputError(
                f"book {q.book.instrument_id} ({q.book.venue}) does not belong to {q.contract_id}"
            )


def detect_nested_violations(
    quotes: Sequence[ContractQuote], *, config: StructuralConfig | None = None
) -> list[StructuralOpportunity]:
    """All executable nested-threshold violations in a reviewed family, best first."""
    cfg = config if config is not None else DEFAULT_CONFIG
    if len(quotes) < 2:
        return []
    _validate_family(quotes, cfg)
    usable = [q for q in quotes if not book_problems(q.book, cfg)]
    results: list[StructuralOpportunity] = []
    for sup in usable:
        for sub in usable:
            if not is_subset_event(sub.mapping, sup.mapping):
                continue
            best_sub_bid = sub.book.best_bid
            best_sup_ask = sup.book.best_ask
            if best_sub_bid is None or best_sup_ask is None:
                continue
            if best_sub_bid.price <= best_sup_ask.price:
                continue  # consistent at the touch; no executable violation
            specs = (
                LegSpec(
                    contract_id=sup.contract_id,
                    instrument_id=sup.book.instrument_id,
                    venue=sup.venue,
                    side=Side.BUY,
                    levels=sup.book.asks,
                    fee_schedule=sup.fee_schedule,
                ),
                LegSpec(
                    contract_id=sub.contract_id,
                    instrument_id=sub.book.instrument_id,
                    venue=sub.venue,
                    side=Side.SELL,
                    levels=sub.book.bids,
                    fee_schedule=sub.fee_schedule,
                ),
            )
            note = (
                f"nested violation: bid {best_sub_bid.price} for {sub.contract_id} "
                f"({sub.mapping.operator} {sub.mapping.strike}) exceeds ask {best_sup_ask.price} "
                f"for {sup.contract_id} ({sup.mapping.operator} {sup.mapping.strike}); "
                "settlement value of the package >= 0"
            )
            opp = walk_package(
                OpportunityKind.NESTED_THRESHOLD,
                specs,
                constant_payoff=ZERO,
                config=cfg,
                notes=(note,),
            )
            if opp is not None:
                results.append(opp)
    return sort_opportunities(results)
