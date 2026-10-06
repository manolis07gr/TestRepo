"""Mutually-exclusive & exhaustive outcome sets (scope s.10.4, test T014).

Exactly one contract of such a set pays 1, so (canonical YES books):

* under-round: ``sum(asks) + sum(fees) < 1 - tolerance``  ->  buy every YES (pays exactly 1);
* over-round:  ``sum(bids) - sum(fees) > 1 + tolerance``  ->  sell every YES / buy every NO
  (owes exactly 1 at settlement).

Executable quantity is the minimum displayed depth across legs (optionally walking levels
while still profitable). Exhaustiveness cannot be inferred from titles: the set must be
*declared* mutually exclusive AND exhaustive by a human (e.g. a Kalshi event's range buckets
plus both tails), otherwise the detector refuses to run. Void/cancellation of a leg breaks
the guaranteed payoff; that settlement risk is not priced here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cma.domain.enums import Side
from cma.domain.numbers import ONE
from cma.mapping.equivalence import family_mismatches
from cma.mapping.review import is_machine_actor
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ExhaustiveSet:
    """Human declaration that ``contract_ids`` partition the outcome space."""

    set_id: str
    contract_ids: frozenset[str]
    mutually_exclusive: bool
    exhaustive: bool
    declared_by: str
    basis: str = ""  # e.g. "KXBTC event buckets + both tails per contract terms"


def _validate(
    quotes: Sequence[ContractQuote], declaration: ExhaustiveSet, config: StructuralConfig
) -> None:
    if not (declaration.mutually_exclusive and declaration.exhaustive):
        raise StructuralInputError(
            f"set {declaration.set_id!r} is not declared mutually exclusive AND exhaustive; "
            "refusing to apply the sum-to-one constraint"
        )
    declarer = declaration.declared_by.strip()
    if not declarer or is_machine_actor(declarer):
        raise StructuralInputError("exhaustive-set declarations must come from a human reviewer")
    ids = [q.contract_id for q in quotes]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise StructuralInputError(f"need >= 2 distinct contracts, got {ids}")
    if set(ids) != set(declaration.contract_ids):
        raise StructuralInputError(
            f"quotes {sorted(ids)} do not match declared set {sorted(declaration.contract_ids)}"
        )
    mappings = [q.mapping for q in quotes]
    require_status(mappings, config.min_status)
    mismatches = family_mismatches(mappings)
    if mismatches:
        raise StructuralInputError(
            "set members do not share one observation family: " + "; ".join(mismatches)
        )
    for q in quotes:
        if q.book.venue is not q.venue:
            raise StructuralInputError(
                f"book {q.book.instrument_id} does not belong to {q.contract_id}"
            )


def detect_exhaustive_mispricing(
    quotes: Sequence[ContractQuote],
    declaration: ExhaustiveSet,
    *,
    config: StructuralConfig | None = None,
) -> list[StructuralOpportunity]:
    """Under-/over-round opportunities of a declared exhaustive set (best first)."""
    cfg = config if config is not None else DEFAULT_CONFIG
    _validate(quotes, declaration, cfg)
    if any(book_problems(q.book, cfg) for q in quotes):
        return []  # every leg must be executable; fail closed
    results: list[StructuralOpportunity] = []
    for side, constant, kind in (
        (Side.BUY, ONE, OpportunityKind.EXHAUSTIVE_UNDER_ROUND),
        (Side.SELL, -ONE, OpportunityKind.EXHAUSTIVE_OVER_ROUND),
    ):
        specs = tuple(
            LegSpec(
                contract_id=q.contract_id,
                instrument_id=q.book.instrument_id,
                venue=q.venue,
                side=side,
                levels=q.book.asks if side is Side.BUY else q.book.bids,
                fee_schedule=q.fee_schedule,
            )
            for q in sorted(quotes, key=lambda q: q.contract_id)
        )
        if any(not s.levels for s in specs):
            continue
        what = (
            "buy every YES (pays exactly 1)"
            if side is Side.BUY
            else ("sell every YES / buy every NO (owes exactly 1)")
        )
        note = (
            f"{kind.value} on declared set {declaration.set_id!r} "
            f"(declared by {declaration.declared_by}): {what}; void/cancel risk not priced"
        )
        opp = walk_package(kind, specs, constant_payoff=constant, config=cfg, notes=(note,))
        if opp is not None:
            results.append(opp)
    return sort_opportunities(results)
