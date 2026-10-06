"""YES/NO complementarity on venues with separate YES and NO token books (scope s.10.4).

One YES token plus one NO token of the same contract always settle to exactly 1, so:

* buy both:  ``YES ask + NO ask + fees < 1 - tolerance``;
* sell both: ``YES bid + NO bid - fees > 1 + tolerance`` (requires inventory or minting a
  complete set, e.g. a Polymarket split - noted on the opportunity).

Inputs are the two token books in their *own* token prices. Legs are reported in canonical
YES terms (buying NO at ``q`` is a canonical SELL at ``1 - q`` with ``outcome=NO``) and fees
use the canonical YES price, per the shared ``FeeSchedule`` convention.
"""

from __future__ import annotations

from cma.domain.enums import Outcome, Side
from cma.domain.fees import FeeSchedule
from cma.domain.models import BookLevel, BookSnapshot, ContractMapping
from cma.domain.numbers import ONE, ZERO
from cma.models.structural.types import (
    DEFAULT_CONFIG,
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


def _complement_levels(levels: tuple[BookLevel, ...]) -> tuple[BookLevel, ...]:
    """NO-token levels -> canonical YES-price levels (order preserved: best stays first)."""
    return tuple(BookLevel(ONE - lvl.price, lvl.quantity) for lvl in levels)


def detect_complement_mispricing(
    mapping: ContractMapping,
    yes_book: BookSnapshot,
    no_book: BookSnapshot,
    *,
    yes_fee_schedule: FeeSchedule,
    no_fee_schedule: FeeSchedule | None = None,
    config: StructuralConfig | None = None,
) -> list[StructuralOpportunity]:
    """Buy-both / sell-both mispricing between a contract's YES and NO token books."""
    cfg = config if config is not None else DEFAULT_CONFIG
    require_status([mapping], cfg.min_status)
    for book in (yes_book, no_book):
        if book.venue is not mapping.venue:
            raise StructuralInputError(
                f"book {book.instrument_id} ({book.venue}) is not on {mapping.venue}"
            )
    if yes_book.instrument_id == no_book.instrument_id:
        raise StructuralInputError("YES and NO books must be different token instruments")
    if book_problems(yes_book, cfg) or book_problems(no_book, cfg):
        return []
    no_fees = no_fee_schedule if no_fee_schedule is not None else yes_fee_schedule

    def spec(
        book: BookSnapshot, side: Side, levels: tuple[BookLevel, ...], outcome: Outcome
    ) -> LegSpec:
        return LegSpec(
            contract_id=mapping.contract_id,
            instrument_id=book.instrument_id,
            venue=mapping.venue,
            side=side,
            levels=levels,
            fee_schedule=yes_fee_schedule if outcome is Outcome.YES else no_fees,
            outcome=outcome,
        )

    packages = (
        (
            OpportunityKind.COMPLEMENT_BUY_BOTH,
            (
                spec(yes_book, Side.BUY, yes_book.asks, Outcome.YES),
                spec(no_book, Side.SELL, _complement_levels(no_book.asks), Outcome.NO),
            ),
            "buy YES and NO tokens: the pair settles to exactly 1",
        ),
        (
            OpportunityKind.COMPLEMENT_SELL_BOTH,
            (
                spec(yes_book, Side.SELL, yes_book.bids, Outcome.YES),
                spec(no_book, Side.BUY, _complement_levels(no_book.bids), Outcome.NO),
            ),
            "sell YES and NO tokens: needs inventory or a minted complete set (owes exactly 1)",
        ),
    )
    results: list[StructuralOpportunity] = []
    for kind, specs, note in packages:
        if any(not s.levels for s in specs):
            continue
        opp = walk_package(kind, specs, constant_payoff=ZERO, config=cfg, notes=(note,))
        if opp is not None:
            results.append(opp)
    return sort_opportunities(results)
