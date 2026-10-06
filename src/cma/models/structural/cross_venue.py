"""Cross-venue arbitrage between equivalence-verified contracts (s.10.3, hypothesis H2).

Only pairs for which :func:`cma.mapping.equivalence.check_equivalence` reports
``equivalent=True`` are considered; anything else returns no opportunity.

* SAME (YES_a == YES_b): buy YES on one venue at its ask and buy NO on the other at
  ``1 - bid`` (canonical: sell YES at the bid) - and the reverse. Settlement value 0.
* COMPLEMENT (YES_a == NO_b): buy YES on both (settles to exactly 1), or sell YES on both
  (owes exactly 1).

Costs: both venues' taker fees, a capital lock-up charge for the settlement-timing
difference ``annual_rate x dt x capital`` (capital = per-unit cash outlay: YES cost for
buys, NO cost for sells) and a per-unit resolution-risk buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cma.domain.enums import Side
from cma.domain.numbers import ONE, ZERO
from cma.mapping.equivalence import EquivalenceRelation, check_equivalence
from cma.models.structural.types import (
    DEFAULT_CONFIG,
    YEAR_NS,
    ContractQuote,
    LegSpec,
    OpportunityKind,
    StructuralConfig,
    StructuralInputError,
    StructuralOpportunity,
    book_problems,
    sort_opportunities,
    unit_outlay,
    walk_package,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossVenueCosts:
    settlement_ts_a_ns: int  # expected payout time of contract a
    settlement_ts_b_ns: int  # expected payout time of contract b
    annual_capital_rate: Decimal = ZERO  # e.g. Decimal("0.05")
    resolution_risk_buffer: Decimal = ZERO  # $ per $1-face unit, for rule/oracle risk
    year_ns: int = YEAR_NS

    def __post_init__(self) -> None:
        if self.annual_capital_rate < ZERO or self.resolution_risk_buffer < ZERO:
            raise ValueError("cost parameters must be non-negative")
        if self.year_ns <= 0:
            raise ValueError("year_ns must be positive")

    @property
    def lockup_years(self) -> Decimal:
        return Decimal(abs(self.settlement_ts_a_ns - self.settlement_ts_b_ns)) / Decimal(
            self.year_ns
        )


def _spec(quote: ContractQuote, side: Side) -> LegSpec:
    return LegSpec(
        contract_id=quote.contract_id,
        instrument_id=quote.book.instrument_id,
        venue=quote.venue,
        side=side,
        levels=quote.book.asks if side is Side.BUY else quote.book.bids,
        fee_schedule=quote.fee_schedule,
    )


def detect_cross_venue_arbitrage(
    a: ContractQuote,
    b: ContractQuote,
    costs: CrossVenueCosts,
    *,
    config: StructuralConfig | None = None,
) -> list[StructuralOpportunity]:
    """Executable arbitrage between two equivalence-verified contracts (best first)."""
    cfg = config if config is not None else DEFAULT_CONFIG
    if a.contract_id == b.contract_id:
        raise StructuralInputError("cross-venue comparison needs two different contracts")
    for q in (a, b):
        if q.book.venue is not q.venue:
            raise StructuralInputError(
                f"book {q.book.instrument_id} does not belong to {q.contract_id}"
            )
    eq = check_equivalence(a.mapping, b.mapping, min_status=cfg.min_status)
    if not eq.equivalent:
        return []
    if book_problems(a.book, cfg) or book_problems(b.book, cfg):
        return []

    lockup = costs.lockup_years
    rate = costs.annual_capital_rate
    buffer = costs.resolution_risk_buffer

    if eq.relation is EquivalenceRelation.SAME:
        packages = (
            ((_spec(a, Side.BUY), _spec(b, Side.SELL)), ZERO, "buy YES on a, buy NO on b"),
            ((_spec(b, Side.BUY), _spec(a, Side.SELL)), ZERO, "buy YES on b, buy NO on a"),
        )
    else:  # COMPLEMENT: YES on a pays exactly when NO on b pays
        packages = (
            ((_spec(a, Side.BUY), _spec(b, Side.BUY)), ONE, "buy YES on a and YES on b"),
            ((_spec(a, Side.SELL), _spec(b, Side.SELL)), -ONE, "buy NO on a and NO on b"),
        )

    results: list[StructuralOpportunity] = []
    for specs, constant, description in packages:
        if any(not s.levels for s in specs):
            continue

        def extra(prices: tuple[Decimal, ...], specs: tuple[LegSpec, ...] = specs) -> Decimal:
            return buffer + rate * lockup * unit_outlay(specs, prices)

        notes = (
            f"{eq.relation.value} pair {a.contract_id} ({a.venue}) / {b.contract_id} ({b.venue}): "
            f"{description}",
            f"settlement timing difference {lockup} years at annual rate {rate}; "
            f"resolution-risk buffer {buffer} per unit",
            "collateral/transfer costs between venues (e.g. USD vs USDC) not modelled",
        )
        opp = walk_package(
            OpportunityKind.CROSS_VENUE,
            specs,
            constant_payoff=constant,
            config=cfg,
            extra_cost_per_unit=extra,
            notes=notes,
        )
        if opp is not None:
            results.append(opp)
    return sort_opportunities(results)
