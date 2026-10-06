"""H2/H4 strategy: executable structural violations -> multi-leg IOC packages.

On any book change inside a reviewed family (nested strikes on one venue, or an
equivalence-verified cross-venue pair) the detector runs on the observed books; the best
opportunity becomes one IOC order per leg sharing a ``group_id``. Legging risk is real:
legs travel and fill independently through the simulator, and the ledger shows it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from cma.backtest.core import OrderIntent, StrategyContext, StrategyOutput
from cma.domain.enums import TimeInForce
from cma.domain.models import ContractMapping, Fill, MarketEvent, PredictionContract
from cma.domain.time import NS_PER_MS
from cma.models.structural.nested import detect_nested_violations
from cma.models.structural.types import ContractQuote, StructuralConfig, StructuralOpportunity


@dataclass
class StructuralArbStrategy:
    contracts: Sequence[tuple[PredictionContract, ContractMapping]]
    config: StructuralConfig = field(default_factory=StructuralConfig)
    min_interval_ms: int = 250
    max_quantity: Decimal = Decimal(25)
    strategy_id: str = "structural_arb"
    version: str = "1.0"
    packages_sent: int = 0
    _families: dict[str, list[tuple[PredictionContract, ContractMapping]]] = field(
        default_factory=dict
    )
    _family_of_inst: dict[str, str] = field(default_factory=dict)
    _last: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for c, m in self.contracts:
            fam = m.event_family or c.event_id
            self._families.setdefault(fam, []).append((c, m))
            self._family_of_inst[c.outcome_instruments.get("YES", c.contract_id)] = fam

    def _quotes(self, fam: str, ctx: StrategyContext) -> list[ContractQuote]:
        quotes = []
        for c, m in self._families[fam]:
            inst = c.outcome_instruments.get("YES", c.contract_id)
            book = ctx.state.book(inst)
            if book is None or not book.is_valid:
                continue
            quotes.append(
                ContractQuote(
                    mapping=m, book=book, fee_schedule=ctx.core.fee_schedule(c.contract_id)
                )
            )
        return quotes

    def on_observation(self, event: MarketEvent, ctx: StrategyContext) -> list[StrategyOutput]:
        fam = self._family_of_inst.get(event.instrument_id)
        if fam is None or len(self._families[fam]) < 2:
            return []
        now = ctx.now_ns
        if now - self._last.get(fam, -(10**18)) < self.min_interval_ms * NS_PER_MS:
            return []
        if any(ctx.working_orders(c.contract_id) for c, _ in self._families[fam]):
            return []  # a package is still in flight
        opps = detect_nested_violations(self._quotes(fam, ctx), config=self.config)
        if not opps:
            return []
        self._last[fam] = now
        return self.package(opps[0])

    def package(self, opp: StructuralOpportunity) -> list[StrategyOutput]:
        qty = min(opp.quantity, self.max_quantity)
        self.packages_sent += 1
        return [
            OrderIntent(
                venue=leg.venue,
                contract_id=leg.contract_id,
                instrument_id=leg.instrument_id,
                side=leg.side,
                quantity=qty,
                limit_price=leg.price,
                tif=TimeInForce.IOC,
                strategy_id=self.strategy_id,
                group_id=opp.group_id,
            )
            for leg in opp.legs
        ]

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> list[StrategyOutput]:
        return []

    def on_timer(self, ctx: StrategyContext) -> list[StrategyOutput]:
        return []
