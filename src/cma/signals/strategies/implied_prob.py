"""H3 strategy: option-implied (risk-neutral) probability vs prediction-market quotes.

The option surface is supplied by a provider (built from Deribit chain snapshots in live
or recorded data). The implied probability is a market benchmark, not an objective
real-world probability (scope s.10.2), so every estimate carries an explicit uncertainty
buffer and expiry alignment must be permitted by an explicit rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from cma.backtest.core import StrategyContext, StrategyOutput
from cma.domain.errors import ExpiryMismatchError
from cma.domain.models import ContractMapping, Fill, MarketEvent, PredictionContract
from cma.domain.time import NS_PER_MS
from cma.models.implied_probability.surface import (
    ExpiryAlignment,
    OptionSurface,
    implied_probability_for_mapping,
)
from cma.signals.engine import FairValueEstimate

SurfaceProvider = Callable[[int], tuple[OptionSurface, int] | None]  # -> (surface, asof_ns)


@dataclass
class ImpliedProbabilityStrategy:
    contracts: Sequence[tuple[PredictionContract, ContractMapping]]
    surface_provider: SurfaceProvider
    alignment: ExpiryAlignment | None = None
    uncertainty_bps: Decimal = Decimal(150)
    max_surface_age_ms: int = 60_000
    min_interval_ms: int = 1_000
    target_quantity: Decimal = Decimal(10)
    strategy_id: str = "implied_prob"
    version: str = "1.0"
    skipped_expiry: int = 0
    _last: dict[str, int] = field(default_factory=dict)
    _by_inst: dict[str, tuple[PredictionContract, ContractMapping]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for c, m in self.contracts:
            self._by_inst[c.outcome_instruments.get("YES", c.contract_id)] = (c, m)

    def on_observation(self, event: MarketEvent, ctx: StrategyContext) -> list[StrategyOutput]:
        pair = self._by_inst.get(event.instrument_id)
        if pair is None:
            return []
        contract, mapping = pair
        now = ctx.now_ns
        if now - self._last.get(contract.contract_id, -(10**18)) < self.min_interval_ms * NS_PER_MS:
            return []
        got = self.surface_provider(now)
        if got is None:
            return []
        surface, asof = got
        if asof > now or now - asof > self.max_surface_age_ms * NS_PER_MS:
            return []
        try:
            ip = implied_probability_for_mapping(mapping, surface, rule=self.alignment)
        except ExpiryMismatchError:
            self.skipped_expiry += 1  # no silent expiry substitution (T042)
            return []
        self._last[contract.contract_id] = now
        return [
            FairValueEstimate(
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                venue=contract.venue,
                contract_id=contract.contract_id,
                instrument_id=event.instrument_id,
                asof_ts_ns=now,
                fair_probability=ip.value,
                feature_watermark_ns=asof,
                uncertainty_bps=self.uncertainty_bps,
                mapping_version=mapping.version,
                target_quantity=self.target_quantity,
                reason=f"implied {ip.method} expiries={ip.expiries_used}",
            )
        ]

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> list[StrategyOutput]:
        return []

    def on_timer(self, ctx: StrategyContext) -> list[StrategyOutput]:
        return []
