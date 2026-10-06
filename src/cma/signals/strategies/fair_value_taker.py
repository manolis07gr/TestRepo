"""H1 strategy: external-market fair value vs stale prediction-market quotes (taker).

For each mapped price contract the strategy computes a semantics-aware fair YES
probability from the latest *observed* reference price (point or trailing-average
settlement, see ``cma.features.fair_value``) and hands it to the signal engine, which
decides whether the executable edge clears fees, slippage and buffers. Positions are held
to settlement by default (fees are paid once).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from cma.backtest.core import StrategyContext, StrategyOutput
from cma.domain.enums import Operator
from cma.domain.models import ContractMapping, Fill, MarketEvent, PredictionContract
from cma.domain.numbers import from_float
from cma.domain.time import NS_PER_MS, NS_PER_S
from cma.features.fair_value import AveragingState, contract_probability, window_seconds
from cma.signals.engine import FairValueEstimate

PROB_QUANTUM = Decimal("0.0001")


@dataclass(frozen=True)
class FairValueTakerConfig:
    reference_instrument: str
    vol_window_s: int = 900
    vol_sample_s: int = 1
    fixed_vol: float | None = None
    vol_multiplier: float = 1.0
    vol_floor: float = 0.10
    vol_cap: float = 3.0
    min_time_to_expiry_s: float = 2.0
    max_time_to_expiry_s: float = 4 * 3600.0
    reeval_min_interval_ms: int = 100
    reeval_on_move_bps: float = 0.5
    max_reference_age_ms: int = 1_000
    uncertainty_bps: Decimal = Decimal(0)
    target_quantity: Decimal = Decimal(10)
    one_order_in_flight: bool = True


@dataclass
class _Tracked:
    contract: PredictionContract
    mapping: ContractMapping
    instrument_id: str
    strikes: tuple[float, ...]
    last_eval_ns: int = 0
    last_spot: float = 0.0


@dataclass
class FairValueTakerStrategy:
    cfg: FairValueTakerConfig
    contracts: Sequence[tuple[PredictionContract, ContractMapping]]
    strategy_id: str = "fv_taker"
    version: str = "1.0"
    estimates_made: int = 0
    vol_recompute_ms: int = 1_000
    _tracked: dict[str, _Tracked] = field(default_factory=dict)
    _vol_cache: tuple[int, float | None] = (-1, None)
    _active_cache: list[str] = field(default_factory=list)
    _active_computed_at: int = 0
    _active_valid_until: int | None = None
    _by_instrument: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for contract, mapping in self.contracts:
            if mapping.operator not in (
                Operator.GT,
                Operator.GE,
                Operator.LT,
                Operator.LE,
                Operator.BETWEEN,
            ):
                continue
            window_seconds(mapping.observation_method)  # validates the method early
            inst = contract.outcome_instruments.get("YES", contract.contract_id)
            self._tracked[contract.contract_id] = _Tracked(
                contract=contract,
                mapping=mapping,
                instrument_id=inst,
                strikes=tuple(float(k) for k in mapping.strikes),
            )
            self._by_instrument[inst] = contract.contract_id

    # ------------------------------------------------------------------ callbacks

    def _active(self, now_ns: int) -> list[str]:
        """Contracts open at ``now_ns`` (cached; recomputed when a boundary is crossed)."""
        if (
            self._active_valid_until is None
            or now_ns >= self._active_valid_until
            or (now_ns < self._active_computed_at)
        ):
            active = []
            next_boundary: int | None = None
            for cid, tr in self._tracked.items():
                o, c = tr.contract.open_ts_ns, tr.contract.close_ts_ns
                if (o is None or o <= now_ns) and (c is None or now_ns < c):
                    active.append(cid)
                for b in (o, c):
                    if (
                        b is not None
                        and b > now_ns
                        and (next_boundary is None or b < next_boundary)
                    ):
                        next_boundary = b
            self._active_cache = active
            self._active_computed_at = now_ns
            self._active_valid_until = next_boundary if next_boundary is not None else 2**62
        return self._active_cache

    def on_observation(self, event: MarketEvent, ctx: StrategyContext) -> list[StrategyOutput]:
        if event.instrument_id == self.cfg.reference_instrument:
            targets: Iterable[str] = self._active(ctx.now_ns)
            forced = False
        else:
            cid = self._by_instrument.get(event.instrument_id)
            if cid is None:
                return []
            targets = (cid,)
            forced = True
        out: list[StrategyOutput] = []
        for cid in targets:
            est = self._evaluate(self._tracked[cid], ctx, forced)
            if est is not None:
                out.append(est)
        return out

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> list[StrategyOutput]:
        return []

    def on_timer(self, ctx: StrategyContext) -> list[StrategyOutput]:
        return []

    # ------------------------------------------------------------------ model

    def sigma(self, ctx: StrategyContext) -> float | None:
        if self.cfg.fixed_vol is not None:
            return self.cfg.fixed_vol * self.cfg.vol_multiplier
        cached_at, cached = self._vol_cache
        if cached_at >= 0 and ctx.now_ns - cached_at < self.vol_recompute_ms * NS_PER_MS:
            return cached
        series = ctx.state.ref.get(self.cfg.reference_instrument)
        if series is None:
            return None
        vol = series.realized_vol(
            ctx.now_ns, self.cfg.vol_window_s * NS_PER_S, self.cfg.vol_sample_s * NS_PER_S
        )
        value = (
            None
            if vol is None
            else min(self.cfg.vol_cap, max(self.cfg.vol_floor, vol * self.cfg.vol_multiplier))
        )
        self._vol_cache = (ctx.now_ns, value)
        return value

    def _evaluate(
        self, tr: _Tracked, ctx: StrategyContext, forced: bool
    ) -> FairValueEstimate | None:
        now = ctx.now_ns
        m = tr.mapping
        c = tr.contract
        if (c.open_ts_ns is not None and now < c.open_ts_ns) or (
            c.close_ts_ns is not None and now >= c.close_ts_ns
        ):
            return None
        tte_s = (m.observation_end_ns - now) / NS_PER_S
        if tte_s < self.cfg.min_time_to_expiry_s or tte_s > self.cfg.max_time_to_expiry_s:
            return None
        series = ctx.state.ref.get(self.cfg.reference_instrument)
        if series is None:
            return None
        latest = series.price_at(now, self.cfg.max_reference_age_ms * NS_PER_MS)
        if latest is None:
            return None
        wm, spot = latest
        if not forced:
            if now - tr.last_eval_ns < self.cfg.reeval_min_interval_ms * NS_PER_MS:
                return None
            if tr.last_spot > 0:
                moved_bps = abs(math.log(spot / tr.last_spot)) * 1e4
                if moved_bps < self.cfg.reeval_on_move_bps:
                    return None
        if self.cfg.one_order_in_flight and ctx.working_orders(tr.contract.contract_id):
            return None
        sigma = self.sigma(ctx)
        if sigma is None:
            return None
        w = window_seconds(m.observation_method)
        averaging = None
        if w and tte_s < w:
            start = m.observation_end_ns - w * NS_PER_S
            integral = series.integral(start, min(now, wm))
            if integral is None:
                return None
            elapsed = (min(now, wm) - start) / NS_PER_S
            averaging = AveragingState(elapsed_s=elapsed, integral=integral)
        p = contract_probability(
            operator=m.operator,
            strikes=tr.strikes,
            spot=spot,
            now_ns=wm,
            observation_end_ns=m.observation_end_ns,
            sigma=sigma,
            observation_method=m.observation_method,
            averaging=averaging,
        )
        tr.last_eval_ns = now
        tr.last_spot = spot
        self.estimates_made += 1
        return FairValueEstimate(
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            venue=tr.contract.venue,
            contract_id=tr.contract.contract_id,
            instrument_id=tr.instrument_id,
            asof_ts_ns=now,
            fair_probability=from_float(p, PROB_QUANTUM),
            feature_watermark_ns=wm,
            uncertainty_bps=self.cfg.uncertainty_bps,
            mapping_version=m.version,
            target_quantity=self.cfg.target_quantity,
            reason=f"spot={spot:.2f} sigma={sigma:.3f} tte={tte_s:.1f}s",
        )
