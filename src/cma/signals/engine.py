"""Signal engine: turns fair-value estimates into gated, explainable signals.

Gate order (each suppression is persisted with a reason code):
look-ahead watermark -> mapping status -> data freshness -> clock drift -> book validity
-> liquidity -> net-edge threshold. Signals are emitted iff
``net_edge_bps >= min_net_edge_bps`` (equality emits; T016) and expire after the TTL.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from cma.config import SignalConfig
from cma.domain.enums import ExecutionMode, LiquidityRole, ReasonCode, Venue
from cma.domain.errors import LookaheadError
from cma.domain.fees import FeeSchedule
from cma.domain.models import BookSnapshot, Signal, stable_id
from cma.domain.numbers import BPS, ONE, ZERO
from cma.domain.time import NS_PER_MS
from cma.signals.edge import EdgeBreakdown, best_edge


@dataclass(frozen=True, slots=True)
class FairValueEstimate:
    strategy_id: str
    strategy_version: str
    venue: Venue
    contract_id: str
    instrument_id: str  # canonical YES book to trade against
    asof_ts_ns: int
    fair_probability: Decimal
    feature_watermark_ns: int
    confidence: float = 1.0
    uncertainty_bps: Decimal = ZERO
    mapping_version: int = 0
    target_quantity: Decimal | None = None
    group_id: str | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Suppression:
    ts_ns: int
    strategy_id: str
    contract_id: str
    reason: ReasonCode
    detail: str = ""


@dataclass(frozen=True, slots=True)
class GateInputs:
    data_fresh: bool = True
    freshness_detail: str = ""
    clock_ok: bool = True


DEFAULT_GATES = GateInputs()


@dataclass
class SignalEngine:
    config: SignalConfig
    mode: ExecutionMode
    can_trade: Callable[[str, ExecutionMode], bool]
    fee_resolver: Callable[[str], FeeSchedule]
    strict_lookahead: bool = True
    suppressions: Counter[ReasonCode] = field(default_factory=Counter)
    recent_suppressions: list[Suppression] = field(default_factory=list)
    max_recent: int = 1_000
    signals_emitted: int = 0
    last_edge: EdgeBreakdown | None = None

    def _suppress(
        self, est: FairValueEstimate, ts_ns: int, reason: ReasonCode, detail: str = ""
    ) -> Suppression:
        s = Suppression(ts_ns, est.strategy_id, est.contract_id, reason, detail)
        self.suppressions[reason] += 1
        if len(self.recent_suppressions) < self.max_recent:
            self.recent_suppressions.append(s)
        return s

    def evaluate(
        self,
        est: FairValueEstimate,
        book: BookSnapshot | None,
        *,
        decision_ts_ns: int,
        gates: GateInputs = DEFAULT_GATES,
    ) -> Signal | Suppression:
        self.last_edge = None
        if est.feature_watermark_ns > decision_ts_ns or est.asof_ts_ns > decision_ts_ns:
            if self.strict_lookahead:
                raise LookaheadError(
                    f"{est.strategy_id}/{est.contract_id}: watermark "
                    f"{est.feature_watermark_ns} > decision {decision_ts_ns}"
                )
            return self._suppress(est, decision_ts_ns, ReasonCode.LOOKAHEAD)
        if self.config.require_reviewed_mapping and not self.can_trade(est.contract_id, self.mode):
            return self._suppress(est, decision_ts_ns, ReasonCode.MAPPING_NOT_APPROVED)
        if not gates.data_fresh:
            return self._suppress(
                est, decision_ts_ns, ReasonCode.STALE_DATA, gates.freshness_detail
            )
        if not gates.clock_ok:
            return self._suppress(est, decision_ts_ns, ReasonCode.CLOCK_DRIFT)
        if book is None or not book.is_valid:
            detail = "" if book is None else ",".join(sorted(book.quality_flags))
            return self._suppress(est, decision_ts_ns, ReasonCode.BOOK_INVALID, detail)

        target = est.target_quantity or self.config.default_order_qty
        edge = best_edge(
            fair=est.fair_probability,
            book=book,
            target_quantity=target,
            fee_schedule=self.fee_resolver(est.contract_id),
            adverse_selection_bps=self.config.adverse_selection_buffer_bps,
            uncertainty_bps=self.config.uncertainty_buffer_bps + est.uncertainty_bps,
            min_marginal_net_edge_bps=self.config.min_net_edge_bps,
            max_levels=self.config.max_levels_to_walk,
        )
        if edge is None:
            return self._suppress(est, decision_ts_ns, ReasonCode.NO_LIQUIDITY)
        self.last_edge = edge
        if edge.net_edge_bps < self.config.min_net_edge_bps:
            return self._suppress(
                est,
                decision_ts_ns,
                ReasonCode.BELOW_THRESHOLD,
                f"net {edge.net_edge_bps:.1f}bps < {self.config.min_net_edge_bps}",
            )
        return self.make_signal(est, edge, decision_ts_ns)

    def touch_precheck(
        self,
        est: FairValueEstimate,
        best_bid: tuple[Decimal, Decimal] | None,
        best_ask: tuple[Decimal, Decimal] | None,
        decision_ts_ns: int,
    ) -> Suppression | None:
        """Cheap upper bound on net edge from the touch alone.

        Deeper levels can only add slippage, so if neither side clears the threshold at
        the touch, the full evaluation would suppress with BELOW_THRESHOLD as well. Gates
        (look-ahead, mapping) still run first so suppression reasons stay truthful.
        """
        if est.feature_watermark_ns > decision_ts_ns or est.asof_ts_ns > decision_ts_ns:
            return None  # let evaluate() raise/suppress LOOKAHEAD
        if self.config.require_reviewed_mapping and not self.can_trade(est.contract_id, self.mode):
            return None
        fee = self.fee_resolver(est.contract_id)
        buffers = (
            self.config.adverse_selection_buffer_bps
            + self.config.uncertainty_buffer_bps
            + est.uncertainty_bps
        ) * BPS
        threshold = self.config.min_net_edge_bps * BPS
        fair = est.fair_probability
        best = None
        if best_ask is not None:
            px = best_ask[0]
            net = fair - px - fee.fee_per_contract(price=px, role=LiquidityRole.TAKER) - buffers
            best = net
        if best_bid is not None:
            px = best_bid[0]
            net = px - fair - fee.fee_per_contract(price=px, role=LiquidityRole.TAKER) - buffers
            best = net if best is None else max(best, net)
        if best is None:
            return None
        if best < threshold:
            return self._suppress(est, decision_ts_ns, ReasonCode.BELOW_THRESHOLD, "touch")
        return None

    def make_signal(
        self, est: FairValueEstimate, edge: EdgeBreakdown, decision_ts_ns: int
    ) -> Signal:
        signal_id = stable_id(
            "sig",
            est.strategy_id,
            est.strategy_version,
            est.contract_id,
            decision_ts_ns,
            edge.side.value,
            est.fair_probability,
        )
        self.signals_emitted += 1
        return Signal(
            signal_id=signal_id,
            strategy_id=est.strategy_id,
            strategy_version=est.strategy_version,
            asof_ts_ns=decision_ts_ns,
            venue=est.venue,
            contract_id=est.contract_id,
            instrument_id=est.instrument_id,
            side=edge.side,
            fair_probability=min(ONE, max(ZERO, est.fair_probability)),
            executable_price=edge.best_price,
            gross_edge=edge.gross_edge,
            expected_cost=edge.expected_cost,
            net_edge=edge.net_edge,
            confidence=est.confidence,
            expiry_ts_ns=decision_ts_ns + self.config.ttl_ms * NS_PER_MS,
            quantity=edge.quantity,
            limit_price=edge.limit_price,
            mapping_version=est.mapping_version,
            feature_watermark_ns=est.feature_watermark_ns,
            group_id=est.group_id,
            reason=est.reason,
        )


def signal_threshold_met(net_edge: Decimal, min_net_edge_bps: Decimal) -> bool:
    """Deterministic boundary: equality passes."""
    return net_edge * Decimal(10_000) >= min_net_edge_bps
