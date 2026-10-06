"""Candidate promotion gates and the research decision (scope s.24, s.30, T051, T052).

Decisions: REJECT, COLLECT_MORE_DATA, FORWARD_PAPER_CANDIDATE, CONTINUE_PAPER.
Live-money promotion criteria are intentionally NOT defined (scope s.24); the strongest
outcome here is "forward-paper observation complete, eligible for a separate review".
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cma.backtest.experiment import StressGrid
from cma.domain.enums import Decision
from cma.domain.errors import CMAError


@dataclass(frozen=True)
class NetPnlCriterion:
    """Declared *before* the final test is unlocked."""

    latency_ms: int = 250
    cost: str = "base"
    min_net_pnl: float = 0.0
    require_ci_lower_above: float | None = None


@dataclass(frozen=True)
class PromotionPolicy:
    criterion: NetPnlCriterion = NetPnlCriterion()
    stress_latency_ms: int = 1_000
    stress_cost: str = "fees_x1.5"
    max_top_contract_share: float = 0.5
    max_top_family_share: float = 0.6
    max_top_day_share: float = 0.7
    min_positions: int = 30
    min_profitable_neighbor_frac: float = 0.6
    min_forward_paper_days: int = 14


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    kind: str  # "economic" | "sample" | "robustness" | "operational" | "paper"


@dataclass
class DecisionOutcome:
    decision: Decision
    gates: list[GateResult]
    reasons: list[str] = field(default_factory=list)

    def gate(self, name: str) -> GateResult:
        for g in self.gates:
            if g.name == name:
                return g
        raise KeyError(name)


class PromotionError(CMAError):
    """Attempted promotion without satisfying the gates."""


def out_of_sample_gate(
    final_test: StressGrid,
    criterion: NetPnlCriterion,
    ci_lower: float | None = None,
) -> GateResult:
    """T051: untouched test set must be positive under the declared net-P&L criterion."""
    try:
        r = final_test.get(criterion.latency_ms, criterion.cost)
    except KeyError:
        return GateResult(
            "oos_net_pnl",
            False,
            f"final test lacks the declared scenario {criterion.cost}@{criterion.latency_ms}ms",
            "economic",
        )
    net = r.metrics.net_pnl
    ok = net > criterion.min_net_pnl
    detail = f"net P&L {net:.2f} at {criterion.cost}@{criterion.latency_ms}ms"
    if criterion.require_ci_lower_above is not None:
        ok = ok and ci_lower is not None and ci_lower > criterion.require_ci_lower_above
        detail += f"; CI lower {ci_lower}"
    return GateResult("oos_net_pnl", ok, detail, "economic")


def forward_paper_gate(paper_days_completed: float, policy: PromotionPolicy) -> GateResult:
    """T052: the configured forward-paper observation window must be completed."""
    ok = paper_days_completed >= policy.min_forward_paper_days
    return GateResult(
        "forward_paper_period",
        ok,
        f"{paper_days_completed:.1f} of {policy.min_forward_paper_days} paper days completed",
        "paper",
    )


def decide(
    *,
    final_test: StressGrid,
    policy: PromotionPolicy,
    sensitivity_net_pnls: Sequence[float] = (),
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    data_quality_ok: bool = True,
    mappings_approved_paper: bool = False,
    tests_passed: bool = True,
    paper_days_completed: float = 0.0,
    documented_concentration: bool = False,
) -> DecisionOutcome:
    gates: list[GateResult] = []
    crit = policy.criterion
    gates.append(out_of_sample_gate(final_test, crit, ci_lower))
    base = None
    with contextlib.suppress(KeyError):
        base = final_test.get(crit.latency_ms, crit.cost)
    for name, lat, cost in (
        ("stress_latency_viable", policy.stress_latency_ms, crit.cost),
        ("stress_cost_viable", crit.latency_ms, policy.stress_cost),
    ):
        try:
            r = final_test.get(lat, cost)
            gates.append(
                GateResult(
                    name,
                    r.metrics.net_pnl > 0,
                    f"net {r.metrics.net_pnl:.2f} at {cost}@{lat}ms",
                    "economic",
                )
            )
        except KeyError:
            gates.append(GateResult(name, False, f"scenario {cost}@{lat}ms not run", "economic"))
    if base is not None:
        conc = base.metrics.concentration
        shares = [
            (label, float(conc.get(key, math.nan)), limit, conc.get(count_key or ""))
            for label, key, limit, count_key in (
                ("contract", "top_contract_share_of_gains", policy.max_top_contract_share, None),
                ("family", "top_family_share_of_gains", policy.max_top_family_share, "n_families"),
                ("day", "top_day_share_of_gains", policy.max_top_day_share, "n_days"),
            )
        ]
        # NaN = not assessable: no unit of that kind made money (the out-of-sample gate
        # already fails such a strategy) or the sample holds a single family/day
        ok = documented_concentration or all(
            math.isnan(share) or share <= limit for _, share, limit, _ in shares
        )

        def _share(label: str, share: float, count: Any) -> str:
            if not math.isnan(share):
                return f"{label} {share:.0%}"
            if count is not None and int(count) < 2:
                return f"{label} n/a ({int(count)} in sample)"
            return f"{label} n/a (none profitable)"

        detail = ", ".join(_share(label, share, n) for label, share, _, n in shares)
        gates.append(GateResult("profit_concentration", ok, f"top {detail} of gains", "robustness"))
        n = base.metrics.n_positions
        gates.append(
            GateResult(
                "sample_size",
                n >= policy.min_positions,
                f"{n} positions (min {policy.min_positions}); CI [{ci_lower}, {ci_upper}]",
                "sample",
            )
        )
    if sensitivity_net_pnls:
        frac = sum(1 for v in sensitivity_net_pnls if v > 0) / len(sensitivity_net_pnls)
        gates.append(
            GateResult(
                "parameter_stability",
                frac >= policy.min_profitable_neighbor_frac,
                f"{frac:.0%} of {len(sensitivity_net_pnls)} neighbouring settings profitable",
                "robustness",
            )
        )
    gates.append(GateResult("data_quality", data_quality_ok, "data-quality review", "operational"))
    gates.append(
        GateResult(
            "mappings_approved_paper",
            mappings_approved_paper,
            "all traded mappings APPROVED_PAPER" if mappings_approved_paper else "not approved",
            "operational",
        )
    )
    gates.append(GateResult("ci_gates", tests_passed, "mandatory tests / CI", "operational"))
    paper = forward_paper_gate(paper_days_completed, policy)
    gates.append(paper)

    reasons: list[str] = []
    econ_fail = [g for g in gates if g.kind == "economic" and not g.passed]
    sample_fail = [g for g in gates if g.kind == "sample" and not g.passed]
    robust_fail = [g for g in gates if g.kind == "robustness" and not g.passed]
    ops_fail = [g for g in gates if g.kind == "operational" and not g.passed]
    clearly_negative = ci_upper is not None and ci_upper < 0

    if sample_fail and not clearly_negative:
        reasons += [f"{g.name}: {g.detail}" for g in sample_fail]
        reasons += [f"{g.name}: {g.detail}" for g in econ_fail]
        decision = Decision.COLLECT_MORE_DATA
    elif econ_fail or robust_fail:
        reasons += [f"{g.name}: {g.detail}" for g in econ_fail + robust_fail]
        decision = Decision.REJECT
    elif ops_fail:
        reasons += [f"{g.name}: {g.detail}" for g in ops_fail]
        decision = Decision.COLLECT_MORE_DATA
    elif paper_days_completed <= 0:
        decision = Decision.FORWARD_PAPER_CANDIDATE
        reasons.append("all historical gates passed; start forward paper observation")
    else:
        decision = Decision.CONTINUE_PAPER
        if paper.passed:
            reasons.append(
                "paper window complete: eligible for a separately scoped live-money review "
                "(criteria intentionally undefined in v1)"
            )
        else:
            reasons.append(paper.detail)
    return DecisionOutcome(decision=decision, gates=gates, reasons=reasons)


def assert_promotable_to_paper(outcome: DecisionOutcome) -> None:
    """Hard gate used by the paper executor: only historical candidates may start paper."""
    if outcome.decision not in (Decision.FORWARD_PAPER_CANDIDATE, Decision.CONTINUE_PAPER):
        raise PromotionError(f"decision {outcome.decision} cannot start forward paper trading")


def assert_paper_complete(outcome: DecisionOutcome) -> None:
    """T052: promotion beyond paper requires the observation window to be complete."""
    gate = outcome.gate("forward_paper_period")
    if not gate.passed or outcome.decision is not Decision.CONTINUE_PAPER:
        raise PromotionError(f"forward-paper gate not satisfied: {gate.detail}")
