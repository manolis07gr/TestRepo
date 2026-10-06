"""Cross-contract semantic equivalence (scope s.9/s.10.3, hypothesis H2, test T043).

Two contracts are comparable only when *every* resolution-relevant field matches exactly:
underlying(s), operator (or exact complement at the same strike), strikes, observation
window in UTC, observation method, timezone, resolution source, rounding rule and
early-close rule - and both mappings have been reviewed. Titles are never compared.

``relation`` describes the semantic relation (SAME, COMPLEMENT or NONE) irrespective of
review status; ``equivalent`` additionally requires both mappings to be >= REVIEWED.
Consumers must gate on ``equivalent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from cma.domain.enums import MappingStatus, Operator
from cma.domain.models import ContractMapping
from cma.mapping.semantics import fmt_ns, is_unspecified


class EquivalenceRelation(StrEnum):
    SAME = "SAME"  # YES(a) pays iff YES(b) pays
    COMPLEMENT = "COMPLEMENT"  # YES(a) pays iff NO(b) pays
    NONE = "NONE"


# Exact complements at an identical strike: X > K  <=>  not (X <= K), X >= K <=> not (X < K).
EXACT_COMPLEMENT: Final = {
    Operator.GT: Operator.LE,
    Operator.LE: Operator.GT,
    Operator.GE: Operator.LT,
    Operator.LT: Operator.GE,
}

# Fields that must be *specified* (not UNKNOWN/UNSPECIFIED) on both sides to match.
_MUST_BE_SPECIFIED: Final = (
    "observation_method",
    "timezone",
    "resolution_source",
    "rounding_rule",
    "early_close_rule",
)


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    relation: EquivalenceRelation
    equivalent: bool
    reasons: list[str] = field(default_factory=list)
    mismatched_fields: tuple[str, ...] = ()


def _operator_relation(a: Operator, b: Operator) -> EquivalenceRelation:
    if a is b:
        return EquivalenceRelation.SAME
    if EXACT_COMPLEMENT.get(a) is b:
        return EquivalenceRelation.COMPLEMENT
    return EquivalenceRelation.NONE


def check_equivalence(
    a: ContractMapping,
    b: ContractMapping,
    *,
    min_status: MappingStatus = MappingStatus.REVIEWED,
) -> EquivalenceResult:
    """Compare two mappings field by field; every mismatch is listed in ``reasons``."""
    reasons: list[str] = []
    fields_: list[str] = []

    def mismatch(name: str, detail: str) -> None:
        reasons.append(f"{name}: {detail}")
        if name not in fields_:
            fields_.append(name)

    relation = _operator_relation(a.operator, b.operator)
    if relation is EquivalenceRelation.NONE:
        detail = f"{a.operator} vs {b.operator} are neither identical nor exact complements"
        if {a.operator, b.operator} == {Operator.UP, Operator.DOWN}:
            detail += " (UP/DOWN tie handling makes them inexact complements)"
        mismatch("operator", detail)

    comparisons: tuple[tuple[str, object, object, str, str], ...] = (
        ("underlyings", a.underlyings, b.underlyings, str(a.underlyings), str(b.underlyings)),
        (
            "strikes",
            tuple(s.normalize() for s in a.strikes),
            tuple(s.normalize() for s in b.strikes),
            str([str(s) for s in a.strikes]),
            str([str(s) for s in b.strikes]),
        ),
        (
            "observation_start",
            a.observation_start_ns,
            b.observation_start_ns,
            fmt_ns(a.observation_start_ns),
            fmt_ns(b.observation_start_ns),
        ),
        (
            "observation_end",
            a.observation_end_ns,
            b.observation_end_ns,
            fmt_ns(a.observation_end_ns),
            fmt_ns(b.observation_end_ns),
        ),
        (
            "observation_method",
            a.observation_method,
            b.observation_method,
            a.observation_method,
            b.observation_method,
        ),
        ("timezone", a.timezone, b.timezone, a.timezone, b.timezone),
        (
            "resolution_source",
            a.resolution_source,
            b.resolution_source,
            a.resolution_source,
            b.resolution_source,
        ),
        ("rounding_rule", a.rounding_rule, b.rounding_rule, a.rounding_rule, b.rounding_rule),
        (
            "early_close_rule",
            a.early_close_rule,
            b.early_close_rule,
            a.early_close_rule,
            b.early_close_rule,
        ),
    )
    for name, va, vb, sa, sb in comparisons:
        if va != vb:
            mismatch(name, f"{sa} != {sb}")

    for name in _MUST_BE_SPECIFIED:
        for label, m in (("a", a), ("b", b)):
            if is_unspecified(str(getattr(m, name))):
                mismatch(name, f"unspecified on {label} ({m.contract_id}); cannot be compared")
    if not a.underlyings or not b.underlyings:
        mismatch("underlyings", "empty underlying list")

    semantic_ok = not fields_
    final_relation = relation if semantic_ok else EquivalenceRelation.NONE

    for label, m in (("a", a), ("b", b)):
        if not m.review_status.at_least(min_status):
            mismatch(
                "review_status",
                f"{label} ({m.contract_id}) is {m.review_status}; requires >= {min_status}",
            )
    equivalent = (
        semantic_ok
        and final_relation is not EquivalenceRelation.NONE
        and ("review_status" not in fields_)
    )
    return EquivalenceResult(
        relation=final_relation,
        equivalent=equivalent,
        reasons=reasons,
        mismatched_fields=tuple(fields_),
    )


def family_mismatches(mappings: list[ContractMapping] | tuple[ContractMapping, ...]) -> list[str]:
    """Reasons why ``mappings`` do not observe the same variable at the same time/source.

    Used by structural detectors: nested/exhaustive constraints are only valid within one
    observation family. Operators and strikes are intentionally not compared.
    """
    if not mappings:
        return []
    ref = mappings[0]
    reasons: list[str] = []
    for m in mappings[1:]:
        for name in (
            "underlyings",
            "observation_start_ns",
            "observation_end_ns",
            "observation_method",
            "timezone",
            "resolution_source",
            "rounding_rule",
            "early_close_rule",
        ):
            va, vb = getattr(ref, name), getattr(m, name)
            if va != vb:
                reasons.append(f"{name}: {ref.contract_id}={va!r} vs {m.contract_id}={vb!r}")
    for m in mappings:
        for name in _MUST_BE_SPECIFIED:
            if is_unspecified(str(getattr(m, name))):
                reasons.append(f"{name}: unspecified on {m.contract_id}")
    return reasons
