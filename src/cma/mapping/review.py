"""Human review checklist and deterministic mapping validation (scope s.9).

A mapping may leave DRAFT only when (a) a human reviewer affirms every checklist item and
(b) :func:`validate_mapping` finds no problem when the mapping is compared against the
contract's own listing metadata. Machine/LLM actors can propose but never review or approve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from typing import Any, ClassVar, Final

from cma.domain.enums import Operator
from cma.domain.errors import CMAError
from cma.domain.models import ContractMapping, PredictionContract
from cma.domain.time import NS_PER_MIN
from cma.mapping.semantics import (
    THRESHOLD_OPERATORS,
    TWO_POINT_OPERATORS,
    decimal_from_metadata,
    fmt_ns,
    is_unspecified,
    is_valid_iana_timezone,
    mapping_family_key,
    observation_method_spec,
)

# Default tolerance between the mapped observation instant and the contract's own close /
# resolve timestamp. Deliberately well below one hour so a DST mistake cannot pass.
DEFAULT_OBSERVATION_TOLERANCE_NS: Final = 5 * NS_PER_MIN

# Actor-id prefixes that identify machine proposers (parsers, LLM assistants, bots).
MACHINE_ACTOR_PREFIXES: Final = ("parser:", "llm:", "machine:", "bot:", "auto:", "model:", "agent:")

# Kalshi market ``strike_type`` -> (operator, which strike fields define it).
KALSHI_STRIKE_TYPES: Final[Mapping[str, tuple[Operator, str]]] = {
    "greater": (Operator.GT, "floor"),
    "greater_or_equal": (Operator.GE, "floor"),
    "less": (Operator.LT, "cap"),
    "less_or_equal": (Operator.LE, "cap"),
    "between": (Operator.BETWEEN, "both"),
}
# Kalshi strike types whose payoff cannot be validated deterministically.
NON_DETERMINISTIC_STRIKE_TYPES: Final = frozenset({"functional", "custom", "structured"})


class MappingReviewError(CMAError, ValueError):
    """A review/approval transition was refused (checklist, validation or workflow rule)."""


def is_machine_actor(actor: str) -> bool:
    lowered = actor.strip().lower()
    return any(lowered.startswith(prefix) for prefix in MACHINE_ACTOR_PREFIXES)


def require_actor(actor: str, role: str, *, human: bool) -> str:
    cleaned = actor.strip()
    if not cleaned:
        raise MappingReviewError(f"{role} must be a non-empty actor id")
    if human and is_machine_actor(cleaned):
        raise MappingReviewError(
            f"{role} {cleaned!r} is a machine actor; machine/LLM proposals must be reviewed and "
            "approved by humans"
        )
    return cleaned


def same_actor(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewChecklist:
    """Every item must be affirmed by the human reviewer (scope s.9)."""

    strike_definition: bool = False
    observation_window: bool = False
    timezone: bool = False
    settlement_source: bool = False
    rounding: bool = False
    early_close_rule: bool = False
    outcome_semantics: bool = False
    evidence: str = ""  # free text: what was checked (rules text version, URL, date)

    ITEMS: ClassVar[tuple[str, ...]] = (
        "strike_definition",
        "observation_window",
        "timezone",
        "settlement_source",
        "rounding",
        "early_close_rule",
        "outcome_semantics",
    )

    @classmethod
    def all_affirmed(cls, evidence: str = "") -> ReviewChecklist:
        return cls(**dict.fromkeys(cls.ITEMS, True), evidence=evidence)

    def missing(self) -> tuple[str, ...]:
        return tuple(item for item in self.ITEMS if getattr(self, item) is not True)

    @property
    def complete(self) -> bool:
        return not self.missing()

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewChecklist:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown checklist fields {sorted(unknown)}")
        values: dict[str, Any] = {}
        for item in cls.ITEMS:
            value = data.get(item, False)
            if not isinstance(value, bool):
                raise ValueError(f"checklist item {item!r} must be a boolean, got {value!r}")
            values[item] = value
        evidence = data.get("evidence", "")
        return cls(**values, evidence=str(evidence))


def _window_problems(mapping: ContractMapping) -> list[str]:
    spec = observation_method_spec(mapping.observation_method)
    if spec is None:
        return [
            f"observation_method {mapping.observation_method!r} is not a known method "
            "(POINT, AVG_<N>S_BEFORE, CANDLE_CLOSE_<N><unit>, CANDLE_OPEN_CLOSE_<N><unit>, "
            "TWAP_<N>S)"
        ]
    start, end = mapping.observation_start_ns, mapping.observation_end_ns
    problems: list[str] = []
    if mapping.operator in TWO_POINT_OPERATORS:
        if not spec.two_point:
            problems.append(
                f"operator {mapping.operator} compares two observations but method "
                f"{spec.method} observes a single value"
            )
        if start is None or start >= end:
            problems.append(f"operator {mapping.operator} needs observation_start < end")
    elif spec.two_point:
        problems.append(f"method {spec.method} is a two-point comparison; operator must be UP/DOWN")
    if spec.window_ns == 0:
        if start is not None and start != end:
            problems.append("POINT observation must not declare a window start")
    elif spec.window_ns is not None:
        if start is None:
            problems.append(f"method {spec.method} requires observation_start")
        elif end - start != spec.window_ns:
            problems.append(
                f"window [{fmt_ns(start)}, {fmt_ns(end)}) is {end - start} ns but method "
                f"{spec.method} implies {spec.window_ns} ns"
            )
    return problems


def _metadata_strike_problems(mapping: ContractMapping, contract: PredictionContract) -> list[str]:
    md = contract.settlement_metadata
    problems: list[str] = []
    if "strike_type" in md:
        strike_type = str(md["strike_type"]).strip().lower()
        if strike_type in NON_DETERMINISTIC_STRIKE_TYPES:
            return [f"venue strike_type {strike_type!r} cannot be validated deterministically"]
        if strike_type not in KALSHI_STRIKE_TYPES:
            return [f"unknown venue strike_type {strike_type!r}"]
        operator, which = KALSHI_STRIKE_TYPES[strike_type]
        floor = decimal_from_metadata(md.get("floor_strike"))
        cap = decimal_from_metadata(md.get("cap_strike"))
        expected: tuple[Decimal | None, ...]
        if which == "floor":
            expected = (floor,)
        elif which == "cap":
            expected = (cap,)
        else:
            expected = (floor, cap)
        if any(s is None for s in expected):
            return [f"strike_type {strike_type!r} but floor/cap strike metadata missing"]
        if mapping.operator is not operator:
            problems.append(
                f"operator {mapping.operator} disagrees with venue strike_type {strike_type!r} "
                f"({operator})"
            )
        if mapping.strikes != expected:
            problems.append(
                f"strikes {[str(s) for s in mapping.strikes]} disagree with venue metadata "
                f"{[str(s) for s in expected]}"
            )
        return problems
    for key in ("strike", "group_item_title"):
        if key in md:
            value = decimal_from_metadata(
                str(md[key]).replace("$", "").replace("↑", "").replace("↓", "").strip()
            )
            if value is None:
                return [f"venue metadata {key}={md[key]!r} is not a number"]
            if len(mapping.strikes) != 1 or mapping.strikes[0] != value:
                return [
                    f"strikes {[str(s) for s in mapping.strikes]} disagree with venue "
                    f"metadata {key}={value}"
                ]
            return []
    return problems


def validate_mapping(
    mapping: ContractMapping,
    contract: PredictionContract | None,
    *,
    observation_tolerance_ns: int = DEFAULT_OBSERVATION_TOLERANCE_NS,
) -> list[str]:
    """Deterministic checks that must all pass before a mapping can become REVIEWED.

    Returns the list of problems (empty = valid). Never raises for a semantic problem.
    """
    problems: list[str] = []
    if contract is None:
        return ["no contract listing metadata available for deterministic validation"]
    if contract.contract_id != mapping.contract_id or contract.venue is not mapping.venue:
        problems.append(
            f"mapping {mapping.venue}:{mapping.contract_id} does not match contract "
            f"{contract.venue}:{contract.contract_id}"
        )
    if not is_valid_iana_timezone(mapping.timezone):
        problems.append(f"timezone {mapping.timezone!r} is not a valid IANA region name")
    if not mapping.underlyings or any(is_unspecified(u) for u in mapping.underlyings):
        problems.append("underlyings must be specified")
    for name in ("resolution_source", "observation_method", "rounding_rule", "early_close_rule"):
        if is_unspecified(str(getattr(mapping, name))):
            problems.append(f"{name} is unspecified; determine it from the rules before review")
    if mapping.operator in THRESHOLD_OPERATORS and any(s <= 0 for s in mapping.strikes):
        problems.append("threshold strikes must be positive")
    problems.extend(_window_problems(mapping))

    anchors = [ts for ts in (contract.close_ts_ns, contract.resolve_ts_ns) if ts is not None]
    if not anchors:
        problems.append("contract has no close/resolve timestamp to anchor the observation")
    else:
        nearest = min(anchors, key=lambda ts: abs(ts - mapping.observation_end_ns))
        if abs(nearest - mapping.observation_end_ns) > observation_tolerance_ns:
            problems.append(
                f"observation_end {fmt_ns(mapping.observation_end_ns)} is not within "
                f"{observation_tolerance_ns} ns of the contract close/resolve time "
                f"(close={fmt_ns(contract.close_ts_ns)}, resolve={fmt_ns(contract.resolve_ts_ns)})"
            )
    if contract.open_ts_ns is not None and mapping.observation_end_ns < contract.open_ts_ns:
        problems.append("observation_end precedes the contract open time")
    problems.extend(_metadata_strike_problems(mapping, contract))

    expected_family = mapping_family_key(mapping)
    if mapping.event_family != expected_family:
        problems.append(f"event_family {mapping.event_family!r} != canonical {expected_family!r}")
    return problems
