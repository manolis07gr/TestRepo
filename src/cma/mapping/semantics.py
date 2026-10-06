"""Shared mapping semantics: family keys, observation-method windows and (de)serialisation.

Everything here is deterministic and venue-agnostic. Observation instants are UTC
nanoseconds; ``observation_start_ns``/``observation_end_ns`` describe the half-open window
``[start, end)`` that determines the settlement value:

* ``POINT``                    - value at ``end``; ``start`` is ``None``.
* ``AVG_<N>S_BEFORE``          - simple average over ``[end - N s, end)`` (Kalshi crypto: N=60).
* ``CANDLE_CLOSE_<N><S|M|H|D>``- close of the candle ``[start, start + N units)``; ``end`` is
  the candle's closing boundary (Polymarket "1 minute candle for 12:00 ET": 12:00-12:01 ET).
* ``CANDLE_OPEN_CLOSE_<N><unit>`` - two-point comparison of a candle's close with its open
  (hourly Up/Down markets): ``[start, end)`` is the candle.
* ``TWAP_<N>S``                - two-point comparison of N-second TWAPs taken at ``start`` and
  ``end`` (Chainlink-resolved 5m/15m/4h Up/Down markets).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cma.domain.enums import MappingStatus, Operator, Venue
from cma.domain.models import ContractMapping
from cma.domain.time import (
    NS_PER_DAY,
    NS_PER_HOUR,
    NS_PER_MIN,
    NS_PER_S,
    iso_from_ns,
    ns_from_iso8601,
)

# Values that mean "not determined"; they never satisfy review or equivalence checks.
UNSPECIFIED_VALUES: Final = frozenset({"", "UNKNOWN", "UNSPECIFIED", "TBD", "N/A"})

THRESHOLD_OPERATORS: Final = frozenset(
    {Operator.GT, Operator.GE, Operator.LT, Operator.LE, Operator.BETWEEN}
)
TWO_POINT_OPERATORS: Final = frozenset({Operator.UP, Operator.DOWN})

_UNIT_NS: Final = {"S": NS_PER_S, "M": NS_PER_MIN, "H": NS_PER_HOUR, "D": NS_PER_DAY}
_AVG_RE: Final = re.compile(r"AVG_(?P<n>\d+)S_BEFORE")
_TWAP_RE: Final = re.compile(r"TWAP_(?P<n>\d+)S")
_CANDLE_RE: Final = re.compile(r"CANDLE_(?P<kind>CLOSE|OPEN_CLOSE)_(?P<n>\d+)(?P<unit>[SMHD])")


def is_unspecified(value: str) -> bool:
    return value.strip().upper() in UNSPECIFIED_VALUES


def is_valid_iana_timezone(name: str) -> bool:
    """True for loadable IANA region names (``America/New_York``) and ``UTC``.

    Legacy abbreviations such as ``EST``/``EST5EDT`` are rejected even though zoneinfo can
    load some of them: they are fixed offsets or ambiguous and a classic source of DST bugs.
    """
    if name != "UTC" and "/" not in name:
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class ObservationMethodSpec:
    """Validation facts about an ``observation_method`` string."""

    method: str
    window_ns: int | None  # required end - start; 0 = point; None = set by the contract
    two_point: bool  # compares a start value with an end value (UP/DOWN)


def observation_method_spec(method: str) -> ObservationMethodSpec | None:
    """Parse a known observation method; ``None`` for unknown methods (fail closed)."""
    if method == "POINT":
        return ObservationMethodSpec(method, 0, two_point=False)
    if m := _AVG_RE.fullmatch(method):
        return ObservationMethodSpec(method, int(m["n"]) * NS_PER_S, two_point=False)
    if m := _TWAP_RE.fullmatch(method):
        return ObservationMethodSpec(method, None, two_point=True)
    if m := _CANDLE_RE.fullmatch(method):
        window = int(m["n"]) * _UNIT_NS[m["unit"]]
        return ObservationMethodSpec(method, window, two_point=m["kind"] == "OPEN_CLOSE")
    return None


def canonical_event_family(
    underlyings: Sequence[str], observation_method: str, observation_end_ns: int
) -> str:
    """Risk-aggregation key: same variable, same observation method, same instant."""
    return f"{'+'.join(underlyings)}|{observation_method}|{iso_from_ns(observation_end_ns)}"


def mapping_family_key(mapping: ContractMapping) -> str:
    return canonical_event_family(
        mapping.underlyings, mapping.observation_method, mapping.observation_end_ns
    )


def semantic_key(mapping: ContractMapping) -> tuple[object, ...]:
    """Every field that changes what the contract pays. A change => new mapping version."""
    return (
        mapping.venue,
        mapping.contract_id,
        mapping.underlyings,
        mapping.operator,
        tuple(s.normalize() for s in mapping.strikes),
        mapping.observation_start_ns,
        mapping.observation_end_ns,
        mapping.observation_method,
        mapping.timezone,
        mapping.resolution_source,
        mapping.rounding_rule,
        mapping.early_close_rule,
        mapping.outcome_semantics,
        mapping.event_family,
    )


def fmt_ns(ns: int | None) -> str:
    return "None" if ns is None else iso_from_ns(ns)


# --------------------------------------------------------------------------------------
# (De)serialisation. Decimals travel as strings, instants as ISO-8601 with 9 fractional
# digits (exact round trip), enums by value.
# --------------------------------------------------------------------------------------


def decimal_from_metadata(value: object) -> Decimal | None:
    """Venue metadata number -> Decimal. Floats use their shortest repr (explicit boundary)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, int):
            result = Decimal(value)
        elif isinstance(value, float):
            result = Decimal(repr(value))
        elif isinstance(value, str):
            result = Decimal(value.strip().replace(",", ""))
        else:
            return None
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _strict_decimal(value: object, what: str) -> Decimal:
    if isinstance(value, bool | float):
        raise ValueError(f"{what} must be a quoted decimal string, not {value!r}")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            result = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"{what}: not a decimal {value!r}") from exc
        if result.is_finite():
            return result
    raise ValueError(f"{what}: not a finite decimal {value!r}")


def _opt_iso(ns: int | None) -> str | None:
    return None if ns is None else iso_from_ns(ns)


def _opt_ns(value: object, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return ns_from_iso8601(value)
    raise ValueError(f"{what} must be an ISO-8601 string or null, got {value!r}")


def _req_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"mapping field {key!r} must be a string, got {value!r}")
    return value


def mapping_to_dict(mapping: ContractMapping, *, include_status: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "venue": mapping.venue.value,
        "contract_id": mapping.contract_id,
        "underlyings": list(mapping.underlyings),
        "operator": mapping.operator.value,
        "strikes": [str(s) for s in mapping.strikes],
        "observation_start": _opt_iso(mapping.observation_start_ns),
        "observation_end": iso_from_ns(mapping.observation_end_ns),
        "observation_method": mapping.observation_method,
        "timezone": mapping.timezone,
        "resolution_source": mapping.resolution_source,
        "rounding_rule": mapping.rounding_rule,
        "early_close_rule": mapping.early_close_rule,
        "outcome_semantics": mapping.outcome_semantics,
        "event_family": mapping.event_family,
        "notes": mapping.notes,
    }
    if include_status:
        data["version"] = mapping.version
        data["review_status"] = mapping.review_status.value
        data["reviewer"] = mapping.reviewer
        data["reviewed_at"] = _opt_iso(mapping.reviewed_at_ns)
    return data


def mapping_from_dict(
    data: Mapping[str, Any],
    *,
    version: int | None = None,
    review_status: MappingStatus | None = None,
    reviewer: str | None = None,
    reviewed_at_ns: int | None = None,
) -> ContractMapping:
    """Inverse of :func:`mapping_to_dict`; keyword overrides win over stored status fields."""
    underlyings = data.get("underlyings")
    strikes = data.get("strikes")
    if not isinstance(underlyings, list) or not all(isinstance(u, str) for u in underlyings):
        raise ValueError(f"underlyings must be a list of strings, got {underlyings!r}")
    if not isinstance(strikes, list):
        raise ValueError(f"strikes must be a list, got {strikes!r}")
    end_ns = _opt_ns(data.get("observation_end"), "observation_end")
    if end_ns is None:
        raise ValueError("observation_end is required")
    stored_version = data.get("version", 1)
    if not isinstance(stored_version, int):
        raise ValueError(f"version must be an integer, got {stored_version!r}")
    stored_reviewer = data.get("reviewer")
    return ContractMapping(
        venue=Venue(_req_str(data, "venue")),
        contract_id=_req_str(data, "contract_id"),
        underlyings=tuple(underlyings),
        operator=Operator(_req_str(data, "operator")),
        strikes=tuple(_strict_decimal(s, "strike") for s in strikes),
        observation_start_ns=_opt_ns(data.get("observation_start"), "observation_start"),
        observation_end_ns=end_ns,
        observation_method=_req_str(data, "observation_method"),
        timezone=_req_str(data, "timezone"),
        resolution_source=_req_str(data, "resolution_source"),
        rounding_rule=str(data.get("rounding_rule", "NONE")),
        early_close_rule=str(data.get("early_close_rule", "NONE")),
        outcome_semantics=str(data.get("outcome_semantics", "")),
        event_family=str(data.get("event_family", "")),
        version=version if version is not None else stored_version,
        review_status=(
            review_status
            if review_status is not None
            else MappingStatus(str(data.get("review_status", MappingStatus.DRAFT.value)))
        ),
        reviewer=(
            reviewer
            if reviewer is not None
            else (stored_reviewer if isinstance(stored_reviewer, str) else None)
        ),
        reviewed_at_ns=(
            reviewed_at_ns
            if reviewed_at_ns is not None
            else _opt_ns(data.get("reviewed_at"), "reviewed_at")
        ),
        notes=str(data.get("notes", "")),
    )
