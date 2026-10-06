"""Deterministic DRAFT mapping proposals from venue contract metadata (scope s.9).

Parsers never approve anything: a successful parse is a DRAFT proposal that must pass
human review (:meth:`cma.mapping.registry.MappingRegistry.review`). Whenever the payoff
cannot be established deterministically - unknown series/template, missing or conflicting
metadata, a source/time/comparison not stated in the rules text, a local time that does
not exist or is ambiguous across DST - the parser returns ``ParseResult(mapping=None,
reason=...)`` instead of guessing (scope s.26: stop and flag ambiguity).

Supported inputs
----------------
* Kalshi crypto series listed in ``config/mappings/series_semantics.yaml`` with
  ``strike_type`` in {greater, greater_or_equal, less, less_or_equal, between} and
  ``floor_strike``/``cap_strike`` metadata; the event ticker ``<SERIES>-<YY><MON><DD><HH>``
  gives the stated ET time.
* Polymarket "Bitcoin above ___ on <date>?" / "Will the price of Bitcoin be above $X on
  <date>?" (Binance 1-minute candle close at the time stated in the rules), hourly
  "Bitcoin Up or Down - <date>, <h>PM ET" (Binance 1-hour candle) and slug-identified
  5m/15m/4h Up/Down markets (Chainlink 60-second TWAPs).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

from cma.domain.enums import MappingStatus, Operator, Venue
from cma.domain.errors import AmbiguousLocalTimeError, NonexistentLocalTimeError
from cma.domain.models import ContractMapping, PredictionContract
from cma.domain.time import (
    NS_PER_HOUR,
    NS_PER_MIN,
    NS_PER_S,
    datetime_from_ns,
    iso_from_ns,
    local_to_utc_ns,
    ns_from_iso8601,
    utc_ns_to_local,
)
from cma.mapping.review import KALSHI_STRIKE_TYPES, NON_DETERMINISTIC_STRIKE_TYPES
from cma.mapping.semantics import canonical_event_family, decimal_from_metadata

DEFAULT_SERIES_SEMANTICS_PATH: Final = (
    Path(__file__).resolve().parents[3] / "config" / "mappings" / "series_semantics.yaml"
)
SERIES_SEMANTICS_SCHEMA: Final = "cma.series_semantics/v1"
KALSHI_PARSER_ID: Final = "parser:kalshi-series/v1"
POLYMARKET_PARSER_ID: Final = "parser:polymarket-rules/v1"
# Venue close time vs parsed observation instant: must agree within this tolerance.
DEFAULT_CLOSE_TOLERANCE_NS: Final = 5 * NS_PER_MIN
# A title date without a year must lie within this distance of the contract's close time.
YEAR_INFERENCE_WINDOW_DAYS: Final = 3

_MONTHS: Final = {
    name: i
    for i, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}
_ASSET_CODES: Final = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH"}
# Settlement sources other than the template's; their presence makes the rules ambiguous.
_OTHER_SOURCES: Final = (
    "binance",
    "coinbase",
    "kraken",
    "bitstamp",
    "chainlink",
    "cf benchmarks",
    "pyth",
    "okx",
    "bybit",
)
_CANDLE_PATTERNS: Final = {
    "1M": re.compile(r"\b(?:1[- ]?minute|one[- ]minute|1m)\b"),
    "1H": re.compile(r"\b(?:1[- ]?hour|one[- ]hour|1h)\b"),
}
_TZ_NAMES: Final = {
    "et": "America/New_York",
    "edt": "America/New_York",
    "est": "America/New_York",
    "utc": "UTC",
    "gmt": "UTC",
}
_SLUG_INTERVALS_NS: Final = {"5m": 5 * NS_PER_MIN, "15m": 15 * NS_PER_MIN, "4h": 4 * NS_PER_HOUR}

_KALSHI_TICKER_RE: Final = re.compile(
    r"^(?P<series>[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hh>\d{2})"
    r"(?:-(?P<suffix>[A-Z0-9.]+))?$"
)
_ASSET_RE: Final = r"(?P<asset>bitcoin|btc|ethereum|eth)"
_DATE_RE: Final = r"(?P<month>[a-z]+)\.? (?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,? (?P<year>\d{4}))?"
_STRIKE_RE: Final = r"\$?(?P<strike>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<k>k)?"
_THRESHOLD_TITLE_RES: Final = (
    re.compile(
        rf"^will the price of {_ASSET_RE} be (?P<dir>above|below) {_STRIKE_RE} on {_DATE_RE}\?$"
    ),
    re.compile(
        rf"^{_ASSET_RE} (?P<dir>above|below) (?:{_STRIKE_RE}|(?P<blank>_+)) on {_DATE_RE}\?$"
    ),
)
_UPDOWN_HOURLY_RE: Final = re.compile(
    rf"^{_ASSET_RE} up or down - {_DATE_RE},? (?P<hour>\d{{1,2}})(?::(?P<minute>\d{{2}}))? ?"
    r"(?P<ampm>am|pm) (?P<tz>et|utc)$"
)
_UPDOWN_RANGE_RE: Final = re.compile(
    rf"^{_ASSET_RE} up or down - {_DATE_RE},? (?P<hour>\d{{1,2}})(?::(?P<minute>\d{{2}}))? ?"
    r"(?P<ampm>am|pm)? ?- ?\d{1,2}(?::\d{2})? ?(?:am|pm) (?P<tz>et|utc)$"
)
_UPDOWN_DAILY_RE: Final = re.compile(rf"^{_ASSET_RE} up or down on {_DATE_RE}\??$")
_UPDOWN_SLUG_RE: Final = re.compile(
    r"^(?P<coin>btc|eth)-updown-(?P<interval>5m|15m|4h)-(?P<start>\d{9,11})$"
)
_RULE_TIME_RE: Final = re.compile(
    r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?\s*"
    r"(?:in the\s+)?(?P<tz>et|edt|est|utc|gmt)\b"
)
_GE_PHRASES: Final = (
    "higher than or equal to",
    "greater than or equal to",
    "equal to or higher than",
    "equal to or greater than",
    "at or above",
)
_GT_PHRASES: Final = ("higher than", "greater than", "above")
_LE_PHRASES: Final = (
    "lower than or equal to",
    "less than or equal to",
    "equal to or lower than",
    "equal to or less than",
    "at or below",
)
_LT_PHRASES: Final = ("lower than", "less than", "below")


# --------------------------------------------------------------------------------------
# Results and semantics tables
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParseResult:
    """A DRAFT proposal, or ``mapping=None`` with the ambiguity ``reason``."""

    mapping: ContractMapping | None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    proposed_by: str = ""

    @property
    def ok(self) -> bool:
        return self.mapping is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class KalshiSeriesSemantics:
    series: str
    underlying: str
    resolution_source: str
    observation_method: str
    observation_window_ns: int
    timezone: str
    rounding_rule: str
    early_close_rule: str
    rules_keywords: tuple[str, ...]
    auto_parse: bool
    verified: bool
    source_url: str
    verification_note: str
    note: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AssetSource:
    underlying: str
    resolution_source: str
    pair_keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PolymarketTemplate:
    template_id: str
    market_kind: str  # THRESHOLD | UPDOWN_HOURLY | UPDOWN_SLUG
    source_keyword: str
    candle: str | None
    observation_method: str
    timezone: str
    rounding_rule: str
    early_close_rule: str
    rule_effective_from_ns: int | None
    assets: Mapping[str, AssetSource]
    verified: bool
    source_url: str
    verification_note: str
    note: str


@dataclass(frozen=True, slots=True)
class SeriesSemantics:
    kalshi: Mapping[str, KalshiSeriesSemantics]
    polymarket: Mapping[str, PolymarketTemplate]

    def template(self, market_kind: str) -> PolymarketTemplate | None:
        for tmpl in self.polymarket.values():
            if tmpl.market_kind == market_kind:
                return tmpl
        return None


def _str(row: Mapping[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value.strip()


def _bool(row: Mapping[str, Any], key: str, where: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{where}: {key} must be true/false")
    return value


def _window_from_method(method: str, where: str) -> int:
    m = re.fullmatch(r"AVG_(\d+)S_BEFORE", method)
    if m is None:
        raise ValueError(f"{where}: Kalshi observation_method must be AVG_<N>S_BEFORE")
    return int(m.group(1)) * NS_PER_S


def parse_series_semantics(data: Mapping[str, Any]) -> SeriesSemantics:
    if data.get("schema") != SERIES_SEMANTICS_SCHEMA:
        raise ValueError(f"series semantics schema must be {SERIES_SEMANTICS_SCHEMA!r}")
    kalshi: dict[str, KalshiSeriesSemantics] = {}
    for series, row in (data.get("kalshi_series") or {}).items():
        where = f"kalshi_series.{series}"
        if not isinstance(row, Mapping):
            raise ValueError(f"{where} must be a mapping")
        method = _str(row, "observation_method", where)
        keywords = row.get("rules_keywords") or []
        if not isinstance(keywords, list):
            raise ValueError(f"{where}: rules_keywords must be a list")
        kalshi[str(series)] = KalshiSeriesSemantics(
            series=str(series),
            underlying=_str(row, "underlying", where),
            resolution_source=_str(row, "resolution_source", where),
            observation_method=method,
            observation_window_ns=_window_from_method(method, where),
            timezone=_str(row, "timezone", where),
            rounding_rule=_str(row, "rounding_rule", where),
            early_close_rule=_str(row, "early_close_rule", where),
            rules_keywords=tuple(str(k) for k in keywords),
            auto_parse=_bool(row, "auto_parse", where),
            verified=_bool(row, "verified", where),
            source_url=str(row.get("source_url") or ""),
            verification_note=str(row.get("verification_note") or ""),
            note=str(row.get("note") or ""),
        )
    templates: dict[str, PolymarketTemplate] = {}
    for template_id, row in (data.get("polymarket_templates") or {}).items():
        where = f"polymarket_templates.{template_id}"
        if not isinstance(row, Mapping):
            raise ValueError(f"{where} must be a mapping")
        assets: dict[str, AssetSource] = {}
        for asset, spec in (row.get("assets") or {}).items():
            if not isinstance(spec, Mapping):
                raise ValueError(f"{where}.assets.{asset} must be a mapping")
            keywords = spec.get("pair_keywords") or []
            assets[str(asset)] = AssetSource(
                underlying=_str(spec, "underlying", where),
                resolution_source=_str(spec, "resolution_source", where),
                pair_keywords=tuple(str(k).lower() for k in keywords),
            )
        effective = row.get("rule_effective_from")
        candle = row.get("candle")
        templates[str(template_id)] = PolymarketTemplate(
            template_id=str(template_id),
            market_kind=_str(row, "market_kind", where),
            source_keyword=_str(row, "source_keyword", where).lower(),
            candle=str(candle) if candle is not None else None,
            observation_method=_str(row, "observation_method", where),
            timezone=_str(row, "timezone", where),
            rounding_rule=_str(row, "rounding_rule", where),
            early_close_rule=_str(row, "early_close_rule", where),
            rule_effective_from_ns=ns_from_iso8601(str(effective)) if effective else None,
            assets=MappingProxyType(assets),
            verified=_bool(row, "verified", where),
            source_url=str(row.get("source_url") or ""),
            verification_note=str(row.get("verification_note") or ""),
            note=str(row.get("note") or ""),
        )
    return SeriesSemantics(kalshi=MappingProxyType(kalshi), polymarket=MappingProxyType(templates))


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime_ns: int) -> SeriesSemantics:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")
    return parse_series_semantics(data)


def load_series_semantics(path: Path | None = None) -> SeriesSemantics:
    target = (path or DEFAULT_SERIES_SEMANTICS_PATH).resolve()
    return _load_cached(str(target), target.stat().st_mtime_ns)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


class _Ambiguous(Exception):
    """Internal: carries the reason a contract cannot be mapped deterministically."""


def _norm(text: str) -> str:
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        text = text.replace(dash, "-")
    text = text.replace("\xa0", " ").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def _local_ns(day: date, hour: int, minute: int, tz: str) -> int:
    try:
        return local_to_utc_ns(datetime.combine(day, time(hour, minute)), tz)
    except (NonexistentLocalTimeError, AmbiguousLocalTimeError) as exc:
        raise _Ambiguous(f"stated time is not a unique instant: {exc}") from exc


def _hour24(hour: int, minute: int | None, ampm: str | None) -> tuple[int, int]:
    minute_v = 0 if minute is None else minute
    if minute_v > 59:
        raise _Ambiguous(f"invalid minute {minute_v}")
    if ampm:
        if not 1 <= hour <= 12:
            raise _Ambiguous(f"invalid 12-hour time {hour} {ampm}")
        return hour % 12 + (12 if ampm.startswith("p") else 0), minute_v
    if minute is None:
        raise _Ambiguous(f"hour {hour} without minutes or AM/PM is ambiguous")
    if hour in {0, 12} or 13 <= hour <= 23:
        return hour, minute_v
    raise _Ambiguous(f"{hour}:{minute:02d} without AM/PM could be morning or evening")


def _check_tz_abbreviation(abbr: str, instant_ns: int, tz: str) -> None:
    """'EDT'/'EST' are only accepted when they match the DST state on that date."""
    if abbr not in ("edt", "est"):
        return
    offset = utc_ns_to_local(instant_ns, tz).utcoffset()
    expected = timedelta(hours=-4) if abbr == "edt" else timedelta(hours=-5)
    if offset != expected:
        raise _Ambiguous(f"rules say {abbr.upper()} but {tz} is not on that offset at that date")


def _resolve_date(
    month_name: str, day_text: str, year_text: str | None, contract: PredictionContract
) -> date:
    month = _MONTHS.get(month_name.lower().rstrip("."))
    if month is None:
        raise _Ambiguous(f"unknown month {month_name!r}")
    day = int(day_text)
    refs = [ts for ts in (contract.close_ts_ns, contract.resolve_ts_ns) if ts is not None]
    ref_date = datetime_from_ns(refs[0]).date() if refs else None
    if year_text is not None:
        try:
            resolved = date(int(year_text), month, day)
        except ValueError as exc:
            raise _Ambiguous(f"invalid date: {exc}") from exc
        if ref_date is not None and abs((resolved - ref_date).days) > YEAR_INFERENCE_WINDOW_DAYS:
            raise _Ambiguous(f"title date {resolved} is far from the close date {ref_date}")
        return resolved
    if ref_date is None:
        raise _Ambiguous("title has no year and the contract has no close/resolve time")
    candidates = []
    for year in (ref_date.year - 1, ref_date.year, ref_date.year + 1):
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            continue
    if not candidates:
        raise _Ambiguous(f"invalid date {month_name} {day}")
    best = min(candidates, key=lambda d: abs((d - ref_date).days))
    if abs((best - ref_date).days) > YEAR_INFERENCE_WINDOW_DAYS:
        raise _Ambiguous(f"title date {month_name} {day} is far from the close date {ref_date}")
    return best


def _parse_strike(number: str, k_suffix: str | None) -> Decimal:
    value = decimal_from_metadata(number.replace(",", ""))
    if value is None or value <= 0:
        raise _Ambiguous(f"invalid strike {number!r}")
    return value * 1000 if k_suffix else value


def _metadata_strike(contract: PredictionContract) -> Decimal:
    raw = contract.settlement_metadata.get("group_item_title")
    if raw is None:
        raise _Ambiguous(
            "title strike is blank and settlement_metadata.group_item_title is missing"
        )
    matches = re.findall(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k)?", _norm(str(raw)))
    if len(matches) != 1:
        raise _Ambiguous(f"group_item_title {raw!r} does not contain exactly one number")
    number, k_suffix = matches[0]
    return _parse_strike(number, k_suffix or None)


def _require_source(rules: str, tmpl: PolymarketTemplate, asset: AssetSource) -> None:
    if not rules:
        raise _Ambiguous("rules text is empty; settlement source cannot be established")
    if tmpl.source_keyword not in rules:
        raise _Ambiguous(f"rules text does not name the {tmpl.source_keyword} source")
    if not any(kw in rules for kw in asset.pair_keywords):
        raise _Ambiguous(f"rules text does not name the pair ({', '.join(asset.pair_keywords)})")
    others = [s for s in _OTHER_SOURCES if s != tmpl.source_keyword and s in rules]
    if others:
        raise _Ambiguous(f"rules text also names other sources {others}; ambiguous")
    if tmpl.candle is not None:
        if not _CANDLE_PATTERNS[tmpl.candle].search(rules):
            raise _Ambiguous(f"rules text does not state a {tmpl.candle} candle")
        clashing = [
            c for c, pat in _CANDLE_PATTERNS.items() if c != tmpl.candle and pat.search(rules)
        ]
        if clashing:
            raise _Ambiguous(f"rules text mentions conflicting candle intervals {clashing}")


def _rule_time(rules: str) -> tuple[int, int, str, frozenset[str]]:
    """Exactly one distinct (hour, minute, timezone) statement in the rules text.

    Returns ``(hour24, minute, IANA zone, abbreviations used)``.
    """
    distinct: set[tuple[int, int, str]] = set()
    abbrs: set[str] = set()
    for m in _RULE_TIME_RE.finditer(rules):
        minute = int(m["minute"]) if m["minute"] else None
        ampm = m["ampm"].replace(".", "") if m["ampm"] else None
        hour, minute_v = _hour24(int(m["hour"]), minute, ampm)
        distinct.add((hour, minute_v, _TZ_NAMES[m["tz"]]))
        abbrs.add(m["tz"])
    if len(distinct) != 1:
        raise _Ambiguous(
            "rules text must state exactly one observation time with a timezone; found "
            f"{sorted(distinct) if distinct else 'none'}"
        )
    hour, minute_v, tz = next(iter(distinct))
    return hour, minute_v, tz, frozenset(abbrs)


def _threshold_operator(rules: str, direction: str) -> Operator:
    if direction == "above":
        if any(p in rules for p in _GE_PHRASES):
            return Operator.GE
        if any(p in rules for p in _GT_PHRASES):
            return Operator.GT
    else:
        if any(p in rules for p in _LE_PHRASES):
            return Operator.LE
        if any(p in rules for p in _LT_PHRASES):
            return Operator.LT
    raise _Ambiguous(f"rules text does not state the {direction!r} comparison")


def _updown_tie_rule(rules: str) -> str:
    """Tie handling of an Up/Down comparison, from the rules text (never assumed)."""
    if any(p in rules for p in ("greater than or equal to", "higher than or equal to")):
        return "TIES_RESOLVE_YES"
    if any(p in rules for p in ("greater than", "higher than")):
        return "TIES_RESOLVE_NO"
    raise _Ambiguous("rules text does not state how the end value compares with the start")


def _require_yes(contract: PredictionContract, expected: str) -> None:
    if _norm(contract.yes_semantics) != expected:
        raise _Ambiguous(
            f"YES outcome is {contract.yes_semantics!r}; expected {expected!r} orientation"
        )


def _check_close(contract: PredictionContract, end_ns: int, tolerance_ns: int) -> list[str]:
    anchors = [ts for ts in (contract.close_ts_ns, contract.resolve_ts_ns) if ts is not None]
    if not anchors:
        return ["contract has no close/resolve time; observation instant not cross-checked"]
    if min(abs(ts - end_ns) for ts in anchors) > tolerance_ns:
        raise _Ambiguous(
            f"parsed observation end {iso_from_ns(end_ns)} disagrees with contract close/resolve "
            f"times {[iso_from_ns(t) for t in anchors]}"
        )
    return []


def _draft(
    contract: PredictionContract,
    *,
    underlying: str,
    operator: Operator,
    strikes: tuple[Decimal, ...],
    start_ns: int | None,
    end_ns: int,
    method: str,
    timezone: str,
    source: str,
    rounding_rule: str,
    early_close_rule: str,
    outcome: str,
    notes: str,
) -> ContractMapping:
    return ContractMapping(
        venue=contract.venue,
        contract_id=contract.contract_id,
        underlyings=(underlying,),
        operator=operator,
        strikes=strikes,
        observation_start_ns=start_ns,
        observation_end_ns=end_ns,
        observation_method=method,
        timezone=timezone,
        resolution_source=source,
        rounding_rule=rounding_rule,
        early_close_rule=early_close_rule,
        outcome_semantics=outcome,
        event_family=canonical_event_family((underlying,), method, end_ns),
        version=1,
        review_status=MappingStatus.DRAFT,
        notes=notes,
    )


def _local_text(ns: int, tz: str) -> str:
    return utc_ns_to_local(ns, tz).strftime("%Y-%m-%d %H:%M:%S %Z")


# --------------------------------------------------------------------------------------
# Kalshi
# --------------------------------------------------------------------------------------


def parse_kalshi_contract(
    contract: PredictionContract,
    semantics: SeriesSemantics | None = None,
    *,
    close_tolerance_ns: int = DEFAULT_CLOSE_TOLERANCE_NS,
) -> ParseResult:
    """DRAFT proposal for a Kalshi crypto market, or ``None`` with the ambiguity reason."""
    try:
        mapping, warnings = _parse_kalshi(contract, semantics, close_tolerance_ns)
    except _Ambiguous as exc:
        return ParseResult(
            None, reason=f"{contract.contract_id}: {exc}", proposed_by=KALSHI_PARSER_ID
        )
    return ParseResult(mapping, warnings=tuple(warnings), proposed_by=KALSHI_PARSER_ID)


def _parse_kalshi(
    contract: PredictionContract, semantics: SeriesSemantics | None, close_tolerance_ns: int
) -> tuple[ContractMapping, list[str]]:
    if contract.venue is not Venue.KALSHI:
        raise _Ambiguous("not a Kalshi contract")
    table = semantics if semantics is not None else load_series_semantics()
    series = contract.series_id or contract.native_id.split("-", 1)[0]
    sem = table.kalshi.get(series)
    if sem is None:
        raise _Ambiguous(f"series {series!r} has no entry in series_semantics.yaml")
    if not sem.auto_parse:
        raise _Ambiguous(f"series {series} is not enabled for automatic parsing: {sem.note}")
    ticker = _KALSHI_TICKER_RE.match(contract.native_id)
    if ticker is None or ticker["series"] != series:
        raise _Ambiguous(
            f"ticker {contract.native_id!r} is not <SERIES>-<YY><MON><DD><HH>-<suffix> for {series}"
        )

    md = contract.settlement_metadata
    if md.get("strike_type") is None:
        raise _Ambiguous("settlement_metadata.strike_type missing")
    strike_type = str(md["strike_type"]).strip().lower()
    if strike_type in NON_DETERMINISTIC_STRIKE_TYPES:
        raise _Ambiguous(f"strike_type {strike_type!r} has no deterministic payoff mapping")
    if strike_type not in KALSHI_STRIKE_TYPES:
        raise _Ambiguous(f"unknown strike_type {strike_type!r}")
    operator, which = KALSHI_STRIKE_TYPES[strike_type]
    floor = decimal_from_metadata(md.get("floor_strike"))
    cap = decimal_from_metadata(md.get("cap_strike"))
    if which == "floor":
        if floor is None or cap is not None:
            raise _Ambiguous(f"strike_type {strike_type} needs floor_strike only")
        strikes: tuple[Decimal, ...] = (floor,)
    elif which == "cap":
        if cap is None or floor is not None:
            raise _Ambiguous(f"strike_type {strike_type} needs cap_strike only")
        strikes = (cap,)
    else:
        if floor is None or cap is None or floor >= cap:
            raise _Ambiguous("strike_type between needs floor_strike < cap_strike")
        strikes = (floor, cap)
    if any(s <= 0 for s in strikes):
        raise _Ambiguous("strikes must be positive")
    suffix = ticker["suffix"]
    if suffix and suffix.startswith("T"):
        ticker_strike = decimal_from_metadata(suffix[1:])
        if ticker_strike is None or ticker_strike not in strikes:
            raise _Ambiguous(f"ticker strike {suffix!r} disagrees with floor/cap metadata")

    month = _MONTHS.get(ticker["mon"].lower())
    if month is None:
        raise _Ambiguous(f"unknown month code {ticker['mon']!r}")
    try:
        day = date(2000 + int(ticker["yy"]), month, int(ticker["dd"]))
    except ValueError as exc:
        raise _Ambiguous(f"invalid ticker date: {exc}") from exc
    hour = int(ticker["hh"])
    if hour > 23:
        raise _Ambiguous(f"invalid ticker hour {hour}")
    end_ns = _local_ns(day, hour, 0, sem.timezone)
    start_ns = end_ns - sem.observation_window_ns

    warnings: list[str] = []
    if contract.close_ts_ns is None:
        warnings.append("contract has no close time; ticker time not cross-checked")
    elif abs(contract.close_ts_ns - end_ns) > close_tolerance_ns:
        raise _Ambiguous(
            f"close time {iso_from_ns(contract.close_ts_ns)} disagrees with the ticker's stated "
            f"time {iso_from_ns(end_ns)}"
        )
    rules = contract.rules_text.lower()
    if rules:
        missing = [kw for kw in sem.rules_keywords if kw.lower() not in rules]
        if missing:
            raise _Ambiguous(f"rules text does not mention {missing} expected for {series}")
    else:
        warnings.append("rules text unavailable; source keywords not cross-checked")
    if contract.can_close_early and "NO_EARLY_CLOSE" in sem.early_close_rule:
        raise _Ambiguous("venue flags can_close_early but the series terms say no early close")
    if not sem.verified:
        warnings.append(f"series {series} semantics UNVERIFIED: {sem.verification_note}")

    stated = _local_text(end_ns, sem.timezone)
    variable = f"{sem.observation_method} of {sem.resolution_source} before {stated}"
    if operator is Operator.BETWEEN:
        outcome = (
            f"YES iff {strikes[0]} <= {variable} <= {strikes[1]} "
            "(Kalshi 'between' is inclusive at both ends)"
        )
    else:
        symbol = {Operator.GT: ">", Operator.GE: ">=", Operator.LT: "<", Operator.LE: "<="}
        outcome = f"YES iff {variable} {symbol[operator]} {strikes[0]}"
    notes = (
        f"DRAFT proposed by {KALSHI_PARSER_ID} from series_semantics.yaml row {series} "
        f"(verified={sem.verified}; {sem.verification_note}). strike_type={strike_type}. "
        "Missing/incomplete index data resolves affected strikes No. Requires human review."
    )
    if operator is Operator.BETWEEN:
        notes += " BETWEEN semantics for this Kalshi contract: low <= X <= high (inclusive)."
    mapping = _draft(
        contract,
        underlying=sem.underlying,
        operator=operator,
        strikes=strikes,
        start_ns=start_ns,
        end_ns=end_ns,
        method=sem.observation_method,
        timezone=sem.timezone,
        source=sem.resolution_source,
        rounding_rule=sem.rounding_rule,
        early_close_rule=sem.early_close_rule,
        outcome=outcome,
        notes=notes,
    )
    return mapping, warnings


# --------------------------------------------------------------------------------------
# Polymarket
# --------------------------------------------------------------------------------------


def parse_polymarket_contract(
    contract: PredictionContract,
    semantics: SeriesSemantics | None = None,
    *,
    close_tolerance_ns: int = DEFAULT_CLOSE_TOLERANCE_NS,
) -> ParseResult:
    """DRAFT proposal for a supported Polymarket crypto market, or ``None`` with a reason."""
    try:
        mapping, warnings = _parse_polymarket(contract, semantics, close_tolerance_ns)
    except _Ambiguous as exc:
        return ParseResult(
            None, reason=f"{contract.contract_id}: {exc}", proposed_by=POLYMARKET_PARSER_ID
        )
    return ParseResult(mapping, warnings=tuple(warnings), proposed_by=POLYMARKET_PARSER_ID)


def _template(table: SeriesSemantics, kind: str) -> PolymarketTemplate:
    tmpl = table.template(kind)
    if tmpl is None:
        raise _Ambiguous(f"no Polymarket template of kind {kind} in series_semantics.yaml")
    return tmpl


def _asset(tmpl: PolymarketTemplate, asset_word: str) -> tuple[str, AssetSource]:
    code = _ASSET_CODES[asset_word]
    spec = tmpl.assets.get(code)
    if spec is None:
        raise _Ambiguous(f"template {tmpl.template_id} has no source for asset {code}")
    return code, spec


def _parse_polymarket(
    contract: PredictionContract, semantics: SeriesSemantics | None, close_tolerance_ns: int
) -> tuple[ContractMapping, list[str]]:
    if contract.venue is not Venue.POLYMARKET:
        raise _Ambiguous("not a Polymarket contract")
    table = semantics if semantics is not None else load_series_semantics()
    title = _norm(contract.title)
    rules = _norm(contract.rules_text)
    slug = _norm(str(contract.settlement_metadata.get("slug") or contract.native_id))
    if m := _UPDOWN_SLUG_RE.match(slug):
        return _parse_slug_updown(contract, table, m, title, rules)
    if _UPDOWN_DAILY_RE.match(title):
        raise _Ambiguous(
            "daily Up/Down markets (12:00 ET close vs previous day; tie resolves 50-50 SCALAR) "
            "are not supported by the deterministic parser"
        )
    if m := _UPDOWN_HOURLY_RE.match(title):
        return _parse_hourly_updown(contract, table, m, rules, close_tolerance_ns)
    for pattern in _THRESHOLD_TITLE_RES:
        if m := pattern.match(title):
            return _parse_threshold(contract, table, m, rules, close_tolerance_ns)
    raise _Ambiguous(f"title {contract.title!r} matches no supported template")


def _parse_threshold(
    contract: PredictionContract,
    table: SeriesSemantics,
    m: re.Match[str],
    rules: str,
    close_tolerance_ns: int,
) -> tuple[ContractMapping, list[str]]:
    tmpl = _template(table, "THRESHOLD")
    _, spec = _asset(tmpl, m["asset"])
    _require_yes(contract, "yes")
    if m.groupdict().get("blank"):
        strike = _metadata_strike(contract)
    else:
        strike = _parse_strike(m["strike"], m["k"])
    _require_source(rules, tmpl, spec)
    operator = _threshold_operator(rules, m["dir"])
    hour, minute, tz, abbrs = _rule_time(rules)
    day = _resolve_date(m["month"], m["day"], m["year"], contract)
    start_ns = _local_ns(day, hour, minute, tz)
    for abbr in sorted(abbrs):
        _check_tz_abbreviation(abbr, start_ns, tz)
    end_ns = start_ns + NS_PER_MIN  # the 1-minute candle closes one minute after it opens
    warnings = _check_close(contract, end_ns, close_tolerance_ns)
    if tz != tmpl.timezone:
        warnings.append(
            f"rules state the observation time in {tz}, not the template's usual {tmpl.timezone}"
        )
    symbol = {Operator.GT: ">", Operator.GE: ">=", Operator.LT: "<", Operator.LE: "<="}[operator]
    outcome = (
        f"YES iff the final close of the {spec.resolution_source} candle opening "
        f"{_local_text(start_ns, tz)} is {symbol} {strike}"
    )
    notes = (
        f"DRAFT proposed by {POLYMARKET_PARSER_ID} via template {tmpl.template_id} "
        f"(verified={tmpl.verified}; {tmpl.verification_note}). Candle window [start, end) is "
        "the 1-minute candle; early_close_rule must be determined by the reviewer."
    )
    mapping = _draft(
        contract,
        underlying=spec.underlying,
        operator=operator,
        strikes=(strike,),
        start_ns=start_ns,
        end_ns=end_ns,
        method=tmpl.observation_method,
        timezone=tz,
        source=spec.resolution_source,
        rounding_rule=tmpl.rounding_rule,
        early_close_rule=tmpl.early_close_rule,
        outcome=outcome,
        notes=notes,
    )
    return mapping, warnings


def _parse_hourly_updown(
    contract: PredictionContract,
    table: SeriesSemantics,
    m: re.Match[str],
    rules: str,
    close_tolerance_ns: int,
) -> tuple[ContractMapping, list[str]]:
    tmpl = _template(table, "UPDOWN_HOURLY")
    _, spec = _asset(tmpl, m["asset"])
    _require_yes(contract, "up")
    _require_source(rules, tmpl, spec)
    tie_rule = _updown_tie_rule(rules)
    hour, minute = _hour24(int(m["hour"]), int(m["minute"]) if m["minute"] else None, m["ampm"])
    tz = _TZ_NAMES[m["tz"]]
    if tz != tmpl.timezone:
        raise _Ambiguous(f"title time is in {tz}, template expects {tmpl.timezone}")
    day = _resolve_date(m["month"], m["day"], m["year"], contract)
    start_ns = _local_ns(day, hour, minute, tz)
    end_ns = start_ns + NS_PER_HOUR
    warnings = _check_close(contract, end_ns, close_tolerance_ns)
    tie_text = (
        "close >= open (ties resolve Up)"
        if tie_rule == "TIES_RESOLVE_YES"
        else ("close > open (ties resolve Down)")
    )
    outcome = (
        f"YES (Up) iff {spec.resolution_source} 1-hour candle opening "
        f"{_local_text(start_ns, tz)} has {tie_text}"
    )
    notes = (
        f"DRAFT proposed by {POLYMARKET_PARSER_ID} via template {tmpl.template_id} "
        f"(verified={tmpl.verified}; {tmpl.verification_note}). Operator UP here means "
        f"{tie_text}; tie handling is encoded in rounding_rule={tie_rule}."
    )
    mapping = _draft(
        contract,
        underlying=spec.underlying,
        operator=Operator.UP,
        strikes=(),
        start_ns=start_ns,
        end_ns=end_ns,
        method=tmpl.observation_method,
        timezone=tz,
        source=spec.resolution_source,
        rounding_rule=tie_rule,
        early_close_rule=tmpl.early_close_rule,
        outcome=outcome,
        notes=notes,
    )
    return mapping, warnings


def _parse_slug_updown(
    contract: PredictionContract,
    table: SeriesSemantics,
    m: re.Match[str],
    title: str,
    rules: str,
) -> tuple[ContractMapping, list[str]]:
    tmpl = _template(table, "UPDOWN_SLUG")
    _, spec = _asset(tmpl, m["coin"])
    _require_yes(contract, "up")
    start_ns = int(m["start"]) * NS_PER_S
    end_ns = start_ns + _SLUG_INTERVALS_NS[m["interval"]]
    if tmpl.rule_effective_from_ns is not None and start_ns < tmpl.rule_effective_from_ns:
        raise _Ambiguous(
            f"market starts {iso_from_ns(start_ns)}, before the TWAP rule took effect "
            f"({iso_from_ns(tmpl.rule_effective_from_ns)}); older rule versions unsupported"
        )
    _require_source(rules, tmpl, spec)
    if "twap" not in rules and "time-weighted" not in rules and "time weighted" not in rules:
        raise _Ambiguous("rules text does not state that prices are TWAPs")
    tie_rule = _updown_tie_rule(rules)
    if tie_rule != tmpl.rounding_rule:
        raise _Ambiguous(
            f"rules tie handling {tie_rule} differs from template {tmpl.rounding_rule}"
        )
    warnings: list[str] = []
    # The slug start is authoritative; the title is only a cross-check (a contradiction
    # fails closed, an unreadable title merely skips the check).
    tm = _UPDOWN_RANGE_RE.match(title) or _UPDOWN_HOURLY_RE.match(title)
    if tm is None or tm["ampm"] is None:
        warnings.append("title start time not recognised; slug start time not cross-checked")
    else:
        tz = _TZ_NAMES[tm["tz"]]
        minute_text = tm["minute"]
        hour, minute = _hour24(
            int(tm["hour"]), int(minute_text) if minute_text else None, tm["ampm"]
        )
        day = _resolve_date(tm["month"], tm["day"], tm["year"], contract)
        titled = _local_ns(day, hour, minute, tz)
        if titled != start_ns:
            raise _Ambiguous(
                f"title start {iso_from_ns(titled)} disagrees with slug start "
                f"{iso_from_ns(start_ns)}"
            )
    outcome = (
        f"YES (Up) iff {spec.resolution_source} at {iso_from_ns(end_ns)} >= the value at "
        f"{iso_from_ns(start_ns)} (ties resolve Up)"
    )
    notes = (
        f"DRAFT proposed by {POLYMARKET_PARSER_ID} via template {tmpl.template_id} "
        f"(verified={tmpl.verified}; {tmpl.verification_note}). {tmpl.note}"
    )
    mapping = _draft(
        contract,
        underlying=spec.underlying,
        operator=Operator.UP,
        strikes=(),
        start_ns=start_ns,
        end_ns=end_ns,
        method=tmpl.observation_method,
        timezone=tmpl.timezone,
        source=spec.resolution_source,
        rounding_rule=tie_rule,
        early_close_rule=tmpl.early_close_rule,
        outcome=outcome,
        notes=notes,
    )
    return mapping, warnings


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def propose_mapping(
    contract: PredictionContract, semantics: SeriesSemantics | None = None
) -> ParseResult:
    """Venue dispatch. The result (if any) must go through ``MappingRegistry.propose``."""
    if contract.venue is Venue.KALSHI:
        return parse_kalshi_contract(contract, semantics)
    if contract.venue is Venue.POLYMARKET:
        return parse_polymarket_contract(contract, semantics)
    return ParseResult(
        None, reason=f"{contract.contract_id}: no parser for venue {contract.venue}", proposed_by=""
    )


__all__ = [
    "DEFAULT_SERIES_SEMANTICS_PATH",
    "KALSHI_PARSER_ID",
    "POLYMARKET_PARSER_ID",
    "AssetSource",
    "KalshiSeriesSemantics",
    "ParseResult",
    "PolymarketTemplate",
    "SeriesSemantics",
    "load_series_semantics",
    "parse_kalshi_contract",
    "parse_polymarket_contract",
    "parse_series_semantics",
    "propose_mapping",
]
