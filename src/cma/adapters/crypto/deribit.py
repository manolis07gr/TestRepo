"""Deribit options: ``public/get_book_summary_by_currency`` -> :class:`OptionQuote`.

The JSON-RPC result rows carry ``instrument_name``, ``bid_price``, ``ask_price``,
``mid_price``, ``mark_price``, ``mark_iv`` (percent), ``underlying_price``,
``underlying_index``, ``interest_rate``, ``open_interest``, ``volume`` and
``creation_timestamp`` (ms). The format follows Deribit's API v2 documentation as of
October 2026 and was implemented without live access.

Conventions
    * Instrument names look like ``BTC-27DEC24-100000-C``. Day numbers have no leading
      zero (``BTC-5JUL24-...``), linear options carry the quote currency
      (``BTC_USDC-...``), and fractional strikes use ``d`` as the decimal point
      (``XRP_USDC-30AUG24-0d625-C``). Expiry is 08:00 UTC on the named date.
    * ``mark_iv`` comes as a percentage and is stored as a fraction (52.31 -> 0.5231).
    * Option prices (bid/ask/mark/mid) are in the quote currency of the instrument,
      i.e. in the underlying (BTC/ETH) for inverse options and in USDC for linear ones.
      A null or non-positive bid/ask means "no quote" (``None``).

Option quotes are research inputs (implied distributions), not book state, so
:class:`DeribitAdapter` validates recorded summary pages and emits no market events.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final

from cma.adapters.base import (
    MalformedPayloadError,
    as_list,
    as_mapping,
    check_venue,
    epoch_to_ns,
    load_json,
    malformed_guard,
    opt_str,
    req_str,
    require,
    to_decimal,
)
from cma.domain.enums import Venue
from cma.domain.models import MarketEvent, RawMessage, instrument_key
from cma.domain.time import ns_from_datetime
from cma.ingestion.rest import FetchedPage, HttpFetcher
from cma.storage.raw import RawRecorder

VENUE: Final = Venue.DERIBIT
DERIBIT_REST_URL: Final = "https://www.deribit.com/api/v2"
BOOK_SUMMARY_STREAM_PREFIX: Final = "rest:book_summary:"
EXPIRY_HOUR_UTC: Final = 8

_MONTHS: Final = {
    m: i + 1
    for i, m in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    )
}
_OPTION_RE = re.compile(
    r"^(?P<base>[A-Z0-9]+)(?:_(?P<quote>[A-Z0-9]+))?-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})"
    r"-(?P<strike>\d+(?:[.d]\d+)?)-(?P<cp>[CP])$"
)


def book_summary_stream(currency: str, kind: str = "option") -> str:
    return f"{BOOK_SUMMARY_STREAM_PREFIX}{kind}:{currency.upper()}"


@dataclass(frozen=True, slots=True)
class OptionName:
    underlying: str
    quote_currency: str | None
    expiry_ts_ns: int
    strike: Decimal
    is_call: bool


def parse_instrument_name(name: str) -> OptionName:
    m = _OPTION_RE.match(name.strip())
    if m is None:
        raise MalformedPayloadError(f"not a Deribit option instrument name: {name!r}")
    month = _MONTHS.get(m.group("mon"))
    if month is None:
        raise MalformedPayloadError(f"{name}: unknown month {m.group('mon')!r}")
    try:
        expiry = datetime(
            2000 + int(m.group("yy")), month, int(m.group("day")), EXPIRY_HOUR_UTC, tzinfo=UTC
        )
    except ValueError as exc:
        raise MalformedPayloadError(f"{name}: invalid expiry date ({exc})") from exc
    strike = Decimal(m.group("strike").replace("d", "."))
    if strike <= 0:
        raise MalformedPayloadError(f"{name}: non-positive strike")
    return OptionName(
        underlying=m.group("base"),
        quote_currency=m.group("quote"),
        expiry_ts_ns=ns_from_datetime(expiry),
        strike=strike,
        is_call=m.group("cp") == "C",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionQuote:
    """One option's book summary (prices in the instrument's quote currency)."""

    instrument_name: str
    underlying: str
    quote_currency: str
    expiry_ts_ns: int
    strike: Decimal
    is_call: bool
    bid: Decimal | None
    ask: Decimal | None
    mid: Decimal | None
    mark_price: Decimal | None
    mark_iv: Decimal | None  # fraction: 0.5231 == 52.31 %
    underlying_price: Decimal | None
    underlying_index: str | None
    interest_rate: Decimal | None
    open_interest: Decimal | None
    volume: Decimal | None
    ts_ns: int

    @property
    def instrument_id(self) -> str:
        return instrument_key(VENUE, self.instrument_name)


def _opt_dec(row: Mapping[str, Any], key: str, what: str) -> Decimal | None:
    value = row.get(key)
    return None if value in (None, "") else to_decimal(value, f"{what}.{key}")


def _quote(row: Mapping[str, Any], key: str, what: str) -> Decimal | None:
    value = _opt_dec(row, key, what)
    return value if value is not None and value > 0 else None


def parse_option_row(row: Mapping[str, Any]) -> OptionQuote:
    name = req_str(row, "instrument_name", "Deribit summary")
    what = f"Deribit {name}"
    parsed = parse_instrument_name(name)
    mark_iv = _opt_dec(row, "mark_iv", what)
    return OptionQuote(
        instrument_name=name,
        underlying=parsed.underlying,
        quote_currency=(
            opt_str(row, "quote_currency", what) or parsed.quote_currency or parsed.underlying
        ),
        expiry_ts_ns=parsed.expiry_ts_ns,
        strike=parsed.strike,
        is_call=parsed.is_call,
        bid=_quote(row, "bid_price", what),
        ask=_quote(row, "ask_price", what),
        mid=_quote(row, "mid_price", what),
        mark_price=_opt_dec(row, "mark_price", what),
        mark_iv=None if mark_iv is None else mark_iv / 100,
        underlying_price=_opt_dec(row, "underlying_price", what),
        underlying_index=opt_str(row, "underlying_index", what),
        interest_rate=_opt_dec(row, "interest_rate", what),
        open_interest=_opt_dec(row, "open_interest", what),
        volume=_opt_dec(row, "volume", what),
        ts_ns=epoch_to_ns(require(row, "creation_timestamp", what), "ms", what),
    )


def parse_book_summary(data: Any) -> list[OptionQuote]:
    """JSON-RPC response (or bare result list) -> option quotes; non-option rows skipped."""
    with malformed_guard(VENUE, "book summary"):
        if isinstance(data, Mapping):
            if data.get("error") is not None:
                raise MalformedPayloadError(f"Deribit error response: {data['error']!r}")
            rows = as_list(require(data, "result", "Deribit response"), "Deribit result")
        else:
            rows = as_list(data, "Deribit result")
        quotes = []
        for i, row in enumerate(rows):
            item = as_mapping(row, f"Deribit result[{i}]")
            name = str(item.get("instrument_name", ""))
            if not name.endswith(("-C", "-P")):
                continue  # futures/perpetual rows when kind != option
            quotes.append(parse_option_row(item))
        return quotes


class DeribitAdapter:
    """Validates recorded book-summary pages; option quotes are not market events."""

    @property
    def venue(self) -> Venue:
        return VENUE

    def parse(self, raw: RawMessage) -> list[MarketEvent]:
        check_venue(raw, VENUE)
        if raw.stream.startswith(BOOK_SUMMARY_STREAM_PREFIX):
            parse_book_summary(load_json(raw.payload))
        return []

    def idempotency_key(self, stream: str, payload: str) -> str | None:
        return None


class DeribitClient:
    def __init__(
        self,
        fetcher: HttpFetcher,
        *,
        base_url: str = DERIBIT_REST_URL,
        recorder: RawRecorder | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._base = base_url.rstrip("/")
        self._recorder = recorder

    @property
    def connection_id(self) -> str:
        return self._fetcher.connection_id

    async def get_book_summary_page(self, currency: str, *, kind: str = "option") -> FetchedPage:
        return await self._fetcher.get(
            f"{self._base}/public/get_book_summary_by_currency",
            params={"currency": currency.upper(), "kind": kind},
            what="Deribit get_book_summary_by_currency",
        )

    def raw_message(self, currency: str, page: FetchedPage, *, kind: str = "option") -> RawMessage:
        return RawMessage(
            venue=VENUE,
            stream=book_summary_stream(currency, kind),
            recv_ts_ns=page.recv_ts_ns,
            payload=page.text,
            connection_id=self._fetcher.connection_id,
            connection_seq=page.seq,
        )

    async def get_option_quotes(self, currency: str) -> list[OptionQuote]:
        page = await self.get_book_summary_page(currency)
        quotes = parse_book_summary(load_json(page.text))
        if self._recorder is not None:
            self._recorder.append(self.raw_message(currency, page))
        return quotes
