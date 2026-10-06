"""Raw store -> normalized events (the replayable derivation step).

Every raw message is re-parsed by the *same* venue adapters used live, passed through the
same :class:`~cma.normalization.pipeline.Normalizer`, and returned in receive-time order.
Malformed payloads are skipped and counted (they were quarantined at capture time too).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from cma.adapters.base import MalformedPayloadError, VenueAdapter
from cma.domain.enums import Venue
from cma.domain.models import MarketEvent
from cma.normalization.pipeline import Normalizer
from cma.storage.raw import RawReader


def default_adapters() -> dict[Venue, VenueAdapter]:
    from cma.adapters.crypto.binance import BinanceAdapter
    from cma.adapters.crypto.coinbase import CoinbaseAdapter
    from cma.adapters.crypto.deribit import DeribitAdapter
    from cma.adapters.kalshi.parser import KalshiAdapter
    from cma.adapters.polymarket.parser import PolymarketAdapter

    return {
        Venue.KALSHI: KalshiAdapter(),
        Venue.POLYMARKET: PolymarketAdapter(),
        Venue.COINBASE: CoinbaseAdapter(),
        Venue.BINANCE: BinanceAdapter(),
        Venue.DERIBIT: DeribitAdapter(),
    }


def normalize_raw(
    raw_root: Path,
    instruments: Iterable[str] | None,
    normalizer: Normalizer,
    *,
    adapters: dict[Venue, VenueAdapter] | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> list[MarketEvent]:
    wanted = None if instruments is None else set(instruments)
    table = adapters or default_adapters()
    out: list[MarketEvent] = []
    for raw in RawReader(raw_root).iter_messages(start_ns=start_ns, end_ns=end_ns):
        adapter = table.get(raw.venue)
        if adapter is None:
            continue
        try:
            events = adapter.parse(raw)
        except MalformedPayloadError as exc:
            if on_error is not None:
                on_error(exc)
            continue
        for ev in events:
            if wanted is not None and ev.instrument_id not in wanted:
                continue
            norm = normalizer.process(ev)
            if norm is not None:
                out.append(norm)
    out.sort(key=lambda e: (e.recv_ts_ns, e.venue_ts_ns))
    return out
