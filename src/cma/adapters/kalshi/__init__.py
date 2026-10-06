"""Kalshi adapter: parsing (``parser``), REST client/signing (``client``), WS command
frames (``commands``) and the REST orderbook poller fallback (``poller``)."""

from cma.adapters.kalshi.client import (
    KALSHI_LEGACY_REST_URL,
    KALSHI_LEGACY_WS_URL,
    KALSHI_REST_URL,
    KALSHI_WS_URL,
    KalshiAuthError,
    KalshiRestClient,
    KalshiSigner,
    signing_available,
    ws_auth_headers,
)
from cma.adapters.kalshi.commands import (
    build_get_snapshot_command,
    build_subscribe_command,
    build_subscriptions,
)
from cma.adapters.kalshi.parser import (
    KalshiAdapter,
    KalshiEvent,
    KalshiSeries,
    KalshiTicker,
    fee_schedule_for_series,
    kalshi_instrument,
    orderbook_stream,
    parse_events_page,
    parse_market,
    parse_market_response,
    parse_markets_page,
    parse_rest_orderbook,
    parse_series,
    parse_ticker_message,
    parse_trades_page,
)
from cma.adapters.kalshi.poller import KalshiRestBookPoller

__all__ = [
    "KALSHI_LEGACY_REST_URL",
    "KALSHI_LEGACY_WS_URL",
    "KALSHI_REST_URL",
    "KALSHI_WS_URL",
    "KalshiAdapter",
    "KalshiAuthError",
    "KalshiEvent",
    "KalshiRestBookPoller",
    "KalshiRestClient",
    "KalshiSeries",
    "KalshiSigner",
    "KalshiTicker",
    "build_get_snapshot_command",
    "build_subscribe_command",
    "build_subscriptions",
    "fee_schedule_for_series",
    "kalshi_instrument",
    "orderbook_stream",
    "parse_events_page",
    "parse_market",
    "parse_market_response",
    "parse_markets_page",
    "parse_rest_orderbook",
    "parse_series",
    "parse_ticker_message",
    "parse_trades_page",
    "signing_available",
    "ws_auth_headers",
]
