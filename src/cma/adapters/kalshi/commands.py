"""Kalshi WebSocket command frames (Trade API v2 ``cmd`` messages).

The frame shapes follow Kalshi's documentation as of October 2026; they are unverified
against the live service. Commands carry a per-connection ``id``, and sessions rebuild
the subscription list from ``id = 1`` on every connection.

The orderbook channel is subscribed with ONE market per subscription. Kalshi numbers
``seq`` per subscription (``sid``), while the canonical book builder checks contiguity
per instrument; giving each market its own sid makes the two agree. The trade and
ticker channels carry no sequence, so they are batched into one subscription.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Final

ORDERBOOK_CHANNEL: Final = "orderbook_delta"
TRADE_CHANNEL: Final = "trade"
TICKER_CHANNEL: Final = "ticker"


def _frame(cmd_id: int, cmd: str, params: dict[str, object]) -> str:
    return json.dumps({"id": cmd_id, "cmd": cmd, "params": params}, separators=(",", ":"))


def build_subscribe_command(
    cmd_id: int, channels: Sequence[str], market_tickers: Sequence[str] | None = None
) -> str:
    """``{"id": n, "cmd": "subscribe", "params": {"channels": [...], "market_tickers": [...]}}``."""
    params: dict[str, object] = {"channels": list(channels)}
    if market_tickers:
        params["market_tickers"] = list(market_tickers)
    return _frame(cmd_id, "subscribe", params)


def build_subscriptions(
    market_tickers: Sequence[str],
    *,
    orderbook: bool = True,
    trades: bool = True,
    ticker: bool = False,
    start_id: int = 1,
) -> list[str]:
    """Subscription frames for a fresh connection.

    The orderbook channel gets one subscription per market; trade/ticker share one.
    """
    frames: list[str] = []
    cmd_id = start_id
    if orderbook:
        for market in market_tickers:
            frames.append(build_subscribe_command(cmd_id, [ORDERBOOK_CHANNEL], [market]))
            cmd_id += 1
    batched = [c for c, on in ((TRADE_CHANNEL, trades), (TICKER_CHANNEL, ticker)) if on]
    if batched and market_tickers:
        frames.append(build_subscribe_command(cmd_id, batched, list(market_tickers)))
    return frames


def build_get_snapshot_command(cmd_id: int, sid: int, market_tickers: Sequence[str]) -> str:
    """In-band re-snapshot after a gap: ``update_subscription`` with ``get_snapshot``."""
    return _frame(
        cmd_id,
        "update_subscription",
        {"sids": [sid], "market_tickers": list(market_tickers), "action": "get_snapshot"},
    )
