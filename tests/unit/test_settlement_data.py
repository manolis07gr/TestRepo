"""History fetcher for the settlement study: both Kalshi spellings, paging, resume, chunks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from cma.ingestion.rest import BackoffPolicy, RetryPolicy
from cma.research import settlement_data as sd

pytestmark = pytest.mark.unit

SINCE = 1_790_000_000


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def _api_market(name: str, close_ts: int, result: str = "yes") -> dict[str, Any]:
    return {
        "ticker": name,
        "event_ticker": f"E-{name}",
        "open_time": _iso(close_ts - 900),
        "close_time": _iso(close_ts),
        "result": result,
        "floor_strike": 100.0,
        "strike_type": "greater_or_equal",
        "expiration_value": "100.5",
    }


def _run(handler: Callable[[httpx.Request], httpx.Response], coro: Any) -> Any:
    async def go() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cl:
            return await coro(sd._Http(cl, rate_per_s=1000.0, retry=RetryPolicy(max_attempts=1)))

    return asyncio.run(go())


def test_candle_from_api_reads_live_archived_and_legacy_spellings() -> None:
    def side(fmt: Callable[[float], Any], suffix: str, o: float, h: float, lo: float, c: float):
        return {
            f"open{suffix}": fmt(o),
            f"high{suffix}": fmt(h),
            f"low{suffix}": fmt(lo),
            f"close{suffix}": fmt(c),
        }

    expected = (100, 0.50, 0.52, 0.49, 0.51, 0.51, 0.53, 0.50, 0.52, 12.5)
    live = {
        "end_period_ts": 100,
        "volume_fp": "12.5",
        "yes_bid": side(lambda v: f"{v:.4f}", "_dollars", 0.50, 0.52, 0.49, 0.51),
        "yes_ask": side(lambda v: f"{v:.4f}", "_dollars", 0.51, 0.53, 0.50, 0.52),
    }
    archived = {
        "end_period_ts": 100,
        "volume": "12.5",
        "yes_bid": side(lambda v: f"{v:.4f}", "", 0.50, 0.52, 0.49, 0.51),
        "yes_ask": side(lambda v: f"{v:.4f}", "", 0.51, 0.53, 0.50, 0.52),
    }
    legacy = {
        "end_period_ts": 100,
        "volume": 12.5,
        "yes_bid": side(lambda v: round(v * 100), "", 0.50, 0.52, 0.49, 0.51),
        "yes_ask": side(lambda v: round(v * 100), "", 0.51, 0.53, 0.50, 0.52),
    }
    for c in (live, archived, legacy):
        assert sd.candle_from_api(c) == pytest.approx(expected)
    empty = sd.candle_from_api({"end_period_ts": 5, "yes_bid": {}, "yes_ask": None})
    assert empty[1:9] == (None,) * 8
    assert empty[9] == 0.0


def test_market_from_api_keeps_only_settled_yes_no_markets() -> None:
    raw = _api_market("KXBTC15M-A", SINCE + 900, "no")
    m = sd.market_from_api(raw, series="KXBTC15M", archived=True)
    assert m is not None
    assert (m.close_ts - m.open_ts, m.result, m.expiration_value, m.archived) == (
        900,
        "no",
        100.5,
        True,
    )
    for bad in ({"result": ""}, {"result": "void"}, {"close_time": None}):
        assert sd.market_from_api(raw | bad, series="KXBTC15M", archived=False) is None


def test_fetch_markets_pages_both_listings_and_stops_past_since(tmp_path: Path) -> None:
    live = {
        None: ([_api_market("L1", SINCE + 2000), _api_market("L2", SINCE + 1000)], "c1"),
        "c1": ([_api_market("L3", SINCE + 900)], ""),
    }
    hist = {
        None: ([_api_market("H1", SINCE + 500), _api_market("H0", SINCE - 100)], "h1"),
        "h1": ([_api_market("H-old", SINCE - 5000)], ""),
    }
    seen: list[httpx.URL] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url)
        pages = hist if "/historical/" in req.url.path else live
        markets, cursor = pages[req.url.params.get("cursor")]
        return httpx.Response(200, json={"markets": markets, "cursor": cursor})

    out = _run(
        handler, lambda http: sd.fetch_markets(http, "KXBTC15M", since_ts=SINCE, data_dir=tmp_path)
    )
    assert [m.ticker for m in out] == ["H1", "L3", "L2", "L1"]
    assert [m.archived for m in out] == [True, False, False, False]
    assert not any(u.params.get("cursor") == "h1" for u in seen)  # stopped past since
    first_live = next(u for u in seen if "/historical/" not in u.path)
    assert first_live.params["status"] == "settled"
    assert first_live.params["min_close_ts"] == str(SINCE)
    assert sd.load_markets(tmp_path, "KXBTC15M") == out


def _settled(name: str, *, archived: bool) -> sd.SettledMarket:
    return sd.SettledMarket(
        name,
        f"E-{name}",
        "KXBTC15M",
        SINCE,
        SINCE + 900,
        "greater_or_equal",
        100.0,
        None,
        "yes",
        100.5,
        archived,
    )


def test_fetch_candles_resumes_and_reads_old_markets_from_the_archive(tmp_path: Path) -> None:
    sd.candles_path(tmp_path, "KXBTC15M").write_text(
        json.dumps({"ticker": "A", "candles": []}) + "\n" + '{"ticker": "trunc', encoding="utf-8"
    )
    candle = {
        "end_period_ts": SINCE + 60,
        "yes_bid": {"close_dollars": "0.40"},
        "yes_ask": {"close_dollars": "0.41"},
        "volume_fp": "3",
    }
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if "/C/" in req.url.path:
            return httpx.Response(404, json={"error": "not found"})
        if "/D/" in req.url.path:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"candlesticks": [candle]})

    markets = [_settled(n, archived=(n == "B")) for n in ("A", "B", "C", "D")]
    written = _run(handler, lambda http: sd.fetch_candles(http, markets, data_dir=tmp_path))
    assert written == 2  # B and C; A was on disk, D failed and is retried next run
    assert any(p.endswith("/historical/markets/B/candlesticks") for p in seen)
    assert any(p.endswith("/series/KXBTC15M/markets/C/candlesticks") for p in seen)
    assert not any("/A/" in p for p in seen)
    loaded = sd.load_candles(tmp_path, "KXBTC15M")
    assert set(loaded) == {"A", "B", "C"}
    assert loaded["C"] == []
    assert loaded["B"][0][4] == pytest.approx(0.40)
    assert loaded["B"][0][8] == pytest.approx(0.41)


def test_fetch_coinbase_and_dvol_chunks_resume_and_follow_continuations(tmp_path: Path) -> None:
    start = SINCE - SINCE % 60
    calls = {"coinbase": 0, "deribit": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if "coinbase" in req.url.host:
            calls["coinbase"] += 1
            s = int(datetime.fromisoformat(req.url.params["start"]).timestamp())
            rows = [[s + 60 * i, 1.0, 3.0, 2.0, 2.5, 7.0] for i in reversed(range(300))]
            return httpx.Response(200, json=rows)
        calls["deribit"] += 1
        lo, hi = int(req.url.params["start_timestamp"]), int(req.url.params["end_timestamp"])
        pts = list(range(lo, hi + 1, 3_600_000))
        page, rest = pts[-600:], pts[:-600]
        data = [[t, 0, 0, 0, 40.0 + t / 1e12] for t in page]
        cont = rest[-1] if rest else None
        return httpx.Response(200, json={"result": {"data": data, "continuation": cont}})

    n = _run(
        handler,
        lambda http: sd.fetch_coinbase(
            http, start_ts=start, end_ts=start + 600 * 60, data_dir=tmp_path
        ),
    )
    assert n == 2
    rows = sd.load_chunked_rows(sd.coinbase_path(tmp_path))
    assert len(rows) == 600
    assert rows[0] == [start, 2.0, 3.0, 1.0, 2.5, 7.0]  # [ts, o, h, l, c, v], oldest first
    again = _run(
        handler,
        lambda http: sd.fetch_coinbase(
            http, start_ts=start, end_ts=start + 600 * 60, data_dir=tmp_path
        ),
    )
    assert again == 0
    assert calls["coinbase"] == 2

    n = _run(
        handler,
        lambda http: sd.fetch_dvol(http, start_ts=start, end_ts=start + 60, data_dir=tmp_path),
    )
    assert n == 1
    dv = sd.load_chunked_rows(sd.dvol_path(tmp_path))
    assert len(dv) == sd.DVOL_CHUNK_BARS  # 600 + 400 hourly closes over two pages
    assert calls["deribit"] == 2
    assert dv[0][0] == start - start % 3600


def test_load_chunked_rows_dedups_and_skips_a_cut_line(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        json.dumps({"start": 0, "rows": [[2, 1.0], [1, 1.0]]})
        + "\n"
        + json.dumps({"start": 1, "rows": [[2, 2.0], [3, 3.0]]})
        + "\n"
        + '{"start": 2, "ro',
        encoding="utf-8",
    )
    assert sd.load_chunked_rows(path) == [[1, 1.0], [2, 2.0], [3, 3.0]]
    assert sd.load_chunked_rows(tmp_path / "missing.jsonl") == []


def test_fetch_all_runs_every_stage_without_credentials(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "kalshi" in req.url.host:
            return httpx.Response(200, json={"markets": [], "cursor": ""})
        if "coinbase" in req.url.host:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"result": {"data": [], "continuation": None}})

    summary = asyncio.run(
        sd.fetch_all(
            series=["KXBTC15M"],
            since_ts=SINCE,
            until_ts=SINCE + 3600,
            data_dir=tmp_path,
            kalshi_rate_per_s=1000.0,
            signed=False,
            transport=httpx.MockTransport(handler),
        )
    )
    assert summary["signed"] is False
    assert summary["KXBTC15M_markets"] == 0
    assert summary["coinbase_chunks_new"] > 0
    assert summary["dvol_chunks_new"] > 0
    assert sd.markets_path(tmp_path, "KXBTC15M").exists()


def test_backoff_policy_is_importable_for_callers() -> None:
    assert BackoffPolicy().max_attempts == 8
