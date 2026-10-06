"""Live staleness study: move detection, stale-quote lifetime, executable edge, end to end."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from cma.domain.enums import Venue
from cma.domain.fees import KALSHI_STANDARD
from cma.domain.models import RawMessage
from cma.domain.time import NS_PER_MS, NS_PER_S
from cma.research.live_study import (
    AboveContract,
    Quote,
    StudyConfig,
    analyze,
    clustered_stats,
    detect_moves,
    executable_edge_c,
    live_decision,
    reaction_ms,
    render_markdown,
    stale_lifetime_ms,
    stream_quotes,
)
from cma.storage.raw import RawRecorder
from tests.factories import contract, snapshot

pytestmark = pytest.mark.unit
T = 1_800_000_000 * NS_PER_S
KX = "KALSHI:KXBTCD-TEST-T100000"


def test_detect_moves_crossing_direction_and_cooldown() -> None:
    ts = T + np.arange(0, 20_000, 100, dtype=np.int64) * NS_PER_MS
    mid = np.full(ts.size, 100_000.0)
    mid[50:] = 100_110.0  # +11 bps at t = 5 s
    mid[60:] = 100_220.0  # another +11 bps 1 s later: inside the 5 s cooldown
    mid[150:] = 100_000.0  # -22 bps at 15 s
    moves = detect_moves(ts, mid, threshold_bps=5.0, window_ms=1_000, cooldown_ms=5_000)
    assert [(int((t - T) // NS_PER_MS), d) for t, d, _ in moves] == [(5_000, 1), (15_000, -1)]
    assert moves[0][2] == pytest.approx(10.995, abs=0.01)
    assert detect_moves(ts, mid, threshold_bps=50.0) == []


def _quote(rows: list[tuple[int, float, float]]) -> Quote:
    return Quote(
        ts=np.asarray([T + ms * NS_PER_MS for ms, _, _ in rows], dtype=np.int64),
        bid=np.asarray([b for _, b, _ in rows], dtype=float),
        ask=np.asarray([a for _, _, a in rows], dtype=float),
        bid_qty=np.full(len(rows), 50.0),
        ask_qty=np.full(len(rows), 50.0),
    )


def test_stale_lifetime_up_and_down_moves() -> None:
    q = _quote([(0, 0.49, 0.51), (300, 0.49, 0.51), (700, 0.57, 0.59), (900, 0.40, 0.59)])
    t0 = T + 100 * NS_PER_MS
    assert stale_lifetime_ms(q, t0, +1, 30_000) == pytest.approx(600.0)  # ask 0.51 -> 0.59
    assert stale_lifetime_ms(q, t0, -1, 30_000) == pytest.approx(800.0)  # bid 0.49 -> 0.40
    assert stale_lifetime_ms(q, t0, +1, 500) is None  # survives a 500 ms horizon
    assert stale_lifetime_ms(q, T - NS_PER_S, +1, 30_000) is None  # no quote yet


def test_reaction_times_the_repricing_not_the_first_requote() -> None:
    # the ask ticks up 1c at 100 ms (noise re-quote), the book reprices at 700 ms
    q = _quote([(0, 0.49, 0.51), (100, 0.49, 0.52), (700, 0.57, 0.59)])
    t0 = T + 50 * NS_PER_MS
    assert stale_lifetime_ms(q, t0, +1, 30_000) == pytest.approx(50.0)
    kw = {"mid_anchor": 0.50, "horizon_ms": 30_000}
    assert reaction_ms(q, t0, +1, target=0.04, **kw) == pytest.approx(650.0)
    assert reaction_ms(q, t0, +1, target=0.20, **kw) is None  # never covers 20c
    assert reaction_ms(q, T + 800 * NS_PER_MS, +1, target=0.04, **kw) == 0.0  # already moved
    assert reaction_ms(q, t0, -1, target=0.04, **kw) is None  # wrong direction


def test_executable_edge_uses_order_rounded_kalshi_fee() -> None:
    c = AboveContract(KX, KX, "KXBTCD", 100_000.0, T + 3600 * NS_PER_S, KALSHI_STANDARD)
    q = _quote([(0, 0.49, 0.55)])
    edge, qty = executable_edge_c(c, q, T, direction=1, fair=0.60, qty=10)
    # 10 contracts at 0.55: 0.07*10*0.55*0.45 = 0.17325 -> $0.18 -> 1.8 c/contract
    assert edge == pytest.approx(5.0 - 1.8) and qty == 50.0
    edge_sell, _ = executable_edge_c(c, q, T, direction=-1, fair=0.45, qty=10)
    assert edge_sell == pytest.approx(4.0 - 1.8)


def test_analyze_finds_a_stale_quote_after_a_reference_jump() -> None:
    close = T + 3600 * NS_PER_S
    c = dataclasses.replace(
        contract(KX, resolve_ts_ns=close),
        series_id="KXBTCD",
        settlement_metadata={"strike_type": "greater", "floor_strike": 100_000.0},
    )
    events = []
    for k in range(-1200, 600):  # reference every 100 ms from T-120 s to T+60 s
        t = T + k * 100 * NS_PER_MS
        px = 100_000.0 if k < 0 else 100_110.0
        events.append(
            snapshot(
                "COINBASE:BTC-USD",
                None,
                t,
                [(f"{px - 0.5:.2f}", "1")],
                [(f"{px + 0.5:.2f}", "1")],
                venue=Venue.COINBASE,
            )
        )
    for k in range(-120, 0):  # Kalshi book quoted around 50c until the jump
        events.append(snapshot(KX, None, T + k * NS_PER_S, [("0.49", "100")], [("0.51", "100")]))
    for k in range(0, 60):  # re-quoted 600 ms after the jump
        t = T + 600 * NS_PER_MS + k * NS_PER_S
        events.append(snapshot(KX, None, t, [("0.57", "100")], [("0.59", "100")]))
    events.sort(key=lambda e: e.recv_ts_ns)
    result = analyze(events, [c], cfg=StudyConfig(), dvol=[(T - 600 * NS_PER_S, 50.0)])
    assert result["moves_by_threshold"] == {"3": 1, "5": 1, "10": 1}
    rows = {r["threshold_bps"]: r for r in result["summary"]}
    row = rows[10.0]
    assert row["samples"] == 1 and row["lifetime_ms"]["median"] == pytest.approx(600.0)
    # the book reprices (mid 50c -> 58c vs a predicted +8.2c) 600 ms after the jump
    assert row["reaction_ms"]["median"] == pytest.approx(600.0)
    assert row["reaction_ms"]["share_beyond_horizon"] == 0.0
    edges = row["edge_by_latency"]
    assert edges["0"]["mean_c"] > 3.0 and edges["500"]["share_positive"] == 1.0
    assert edges["1000"]["mean_c"] < 0  # after the re-quote the taker pays the new ask + fee
    # random-time baseline: mostly fairly priced (one grid point lands on the jump itself)
    assert result["baseline"]["mean_c"] < 0 and result["baseline"]["share_positive"] < 0.1
    # market-anchored: Kalshi mid 50c before the move + model change (+8.2c) = 58.2c fair;
    # buying the stale 51c ask with a 1.8c fee nets ~5.4c
    anchored = row["anchored_edge_by_latency"]
    assert anchored["0"]["mean_c"] == pytest.approx(50.0 + 8.2 - 51.0 - 1.8, abs=0.1)
    assert anchored["1000"]["mean_c"] < 0
    assert result["baseline_anchored"]["mean_c"] < 0  # no news: pay half-spread + fee
    (pooled,) = (r for r in result["pooled"] if r["threshold_bps"] == 10.0)
    assert pooled["series"] == "all" and pooled["anchored_edge_by_latency"]["0"]["moves"] == 1
    # one move: positive at 100 ms, far too little data to reject or promote
    assert result["decision"]["decision"] == "COLLECT_MORE_DATA"
    text = render_markdown(result)
    assert "Market-anchored executable edge" in text and "Decision: `COLLECT_MORE_DATA`" in text


def test_clustered_stats_treats_each_move_as_one_cluster() -> None:
    st = clustered_stats([(1.0, 1), (3.0, 1), (2.0, 2), (6.0, 2)])
    # mean 3; cluster residual sums -2 and +2 -> sqrt(2/1 * 8) / 4 = 1
    assert st["mean_c"] == pytest.approx(3.0) and st["se_c"] == pytest.approx(1.0)
    assert (st["n"], st["moves"], st["share_positive"]) == (4, 2, 1.0)
    assert clustered_stats([(1.0, 1), (2.0, 1)])["se_c"] is None  # one move: undefined
    assert clustered_stats([])["n"] == 0


def _pooled(cells: dict[float, tuple[float, float, int]]) -> dict[str, object]:
    """threshold -> (mean, se, moves), same numbers at 100 and 250 ms."""
    rows = []
    for thr, (mean, se, moves) in cells.items():
        st = {"n": 4 * moves, "moves": moves, "mean_c": mean, "se_c": se, "share_positive": 0.1}
        rows.append({"threshold_bps": thr, "anchored_edge_by_latency": {"100": st, "250": st}})
    return {"pooled": rows}


@pytest.mark.parametrize(
    ("cells", "expected", "why"),
    [
        ({3.0: (-1.5, 0.2, 90), 5.0: (-2.0, 0.3, 40), 10.0: (-1.0, 0.9, 8)}, "REJECT", "beyond"),
        ({5.0: (-2.0, 0.3, 40), 10.0: (0.4, 0.9, 8)}, "COLLECT_MORE_DATA", "positive"),
        ({5.0: (-2.0, 0.3, 12)}, "COLLECT_MORE_DATA", "only 12 moves"),
        ({5.0: (-0.2, 0.3, 40)}, "COLLECT_MORE_DATA", "within 2 SE"),
        ({3.0: (-1.0, 0.2, 40)}, "COLLECT_MORE_DATA", "no 5 bps samples"),
    ],
)
def test_live_decision_rule(
    cells: dict[float, tuple[float, float, int]], expected: str, why: str
) -> None:
    dec = live_decision(_pooled(cells))
    assert dec["decision"] == expected and why in dec["reasons"][0]
    assert "One live window never promotes" in dec["rule"]


def test_live_decision_without_reference_data() -> None:
    dec = live_decision({"error": "no reference (Coinbase BTC-USD) data"})
    assert dec["decision"] == "COLLECT_MORE_DATA" and "no reference" in dec["reasons"][0]


def _kalshi_ws(kind: str, seq: int, **msg: object) -> str:
    body = {"market_ticker": "KXBTCD-TEST-T100000", **msg}
    return json.dumps({"type": kind, "sid": 7, "seq": seq, "msg": body})


def test_stream_quotes_survives_transient_crosses_and_reconnects(tmp_path: Path) -> None:
    """A momentary cross must not kill the book, and a reconnect snapshot (seq back to 1)
    must replace it; both used to leave the book dead or frozen for the rest of the run."""
    snap = {"yes_dollars_fp": [["0.4900", "100.00"]], "no_dollars_fp": [["0.4900", "100.00"]]}
    rows = [  # (ms, connection, payload); canonical YES book starts at 0.49 / 0.51
        (0, "k1", _kalshi_ws("orderbook_snapshot", 1, **snap)),
        # a YES bid at 0.55 shows up before the NO bid it trades against is removed
        (
            10,
            "k1",
            _kalshi_ws("orderbook_delta", 2, price_dollars="0.5500", delta_fp="5.00", side="yes"),
        ),
        (
            11,
            "k1",
            _kalshi_ws("orderbook_delta", 3, price_dollars="0.4900", delta_fp="-100.00", side="no"),
        ),
        (
            20,
            "k1",
            _kalshi_ws("orderbook_delta", 4, price_dollars="0.4000", delta_fp="50.00", side="no"),
        ),
        # reconnect: fresh snapshot, sequence restarts at 1
        (
            500,
            "k2",
            _kalshi_ws(
                "orderbook_snapshot",
                1,
                yes_dollars_fp=[["0.5700", "10.00"]],
                no_dollars_fp=[["0.4100", "20.00"]],
            ),
        ),
        (
            600,
            "k2",
            _kalshi_ws("orderbook_delta", 2, price_dollars="0.5800", delta_fp="3.00", side="yes"),
        ),
    ]
    with RawRecorder(tmp_path) as rec:
        for i, (ms, conn, payload) in enumerate(rows):
            rec.append(
                RawMessage(
                    venue=Venue.KALSHI,
                    stream="ws",
                    recv_ts_ns=T + ms * NS_PER_MS,
                    payload=payload,
                    connection_id=conn,
                    connection_seq=i,
                )
            )
    stats: dict[str, dict[str, int]] = {}
    quotes, n = stream_quotes(tmp_path, [KX], stats=stats)
    q = quotes[KX]
    tops = [
        (int((t - T) // NS_PER_MS), float(b), None if np.isnan(a) else float(a))
        for t, b, a in zip(q.ts, q.bid, q.ask, strict=True)
    ]
    assert n == 6
    assert tops == [
        (0, 0.49, 0.51),
        (11, 0.55, None),  # crossed at 10 ms: skipped, not fatal; the NO side is now empty
        (20, 0.55, 0.60),
        (500, 0.57, 0.59),  # reconnect snapshot (sequence back to 1) replaces the book
        (600, 0.58, 0.59),
    ]
    k = stats["KALSHI"]
    # a reconnect is a new (connection, sid) subscription with its own sequence
    assert (k["crossed"], k["books"], k["books_invalid_at_end"]) == (1, 1, 0)
    assert (k["subscriptions"], k["subscription_gaps"]) == (2, 0)


def _ws(kind: str, sid: int, seq: int, market: str, **msg: object) -> str:
    return json.dumps(
        {"type": kind, "sid": sid, "seq": seq, "msg": {"market_ticker": market, **msg}}
    )


def test_stream_quotes_checks_kalshi_sequence_per_subscription(tmp_path: Path) -> None:
    """Kalshi folds every market of a channel into one subscription: ``seq`` is shared, so
    one market's sequence numbers are never contiguous. Continuity is per subscription;
    a real gap there invalidates every book it carries until each book's next snapshot."""
    a, b = "KXBTCD-TEST-T100000", "KXBTCD-TEST-T101000"
    book = {"yes_dollars_fp": [["0.4900", "10.00"]], "no_dollars_fp": [["0.4900", "10.00"]]}
    rows = [
        (0, _ws("orderbook_snapshot", 1, 1, a, **book)),
        (1, _ws("orderbook_snapshot", 1, 2, b, **book)),
        (5, _ws("orderbook_delta", 1, 3, a, price_dollars="0.5000", delta_fp="4.00", side="yes")),
        (6, _ws("orderbook_delta", 1, 4, b, price_dollars="0.5000", delta_fp="4.00", side="yes")),
        (7, _ws("orderbook_delta", 1, 5, a, price_dollars="0.4800", delta_fp="1.00", side="no")),
        (7, _ws("orderbook_delta", 1, 5, a, price_dollars="0.4800", delta_fp="1.00", side="no")),
        # a market we do not analyse shares the subscription: its seq still counts
        (
            8,
            _ws(
                "orderbook_delta", 1, 6, "OTHER", price_dollars="0.1000", delta_fp="1.00", side="no"
            ),
        ),
        # seq 7 never arrives: both books may be wrong from here on
        (9, _ws("orderbook_delta", 1, 8, b, price_dollars="0.4700", delta_fp="1.00", side="no")),
        (10, _ws("orderbook_delta", 1, 9, a, price_dollars="0.4600", delta_fp="1.00", side="no")),
        (12, _ws("orderbook_snapshot", 1, 10, a, yes_dollars_fp=[["0.5200", "3.00"]])),
    ]
    with RawRecorder(tmp_path) as rec:
        for i, (ms, payload) in enumerate(rows):
            rec.append(
                RawMessage(
                    venue=Venue.KALSHI,
                    stream="ws",
                    recv_ts_ns=T + ms * NS_PER_MS,
                    payload=payload,
                    connection_id="k1",
                    connection_seq=i,
                )
            )
    stats: dict[str, dict[str, int]] = {}
    ka, kb = f"KALSHI:{a}", f"KALSHI:{b}"
    quotes, _ = stream_quotes(tmp_path, [ka, kb], stats=stats)

    def tops(inst: str) -> list[tuple[int, float, float | None]]:
        q = quotes[inst]
        return [
            (int((t - T) // NS_PER_MS), float(x), None if np.isnan(y) else float(y))
            for t, x, y in zip(q.ts, q.bid, q.ask, strict=True)
        ]

    # interleaved deltas apply although each market's own seq jumps (1 -> 3 -> 5)
    assert tops(ka) == [(0, 0.49, 0.51), (5, 0.50, 0.51), (12, 0.52, None)]
    assert tops(kb) == [(1, 0.49, 0.51), (6, 0.50, 0.51)]  # stale after the gap: dropped
    k = stats["KALSHI"]
    assert (k["subscriptions"], k["subscription_gaps"]) == (1, 1)
    assert k["books_invalid_at_end"] == 1  # b never got a new snapshot
