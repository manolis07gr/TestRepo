"""Live staleness study: move detection, stale-quote lifetime, executable edge, end to end."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cma.domain.enums import Venue
from cma.domain.fees import KALSHI_STANDARD
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
    render_markdown,
    stale_lifetime_ms,
)
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
