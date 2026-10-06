"""15-minute vs hourly-ladder consistency check: riskless pairs, persistence, decision."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from cma.research import ladder_check as lc
from cma.research.settlement_data import SettledMarket
from cma.research.settlement_study import fee_c

pytestmark = pytest.mark.unit

C0 = 1_767_229_200  # 2026-01-01T01:00:00Z, on the hour
CFG = lc.LadderConfig()


def _m15(i: int, strike: float, value: float) -> SettledMarket:
    close = C0 + 3600 * i
    return SettledMarket(
        ticker=f"KXBTC15M-H{i}",
        event_ticker=f"E{i}",
        series="KXBTC15M",
        open_ts=close - 900,
        close_ts=close,
        strike_type="greater_or_equal",
        floor_strike=strike,
        cap_strike=None,
        result="yes" if value >= strike else "no",
        expiration_value=value,
        archived=False,
    )


def _candles(open_ts: int, quotes: dict[int, tuple[float, float]]) -> list[tuple[Any, ...]]:
    """Candles 0..15 (end = open + 60 k) with the given (bid, ask) closes; default 0.49/0.51."""
    rows = []
    for k in range(16):
        bid, ask = quotes.get(k, (0.49, 0.51))
        rows.append((open_ts + 60 * k, bid, bid, bid, bid, ask, ask, ask, ask, 10.0))
    return rows


def _world(
    n: int, overrides: dict[int, dict[str, dict[int, tuple[float, float]]]] | None = None
) -> tuple[list[SettledMarket], dict[str, Any], dict[int, dict[str, Any]]]:
    """n hours; 15-minute strike 100.50 between ladder strikes 99.99 / 100.99. Consistent
    default quotes: lo 0.60/0.62, 15-minute 0.49/0.51, hi 0.38/0.40."""
    overrides = overrides or {}
    markets, candles, ladder = [], {}, {}
    for i in range(n):
        m = _m15(i, 100.50, 100.70)
        markets.append(m)
        o = overrides.get(i, {})
        candles[m.ticker] = _candles(m.open_ts, o.get("m15", {}))
        lo_q = dict.fromkeys(range(16), (0.6, 0.62)) | o.get("lo", {})
        hi_q = dict.fromkeys(range(16), (0.38, 0.4)) | o.get("hi", {})
        ladder[m.close_ts] = {
            "close_ts": m.close_ts,
            "event": f"KXBTCD-{i}",
            "k15": 100.50,
            "spacing": 1,
            "lo": {"ticker": "lo", "strike": 99.99, "candles": _candles(m.open_ts, lo_q)},
            "hi": {"ticker": "hi", "strike": 100.99, "candles": _candles(m.open_ts, hi_q)},
        }
    return markets, candles, ladder


def test_every_pair_pays_at_least_one_dollar_in_every_outcome() -> None:
    k_lo, k15, k_hi = 99.99, 100.50, 100.99
    for pair, value in itertools.product(lc.PAIRS, (99.0, 99.99, 100.0, 100.5, 100.99, 101.5)):
        assert lc.pair_payoff(pair, value, k15, k_lo, k_hi) >= 1
    assert lc.pair_payoff("a", 100.7, k15, k_lo, k_hi) == 2  # both legs win inside the band
    assert lc.pair_payoff("b", 100.2, k15, k_lo, k_hi) == 2


def test_riskless_profit_pays_both_fees() -> None:
    assert lc.riskless_profit_c(0.45, 0.50, 100) == pytest.approx(
        5.0 - fee_c(0.45, 100) - fee_c(0.50, 100)
    )
    assert lc.riskless_profit_c(0.50, 0.50, 100) < 0


def test_consistent_prices_give_no_opportunity_and_reject() -> None:
    res = lc.analyze(*_world(60))
    assert res["data"]["paired_hours"] == 60
    assert res["opportunities"]["count"] == 0
    assert res["decision"]["decision"] == "REJECT"
    assert res["description"]["pre_fee_crossed_share"] == {"a": 0.0, "b": 0.0, "c": 0.0}
    assert res["description"]["mid_outside_ladder_band_share"] == 0.0


def test_a_gap_that_lasts_a_minute_is_found_and_priced_at_the_next_minute() -> None:
    # hour 3: the 15-minute ask drops to 0.30 at minutes 4-5 while the ladder bid above is 0.38
    over = {3: {"m15": {4: (0.28, 0.30), 5: (0.29, 0.31)}}}
    res = lc.analyze(*_world(10, over))
    opps = res["opportunities"]
    assert opps["count"] == 1
    o = opps["list"][0]
    assert (o["pair"], o["k"]) == ("a", 4)
    expected = lc.riskless_profit_c(0.31, 1 - 0.38, 100)  # filled at minute 5's quotes
    assert o["profit_fill_c"] == pytest.approx(expected)
    # settlement 100.70: 15-minute YES wins and ladder NO above 100.99 wins too
    assert o["realised_c"] == pytest.approx(
        100 * (2 - 0.31 - 0.62) - fee_c(0.31, 100) - fee_c(0.62, 100)
    )


def test_a_one_minute_gap_is_not_an_opportunity() -> None:
    over = {2: {"m15": {6: (0.28, 0.30)}}}  # gone a minute later
    res = lc.analyze(*_world(10, over))
    assert res["opportunities"]["count"] == 0
    assert res["description"]["after_fee_positive_share"]["a"] > 0


def test_ladder_out_of_order_and_rich_15_minute_market() -> None:
    over = {
        1: {"hi": {3: (0.66, 0.68), 4: (0.66, 0.68)}},  # above-strike bid over below-strike ask
        5: {"m15": {7: (0.70, 0.72), 8: (0.70, 0.72)}},  # 15-minute bid over the ask below
    }
    res = lc.analyze(*_world(10, over))
    pairs = {o["close_ts"]: o["pair"] for o in res["opportunities"]["list"]}
    assert pairs[C0 + 3600 * 5] == "b"
    # hour 1: both a and c are riskless (hi bid 0.66 vs 15-minute ask 0.51 and lo ask 0.62);
    # the larger profit at the fill wins
    assert pairs[C0 + 3600 * 1] == "a"


def test_decision_live_size_check_when_frequent_and_large_enough() -> None:
    over = {i: {"m15": {3: (0.20, 0.22), 4: (0.20, 0.22)}} for i in range(0, 50, 10)}  # 5 / 50
    res = lc.analyze(*_world(50, over))
    assert res["opportunities"]["share_of_hours"] == pytest.approx(0.10)
    assert res["decision"]["decision"] == "LIVE_SIZE_CHECK"


def test_hours_without_ladder_or_quotes_are_counted_not_paired(tmp_path: Path) -> None:
    markets, candles, ladder = _world(4)
    del ladder[markets[0].close_ts]
    ladder[markets[1].close_ts] = {"close_ts": markets[1].close_ts, "status": "missing"}
    candles[markets[2].ticker] = _candles(
        markets[2].open_ts, dict.fromkeys(range(16), (0.0, 0.999))
    )
    res = lc.analyze(markets, candles, ladder)
    assert res["data"]["ladder_missing"] == 2
    assert res["data"]["no_sane_minute"] == 1
    assert res["data"]["paired_hours"] == 1
    paths = lc.write_outputs(res, tmp_path)
    assert "Same-venue consistency" in paths[1].read_text()
