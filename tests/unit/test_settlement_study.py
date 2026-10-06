"""Hold-to-settlement study: inputs, signals, fills, statistics and the decision, end to end
on synthetic 15-minute markets where either the model or Kalshi is the one mispricing."""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cma.domain.time import NS_PER_S
from cma.features.fair_value import prob_above
from cma.research import settlement_study as ss
from cma.research.settlement_data import Candle, SettledMarket

pytestmark = pytest.mark.unit

T0 = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
CFG = ss.SettlementConfig()


def _market(i: int, open_ts: int, strike: float, settle: float) -> SettledMarket:
    tk = f"KXBTC15M-T{i:05d}"
    return SettledMarket(
        ticker=tk,
        event_ticker=tk,
        series="KXBTC15M",
        open_ts=open_ts,
        close_ts=open_ts + 900,
        strike_type="greater_or_equal",
        floor_strike=strike,
        cap_strike=None,
        result="yes" if settle >= strike else "no",
        expiration_value=settle,
        archived=False,
    )


def _world(
    n_markets: int, *, true_vol: float, kalshi_vol: float, dvol_pct: float, seed: int = 7
) -> tuple[list[SettledMarket], dict[str, list[Candle]], list[list[float]], list[list[float]]]:
    """A BTC path with ``true_vol``; Kalshi quotes the log-normal digital with ``kalshi_vol``
    (1c spread); DVOL reads ``dvol_pct``. The candle starting at T0 + 60 j closes at
    price[j + 1], the price at the end of that minute."""
    rng = random.Random(seed)
    minutes = n_markets * 15 + 240
    sig = true_vol / math.sqrt(ss.MINUTES_PER_YEAR)
    price = [60_000.0]
    for _ in range(minutes):
        price.append(price[-1] * math.exp(rng.gauss(0.0, sig)))
    coinbase = [[T0 + 60 * j, *([price[j + 1]] * 4), 1.0] for j in range(minutes)]
    dvol = [[T0 + 3600 * h, dvol_pct] for h in range(minutes // 60 + 2)]  # hourly closes

    def at(t: int) -> float:
        return price[(t - T0) // 60]

    markets: list[SettledMarket] = []
    candles: dict[str, list[Candle]] = {}
    first_open = T0 + 240 * 60
    for i in range(n_markets):
        o = first_open + 900 * i
        m = _market(i, o, at(o), at(o + 900))
        markets.append(m)
        rows: list[Candle] = []
        for k in range(1, 16):
            e = o + 60 * k
            if k < 15:
                f = prob_above(
                    spot=at(e),
                    strike=m.floor_strike or 0.0,
                    now_ns=e * NS_PER_S,
                    observation_end_ns=m.close_ts * NS_PER_S,
                    sigma=kalshi_vol,
                    observation_method="AVG_60S_BEFORE",
                )
            else:
                f = 1.0 if m.result == "yes" else 0.0
            mid = min(max(f, 0.02), 0.98)
            bid, ask = round(mid - 0.005, 3), round(mid + 0.005, 3)
            rows.append((e, bid, bid, bid, bid, ask, ask, ask, ask, 100.0))
        candles[m.ticker] = rows
    return markets, candles, coinbase, dvol


# ----------------------------------------------------------------------------- pieces


def test_fee_matches_kalshi_quadratic_schedule() -> None:
    assert ss.fee_c(0.50, 100) == pytest.approx(1.75)
    assert ss.fee_c(0.10, 100) == pytest.approx(0.63)
    assert ss.fee_c(0.90, 100) == pytest.approx(0.63)
    # one contract: rounded up to a whole cent
    assert ss.fee_c(0.50, 1) == pytest.approx(2.0)


def test_best_side_buys_yes_below_fair_and_no_above_it() -> None:
    side, edge = ss.best_side(0.60, 0.50, 0.52, 100)
    assert side == 1
    assert edge == pytest.approx(100 * (0.60 - 0.52) - ss.fee_c(0.52, 100))
    side, edge = ss.best_side(0.40, 0.48, 0.50, 100)
    assert side == -1
    assert edge == pytest.approx(100 * (0.48 - 0.40) - ss.fee_c(0.52, 100))


def test_reference_reads_only_finished_bars() -> None:
    cb = [[T0 + 60 * j, 100.0 + j, 101.0 + j, 99.0 + j, 100.5 + j, 1.0] for j in range(200)]
    dv = [[T0 + 3600 * h, 40.0 + h] for h in range(10)]  # hourly DVOL closes
    ref = ss.Reference(cb, dv, CFG)
    t = T0 + 60 * 10
    assert ref.spot(t) == pytest.approx(100.5 + 9)  # the bar that started at t - 60
    assert ref.spot(t + 59) == pytest.approx(100.5 + 9)  # bar 10 has not finished yet
    assert ref.sigma(t, "dvol") is None  # the first hour has not finished
    assert ref.sigma(T0 + 3600, "dvol") == pytest.approx(0.40)
    assert ref.sigma(T0 + 2 * 3600 - 1, "dvol") == pytest.approx(0.40)  # hour 1 still open
    assert ref.sigma(T0 + 2 * 3600, "dvol") == pytest.approx(0.41)
    assert ref.sigma(T0 + 14 * 3600, "dvol") is None  # last close older than 3 hours
    assert ref.minute_average(t) == pytest.approx((100 + 101 + 99 + 100.5) / 4 + 9)
    assert ref.spot(T0) is None  # nothing finished yet
    assert ref.spot(T0 + 60 * 400) is None  # stale beyond 5 minutes
    assert ref.sigma(t, "rv60") is None  # fewer than 30 returns
    assert ref.sigma(T0 + 60 * 100, "rv60") is not None
    with pytest.raises(ValueError, match="unknown"):
        ref.sigma(t, "vix")


def test_realised_vol_recovers_the_path_volatility() -> None:
    rng = random.Random(3)
    p, cb = 50_000.0, []
    sig = 0.6 / math.sqrt(ss.MINUTES_PER_YEAR)
    for j in range(3000):
        p *= math.exp(rng.gauss(0.0, sig))
        cb.append([T0 + 60 * j, p, p, p, p, 1.0])
    ref = ss.Reference(cb, [], CFG)
    vols = [ref.sigma(T0 + 60 * j, "rv60") for j in range(100, 3000, 50)]
    assert np.mean([v for v in vols if v is not None]) == pytest.approx(0.6, rel=0.05)


def test_basis_uses_only_marks_already_published() -> None:
    cb = [[T0 + 60 * j, 100.0, 100.0, 100.0, 100.0, 1.0] for j in range(200)]
    ref = ss.Reference(cb, [], CFG)
    # strike 1% above Coinbase at the first mark, settlement 2% above at the second
    m = _market(0, T0 + 900, 101.0, 102.0)
    basis = ss.Basis([m], ref, n=8)
    assert basis.at(T0 + 899) is None
    assert basis.at(T0 + 900 + 120) == pytest.approx(math.log(1.01))
    # after the close both marks count: median of log(1.01) and log(1.02)
    assert basis.at(T0 + 1800) == pytest.approx((math.log(1.01) + math.log(1.02)) / 2)
    assert basis.stats_bp()["marks"] == 2


def _minutes(
    quotes: dict[int, tuple[float, float]], fair: dict[int, float]
) -> dict[int, ss.Minute]:
    return {
        k: ss.Minute(k=k, end_ts=T0 + 60 * k, bid=b, ask=a, fair={"dvol": fair.get(k)})
        for k, (b, a) in quotes.items()
    }


def test_first_signal_is_traded_at_the_next_minutes_quote() -> None:
    m = _market(0, T0, 100.0, 101.0)  # settles YES
    q = dict.fromkeys(range(1, 16), (0.49, 0.5))
    q[4] = (0.53, 0.54)
    fair = dict.fromkeys(range(2, 14), 0.50)
    fair[3] = 0.60  # 10c above the ask, ~8.25c after the fee
    fair[5] = 0.70  # a later, larger signal is ignored
    minutes = _minutes(q, fair)
    sig = ss.first_signal(minutes, "dvol", 5.0, CFG)
    assert sig is not None
    assert sig[:2] == (3, 1)
    trade = ss.fill_trade(m, minutes, sig, 1, CFG)
    assert trade is not None
    assert trade.fill == pytest.approx(0.54)  # minute 4's ask, not minute 3's
    assert trade.payoff == 1
    assert trade.pnl_c == pytest.approx(100 * (1 - 0.54) - ss.fee_c(0.54, 100))
    same = ss.fill_trade(m, minutes, sig, 0, CFG)
    assert same is not None
    assert same.fill == pytest.approx(0.50)
    assert ss.first_signal(minutes, "dvol", 30.0, CFG) is None


def test_no_side_pays_when_the_market_settles_no() -> None:
    m = _market(0, T0, 100.0, 99.0)  # settles NO
    minutes = _minutes(dict.fromkeys(range(1, 16), (0.6, 0.61)), {2: 0.40})
    sig = ss.first_signal(minutes, "dvol", 0.0, CFG)
    assert sig is not None
    assert sig[1] == -1
    trade = ss.fill_trade(m, minutes, sig, 1, CFG)
    assert trade is not None
    assert trade.fill == pytest.approx(0.40)  # NO bought at 1 - bid
    assert trade.pnl_c == pytest.approx(100 * (1 - 0.40) - ss.fee_c(0.40, 100))


def test_signals_need_a_sane_book_and_a_fill_before_the_last_minute() -> None:
    m = _market(0, T0, 100.0, 101.0)
    q = dict.fromkeys(range(1, 16), (0.0, 0.999))  # empty book
    fair = dict.fromkeys(range(2, 14), 0.5)
    assert ss.first_signal(_minutes(q, fair), "dvol", 0.0, CFG) is None
    q = dict.fromkeys(range(1, 16), (0.49, 0.5))
    fair = {13: 0.70}
    minutes = _minutes(q, fair)
    sig = ss.first_signal(minutes, "dvol", 0.0, CFG)
    assert sig is not None
    assert sig[0] == 13
    assert ss.fill_trade(m, minutes, sig, 1, CFG) is not None  # minute 14 is still tradable
    assert ss.fill_trade(m, minutes, sig, 2, CFG) is None  # minute 15 ends at the close
    out_of_band = _minutes(q, {2: 0.95})
    assert ss.first_signal(out_of_band, "dvol", 0.0, CFG) is None


def _trade(day: str, pnl: float, payoff: int = 1) -> ss.Trade:
    return ss.Trade(
        ticker=f"T{day}{pnl}",
        day=day,
        month=day[:7],
        close_ts=0,
        side=1,
        k=2,
        signal_edge_c=1.0,
        fill=0.5,
        fee_c=1.75,
        payoff=payoff,
        pnl_c=pnl,
    )


def test_trade_stats_cluster_by_day_and_fee_stress() -> None:
    trades = [_trade("2026-01-01", 10.0), _trade("2026-01-01", 20.0), _trade("2026-01-02", -6.0)]
    s = ss.trade_stats(trades)
    mean = 8.0
    resid = {"a": (10 - mean) + (20 - mean), "b": -6 - mean}
    se = math.sqrt(2 / 1 * sum(r * r for r in resid.values())) / 3
    assert s["mean_c"] == pytest.approx(mean)
    assert s["se_c"] == pytest.approx(se)
    assert s["days"] == 2
    assert s["t"] == pytest.approx(mean / se)
    assert ss.trade_stats(trades, fee_mult=1.5)["mean_c"] == pytest.approx(mean - 0.875)
    assert ss.trade_stats([])["mean_c"] is None


def test_top_day_share() -> None:
    assert ss.top_day_share([_trade("d1", 30.0), _trade("d2", 10.0)]) == pytest.approx(0.75)
    assert ss.top_day_share([_trade("d1", -5.0)]) is None


def test_split_is_chronological() -> None:
    ms = [_market(i, T0 + 900 * i, 1.0, 1.0) for i in range(10)][::-1]
    a, b = ss.split_markets(ms, 0.6)
    assert [m.ticker for m in a] == [f"KXBTC15M-T{i:05d}" for i in range(6)]
    assert max(m.close_ts for m in a) < min(m.close_ts for m in b)


def test_ols_hc1_recovers_coefficients() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(2000), rng.normal(size=2000)])
    y = 0.5 + 2.0 * x[:, 1] + rng.normal(scale=0.1, size=2000)
    beta, se = ss.ols_hc1(y, x)
    assert beta == pytest.approx([0.5, 2.0], abs=0.02)
    assert all(0 < s < 0.01 for s in se)


# ----------------------------------------------------------------------------- decision


def _selected(**over: Any) -> dict[str, Any]:
    base = {
        "spec": "dvol|2",
        "in_sample": {"t": 3.0, "n": 900},
        "out_of_sample": {"mean_c": 1.0, "se_c": 0.2, "t": 5.0, "n": 600},
        "oos_fee_stress": {"mean_c": 0.5},
        "oos_delay_stress": {"mean_c": 0.4},
        "oos_top_day_share": 0.1,
        "oos_neighbour_frac": 1.0,
        "oos_months_positive_frac": 1.0,
    }
    return base | over


def test_decision_rule() -> None:
    assert ss.settlement_decision({"selected": None}, CFG)["decision"] == "REJECT"
    assert ss.settlement_decision({"selected": _selected()}, CFG)["decision"] == (
        "FORWARD_PAPER_CANDIDATE"
    )
    weak = _selected(out_of_sample={"mean_c": 0.3, "se_c": 0.2, "t": 1.5, "n": 600})
    assert ss.settlement_decision({"selected": weak}, CFG)["decision"] == "COLLECT_MORE_DATA"
    gate = _selected(oos_delay_stress={"mean_c": -0.1})
    out = ss.settlement_decision({"selected": gate}, CFG)
    assert out["decision"] == "COLLECT_MORE_DATA"
    assert "gate two_minute_fill_positive: FAIL" in out["reasons"]
    loss = _selected(out_of_sample={"mean_c": -0.2, "se_c": 0.1, "t": -2.0, "n": 600})
    assert ss.settlement_decision({"selected": loss}, CFG)["decision"] == "REJECT"


# ----------------------------------------------------------------------------- end to end


def test_mispriced_kalshi_is_found_and_survives_out_of_sample(tmp_path: Path) -> None:
    """Kalshi prices with 120% vol on a 50% vol path; the model (DVOL 50) is right."""
    markets, candles, cb, dv = _world(1200, true_vol=0.5, kalshi_vol=1.2, dvol_pct=50.0)
    res = ss.analyze(markets, candles, cb, dv)
    assert res["data"]["markets_usable"] == 1200
    assert res["data"]["result_matches_settlement_values"] == 1200
    assert abs(res["data"]["basis"]["median_bp"]) < 1e-6
    sel = res["selected"]
    assert sel is not None
    assert sel["out_of_sample"]["mean_c"] > 0
    assert res["decision"]["decision"] != "REJECT"
    # the model beats the mispriced quotes on Brier score and the gap carries information
    d5 = next(x for x in res["diagnostics"] if x["minutes_before_close"] == 5)
    assert d5["brier"]["model_dvol"] < d5["brier"]["kalshi_mid"]
    assert d5["outcome_on_mid_and_model_gap"]["dvol"]["gap_t"] > 2
    paths = ss.write_outputs(res, tmp_path)
    assert json.loads(paths[0].read_text())["decision"]["decision"] == res["decision"]["decision"]
    md = paths[1].read_text()
    assert "Hold-to-settlement study" in md
    assert "Who forecasts settlement better?" in md


def test_wrong_model_against_a_correct_market_is_rejected() -> None:
    """Kalshi prices with the true 50% vol; DVOL claims 90%: the model's 'edges' lose."""
    markets, candles, cb, dv = _world(1200, true_vol=0.5, kalshi_vol=0.5, dvol_pct=90.0)
    res = ss.analyze(markets, candles, cb, dv)
    assert res["decision"]["decision"] == "REJECT"
    d5 = next(x for x in res["diagnostics"] if x["minutes_before_close"] == 5)
    assert d5["brier"]["kalshi_mid"] < d5["brier"]["model_dvol"]
    assert "REJECT" in ss.render_markdown(res)
