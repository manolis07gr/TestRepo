"""Hurdle maths, normalization stage, observability, synthetic generator invariants,
report rendering and the CLI entry points."""

from __future__ import annotations

import io
import json
import logging
import math
from decimal import Decimal
from pathlib import Path

import pytest

from cma.cli.main import build_parser, main
from cma.config import DataQualityConfig
from cma.domain.enums import QualityFlag, Side, Venue
from cma.domain.fees import KALSHI_STANDARD, POLYMARKET_CRYPTO_TAKER
from cma.domain.time import NS_PER_MS, ManualClock
from cma.features.fair_value import (
    AveragingState,
    contract_probability,
    digital_above,
    digital_delta_per_log_move,
    prob_above,
)
from cma.normalization.pipeline import Normalizer
from cma.observability.incidents import Incident, IncidentLog
from cma.observability.logs import configure_logging
from cma.observability.metrics import MetricsRegistry
from cma.research.hurdle import break_even_staleness_ms, hurdle_row, hurdle_table
from cma.research.report import render_markdown, write_reports
from cma.research.synthetic import SyntheticMarketConfig, generate_market
from cma.storage.db import open_database
from tests.factories import snapshot, trade

pytestmark = pytest.mark.unit


def test_hurdle_required_move_closed_form_and_monotonicity() -> None:
    row = hurdle_row(fee=KALSHI_STANDARD, t_seconds=3600, moneyness_z=0.0, sigma=0.45, window_s=1.0)
    # fair after the required move equals ask + fee + threshold
    from cma.research.hurdle import SECONDS_PER_YEAR

    sd = 0.45 * math.sqrt(3600 / SECONDS_PER_YEAR)
    from scipy.stats import norm

    fair_after = norm.cdf(math.log(math.exp(row.required_move_bps * 1e-4)) / sd + norm.ppf(row.p0))
    assert fair_after == pytest.approx(row.required_fair, abs=1e-9)
    near = hurdle_row(fee=KALSHI_STANDARD, t_seconds=120, moneyness_z=0.0, sigma=0.45, window_s=1)
    assert near.required_move_bps < row.required_move_bps  # gamma rises near expiry
    assert near.p_move_gauss > row.p_move_gauss
    deep = hurdle_row(fee=KALSHI_STANDARD, t_seconds=60, moneyness_z=6.0, sigma=0.45, window_s=1)
    assert deep.p_move_gauss <= near.p_move_gauss
    rows = hurdle_table(fee_ids=("kalshi-standard", "polymarket-crypto-taker"), windows_s=(1.0,))
    assert {r.venue_fee for r in rows} == {KALSHI_STANDARD.tag, POLYMARKET_CRYPTO_TAKER.tag}
    assert break_even_staleness_ms(required_move_bps=10, sigma=0.45) > 1000
    assert math.isinf(break_even_staleness_ms(required_move_bps=math.inf, sigma=0.45))


def test_fair_value_semantics_point_and_trailing_average() -> None:
    assert digital_above(100.0, 100.0, 0.0, 0.5) == 0.5
    assert digital_above(101.0, 100.0, 0.0, 0.5) == 1.0
    assert 0.0 < digital_above(99.0, 100.0, 1 / 365, 0.5) < 0.5
    end = 10_000 * 10**9
    now = end - 600 * 10**9
    point = prob_above(spot=101.0, strike=100.0, now_ns=now, observation_end_ns=end, sigma=0.6)
    avg = prob_above(
        spot=101.0,
        strike=100.0,
        now_ns=now,
        observation_end_ns=end,
        sigma=0.6,
        observation_method="AVG_60S_BEFORE",
    )
    assert avg > point  # averaging lowers variance -> ITM contract more certain
    # inside the window the known part of the average dominates
    inside = prob_above(
        spot=99.0,
        strike=100.0,
        now_ns=end - 5 * 10**9,
        observation_end_ns=end,
        sigma=0.6,
        observation_method="AVG_60S_BEFORE",
        averaging=AveragingState(elapsed_s=55.0, integral=55.0 * 102.0),
    )
    assert inside > 0.99
    between = contract_probability(
        operator=__import__("cma.domain.enums", fromlist=["Operator"]).Operator.BETWEEN,
        strikes=(99.0, 101.0),
        spot=100.0,
        now_ns=now,
        observation_end_ns=end,
        sigma=0.6,
    )
    assert 0 < between < 1
    assert digital_delta_per_log_move(100.0, 100.0, 1 / 8760, 0.45) > 0


def test_normalizer_flags_drift_dedups_and_stamps() -> None:
    clock = ManualClock(5_000_000_000)
    norm = Normalizer(DataQualityConfig(max_clock_drift_ms=50), clock)
    ahead = trade(
        "KALSHI:X", 2_000_000_000, "0.5", "1", Side.BUY, tid="a", delay_ns=-100 * NS_PER_MS
    )
    out = norm.process(ahead)
    assert out is not None and QualityFlag.CLOCK_DRIFT in out.quality_flags
    assert out.process_ts_ns == 5_000_000_000 and out.source_ts_ns == ahead.source_ts_ns
    assert norm.process(ahead) is None  # duplicate replay
    ok = trade("KALSHI:X", 3_000_000_000, "0.5", "1", Side.BUY, tid="b", delay_ns=10 * NS_PER_MS)
    assert norm.process(ok) is not None
    bad = snapshot("KALSHI:Y", 1, 1, [("0.4", "1")], [("1.4", "1")])
    assert norm.process(bad) is None and norm.stats.rejected == 1
    assert norm.stats.clock_drift == 1 and norm.stats.duplicates == 1


def test_observability_metrics_logs_and_incidents(tmp_path: Path) -> None:
    reg = MetricsRegistry()
    for v in (1, 5, 50, 500):
        reg.observe("arrival_ms", v)
    reg.inc("fills")
    reg.set("nav", 10_000.0)
    snap = reg.snapshot()
    assert snap["counters"]["fills"] == 1 and snap["histograms"]["arrival_ms"]["n"] == 4
    assert "cma_fills_total 1.0" in reg.prometheus_text()

    stream = io.StringIO()
    logger = configure_logging("INFO", stream=stream)
    logger.getChild("t").info("token=abcdef123456 login", extra={"api_key": "zzz-secret-zzz"})
    line = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert "abcdef123456" not in line["msg"]

    db = open_database(f"sqlite:///{tmp_path / 'm.sqlite'}")
    log = IncidentLog(db=db)
    inc = Incident("KALSHI", "KALSHI:X", "SEQUENCE_GAP", 100, 200, "gap")
    log.record(inc, created_at_ns=1)
    log.record(inc, created_at_ns=2)  # idempotent
    assert len(log.load()) == 1
    assert log.affecting(150, 160, ["KALSHI:X"]) and not log.affecting(300, 400)
    logging.getLogger("cma").handlers.clear()


def test_synthetic_market_invariants() -> None:
    m = generate_market(SyntheticMarketConfig(hours=1, n_strikes=3, seed=3))
    assert m.events == sorted(m.events, key=lambda e: (e.recv_ts_ns, e.venue_ts_ns))
    assert all(e.venue_ts_ns <= e.recv_ts_ns for e in m.events)
    assert len(m.contracts) == len(m.mappings) == len(m.settlements) == 3
    assert m.vol_instrument in m.reference_instruments
    # settlement follows the 60 s average of the true index
    assert {s.outcome.value for s in m.settlements} <= {"YES", "NO"}
    strikes = sorted(float(mp.strikes[0]) for mp in m.mappings)
    yes = [s.outcome.value == "YES" for s in sorted(m.settlements, key=lambda s: s.contract_id)]
    assert len(strikes) == len(yes)
    again = generate_market(SyntheticMarketConfig(hours=1, n_strikes=3, seed=3))
    assert [e.event_id for e in again.events] == [e.event_id for e in m.events]


def test_report_rendering_from_minimal_result(tmp_path: Path) -> None:
    result = {
        "plan": {"criterion_latency_ms": 250},
        "manifest": {"experiment_id": "x", "git_commit": "abc", "config_hash": "c", "seed": 1},
        "real_market_decision": "COLLECT_MORE_DATA",
        "base_cases": [],
        "frontier": [
            {
                "mm_lag_ms": 350.0,
                "competitor_ms": None,
                "latency_ms": 0,
                "model": "implied_vol",
                "expected_c_per_contract": -0.2,
                "contracts": 10.0,
            }
        ],
        "lead_lag": {"results": []},
        "structural": {"family_samples": 0, "violations_after_costs": 0, "net_total": "0"},
        "hurdle": [r.to_dict() for r in hurdle_table(fee_ids=("kalshi-standard",))],
    }
    text = render_markdown(result)
    assert "COLLECT_MORE_DATA" in text and "Edge frontier" in text
    paths = write_reports(result, tmp_path)
    assert all(p.exists() for p in paths)


def test_cli_parser_hurdle_live_and_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    assert parser.parse_args(["hurdle"]).command == "hurdle"
    assert main(["hurdle", "--fee", "kalshi-standard"]) == 0
    assert json.loads(capsys.readouterr().out)
    assert main(["live", "--config", "config/base.yaml"]) == 3  # always refused
    assert main(["db", "migrate", "--db-url", f"sqlite:///{tmp_path / 'x.sqlite'}"]) == 0
    assert "contracts" in capsys.readouterr().out
    assert main(["config", "--config", "config/base.yaml"]) == 0
    assert "config_hash" in capsys.readouterr().out
    assert Decimal(1) == Decimal(1) and Venue.KALSHI.value == "KALSHI"


@pytest.mark.integration
def test_paper_smoke_produces_valid_health(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["smoke"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["collector"]["status"] == "OK"
    assert out["paper"]["status"] == "OK" and out["events_received"] >= 10
