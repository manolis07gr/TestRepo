"""Engineering benchmarks for scope s.22 (with hardware + dataset metadata).

* ingestion: raw Kalshi/Coinbase WS payloads -> adapter.parse -> Normalizer (events/sec)
* per-event normalization latency p99
* feature + signal hot path p99 (fair-value strategy + signal engine per observation)
* replay speed vs real time on a synthetic dataset

Usage: python scripts/benchmark.py [--hours 2] [--out docs/benchmarks.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from cma.adapters.crypto.coinbase import CoinbaseAdapter
from cma.adapters.kalshi.parser import KalshiAdapter
from cma.backtest.experiment import DatasetSpec, StrategySpec, run_single
from cma.config import AppConfig, CostScenario, DataQualityConfig
from cma.domain.enums import Venue
from cma.domain.models import RawMessage
from cma.domain.time import NS_PER_S, ManualClock, iso_from_ns
from cma.normalization.pipeline import Normalizer


def _payloads(n: int) -> list[RawMessage]:
    t0 = 1_790_000_000 * NS_PER_S
    out = []
    for i in range(n):
        if i % 2:
            payload = json.dumps(
                {
                    "type": "orderbook_delta",
                    "sid": 1,
                    "seq": i,
                    "msg": {
                        "market_ticker": "KXBTCD-BENCH-T110000",
                        "price_dollars": "0.4500",
                        "delta_fp": "1.00" if i % 4 == 1 else "-1.00",
                        "side": "yes",
                        "ts_ms": (t0 + i * 1_000_000) // 1_000_000,
                    },
                }
            )
            venue = Venue.KALSHI
        else:
            px = 110_000 + (i % 50)
            payload = json.dumps(
                {
                    "type": "ticker",
                    "sequence": i,
                    "product_id": "BTC-USD",
                    "price": f"{px}.00",
                    "best_bid": f"{px - 1}.00",
                    "best_bid_size": "0.5",
                    "best_ask": f"{px + 1}.00",
                    "best_ask_size": "0.5",
                    "side": "buy",
                    "time": iso_from_ns(t0 + i * 1_000_000),
                    "trade_id": i,
                    "last_size": "0.01",
                }
            )
            venue = Venue.COINBASE
        out.append(
            RawMessage(
                venue=venue,
                stream="ws",
                recv_ts_ns=t0 + i * 1_000_000 + 5,
                payload=payload,
                connection_id=venue.value,
                connection_seq=i,
            )
        )
    return out


def bench_ingestion(n: int = 50_000) -> dict[str, float]:
    adapters = {Venue.KALSHI: KalshiAdapter(), Venue.COINBASE: CoinbaseAdapter()}
    norm = Normalizer(DataQualityConfig(), ManualClock(1))
    raws = _payloads(n)
    per_event = []
    t0 = time.perf_counter()
    count = 0
    for raw in raws:
        s = time.perf_counter_ns()
        for ev in adapters[raw.venue].parse(raw):
            if norm.process(ev) is not None:
                count += 1
        per_event.append(time.perf_counter_ns() - s)
    dt = time.perf_counter() - t0
    arr = np.asarray(per_event) / 1e6
    return {
        "messages": n,
        "events": count,
        "events_per_s": count / dt,
        "normalize_p50_ms": float(np.percentile(arr, 50)),
        "normalize_p99_ms": float(np.percentile(arr, 99)),
    }


def bench_replay(hours: int) -> dict[str, float]:
    params = {"hours": hours, "seed": 3, "mm_lag_ms": 350.0, "competitor_latency_ms": 120.0}
    t = time.perf_counter()
    ds = DatasetSpec("synthetic", params).load()
    gen_s = time.perf_counter() - t
    cfg = AppConfig.model_validate({"mode": "BACKTEST", "signal": {"ttl_ms": 10_000}})
    strat = StrategySpec("fv_taker", "1.0", {"vol_instrument": "DERIBIT:BTC-DVOL"})
    t = time.perf_counter()
    _r, core = run_single(ds, strat, cfg, latency_ms=250, cost=CostScenario(name="base"), seed=1)
    wall = time.perf_counter() - t
    sim_span_s = (ds.events[-1].recv_ts_ns - ds.events[0].recv_ts_ns) / NS_PER_S

    # hot path: strategy + signal engine per observation, timed on a replayed core
    strategy = core.strategies[0]
    samples = []
    from cma.backtest.core import StrategyContext

    for ev in ds.events[-20_000:]:
        ctx = StrategyContext(now_ns=ev.recv_ts_ns, core=core)
        s = time.perf_counter_ns()
        outs = strategy.on_observation(ev, ctx)
        for out in outs:
            book = core.state.book(out.instrument_id, 5)
            core.signal_engine.evaluate(out, book, decision_ts_ns=ev.recv_ts_ns)
        samples.append(time.perf_counter_ns() - s)
    arr = np.asarray(samples) / 1e6
    return {
        "hours": hours,
        "events": len(ds.events),
        "generate_s": gen_s,
        "replay_wall_s": wall,
        "replay_events_per_s": len(ds.events) / wall,
        "replay_x_realtime": sim_span_s / wall,
        "hot_path_p50_ms": float(np.percentile(arr, 50)),
        "hot_path_p99_ms": float(np.percentile(arr, 99)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=2)
    ap.add_argument("--out", default="docs/benchmarks.json")
    args = ap.parse_args()
    result = {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "python": sys.version.split()[0],
            "processor": platform.processor() or "unknown",
        },
        "targets": {
            "events_per_s": 10_000,
            "normalize_p99_ms": 5,
            "hot_path_p99_ms": 25,
            "replay_x_realtime": 10,
        },
        "ingestion": bench_ingestion(),
        "replay": bench_replay(args.hours),
        "dataset": "synthetic market (seed 3, maker lag 350 ms, competitor 120 ms); "
        "ingestion payloads: alternating Kalshi orderbook_delta / Coinbase ticker",
        "decimal_note": str(Decimal("0.1") + Decimal("0.2")),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
