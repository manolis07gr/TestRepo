"""Raw store -> normalized dataset dir -> replay: the real-data path, end to end, offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cma.backtest.experiment import DatasetSpec, StrategySpec, run_single
from cma.config import AppConfig, CostScenario, DataQualityConfig
from cma.domain.enums import Venue
from cma.domain.models import RawMessage
from cma.domain.time import ManualClock
from cma.normalization.pipeline import Normalizer
from cma.storage.datasets import load_dataset_dir, save_dataset_dir
from cma.storage.normalize import normalize_raw
from cma.storage.raw import RawRecorder

pytestmark = pytest.mark.integration
FIX = Path(__file__).resolve().parents[1] / "fixtures" / "adapters"


def test_raw_capture_normalizes_through_the_live_adapters(tmp_path: Path) -> None:
    rec = RawRecorder(tmp_path / "raw")
    t = 1_790_000_000_000_000_000
    payloads = [
        (Venue.KALSHI, "ws", (FIX / "kalshi" / "ws_orderbook_snapshot.json").read_text()),
        (Venue.KALSHI, "ws", (FIX / "kalshi" / "ws_orderbook_delta_yes.json").read_text()),
        (Venue.COINBASE, "ws", (FIX / "coinbase" / "ws_ticker.json").read_text()),
        (Venue.KALSHI, "ws", "{not json"),
    ]
    for i, (venue, stream, payload) in enumerate(payloads):
        rec.append(
            RawMessage(venue=venue, stream=stream, recv_ts_ns=t + i, payload=payload,
                       connection_id=f"c-{venue.value}", connection_seq=i)
        )
    rec.close()
    errors: list[Exception] = []
    events = normalize_raw(
        tmp_path / "raw",
        None,
        Normalizer(DataQualityConfig(), ManualClock(t + 100)),
        on_error=errors.append,
    )
    assert len(errors) == 1  # malformed payload skipped, not fatal
    assert events and all(e.process_ts_ns == t + 100 for e in events)
    assert events == sorted(events, key=lambda e: (e.recv_ts_ns, e.venue_ts_ns))
    assert {e.venue for e in events} == {Venue.KALSHI, Venue.COINBASE}


def test_dataset_dir_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    spec = DatasetSpec("synthetic", {"hours": 1, "n_strikes": 3, "seed": 9})
    ds = spec.load()
    digest = save_dataset_dir(ds, tmp_path / "ds")
    back = load_dataset_dir(tmp_path / "ds")
    assert back.dataset_hash == digest
    assert len(back.events) == len(ds.events)
    assert back.contracts == ds.contracts
    assert [e.event_id for e in back.events] == [e.event_id for e in ds.events]
    assert [m.contract_id for m in back.mappings] == [m.contract_id for m in ds.mappings]
    assert back.settlements == ds.settlements
    cfg = AppConfig.model_validate({"mode": "BACKTEST", "signal": {"min_net_edge_bps": 50}})
    strat = StrategySpec("fv_taker", "1.0", {"vol_instrument": "DERIBIT:BTC-DVOL"})
    a, _ = run_single(ds, strat, cfg, latency_ms=250, cost=CostScenario(name="base"), seed=1)
    b, _ = run_single(back, strat, cfg, latency_ms=250, cost=CostScenario(name="base"), seed=1)
    assert a.ledger_hash == b.ledger_hash  # derived dataset replays identically
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    (tmp_path / "ds" / "settlements.json").write_text("[]")
    with pytest.raises(ValueError, match="does not match"):
        load_dataset_dir(tmp_path / "ds")
    assert manifest["dataset_hash"] == digest
