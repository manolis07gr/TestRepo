"""Experiment runner: immutable dataset + strategy specs -> stress grid -> manifest.

Every run is reproducible from its manifest (git commit, config hash, dataset spec and
hash, strategy spec/version, seed, latency grid, cost scenarios, partitions, environment).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any

from cma.backtest.core import StaticMappings, Strategy, TradingCore
from cma.backtest.engine import ReplayEngine, ledger_hash
from cma.backtest.metrics import TradingMetrics, compute_metrics
from cma.config import MANDATORY_LATENCY_GRID_MS, AppConfig, CostScenario, config_hash
from cma.domain.enums import ExecutionMode, Venue
from cma.domain.errors import ExperimentPublicationError
from cma.domain.models import (
    ContractMapping,
    MarketEvent,
    PredictionContract,
    Settlement,
)
from cma.domain.time import NS_PER_S
from cma.execution.latency import LatencyModel, LatencyProfile

# ----------------------------------------------------------------------------- datasets


@dataclass
class LoadedDataset:
    name: str
    events: list[MarketEvent]
    contracts: list[PredictionContract]
    mappings: list[ContractMapping]
    settlements: list[Settlement]
    closes: list[tuple[int, str]]
    reference_instruments: frozenset[str]
    dataset_hash: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def start_ns(self) -> int:
        return self.events[0].recv_ts_ns if self.events else 0

    @property
    def end_ns(self) -> int:
        return self.events[-1].recv_ts_ns if self.events else 0

    def window(self, start_ns: int, end_ns: int) -> LoadedDataset:
        """Restrict to contracts that *list and resolve* inside [start, end)."""
        keep = {
            c.contract_id
            for c in self.contracts
            if (c.open_ts_ns or 0) >= start_ns and (c.close_ts_ns or 0) <= end_ns
        }
        yes_inst = {
            c.outcome_instruments.get("YES", c.contract_id)
            for c in self.contracts
            if c.contract_id in keep
        }
        events = [
            e
            for e in self.events
            if start_ns <= e.recv_ts_ns < end_ns
            and (e.instrument_id in self.reference_instruments or e.instrument_id in yes_inst)
        ]
        return LoadedDataset(
            name=f"{self.name}[{start_ns}:{end_ns}]",
            events=events,
            contracts=[c for c in self.contracts if c.contract_id in keep],
            mappings=[m for m in self.mappings if m.contract_id in keep],
            settlements=[s for s in self.settlements if s.contract_id in keep],
            closes=[(t, c) for t, c in self.closes if c in keep],
            reference_instruments=self.reference_instruments,
            dataset_hash=_sha({"parent": self.dataset_hash, "start": start_ns, "end": end_ns}),
            provenance={**self.provenance, "window": [start_ns, end_ns]},
        )


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    kind: str  # "synthetic" | "directory"
    params: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}

    def load(self) -> LoadedDataset:
        if self.kind == "synthetic":
            from cma.research.synthetic import SyntheticMarketConfig, generate_market

            cfg = SyntheticMarketConfig(**self.params)
            m = generate_market(cfg)
            return LoadedDataset(
                name=f"synthetic:{cfg.label()}",
                events=m.events,
                contracts=m.contracts,
                mappings=m.mappings,
                settlements=m.settlements,
                closes=m.closes,
                reference_instruments=m.reference_instruments,
                dataset_hash=_sha(self.to_dict()),
                provenance={"generator": "cma.research.synthetic", "stats": m.stats},
            )
        if self.kind == "directory":
            from cma.storage.datasets import load_dataset_dir

            return load_dataset_dir(Path(str(self.params["path"])))
        raise ValueError(f"unknown dataset kind {self.kind!r}")


# ----------------------------------------------------------------------------- strategies


StrategyBuilder = Callable[[LoadedDataset, Mapping[str, Any]], list[Strategy]]
_STRATEGIES: dict[str, StrategyBuilder] = {}


def register_strategy(name: str) -> Callable[[StrategyBuilder], StrategyBuilder]:
    def deco(fn: StrategyBuilder) -> StrategyBuilder:
        _STRATEGIES[name] = fn
        return fn

    return deco


@register_strategy("fv_taker")
def _build_fv_taker(ds: LoadedDataset, params: Mapping[str, Any]) -> list[Strategy]:
    from cma.signals.strategies.fair_value_taker import (
        FairValueTakerConfig,
        FairValueTakerStrategy,
    )

    maps = {m.contract_id: m for m in ds.mappings}
    pairs = [(c, maps[c.contract_id]) for c in ds.contracts if c.contract_id in maps]
    ref = (
        params.get("reference_instrument")
        or sorted(r for r in ds.reference_instruments if not r.endswith("DVOL"))[0]
    )
    kwargs = {k: v for k, v in params.items() if k != "reference_instrument"}
    for key in ("uncertainty_bps", "target_quantity"):
        if key in kwargs:
            kwargs[key] = Decimal(str(kwargs[key]))
    return [FairValueTakerStrategy(FairValueTakerConfig(reference_instrument=ref, **kwargs), pairs)]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    version: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def build(self, ds: LoadedDataset) -> list[Strategy]:
        try:
            builder = _STRATEGIES[self.name]
        except KeyError as exc:
            raise KeyError(f"unknown strategy {self.name!r}; known {sorted(_STRATEGIES)}") from exc
        return builder(ds, self.params)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "params": dict(self.params)}


# ----------------------------------------------------------------------------- runs


@dataclass
class RunResult:
    latency_ms: int
    cost: str
    metrics: TradingMetrics
    ledger_hash: str
    n_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "ledger_hash": self.ledger_hash,
            "n_events": self.n_events,
            "metrics": self.metrics.to_dict(),
        }


def run_single(
    ds: LoadedDataset,
    strategy: StrategySpec,
    cfg: AppConfig,
    *,
    latency_ms: int,
    cost: CostScenario,
    seed: int,
    venue_taker_delay_ms: Mapping[Venue, int] | None = None,
) -> tuple[RunResult, TradingCore]:
    strategies = strategy.build(ds)
    profile = LatencyProfile.from_config(cfg.simulation, latency_ms + cost.extra_latency_ms)

    def factory(engine: ReplayEngine) -> TradingCore:
        return TradingCore(
            config=cfg,
            strategies=strategies,
            contracts={c.contract_id: c for c in ds.contracts},
            mappings=StaticMappings({m.contract_id: m for m in ds.mappings}),
            latency=LatencyModel(profile, seed=seed),
            scheduler=engine,
            mode=ExecutionMode.BACKTEST,
            reference_instruments=ds.reference_instruments,
            fee_multiplier=cost.fee_multiplier,
            stress_slippage_ticks=cost.extra_slippage_ticks,
            venue_taker_delay_ms=venue_taker_delay_ms,
        )

    engine = ReplayEngine(
        factory,
        settlements=ds.settlements,
        closes=ds.closes,
        extra_feed_ns=round(cfg.simulation.extra_feed_latency_ms * 1e6),
    )
    engine.run(ds.events)
    core = engine.core
    result = RunResult(
        latency_ms=latency_ms,
        cost=cost.name,
        metrics=compute_metrics(core),
        ledger_hash=ledger_hash(core.records.fills),
        n_events=len(ds.events),
    )
    return result, core


@dataclass
class StressGrid:
    results: list[RunResult]

    def get(self, latency_ms: int, cost: str) -> RunResult:
        for r in self.results:
            if r.latency_ms == latency_ms and r.cost == cost:
                return r
        raise KeyError((latency_ms, cost))

    def latencies(self) -> list[int]:
        return sorted({r.latency_ms for r in self.results})

    def costs(self) -> list[str]:
        return list(dict.fromkeys(r.cost for r in self.results))

    def table(self, field_name: str = "net_pnl") -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for r in self.results:
            out.setdefault(r.cost, {})[r.latency_ms] = float(getattr(r.metrics, field_name))
        return out


def run_stress_grid(
    ds: LoadedDataset,
    strategy: StrategySpec,
    cfg: AppConfig,
    *,
    latencies_ms: Sequence[int] | None = None,
    costs: Sequence[CostScenario] | None = None,
    seed: int | None = None,
    venue_taker_delay_ms: Mapping[Venue, int] | None = None,
) -> StressGrid:
    lats = list(latencies_ms) if latencies_ms is not None else cfg.simulation.stress_grid()
    scenarios = list(costs) if costs is not None else list(cfg.research.cost_scenarios)
    results = []
    for cost in scenarios:
        for lat in lats:
            r, _ = run_single(
                ds,
                strategy,
                cfg,
                latency_ms=lat,
                cost=cost,
                seed=cfg.simulation.seed if seed is None else seed,
                venue_taker_delay_ms=venue_taker_delay_ms,
            )
            results.append(r)
    return StressGrid(results)


def validate_publication(grid: StressGrid, manifest: Mapping[str, Any]) -> None:
    """T049/T050/T055: refuse to publish an incomplete experiment."""
    missing = sorted(set(MANDATORY_LATENCY_GRID_MS) - set(grid.latencies()))
    if missing:
        raise ExperimentPublicationError(f"missing mandated latency scenarios (ms): {missing}")
    costs = grid.costs()
    if "base" not in costs or len([c for c in costs if c != "base"]) < 2:
        raise ExperimentPublicationError(
            f"cost stress requires base + >=2 adverse scenarios, got {costs}"
        )
    for c in costs:
        have = {r.latency_ms for r in grid.results if r.cost == c}
        if c == "base" and not set(MANDATORY_LATENCY_GRID_MS) <= have:
            raise ExperimentPublicationError("base cost scenario lacks the full latency grid")
    required = ("git_commit", "config_hash", "dataset", "dataset_hash", "strategy", "seed")
    absent = [k for k in required if not manifest.get(k)]
    if absent:
        raise ExperimentPublicationError(f"manifest missing {absent}")


# ----------------------------------------------------------------------------- manifest


def git_commit(repo: Path | None = None) -> str:
    root = repo or Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_info() -> dict[str, str]:
    pkgs = {}
    for name in ("numpy", "pandas", "scipy", "pyarrow", "pydantic"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = "absent"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        **pkgs,
    }


def build_manifest(
    *,
    experiment_id: str,
    created_ns: int,
    cfg: AppConfig,
    dataset: DatasetSpec,
    dataset_hash: str,
    strategy: StrategySpec,
    seed: int,
    grid: StressGrid | None = None,
    partitions: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "created_ns": created_ns,
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config": json.loads(cfg.model_dump_json()),
        "dataset": dataset.to_dict(),
        "dataset_hash": dataset_hash,
        "strategy": strategy.to_dict(),
        "seed": seed,
        "latency_grid_ms": grid.latencies() if grid else cfg.simulation.stress_grid(),
        "cost_scenarios": [c.model_dump(mode="json") for c in cfg.research.cost_scenarios],
        "ledger_hashes": (
            {f"{r.cost}@{r.latency_ms}": r.ledger_hash for r in grid.results} if grid else {}
        ),
        "partitions": dict(partitions or {}),
        "environment": environment_info(),
        "reproduce": f"cma reproduce --manifest <path-to-this-file> (experiment {experiment_id})",
        **dict(extra or {}),
    }


def reproduce_from_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    """Re-run every grid cell recorded in the manifest and compare ledger hashes."""
    cfg = AppConfig.model_validate(manifest["config"])
    ds_spec = DatasetSpec(manifest["dataset"]["kind"], manifest["dataset"]["params"])
    strat = StrategySpec(
        manifest["strategy"]["name"],
        manifest["strategy"]["version"],
        manifest["strategy"]["params"],
    )
    ds = ds_spec.load()
    window = manifest.get("partitions", {}).get("evaluated_window")
    if window:
        ds = ds.window(int(window[0]), int(window[1]))
    taker_delay = {Venue(k): int(v) for k, v in manifest.get("venue_taker_delay_ms", {}).items()}
    out = {}
    scenarios = {c.name: c for c in cfg.research.cost_scenarios}
    for key, expected in manifest["ledger_hashes"].items():
        cost_name, lat = key.rsplit("@", 1)
        r, _ = run_single(
            ds,
            strat,
            cfg,
            latency_ms=int(lat),
            cost=scenarios[cost_name],
            seed=int(manifest["seed"]),
            venue_taker_delay_ms=taker_delay,
        )
        out[key] = r.ledger_hash == expected
    return out


def default_partitions(start_ns: int, end_ns: int, cfg: AppConfig) -> dict[str, list[int]]:
    """Chronological train / validation / final-test boundaries (hour-aligned)."""
    span = end_ns - start_ns
    hour = 3600 * NS_PER_S
    test_start = start_ns + round(span * (1 - cfg.research.final_test_fraction) / hour) * hour
    val_start = test_start - round(span * cfg.research.validation_fraction / hour) * hour
    return {
        "train": [start_ns, val_start],
        "validation": [val_start, test_start],
        "final_test": [test_start, end_ns],
    }
