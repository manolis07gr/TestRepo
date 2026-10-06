"""``cma`` command-line interface.

Research workflow::

    cma collect   --config config/base.yaml --duration 3600     # record raw venue data
    cma dataset build --raw data/raw --db-url sqlite:///data/cma.sqlite --out data/ds/x
    cma stress    --dataset data/ds/x --out reports/runs/x      # latency x cost grid + manifest
    cma reproduce --manifest reports/runs/x/manifest.json       # one-command re-run
    cma evaluate  --out reports/edge_evaluation                 # full synthetic edge study
    cma smoke                                                   # paper smoke test on mocks

Live order routing does not exist in v1: ``cma live`` always refuses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from cma.config import AppConfig, CostScenario, load_config
from cma.domain.time import SystemClock, iso_from_ns
from cma.security import redact_mapping

DEFAULT_CONFIGS = ("config/base.yaml",)


def _load_cfg(paths: Sequence[str] | None, **overrides: Any) -> AppConfig:
    files = list(paths or [p for p in DEFAULT_CONFIGS if Path(p).exists()])
    return load_config(*files, overrides=overrides or None) if files else AppConfig()


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))


# ----------------------------------------------------------------------------- commands


def cmd_config_show(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    _print({"config_hash": cfg.hash(), "config": redact_mapping(cfg.model_dump(mode="json"))})
    return 0


def cmd_db_migrate(args: argparse.Namespace) -> int:
    from cma.storage.db import Database, migrate

    db = Database(args.db_url)
    applied = migrate(db)
    _print({"applied": applied, "tables": sorted(db.table_names())})
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from cma.ingestion.collector import build_collector
    from cma.storage.db import open_database

    cfg = _load_cfg(args.config)
    db = open_database(args.db_url or cfg.storage.db_url)
    collector = build_collector(cfg, db=db, raw_root=args.raw_dir or cfg.storage.raw_dir)

    async def main() -> None:
        await collector.start()
        try:
            elapsed = 0.0
            while args.duration is None or elapsed < args.duration:
                await asyncio.sleep(min(args.report_every, 1e9))
                elapsed += args.report_every
                _print(collector.health_report())
        finally:
            await collector.stop()
            await collector.aclose()

    asyncio.run(main())
    return 0


def cmd_dataset_build(args: argparse.Namespace) -> int:
    from cma.backtest.experiment import LoadedDataset
    from cma.domain.models import Settlement
    from cma.mapping.registry import MappingRegistry
    from cma.normalization.pipeline import Normalizer
    from cma.storage.contracts import ContractStore
    from cma.storage.datasets import save_dataset_dir, settlement_from_dict
    from cma.storage.db import open_database
    from cma.storage.normalize import normalize_raw

    cfg = _load_cfg(args.config)
    db = open_database(args.db_url or cfg.storage.db_url)
    contracts = ContractStore(db).load()
    registry = MappingRegistry.from_yaml(Path(args.mappings_dir))
    mappings = [
        m
        for c in contracts
        if (m := registry.get(c.contract_id)) is not None and m.review_status.rank >= 2
    ]
    keep = {m.contract_id for m in mappings}
    contracts = [c for c in contracts if c.contract_id in keep]
    refs = frozenset(args.reference)
    wanted = refs | {c.outcome_instruments.get("YES", c.contract_id) for c in contracts}
    normalizer = Normalizer(cfg.data_quality, SystemClock())
    events = normalize_raw(Path(args.raw), wanted, normalizer)
    settlements: list[Settlement] = []
    if args.settlements:
        settlements = [
            settlement_from_dict(d) for d in json.loads(Path(args.settlements).read_text())
        ]
    ds = LoadedDataset(
        name=args.name or Path(args.out).name,
        events=events,
        contracts=contracts,
        mappings=mappings,
        settlements=settlements,
        closes=[(c.close_ts_ns, c.contract_id) for c in contracts if c.close_ts_ns],
        reference_instruments=refs,
        dataset_hash="",
        provenance={"raw_root": str(args.raw), "normalizer": vars(normalizer.stats)},
    )
    digest = save_dataset_dir(ds, Path(args.out))
    _print({"dataset_hash": digest, "events": len(events), "contracts": len(contracts)})
    return 0


def cmd_dataset_synth(args: argparse.Namespace) -> int:
    from cma.backtest.experiment import DatasetSpec
    from cma.storage.datasets import save_dataset_dir

    params = {
        "hours": args.hours,
        "seed": args.seed,
        "mm_lag_ms": args.mm_lag_ms,
        "competitor_latency_ms": args.competitor_ms,
    }
    ds = DatasetSpec("synthetic", params).load()
    digest = save_dataset_dir(ds, Path(args.out), provenance={"synthetic_params": params})
    _print({"dataset_hash": digest, "events": len(ds.events), "contracts": len(ds.contracts)})
    return 0


def _dataset_spec(arg: str) -> Any:
    from cma.backtest.experiment import DatasetSpec

    if arg.startswith("synthetic:"):
        return DatasetSpec("synthetic", json.loads(arg.removeprefix("synthetic:") or "{}"))
    return DatasetSpec("directory", {"path": arg})


def cmd_backtest(args: argparse.Namespace) -> int:
    from cma.backtest.experiment import StrategySpec, run_single

    cfg = _load_cfg(args.config, mode="BACKTEST")
    ds = _dataset_spec(args.dataset).load()
    strat = StrategySpec(args.strategy, args.strategy_version, json.loads(args.params))
    cost = next((c for c in cfg.research.cost_scenarios if c.name == args.cost), None)
    if cost is None:
        cost = CostScenario(name=args.cost)
    r, _core = run_single(ds, strat, cfg, latency_ms=args.latency_ms, cost=cost, seed=args.seed)
    _print(r.to_dict())
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    from cma.backtest.experiment import (
        StrategySpec,
        build_manifest,
        run_stress_grid,
        validate_publication,
    )

    cfg = _load_cfg(args.config, mode="BACKTEST")
    spec = _dataset_spec(args.dataset)
    ds = spec.load()
    strat = StrategySpec(args.strategy, args.strategy_version, json.loads(args.params))
    grid = run_stress_grid(ds, strat, cfg, seed=args.seed)
    manifest = build_manifest(
        experiment_id=args.experiment_id or f"stress-{ds.dataset_hash[:10]}",
        created_ns=SystemClock().now_ns(),
        cfg=cfg,
        dataset=spec,
        dataset_hash=ds.dataset_hash,
        strategy=strat,
        seed=args.seed,
        grid=grid,
    )
    validate_publication(grid, manifest)  # refuses incomplete experiments (T049/T050/T055)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (out / "grid.json").write_text(
        json.dumps([r.to_dict() for r in grid.results], indent=1, default=str)
    )
    _print({"out": str(out), "net_pnl": grid.table("net_pnl")})
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    from cma.backtest.experiment import reproduce_from_manifest

    manifest = json.loads(Path(args.manifest).read_text())
    result = reproduce_from_manifest(manifest)
    _print({"identical": all(result.values()), "cells": result})
    return 0 if all(result.values()) else 1


def cmd_leadlag(args: argparse.Namespace) -> int:
    from cma.models.lead_lag.discovery import LeadLagConfig, discover_lead_lag
    from cma.research.evaluation import mid_series

    ds = _dataset_spec(args.dataset).load()
    x_ts, x_v = mid_series(ds.events, args.x)
    y_ts, y_v = mid_series(ds.events, args.y)
    res = discover_lead_lag(
        x_ts, x_v, y_ts, y_v, LeadLagConfig(max_lag_ms=args.max_lag_ms, cost_hurdle=args.hurdle)
    )
    _print(res.to_dict())
    return 0


def cmd_hurdle(args: argparse.Namespace) -> int:
    from cma.research.hurdle import hurdle_table

    rows = hurdle_table(fee_ids=tuple(args.fee), sigmas=(args.sigma,))
    _print([r.to_dict() for r in rows])
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from cma.research.evaluation import EvaluationPlan, run_evaluation
    from cma.research.report import write_reports

    plan = EvaluationPlan.quick() if args.quick else EvaluationPlan()
    if args.workers:
        plan = EvaluationPlan(**{**plan.__dict__, "workers": args.workers})
    out = Path(args.out)
    result = run_evaluation(out, plan)
    paths = write_reports(result, out)
    _print({"out": str(out), "reports": [str(p) for p in paths], "timings": result["timings_s"]})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from cma.research.report import write_reports

    result = json.loads(Path(args.evaluation).read_text())
    paths = write_reports(result, Path(args.out))
    _print({"reports": [str(p) for p in paths]})
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Paper-mode smoke test: collectors on scripted mock feeds -> valid health status."""
    from cma.research.smoke import run_smoke

    report = run_smoke()
    _print(report)
    return 0 if report["collector"]["status"] in ("OK", "DEGRADED") and report["paper"] else 1


def cmd_paper(args: argparse.Namespace) -> int:
    from cma.research.paper_run import run_paper

    cfg = _load_cfg(args.config)
    report = run_paper(cfg, duration_s=args.duration, mappings_dir=Path(args.mappings_dir))
    _print(report)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from cma.domain.errors import LiveTradingDisabledError
    from cma.execution.live.adapter import LiveExecutionAdapter

    cfg = _load_cfg(args.config)
    try:
        LiveExecutionAdapter(cfg)
    except LiveTradingDisabledError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    return 4  # pragma: no cover - unreachable in v1


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cma", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--config", nargs="*", help="YAML files merged left to right")
        return sp

    s = with_config(sub.add_parser("config", help="show the merged, redacted configuration"))
    s.set_defaults(func=cmd_config_show)

    s = sub.add_parser("db", help="metadata database")
    s.add_argument("action", choices=["migrate"])
    s.add_argument("--db-url", default="sqlite:///data/cma.sqlite")
    s.set_defaults(func=cmd_db_migrate)

    s = with_config(sub.add_parser("collect", help="record raw venue data (needs network)"))
    s.add_argument("--duration", type=float, default=None, help="seconds (default: forever)")
    s.add_argument("--raw-dir")
    s.add_argument("--db-url")
    s.add_argument("--report-every", type=float, default=30.0)
    s.set_defaults(func=cmd_collect)

    ds = sub.add_parser("dataset", help="build normalized datasets")
    dsub = ds.add_subparsers(dest="dataset_command", required=True)
    b = with_config(dsub.add_parser("build", help="raw store -> normalized dataset dir"))
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--db-url")
    b.add_argument("--mappings-dir", default="config/mappings/registry")
    b.add_argument("--reference", nargs="+", default=["COINBASE:BTC-USD"])
    b.add_argument("--settlements", help="JSON list of settlements")
    b.add_argument("--name")
    b.set_defaults(func=cmd_dataset_build)
    y = dsub.add_parser("synth", help="synthetic dataset dir")
    y.add_argument("--out", required=True)
    y.add_argument("--hours", type=int, default=6)
    y.add_argument("--seed", type=int, default=1)
    y.add_argument("--mm-lag-ms", type=float, default=350.0)
    y.add_argument("--competitor-ms", type=float, default=120.0)
    y.set_defaults(func=cmd_dataset_synth)

    def with_run(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--dataset", required=True, help="dir or synthetic:{json params}")
        sp.add_argument("--strategy", default="fv_taker")
        sp.add_argument("--strategy-version", default="1.0")
        sp.add_argument("--params", default="{}", help="strategy params JSON")
        sp.add_argument("--seed", type=int, default=7)
        return with_config(sp)

    s = with_run(sub.add_parser("backtest", help="single latency/cost run"))
    s.add_argument("--latency-ms", type=int, default=250)
    s.add_argument("--cost", default="base")
    s.set_defaults(func=cmd_backtest)

    s = with_run(sub.add_parser("stress", help="full latency x cost grid + manifest"))
    s.add_argument("--out", required=True)
    s.add_argument("--experiment-id")
    s.set_defaults(func=cmd_stress)

    s = sub.add_parser("reproduce", help="re-run a published experiment from its manifest")
    s.add_argument("--manifest", required=True)
    s.set_defaults(func=cmd_reproduce)

    s = sub.add_parser("leadlag", help="lead-lag discovery between two instruments")
    s.add_argument("--dataset", required=True)
    s.add_argument("--x", required=True)
    s.add_argument("--y", required=True)
    s.add_argument("--max-lag-ms", type=int, default=5_000)
    s.add_argument("--hurdle", type=float, default=0.0)
    s.set_defaults(func=cmd_leadlag)

    s = sub.add_parser("hurdle", help="analytical cost hurdle table")
    s.add_argument("--fee", nargs="+", default=["kalshi-standard", "polymarket-crypto-taker"])
    s.add_argument("--sigma", type=float, default=0.45)
    s.set_defaults(func=cmd_hurdle)

    s = sub.add_parser("evaluate", help="full edge evaluation (synthetic study) + reports")
    s.add_argument("--out", default="reports/edge_evaluation")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--workers", type=int, default=0)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("report", help="render reports from an evaluation.json")
    s.add_argument("--evaluation", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("smoke", help="paper-mode smoke test against mock feeds")
    s.set_defaults(func=cmd_smoke)

    s = with_config(sub.add_parser("paper", help="forward paper trading (needs network)"))
    s.add_argument("--duration", type=float, default=3600.0)
    s.add_argument("--mappings-dir", default="config/mappings/registry")
    s.set_defaults(func=cmd_paper)

    s = with_config(sub.add_parser("live", help="always refused in v1"))
    s.set_defaults(func=cmd_live)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]

_ = (Decimal, iso_from_ns)
