"""End-to-end edge evaluation (the research cycle of scope s.11, s.13, s.24, s.30).

Runs, in order:

1. Analytical cost hurdle (``cma.research.hurdle``).
2. Edge frontier: synthetic markets sweeping market-maker reaction lag and competitor
   presence, replayed through the full pipeline at every mandated latency.
3. Base case calibrated to public evidence (maker reaction ~350 ms median, competing
   arbitrageurs ~120 ms, Kalshi taker fees, one-tick markets) with chronological
   train / validation / locked final-test windows, a logged parameter-sensitivity search on
   validation, the full latency x cost stress grid on the untouched test window, bootstrap
   confidence intervals and the promotion gates.
4. Venue variant: identical market under Polymarket's crypto fee and 150 ms taker delay.
5. Lead-lag discovery (reference mid -> contract mid) with regime stratification, and a
   structural (nested-strike) scan on the same data.

Everything is seeded and recorded in a manifest. Synthetic results quantify *edge
potential under stated assumptions*; they are not evidence of real-market profitability.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from cma.backtest.experiment import (
    DatasetSpec,
    LoadedDataset,
    StrategySpec,
    StressGrid,
    _sha,
    build_manifest,
    default_partitions,
    run_single,
    validate_publication,
)
from cma.config import AppConfig, CostScenario
from cma.domain.enums import BookSide, Decision, Venue
from cma.domain.errors import ExperimentPublicationError
from cma.domain.models import BookDeltaEvent, BookSnapshotEvent, MarketEvent
from cma.domain.time import NS_PER_MS, NS_PER_S, iso_from_ns
from cma.ingestion.book import L2BookBuilder
from cma.research.gates import DecisionOutcome, NetPnlCriterion, PromotionPolicy, decide
from cma.research.hurdle import hurdle_table

IV_INSTRUMENT = "DERIBIT:BTC-DVOL"
# Default model reads the implied-vol index (same vol information as the makers), so the
# frontier isolates *latency*; the realized-vol model is evaluated separately (model risk).
BASE_STRATEGY = StrategySpec("fv_taker", "1.0", {"vol_instrument": IV_INSTRUMENT})
REALIZED_VOL_STRATEGY = StrategySpec("fv_taker", "1.0", {})


@dataclass(frozen=True)
class EvaluationPlan:
    seed: int = 20261006
    frontier_hours: int = 8
    frontier_mm_lags_ms: tuple[float, ...] = (150.0, 350.0, 1000.0, 3000.0)
    frontier_competitors_ms: tuple[float | None, ...] = (None, 120.0)
    latencies_ms: tuple[int, ...] = (0, 100, 250, 500, 1000, 2000, 5000)
    base_hours: int = 48
    base_mm_lag_ms: float = 350.0
    base_competitor_ms: float | None = 120.0
    sensitivity_edges_bps: tuple[int, ...] = (50, 100, 200, 300)
    sensitivity_vol_mult: tuple[float, ...] = (0.9, 1.0, 1.1)
    frontier_realized_vol_scenarios: tuple[tuple[float, float | None], ...] = (
        (350.0, 120.0),
        (3000.0, None),
        (3000.0, 120.0),
    )
    criterion_latency_ms: int = 250
    polymarket_taker_delay_ms: int = 150
    workers: int = 4
    warmup_minutes: int = 20

    @classmethod
    def quick(cls) -> EvaluationPlan:
        return cls(
            frontier_hours=2,
            frontier_mm_lags_ms=(350.0, 3000.0),
            frontier_competitors_ms=(None, 120.0),
            base_hours=5,
            sensitivity_edges_bps=(100, 200),
            sensitivity_vol_mult=(1.0,),
            frontier_realized_vol_scenarios=((3000.0, None),),
        )


EVAL_TTL_MS = 10_000  # strategy TTL for latency stress: arrival up to 10 s after decision


def _cfg(min_edge_bps: int = 100) -> AppConfig:
    return AppConfig.model_validate(
        {
            "mode": "BACKTEST",
            "signal": {"min_net_edge_bps": min_edge_bps, "ttl_ms": EVAL_TTL_MS},
        }
    )


def _market_params(
    plan: EvaluationPlan, *, hours: int, mm_lag: float, comp: float | None, seed: int, **kw: Any
) -> dict[str, Any]:
    return {
        "hours": hours,
        "seed": seed,
        "mm_lag_ms": mm_lag,
        "competitor_latency_ms": comp,
        **kw,
    }


# ----------------------------------------------------------------------------- truth


_SHARED: dict[str, tuple[LoadedDataset, Any]] = {}


def synthetic_dataset(params: Mapping[str, Any]) -> tuple[LoadedDataset, Any]:
    """Generate (or reuse) a synthetic market; returns (dataset, market-with-truth)."""
    from cma.research.synthetic import SyntheticMarketConfig, generate_market

    key = json.dumps(dict(params), sort_keys=True, default=str)
    hit = _SHARED.get(key)
    if hit is not None:
        return hit
    m = generate_market(SyntheticMarketConfig(**params))
    ds = LoadedDataset(
        name=f"synthetic:{m.config.label()}",
        events=m.events,
        contracts=m.contracts,
        mappings=m.mappings,
        settlements=m.settlements,
        closes=m.closes,
        reference_instruments=m.reference_instruments,
        dataset_hash=_sha({"kind": "synthetic", "params": dict(params)}),
        provenance={"generator": "cma.research.synthetic", "stats": m.stats},
    )
    _SHARED[key] = (ds, m)
    return ds, m


def ex_ante_edge(core: Any, market: Any) -> dict[str, float]:
    """Expected P&L of every fill under the TRUE model (simulation-only diagnostic).

    E[payoff | information at fill time] is the true fair value computed from the true
    index path and volatility, so ``side*(fair_true - price)*qty - fee`` isolates the
    edge from settlement luck. Not available for real data.
    """
    from cma.research.synthetic import _fair_on_grid

    cfg = market.config
    fills = core.records.fills
    if not fills:
        return {"expected_net_pnl": 0.0, "expected_gross_pnl": 0.0, "expected_c_per_contract": 0.0}
    dt_ns = cfg.dt_ms * NS_PER_MS
    index = market.truth["index"]
    sigma = market.truth["sigma"]
    n = index.size
    t_grid = cfg.start_ns + np.arange(n, dtype=np.int64) * dt_ns
    prefix = np.concatenate([[0.0], np.cumsum(index)])
    window_s = 60.0 if cfg.observation_method == "AVG_60S_BEFORE" else 0.0
    maps = {m.contract_id: m for m in market.mappings}
    by_contract: dict[str, list[Any]] = {}
    for f in fills:
        by_contract.setdefault(f.contract_id, []).append(f)
    gross = 0.0
    fees = 0.0
    qty_total = 0.0
    for cid, fs in by_contract.items():
        mp = maps[cid]
        k = np.clip(
            (np.array([f.fill_ts_ns for f in fs], dtype=np.int64) - cfg.start_ns) // dt_ns, 0, n - 1
        )
        fair = _fair_on_grid(
            index,
            prefix,
            t_grid,
            k,
            float(mp.strikes[0]),
            mp.observation_end_ns,
            sigma,
            window_s,
            cfg.dt_ms / 1000.0,
        )
        for f, fv in zip(fs, fair, strict=True):
            sign = 1.0 if f.side.value == "BUY" else -1.0
            q = float(f.quantity)
            gross += sign * (float(fv) - float(f.price)) * q
            fees += float(f.fee)
            qty_total += q
    net = gross - fees
    return {
        "expected_net_pnl": net,
        "expected_gross_pnl": gross,
        "expected_c_per_contract": 100.0 * net / qty_total if qty_total else 0.0,
    }


# ----------------------------------------------------------------------------- frontier


def _frontier_task(
    args: tuple[dict[str, Any], tuple[int, ...], int, str],
) -> list[dict[str, Any]]:
    params, latencies, seed, model = args
    ds, market = synthetic_dataset(params)
    cfg = _cfg()
    strategy = BASE_STRATEGY if model == "implied_vol" else REALIZED_VOL_STRATEGY
    rows = []
    for lat in latencies:
        r, core = run_single(
            ds, strategy, cfg, latency_ms=lat, cost=CostScenario(name="base"), seed=seed
        )
        m = r.metrics
        ex = ex_ante_edge(core, market)
        rows.append(
            {
                "mm_lag_ms": params["mm_lag_ms"],
                "competitor_ms": params["competitor_latency_ms"],
                "model": model,
                "fee_schedule": params.get("fee_schedule_id", "kalshi-standard"),
                "latency_ms": lat,
                "hours": params["hours"],
                "n_fills": m.n_fills,
                "contracts": m.filled_contracts,
                "fees": m.fees,
                "realized_net_pnl": m.net_pnl,
                "expected_net_pnl": ex["expected_net_pnl"],
                "expected_c_per_contract": ex["expected_c_per_contract"],
                "markout_60s_net_pnl": m.markout_net_pnl.get("60000ms", math.nan),
                "markout_5s_c_per_contract": m.markout_net_c_per_contract.get("5000ms", math.nan),
                "fill_rate": m.fill_rate,
                "events": r.n_events,
                "maker_updates": market.stats.get("maker_updates", 0),
                "competitor_takes": market.stats.get("competitor_takes", 0),
            }
        )
    _SHARED.clear()
    return rows


def run_frontier(plan: EvaluationPlan) -> list[dict[str, Any]]:
    tasks = []
    for mm in plan.frontier_mm_lags_ms:
        for comp in plan.frontier_competitors_ms:
            params = _market_params(
                plan, hours=plan.frontier_hours, mm_lag=mm, comp=comp, seed=plan.seed
            )
            tasks.append((params, plan.latencies_ms, plan.seed, "implied_vol"))
            if (mm, comp) in plan.frontier_realized_vol_scenarios:
                tasks.append((params, plan.latencies_ms, plan.seed, "realized_vol"))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=plan.workers, mp_context=_fork()) as pool:
        for part in pool.map(_frontier_task, tasks):
            rows.extend(part)
    return rows


def _fork() -> Any:
    import multiprocessing

    return multiprocessing.get_context("fork")


# ----------------------------------------------------------------------------- base case


def _window_with_warmup(ds: LoadedDataset, start: int, end: int, warmup_ns: int) -> LoadedDataset:
    """Contracts listing/resolving in [start, end) plus reference-only warm-up history."""
    core = ds.window(start, end)
    warm = [
        e
        for e in ds.events
        if start - warmup_ns <= e.recv_ts_ns < start and e.instrument_id in ds.reference_instruments
    ]
    return replace(core, events=warm + core.events)


def _sensitivity_task(
    args: tuple[dict[str, Any], int, float, list[int], int, int, int],
) -> dict[str, Any]:
    params, edge_bps, vol_mult, window, warmup_ns, latency, seed = args
    ds, market = synthetic_dataset(params)
    part = _window_with_warmup(ds, window[0], window[1], warmup_ns)
    strat = StrategySpec(
        "fv_taker", "1.0", {"vol_multiplier": vol_mult, "vol_instrument": IV_INSTRUMENT}
    )
    r, core = run_single(
        part, strat, _cfg(edge_bps), latency_ms=latency, cost=CostScenario(name="base"), seed=seed
    )
    pnls = list(_position_pnls(core).values())
    ex = ex_ante_edge(core, market)
    return {
        "min_net_edge_bps": edge_bps,
        "vol_multiplier": vol_mult,
        "net_pnl": r.metrics.net_pnl,
        "expected_net_pnl": ex["expected_net_pnl"],
        "markout_60s_net_pnl": r.metrics.markout_net_pnl.get("60000ms", math.nan),
        "n_positions": r.metrics.n_positions,
        "contracts": r.metrics.filled_contracts,
        "position_pnls": pnls,
    }


def _position_pnls(core: Any) -> dict[str, float]:
    from cma.backtest.metrics import position_pnls

    return position_pnls(core.records, core)


def _family_pnls(core: Any) -> list[float]:
    """Net P&L per event family (hourly ladder): strikes of one family are one bet."""
    fam: dict[str, float] = {}
    for cid, pnl in _position_pnls(core).items():
        key = core._family_of(cid) or cid
        fam[key] = fam.get(key, 0.0) + pnl
    return list(fam.values())


def _grid_task(
    args: tuple[
        dict[str, Any], dict[str, Any], int, dict[str, Any], list[int], int, int, dict[str, int]
    ],
) -> tuple[dict[str, Any], list[float]]:
    params, strat_params, edge_bps, cost, window, warmup_ns, latency, taker_delay = args
    ds, market = synthetic_dataset(params)
    part = _window_with_warmup(ds, window[0], window[1], warmup_ns)
    scenario = CostScenario(**cost)
    r, core = run_single(
        part,
        StrategySpec("fv_taker", "1.0", strat_params),
        _cfg(edge_bps),
        latency_ms=latency,
        cost=scenario,
        seed=int(params["seed"]),
        venue_taker_delay_ms={Venue(k): v for k, v in taker_delay.items()},
    )
    out = r.to_dict()
    out["ex_ante"] = ex_ante_edge(core, market)
    out["family_pnls"] = _family_pnls(core)
    return out, list(_position_pnls(core).values())


def _model_risk_task(args: tuple[dict[str, Any], list[int], int, int, int]) -> dict[str, Any]:
    params, window, warmup_ns, latency, edge_bps = args
    ds, market = synthetic_dataset(params)
    part = _window_with_warmup(ds, window[0], window[1], warmup_ns)
    out = {}
    for label, strat in (("implied_vol", BASE_STRATEGY), ("realized_vol", REALIZED_VOL_STRATEGY)):
        r, core = run_single(
            part,
            strat,
            _cfg(edge_bps),
            latency_ms=latency,
            cost=CostScenario(name="base"),
            seed=int(params["seed"]),
        )
        ex = ex_ante_edge(core, market)
        out[label] = {
            "realized_net_pnl": r.metrics.net_pnl,
            "expected_net_pnl": ex["expected_net_pnl"],
            "expected_c_per_contract": ex["expected_c_per_contract"],
            "markout_60s_net_pnl": r.metrics.markout_net_pnl.get("60000ms", math.nan),
            "contracts": r.metrics.filled_contracts,
        }
    return {"latency_ms": latency, **out}


def _grid_from_dicts(rows: Sequence[dict[str, Any]]) -> StressGrid:
    from cma.backtest.experiment import RunResult
    from cma.backtest.metrics import TradingMetrics

    results = [
        RunResult(
            latency_ms=int(d["latency_ms"]),
            cost=str(d["cost"]),
            metrics=TradingMetrics(**d["metrics"]),
            ledger_hash=str(d["ledger_hash"]),
            n_events=int(d["n_events"]),
        )
        for d in rows
    ]
    return StressGrid(results)


@dataclass
class BaseCaseResult:
    label: str
    params: dict[str, Any]
    partitions: dict[str, list[int]]
    sensitivity: list[dict[str, Any]]
    multiple_testing: dict[str, Any]
    selected: dict[str, Any]
    grid: StressGrid
    base_position_pnls: list[float]
    ci: dict[str, Any]
    decision: DecisionOutcome
    publication_ok: bool
    publication_error: str = ""
    final_test_access_log: list[str] = field(default_factory=list)
    ex_ante: dict[str, dict[str, float]] = field(default_factory=dict)
    model_risk: list[dict[str, Any]] = field(default_factory=list)


def run_base_case(
    plan: EvaluationPlan,
    *,
    label: str,
    fee_schedule_id: str = "kalshi-standard",
    taker_delay: Mapping[str, int] | None = None,
) -> BaseCaseResult:
    from cma.research.stats import benjamini_hochberg, bootstrap_mean_ci

    cfg = _cfg()
    params = _market_params(
        plan,
        hours=plan.base_hours,
        mm_lag=plan.base_mm_lag_ms,
        comp=plan.base_competitor_ms,
        seed=plan.seed + 1,
        fee_schedule_id=fee_schedule_id,
    )
    _SHARED.clear()
    ds, _market = synthetic_dataset(params)  # generated once; forked workers share it
    start = min(c.open_ts_ns or 0 for c in ds.contracts)
    end = max(c.close_ts_ns or 0 for c in ds.contracts)
    parts = default_partitions(start, end, cfg)
    warmup_ns = plan.warmup_minutes * 60 * NS_PER_S
    delay = dict(taker_delay or {})

    # --- parameter sensitivity on VALIDATION only (logged as hypotheses)
    tasks = [
        (params, e, v, parts["validation"], warmup_ns, plan.criterion_latency_ms, plan.seed)
        for e in plan.sensitivity_edges_bps
        for v in plan.sensitivity_vol_mult
    ]
    with ProcessPoolExecutor(max_workers=plan.workers, mp_context=_fork()) as pool:
        sens = list(pool.map(_sensitivity_task, tasks))
    pvals = []
    for row in sens:
        x = np.asarray(row.pop("position_pnls"), dtype=float)
        if x.size >= 3 and np.std(x, ddof=1) > 0:
            tstat = float(np.mean(x) / (np.std(x, ddof=1) / math.sqrt(x.size)))
            from scipy.stats import t as student_t

            p = float(student_t.sf(tstat, df=x.size - 1))
        else:
            p = 1.0
        row["p_value_mean_gt_0"] = p
        pvals.append(p)
    bh = benjamini_hochberg(np.asarray(pvals), alpha=cfg.research.fdr_alpha)
    multiple = {
        "n_hypotheses": len(pvals),
        "alpha": cfg.research.fdr_alpha,
        "bh_adjusted": [float(x) for x in bh.adjusted],
        "n_rejected": int(bh.n_rejected),
    }

    def _score(r: dict[str, Any]) -> float:
        v = r.get("markout_60s_net_pnl", math.nan)
        return float(v) if v is not None and math.isfinite(float(v)) else -math.inf

    best = max(sens, key=lambda r: (_score(r), -abs(r["min_net_edge_bps"] - 100)))
    selected: dict[str, Any]
    if _score(best) <= 0:
        selected = {
            "min_net_edge_bps": 100,
            "vol_multiplier": 1.0,
            "why": "pre-declared default (no validation setting had positive 60 s mark-outs)",
        }
    else:
        selected = {
            "min_net_edge_bps": best["min_net_edge_bps"],
            "vol_multiplier": best["vol_multiplier"],
            "why": "best fee-adjusted 60 s mark-out P&L on validation",
        }

    # --- unlock the final test ONCE and run the full stress grid
    access_log = [
        f"final test {iso_from_ns(parts['final_test'][0])}..{iso_from_ns(parts['final_test'][1])}"
        f" unlocked once for {label} with params {selected}"
    ]
    costs = [c.model_dump(mode="json") for c in cfg.research.cost_scenarios]
    strat_params = {"vol_multiplier": selected["vol_multiplier"], "vol_instrument": IV_INSTRUMENT}
    grid_tasks = [
        (
            params,
            strat_params,
            int(selected["min_net_edge_bps"]),
            cost,
            parts["final_test"],
            warmup_ns,
            lat,
            delay,
        )
        for cost in costs
        for lat in plan.latencies_ms
    ]
    with ProcessPoolExecutor(max_workers=plan.workers, mp_context=_fork()) as pool:
        outs = list(pool.map(_grid_task, grid_tasks))
        model_risk = list(
            pool.map(
                _model_risk_task,
                [
                    (params, parts["final_test"], warmup_ns, lat, int(selected["min_net_edge_bps"]))
                    for lat in plan.latencies_ms
                ],
            )
        )
    _SHARED.clear()
    grid = _grid_from_dicts([o[0] for o in outs])
    ex_ante = {f"{o[0]['cost']}@{o[0]['latency_ms']}": o[0]["ex_ante"] for o in outs}
    base_idx = next(
        i
        for i, t in enumerate(grid_tasks)
        if t[3]["name"] == "base" and t[6] == plan.criterion_latency_ms
    )
    base_pnls = outs[base_idx][1]
    family = [float(x) for x in outs[base_idx][0]["family_pnls"]]
    ci: dict[str, Any]
    if len(family) >= 3:
        ci_res = bootstrap_mean_ci(np.asarray(family), n_boot=4000, seed=plan.seed)
        ci = {
            "unit": "event family (hourly ladder)",
            "n": len(family),
            "mean": float(ci_res.estimate),
            "lower": float(ci_res.lower),
            "upper": float(ci_res.upper),
        }
    else:
        ci = {
            "unit": "event family",
            "n": len(family),
            "mean": float(np.mean(family or [0.0])),
            "lower": math.nan,
            "upper": math.nan,
        }
    policy = PromotionPolicy(
        criterion=NetPnlCriterion(latency_ms=plan.criterion_latency_ms, require_ci_lower_above=0.0)
    )
    decision = decide(
        final_test=grid,
        policy=policy,
        sensitivity_net_pnls=[r["net_pnl"] for r in sens],
        ci_lower=None if math.isnan(ci["lower"]) else ci["lower"],
        ci_upper=None if math.isnan(ci["upper"]) else ci["upper"],
        data_quality_ok=True,
        mappings_approved_paper=False,  # synthetic mappings are REVIEWED, never paper-approved
        tests_passed=True,
        paper_days_completed=0.0,
    )
    manifest_stub = {
        "git_commit": "pending",
        "config_hash": "c",
        "dataset": params,
        "dataset_hash": ds.dataset_hash,
        "strategy": strat_params,
        "seed": plan.seed,
    }
    try:
        validate_publication(grid, manifest_stub)
        pub_ok, pub_err = True, ""
    except ExperimentPublicationError as exc:
        pub_ok, pub_err = False, str(exc)
    return BaseCaseResult(
        label=label,
        params=params,
        partitions=parts,
        sensitivity=sens,
        multiple_testing=multiple,
        selected=selected,
        grid=grid,
        base_position_pnls=base_pnls,
        ci=ci,
        decision=decision,
        publication_ok=pub_ok,
        publication_error=pub_err,
        final_test_access_log=access_log,
        ex_ante=ex_ante,
        model_risk=model_risk,
    )


# ----------------------------------------------------------------------------- lead-lag


def mid_series(events: Sequence[MarketEvent], instrument_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Venue-time mid of a canonical book, sampled after every update while valid."""
    b: L2BookBuilder | None = None
    ts: list[int] = []
    mids: list[float] = []
    for ev in events:
        if ev.instrument_id != instrument_id or not isinstance(
            ev, BookSnapshotEvent | BookDeltaEvent
        ):
            continue
        if b is None:
            b = L2BookBuilder(venue=ev.venue, instrument_id=instrument_id, require_sequence=False)
        b.apply(ev)
        if not b.is_valid:
            continue
        bb, ba = b.best_bid(), b.best_ask()
        if bb is None or ba is None:
            continue
        ts.append(ev.venue_ts_ns)
        mids.append(float((bb[0] + ba[0]) / 2))
    return np.asarray(ts, dtype=np.int64), np.asarray(mids, dtype=float)


def run_lead_lag(plan: EvaluationPlan, max_contracts: int = 4) -> dict[str, Any]:
    from cma.models.lead_lag.discovery import LeadLagConfig, discover_lead_lag, stratified_lead_lag

    params = _market_params(
        plan,
        hours=min(plan.base_hours, 6),
        mm_lag=plan.base_mm_lag_ms,
        comp=plan.base_competitor_ms,
        seed=plan.seed + 1,
    )
    ds, _m = synthetic_dataset(params)
    ref = sorted(ds.reference_instruments)[0]
    x_ts, x_v = mid_series(ds.events, ref)
    # near-the-money contracts: middle strikes of each hourly ladder
    by_event: dict[str, list[str]] = {}
    for c in ds.contracts:
        by_event.setdefault(c.event_id, []).append(c.contract_id)
    picks = []
    for cids in by_event.values():
        picks.append(sorted(cids)[len(cids) // 2])
    picks = picks[:max_contracts]
    cfg = LeadLagConfig(
        grid_ms=100,
        max_lag_ms=3_000,
        horizons_ms=(500, 1_000, 2_000),
        permutation_samples=199,
        cost_hurdle=0.0175 + 0.005,  # taker fee at p=0.5 + half tick (probability units)
        seed=plan.seed,
        min_obs=500,
    )
    results = []
    for cid in picks:
        y_ts, y_v = mid_series(ds.events, cid)
        if y_ts.size < 50:
            continue
        lo, hi = y_ts[0], y_ts[-1]
        mask = (x_ts >= lo - 60 * NS_PER_S) & (x_ts <= hi)
        res = discover_lead_lag(x_ts[mask], x_v[mask], y_ts, y_v, cfg)
        expiry = next(c.close_ts_ns for c in ds.contracts if c.contract_id == cid) or hi
        tte_bucket = np.where(
            (expiry - y_ts) > 30 * 60 * NS_PER_S,
            ">30m",
            np.where((expiry - y_ts) > 10 * 60 * NS_PER_S, "10-30m", "<10m"),
        )
        strata: dict[str, Any]
        try:
            strat = stratified_lead_lag(
                x_ts[mask], x_v[mask], y_ts, y_v, cfg, strata=tte_bucket, strata_on="y_obs"
            )
            strata = {str(k): _ll_summary(v) for k, v in strat}
        except Exception as exc:  # stability analysis must not hide the main result
            strata = {"error": str(exc)}
        results.append({"contract": cid, "overall": _ll_summary(res), "by_time_to_expiry": strata})
    return {
        "config": cfg.to_dict(),
        "reference": ref,
        "true_maker_lag_ms": plan.base_mm_lag_ms,
        "results": results,
    }


def _ll_summary(r: Any) -> dict[str, Any]:
    return {
        "best_lag_ms": r.best_lag_ms,
        "best_positive_lag_ms": r.best_positive_lag_ms,
        "hy_best_lag_ms": r.hy_best_lag_ms,
        "corr_at_best": _num(r.corr_at_best),
        "corr_at_zero": _num(r.corr_at_zero),
        "p_value": _num(r.p_value),
        "oos_r2": _num(r.oos_r2),
        "incremental_oos_r2": _num(r.incremental_oos_r2),
        "hit_rate": _num(r.hit_rate),
        "economic_edge_estimate": _num(r.economic_edge_estimate),
        "n_obs": r.n_obs,
        "qualifies": r.qualifies,
        "reasons": list(r.reasons),
    }


def _num(x: float) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


# ----------------------------------------------------------------------------- structural


def run_structural_scan(plan: EvaluationPlan, sample_every_s: int = 5) -> dict[str, Any]:
    from cma.domain.fees import get_fee_schedule
    from cma.models.structural.nested import detect_nested_violations
    from cma.models.structural.types import ContractQuote, StructuralConfig

    params = _market_params(
        plan,
        hours=min(plan.base_hours, 6),
        mm_lag=plan.base_mm_lag_ms,
        comp=plan.base_competitor_ms,
        seed=plan.seed + 1,
    )
    ds, _m = synthetic_dataset(params)
    maps = {m.contract_id: m for m in ds.mappings}
    fee = get_fee_schedule("kalshi-standard")
    builders: dict[str, L2BookBuilder] = {}
    families: dict[str, list[str]] = {}
    for m in ds.mappings:
        families.setdefault(m.event_family, []).append(m.contract_id)
    next_sample = ds.events[0].recv_ts_ns + sample_every_s * NS_PER_S
    samples = 0
    found = 0
    gross_total = Decimal(0)
    net_total = Decimal(0)
    examples: list[dict[str, Any]] = []
    for ev in ds.events:
        if ev.instrument_id in maps and isinstance(ev, BookSnapshotEvent | BookDeltaEvent):
            b = builders.setdefault(
                ev.instrument_id, L2BookBuilder(venue=ev.venue, instrument_id=ev.instrument_id)
            )
            b.apply(ev)
        if ev.recv_ts_ns >= next_sample:
            next_sample += sample_every_s * NS_PER_S
            for _fam, cids in families.items():
                quotes = [
                    ContractQuote(mapping=maps[c], book=builders[c].snapshot(), fee_schedule=fee)
                    for c in cids
                    if c in builders and builders[c].is_valid
                ]
                if len(quotes) < 2:
                    continue
                samples += 1
                opps = detect_nested_violations(quotes, config=StructuralConfig())
                for o in opps:
                    found += 1
                    gross_total += o.gross_edge_per_unit * o.quantity
                    net_total += o.net_profit
                    if len(examples) < 5:
                        examples.append(
                            {
                                "ts": iso_from_ns(ev.recv_ts_ns),
                                "contracts": list(o.contract_ids),
                                "net_edge_per_unit": str(o.net_edge_per_unit),
                                "quantity": str(o.quantity),
                            }
                        )
    return {
        "family_samples": samples,
        "violations_after_costs": found,
        "violation_rate": found / samples if samples else 0.0,
        "gross_total": str(gross_total),
        "net_total": str(net_total),
        "examples": examples,
        "note": "synthetic makers re-quote strikes independently with random lags; "
        "violations need bid(K_high) - ask(K_low) > both taker fees",
    }


# ----------------------------------------------------------------------------- driver


def run_evaluation(out_dir: Path, plan: EvaluationPlan | None = None) -> dict[str, Any]:
    plan = plan or EvaluationPlan()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings: dict[str, float] = {}

    hurdle = [r.to_dict() for r in hurdle_table()]
    timings["hurdle_s"] = time.time() - t0

    t = time.time()
    frontier = run_frontier(plan)
    timings["frontier_s"] = time.time() - t

    t = time.time()
    kalshi = run_base_case(plan, label="kalshi_fee")
    timings["base_kalshi_s"] = time.time() - t

    t = time.time()
    poly = run_base_case(
        plan,
        label="polymarket_fee_speedbump",
        fee_schedule_id="polymarket-crypto-taker",
        taker_delay={Venue.KALSHI.value: plan.polymarket_taker_delay_ms},
    )
    timings["base_polymarket_s"] = time.time() - t

    t = time.time()
    lead_lag = run_lead_lag(plan)
    timings["lead_lag_s"] = time.time() - t

    t = time.time()
    structural = run_structural_scan(plan)
    timings["structural_s"] = time.time() - t

    def base_dict(b: BaseCaseResult) -> dict[str, Any]:
        return {
            "label": b.label,
            "market_params": b.params,
            "partitions": {
                k: [iso_from_ns(v[0]), iso_from_ns(v[1])] for k, v in b.partitions.items()
            },
            "sensitivity_validation": b.sensitivity,
            "multiple_testing": b.multiple_testing,
            "selected_params": b.selected,
            "final_test_grid": [r.to_dict() for r in b.grid.results],
            "net_pnl_table": b.grid.table("net_pnl"),
            "fills_table": b.grid.table("n_fills"),
            "ci_mean_position_pnl": b.ci,
            "ex_ante": b.ex_ante,
            "model_risk": b.model_risk,
            "markout_60s_table": {
                c: {
                    r.latency_ms: r.metrics.markout_net_pnl.get("60000ms", math.nan)
                    for r in b.grid.results
                    if r.cost == c
                }
                for c in b.grid.costs()
            },
            "decision": b.decision.decision.value,
            "decision_reasons": b.decision.reasons,
            "gates": [g.__dict__ for g in b.decision.gates],
            "publication_ok": b.publication_ok,
            "publication_error": b.publication_error,
            "final_test_access_log": b.final_test_access_log,
        }

    cfg = _cfg()
    manifest = build_manifest(
        experiment_id=f"edge-eval-{plan.seed}",
        created_ns=time.time_ns(),
        cfg=cfg,
        dataset=DatasetSpec("synthetic", kalshi.params),
        dataset_hash=DatasetSpec("synthetic", kalshi.params).load().dataset_hash,
        strategy=StrategySpec(
            "fv_taker", "1.0", {"vol_multiplier": kalshi.selected["vol_multiplier"]}
        ),
        seed=plan.seed,
        grid=kalshi.grid,
        partitions=dict(kalshi.partitions) | {"evaluated_window": kalshi.partitions["final_test"]},
        extra={"plan": plan.__dict__},
    )
    result = {
        "plan": plan.__dict__,
        "hurdle": hurdle,
        "frontier": frontier,
        "base_cases": [base_dict(kalshi), base_dict(poly)],
        "lead_lag": lead_lag,
        "structural": structural,
        "real_market_decision": Decision.COLLECT_MORE_DATA.value,
        "timings_s": timings,
        "manifest": manifest,
    }
    (out_dir / "evaluation.json").write_text(
        json.dumps(result, indent=1, default=_json_default) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, default=_json_default) + "\n", encoding="utf-8"
    )
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def scenario_label(mm_lag_ms: float, competitor_ms: float | None) -> str:
    comp = "no competitor" if competitor_ms is None else f"competitor {competitor_ms:g} ms"
    return f"maker lag {mm_lag_ms:g} ms, {comp}"


def book_side_of(side: BookSide) -> str:  # pragma: no cover - helper for reports
    return side.value


__all__ = [
    "EvaluationPlan",
    "run_base_case",
    "run_evaluation",
    "run_frontier",
    "run_lead_lag",
    "run_structural_scan",
]

_ = NS_PER_MS
