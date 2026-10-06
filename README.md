# Cross-Market Prediction Trading Research Platform (`cma`)

Research-first, **paper-only** platform that tests whether information from faster
markets (BTC/ETH spot, implied vol) predicts short-horizon repricing of prediction-market
contracts (Kalshi, Polymarket) strongly enough to leave **executable net alpha** after
fees, spread, depth, latency, partial fills and adverse selection. Built to the
"Technical Scope, Architecture, Validation Plan & Test Specification" v1.0 (5 Oct 2026).

> **Edge evaluation:** [`reports/edge_evaluation/DECISION_REPORT.md`](reports/edge_evaluation/DECISION_REPORT.md)
> (charts: `reports/edge_evaluation/edge_evaluation.html`; live study: [`reports/live_study/summary.md`](reports/live_study/summary.md)).
> Real-market decision: **COLLECT_MORE_DATA** by a rule fixed before the run. On 1.9 h of authenticated
> Kalshi BTC order books (702 contracts, 10.2M book updates) against Coinbase BTC-USD, Kalshi had usually
> repriced before a BTC move reached a cloud machine (60% of quotes already halfway, three quarters within
> ~230 ms), and taking the quote after a ≥3 bp move lost 0.78¢ per contract at 100 ms after the fee
> (±0.25¢ at 2 SE, 72 moves). The window was calm (9 moves ≥5 bp, none ≥10 bp), so the ≥5 bp test
> (+0.11 ± 0.64¢) cannot decide. In a calibrated simulation through the production pipeline a lead-lag
> *taker* clears the 0.07·p(1−p) fee only when makers take ≳1 s to re-quote; both synthetic base cases are
> **REJECT**.

## What is in the box

| Area | Module(s) | Highlights |
|---|---|---|
| Domain | `cma.domain` | Decimal money/probabilities, canonical YES book, immutable source/recv/process timestamps, versioned fee schedules (Kalshi `0.07·C·P(1−P)` order-rounded; Polymarket crypto `0.07·C·p(1−p)`) |
| Data | `cma.adapters`, `cma.ingestion`, `cma.storage.raw` | Kalshi / Polymarket / Coinbase / Binance / Deribit parsers (current dollar/`_fp` schemas + legacy fallback), WS lifecycle with backoff/resubscribe, REST rate limits, quarantine, append-only hashed raw store, crash-safe dedup |
| Books | `cma.ingestion.book` | snapshot + delta reconstruction, sequence-gap invalidation, duplicate/old-delta immunity, crossed/locked detection (fail closed) |
| Semantics | `cma.mapping` | reviewed mapping registry (UNMAPPED → DRAFT → REVIEWED → APPROVED_PAPER; LIVE reserved), four-eyes approval, equivalence checks (cutoff, timezone, source, method), deterministic Kalshi/Polymarket parsers |
| Models | `cma.models` | lead-lag discovery (grid CCF + Hayashi–Yoshida, block-permutation null, OOS predictive + economic gates), option-implied probabilities (smile-consistent digitals, explicit expiry alignment), structural/cross-venue constraint detectors |
| Signals & risk | `cma.signals`, `cma.risk` | edge = fair-value edge − fees − slippage − adverse-selection − uncertainty; gated signals with reason codes and TTL; contract/family/portfolio limits, daily stop, kill switch |
| Execution | `cma.execution` | latency-aware simulator: venue-time books, liquidity-consumption overlay, conservative maker queue, cancel latency, venue speed bumps; forward paper executor sharing the same core; live adapter that cannot be constructed in v1 |
| Research | `cma.backtest`, `cma.research` | discrete-event replay, latency × cost stress grids, walk-forward/purged splits, locked final test, bootstrap CIs, FDR, promotion gates, manifests + one-command reproduction, calibrated synthetic markets, analytical cost hurdle, decision reports |

## Quick start

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"            # add ".[kalshi-auth]" for Kalshi WebSocket signing

pytest -q                             # unit, property, integration (mocks), replay, acceptance, security
python scripts/check_test_ids.py      # all 55 scope test IDs present
cma smoke                             # paper-mode smoke test on mock feeds -> health status
cma hurdle                            # analytical fee/latency hurdle table
cma evaluate --out reports/edge_evaluation   # full synthetic edge study (~30 min, 4 cores)
python scripts/benchmark.py           # s.22 performance numbers -> docs/benchmarks.json
cma live                              # always refused in v1 (exit code 3)
```

## Real-data workflow

```bash
export KALSHI_API_KEY_ID=...  KALSHI_PRIVATE_KEY_PATH=...   # secrets: environment only
cma db migrate --db-url sqlite:///data/cma.sqlite
cma collect --config config/base.yaml --duration 1209600     # >= 14 days of raw capture
# review mappings in config/mappings/registry (DRAFT -> REVIEWED -> APPROVED_PAPER)
cma dataset build --raw data/raw --out data/datasets/btc-2w --reference COINBASE:BTC-USD
cma leadlag --dataset data/datasets/btc-2w --x COINBASE:BTC-USD --y KALSHI:<ticker>
cma stress --dataset data/datasets/btc-2w --out reports/runs/btc-2w   # refuses incomplete grids
cma reproduce --manifest reports/runs/btc-2w/manifest.json
cma paper --config config/base.yaml config/paper.yaml --duration 86400
```

## Non-negotiables (enforced in code and tests)

* No look-ahead: features declare source watermarks; anything after the decision time raises.
* Replay ordering: venue events at venue time, observations at receive time, our orders
  meet the book only after simulated arrival; identical inputs give byte-identical ledgers.
* Costs before P&L: fees (versioned per fill), depth-walk slippage, latency, partial fills,
  queue position; 0 ms latency is diagnostic only; results must cover 0/100/250/500/1000/
  2000/5000 ms and ≥ 2 adverse cost scenarios or publication is refused.
* Fail closed: gaps, crossed books, stale data, clock drift and unreviewed mappings suppress
  signals. Live trading is impossible in v1.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/RUNBOOK.md`](docs/RUNBOOK.md) and
[`docs/EDGE_EVALUATION_METHOD.md`](docs/EDGE_EVALUATION_METHOD.md).
