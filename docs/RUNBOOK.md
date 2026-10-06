# Runbook

## Network and credentials

| Host | Purpose |
|---|---|
| `external-api.kalshi.com`, `external-api-ws.kalshi.com` (or `api.elections.kalshi.com`) | Kalshi REST / WebSocket (WS needs API-key auth even for market data) |
| `gamma-api.polymarket.com`, `clob.polymarket.com`, `ws-subscriptions-clob.polymarket.com`, `data-api.polymarket.com` | Polymarket discovery, books, stream, trades (`/v2/trades`) |
| `api.exchange.coinbase.com`, `ws-feed.exchange.coinbase.com` | BTC-USD reference (BRTI constituent) |
| `www.deribit.com` | option chain / DVOL implied volatility |

Secrets are read from the environment only (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`);
config files may only name the environment variables. Logs pass through a redacting filter.

## Collect

```bash
cma db migrate --db-url sqlite:///data/cma.sqlite
cma collect --config config/base.yaml --duration 1209600 --report-every 60
```

The collector prints `health_report()` (feed connection state, message rates, reconnects,
gaps, source→receive latency, book validity, quarantine counts, open incidents). Books are
invalidated on disconnect and re-validated only by a fresh snapshot.

## Map and review

`MappingRegistry` proposals from `cma.mapping.parsers` are always DRAFT. A reviewer affirms
the checklist (strike, observation window, timezone, settlement source, rounding, early
close, outcome semantics) → REVIEWED (backtest-eligible); a *different* approver promotes to
APPROVED_PAPER (forward-paper eligible). LIVE_ELIGIBLE always raises in v1.

## Evaluate

```bash
cma dataset build --raw data/raw --out data/datasets/<name>
cma stress  --dataset data/datasets/<name> --out reports/runs/<name>
cma reproduce --manifest reports/runs/<name>/manifest.json
```

`cma stress` refuses to publish unless the grid contains 0/100/250/500/1000/2000/5000 ms and
base + ≥2 adverse cost scenarios, and the manifest names git commit, config hash, dataset
hash, strategy version and seed.

## Paper

```bash
cma paper --config config/base.yaml config/paper.yaml --duration 86400
```

Only APPROVED_PAPER contracts trade. Fills and portfolio state persist to `paper_fills` /
`paper_state` for restart recovery. The kill switch blocks new orders synchronously and
cancels every working simulated order.

## CI gates

`ruff format --check`, `ruff check`, `mypy --strict`, all tests (no network), test-ID
coverage, per-package coverage (≥ 90% domain/portfolio/simulator/risk, ≥ 80% core), fixture
reproducibility, migration from an empty DB, paper smoke test, `cma live` must fail.
Set `OPENBLAS_NUM_THREADS=1` (BLAS thread contention slows small dot products).
