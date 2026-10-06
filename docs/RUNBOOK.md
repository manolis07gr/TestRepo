# Runbook

## Network and credentials

| Host | Purpose |
|---|---|
| `external-api.kalshi.com`, `external-api-ws.kalshi.com` (or `api.elections.kalshi.com`) | Kalshi REST / WebSocket (WS needs API-key auth even for market data) |
| `gamma-api.polymarket.com`, `clob.polymarket.com`, `ws-subscriptions-clob.polymarket.com`, `data-api.polymarket.com` | Polymarket discovery, books, stream, trades (`/v2/trades`) |
| `api.exchange.coinbase.com`, `ws-feed.exchange.coinbase.com` | BTC-USD reference (BRTI constituent) |
| `www.deribit.com` | option chain / DVOL implied volatility |

Secrets are read from the environment only (`KALSHI_API_KEY_ID`, and the RSA key as
`KALSHI_PRIVATE_KEY` inline or `KALSHI_PRIVATE_KEY_PATH`); config files may only name the
environment variables. The inline key may be pasted as a full PEM, with `\n` escapes, or as
just the base64 body. `python scripts/check_kalshi_auth.py` verifies the credentials with an
authenticated WebSocket handshake without printing them. Logs pass through a redacting
filter.

## Collect

```bash
cma db migrate --db-url sqlite:///data/cma.sqlite
cma collect --config config/base.yaml --duration 1209600 --report-every 60
```

The collector prints `health_report()` (feed connection state, message rates, reconnects,
gaps, source→receive latency, book validity, quarantine counts, open incidents). Books are
invalidated on disconnect and re-validated only by a fresh snapshot.

## Live staleness study

```bash
python scripts/live_study.py collect --duration 6900 --report-every 60   # ~100 MB/min raw
python scripts/live_study.py analyze --out reports/live_study            # ~15-30 min / 2 h
cma report --evaluation reports/edge_evaluation/evaluation.json \
  --out reports/edge_evaluation --live reports/live_study/summary.json
```

Uses `config/base.yaml` + `config/live_study.yaml` (raw store `data/live_study/raw`). Analysis
streams the raw store once, so memory follows top-of-book changes rather than raw messages.
Method and the pre-registered decision rule: `docs/EDGE_EVALUATION_METHOD.md` section 6.

Longer collections are analysed in windows (for example one per day) and pooled exactly:
`analyze --out data/live_study/windows/<day>` also writes `samples.jsonl.gz`, and
`python scripts/live_study.py pool --windows data/live_study/windows/* --out reports/live_study`
rebuilds every table and the decision from all windows' samples (lead-lag stays per window).
`python scripts/latency_probe.py` prints round-trip times from the collecting machine to Kalshi
and Coinbase (read-only); read the study's latency grid against them. From a Claude cloud session
(through its HTTPS proxy) both were about 60 ms in October 2026.

## Hold-to-settlement study (history)

```bash
python scripts/settlement_study.py fetch --since 2025-12-01     # ~50 min at 10 req/s; resumable
python scripts/settlement_study.py analyze --out reports/settlement_study
cma report --evaluation reports/edge_evaluation/evaluation.json --out reports/edge_evaluation \
  --live reports/live_study/summary.json --settlement reports/settlement_study/summary.json
```

`fetch` downloads every settled `KXBTC15M` market with its 1-minute YES bid/ask candles (Kalshi
serves markets settled before its archive cutoff from `/historical/...`), Coinbase BTC-USD
1-minute candles and hourly Deribit DVOL into `data/settlement/` (~60 MB, git-ignored).
Kalshi requests are signed when `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY` are set, which only
raises the read limit; the fetcher has no order path. Re-running `fetch` completes a partial
download. Method and the pre-registered rule: `docs/EDGE_EVALUATION_METHOD.md` section 7.

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

## Performance (scope s.22)

`python scripts/benchmark.py` writes `docs/benchmarks.json` with hardware metadata. Reference
run (4 vCPU x86_64, Python 3.12, `OPENBLAS_NUM_THREADS=1`):

| Measure | Result | Target |
|---|---|---|
| Ingestion (parse + normalize, Kalshi deltas / Coinbase tickers) | ~28,000 events/s | ≥ 10,000 |
| Per-event normalization p99 | 0.07 ms | ≤ 5 ms |
| Strategy + signal hot path p99 | 0.36 ms | ≤ 25 ms |
| Replay of a 2 h synthetic market (82k events) | ~15,000 events/s, ~1,300× real time | ≥ 10× |
