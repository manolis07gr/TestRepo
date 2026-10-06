# Cross-market edge evaluation — decision report

Experiment `edge-eval-20261006` · git `3811f8c671f92d121e687b460a80a3cb403d3a66` · config `ac775e935df5` · dataset `7c5562ded542` · seed 20261006

## Decision

* **Real markets (Kalshi / Polymarket BTC contracts): `COLLECT_MORE_DATA`.** No real venue data could be collected from this environment (market-data hosts are blocked by its network policy), so no real-market claim is made. The platform is ready to collect and evaluate as soon as access exists (see *Next steps*).
* Synthetic base case `kalshi_fee`: **`REJECT`** — oos_net_pnl: net P&L -490.20 at base@250ms; CI lower -117.04; stress_latency_viable: net -249.45 at base@1000ms; stress_cost_viable: net -177.92 at fees_x1.5@250ms; parameter_stability: 42% of 12 neighbouring settings profitable
* Synthetic base case `polymarket_fee_speedbump`: **`REJECT`** — oos_net_pnl: net P&L -509.01 at base@250ms; CI lower -105.47; stress_latency_viable: net -341.21 at base@1000ms; stress_cost_viable: net -200.67 at fees_x1.5@250ms; parameter_stability: 50% of 12 neighbouring settings profitable

The synthetic study answers a narrower question than profitability: *how much quote staleness, competition and latency can a lead-lag taker afford after real venue fees?* It uses the full production pipeline (replay → features → signals → risk → latency-aware execution simulator → accounting) on a calibrated synthetic market with a known ground truth.

## 1. Data, instruments and exclusions

* Base case: synthetic BTC index (Markov-switching vol 40%/85%, jumps), reference exchange mid (Coinbase-like, 5 Hz), hourly ladder of 9 "index > K" contracts settling on the 60 s average before expiry (Kalshi BRTI semantics), 48 hours, seed 20261007.
* Market makers re-quote one-tick markets from a view delayed by a log-normal lag (median 350.0 ms); competing arbitrageurs with 120.0 ms latency take quotes stale by > fees + 2¢; noise traders hit the touch at random.
* Fee schedule: `kalshi-standard` (variant: Polymarket crypto taker fee + 150 ms taker speed bump).
* Data-quality exclusions: none (synthetic data is gap-free by construction; the gap/duplicate/crossed-book machinery is exercised by the replay fixtures and tests).

## 2. Strategy, model and mapping versions

* Strategy `fv_taker` v1.0: fair YES probability from the observed reference mid with settlement-exact semantics (point vs 60 s trailing average), volatility from an implied-vol index feed; IOC limit orders at the price where marginal net edge still clears the threshold; positions held to settlement (fees paid once).
* Selected parameters: `{"min_net_edge_bps": 300, "vol_multiplier": 1.1, "why": "best fee-adjusted 60 s mark-out P&L on validation"}`.
* Mappings: synthetic contracts with exact semantics, status `REVIEWED` (never `APPROVED_PAPER`, so no synthetic result can start forward paper trading).

## 3. Train / validation / final-test segmentation

| partition | start | end |
|---|---|---|
| train | 2026-09-01T00:00:00.000000000Z | 2026-09-02T04:00:00.000000000Z |
| validation | 2026-09-02T04:00:00.000000000Z | 2026-09-02T14:00:00.000000000Z |
| final_test | 2026-09-02T14:00:00.000000000Z | 2026-09-03T00:00:00.000000000Z |

* Parameter search on **validation only**: 12 settings (min net edge × vol multiplier), each logged as a hypothesis (one-sided t-test of mean net P&L per event family > 0 (validation window)); Benjamini–Hochberg at α=0.05: 0 rejections.
* Final-test access log: final test 2026-09-02T14:00:00.000000000Z..2026-09-03T00:00:00.000000000Z unlocked once for kalshi_fee with params {'min_net_edge_bps': 300, 'vol_multiplier': 1.1, 'why': 'best fee-adjusted 60 s mark-out P&L on validation'}

## 4. Final-test results (base cost, 250 ms)

| metric | value |
|---|---|
| realized net P&L ($) | -490.20 |
| realized gross P&L ($) | -367.27 |
| fees ($) | 122.93 |
| slippage vs signal price ($) | -14.50 |
| ex-ante expected net P&L ($, true model) | -103.33 |
| ex-ante expected net ¢ / contract | -0.74 |
| fee-adjusted 60 s mark-out P&L ($) | -46.63 |
| signals / orders / fills | 3254 / 3254 / 1401 |
| contracts filled | 13,966 |
| fill rate | 0.429 |
| positions (contracts traded) | 90 |
| hit rate (positions) | 0.111 |
| profit factor | 0.62 |
| max drawdown ($) | 613.49 |
| CVaR 5% per position ($) | -74.39 |
| mean / median signal net edge (bps) | 414.5 / 344.0 |

Risk limits: position limits (contract / family / portfolio exposure) apply; the NAV-triggered daily loss stop is disabled in the edge study so one bad hour cannot silence the rest of the sample. With the production 2% daily loss stop on, the 250 ms base run hit the stop at 16:50 UTC and traded 3 of 10 hourly ladders (net P&L -175.89; 4,164 orders refused).

Bootstrap 95% CI of mean net P&L per event family (hourly ladder), n = 10: [-117.04, 22.22] around -49.02. Hold-to-settlement P&L of one hourly ladder is one correlated bet on BTC, so the family — not the contract — is the unit of independent evidence.

## 5. Results by latency and cost / slippage stress (final test)

Realized net P&L ($):

| cost scenario | 0 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | 5000 ms |
|---|---|---|---|---|---|---|---|
| base | -686.77 | -716.79 | -490.20 | -441.47 | -249.45 | -208.77 | -102.64 |
| fees_x1.5 | -500.45 | -322.94 | -177.92 | -190.02 | -52.01 | -67.75 | -79.38 |
| slip_+1tick | -132.06 | -68.49 | -70.42 | -82.94 | -21.45 | -11.48 | 24.17 |
| fees_x2_slip_+1tick | -80.02 | -47.04 | -28.10 | -36.21 | -25.12 | -9.85 | 37.47 |

Realized hold-to-settlement P&L is dominated by where BTC finished each hour, and each cost scenario trades a different subset of opportunities, so this table need not be monotone in costs. The mark-out and ex-ante tables below measure edge.

Fee-adjusted 60 s mark-out P&L ($) — low-variance edge estimate also available on real data:

| cost scenario | 0 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | 5000 ms |
|---|---|---|---|---|---|---|---|
| base | -9.43 | -113.20 | -46.63 | -61.56 | -21.69 | 3.07 | -3.79 |
| fees_x1.5 | -194.68 | -134.41 | -45.70 | -56.55 | -2.24 | 11.12 | -1.40 |
| slip_+1tick | -62.56 | -21.45 | -33.97 | -52.57 | -28.10 | -28.81 | -6.67 |
| fees_x2_slip_+1tick | -36.24 | -20.62 | -18.64 | -34.70 | -36.05 | -16.57 | -0.65 |

Ex-ante expected net ¢ per contract under the true model (simulation-only diagnostic that removes settlement luck):

| cost scenario | 0 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | 5000 ms |
|---|---|---|---|---|---|---|---|
| base | -0.18 | -0.48 | -0.74 | -1.04 | -1.23 | -1.43 | -1.28 |
| fees_x1.5 | -0.42 | -0.78 | -1.08 | -1.46 | -1.74 | -1.83 | -1.79 |
| slip_+1tick | -0.87 | -1.54 | -1.95 | -2.57 | -2.53 | -2.71 | -2.37 |
| fees_x2_slip_+1tick | -1.61 | -2.38 | -2.30 | -3.02 | -3.41 | -3.64 | -3.31 |

Model risk — same final test, implied-vol vs realized-vol fair value (expected ¢ / contract):

| latency | implied vol | realized vol |
|---|---|---|
| 0 ms | -0.18 | -1.08 |
| 100 ms | -0.48 | -1.21 |
| 250 ms | -0.74 | -1.31 |
| 500 ms | -1.04 | -1.38 |
| 1000 ms | -1.23 | -1.41 |
| 2000 ms | -1.43 | -1.40 |
| 5000 ms | -1.28 | -1.33 |

Venue variant `polymarket_fee_speedbump` (Polymarket crypto fee 0.07·p(1−p), 150 ms taker delay) — decision `REJECT`; realized net P&L ($):

| cost scenario | 0 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | 5000 ms |
|---|---|---|---|---|---|---|---|
| base | -750.86 | -572.49 | -509.01 | -453.66 | -341.21 | -229.48 | -147.48 |
| fees_x1.5 | -324.12 | -233.41 | -200.67 | -186.31 | -142.05 | -116.18 | -82.39 |
| slip_+1tick | -81.13 | -75.67 | -82.85 | -14.65 | -29.57 | -9.51 | -5.69 |
| fees_x2_slip_+1tick | -43.75 | -27.89 | -37.07 | -28.08 | -46.17 | -6.29 | 52.00 |

## 6. Where the edge goes (simulation-only attribution)

Each fill's net edge as the signal saw it (model fair value minus price, fees and buffers) versus its true net edge at fill time (true fair value minus price minus fee). When the perceived edge systematically exceeds the true edge, the strategy is adversely selected: it trades most when its own inputs (stale spot, vol) are wrong.

**Base case `kalshi_fee` (final test, 250 ms)**

| perceived net edge | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| 3–5¢ | 12,839 | 3.41 | -0.74 |
| 5–10¢ | 650 | 6.84 | -0.51 |
| ≥ 10¢ | 477 | 14.55 | -1.07 |

| time to expiry | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| < 60 s | 710 | 5.72 | -0.85 |
| 60–300 s | 2,990 | 3.75 | -0.63 |
| 300–1800 s | 5,962 | 3.94 | -0.72 |
| ≥ 1800 s | 4,304 | 3.82 | -0.83 |

**Base case `polymarket_fee_speedbump` (final test, 250 ms)**

| perceived net edge | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| 3–5¢ | 10,175 | 3.38 | -0.87 |
| 5–10¢ | 470 | 7.07 | -0.84 |
| ≥ 10¢ | 409 | 14.94 | -1.35 |

| time to expiry | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| < 60 s | 480 | 5.49 | -0.97 |
| 60–300 s | 2,070 | 3.72 | -0.94 |
| 300–1800 s | 4,637 | 4.02 | -0.88 |
| ≥ 1800 s | 3,867 | 3.84 | -0.85 |

**Frontier: maker lag 150 ms, competitor 120 ms, 250 ms**

| perceived net edge | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| < 1.5¢ | 7,917 | 1.20 | -1.29 |
| 1.5–2¢ | 2,440 | 1.71 | -1.46 |
| 2–3¢ | 904 | 2.42 | -1.53 |
| 3–5¢ | 530 | 3.78 | -2.06 |
| 5–10¢ | 383 | 7.13 | -1.04 |
| ≥ 10¢ | 292 | 14.83 | -1.91 |

**Frontier: maker lag 350 ms, competitor 120 ms, 250 ms**

| perceived net edge | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| < 1.5¢ | 9,247 | 1.19 | -0.76 |
| 1.5–2¢ | 2,742 | 1.71 | -0.67 |
| 2–3¢ | 1,234 | 2.43 | -0.49 |
| 3–5¢ | 610 | 3.66 | -0.88 |
| 5–10¢ | 340 | 7.34 | -1.13 |
| ≥ 10¢ | 242 | 14.05 | -1.30 |

**Frontier: maker lag 1000 ms, competitor 120 ms, 250 ms**

| perceived net edge | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| < 1.5¢ | 26,096 | 1.20 | 0.14 |
| 1.5–2¢ | 8,948 | 1.71 | 0.51 |
| 2–3¢ | 4,202 | 2.37 | 0.97 |
| 3–5¢ | 1,088 | 3.63 | 1.79 |
| 5–10¢ | 230 | 6.57 | 1.58 |
| ≥ 10¢ | 110 | 16.59 | 7.47 |

**Frontier: maker lag 3000 ms, competitor 120 ms, 250 ms**

| perceived net edge | contracts | perceived ¢ | true ¢ |
|---|---|---|---|
| < 1.5¢ | 68,184 | 1.21 | 0.62 |
| 1.5–2¢ | 30,597 | 1.71 | 0.99 |
| 2–3¢ | 15,980 | 2.33 | 1.37 |
| 3–5¢ | 2,723 | 3.51 | 2.14 |
| 5–10¢ | 408 | 6.88 | 5.61 |
| ≥ 10¢ | 244 | 29.85 | 29.08 |

## 7. Lead-lag estimates and stability by regime

Reference mid → contract mid, true maker lag 350.0 ms (median) plus feed delays. Lags in ms; positive = reference leads.

| contract | CCF lag | HY lag | corr@lag | corr@0 | p | ΔOOS R² | hit rate | econ edge | qualifies |
|---|---|---|---|---|---|---|---|---|---|
| SYNBTC-20260901T0100-T110000 | 200 | 200 | 0.098 | 0.046 | 0.005 | 0.029 | 0.696 | -0.0486 | False |
| SYNBTC-20260901T0200-T110000 | 300 | 200 | 0.054 | 0.018 | 0.250 | -23.400 | 0.923 | -0.0225 | False |
| SYNBTC-20260901T0300-T111000 | 300 | 200 | 0.116 | 0.052 | 0.005 | 0.291 | 0.844 | n/a | False |
| SYNBTC-20260901T0400-T110500 | 300 | 200 | 0.120 | 0.040 | 0.005 | 0.062 | 0.699 | -0.0108 | False |

* `SYNBTC-20260901T0100-T110000` by time to expiry: 10-30m: lag 200 ms, ΔOOS R² 0.337, qualifies False; <10m: lag 200 ms, ΔOOS R² 0.039, qualifies False; >30m: lag 400 ms, ΔOOS R² 0.158, qualifies False
  * Failing gate: FAIL economic: |pred| > hurdle 0.0225 on 0.002197 of OOS rows (min 0.01); mean edge after hurdle -0.04859 (must be > 0); net value per row -0.0001067 vs AR baseline 0 (X must add value)
* `SYNBTC-20260901T0200-T110000` by time to expiry: 10-30m: lag None ms, ΔOOS R² n/a, qualifies False; >30m: lag 300 ms, ΔOOS R² -22.939, qualifies False
  * Failing gate: FAIL significance: max|corr| over positive lags = 0.03864, family-wise circular_shift p=0.25 vs alpha=0.05 (null q95=0.04464, 199 draws)

A statistically real lead (significant, out-of-sample predictive) is necessary but not sufficient: the economic gate requires predicted moves larger than fees + half-spread.

## 8. Profit / loss concentration

| diagnostic | value |
|---|---|
| top contract share of gains | 0.245 |
| top family share of gains | 0.466 |
| top day share of gains | n/a |
| families / days | 10 / 1 |

## 9. Structural (nested-strike) consistency

15116 family snapshots scanned; 0 executable violations after both taker fees (net total 0). synthetic makers re-quote strikes independently with random lags; violations need bid(K_high) - ask(K_low) > both taker fees

## 10. Edge frontier (simulation)

Ex-ante expected net ¢ per contract (true-model value minus price minus fees), implied-vol model, base fees, by maker reaction lag, competitor latency and *our* outbound latency (0 ms is diagnostic only):

| maker lag | competitor | 0 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | 5000 ms |
|---|---|---|---|---|---|---|---|---|
| 150 ms | none | -0.94 | -1.18 | -1.34 | -1.43 | -1.50 | -1.55 | -1.54 |
| 150 ms | 120 ms | -1.04 | -1.25 | -1.38 | -1.50 | -1.51 | -1.54 | -1.54 |
| 350 ms | none | 0.23 | 0.05 | -0.27 | -0.83 | -1.32 | -1.75 | -1.69 |
| 350 ms | 120 ms | -0.25 | -0.46 | -0.74 | -1.11 | -1.40 | -1.79 | -1.71 |
| 1000 ms | none | 1.48 | 1.42 | 1.37 | 1.18 | 0.83 | -0.61 | -1.62 |
| 1000 ms | 120 ms | 0.66 | 0.51 | 0.38 | 0.19 | -0.16 | -1.23 | -1.83 |
| 3000 ms | none | 2.15 | 2.11 | 2.08 | 2.07 | 1.93 | 1.90 | 1.10 |
| 3000 ms | 120 ms | 1.14 | 1.01 | 0.93 | 0.79 | 0.59 | 0.33 | -0.76 |

* At 250 ms with no competing arbitrageur: expected edge is positive only for maker lags 1000 ms, 3000 ms (grid: 150 ms, 350 ms, 1000 ms, 3000 ms).
* At 250 ms with a competing arbitrageur: expected edge is positive only for maker lags 1000 ms, 3000 ms (grid: 150 ms, 350 ms, 1000 ms, 3000 ms).

Contracts filled (same grid):

| maker lag | competitor | 0 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | 5000 ms |
|---|---|---|---|---|---|---|---|---|
| 150 ms | none | 21,839 | 16,631 | 12,426 | 11,071 | 9,140 | 7,388 | 5,235 |
| 150 ms | 120 ms | 21,530 | 16,587 | 12,466 | 10,861 | 9,170 | 7,268 | 5,225 |
| 350 ms | none | 31,284 | 23,349 | 15,268 | 9,887 | 7,623 | 6,825 | 5,875 |
| 350 ms | 120 ms | 29,535 | 21,775 | 14,415 | 9,534 | 7,523 | 6,835 | 5,905 |
| 1000 ms | none | 79,758 | 67,388 | 52,556 | 34,001 | 15,427 | 6,285 | 7,565 |
| 1000 ms | 120 ms | 67,701 | 54,098 | 40,674 | 26,423 | 12,392 | 5,585 | 7,375 |
| 3000 ms | none | 211,486 | 193,976 | 167,894 | 133,580 | 88,413 | 44,718 | 9,254 |
| 3000 ms | 120 ms | 168,693 | 141,456 | 118,136 | 89,037 | 56,734 | 27,869 | 6,464 |

Realized-vol model (model-risk comparison), expected ¢ / contract:

| maker lag | competitor | 0 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | 5000 ms |
|---|---|---|---|---|---|---|---|---|
| 350 ms | 120 ms | -0.90 | -1.10 | -1.25 | -1.41 | -1.51 | -1.52 | -1.51 |
| 3000 ms | none | 2.23 | 2.13 | 2.00 | 1.93 | 1.62 | 1.12 | -0.42 |
| 3000 ms | 120 ms | 0.99 | 0.83 | 0.66 | 0.44 | 0.11 | -0.42 | -1.34 |

## 11. Analytical cost hurdle

Move needed (bps of BTC) for a one-tick stale quote to clear the taker fee + 1¢ net, and the chance it happens inside a 350 ms reaction window (Gaussian vs variance-matched Student-t ν=3), σ = 45%. Opportunities per hour are a rate per hour spent at that time to expiry, before any competition:

| T (s) | z | p0 | fee ¢ | ¢ per bp | move bps | P gauss | P fat | opps/h (fat) |
|---|---|---|---|---|---|---|---|---|
| 120 | 0.0 | 0.50 | 1.75 | 4.55 | 0.7 | 1.3e-01 | 7.9e-02 | 815.8 |
| 120 | 0.5 | 0.69 | 1.48 | 4.01 | 0.8 | 1.1e-01 | 6.9e-02 | 711.5 |
| 120 | 1.0 | 0.84 | 0.91 | 2.76 | 0.9 | 5.2e-02 | 4.3e-02 | 446.2 |
| 120 | 2.0 | 0.98 | 0.12 | 0.62 | 4.2 | 4.9e-19 | 5.9e-04 | 6.1 |
| 300 | 0.0 | 0.50 | 1.75 | 2.88 | 1.1 | 1.7e-02 | 2.6e-02 | 264.0 |
| 300 | 0.5 | 0.69 | 1.48 | 2.54 | 1.2 | 1.1e-02 | 2.2e-02 | 225.2 |
| 300 | 1.0 | 0.84 | 0.91 | 1.75 | 1.5 | 2.1e-03 | 1.3e-02 | 132.9 |
| 300 | 2.0 | 0.98 | 0.12 | 0.39 | 6.7 | 4.9e-45 | 1.5e-04 | 1.6 |
| 900 | 0.0 | 0.50 | 1.75 | 1.66 | 2.0 | 3.5e-05 | 5.6e-03 | 57.7 |
| 900 | 0.5 | 0.69 | 1.48 | 1.47 | 2.1 | 1.1e-05 | 4.7e-03 | 48.6 |
| 900 | 1.0 | 0.84 | 0.91 | 1.01 | 2.5 | 1.0e-07 | 2.7e-03 | 27.7 |
| 900 | 2.0 | 0.98 | 0.12 | 0.23 | 11.5 | 7.3e-131 | 2.9e-05 | 0.3 |
| 1,800 | 0.0 | 0.50 | 1.75 | 1.17 | 2.8 | 5.0e-09 | 2.0e-03 | 21.1 |
| 1,800 | 0.5 | 0.69 | 1.48 | 1.04 | 2.9 | 5.4e-10 | 1.7e-03 | 17.7 |
| 1,800 | 1.0 | 0.84 | 0.91 | 0.71 | 3.6 | 5.0e-14 | 9.7e-04 | 10.0 |
| 1,800 | 2.0 | 0.98 | 0.12 | 0.16 | 16.3 | 1.2e-258 | 1.0e-05 | 0.1 |
| 3,600 | 0.0 | 0.50 | 1.75 | 0.83 | 3.9 | 1.3e-16 | 7.4e-04 | 7.6 |
| 3,600 | 0.5 | 0.69 | 1.48 | 0.73 | 4.2 | 1.7e-18 | 6.2e-04 | 6.4 |
| 3,600 | 1.0 | 0.84 | 0.91 | 0.50 | 5.0 | 1.8e-26 | 3.5e-04 | 3.6 |
| 3,600 | 2.0 | 0.98 | 0.12 | 0.11 | 23.0 | 0.0e+00 | 3.7e-06 | 0.0 |

## 12. Known limitations and unresolved issues

* No real venue data was examined (network policy). All profitability numbers are model-world results; they bound what is plausible, they do not estimate real P&L.
* Synthetic makers re-quote from a lagged view with one-tick spreads and fixed depth distributions; real quoting (inventory skew, widening into events, cancels) differs.
* Competition is modelled as one arbitrageur class with a single latency and threshold.
* The ex-ante 'truth' is the model world's fair value (current regime vol, exact 60 s averaging); it measures edge against well-informed makers, not against real ones.
* Fee formulas reflect official documentation as of 2026-10 (Kalshi 0.07·C·P·(1−P) rounded up per order; Polymarket crypto 0.07·C·p·(1−p)); maker rebates are not credited; verify per-series values at runtime.
* Kalshi WebSocket market data requires API credentials; without them the collector falls back to REST polling, which cannot measure sub-second lead-lag.
* Cross-venue BTC contracts settle on different references (Kalshi BRTI 60 s average vs Polymarket Binance BTCUSDT candles or Chainlink TWAP): they are basis trades, never riskless equivalents; the mapping registry rejects them as equivalent.
* Parsers were written from documented schemas without live payloads; schema drift is quarantined, not silently accepted, and must be checked on first live collection.

## 13. Next steps to reach a real-market decision

* Allow the market-data hosts in the environment network policy (Kalshi external-api, Polymarket clob/gamma/data-api, Coinbase exchange, Deribit) and add Kalshi API credentials via environment variables (never config files).
* `cma collect --duration 1209600` for ≥ 14 days to capture synchronized books (`KXBTCD`, Polymarket BTC markets, Coinbase BTC-USD, Deribit DVOL/options).
* Review and approve mappings (`config/mappings/registry`), build a dataset (`cma dataset build`), run `cma stress` and `cma leadlag`; publish only manifests that pass `validate_publication`.
* Measure the real maker reaction-lag distribution and competitor fill speed: the frontier above says whether any latency budget can be profitable before money is spent on infrastructure.

