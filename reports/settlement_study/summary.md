# Hold-to-settlement study: Kalshi BTC 15-minute markets

**Decision: REJECT** (rule fixed before the first full run: docs/EDGE_EVALUATION_METHOD.md section 7).

* selected 60-minute realised vol, edge ≥ 10¢ (in-sample t -2.23, 2450 trades); no specification made money in-sample either
* out of sample: -0.97c per contract ± 1.24 (2 SE) on 2464 trades
* gate oos_mean_above_2se: FAIL
* gate fees_x1_5_positive: FAIL
* gate two_minute_fill_positive: FAIL
* gate min_trades: pass
* gate top_day_share: FAIL
* gate neighbours_profitable: FAIL
* gate months_positive: FAIL

## Data

* 27549 settled `KXBTC15M` markets with candles (of 27991 listed); YES settled 49.9% of the time. Kalshi's result agrees with its own settlement values in 25064 of the 25065 markets that publish both.
* In-sample: 16529 markets, 2025-12-15 to 2026-06-12. Out-of-sample: 11020 markets, 2026-06-12 to 2026-10-06.
* Coinbase-to-BRTI basis over 27700 quarter-hour marks: median +0.64 bp (5–95%: -1.34 to +2.77 bp).
* Reference data: 445802 Coinbase minutes, 7441 hourly DVOL closes.

## Strategy grid

Buy the side the model favours when its edge after the taker fee is at least θ; one trade per market, filled at the next minute's quote, held to settlement. Cents per contract after fees, ± 2 standard errors clustered by day.

| Volatility | θ (¢) | In-sample trades | In-sample P&L | t | Out-of-sample trades | Out-of-sample P&L | t | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dvol | 0 | 15780 | -2.89 ± 0.81 | -7.1 | 10860 | -2.72 ± 0.88 | -6.2 | 40.8% |
| dvol | 1 | 15382 | -2.96 ± 0.85 | -7.0 | 10689 | -2.73 ± 0.85 | -6.4 | 38.9% |
| dvol | 2 | 14822 | -2.90 ± 0.86 | -6.8 | 10404 | -2.90 ± 0.91 | -6.3 | 36.9% |
| dvol | 3 | 14060 | -2.97 ± 0.88 | -6.8 | 9977 | -2.58 ± 0.95 | -5.4 | 35.3% |
| dvol | 5 | 12394 | -2.91 ± 0.85 | -6.9 | 8953 | -2.38 ± 0.98 | -4.9 | 31.4% |
| dvol | 10 | 8307 | -2.47 ± 0.85 | -5.8 | 6077 | -1.98 ± 0.97 | -4.1 | 23.1% |
| rv60 | 0 | 15535 | -1.73 ± 0.71 | -4.9 | 10765 | -1.67 ± 0.83 | -4.1 | 47.3% |
| rv60 | 1 | 14712 | -1.73 ± 0.76 | -4.6 | 10378 | -1.27 ± 0.85 | -3.0 | 45.4% |
| rv60 | 2 | 13334 | -1.87 ± 0.73 | -5.1 | 9704 | -1.15 ± 0.86 | -2.7 | 42.7% |
| rv60 | 3 | 11630 | -1.67 ± 0.77 | -4.3 | 8807 | -1.26 ± 0.84 | -3.0 | 39.9% |
| rv60 | 5 | 8137 | -1.89 ± 0.85 | -4.5 | 6711 | -1.22 ± 1.00 | -2.4 | 35.1% |
| rv60 | 10 | 2450 | -1.68 ± 1.50 | -2.2 | 2464 | -0.97 ± 1.24 | -1.6 | 29.1% |

## Selected specification out of sample (60-minute realised vol, edge ≥ 10¢)

| Variant | Trades | P&L (¢/contract) | t |
|---|---:|---:|---:|
| Next-minute fill (primary) | 2464 | -0.97 ± 1.24 | -1.6 |
| Fees × 1.5 | 2464 | -1.43 ± 1.24 | -2.3 |
| Fill two minutes later | 1958 | -1.48 ± 1.46 | -2.0 |
| Same-minute fill (optimistic) | 2481 | -0.82 ± 1.25 | -1.3 |

| Month | Trades | P&L (¢/contract) |
|---|---:|---:|
| 2026-06 | 378 | -1.66 ± 2.30 |
| 2026-07 | 653 | -1.51 ± 2.81 |
| 2026-08 | 855 | +0.14 ± 1.70 |
| 2026-09 | 457 | -1.55 ± 3.42 |
| 2026-10 | 121 | -1.66 ± 6.43 |

* Bought YES: 1164 trades, -0.93 ± 2.12¢; bought NO: 1300 trades, -1.01 ± 1.80¢.
* Largest day's share of the P&L: n/a; neighbouring thresholds profitable: 0.0%; months positive: 20.0%.

## Who forecasts settlement better?

Brier score (lower is better) of Kalshi's mid and of the model, and the regression of the outcome on the mid and the model–mid gap: a gap coefficient near zero means the model adds nothing the price does not already contain.

| Minutes before close | Markets | Brier: Kalshi mid | Brier: model (DVOL) | Brier: model (60-min vol) | Gap coefficient (DVOL) | Gap coefficient (60-min vol) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 27162 | 0.1883 | 0.1958 | 0.1894 | -0.009 ± 0.066 | +0.253 ± 0.109 |
| 5 | 26762 | 0.1189 | 0.1314 | 0.1215 | +0.002 ± 0.044 | +0.124 ± 0.087 |
| 2 | 22899 | 0.0713 | 0.0841 | 0.0750 | +0.040 ± 0.039 | +0.057 ± 0.080 |

## Calibration 5 minutes before close

| Kalshi mid | Markets | Mean mid | YES settled | Buy YES at ask (¢) | Buy NO at 1 − bid (¢) |
|---|---:|---:|---:|---:|---:|
| 0.00–0.05 | 3338 | 0.026 | 0.022 ± 0.005 | -0.93 | +0.02 |
| 0.05–0.10 | 2314 | 0.072 | 0.059 ± 0.010 | -2.26 | +0.37 |
| 0.10–0.20 | 2737 | 0.145 | 0.144 ± 0.013 | -1.79 | -1.44 |
| 0.20–0.30 | 1887 | 0.246 | 0.231 ± 0.019 | -3.58 | -0.48 |
| 0.30–0.40 | 1611 | 0.348 | 0.346 ± 0.024 | -2.52 | -2.11 |
| 0.40–0.50 | 1420 | 0.448 | 0.435 ± 0.026 | -3.71 | -1.22 |
| 0.50–0.60 | 1595 | 0.549 | 0.541 ± 0.025 | -3.31 | -1.66 |
| 0.60–0.70 | 1562 | 0.651 | 0.643 ± 0.024 | -3.16 | -1.49 |
| 0.70–0.80 | 1812 | 0.750 | 0.740 ± 0.021 | -3.09 | -1.08 |
| 0.80–0.90 | 2599 | 0.853 | 0.868 ± 0.013 | -0.03 | -3.20 |
| 0.90–0.95 | 2392 | 0.926 | 0.936 ± 0.010 | +0.14 | -2.00 |
| 0.95–1.00 | 3495 | 0.974 | 0.980 ± 0.005 | +0.19 | -1.11 |

Minute bars only: this tests minute-scale strategies, not latency. Fills assume the quoted price had at least 100 contracts behind it.
