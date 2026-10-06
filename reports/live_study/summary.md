# Live staleness study: Kalshi BTC vs Coinbase

Window 2026-10-06T14:16:31.919540229Z to 2026-10-06T16:12:36.154993494Z (1.93 h). Reference updates: 56114; vol 0.364.
Quote updates by series: {'KXBTC15M': 106820, 'KXBTCD': 486573}
Moves found by threshold (bps in 1 s): {'3': 73, '5': 9, '10': 0}
Kalshi book replay: 702 books, 1036373 snapshots (308 after a sequence restart), 136685 deltas; 0 crossed updates skipped, 1034609 sequence gaps, 0 negative sizes, 10 books invalid at the end.

## Decision: `COLLECT_MORE_DATA` (stale-quote taker on these books)

* positive mean edge: 5 bps at 100 ms, 5 bps at 250 ms
* 3 bps at 100 ms: -0.46¢ ± 0.29 (2 SE), 632 samples on 72 moves
* 3 bps at 250 ms: -0.71¢ ± 0.28 (2 SE), 632 samples on 72 moves
* 5 bps at 100 ms: +1.24¢ ± 1.20 (2 SE), 91 samples on 9 moves
* 5 bps at 250 ms: +0.91¢ ± 1.27 (2 SE), 91 samples on 9 moves

Rule, fixed before the first live run: REJECT when no move threshold shows a positive pooled market-anchored edge at 100 ms or 250 ms and the 5 bps edge at 100 ms is negative by more than 2 move-clustered SE on >= 30 moves; otherwise COLLECT_MORE_DATA. One live window never promotes to paper.

## Market-anchored executable edge after fees (primary latency measure)

Fair value = Kalshi's own mid 1 s before the move + the model's predicted change from the BTC move; edge = fair - price - taker fee, for the book observed L ms after the move. Reaction = time until Kalshi's mid covered half the model's predicted repricing (the maker reaction lag); lifetime = how long the quote a taker would hit survived (ends at any re-quote or fill). Standard errors are clustered by move (every in-play strike reacts to the same move).

All series pooled:

| move ≥ bps | samples | moves | reaction median ms | lifetime median ms | 0 ms: mean ± 2 SE ¢ (share>0) | 100 ms: mean ± 2 SE ¢ (share>0) | 250 ms: mean ± 2 SE ¢ (share>0) | 500 ms: mean ± 2 SE ¢ (share>0) | 1000 ms: mean ± 2 SE ¢ (share>0) |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 632 | 72 | 0 | 642 | -0.57 ± 0.26 (0.28) | -0.46 ± 0.29 (0.30) | -0.71 ± 0.28 (0.26) | -1.03 ± 0.25 (0.20) | -1.14 ± 0.23 (0.19) |
| 5 | 91 | 9 | 247 | 481 | 0.92 ± 0.77 (0.58) | 1.24 ± 1.20 (0.74) | 0.91 ± 1.27 (0.66) | -0.13 ± 0.64 (0.52) | -0.50 ± 0.67 (0.41) |

By series:

| move ≥ bps | series | samples | lifetime p25 / median / p75 ms | beyond 30 s | 0 ms: share>0 / mean ¢ | 100 ms: share>0 / mean ¢ | 250 ms: share>0 / mean ¢ | 500 ms: share>0 / mean ¢ | 1000 ms: share>0 / mean ¢ |
|---|---|---|---|---|---|---|---|---|---|
| 3 | KXBTC15M | 41 | 322 / 690 / 1866 | 0.07 | 0.51 / 0.36 | 0.61 / 0.75 | 0.61 / 0.73 | 0.54 / 0.49 | 0.46 / 0.30 |
| 3 | KXBTCD | 591 | 216 / 640 / 2782 | 0.17 | 0.26 / -0.63 | 0.28 / -0.54 | 0.23 / -0.81 | 0.18 / -1.14 | 0.17 / -1.24 |
| 5 | KXBTC15M | 5 | 256 / 274 / 398 | 0.00 | 1.00 / 3.78 | 1.00 / 4.44 | 1.00 / 4.44 | 0.80 / 1.82 | 1.00 / 2.29 |
| 5 | KXBTCD | 86 | 266 / 544 / 967 | 0.06 | 0.56 / 0.75 | 0.72 / 1.05 | 0.64 / 0.70 | 0.50 / -0.25 | 0.37 / -0.67 |

No-move baseline (anchored): n=7640, share with positive edge 0.02, mean -1.46 ¢, p95 -0.43 ¢.

## Model-absolute edge (secondary: includes level disagreement)

| move ≥ bps | series | samples | lifetime p25 / median / p75 ms | beyond 30 s | 0 ms: share>0 / mean ¢ | 100 ms: share>0 / mean ¢ | 250 ms: share>0 / mean ¢ | 500 ms: share>0 / mean ¢ | 1000 ms: share>0 / mean ¢ |
|---|---|---|---|---|---|---|---|---|---|
| 3 | KXBTC15M | 41 | 322 / 690 / 1866 | 0.07 | 0.66 / 1.02 | 0.66 / 1.41 | 0.63 / 1.40 | 0.63 / 1.15 | 0.63 / 0.96 |
| 3 | KXBTCD | 591 | 216 / 640 / 2782 | 0.17 | 0.47 / -0.39 | 0.46 / -0.30 | 0.43 / -0.56 | 0.41 / -0.89 | 0.39 / -0.99 |
| 5 | KXBTC15M | 5 | 256 / 274 / 398 | 0.00 | 1.00 / 4.73 | 1.00 / 5.38 | 1.00 / 5.38 | 1.00 / 2.77 | 1.00 / 3.23 |
| 5 | KXBTCD | 86 | 266 / 544 / 967 | 0.06 | 0.62 / 0.81 | 0.65 / 1.12 | 0.57 / 0.77 | 0.49 / -0.18 | 0.43 / -0.60 |

No-move baseline (model-absolute): n=7640, share with positive edge 0.62, mean 1.15 ¢, p95 4.59 ¢.

## Lead-lag (reference mid -> contract mid)

* KALSHI:KXBTC15M-26OCT061115-15 (16006 updates): lag 100 ms (HY 0 ms), p 0.005, ΔOOS R² -0.048, qualifies False
* KALSHI:KXBTC15M-26OCT061145-45 (13834 updates): lag 100 ms (HY 0 ms), p 0.005, ΔOOS R² -0.056, qualifies False
* KALSHI:KXBTCD-26OCT0612-T86199.99 (13920 updates): lag 100 ms (HY 0 ms), p 0.005, ΔOOS R² 0.065, qualifies False
* KALSHI:KXBTC15M-26OCT061045-45 (13833 updates): lag 100 ms (HY 0 ms), p 0.005, ΔOOS R² -1.882, qualifies False

