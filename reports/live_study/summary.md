# Live staleness study: Kalshi BTC vs Coinbase

Window 2026-10-06T14:16:31.919540229Z to 2026-10-06T16:12:36.154993494Z (1.93 h). Reference updates: 56114; vol 0.364.
Quote updates by series: {'KXBTC15M': 1150767, 'KXBTCD': 1420237}
Moves found by threshold (bps in 1 s): {'3': 73, '5': 9, '10': 0}
Kalshi book replay: 702 books, 10238988 deltas and 1036373 snapshots applied; 12 gaps in the subscription sequence (2 subscriptions), 2331 updates skipped while a book awaited a snapshot, 0 crossed states skipped, 0 negative sizes; 4 books invalid at the end.

## Decision: `COLLECT_MORE_DATA` (stale-quote taker on these books)

* positive mean edge: 5 bps at 100 ms
* 3 bps at 100 ms: -0.78¢ ± 0.25 (2 SE), 632 samples on 72 moves
* 3 bps at 250 ms: -0.95¢ ± 0.24 (2 SE), 632 samples on 72 moves
* 5 bps at 100 ms: +0.11¢ ± 0.64 (2 SE), 91 samples on 9 moves
* 5 bps at 250 ms: -0.15¢ ± 0.71 (2 SE), 91 samples on 9 moves

Rule, fixed before the first live run: REJECT when no move threshold shows a positive pooled market-anchored edge at 100 ms or 250 ms and the 5 bps edge at 100 ms is negative by more than 2 move-clustered SE on >= 30 moves; otherwise COLLECT_MORE_DATA. One live window never promotes to paper.

## Market-anchored executable edge after fees (primary latency measure)

Fair value = Kalshi's own mid 1 s before the move + the model's predicted change from the BTC move; edge = fair - price - taker fee, for the book observed L ms after the move. Reaction = time until Kalshi's mid covered half the model's predicted repricing (the maker reaction lag); lifetime = how long the quote a taker would hit survived (ends at any re-quote or fill). Standard errors are clustered by move (every in-play strike reacts to the same move).

All series pooled:

| move ≥ bps | samples | moves | reaction median ms | lifetime median ms | 0 ms: mean ± 2 SE ¢ (share>0) | 100 ms: mean ± 2 SE ¢ (share>0) | 250 ms: mean ± 2 SE ¢ (share>0) | 500 ms: mean ± 2 SE ¢ (share>0) | 1000 ms: mean ± 2 SE ¢ (share>0) |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 632 | 72 | 0 | 647 | -0.93 ± 0.24 (0.22) | -0.78 ± 0.25 (0.24) | -0.95 ± 0.24 (0.21) | -1.06 ± 0.26 (0.19) | -1.10 ± 0.26 (0.18) |
| 5 | 91 | 9 | 0 | 374 | -0.06 ± 0.50 (0.51) | 0.11 ± 0.64 (0.52) | -0.15 ± 0.71 (0.46) | -0.30 ± 0.70 (0.44) | -0.97 ± 1.06 (0.31) |

By series:

| move ≥ bps | series | samples | lifetime p25 / median / p75 ms | beyond 30 s | 0 ms: share>0 / mean ¢ | 100 ms: share>0 / mean ¢ | 250 ms: share>0 / mean ¢ | 500 ms: share>0 / mean ¢ | 1000 ms: share>0 / mean ¢ |
|---|---|---|---|---|---|---|---|---|---|
| 3 | KXBTC15M | 41 | 146 / 574 / 1755 | 0.10 | 0.46 / 0.10 | 0.54 / 0.69 | 0.54 / 0.54 | 0.54 / 0.58 | 0.49 / 0.62 |
| 3 | KXBTCD | 591 | 161 / 648 / 4478 | 0.18 | 0.20 / -1.00 | 0.22 / -0.88 | 0.19 / -1.05 | 0.17 / -1.18 | 0.16 / -1.22 |
| 5 | KXBTC15M | 5 | 47 / 63 / 329 | 0.00 | 1.00 / 2.94 | 1.00 / 2.44 | 1.00 / 2.42 | 1.00 / 2.24 | 0.80 / 2.27 |
| 5 | KXBTCD | 86 | 84 / 419 / 716 | 0.08 | 0.48 / -0.24 | 0.49 / -0.02 | 0.43 / -0.30 | 0.41 / -0.45 | 0.28 / -1.16 |

No-move baseline (anchored): n=7640, share with positive edge 0.03, mean -1.45 ¢, p95 -0.41 ¢.

## Model-absolute edge (secondary: includes level disagreement)

| move ≥ bps | series | samples | lifetime p25 / median / p75 ms | beyond 30 s | 0 ms: share>0 / mean ¢ | 100 ms: share>0 / mean ¢ | 250 ms: share>0 / mean ¢ | 500 ms: share>0 / mean ¢ | 1000 ms: share>0 / mean ¢ |
|---|---|---|---|---|---|---|---|---|---|
| 3 | KXBTC15M | 41 | 146 / 574 / 1755 | 0.10 | 0.59 / 0.51 | 0.63 / 1.10 | 0.59 / 0.94 | 0.59 / 0.98 | 0.63 / 1.03 |
| 3 | KXBTCD | 591 | 161 / 648 / 4478 | 0.18 | 0.40 / -0.87 | 0.42 / -0.75 | 0.40 / -0.92 | 0.38 / -1.05 | 0.37 / -1.09 |
| 5 | KXBTC15M | 5 | 47 / 63 / 329 | 0.00 | 1.00 / 3.29 | 1.00 / 2.78 | 1.00 / 2.76 | 1.00 / 2.59 | 1.00 / 2.61 |
| 5 | KXBTCD | 86 | 84 / 419 / 716 | 0.08 | 0.50 / -0.35 | 0.49 / -0.14 | 0.45 / -0.42 | 0.44 / -0.56 | 0.36 / -1.27 |

No-move baseline (model-absolute): n=7640, share with positive edge 0.62, mean 1.15 ¢, p95 4.59 ¢.

## Lead-lag (reference mid -> contract mid)

* KALSHI:KXBTC15M-26OCT061115-15 (177282 updates): Kalshi moved first (peak at -100 ms) (HY -100 ms), p 0.030, ΔOOS R² -0.021, qualifies False
* KALSHI:KXBTC15M-26OCT061215-15 (173405 updates): Kalshi moved first (peak at -500 ms) (HY -200 ms), p 0.005, ΔOOS R² -0.093, qualifies False
* KALSHI:KXBTC15M-26OCT061145-45 (172268 updates): Kalshi moved first (peak at -500 ms) (HY -100 ms), p 0.005, ΔOOS R² -0.037, qualifies False
* KALSHI:KXBTC15M-26OCT061100-00 (134730 updates): Coinbase led by 100 ms (HY 0 ms), p 0.005, ΔOOS R² -0.173, qualifies False

