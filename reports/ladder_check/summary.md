# Same-venue consistency: Kalshi 15-minute BTC markets vs the hourly ladder

**Decision: REJECT** (rule fixed before the first run: docs/EDGE_EVALUATION_METHOD.md section 8).

* 42 opportunities in 6405 paired hours (0.66%; the rule needs 2%)
* median riskless profit 0.65¢ per pair (the rule needs 1¢)

## Data

* 6405 hours where the 15-minute market closing on the hour and the hourly ladder strikes around its strike both have usable minute quotes (2025-12-15 to 2026-10-06; 6895 on-the-hour 15-minute markets, ladder strikes not found for 310, no usable minute in 180).
* Ladder strike spacing by hour: $100: 4524, $250: 1835, $500: 46.
* Settlement value known for 6403 hours; it agrees with the 15-minute result in 6403.

## Opportunities (riskless after both taker fees, still there a minute later)

* 42 of 6405 hours (0.66%); by pair: 15-minute YES + ladder NO above: 26, 15-minute NO + ladder YES below: 16, ladder YES below + ladder NO above: 0.
* Riskless profit at the next minute's quotes: median +0.65¢, mean +1.44¢, 90th percentile +3.26¢, largest +12.34¢ per pair (each pair pays at least $1).
* With the settlement value (42 of them): mean realised +1.44¢, worst +0.04¢ per pair.

| Month | Paired hours | Opportunities |
|---|---:|---:|
| 2025-12 | 181 | 2 |
| 2026-01 | 542 | 1 |
| 2026-02 | 626 | 0 |
| 2026-03 | 688 | 2 |
| 2026-04 | 691 | 9 |
| 2026-05 | 715 | 7 |
| 2026-06 | 696 | 8 |
| 2026-07 | 718 | 2 |
| 2026-08 | 711 | 7 |
| 2026-09 | 702 | 4 |
| 2026-10 | 135 | 0 |

## How close the prices come to an arbitrage

Over 57920 hour-minutes with all three books quoted:

| Pair | Crossed before fees | Riskless after both fees |
|---|---:|---:|
| 15-minute YES + ladder NO above | 0.958% | 0.302% |
| 15-minute NO + ladder YES below | 0.860% | 0.257% |
| ladder YES below + ladder NO above | 0.000% | 0.000% |

* The 15-minute mid sits outside the ladder's band [mid above, mid below] in 5.19% of minutes.
* Best after-fee result per hour (negative = no arbitrage): median -5.94¢, 90th percentile -0.68¢, 99th +1.27¢, best +12.34¢.

Minute closing quotes only: no depth, and both legs are assumed to fill at the quoted price for 100 contracts.
