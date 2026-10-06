# Edge evaluation method

The evaluation (`cma evaluate`, `cma.research.evaluation`) answers: *can a lead-lag taker
on BTC price contracts make money after real fees, and what market conditions would it
need?* It never treats correlation as an edge and never reports P&L without costs.

## 1. Analytical hurdle (`cma.research.hurdle`)

For a one-tick market around fair value p0 and a stale ask a = p0 + tick/2, a trade clears
a net threshold θ only if the move lifted fair value to p* = a + fee(a) + θ. For a log-normal
digital the required log move is exactly `r* = σ√T · (Φ⁻¹(p*) − Φ⁻¹(p0))`; the probability
of such a move inside a reaction window Δ is evaluated for Gaussian and variance-matched
Student-t(3) returns. This bounds opportunity frequency *before* competition.

## 2. Calibrated synthetic markets (`cma.research.synthetic`)

* Index: log random walk on a 50 ms grid, Markov-switching vol (40% / 85%), Poisson jumps.
* Reference exchange (Coinbase-like) top of book at 5 Hz with a mean-reverting basis.
* Implied-vol index (DVOL-like) every 5 s: true regime vol ± 3%.
* Kalshi-like hourly ladder of "index > K" contracts settling on the 60 s average.
* Makers: one-tick quotes around the true fair value as of a log-normally lagged view
  (median = the swept *maker lag*), re-quoted with hysteresis; depth drawn per level.
* Noise traders hit the touch; competing arbitrageurs with their own latency take quotes
  stale by more than fees + 2¢ (the swept *competitor latency*).
* Feed delays (25 ms reference, 60 ms prediction venue) with jitter, monotone per connection.

Calibration anchors (public evidence gathered 2026-10, see the decision report): Polymarket
BTC 15-minute quotes reprice after large Binance moves with a median ≈ 347 ms; spreads are
usually one tick; bots dominate short-dated crypto taker flow; both venues charge crypto
takers 0.07·p(1−p) per contract; Polymarket adds a 150 ms taker delay.

## 3. Pipeline under test

Every run replays the synthetic event stream through the production code: normalization,
observed books, watermarked features, the fair-value strategy (semantics-exact 60 s
average), signal gates, risk limits, the latency-aware simulator and Decimal accounting.
Position limits (contract / family / portfolio exposure) stay on. The NAV-triggered daily
loss stop is disabled in the edge study: it is a loss control, not an edge source, and it
truncates samples path-dependently (one bad hour silences the rest of the UTC day). The
base case re-runs its criterion cell with the production stop on and reports what it did.

## 4. Metrics that separate edge from luck

* **Realized net P&L** (hold to settlement) — the official criterion, but each hourly ladder
  is one correlated bet on BTC, so it is noisy; CIs bootstrap *families*, not contracts.
* **Fee-adjusted mark-outs** (mid at +1/5/30/60 s minus fill price, minus fees) — low
  variance and available on real data; used for parameter selection on validation.
* **Ex-ante expected P&L** (simulation only): each fill valued at the *true* fair value at
  fill time from the true index path and vol. It removes settlement noise entirely and is
  the cleanest estimate of edge in the model world. The synthetic truth and the production
  fair-value model are cross-checked in a unit test (identical inputs → identical
  probabilities, inside and before the 60 s averaging window).
* **Edge attribution** (simulation only): per fill, the net edge the signal *believed* it
  had versus the true net edge at fill time, bucketed by perceived edge and by time to
  expiry. A perceived edge that systematically exceeds the true edge is adverse selection
  (the strategy trades most when its own spot or vol input is stale).

## 5. Protocol

1. Frontier: sweep maker lag × competitor latency × our latency (0 ms diagnostic only) for
   the implied-vol model, plus the realized-vol model on selected scenarios (model risk).
2. Base case: 48 h market → chronological train / validation / final test; 12 settings
   searched on validation only, each logged as a hypothesis tested on *event-family* P&L
   (one-sided t-test; Benjamini–Hochberg), selected by validation 60 s mark-outs; final
   test unlocked once; full latency × 4 cost-scenario grid; family bootstrap CI;
   promotion gates.
3. Venue variant: Polymarket fee + 150 ms taker delay on the same market dynamics.
4. Lead-lag discovery on reference → contract mids (CCF + Hayashi–Yoshida, block-permutation
   significance, out-of-sample predictive and economic gates), stratified by time to expiry.
5. Structural scan for nested-strike violations net of both taker fees.

Synthetic results bound what is plausible; they cannot promote a strategy (synthetic
mappings are never APPROVED_PAPER) and they are not estimates of real-market P&L.

## 6. Real-market check (`cma.research.live_study`)

The synthetic study says what maker speed a latency taker needs; the live study measures
what real Kalshi makers do. It runs on the raw store, on the *receive-time* timeline of the
collecting machine (what that machine could actually have acted on).

1. **Capture.** `scripts/live_study.py collect --duration 6900` records Kalshi's
   authenticated WebSocket order books for the BTC above-strike series (`KXBTCD`,
   `KXBTC15M`; new markets subscribed within a minute of listing) next to the Coinbase
   BTC-USD ticker.
2. **Replay.** `stream_quotes` rebuilds every book in one pass and keeps only top-of-book
   changes. A momentarily crossed book (the deltas of one match arrive one at a time) is
   skipped rather than invalidated; a snapshot whose sequence restarted (re-subscription
   after a reconnect) replaces the book; sequence gaps and negative sizes invalidate a book
   until its next snapshot. Per-venue counters land in `summary.json` (`book_quality`).
3. **Moves.** A move is a 1 s Coinbase log return of at least 3, 5 or 10 bps (5 s cooldown).
   Each move is paired with every contract 90 s to 6 h from close whose model fair value is
   in 10–90¢ and moves by at least 1¢ (log-normal digital on the 60 s settlement average,
   volatility from Deribit DVOL).
4. **Maker reaction.** Time from seeing the move until Kalshi's mid covers half the model's
   predicted repricing (0 if it already had; right-censored at 30 s, censoring-aware
   median). This is the quantity the synthetic frontier is indexed by. The life of the
   quote a taker would hit is reported too, but it ends at any re-quote or fill and so
   understates the repricing lag.
5. **Executable edge.** The book as observed L = 0/100/250/500/1000 ms after the move,
   valued at Kalshi's own mid 1 s before the move plus the model's change in fair value
   (market-anchored: removes level disagreement such as tails, vol and basis, keeps the
   delta a latency trader exploits), minus the price and the Kalshi taker fee for 10
   contracts (rounded up per order). The same valuation at fixed 10 s times without a move
   is the no-news baseline (about minus half the spread plus the fee).
6. **Uncertainty.** Every in-play strike reacts to the same move, so standard errors are
   cluster-robust (CR1) with the move as the cluster, pooled over series per threshold.
7. **Decision rule (fixed before the first live run).** `REJECT` the stale-quote taker when
   no threshold shows a positive pooled market-anchored edge at 100 or 250 ms and the 5 bps
   edge at 100 ms is negative by more than two clustered standard errors on at least 30
   moves; otherwise `COLLECT_MORE_DATA`. One live window never promotes to paper:
   `FORWARD_PAPER_CANDIDATE` needs the ≥ 14-day collection, out-of-sample test and the
   promotion gates.

`scripts/live_study.py analyze --out reports/live_study` writes `summary.json` and
`summary.md`; `cma report --evaluation reports/edge_evaluation/evaluation.json --out
reports/edge_evaluation --live reports/live_study/summary.json` puts the live call and a
real-market section into the decision report, `decision.json` and the HTML page.

## 7. Hold-to-settlement check on history (`cma.research.settlement_study`)

The live study tests a *latency* taker. Its no-move baseline also showed the options-style
model disagreeing with Kalshi's quotes by about a cent after fees at random times; that gap
does not close within seconds, so the question is whether it pays **at settlement**. History
answers that without waiting: Kalshi serves every settled market with its result and
1-minute YES bid/ask candles (`/historical/...` before its archive cutoff), and Coinbase and
Deribit serve 1-minute BTC-USD candles and DVOL. `scripts/settlement_study.py fetch`
downloads them (resumable, rate limited, signed when the key is set; it cannot trade).

This method and the decision rule were fixed and committed **before the first run on the
full history** (only a two-day sample was inspected, for data quality).

1. **Contract.** `KXBTC15M`: YES pays $1 when the 60 s average of CF Benchmarks' BRTI before
   close is at least the same average before open (the market's floor strike). Each
   market's strike and settlement value are BRTI averages at quarter-hour marks.
2. **Decision minutes.** Candles ending 2 to 13 minutes after open (the first minute starts
   on an empty book; from 1 minute before close the settlement average is under way). The
   book must be sane (0 < bid < ask < 1, spread <= 10¢) and the model fair value in 10–90¢.
3. **Model.** Log-normal digital on the 60 s settlement average (`prob_above`,
   `AVG_60S_BEFORE`). Spot: the Coinbase close at the decision minute times the
   Coinbase-to-BRTI basis, the median over the last 8 quarter-hour marks (Coinbase minute
   typical price vs the BRTI mark; only marks already published). Volatility: Deribit DVOL
   (primary) or the trailing 60-minute realised volatility of Coinbase 1-minute returns.
4. **Trade.** In each market, the first decision minute where the model's edge after the
   Kalshi taker fee on the better side (buy YES at the ask, or NO at 1 − bid) is at least
   θ ∈ {0, 1, 2, 3, 5, 10}¢. One trade per market, 100 contracts, held to settlement.
5. **Fill.** Primary: the quote at the end of the *next* minute. A mispricing that only
   lasted until the next quote update is not one a minute-scale strategy can trade, and this
   removes the stale-quote effect of minute bars. Optimistic: the decision minute's quote.
   Stress: two minutes later.
6. **Split and selection.** Chronological: the first 60% of markets by close time are
   in-sample, the rest out-of-sample. The specification (volatility input × θ) with the
   highest in-sample t-statistic among those with at least 200 trades is the one tested out
   of sample. Standard errors cluster by UTC day.
7. **Decision.** `FORWARD_PAPER_CANDIDATE` when the selected specification's out-of-sample
   mean P&L is more than two standard errors above zero and every gate holds: still
   positive with fees × 1.5 and with the two-minute fill, at least 200 trades, no single day
   above 70% of the P&L, at least 60% of the neighbouring thresholds profitable, positive in
   at least half of the months. `REJECT` when its out-of-sample mean is <= 0; otherwise
   `COLLECT_MORE_DATA`. Paper trading, not this test, is the next step after a pass.
8. **Diagnostics.** Brier score and log loss of Kalshi's mid and of the model at 10, 5 and 2
   minutes before close; the regression of outcomes on the mid and the model–mid gap (does
   the model add anything the price does not already contain?); calibration by price
   bucket, including the favourite–longshot ends.

`scripts/settlement_study.py analyze --out reports/settlement_study` writes `summary.json`
and `summary.md`.
