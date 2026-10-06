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

## 4. Metrics that separate edge from luck

* **Realized net P&L** (hold to settlement) — the official criterion, but each hourly ladder
  is one correlated bet on BTC, so it is noisy; CIs bootstrap *families*, not contracts.
* **Fee-adjusted mark-outs** (mid at +1/5/30/60 s minus fill price, minus fees) — low
  variance and available on real data; used for parameter selection on validation.
* **Ex-ante expected P&L** (simulation only): each fill valued at the *true* fair value at
  fill time from the true index path and vol. It removes settlement noise entirely and is
  the cleanest estimate of edge in the model world.

## 5. Protocol

1. Frontier: sweep maker lag × competitor latency × our latency (0 ms diagnostic only) for
   the implied-vol model, plus the realized-vol model on selected scenarios (model risk).
2. Base case: 48 h market → chronological train / validation / final test; 12 settings
   searched on validation only (logged; Benjamini–Hochberg); final test unlocked once;
   full latency × 4 cost-scenario grid; family bootstrap CI; promotion gates.
3. Venue variant: Polymarket fee + 150 ms taker delay on the same market dynamics.
4. Lead-lag discovery on reference → contract mids (CCF + Hayashi–Yoshida, block-permutation
   significance, out-of-sample predictive and economic gates), stratified by time to expiry.
5. Structural scan for nested-strike violations net of both taker fees.

Synthetic results bound what is plausible; they cannot promote a strategy (synthetic
mappings are never APPROVED_PAPER) and they are not estimates of real-market P&L.
