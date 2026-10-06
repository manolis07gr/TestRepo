# Architecture

```
[Venue adapters]               [Reference adapters]
 Kalshi WS/REST (+REST poller)   Coinbase / Binance spot, Deribit options & DVOL, file series
 Polymarket CLOB WS/REST
        \                           /
         -> FeedSession / PollingSession -> RawRecorder (append-only, hashed, deduped)
                         |                         \-> Quarantine + IncidentLog
                  adapter.parse (vendor JSON stops here)
                         |
                  Normalizer (process_ts, clock drift, dedup, probability bounds)
                         |
                  BookManager / L2BookBuilder (snapshot+delta, gaps, crossed)
                         |
          MappingRegistry (reviewed semantics, equivalence)
                         |
          MarketState + features (watermarked) + fair values
           /             |               \
   lead-lag        implied probability    structural / cross-venue
           \             |               /
                SignalEngine (gates, edge, threshold, TTL)
                         |
                RiskEngine (limits, daily stop, kill switch)
                         |
                TradingCore  <- one implementation, two drivers
           /                         \
   ReplayEngine (historical)      PaperTradingSession (live, wall clock)
           \                         /
       ExecutionSimulator (venue-time books, latency, queue, fees) -> Portfolio
                         |
     metrics, stress grids, gates, manifests, decision reports, health
```

## Time model

Every event carries `source_ts_ns` (venue clock), `recv_ts_ns` (our clock) and, once
processed, `process_ts_ns`; none is ever overwritten. A source timestamp ahead of our clock
by more than `max_clock_drift_ms` flags `CLOCK_DRIFT` instead of being corrected, and
`venue_ts_ns = min(source, recv)` so the simulator can never exploit a fast venue clock.

The replay engine schedules each event twice in one priority queue keyed by
`(time, ActionKind, sequence)`: at its venue time for the simulator and at its receive time
(+ optional extra feed delay) for strategies. Order arrivals, cancel arrivals, fill acks,
mark-outs, timers, closes and settlements are scheduled in the same queue. At equal
timestamps venue events precede our arrivals (liquidity that disappears "at the same time"
is gone before we arrive).

## Execution simulator

* Taker orders walk the *available* book at arrival (recorded size minus the size we already
  consumed), never through their limit, with optional stress slippage that also respects
  the limit; IOC remainders cancel, FOK is all-or-nothing, GTC/GTD remainders rest.
* Resting orders join the back of the queue (`queue_ahead` = displayed size at arrival).
  They fill only on trades at their price after the queue ahead is consumed, on trades
  through their price, or when opposite liquidity crosses them. A touch never fills.
  Cancels reach the venue after the cancel latency; fills can happen in between.
* Venue speed bumps (e.g. Polymarket's taker delay) add to taker arrival times.
* Fees use the contract's versioned schedule with Kalshi-style per-order rounding.

## Accounting

Positions are signed YES quantities with exact Decimal cash; average-cost realized P&L,
settlement (YES/NO/void/scalar), worst-case exposure `max(0, C − Q, C)` and the invariant
`NAV = initial + realized − fees + unrealized` (checked in tests).

## Research hygiene

Chronological train / validation / locked final test (`LockedDataset`, access logged),
purged + embargoed walk-forward splits, train-only scalers, Benjamini–Hochberg over every
logged hypothesis, deflated Sharpe, family-level bootstrap CIs, concentration diagnostics,
and promotion gates that can only end in REJECT, COLLECT_MORE_DATA,
FORWARD_PAPER_CANDIDATE or CONTINUE_PAPER — never "live".
