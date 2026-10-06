"""Calibrated synthetic cross-market generator (simulation studies, fixtures, replay tests).

World model (all parameters explicit and seeded):

* An index ("true" BTC price, BRTI-like) follows a log random walk on a 50 ms grid with
  two-state Markov-switching volatility and Poisson jumps.
* A reference exchange (Coinbase-like) publishes top-of-book snapshots at Poisson times;
  its mid tracks the index with a small mean-reverting basis.
* A prediction venue (Kalshi-like) lists an hourly ladder of "index above K" contracts that
  settle on the index's 60-second average before expiry. Each contract's book is quoted by
  a market maker who reprices from a view of the index delayed by a log-normally
  distributed reaction lag (the quantity that creates or destroys lead-lag edge).
* Noise traders hit the touch at Poisson times; optional competing arbitrageurs with their
  own latency pick off quotes that are stale by more than fees + margin, removing the edge
  before slower participants arrive.
* Every event carries a venue timestamp and a receive timestamp (feed delay + jitter, per
  connection monotone), exactly like recorded data.

The point is not to claim real markets look like this; it is to measure *how much
quote staleness, competition and latency the economics can tolerate* (edge frontier), and to
exercise the full pipeline end to end on data with a known ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from cma.domain.enums import (
    BookSide,
    ContractStatus,
    DeltaMode,
    LiquidityRole,
    MappingStatus,
    Operator,
    QualityFlag,
    SettlementOutcome,
    Side,
    Venue,
)
from cma.domain.fees import get_fee_schedule
from cma.domain.models import (
    BookDeltaEvent,
    BookLevel,
    BookSnapshotEvent,
    ContractMapping,
    LevelChange,
    MarketEvent,
    PredictionContract,
    Settlement,
    TradeEvent,
    instrument_key,
)
from cma.domain.time import NS_PER_MS, NS_PER_S, iso_from_ns

SECONDS_PER_YEAR = 365.25 * 24 * 3600
REFERENCE_NATIVE = "BTC-USD"
VOL_NATIVE = "BTC-DVOL"


@dataclass(frozen=True)
class SyntheticMarketConfig:
    seed: int = 1
    start_ns: int = 1_788_220_800 * NS_PER_S  # 2026-09-01T00:00:00Z
    hours: int = 6
    dt_ms: int = 50
    spot0: float = 110_000.0
    vol_low: float = 0.40
    vol_high: float = 0.85
    regime_switches_per_hour: float = 0.5
    jumps_per_hour: float = 1.0
    jump_sd_bps: float = 20.0
    # reference exchange
    ref_rate_hz: float = 5.0
    ref_half_spread_usd: float = 0.5
    ref_basis_sd_bps: float = 0.5
    ref_basis_halflife_s: float = 30.0
    ref_feed_delay_ms: float = 25.0
    ref_feed_jitter_ms: float = 10.0
    # prediction venue
    strike_step: float = 250.0
    n_strikes: int = 9
    tick: str = "0.01"
    fee_schedule_id: str = "kalshi-standard"
    observation_method: str = "AVG_60S_BEFORE"
    pm_feed_delay_ms: float = 60.0
    pm_feed_jitter_ms: float = 20.0
    # market makers
    mm_lag_ms: float = 350.0
    mm_lag_sigma: float = 0.5
    mm_extra_half_spread_ticks: int = 0  # 0 => one-tick markets around fair value
    mm_requote_margin_ticks: float = 0.25  # hysteresis before re-quoting
    mm_vol: float | None = None  # None => makers know the true regime vol (well calibrated)
    mm_vol_bias: float = 1.0
    mm_touch_size: tuple[int, int] = (20, 120)
    mm_depth_sizes: tuple[tuple[int, int], ...] = ((50, 250), (100, 400))
    # noise flow
    noise_trades_per_min: float = 3.0
    noise_size_mean: float = 15.0
    # competing arbitrageurs
    competitor_latency_ms: float | None = 120.0
    competitor_min_edge: float = 0.02
    competitor_vol: float | None = None
    # implied-volatility index feed (DVOL-like): true regime vol with relative noise
    iv_feed: bool = True
    iv_noise_rel: float = 0.03
    iv_update_s: float = 5.0
    iv_feed_delay_ms: float = 100.0
    mapping_status: MappingStatus = MappingStatus.REVIEWED

    def label(self) -> str:
        comp = "none" if self.competitor_latency_ms is None else f"{self.competitor_latency_ms:g}"
        return f"mm{self.mm_lag_ms:g}ms_comp{comp}ms_seed{self.seed}"


@dataclass
class SyntheticMarket:
    config: SyntheticMarketConfig
    events: list[MarketEvent]
    contracts: list[PredictionContract]
    mappings: list[ContractMapping]
    settlements: list[Settlement]
    closes: list[tuple[int, str]]
    reference_instrument: str
    truth: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    vol_instrument: str | None = None

    @property
    def reference_instruments(self) -> frozenset[str]:
        refs = {self.reference_instrument}
        if self.vol_instrument is not None:
            refs.add(self.vol_instrument)
        return frozenset(refs)


# ----------------------------------------------------------------------------- helpers


def _monotone_recv(
    rng: np.random.Generator, source: NDArray[np.int64], mean_ms: float, jitter_ms: float
) -> NDArray[np.int64]:
    delay = np.abs(rng.normal(mean_ms, jitter_ms, size=source.size)) + 1.0
    recv = source + (delay * NS_PER_MS).astype(np.int64)
    return np.maximum.accumulate(recv) if recv.size else recv


def _fair_on_grid(  # noqa: PLR0917 - vectorised kernel over parallel arrays
    s: NDArray[np.float64],
    prefix: NDArray[np.float64],
    t_ns: NDArray[np.int64],
    k_idx: NDArray[np.int64],
    strike: float,
    expiry_ns: int,
    sigma: NDArray[np.float64],
    window_s: float,
    dt_s: float,
) -> NDArray[np.float64]:
    """Fair P(avg-or-point > K) as of each grid index in ``k_idx`` (sigma per grid point)."""
    spot = s[k_idx]
    t_end = (expiry_ns - t_ns[k_idx]) / NS_PER_S
    out = np.empty(k_idx.size)
    sig_all = sigma[k_idx] / math.sqrt(SECONDS_PER_YEAR)
    before = t_end >= window_s
    if np.any(before):
        sig_s = sig_all[before]
        var = sig_s**2 * (t_end[before] - window_s + window_s / 3.0)
        var = np.maximum(var, 1e-18)
        out[before] = ndtr((np.log(spot[before] / strike) - 0.5 * var) / np.sqrt(var))
    inside = ~before
    if np.any(inside):
        r = np.maximum(t_end[inside], 0.0)
        k_start = int(round(((expiry_ns - window_s * NS_PER_S) - t_ns[0]) / (dt_s * NS_PER_S)))  # noqa: RUF046
        k_start = max(k_start, 0)
        k_now = k_idx[inside]
        # known part covers [t_start, t_now) = samples k_start..k_now-1; the current
        # sample belongs to the random remainder (spot * r), so it must not be counted twice
        integral = (prefix[k_now] - prefix[k_start]) * dt_s
        mean = (integral + spot[inside] * r) / window_s
        sd = spot[inside] * sig_all[inside] * np.sqrt(r**3 / 3.0) / window_s
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(
                sd > 0, (mean - strike) / np.where(sd > 0, sd, 1.0), np.sign(mean - strike) * 50
            )
        out[inside] = ndtr(z)
    return np.clip(out, 0.0, 1.0)


@dataclass
class _Book:
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)

    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        return (max(self.bids), self.bids[max(self.bids)]) if self.bids else None

    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        return (min(self.asks), self.asks[min(self.asks)]) if self.asks else None


# ----------------------------------------------------------------------------- generator


def generate_market(cfg: SyntheticMarketConfig) -> SyntheticMarket:
    rng = np.random.default_rng(cfg.seed)
    dt_s = cfg.dt_ms / 1000.0
    n = int(cfg.hours * 3600 / dt_s)
    t_ns = cfg.start_ns + np.arange(n, dtype=np.int64) * cfg.dt_ms * NS_PER_MS

    # --- index path: Markov-switching vol + jumps
    p_switch = cfg.regime_switches_per_hour * dt_s / 3600.0
    switches = rng.random(n) < p_switch
    regime = np.cumsum(switches) % 2
    sigma = np.where(regime == 0, cfg.vol_low, cfg.vol_high)
    dt_y = dt_s / SECONDS_PER_YEAR
    rets = sigma * math.sqrt(dt_y) * rng.standard_normal(n) - 0.5 * sigma**2 * dt_y
    jumps = rng.random(n) < cfg.jumps_per_hour * dt_s / 3600.0
    rets = rets + jumps * rng.normal(0.0, cfg.jump_sd_bps * 1e-4, size=n)
    rets[0] = 0.0
    index = cfg.spot0 * np.exp(np.cumsum(rets))
    prefix = np.concatenate([[0.0], np.cumsum(index)])

    # --- reference exchange mid with AR(1) basis
    phi = 0.5 ** (dt_s / cfg.ref_basis_halflife_s)
    innov_sd = cfg.ref_basis_sd_bps * 1e-4 * math.sqrt(1 - phi**2)
    basis = np.empty(n)
    b = 0.0
    shocks = rng.normal(0.0, innov_sd, size=n)
    for i in range(n):
        b = phi * b + shocks[i]
        basis[i] = b
    ref_mid = index * np.exp(basis)

    events: list[MarketEvent] = []
    ref_inst = instrument_key(Venue.COINBASE, REFERENCE_NATIVE)
    gaps = rng.exponential(
        1.0 / cfg.ref_rate_hz, size=int(cfg.hours * 3600 * cfg.ref_rate_hz * 1.3)
    )
    ref_times = cfg.start_ns + (np.cumsum(gaps) * NS_PER_S).astype(np.int64)
    ref_times = ref_times[ref_times < t_ns[-1]]
    ref_recv = _monotone_recv(rng, ref_times, cfg.ref_feed_delay_ms, cfg.ref_feed_jitter_ms)
    ref_k = ((ref_times - cfg.start_ns) // (cfg.dt_ms * NS_PER_MS)).astype(np.int64)
    half = Decimal(str(cfg.ref_half_spread_usd))
    for i in range(ref_times.size):
        mid = Decimal(f"{ref_mid[ref_k[i]]:.2f}")
        events.append(
            BookSnapshotEvent(
                venue=Venue.COINBASE,
                instrument_id=ref_inst,
                source_ts_ns=int(ref_times[i]),
                recv_ts_ns=int(ref_recv[i]),
                sequence=i + 1,
                payload_hash=f"syn-ref-{cfg.seed}-{i}",
                bids=(BookLevel(mid - half, Decimal(3)),),
                asks=(BookLevel(mid + half, Decimal(3)),),
                quality_flags=frozenset({QualityFlag.TOP_OF_BOOK_ONLY}),
            )
        )

    # --- implied-volatility index (DVOL-like), percent units
    vol_inst: str | None = None
    if cfg.iv_feed:
        vol_inst = instrument_key(Venue.DERIBIT, VOL_NATIVE)
        step = max(1, int(cfg.iv_update_s / dt_s))
        ks_iv = np.arange(0, n, step, dtype=np.int64)
        noise = 1.0 + cfg.iv_noise_rel * rng.standard_normal(ks_iv.size)
        iv_src = t_ns[ks_iv]
        iv_recv = iv_src + int(cfg.iv_feed_delay_ms * NS_PER_MS)
        for j, k in enumerate(ks_iv):
            events.append(
                TradeEvent(
                    venue=Venue.DERIBIT,
                    instrument_id=vol_inst,
                    source_ts_ns=int(iv_src[j]),
                    recv_ts_ns=int(iv_recv[j]),
                    payload_hash=f"syn-iv-{cfg.seed}-{j}",
                    price=Decimal(f"{sigma[k] * noise[j] * 100:.2f}"),
                    size=Decimal(1),
                    aggressor_side=None,
                    trade_id=f"iv-{j}",
                )
            )

    # --- prediction-venue ladder
    tick = Decimal(cfg.tick)
    tick_f = float(tick)
    window_s = 60.0 if cfg.observation_method == "AVG_60S_BEFORE" else 0.0
    fee = get_fee_schedule(cfg.fee_schedule_id)
    contracts: list[PredictionContract] = []
    mappings: list[ContractMapping] = []
    settlements: list[Settlement] = []
    closes: list[tuple[int, str]] = []
    pm_raw: list[tuple[int, int, MarketEvent]] = []  # (source, tiebreak, event-without-recv)
    stats = {"maker_updates": 0, "noise_trades": 0, "competitor_takes": 0}
    lag_steps_med = max(0, round(cfg.mm_lag_ms / cfg.dt_ms))
    comp_steps = (
        None
        if cfg.competitor_latency_ms is None
        else max(0, round(cfg.competitor_latency_ms / cfg.dt_ms))
    )
    steps_per_hour = int(3600 / dt_s)

    for h in range(cfg.hours):
        k_list = h * steps_per_hour
        k_exp = min((h + 1) * steps_per_hour, n) - 1
        listing_ns = int(t_ns[k_list])
        expiry_ns = int(t_ns[k_exp]) + cfg.dt_ms * NS_PER_MS
        center = round(index[k_list] / cfg.strike_step) * cfg.strike_step
        strikes = [
            center + (j - cfg.n_strikes // 2) * cfg.strike_step for j in range(cfg.n_strikes)
        ]
        # settlement value: average of index over the final window (or point)
        if window_s:
            k0 = max(k_exp - int(window_s / dt_s) + 1, 0)
            settle_value = float(np.mean(index[k0 : k_exp + 1]))
        else:
            settle_value = float(index[k_exp])
        ev_tag = iso_from_ns(expiry_ns)[:16].replace("-", "").replace(":", "")
        for strike in strikes:
            native = f"SYNBTC-{ev_tag}-T{strike:.0f}"
            cid = instrument_key(Venue.KALSHI, native)
            family = f"BTC-USD|{cfg.observation_method}|{iso_from_ns(expiry_ns)}"
            contracts.append(
                PredictionContract(
                    venue=Venue.KALSHI,
                    contract_id=cid,
                    native_id=native,
                    event_id=f"SYNBTC-{ev_tag}",
                    title=f"BTC above {strike:.0f} at {iso_from_ns(expiry_ns)} (synthetic)",
                    yes_semantics=f"60s average index > {strike:.0f}",
                    no_semantics=f"60s average index <= {strike:.0f}",
                    open_ts_ns=listing_ns,
                    close_ts_ns=expiry_ns,
                    resolve_ts_ns=expiry_ns,
                    status=ContractStatus.OPEN,
                    tick_size=tick,
                    series_id="SYNBTC",
                    fee_schedule_id=cfg.fee_schedule_id,
                    outcome_instruments={"YES": cid},
                )
            )
            mappings.append(
                ContractMapping(
                    venue=Venue.KALSHI,
                    contract_id=cid,
                    underlyings=("BTC-USD",),
                    operator=Operator.GT,
                    strikes=(Decimal(f"{strike:.0f}"),),
                    observation_start_ns=expiry_ns - int(window_s * NS_PER_S) if window_s else None,
                    observation_end_ns=expiry_ns,
                    observation_method=cfg.observation_method,
                    timezone="America/New_York",
                    resolution_source="SYNTHETIC_INDEX",
                    event_family=family,
                    review_status=cfg.mapping_status,
                    reviewer="synthetic-generator",
                    notes="synthetic contract with known semantics",
                )
            )
            closes.append((expiry_ns, cid))
            yes = settle_value > strike
            settlements.append(
                Settlement(
                    venue=Venue.KALSHI,
                    contract_id=cid,
                    outcome=SettlementOutcome.YES if yes else SettlementOutcome.NO,
                    yes_value=Decimal(1) if yes else Decimal(0),
                    settled_ts_ns=expiry_ns + 30 * NS_PER_S,
                    source="synthetic index 60s average",
                )
            )
            _simulate_contract(
                cfg=cfg,
                rng=rng,
                cid=cid,
                strike=strike,
                expiry_ns=expiry_ns,
                k_list=k_list,
                k_exp=k_exp,
                index=index,
                prefix=prefix,
                t_ns=t_ns,
                window_s=window_s,
                dt_s=dt_s,
                tick=tick,
                tick_f=tick_f,
                fee=fee,
                lag_steps_med=lag_steps_med,
                comp_steps=comp_steps,
                sigma_path=sigma,
                out=pm_raw,
                stats=stats,
            )

    # assign receive times per prediction-venue connection (monotone, jittered)
    pm_raw.sort(key=lambda x: (x[0], x[1]))
    src = np.array([x[0] for x in pm_raw], dtype=np.int64)
    recv = _monotone_recv(rng, src, cfg.pm_feed_delay_ms, cfg.pm_feed_jitter_ms)
    import dataclasses

    for (_, _, ev), r in zip(pm_raw, recv, strict=True):
        events.append(dataclasses.replace(ev, recv_ts_ns=int(r)))
    events.sort(key=lambda e: (e.recv_ts_ns, e.venue_ts_ns))
    truth = {"t_ns": t_ns.astype(np.float64), "index": index, "ref_mid": ref_mid, "sigma": sigma}
    return SyntheticMarket(
        config=cfg,
        events=events,
        contracts=contracts,
        mappings=mappings,
        settlements=settlements,
        closes=closes,
        reference_instrument=ref_inst,
        truth=truth,
        stats=stats,
        vol_instrument=vol_inst,
    )


def _simulate_contract(
    *,
    cfg: SyntheticMarketConfig,
    rng: np.random.Generator,
    cid: str,
    strike: float,
    expiry_ns: int,
    k_list: int,
    k_exp: int,
    index: NDArray[np.float64],
    prefix: NDArray[np.float64],
    t_ns: NDArray[np.int64],
    window_s: float,
    dt_s: float,
    tick: Decimal,
    tick_f: float,
    fee: object,
    lag_steps_med: int,
    comp_steps: int | None,
    sigma_path: NDArray[np.float64],
    out: list[tuple[int, int, MarketEvent]],
    stats: dict[str, int],
) -> None:
    ks = np.arange(k_list, k_exp + 1, dtype=np.int64)
    view = np.maximum(ks - lag_steps_med, k_list)
    mm_sigma = (
        np.full(sigma_path.size, cfg.mm_vol) if cfg.mm_vol is not None else sigma_path
    ) * cfg.mm_vol_bias
    f_mm = _fair_on_grid(index, prefix, t_ns, view, strike, expiry_ns, mm_sigma, window_s, dt_s)
    # one-tick (plus optional extra) quotes around the maker's fair value, re-quoted with
    # hysteresis: only when fair value leaves [bid - m, ask + m] (Schmitt trigger)
    extra = cfg.mm_extra_half_spread_ticks
    m = cfg.mm_requote_margin_ticks
    x = f_mm / tick_f
    up = np.floor(x - m)
    dn = np.floor(x + m)
    cand = np.nonzero((up[1:] != up[:-1]) | (dn[1:] != dn[:-1]))[0] + 1
    units = np.empty(ks.size, dtype=np.int64)
    u = math.floor(x[0])
    change_points = [0]
    last = 0
    for i in cand:
        xi = x[i]
        if xi < u - m or xi >= u + 1 + m:
            units[last:i] = u
            u = math.floor(xi)
            change_points.append(int(i))
            last = int(i)
    units[last:] = u
    bid_units = units - extra
    ask_units = bid_units + 1 + 2 * extra
    max_units = round(1 / tick_f)
    bid_units = np.where(bid_units < 1, 0, bid_units)
    ask_units = np.where(ask_units > max_units - 1, max_units, ask_units)
    change_idx = np.array(change_points, dtype=np.int64)
    # jitter each update's effective time by its own log-normal reaction lag
    lag_draw = cfg.mm_lag_ms * np.exp(cfg.mm_lag_sigma * rng.standard_normal(change_idx.size))
    upd_t = t_ns[ks[change_idx]] + ((lag_draw - lag_steps_med * cfg.dt_ms) * NS_PER_MS).astype(
        np.int64
    )
    upd_t[0] = t_ns[k_list]
    upd_t = np.maximum.accumulate(np.maximum(upd_t, t_ns[k_list]))
    upd_t = np.minimum(upd_t, expiry_ns - 1)

    actions: list[tuple[int, int, int]] = []  # (time, kind, index) kind 0=update 1=noise 2=comp
    for j, idx in enumerate(change_idx):
        actions.append((int(upd_t[j]), 0, int(idx)))
    span_s = (expiry_ns - int(t_ns[k_list])) / NS_PER_S
    n_noise = rng.poisson(cfg.noise_trades_per_min * span_s / 60.0)
    noise_t = np.sort(rng.integers(int(t_ns[k_list]) + NS_PER_S, expiry_ns - 1, size=n_noise))
    for j, tt in enumerate(noise_t):
        actions.append((int(tt), 1, j))
    f_comp: NDArray[np.float64] | None = None
    if comp_steps is not None:
        cview = np.maximum(ks - comp_steps, k_list)
        comp_sigma = (
            np.full(sigma_path.size, cfg.competitor_vol)
            if cfg.competitor_vol is not None
            else sigma_path
        )
        f_comp = _fair_on_grid(
            index, prefix, t_ns, cview, strike, expiry_ns, comp_sigma, window_s, dt_s
        )
        # candidate grid points where the grid-aligned maker quote looks stale to them
        ask_px = ask_units * tick_f
        bid_px = bid_units * tick_f
        cand = np.nonzero(
            (f_comp - ask_px > cfg.competitor_min_edge)
            | (bid_px - f_comp > cfg.competitor_min_edge)
        )[0]
        for idx in cand:
            actions.append((int(t_ns[ks[idx]]), 2, int(idx)))
    actions.sort()

    book = _Book()
    seq = 0
    sub = 0
    noise_side = rng.random(max(n_noise, 1)) < 0.5
    noise_size = rng.geometric(1.0 / cfg.noise_size_mean, size=max(n_noise, 1))
    started = False

    def emit_delta(ts: int, changes: list[LevelChange]) -> None:
        nonlocal seq, sub
        if not changes:
            return
        seq += 1
        sub += 1
        out.append(
            (
                ts,
                sub,
                BookDeltaEvent(
                    venue=Venue.KALSHI,
                    instrument_id=cid,
                    source_ts_ns=ts,
                    recv_ts_ns=ts,
                    sequence=seq,
                    payload_hash=f"syn-{cid}-{seq}",
                    changes=tuple(changes),
                    mode=DeltaMode.ABSOLUTE,
                ),
            )
        )

    def emit_trade(ts: int, price: Decimal, size: Decimal, aggressor: Side) -> None:
        nonlocal sub
        sub += 1
        out.append(
            (
                ts,
                sub,
                TradeEvent(
                    venue=Venue.KALSHI,
                    instrument_id=cid,
                    source_ts_ns=ts,
                    recv_ts_ns=ts,
                    payload_hash=f"syn-{cid}-tr{sub}",
                    price=price,
                    size=size,
                    aggressor_side=aggressor,
                    trade_id=f"{cid}-tr{sub}",
                ),
            )
        )

    def target_levels(i: int) -> tuple[dict[Decimal, Decimal], dict[Decimal, Decimal]]:
        bu, au = int(bid_units[i]), int(ask_units[i])
        bids: dict[Decimal, Decimal] = {}
        asks: dict[Decimal, Decimal] = {}
        sizes = [cfg.mm_touch_size, *cfg.mm_depth_sizes]
        for lvl, (lo, hi) in enumerate(sizes):
            q = Decimal(int(rng.integers(lo, hi + 1)))
            if bu - lvl >= 1:
                bids[tick * (bu - lvl)] = q
            if au + lvl <= max_units - 1:
                asks[tick * (au + lvl)] = q
        return bids, asks

    max_units = round(1 / tick_f)
    for ts, kind, idx in actions:
        if kind == 0:
            new_bids, new_asks = target_levels(idx)
            if not started:
                seq += 1
                sub += 1
                out.append(
                    (
                        ts,
                        sub,
                        BookSnapshotEvent(
                            venue=Venue.KALSHI,
                            instrument_id=cid,
                            source_ts_ns=ts,
                            recv_ts_ns=ts,
                            sequence=seq,
                            payload_hash=f"syn-{cid}-{seq}",
                            bids=tuple(
                                BookLevel(p, q) for p, q in sorted(new_bids.items(), reverse=True)
                            ),
                            asks=tuple(BookLevel(p, q) for p, q in sorted(new_asks.items())),
                        ),
                    )
                )
                book.bids, book.asks = new_bids, new_asks
                started = True
                stats["maker_updates"] += 1
                continue
            changes: list[LevelChange] = []
            for side, old, new in (
                (BookSide.BID, book.bids, new_bids),
                (BookSide.ASK, book.asks, new_asks),
            ):
                for p in old:
                    if p not in new:
                        changes.append(LevelChange(side, p, Decimal(0)))
                for p, q in new.items():
                    if old.get(p) != q:
                        changes.append(LevelChange(side, p, q))
            book.bids, book.asks = dict(new_bids), dict(new_asks)
            emit_delta(ts, changes)
            stats["maker_updates"] += 1
        elif kind == 1 and started:
            buy = bool(noise_side[idx])
            level = book.best_ask() if buy else book.best_bid()
            if level is None:
                continue
            price, avail = level
            size = min(Decimal(int(noise_size[idx])), avail)
            emit_trade(ts, price, size, Side.BUY if buy else Side.SELL)
            side_book = book.asks if buy else book.bids
            remaining = avail - size
            if remaining > 0:
                side_book[price] = remaining
            else:
                del side_book[price]
            emit_delta(ts, [LevelChange(BookSide.ASK if buy else BookSide.BID, price, remaining)])
            stats["noise_trades"] += 1
        elif kind == 2 and started and f_comp is not None:
            fair = float(f_comp[idx])
            for buy in (True, False):
                while True:
                    level = book.best_ask() if buy else book.best_bid()
                    if level is None:
                        break
                    price, avail = level
                    px_f = float(price)
                    unit_fee = float(
                        fee.fee_per_contract(price=price, role=LiquidityRole.TAKER)  # type: ignore[attr-defined]
                    )
                    edge = (fair - px_f) if buy else (px_f - fair)
                    if edge - unit_fee < cfg.competitor_min_edge:
                        break
                    emit_trade(ts, price, avail, Side.BUY if buy else Side.SELL)
                    (book.asks if buy else book.bids).pop(price)
                    emit_delta(
                        ts, [LevelChange(BookSide.ASK if buy else BookSide.BID, price, Decimal(0))]
                    )
                    stats["competitor_takes"] += 1
