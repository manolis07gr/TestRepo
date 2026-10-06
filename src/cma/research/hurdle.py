"""Closed-form cost hurdle for latency (lead-lag) taking on binary price contracts.

Setting: a contract pays $1 if the underlying finishes above K. A market maker quotes a
one-tick market around the fair value p0 and has not yet reacted to a move. We buy at the
stale ask a = p0 + tick/2 (expected position inside the tick) and pay the venue fee. The
trade clears a net threshold theta only if the move pushed fair value to

    p* = a + fee(a) + theta.

For a log-normal digital, fair value is Phi(d2), so the required log move is exactly

    r* = sigma sqrt(T) * (Phi^-1(p*) - Phi^-1(p0)).

The chance that the underlying moves at least r* (either direction; symmetric for sells)
inside a reaction window of Delta seconds is 2 * (1 - F(r* / (sigma_s sqrt(Delta)))) for a
Gaussian or a variance-matched Student-t (fat tails). This is an *upper bound* on
opportunity frequency: it ignores competition, queue/latency races and settlement basis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal

from scipy.stats import norm, t

from cma.domain.enums import LiquidityRole
from cma.domain.fees import FeeSchedule, get_fee_schedule

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class HurdleRow:
    venue_fee: str
    t_seconds: float
    moneyness_z: float  # ln(S/K) / (sigma sqrt T)
    sigma: float
    p0: float
    ask: float
    fee_per_contract: float
    required_fair: float
    delta_per_bp: float  # cents of probability per 1 bp move at p0
    required_move_bps: float  # inf if impossible
    window_s: float
    p_move_gauss: float
    p_move_fat: float
    opportunities_per_hour_gauss: float
    opportunities_per_hour_fat: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _p_exceed(r: float, sigma_s: float, window_s: float, nu: float | None) -> float:
    if not math.isfinite(r):
        return 0.0
    sd = sigma_s * math.sqrt(window_s)
    x = r / sd
    if nu is None:
        return float(2.0 * norm.sf(x))
    scale = math.sqrt(nu / (nu - 2.0))  # variance-matched
    return float(2.0 * t.sf(x * scale, df=nu))


def hurdle_row(
    *,
    fee: FeeSchedule,
    t_seconds: float,
    moneyness_z: float,
    sigma: float,
    window_s: float,
    tick: float = 0.01,
    threshold: float = 0.01,
    nu: float = 3.0,
) -> HurdleRow:
    sqrt_t = sigma * math.sqrt(t_seconds / SECONDS_PER_YEAR)
    p0 = float(norm.cdf(moneyness_z - 0.5 * sqrt_t))
    ask = min(0.99, max(0.01, p0 + tick / 2))
    fee_pc = float(fee.fee_per_contract(price=Decimal(f"{ask:.4f}"), role=LiquidityRole.TAKER))
    p_star = ask + fee_pc + threshold
    sigma_s = sigma / math.sqrt(SECONDS_PER_YEAR)
    if p_star >= 1.0 or p0 <= 0.0:
        r_star = math.inf
    else:
        r_star = sqrt_t * (float(norm.ppf(p_star)) - float(norm.ppf(max(p0, 1e-12))))
    d2 = moneyness_z - 0.5 * sqrt_t
    delta_per_bp = float(norm.pdf(d2)) / sqrt_t * 1e-4 * 100 if sqrt_t > 0 else 0.0
    pg = _p_exceed(r_star, sigma_s, window_s, None)
    pf = _p_exceed(r_star, sigma_s, window_s, nu)
    per_hour = 3600.0 / window_s
    return HurdleRow(
        venue_fee=fee.tag,
        t_seconds=t_seconds,
        moneyness_z=moneyness_z,
        sigma=sigma,
        p0=p0,
        ask=ask,
        fee_per_contract=fee_pc,
        required_fair=p_star,
        delta_per_bp=delta_per_bp,
        required_move_bps=r_star * 1e4 if math.isfinite(r_star) else math.inf,
        window_s=window_s,
        p_move_gauss=pg,
        p_move_fat=pf,
        opportunities_per_hour_gauss=pg * per_hour,
        opportunities_per_hour_fat=pf * per_hour,
    )


def hurdle_table(
    *,
    fee_ids: Sequence[str] = ("kalshi-standard", "polymarket-crypto-taker"),
    t_seconds: Sequence[float] = (120, 300, 900, 1800, 3600),
    moneyness: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    sigmas: Sequence[float] = (0.45,),
    windows_s: Sequence[float] = (0.35, 1.0),
    threshold: float = 0.01,
) -> list[HurdleRow]:
    rows = []
    for fid in fee_ids:
        fee = get_fee_schedule(fid)
        for sigma in sigmas:
            for tt in t_seconds:
                for z in moneyness:
                    for w in windows_s:
                        rows.append(
                            hurdle_row(
                                fee=fee,
                                t_seconds=tt,
                                moneyness_z=z,
                                sigma=sigma,
                                window_s=w,
                                threshold=threshold,
                            )
                        )
    return rows


def break_even_staleness_ms(
    *, required_move_bps: float, sigma: float, target_prob: float = 0.01
) -> float:
    """Window (ms) over which a >= required move has probability ``target_prob``."""
    if not math.isfinite(required_move_bps):
        return math.inf
    sigma_s = sigma / math.sqrt(SECONDS_PER_YEAR)
    z = float(norm.isf(target_prob / 2.0))
    window_s = (required_move_bps * 1e-4 / (z * sigma_s)) ** 2
    return window_s * 1000.0
