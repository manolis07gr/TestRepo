"""Contract-semantics-aware fair values from an external reference price.

This is the "naive external-price mapping" family of models (scope s.13 baseline) made
semantically exact for the two observation methods that matter for crypto contracts:

* ``POINT``: payoff on the price at a single instant -> log-normal digital N(d2).
* ``AVG_<n>S_BEFORE``: payoff on the arithmetic average over the n seconds before the
  observation instant (e.g. index averages used by some venues). Averaging lowers the
  effective variance (~sigma^2 (T_start + w/3)); inside the window the already-observed
  part of the average is known and only the remainder is random.

Floats are used deliberately (statistical model); results are clipped to [0, 1].
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from cma.domain.enums import Operator
from cma.domain.time import NS_PER_S

SECONDS_PER_YEAR = 365.25 * 24 * 3600
_AVG_RE = re.compile(r"^AVG_(\d+)S_BEFORE$")


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _clip(p: float) -> float:
    if math.isnan(p):
        raise ValueError("probability is NaN")
    return min(1.0, max(0.0, p))


def digital_above(spot: float, strike: float, t_years: float, sigma: float) -> float:
    """P(S_T > K) under a driftless log-normal model, with exact limits."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    var = sigma * sigma * max(t_years, 0.0)
    if var <= 0.0:
        if spot > strike:
            return 1.0
        if spot < strike:
            return 0.0
        return 0.5
    sd = math.sqrt(var)
    return _clip(norm_cdf((math.log(spot / strike) - 0.5 * var) / sd))


def window_seconds(observation_method: str) -> int:
    if observation_method == "POINT":
        return 0
    m = _AVG_RE.match(observation_method)
    if m is None:
        raise ValueError(f"unsupported observation method {observation_method!r}")
    return int(m.group(1))


@dataclass(frozen=True, slots=True)
class AveragingState:
    """Known part of an in-progress averaging window."""

    elapsed_s: float  # seconds of the window already observed
    integral: float  # time integral of price over the elapsed part (price * seconds)


def prob_above(
    *,
    spot: float,
    strike: float,
    now_ns: int,
    observation_end_ns: int,
    sigma: float,
    observation_method: str = "POINT",
    averaging: AveragingState | None = None,
) -> float:
    """P(X > K) where X is the contract's observed value (point or trailing average)."""
    w = window_seconds(observation_method)
    t_end = (observation_end_ns - now_ns) / NS_PER_S
    if w == 0:
        return digital_above(spot, strike, max(t_end, 0.0) / SECONDS_PER_YEAR, sigma)
    sigma_s = sigma / math.sqrt(SECONDS_PER_YEAR)  # per sqrt(second)
    if t_end <= 0:
        if averaging is None or averaging.elapsed_s <= 0:
            raise ValueError("observation window finished but no averaging state supplied")
        avg = averaging.integral / averaging.elapsed_s
        return 1.0 if avg > strike else (0.0 if avg < strike else 0.5)
    if t_end >= w:
        # Window not started: variance until window start + averaging over the window.
        t_start = t_end - w
        var = sigma_s**2 * (t_start + w / 3.0)
        if var <= 0:
            return 1.0 if spot > strike else (0.0 if spot < strike else 0.5)
        sd = math.sqrt(var)
        return _clip(norm_cdf((math.log(spot / strike) - 0.5 * var) / sd))
    # Inside the window: A = (I_known + integral of future path) / w.
    if averaging is None:
        raise ValueError("inside the averaging window an AveragingState is required")
    r = t_end
    mean = (averaging.integral + spot * r) / w
    sd = spot * sigma_s * math.sqrt(r**3 / 3.0) / w
    if sd <= 0:
        return 1.0 if mean > strike else (0.0 if mean < strike else 0.5)
    return _clip(norm_cdf((mean - strike) / sd))


def contract_probability(
    *,
    operator: Operator,
    strikes: tuple[float, ...],
    spot: float,
    now_ns: int,
    observation_end_ns: int,
    sigma: float,
    observation_method: str = "POINT",
    averaging: AveragingState | None = None,
) -> float:
    """Fair YES probability for GT/GE/LT/LE/BETWEEN payoffs (continuous: GT == GE)."""

    def above(k: float) -> float:
        return prob_above(
            spot=spot,
            strike=k,
            now_ns=now_ns,
            observation_end_ns=observation_end_ns,
            sigma=sigma,
            observation_method=observation_method,
            averaging=averaging,
        )

    if operator in (Operator.GT, Operator.GE):
        return above(strikes[0])
    if operator in (Operator.LT, Operator.LE):
        return _clip(1.0 - above(strikes[0]))
    if operator is Operator.BETWEEN:
        lo, hi = strikes
        return _clip(above(lo) - above(hi))
    raise ValueError(f"operator {operator} not supported by the spot fair-value model")


def digital_delta_per_log_move(spot: float, strike: float, t_years: float, sigma: float) -> float:
    """dP/d(ln S) for the point digital: phi(d2) / (sigma sqrt(T))."""
    if t_years <= 0 or sigma <= 0:
        return 0.0
    sd = sigma * math.sqrt(t_years)
    d2 = (math.log(spot / strike) - 0.5 * sd * sd) / sd
    return math.exp(-0.5 * d2 * d2) / math.sqrt(2 * math.pi) / sd
