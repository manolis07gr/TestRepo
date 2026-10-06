"""Black (forward-measure) digital probabilities with explicit, overflow-free limits.

Floats are acceptable inside option-pricing math (scope s.7); callers convert explicitly at
the boundary (see :func:`cma.domain.numbers.from_float`). Every function here:

* validates inputs (non-finite or out-of-domain values raise ``ValueError``; never NaN out);
* treats the degenerate limits explicitly: ``T == 0`` or ``sigma == 0`` gives the step
  ``1{F > K}`` with exactly ``0.5`` at ``F == K``;
* forms intermediates in log space (``ln F - ln K``, ``sigma * sqrt(T)`` without squaring
  sigma, log-pdf for the smile term) so extreme moneyness, tiny maturities and huge
  volatilities neither overflow nor produce ``inf - inf``;
* clips probabilities to ``[0, 1]``.

Under the forward measure with lognormal ``S_T``: ``P(S_T > K) = N(d2)``,
``d2 = ln(F/K)/(sigma sqrt T) - sigma sqrt T / 2``. With a strike-dependent smile the
digital is ``-dC/dK = N(d2) - F phi(d1) sqrt(T) dsigma/dK`` (vega x smile slope); the
identity ``F phi(d1) = K phi(d2)`` is used for stability.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Final

from scipy.special import ndtr

from cma.domain.time import NS_PER_DAY

YEAR_NS: Final = 365 * NS_PER_DAY  # ACT/365: crypto options trade continuously
_LOG_SQRT_2PI: Final = 0.5 * math.log(2.0 * math.pi)
_EXP_OVERFLOW: Final = 700.0
_EXP_UNDERFLOW: Final = -745.0

type FloatLike = float | int | Decimal


def _finite(name: str, value: FloatLike) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    result = float(value)  # explicit Decimal/int -> float boundary crossing
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive(name: str, value: FloatLike) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _non_negative(name: str, value: FloatLike) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def clip_probability(p: float) -> float:
    if math.isnan(p):  # defensive: no code path should produce NaN
        raise FloatingPointError("probability computation produced NaN")
    return 0.0 if p < 0.0 else 1.0 if p > 1.0 else p


def year_fraction(start_ns: int, end_ns: int, *, year_ns: int = YEAR_NS) -> float:
    """ACT/365 year fraction between two UTC instants; refuses negative intervals."""
    if end_ns < start_ns:
        raise ValueError("end precedes start (expired)")
    return (end_ns - start_ns) / year_ns


def log_normal_pdf(x: float) -> float:
    """log phi(x); ``-inf`` for infinite |x| (no overflow: x*x -> inf is fine)."""
    return -0.5 * x * x - _LOG_SQRT_2PI


def normal_pdf(x: float) -> float:
    log_pdf = log_normal_pdf(x)
    return 0.0 if log_pdf < _EXP_UNDERFLOW else math.exp(log_pdf)


def signed_exp(log_magnitude: float, sign: float) -> float:
    """``sign * exp(log_magnitude)`` saturating to +-inf / 0 instead of raising."""
    if sign == 0.0 or log_magnitude < _EXP_UNDERFLOW:
        return 0.0
    if log_magnitude > _EXP_OVERFLOW:
        return math.copysign(math.inf, sign)
    return math.copysign(math.exp(log_magnitude), sign)


def _d2(log_fk: float, total_std: float) -> float:
    return log_fk / total_std - 0.5 * total_std


def _step(log_fk: float, *, above: bool) -> float:
    if log_fk == 0.0:
        return 0.5
    return 1.0 if (log_fk > 0.0) is above else 0.0


def _smile_term(strike: float, d2: float, t_years: float, dsigma_dstrike: float) -> float:
    """``K phi(d2) sqrt(T) dsigma/dK`` (== ``F phi(d1) sqrt(T) dsigma/dK``) in log space."""
    if dsigma_dstrike == 0.0 or t_years == 0.0:
        return 0.0
    log_mag = (
        math.log(strike)
        + log_normal_pdf(d2)
        + 0.5 * math.log(t_years)
        + math.log(abs(dsigma_dstrike))
    )
    return signed_exp(log_mag, dsigma_dstrike)


def digital_call_probability(
    forward: FloatLike,
    strike: FloatLike,
    t_years: FloatLike,
    sigma: FloatLike,
    *,
    dsigma_dstrike: FloatLike = 0.0,
) -> float:
    """Forward-measure ``P(S_T > K)``, optionally smile-corrected by ``dsigma/dK``."""
    f = _positive("forward", forward)
    k = _finite("strike", strike)
    t = _non_negative("t_years", t_years)
    vol = _non_negative("sigma", sigma)
    slope = _finite("dsigma_dstrike", dsigma_dstrike)
    if k <= 0.0:
        return 1.0  # lognormal S_T > 0 >= K almost surely
    log_fk = math.log(f) - math.log(k)
    total_std = vol * math.sqrt(t)
    if total_std == 0.0:
        return _step(log_fk, above=True)
    if math.isinf(total_std):
        return 0.0  # all mass escapes to 0 under the forward measure: d2 -> -inf
    d2 = _d2(log_fk, total_std)
    return clip_probability(float(ndtr(d2)) - _smile_term(k, d2, t, slope))


def digital_put_probability(
    forward: FloatLike,
    strike: FloatLike,
    t_years: FloatLike,
    sigma: FloatLike,
    *,
    dsigma_dstrike: FloatLike = 0.0,
) -> float:
    """Forward-measure ``P(S_T < K)`` (``N(-d2)`` computed directly for tail accuracy)."""
    f = _positive("forward", forward)
    k = _finite("strike", strike)
    t = _non_negative("t_years", t_years)
    vol = _non_negative("sigma", sigma)
    slope = _finite("dsigma_dstrike", dsigma_dstrike)
    if k <= 0.0:
        return 0.0
    log_fk = math.log(f) - math.log(k)
    total_std = vol * math.sqrt(t)
    if total_std == 0.0:
        return _step(log_fk, above=False)
    if math.isinf(total_std):
        return 1.0
    d2 = _d2(log_fk, total_std)
    return clip_probability(float(ndtr(-d2)) + _smile_term(k, d2, t, slope))


def black_call_price(
    forward: FloatLike, strike: FloatLike, t_years: FloatLike, sigma: FloatLike
) -> float:
    """Undiscounted Black call price, bounded to ``[max(F - K, 0), F]``."""
    f = _positive("forward", forward)
    k = _finite("strike", strike)
    t = _non_negative("t_years", t_years)
    vol = _non_negative("sigma", sigma)
    intrinsic = max(f - k, 0.0)
    if k <= 0.0:
        return f - k
    total_std = vol * math.sqrt(t)
    if total_std == 0.0:
        return intrinsic
    if math.isinf(total_std):
        return f
    log_fk = math.log(f) - math.log(k)
    d2 = _d2(log_fk, total_std)
    d1 = d2 + total_std
    price = f * float(ndtr(d1)) - k * float(ndtr(d2))
    return min(f, max(intrinsic, price))


def forward_from_spot(
    spot: FloatLike, t_years: FloatLike, *, rate: FloatLike = 0.0, carry_yield: FloatLike = 0.0
) -> float:
    """``F = S exp((r - q) T)``; with the default ``r = q = 0`` the forward is the spot."""
    s = _positive("spot", spot)
    t = _non_negative("t_years", t_years)
    exponent = (_finite("rate", rate) - _finite("carry_yield", carry_yield)) * t
    log_f = math.log(s) + exponent
    if not math.isfinite(log_f) or log_f > 709.0:
        raise ValueError("forward overflows a float")
    return math.exp(log_f)


def digital_probability_from_spot(
    spot: FloatLike,
    strike: FloatLike,
    t_years: FloatLike,
    sigma: FloatLike,
    *,
    rate: FloatLike = 0.0,
    carry_yield: FloatLike = 0.0,
) -> float:
    """Naive external-price fair value ``P(S_T > K)`` from spot (r = q = 0 by default).

    This is the "naive external-price mapping" baseline of scope s.13: a flat-vol
    lognormal digital. It ignores averaging windows, basis between the option underlying
    and the contract's settlement source, and smile/skew.
    """
    forward = forward_from_spot(spot, t_years, rate=rate, carry_yield=carry_yield)
    return digital_call_probability(forward, strike, t_years, sigma)
