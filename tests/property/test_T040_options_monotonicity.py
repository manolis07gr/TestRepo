"""T040 - for the same expiry/model inputs, P(S_T > K) is non-increasing as K increases.

Covers the closed-form Black digital, flat slices, arbitrary (possibly arbitrageable) smiles
- where the raw Breeden-Litzenberger digital can increase in K and the monotone guard must
hold - and total-variance-interpolated slices. Comparisons allow 1e-12 for float rounding.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cma.domain.time import NS_PER_DAY, NS_PER_HOUR
from cma.models.implied_probability import (
    OptionSlice,
    digital_call_probability,
    interpolate_total_variance,
)

pytestmark = pytest.mark.property

EPS = 1e-12
SETTINGS = settings(max_examples=150, deadline=None, derandomize=True)

forwards = st.floats(min_value=1e-3, max_value=1e7, allow_nan=False, allow_infinity=False)
vols = st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False)
expiries = st.sampled_from(
    [NS_PER_HOUR, NS_PER_DAY, 7 * NS_PER_DAY, 90 * NS_PER_DAY, 730 * NS_PER_DAY]
)
log_m = st.floats(min_value=-6.0, max_value=6.0, allow_nan=False, allow_infinity=False)
steps = st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False)


@st.composite
def smiles(
    draw: st.DrawFn, *, expiry: int | None = None, forward: float | None = None
) -> OptionSlice:
    f = forward if forward is not None else draw(forwards)
    n = draw(st.integers(min_value=1, max_value=8))
    ks = sorted(set(draw(st.lists(st.floats(-2.5, 2.5), min_size=n, max_size=n))))
    ks = [k for i, k in enumerate(ks) if i == 0 or k - ks[i - 1] > 1e-6]
    ivs = draw(st.lists(vols, min_size=len(ks), max_size=len(ks)))
    return OptionSlice(
        expiry_ns=expiry if expiry is not None else draw(expiries),
        forward=f,
        strikes=tuple(f * math.exp(k) for k in ks),
        ivs=tuple(ivs),
        asof_ns=0,
    )


def assert_monotone(probability_above, forward: float, k1: float, dk: float) -> None:  # type: ignore[no-untyped-def]
    low, high = forward * math.exp(k1), forward * math.exp(k1 + dk)
    p_low, p_high = probability_above(low), probability_above(high)
    assert 0.0 <= p_high <= 1.0
    assert 0.0 <= p_low <= 1.0
    assert p_high <= p_low + EPS, (low, high, p_low, p_high)


@SETTINGS
@given(forward=forwards, sigma=vols, expiry=expiries, k1=log_m, dk=steps)
def test_t040_closed_form_digital_non_increasing(
    *, forward: float, sigma: float, expiry: int, k1: float, dk: float
) -> None:
    t = expiry / (365 * NS_PER_DAY)
    assert_monotone(lambda k: digital_call_probability(forward, k, t, sigma), forward, k1, dk)


@SETTINGS
@given(forward=forwards, sigma=vols, expiry=expiries, k1=log_m, dk=steps, n=st.integers(1, 6))
def test_t040_flat_slice_non_increasing_and_closed_form(
    *, forward: float, sigma: float, expiry: int, k1: float, dk: float, n: int
) -> None:
    sl = OptionSlice(
        expiry_ns=expiry,
        forward=forward,
        strikes=tuple(forward * math.exp(0.1 * (i - n // 2)) for i in range(n)),
        ivs=(sigma,) * n,
        asof_ns=0,
    )
    assert_monotone(sl.probability_above, forward, k1, dk)
    strike = forward * math.exp(k1)
    assert sl.probability_above(strike) == pytest.approx(
        digital_call_probability(forward, strike, sl.t_years, sigma), abs=1e-12
    )


@SETTINGS
@given(sl=smiles(), k1=log_m, dk=steps)
def test_t040_smiled_slice_non_increasing(sl: OptionSlice, k1: float, dk: float) -> None:
    assert_monotone(sl.probability_above, sl.forward, k1, dk)


@SETTINGS
@given(
    sl=smiles(), i=st.integers(0, 7), j=st.integers(0, 7), nudge=st.sampled_from([-1e-9, 0.0, 1e-9])
)
def test_t040_monotone_across_smile_nodes(sl: OptionSlice, i: int, j: int, nudge: float) -> None:
    nodes = sl.strikes
    a, b = sorted((nodes[i % len(nodes)], nodes[j % len(nodes)]))
    lo, hi = a * (1.0 + nudge), b
    if lo > hi:
        lo, hi = hi, lo
    assert sl.probability_above(hi) <= sl.probability_above(lo) + EPS


@SETTINGS
@given(
    lower=smiles(expiry=2 * NS_PER_DAY, forward=100.0),
    upper=smiles(expiry=9 * NS_PER_DAY, forward=101.0),
    frac=st.floats(min_value=0.01, max_value=0.99),
    k1=log_m,
    dk=steps,
)
def test_t040_interpolated_slice_non_increasing(
    lower: OptionSlice, upper: OptionSlice, frac: float, k1: float, dk: float
) -> None:
    target = lower.expiry_ns + int(frac * (upper.expiry_ns - lower.expiry_ns))
    sl = interpolate_total_variance(lower, upper, target)
    assert_monotone(sl.probability_above, sl.forward, k1, dk)
