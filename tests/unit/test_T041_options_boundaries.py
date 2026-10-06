"""T041 - deep ITM/OTM, tiny T and huge sigma approach the expected limits without NaN/overflow.

Warnings are errors here, so any numpy/scipy overflow or invalid-value warning fails.
"""

from __future__ import annotations

import math

import pytest

from cma.domain.time import NS_PER_DAY
from cma.models.implied_probability import (
    OptionSlice,
    black_call_price,
    digital_call_probability,
    digital_probability_from_spot,
    digital_put_probability,
)

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("error")]

F = 100_000.0


def in_unit_interval(p: float) -> bool:
    return math.isfinite(p) and 0.0 <= p <= 1.0


@pytest.mark.parametrize(
    ("forward", "strike", "t", "sigma", "expected", "tol"),
    [
        # deep in the money: P -> 1
        (F, 1e-300, 1.0, 0.5, 1.0, 0.0),
        (F, 1.0, 1.0, 0.5, 1.0, 1e-12),
        (1e300, 1e-300, 1.0, 0.8, 1.0, 0.0),
        # deep out of the money: P -> 0
        (F, 1e300, 1.0, 0.5, 0.0, 0.0),
        (F, 1e12, 1.0, 0.5, 0.0, 1e-12),
        (1e-300, 1e300, 1.0, 0.8, 0.0, 0.0),
        # tiny maturity: step function, 1/2 at the money
        (F, F, 1e-15, 0.5, 0.5, 1e-7),
        (F, F * (1 - 1e-4), 1e-12, 0.5, 1.0, 1e-12),
        (F, F * (1 + 1e-4), 1e-12, 0.5, 0.0, 1e-12),
        # explicit T = 0 and sigma = 0 limits (exactly 0.5 at F == K)
        (F, F, 0.0, 0.5, 0.5, 0.0),
        (F, F - 1.0, 0.0, 0.5, 1.0, 0.0),
        (F, F + 1.0, 0.0, 0.5, 0.0, 0.0),
        (F, F, 1.0, 0.0, 0.5, 0.0),
        (F, F - 1.0, 1.0, 0.0, 1.0, 0.0),
        (F, F + 1.0, 1.0, 0.0, 0.0, 0.0),
        # huge volatility: the forward-measure mass escapes to 0, P -> 0 for any K > 0
        (F, F, 1.0, 1e3, 0.0, 1e-12),
        (F, 1e-6, 1.0, 1e10, 0.0, 1e-12),
        (F, F, 1.0, 1e300, 0.0, 0.0),
        (F, F, 1e10, 1e300, 0.0, 0.0),  # sigma * sqrt(T) overflows to inf
        # non-positive strike: S_T > K almost surely
        (F, 0.0, 1.0, 0.5, 1.0, 0.0),
        (F, -5.0, 1.0, 0.5, 1.0, 0.0),
    ],
)
def test_t041_digital_limits(
    *, forward: float, strike: float, t: float, sigma: float, expected: float, tol: float
) -> None:
    p = digital_call_probability(forward, strike, t, sigma)
    assert in_unit_interval(p)
    assert abs(p - expected) <= tol
    q = digital_put_probability(forward, strike, t, sigma)
    assert in_unit_interval(q)
    if strike > 0:
        assert abs((p + q) - 1.0) <= 1e-12


@pytest.mark.parametrize("slope", [1e-300, -1e-300, 1e300, -1e300, 1e-3, -1e-3])
@pytest.mark.parametrize(
    ("strike", "t", "sigma"),
    [(F, 1e-12, 0.5), (1e-300, 1.0, 0.5), (F, 1.0, 1e300), (F, 30 / 365, 0.6)],
)
def test_t041_smile_slope_term_never_produces_nan(
    slope: float, strike: float, t: float, sigma: float
) -> None:
    p = digital_call_probability(F, strike, t, sigma, dsigma_dstrike=slope)
    q = digital_put_probability(F, strike, t, sigma, dsigma_dstrike=slope)
    assert in_unit_interval(p)
    assert in_unit_interval(q)


def test_t041_tiny_t_converges_monotonically_to_step() -> None:
    above = [digital_call_probability(F, F * 0.999, t, 0.5) for t in (1e-2, 1e-4, 1e-6, 1e-8)]
    below = [digital_call_probability(F, F * 1.001, t, 0.5) for t in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert above == sorted(above)
    assert above[-1] == pytest.approx(1.0, abs=1e-12)
    assert below == sorted(below, reverse=True)
    assert below[-1] == pytest.approx(0.0, abs=1e-12)


def test_t041_call_price_bounds_at_extremes() -> None:
    for strike, t, sigma in [(1e-300, 1.0, 0.5), (1e300, 1.0, 0.5), (F, 0.0, 0.5), (F, 1.0, 1e300)]:
        c = black_call_price(F, strike, t, sigma)
        assert math.isfinite(c)
        assert max(F - strike, 0.0) <= c <= F


def test_t041_spot_helper_extremes() -> None:
    assert digital_probability_from_spot(F, 1e300, 1.0, 0.5) == 0.0
    assert digital_probability_from_spot(F, 1e-300, 1.0, 0.5) == 1.0
    assert digital_probability_from_spot(F, F, 0.0, 0.5) == 0.5
    with pytest.raises(ValueError, match="overflow"):
        digital_probability_from_spot(F, F, 1.0, 0.5, rate=1e6)


@pytest.mark.parametrize(
    "bad",
    [
        {"forward": math.nan},
        {"forward": 0.0},
        {"forward": -1.0},
        {"forward": math.inf},
        {"strike": math.nan},
        {"strike": math.inf},
        {"t_years": -1e-9},
        {"t_years": math.inf},
        {"sigma": -0.1},
        {"sigma": math.nan},
        {"dsigma_dstrike": math.nan},
    ],
)
def test_t041_invalid_inputs_raise_instead_of_nan(bad: dict[str, float]) -> None:
    args = {"forward": F, "strike": F, "t_years": 1.0, "sigma": 0.5, "dsigma_dstrike": 0.0}
    args.update(bad)
    with pytest.raises(ValueError, match="must be"):
        digital_call_probability(
            args["forward"],
            args["strike"],
            args["t_years"],
            args["sigma"],
            dsigma_dstrike=args["dsigma_dstrike"],
        )


def test_t041_slice_with_extreme_strikes_and_vols() -> None:
    sl = OptionSlice(
        expiry_ns=NS_PER_DAY // 24,
        forward=F,
        strikes=(1e-200, F * 0.5, F, F * 2.0, 1e200),
        ivs=(50.0, 3.0, 0.2, 3.0, 50.0),
        asof_ns=0,
    )
    values = [sl.probability_above(k) for k in (1e-300, 1e-200, 1.0, F, 1e6, 1e200, 1e300)]
    assert all(in_unit_interval(v) for v in values)
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(1.0, abs=1e-12)
    assert values[-1] == pytest.approx(0.0, abs=1e-12)
    assert sl.probability_above(0.0) == 1.0
    with pytest.raises(ValueError, match="finite"):
        sl.probability_above(math.nan)


def test_t041_slice_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="after asof"):
        OptionSlice(expiry_ns=0, forward=F, strikes=(F,), ivs=(0.5,), asof_ns=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        OptionSlice(expiry_ns=NS_PER_DAY, forward=F, strikes=(F, F), ivs=(0.5, 0.5), asof_ns=0)
    with pytest.raises(ValueError, match="positive"):
        OptionSlice(expiry_ns=NS_PER_DAY, forward=F, strikes=(F,), ivs=(0.0,), asof_ns=0)
    with pytest.raises(ValueError, match="positive and finite"):
        OptionSlice(expiry_ns=NS_PER_DAY, forward=math.nan, strikes=(F,), ivs=(0.5,), asof_ns=0)
