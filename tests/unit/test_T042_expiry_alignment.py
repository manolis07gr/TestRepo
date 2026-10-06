"""T042 - an option surface with a mismatched expiry cannot be used silently.

An explicit ExpiryAlignment rule is required (EXACT within a tolerance, or total-variance
interpolation between bracketing slices); interpolated values lie between the bracketing
slices' values.
"""

from __future__ import annotations

import dataclasses

import pytest

from cma.domain.enums import Operator
from cma.domain.errors import ExpiryMismatchError
from cma.domain.models import ContractMapping
from cma.domain.time import NS_PER_DAY, NS_PER_HOUR, NS_PER_MIN, ns_from_iso8601
from cma.models.implied_probability import (
    AlignmentMethod,
    ExpiryAlignment,
    OptionSlice,
    OptionSurface,
    digital_call_probability,
    implied_probability_for_mapping,
)
from tests.unit.test_mapping_fixtures import as_reviewed, fixture_mapping, load_fixture

pytestmark = pytest.mark.unit

ASOF = ns_from_iso8601("2026-10-06T12:00:00Z")
EXP_1 = ns_from_iso8601("2026-10-07T08:00:00Z")  # Deribit-style 08:00 UTC expiries
EXP_2 = ns_from_iso8601("2026-10-09T08:00:00Z")
TARGET = ns_from_iso8601("2026-10-07T21:00:00Z")  # Kalshi 5 PM ET (EDT) observation
FWD = 111_000.0
STRIKES = (100_000.0, 105_000.0, 111_000.0, 117_000.0, 122_000.0)


def surface(
    ivs_1: tuple[float, ...], ivs_2: tuple[float, ...], fwd_2: float = FWD
) -> OptionSurface:
    return OptionSurface(
        [
            OptionSlice(expiry_ns=EXP_1, forward=FWD, strikes=STRIKES, ivs=ivs_1, asof_ns=ASOF),
            OptionSlice(expiry_ns=EXP_2, forward=fwd_2, strikes=STRIKES, ivs=ivs_2, asof_ns=ASOF),
        ],
        underlying="BTC-USD",
    )


SMILE = surface((0.62, 0.52, 0.45, 0.50, 0.58), (0.60, 0.53, 0.48, 0.52, 0.57))
FLAT = surface((0.5,) * 5, (0.5,) * 5)
INTERP = ExpiryAlignment.interpolate_total_variance(max_gap_ns=3 * NS_PER_DAY)


def test_t042_exact_expiry_needs_no_rule() -> None:
    aligned = SMILE.resolve(EXP_1)
    assert aligned.method == AlignmentMethod.EXACT.value
    assert aligned.expiries_used == (EXP_1,)
    assert SMILE.probability_above(111_000.0, EXP_1) == SMILE.slices[0].probability_above(111_000.0)


def test_t042_mismatched_expiry_without_rule_raises() -> None:
    with pytest.raises(ExpiryMismatchError, match="explicit ExpiryAlignment"):
        SMILE.probability_above(111_000.0, TARGET)


def test_t042_exact_rule_respects_tolerance() -> None:
    near = EXP_1 + 30 * NS_PER_MIN
    with pytest.raises(ExpiryMismatchError, match="beyond EXACT tolerance"):
        SMILE.resolve(near, ExpiryAlignment.exact(tolerance_ns=10 * NS_PER_MIN))
    aligned = SMILE.resolve(near, ExpiryAlignment.exact(tolerance_ns=NS_PER_HOUR))
    assert aligned.slice is SMILE.slices[0]
    assert aligned.expiries_used == (EXP_1,)
    assert aligned.notes
    assert "tolerance" in aligned.notes[0]
    with pytest.raises(ExpiryMismatchError):
        SMILE.resolve(TARGET, ExpiryAlignment.exact(tolerance_ns=NS_PER_HOUR))  # 13 h away


def test_t042_exact_rule_refuses_equidistant_slices() -> None:
    midpoint = (EXP_1 + EXP_2) // 2
    with pytest.raises(ExpiryMismatchError, match="ambiguous"):
        SMILE.resolve(midpoint, ExpiryAlignment.exact(tolerance_ns=3 * NS_PER_DAY))


@pytest.mark.parametrize("strike", [104_000.0, 108_000.0, 111_000.0, 113_500.0, 118_000.0])
@pytest.mark.parametrize("surf", [SMILE, FLAT, surface((0.45,) * 5, (0.55,) * 5, fwd_2=111_400.0)])
def test_t042_interpolation_lies_between_bracketing_slices(
    surf: OptionSurface, strike: float
) -> None:
    p1 = surf.slices[0].probability_above(strike)
    p2 = surf.slices[1].probability_above(strike)
    aligned = surf.resolve(TARGET, INTERP)
    p = aligned.slice.probability_above(strike)
    assert min(p1, p2) - 1e-12 <= p <= max(p1, p2) + 1e-12
    assert aligned.method == AlignmentMethod.INTERPOLATE_TOTAL_VARIANCE.value
    assert aligned.expiries_used == (EXP_1, EXP_2)
    assert aligned.slice.expiry_ns == TARGET
    assert surf.probability_above(strike, TARGET, INTERP) == p


def test_t042_flat_vol_interpolation_reproduces_closed_form() -> None:
    sl = FLAT.resolve(TARGET, INTERP).slice
    t = (TARGET - ASOF) / (365 * NS_PER_DAY)
    for strike in (95_000.0, 111_000.0, 125_000.0):
        assert sl.probability_above(strike) == pytest.approx(
            digital_call_probability(FWD, strike, t, 0.5), abs=1e-12
        )


def test_t042_total_variance_is_linear_in_time_at_fixed_log_moneyness() -> None:
    aligned = SMILE.resolve(TARGET, INTERP).slice
    s1, s2 = SMILE.slices
    alpha = (TARGET - EXP_1) / (EXP_2 - EXP_1)
    for k in (-0.08, -0.02, 0.0, 0.03, 0.07):
        expected = (1 - alpha) * s1.total_variance(k) + alpha * s2.total_variance(k)
        assert aligned.total_variance(k) == pytest.approx(expected, rel=1e-12)


def test_t042_interpolation_gap_and_extrapolation_limits() -> None:
    with pytest.raises(ExpiryMismatchError, match="max_gap"):
        SMILE.resolve(TARGET, ExpiryAlignment.interpolate_total_variance(max_gap_ns=NS_PER_DAY))
    with pytest.raises(ExpiryMismatchError, match="extrapolation"):
        SMILE.resolve(EXP_2 + NS_PER_HOUR, INTERP)
    with pytest.raises(ExpiryMismatchError, match="extrapolation"):
        SMILE.resolve(EXP_1 - NS_PER_HOUR, INTERP)
    with pytest.raises(ExpiryMismatchError, match="asof"):
        SMILE.resolve(ASOF, INTERP)


def _mapping_at(target: int) -> ContractMapping:
    base = fixture_mapping(load_fixture("nested_contracts.json")["mappings"][2])
    return as_reviewed(
        dataclasses.replace(
            base, observation_end_ns=target, observation_start_ns=target - NS_PER_MIN
        )
    )


def test_t042_mapping_level_alignment_is_explicit() -> None:
    mapping = _mapping_at(TARGET)
    assert mapping.operator is Operator.GT
    with pytest.raises(ExpiryMismatchError):
        implied_probability_for_mapping(mapping, SMILE)
    result = implied_probability_for_mapping(mapping, SMILE, rule=INTERP)
    assert result.expiries_used == (EXP_1, EXP_2)
    assert result.target_expiry_ns == TARGET
    assert "INTERPOLATE_TOTAL_VARIANCE" in result.method
    assert any("interpolated" in note for note in result.notes)
    p1 = SMILE.slices[0].probability_above(float(mapping.strike))
    p2 = SMILE.slices[1].probability_above(float(mapping.strike))
    assert min(p1, p2) <= result.value_float <= max(p1, p2)
