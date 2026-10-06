"""Option-implied benchmark for mapped contracts, BL consistency and kink handling (H3)."""

from __future__ import annotations

import dataclasses
import itertools
import math
from decimal import Decimal

import pytest

from cma.domain.enums import MappingStatus, Operator
from cma.domain.models import ContractMapping
from cma.domain.time import NS_PER_DAY, NS_PER_HOUR, NS_PER_MIN
from cma.models.implied_probability import (
    PROBABILITY_QUANTUM,
    OptionSlice,
    OptionSurface,
    digital_call_probability,
    digital_probability_from_spot,
    forward_from_spot,
    implied_probability_for_mapping,
)
from tests.unit.test_mapping_fixtures import as_reviewed, fixture_mapping, load_fixture

pytestmark = pytest.mark.unit

BASE = as_reviewed(fixture_mapping(load_fixture("nested_contracts.json")["mappings"][2]))
END = BASE.observation_end_ns
ASOF = END - NS_PER_DAY
FWD = 111_000.0
FLAT = OptionSurface(
    [
        OptionSlice(
            expiry_ns=END, forward=FWD, strikes=(100_000.0, 120_000.0), ivs=(0.5, 0.5), asof_ns=ASOF
        )
    ],
    underlying="BTC-USD@DERIBIT_INDEX",
)
T = 1 / 365


def mapping(operator: Operator, strikes: tuple[str, ...], **kw: object) -> ContractMapping:
    return dataclasses.replace(
        BASE,
        operator=operator,
        strikes=tuple(Decimal(s) for s in strikes),
        **kw,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("operator", "strikes", "expected"),
    [
        (Operator.GT, ("111999.99",), lambda: digital_call_probability(FWD, 111999.99, T, 0.5)),
        (Operator.GE, ("111999.99",), lambda: digital_call_probability(FWD, 111999.99, T, 0.5)),
        (Operator.LT, ("108000",), lambda: 1 - digital_call_probability(FWD, 108000, T, 0.5)),
        (Operator.LE, ("108000",), lambda: 1 - digital_call_probability(FWD, 108000, T, 0.5)),
        (
            Operator.BETWEEN,
            ("110000", "111999.99"),
            lambda: (
                digital_call_probability(FWD, 110000, T, 0.5)
                - digital_call_probability(FWD, 111999.99, T, 0.5)
            ),
        ),
    ],
)
def test_operators_on_flat_surface(
    operator: Operator, strikes: tuple[str, ...], expected: object
) -> None:
    result = implied_probability_for_mapping(mapping(operator, strikes), FLAT)
    target = expected()  # type: ignore[operator]
    assert result.value_float == pytest.approx(target, abs=1e-12)
    assert abs(float(result.value) - target) <= 1e-12
    assert Decimal(0) <= result.value <= Decimal(1)
    assert result.value == result.value.quantize(PROBABILITY_QUANTUM)
    assert result.method == "BREEDEN_LITZENBERGER_SMILE[EXACT]"
    assert result.expiries_used == (END,)
    assert result.contract_id == BASE.contract_id
    assert result.mapping_version == BASE.version
    assert any("basis not modelled" in n for n in result.notes)
    assert any("AVG_60S_BEFORE" in n for n in result.notes)
    if operator in (Operator.GE, Operator.LE):
        assert any("boundary" in n for n in result.notes)


def test_up_down_need_realised_start_value() -> None:
    up = mapping(
        Operator.UP,
        (),
        observation_method="CANDLE_OPEN_CLOSE_1H",
        observation_start_ns=END - NS_PER_HOUR,
        rounding_rule="TIES_RESOLVE_YES",
    )
    with pytest.raises(ValueError, match="start_price"):
        implied_probability_for_mapping(up, FLAT)
    with pytest.raises(ValueError, match="not yet observed"):
        implied_probability_for_mapping(up, FLAT, start_price=110_500.0)  # asof < window start
    started = OptionSurface(
        [
            OptionSlice(
                expiry_ns=END,
                forward=FWD,
                strikes=(FWD,),
                ivs=(0.5,),
                asof_ns=END - 30 * NS_PER_MIN,
            )
        ]
    )
    result = implied_probability_for_mapping(up, started, start_price=110_500.0)
    t = 30 / (365 * 24 * 60)
    assert result.value_float == pytest.approx(
        digital_call_probability(FWD, 110_500.0, t, 0.5), abs=1e-12
    )
    down = dataclasses.replace(up, operator=Operator.DOWN)
    assert implied_probability_for_mapping(
        down, started, start_price=110_500.0
    ).value_float == pytest.approx(1 - result.value_float, abs=1e-12)


def test_draft_mapping_is_flagged_research_only() -> None:
    draft = dataclasses.replace(BASE, review_status=MappingStatus.DRAFT)
    assert any("research only" in n for n in implied_probability_for_mapping(draft, FLAT).notes)


SMILE = OptionSlice(
    expiry_ns=END,
    forward=FWD,
    strikes=(100_000.0, 105_000.0, 111_000.0, 117_000.0, 122_000.0),
    ivs=(0.62, 0.52, 0.45, 0.50, 0.58),
    asof_ns=ASOF,
)


@pytest.mark.parametrize(
    "strike", [101_000.0, 103_000.0, 108_000.0, 114_000.0, 119_500.0, 125_000.0]
)
def test_analytic_bl_digital_matches_central_difference_of_call_prices(strike: float) -> None:
    h = 0.5
    fd = -(SMILE.call_price(strike + h) - SMILE.call_price(strike - h)) / (2 * h)
    assert SMILE.raw_probability_above(strike) == pytest.approx(fd, abs=1e-6)
    assert SMILE.probability_above(strike) == pytest.approx(fd, abs=1e-6)  # away from kinks


def test_kinks_are_smoothed_and_reported() -> None:
    atm = next(k for k in SMILE.kinks if k.strike == FWD)
    assert atm.atom > 0.04  # ~5pp artificial atom from linear-variance interpolation
    assert atm.strike_lo < FWD < atm.strike_hi
    # continuous across the quoted strike, with the atom spread over the window
    below = SMILE.probability_above(FWD * (1 - 1e-12))
    above = SMILE.probability_above(FWD * (1 + 1e-12))
    assert abs(below - above) < 1e-8
    left = SMILE.probability_above(atm.strike_lo * (1 - 1e-9))
    right = SMILE.probability_above(atm.strike_hi * (1 + 1e-9))
    assert left - right > atm.atom
    # the analytic (unsmoothed) digital really does jump at the node
    assert SMILE.raw_probability_above(FWD * (1 - 1e-12)) - SMILE.raw_probability_above(FWD) > 0.04
    surface = OptionSurface([SMILE])
    result = implied_probability_for_mapping(mapping(Operator.GT, ("111000",)), surface)
    assert any("kink window" in n for n in result.notes)
    far = implied_probability_for_mapping(mapping(Operator.GT, ("114000",)), surface)
    assert not any("kink window" in n for n in far.notes)


def test_flat_smile_has_no_kinks_and_guard_is_inactive() -> None:
    flat = FLAT.slices[0]
    assert flat.kinks == ()
    assert not flat.guard_active
    assert flat.implied_vol(150_000.0) == pytest.approx(0.5)


def test_negative_density_smile_is_guarded() -> None:
    # a sharp concave kink: vol collapses at one strike, implying negative density nearby
    arb = OptionSlice(
        expiry_ns=END,
        forward=FWD,
        strikes=(108_000.0, 111_000.0, 114_000.0),
        ivs=(0.9, 0.2, 0.9),
        asof_ns=ASOF,
    )
    grid = [FWD * math.exp(x / 400) for x in range(-200, 201)]
    raw = [arb.raw_probability_above(k) for k in grid]
    guarded = [arb.probability_above(k) for k in grid]
    assert any(b > a + 1e-6 for a, b in itertools.pairwise(raw))  # raw is non-monotone
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(guarded))
    assert arb.guard_active


def test_spot_helper_is_flat_vol_digital_with_zero_rates() -> None:
    assert digital_probability_from_spot(
        Decimal("111000"), Decimal("112000"), T, 0.5
    ) == pytest.approx(digital_call_probability(111_000.0, 112_000.0, T, 0.5), abs=0.0)
    assert forward_from_spot(100.0, 1.0, rate=0.05, carry_yield=0.05) == pytest.approx(100.0)
    assert forward_from_spot(100.0, 2.0, rate=0.03) == pytest.approx(100.0 * math.exp(0.06))
    with pytest.raises(TypeError):
        digital_probability_from_spot(True, 1.0, 1.0, 0.5)
