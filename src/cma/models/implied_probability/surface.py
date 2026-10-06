"""Smile-consistent option-implied probabilities with explicit expiry alignment (s.10.2).

Slice model
-----------
Total implied variance ``w(k) = sigma(k)^2 T`` is linear in log-moneyness ``k = ln(K/F)``
between quoted strikes and flat (constant vol) outside them. The terminal probability is
the Breeden-Litzenberger digital ``P(S_T > K) = -dC/dK`` of Black prices on that smile,
evaluated analytically::

    P(k) = N(d2) - phi(d2) w'(k) / (2 sqrt(w)),    d2 = -k / sqrt(w) - sqrt(w) / 2

(``N(d2) - F phi(d1) sqrt(T) dsigma/dK`` written in ``k``). On a linear piece ``dP/dk`` has
the sign of ``-g(k)`` (Durrleman's density factor), which is an explicit quadratic there,
and ``P`` jumps where the slope of ``w`` changes. Interpolated or noisy smiles can therefore
imply negative densities, i.e. ``P`` *increasing* in ``K``.

Kinks: where the slope of ``w`` changes (every quoted strike) the call curve has a kink, so
the analytic digital jumps there - an artificial atom of the interpolation. Inside a small
window around each such strike (half-width ``0.1 sqrt(w)``, at most a quarter of the
distance to the neighbouring strikes) the central-difference BL digital is used instead,
which spreads the atom linearly; ``P`` is then continuous and contract strikes that coincide
with quoted strikes get a stable value. ``OptionSlice.kinks`` reports the windows and atoms.

Monotone guard: the reported value is the average of the lower envelope
``inf_{k' <= k} P(k')`` and the upper envelope ``sup_{k' >= k} P(k')``, both computed
exactly piece by piece (no grid). The result is non-increasing in ``K``, lies in ``[0, 1]``
and equals the unguarded value wherever that is already consistent (e.g. any flat smile,
where it reduces to the closed-form ``N(d2)``).

Expiry alignment
----------------
Nothing is aligned silently: a target expiry must equal a slice expiry unless an explicit
:class:`ExpiryAlignment` rule allows ``EXACT`` within a tolerance or
``INTERPOLATE_TOTAL_VARIANCE`` (linear in time at fixed log-moneyness between the two
bracketing slices, no extrapolation); otherwise :class:`ExpiryMismatchError` is raised.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Final

from scipy.special import ndtr

from cma.domain.enums import MappingStatus, Operator
from cma.domain.errors import ExpiryMismatchError
from cma.domain.models import ContractMapping
from cma.domain.numbers import ONE, ZERO, from_float
from cma.domain.time import iso_from_ns
from cma.models.implied_probability.black_scholes import (
    YEAR_NS,
    FloatLike,
    black_call_price,
    clip_probability,
    log_normal_pdf,
    signed_exp,
    year_fraction,
)

PROBABILITY_QUANTUM: Final = Decimal("1e-12")
_NODE_MERGE_TOL: Final = 1e-12
# Half-width of the central-difference window at a smile kink, in units of sqrt(w) there,
# capped at a fraction of the distance to the neighbouring quoted strikes.
KINK_WINDOW_SQRT_W: Final = 0.1
KINK_WINDOW_MAX_SPACING_FRACTION: Final = 0.25
KINK_NOTE_THRESHOLD: Final = 1e-3  # atoms larger than this are reported in mapping notes


def _raw_probability(k: float, w: float, slope: float) -> float:
    """Analytic BL ``P(S_T > K)`` at log-moneyness ``k`` on a linear-variance piece."""
    s = math.sqrt(w)
    d2 = -k / s - 0.5 * s
    base = float(ndtr(d2))
    if slope == 0.0:
        return clip_probability(base)
    log_term = log_normal_pdf(d2) + math.log(abs(slope)) - math.log(2.0 * s)
    return clip_probability(base - signed_exp(log_term, slope))


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    if a == 0.0:
        return [] if b == 0.0 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
    if q == 0.0:
        return [0.0]
    return [q / a, c / q]


def _density_sign_pieces(
    k_lo: float, k_hi: float, w_lo: float, slope: float
) -> list[tuple[float, float, bool]]:
    """Split ``[k_lo, k_hi)`` where Durrleman's ``g`` changes sign.

    Returns ``(start, end, increasing)`` sub-intervals; ``increasing`` marks negative density
    (raw ``P`` non-decreasing in ``k``). With ``w = w_lo + slope*u``, ``k = k_lo + u``,
    ``G(u) = w^2 g = (c + slope*u/2)^2 - slope^2/4 * w - slope^2/16 * w^2`` is quadratic.
    """
    length = k_hi - k_lo
    c = w_lo - slope * k_lo / 2.0
    s2 = slope * slope
    a = s2 / 4.0 - s2 * s2 / 16.0
    b = c * slope - s2 * slope / 4.0 - s2 * slope * w_lo / 8.0
    c0 = c * c - s2 * w_lo / 4.0 - s2 * w_lo * w_lo / 16.0
    cuts = sorted({u for u in _quadratic_roots(a, b, c0) if 0.0 < u < length})
    bounds = [0.0, *cuts, length]
    pieces: list[tuple[float, float, bool]] = []
    for u0, u1 in itertools.pairwise(bounds):
        mid = 0.5 * (u0 + u1)
        increasing = (a * mid + b) * mid + c0 < 0.0
        start, end = k_lo + u0, k_lo + u1
        if pieces and pieces[-1][2] is increasing:
            pieces[-1] = (pieces[-1][0], end, increasing)
        else:
            pieces.append((start, end, increasing))
    return pieces


@dataclass(frozen=True, slots=True)
class _Line:
    """Total variance ``w(k) = w_ref + slope (k - k_ref)`` on ``[k_lo, k_hi)``."""

    k_lo: float
    k_hi: float
    k_ref: float
    w_ref: float
    slope: float

    def variance(self, k: float) -> float:
        return self.w_ref + self.slope * (k - self.k_ref)

    def probability(self, k: float) -> float:
        return _raw_probability(k, self.variance(k), self.slope)


@dataclass(frozen=True, slots=True)
class _Piece:
    """A monotone stretch of the (smoothed) digital ``P(k)``.

    ``line`` set: analytic BL on that variance line. ``line`` None: a kink window where ``P``
    is the central-difference BL, i.e. linear between ``start_value`` and ``end_value``.
    """

    k_lo: float  # -inf for the left wing
    k_hi: float  # +inf for the right wing
    increasing: bool  # P non-decreasing here (negative implied density)
    start_value: float  # P at k_lo (right limit); 1.0 for the left wing
    end_value: float  # P left limit at k_hi; 0.0 for the right wing
    line: _Line | None
    lower_before: float = math.inf  # inf of P over k < k_lo
    upper_after: float = -math.inf  # sup of P over k >= k_hi

    def value(self, k: float) -> float:
        if self.line is not None:
            return self.line.probability(k)
        frac = (k - self.k_lo) / (self.k_hi - self.k_lo)
        return self.start_value + frac * (self.end_value - self.start_value)


@dataclass(frozen=True, slots=True)
class KinkWindow:
    """Where linear total-variance interpolation puts a kink (an atom) in the call curve."""

    strike: float
    strike_lo: float  # central-difference window [strike_lo, strike_hi)
    strike_hi: float
    atom: float  # analytic left limit - right limit of P at the strike (<0: negative mass)


@dataclass(frozen=True)
class OptionSlice:
    """One expiry of an implied-volatility smile.

    ``forward`` is the forward for ``expiry_ns`` (equal to spot when r = q = 0), ``strikes``
    strictly increasing and positive, ``ivs`` the matching annualised implied vols (ACT/365
    from ``asof_ns``).
    """

    expiry_ns: int
    forward: float
    strikes: tuple[float, ...]
    ivs: tuple[float, ...]
    asof_ns: int
    t_years: float = field(init=False, compare=False)
    k_nodes: tuple[float, ...] = field(init=False, compare=False, repr=False)
    w_nodes: tuple[float, ...] = field(init=False, compare=False, repr=False)
    kinks: tuple[KinkWindow, ...] = field(init=False, compare=False, repr=False)
    _lines: tuple[_Line, ...] = field(init=False, compare=False, repr=False)
    _line_starts: tuple[float, ...] = field(init=False, compare=False, repr=False)
    _pieces: tuple[_Piece, ...] = field(init=False, compare=False, repr=False)
    _starts: tuple[float, ...] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        strikes = tuple(float(x) for x in self.strikes)
        ivs = tuple(float(x) for x in self.ivs)
        forward = float(self.forward)
        if not math.isfinite(forward) or forward <= 0.0:
            raise ValueError(f"forward must be positive and finite, got {self.forward!r}")
        if not strikes or len(strikes) != len(ivs):
            raise ValueError("strikes and ivs must be non-empty and of equal length")
        if any(not math.isfinite(x) or x <= 0.0 for x in strikes):
            raise ValueError("strikes must be positive and finite")
        if any(b <= a for a, b in itertools.pairwise(strikes)):
            raise ValueError("strikes must be strictly increasing")
        if any(not math.isfinite(v) or v <= 0.0 for v in ivs):
            raise ValueError("implied vols must be positive and finite")
        if self.expiry_ns <= self.asof_ns:
            raise ValueError("slice expiry must be after asof (expired slices are unusable)")
        t = year_fraction(self.asof_ns, self.expiry_ns)
        log_f = math.log(forward)
        k_nodes = tuple(math.log(x) - log_f for x in strikes)
        if any(b <= a for a, b in itertools.pairwise(k_nodes)):
            raise ValueError("strikes collapse to identical log-moneyness")
        w_nodes = tuple(v * v * t for v in ivs)
        if any(not math.isfinite(w) or w <= 0.0 for w in w_nodes):
            raise ValueError("total variance must be positive and finite")
        lines = _build_lines(k_nodes, w_nodes)
        pieces, windows = _build_pieces(k_nodes, w_nodes, lines)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "ivs", ivs)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "t_years", t)
        object.__setattr__(self, "k_nodes", k_nodes)
        object.__setattr__(self, "w_nodes", w_nodes)
        object.__setattr__(self, "_lines", lines)
        object.__setattr__(self, "_line_starts", tuple(line.k_lo for line in lines))
        object.__setattr__(self, "_pieces", pieces)
        object.__setattr__(self, "_starts", tuple(p.k_lo for p in pieces))
        object.__setattr__(
            self,
            "kinks",
            tuple(
                KinkWindow(
                    strike=strikes[i],
                    strike_lo=forward * math.exp(lo),
                    strike_hi=forward * math.exp(hi),
                    atom=atom,
                )
                for i, lo, hi, atom in windows
            ),
        )

    # ---------------------------------------------------------------- smile
    def log_moneyness(self, strike: FloatLike) -> float:
        k = float(strike)
        if not math.isfinite(k) or k <= 0.0:
            raise ValueError(f"strike must be positive and finite, got {strike!r}")
        return math.log(k) - math.log(self.forward)

    def _line(self, k: float) -> _Line:
        return self._lines[bisect_right(self._line_starts, k) - 1]

    def _piece(self, k: float) -> _Piece:
        return self._pieces[bisect_right(self._starts, k) - 1]

    def total_variance(self, k: float) -> float:
        """Interpolated total variance at log-moneyness ``k`` (flat beyond the wings)."""
        return self._line(k).variance(k)

    def implied_vol(self, strike: FloatLike) -> float:
        return math.sqrt(self.total_variance(self.log_moneyness(strike)) / self.t_years)

    def call_price(self, strike: FloatLike) -> float:
        """Undiscounted Black call on the interpolated smile (diagnostics / BL checks)."""
        return black_call_price(self.forward, strike, self.t_years, self.implied_vol(strike))

    # ---------------------------------------------------------------- probabilities
    def raw_probability_above(self, strike: FloatLike) -> float:
        """Analytic BL digital, unguarded and unsmoothed (right-continuous at kinks)."""
        if float(strike) <= 0.0:
            return 1.0
        k = self.log_moneyness(strike)
        return self._line(k).probability(k)

    def probability_above(self, strike: FloatLike) -> float:
        """Guarded smile-consistent ``P(S_T > K)``: non-increasing in ``K``, in ``[0, 1]``."""
        value = float(strike)
        if not math.isfinite(value):
            raise ValueError(f"strike must be finite, got {strike!r}")
        if value <= 0.0:
            return 1.0
        k = self.log_moneyness(value)
        piece = self._piece(k)
        if piece.increasing:
            lower = min(piece.lower_before, piece.start_value)
            upper = max(piece.upper_after, piece.end_value)
        else:
            current = piece.value(k)
            lower = min(piece.lower_before, current)
            upper = max(piece.upper_after, current)
        return clip_probability(0.5 * (lower + upper))

    def probability_below(self, strike: FloatLike) -> float:
        """``P(S_T < K)`` (= ``P(S_T <= K)`` under the continuous model)."""
        return clip_probability(1.0 - self.probability_above(strike))

    def kink_near(self, strike: FloatLike) -> KinkWindow | None:
        """The kink window containing ``strike``, if any (value there is smoothed)."""
        value = float(strike)
        for window in self.kinks:
            if window.strike_lo <= value < window.strike_hi:
                return window
        return None

    @property
    def guard_active(self) -> bool:
        """True when the smoothed smile implies negative density somewhere."""
        return any(p.increasing for p in self._pieces)


def _build_lines(k_nodes: tuple[float, ...], w_nodes: tuple[float, ...]) -> tuple[_Line, ...]:
    lines = [_Line(-math.inf, k_nodes[0], k_nodes[0], w_nodes[0], 0.0)]
    for i in range(len(k_nodes) - 1):
        k0, k1, w0, w1 = k_nodes[i], k_nodes[i + 1], w_nodes[i], w_nodes[i + 1]
        lines.append(_Line(k0, k1, k0, w0, (w1 - w0) / (k1 - k0)))
    lines.append(_Line(k_nodes[-1], math.inf, k_nodes[-1], w_nodes[-1], 0.0))
    return tuple(lines)


def _build_pieces(
    k_nodes: tuple[float, ...], w_nodes: tuple[float, ...], lines: tuple[_Line, ...]
) -> tuple[tuple[_Piece, ...], list[tuple[int, float, float, float]]]:
    """Monotone pieces of the smoothed digital plus the kink windows ``(i, lo, hi, atom)``.

    ``lines[j]`` spans ``[node j-1, node j)`` (wings: j = 0 and j = n). A node where the
    variance slope changes gets a central-difference window of half-width
    ``min(KINK_WINDOW_SQRT_W * sqrt(w), KINK_WINDOW_MAX_SPACING_FRACTION * spacing)``.
    """
    n = len(k_nodes)
    half: list[float] = []
    for i in range(n):
        if lines[i].slope == lines[i + 1].slope:
            half.append(0.0)
            continue
        width = KINK_WINDOW_SQRT_W * math.sqrt(w_nodes[i])
        if i > 0:
            width = min(width, KINK_WINDOW_MAX_SPACING_FRACTION * (k_nodes[i] - k_nodes[i - 1]))
        if i < n - 1:
            width = min(width, KINK_WINDOW_MAX_SPACING_FRACTION * (k_nodes[i + 1] - k_nodes[i]))
        half.append(width)

    raw: list[_Piece] = []
    windows: list[tuple[int, float, float, float]] = []
    for j, line in enumerate(lines):
        lo = line.k_lo + (half[j - 1] if j > 0 else 0.0)
        hi = line.k_hi - (half[j] if j < n else 0.0)
        if math.isinf(lo) or line.slope == 0.0:
            spans = [(lo, hi, False)]
        else:
            spans = _density_sign_pieces(lo, hi, line.variance(lo), line.slope)
        for a, b, increasing in spans:
            start = 1.0 if math.isinf(a) else line.probability(a)
            end = 0.0 if math.isinf(b) else line.probability(b)
            raw.append(_Piece(a, b, increasing, start, end, line))
        if j < n and half[j] > 0.0:
            k, h = k_nodes[j], half[j]
            start, end = line.probability(k - h), lines[j + 1].probability(k + h)
            raw.append(_Piece(k - h, k + h, end > start, start, end, None))
            atom = line.probability(k) - lines[j + 1].probability(k)
            windows.append((j, k - h, k + h, atom))

    pieces = list(raw)
    running_inf = math.inf
    for i, p in enumerate(raw):
        pieces[i] = dataclasses.replace(p, lower_before=running_inf)
        running_inf = min(running_inf, p.start_value if p.increasing else p.end_value)
    running_sup = -math.inf
    for i in range(len(pieces) - 1, -1, -1):
        p = pieces[i]
        pieces[i] = dataclasses.replace(p, upper_after=running_sup)
        running_sup = max(running_sup, p.end_value if p.increasing else p.start_value)
    return tuple(pieces), windows


def interpolate_total_variance(
    lower: OptionSlice, upper: OptionSlice, expiry_ns: int
) -> OptionSlice:
    """Slice at ``expiry_ns`` with ``w(k, T)`` linear in time at fixed log-moneyness.

    The forward is interpolated log-linearly in time. The node set is the union of both
    slices' log-moneyness nodes, so the result is again piecewise linear in ``k``.
    """
    if lower.asof_ns != upper.asof_ns:
        raise ValueError("slices must share the same asof")
    if not lower.expiry_ns < expiry_ns < upper.expiry_ns:
        raise ValueError("target expiry must lie strictly between the two slices")
    alpha = (expiry_ns - lower.expiry_ns) / (upper.expiry_ns - lower.expiry_ns)
    t = year_fraction(lower.asof_ns, expiry_ns)
    nodes: list[float] = []
    for k in sorted({*lower.k_nodes, *upper.k_nodes}):
        if not nodes or k - nodes[-1] > _NODE_MERGE_TOL:
            nodes.append(k)
    log_forward = (1.0 - alpha) * math.log(lower.forward) + alpha * math.log(upper.forward)
    forward = math.exp(log_forward)
    variances = [
        (1.0 - alpha) * lower.total_variance(k) + alpha * upper.total_variance(k) for k in nodes
    ]
    return OptionSlice(
        expiry_ns=expiry_ns,
        forward=forward,
        strikes=tuple(forward * math.exp(k) for k in nodes),
        ivs=tuple(math.sqrt(w / t) for w in variances),
        asof_ns=lower.asof_ns,
    )


class AlignmentMethod(StrEnum):
    EXACT = "EXACT"
    INTERPOLATE_TOTAL_VARIANCE = "INTERPOLATE_TOTAL_VARIANCE"


@dataclass(frozen=True, slots=True)
class ExpiryAlignment:
    """Explicit rule permitting a contract expiry to differ from the option expiries."""

    method: AlignmentMethod
    tolerance_ns: int = 0  # EXACT: max |slice expiry - target|
    max_gap_ns: int = 0  # INTERPOLATE_TOTAL_VARIANCE: max distance between bracketing slices

    def __post_init__(self) -> None:
        if self.tolerance_ns < 0 or self.max_gap_ns < 0:
            raise ValueError("alignment tolerances must be non-negative")

    @classmethod
    def exact(cls, tolerance_ns: int = 0) -> ExpiryAlignment:
        return cls(AlignmentMethod.EXACT, tolerance_ns=tolerance_ns)

    @classmethod
    def interpolate_total_variance(cls, max_gap_ns: int) -> ExpiryAlignment:
        return cls(AlignmentMethod.INTERPOLATE_TOTAL_VARIANCE, max_gap_ns=max_gap_ns)


@dataclass(frozen=True, slots=True)
class AlignedSlice:
    slice: OptionSlice
    method: str
    target_expiry_ns: int
    expiries_used: tuple[int, ...]
    notes: tuple[str, ...] = ()


class OptionSurface:
    """Slices sharing one valuation time, sorted by expiry."""

    def __init__(
        self, slices: Sequence[OptionSlice], *, underlying: str | None = None, source: str = ""
    ) -> None:
        if not slices:
            raise ValueError("an option surface needs at least one slice")
        ordered = tuple(sorted(slices, key=lambda s: s.expiry_ns))
        if len({s.asof_ns for s in ordered}) != 1:
            raise ValueError("all slices must share the same asof_ns")
        if len({s.expiry_ns for s in ordered}) != len(ordered):
            raise ValueError("duplicate slice expiries")
        self._slices = ordered
        self.underlying = underlying
        self.source = source

    @property
    def slices(self) -> tuple[OptionSlice, ...]:
        return self._slices

    @property
    def asof_ns(self) -> int:
        return self._slices[0].asof_ns

    @property
    def expiries(self) -> tuple[int, ...]:
        return tuple(s.expiry_ns for s in self._slices)

    def resolve(self, expiry_ns: int, rule: ExpiryAlignment | None = None) -> AlignedSlice:
        """The slice to use for ``expiry_ns``; raises ``ExpiryMismatchError`` if none is allowed."""
        if expiry_ns <= self.asof_ns:
            raise ExpiryMismatchError(
                f"target expiry {iso_from_ns(expiry_ns)} is not after the surface asof "
                f"{iso_from_ns(self.asof_ns)}"
            )
        for s in self._slices:
            if s.expiry_ns == expiry_ns:
                return AlignedSlice(s, AlignmentMethod.EXACT.value, expiry_ns, (s.expiry_ns,))
        nearest = min(self._slices, key=lambda s: abs(s.expiry_ns - expiry_ns))
        context = (
            f"target {iso_from_ns(expiry_ns)}; slices {[iso_from_ns(e) for e in self.expiries]}"
        )
        if rule is None:
            raise ExpiryMismatchError(
                f"no option slice expires at the contract expiry ({context}); nearest differs by "
                f"{abs(nearest.expiry_ns - expiry_ns)} ns. Pass an explicit ExpiryAlignment rule."
            )
        if rule.method is AlignmentMethod.EXACT:
            gap = abs(nearest.expiry_ns - expiry_ns)
            ties = [s for s in self._slices if abs(s.expiry_ns - expiry_ns) == gap]
            if gap > rule.tolerance_ns:
                raise ExpiryMismatchError(
                    f"nearest slice is {gap} ns away, beyond EXACT tolerance {rule.tolerance_ns} "
                    f"ns ({context})"
                )
            if len(ties) > 1:
                raise ExpiryMismatchError(f"two slices are equally close ({context}); ambiguous")
            note = (
                f"EXACT within tolerance: used slice {iso_from_ns(nearest.expiry_ns)} for target "
                f"{iso_from_ns(expiry_ns)} (difference {nearest.expiry_ns - expiry_ns} ns, "
                "slice's own maturity used)"
            )
            return AlignedSlice(
                nearest, AlignmentMethod.EXACT.value, expiry_ns, (nearest.expiry_ns,), (note,)
            )
        idx = bisect_right(self.expiries, expiry_ns)
        if idx == 0 or idx == len(self._slices):
            raise ExpiryMismatchError(
                "no bracketing slices for interpolation; extrapolation is not permitted "
                f"({context})"
            )
        lower, upper = self._slices[idx - 1], self._slices[idx]
        gap = upper.expiry_ns - lower.expiry_ns
        if gap > rule.max_gap_ns:
            raise ExpiryMismatchError(
                f"bracketing slices are {gap} ns apart, beyond max_gap {rule.max_gap_ns} ns "
                f"({context})"
            )
        interpolated = interpolate_total_variance(lower, upper, expiry_ns)
        note = (
            f"total variance interpolated linearly in time at fixed log-moneyness between "
            f"{iso_from_ns(lower.expiry_ns)} and {iso_from_ns(upper.expiry_ns)}"
        )
        return AlignedSlice(
            interpolated,
            AlignmentMethod.INTERPOLATE_TOTAL_VARIANCE.value,
            expiry_ns,
            (lower.expiry_ns, upper.expiry_ns),
            (note,),
        )

    def probability_above(
        self, strike: FloatLike, expiry_ns: int, rule: ExpiryAlignment | None = None
    ) -> float:
        return self.resolve(expiry_ns, rule).slice.probability_above(strike)

    def probability_below(
        self, strike: FloatLike, expiry_ns: int, rule: ExpiryAlignment | None = None
    ) -> float:
        return self.resolve(expiry_ns, rule).slice.probability_below(strike)


@dataclass(frozen=True, slots=True)
class ImpliedProbability:
    """Option-implied (risk-neutral) probability that a mapped contract pays YES."""

    contract_id: str
    mapping_version: int
    operator: Operator
    value: Decimal  # quantised to PROBABILITY_QUANTUM, in [0, 1]
    value_float: float
    method: str
    target_expiry_ns: int
    expiries_used: tuple[int, ...]
    notes: tuple[str, ...]


def _strike_float(value: Decimal) -> float:
    return float(value)  # explicit Decimal -> float boundary into option math


def implied_probability_for_mapping(
    mapping: ContractMapping,
    surface: OptionSurface,
    *,
    rule: ExpiryAlignment | None = None,
    start_price: FloatLike | None = None,
) -> ImpliedProbability:
    """Market-implied benchmark for ``mapping``'s YES payoff (scope s.10.2).

    GT/GE use ``P(S_T > K)``; LT/LE ``1 - P(S_T > K)``; BETWEEN the difference of digitals
    ``P(S_T > low) - P(S_T > high)``; UP/DOWN require the realised ``start_price`` (a digital
    struck at it). The target expiry is ``mapping.observation_end_ns`` and is aligned to the
    surface only as ``rule`` permits (``ExpiryMismatchError`` otherwise). This is a
    risk-neutral benchmark, not a real-world probability.
    """
    aligned = surface.resolve(mapping.observation_end_ns, rule)
    sl = aligned.slice
    notes = list(aligned.notes)
    op = mapping.operator
    if op in (Operator.GT, Operator.GE):
        value = sl.probability_above(_strike_float(mapping.strikes[0]))
        if op is Operator.GE:
            notes.append("GE treated as GT: the boundary carries no mass in the continuous model")
    elif op in (Operator.LT, Operator.LE):
        value = sl.probability_below(_strike_float(mapping.strikes[0]))
        if op is Operator.LE:
            notes.append("LE treated as LT: the boundary carries no mass in the continuous model")
    elif op is Operator.BETWEEN:
        low, high = (_strike_float(s) for s in mapping.strikes)
        value = max(0.0, sl.probability_above(low) - sl.probability_above(high))
        notes.append("BETWEEN = P(S_T > low) - P(S_T > high); endpoint inclusivity carries no mass")
    else:
        if start_price is None:
            raise ValueError(f"{op} contracts need the realised start_price")
        start = mapping.observation_start_ns
        if start is None or surface.asof_ns < start:
            raise ValueError("UP/DOWN start value is not yet observed at the surface asof")
        above = sl.probability_above(start_price)
        value = above if op is Operator.UP else 1.0 - above
        notes.append(f"{op}: digital struck at the realised start value; ties carry no mass")
    probes: list[float] = [_strike_float(k) for k in mapping.strikes]
    if start_price is not None:
        probes.append(float(start_price))
    for probe in probes:
        window = sl.kink_near(probe)
        if window is not None and abs(window.atom) > KINK_NOTE_THRESHOLD:
            notes.append(
                f"strike {probe:g} is within the kink window of quoted strike {window.strike:g}: "
                f"interpolation atom {window.atom:.4f} spread over "
                f"[{window.strike_lo:g}, {window.strike_hi:g})"
            )
    if mapping.observation_method != "POINT":
        notes.append(
            f"observation method {mapping.observation_method} approximated by the terminal value "
            f"at observation_end"
        )
    if surface.underlying is not None and surface.underlying not in mapping.underlyings:
        notes.append(
            f"basis not modelled: options on {surface.underlying} vs contract on "
            f"{'+'.join(mapping.underlyings)} ({mapping.resolution_source})"
        )
    if not mapping.review_status.at_least(MappingStatus.REVIEWED):
        notes.append(f"mapping status {mapping.review_status}: research only")
    clipped = clip_probability(value)
    decimal_value = min(ONE, max(ZERO, from_float(clipped, PROBABILITY_QUANTUM)))
    return ImpliedProbability(
        contract_id=mapping.contract_id,
        mapping_version=mapping.version,
        operator=op,
        value=decimal_value,
        value_float=clipped,
        method=f"BREEDEN_LITZENBERGER_SMILE[{aligned.method}]",
        target_expiry_ns=mapping.observation_end_ns,
        expiries_used=aligned.expiries_used,
        notes=tuple(notes),
    )


__all__ = [
    "PROBABILITY_QUANTUM",
    "YEAR_NS",
    "AlignedSlice",
    "AlignmentMethod",
    "ExpiryAlignment",
    "ImpliedProbability",
    "KinkWindow",
    "OptionSlice",
    "OptionSurface",
    "implied_probability_for_mapping",
    "interpolate_total_variance",
]
