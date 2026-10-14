"""Data-quality checks for template spectra."""
from itertools import combinations

import numpy as np

from .measure import HALPHA, HBETA


def wavelength_unit_problem(t):
    """Template whose wavelength array cannot be in Å (e.g. stored in µm)."""
    return bool(t.wav[-1] < 2000.0)


def strong_negative_pixels(t, lo=HBETA - 400.0, hi=HALPHA + 400.0):
    """Number of pixels in [lo, hi] with flux < -1 x median|flux| (unphysical)."""
    m = (t.wav >= lo) & (t.wav <= hi)
    if m.sum() < 5:
        return 0
    scale = np.median(np.abs(t.flux[m]))
    return int(np.sum(t.flux[m] < -scale))


def max_abs_over_median(t, lo=HBETA - 400.0, hi=HALPHA + 400.0):
    m = (t.wav >= lo) & (t.wav <= hi)
    if m.sum() < 5:
        return np.nan
    return float(np.max(np.abs(t.flux[m])) / np.median(np.abs(t.flux[m])))


def find_duplicates(templates, min_overlap=1500.0, corr_min=0.9995, npts=800):
    """Pairs of templates of the same SN whose spectra are (near-)identical.

    Spectra are compared on their common wavelength range after removing a
    constant scale factor; returns (t1, t2, corr, scale) tuples.
    """
    by = {}
    for t in templates:
        by.setdefault((t.folder, t.name), []).append(t)
    out = []
    for lst in by.values():
        for a, b in combinations(lst, 2):
            if wavelength_unit_problem(a) or wavelength_unit_problem(b):
                continue
            lo, hi = max(a.wav[0], b.wav[0]), min(a.wav[-1], b.wav[-1])
            if hi - lo < min_overlap:
                continue
            g = np.linspace(lo, hi, npts)
            fa, fb = np.interp(g, a.wav, a.flux), np.interp(g, b.wav, b.flux)
            if np.std(fa) == 0 or np.std(fb) == 0:
                continue
            cc = float(np.corrcoef(fa, fb)[0, 1])
            if cc > corr_min:
                scale = float(np.sum(fa * fb) / np.sum(fa * fa))
                out.append((a, b, cc, scale))
    return out
