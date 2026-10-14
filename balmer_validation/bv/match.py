"""Identify the archive spectrum each template was made from.

Spectra are compared through their *features*: log-flux on a common rest-frame
grid with a quadratic (in wavelength) removed, so differences in flux
calibration or reddening do not matter.  The low-order term that was removed is
kept separately: the template/archive flux ratio at Hβ relative to Hα tells how
differently the two versions are calibrated, which propagates directly into
Hα/Hβ.
"""
import numpy as np

from .measure import HALPHA, HBETA


def _prep(w, f, grid):
    fi = np.interp(grid, w, f, left=np.nan, right=np.nan)
    return fi


def compare(wt, ft, wa, fa, min_overlap=1500.0, step=4.0):
    lo, hi = max(wt[0], wa[0], 3300.0), min(wt[-1], wa[-1], 9500.0)
    if hi - lo < min_overlap:
        return None
    grid = np.arange(lo, hi, step)
    a = _prep(wt, ft, grid)
    b = _prep(wa, fa, grid)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < 100:
        return None
    x = (grid[ok] - grid[ok].mean()) / (np.ptp(grid[ok]) / 2.0)
    la, lb = np.log(a[ok]), np.log(b[ok])
    pa = np.polyfit(x, la, 2)
    pb = np.polyfit(x, lb, 2)
    ra, rb = la - np.polyval(pa, x), lb - np.polyval(pb, x)
    if ra.std() == 0 or rb.std() == 0:
        return None
    corr = float(np.corrcoef(ra, rb)[0, 1])
    # relative calibration: (template/archive) at Hβ divided by the same at Hα
    tilt = np.nan
    xs = lambda lam: (lam - grid[ok].mean()) / (np.ptp(grid[ok]) / 2.0)  # noqa: E731
    if lo < HBETA - 200 and hi > HALPHA + 200:
        d = np.polyval(pa - pb, xs(HBETA)) - np.polyval(pa - pb, xs(HALPHA))
        tilt = float(np.exp(d))
    return dict(corr=corr, overlap=float(hi - lo), hb_ha_calib_ratio=tilt)
