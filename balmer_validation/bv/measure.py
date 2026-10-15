"""Hα equivalent widths and Hα/Hβ flux ratios for SN spectra.

Two methods are implemented:

``replica_measure``
    Re-implementation of the procedure described in Wagle, Chatzopoulos & Baer
    (2025, ApJ 978, 112, Sect. 3.2 / Table A1 note): a single "generic"
    continuum (specutils ``fit_generic_continuum``: 3-pixel median smoothing +
    Chebyshev polynomial of degree 3) fitted over 4000-7000 Å with the Balmer
    lines excluded, followed by specutils ``equivalent_width`` on the
    continuum-normalised spectrum and ``line_flux`` on the continuum-subtracted
    spectrum.  This is our best reconstruction of how the figures under review
    were made; it is used to reproduce them, not as a recommended method.

``robust_measure``
    Local linear continuum through two sidebands outside the full line profile
    (velocity-defined, identical for Hα and Hβ), integrated net/emission/
    absorption parts, Monte-Carlo uncertainties that include both the flux
    noise and the arbitrariness of the window choice, and explicit quality
    flags.  A Balmer decrement is only reported when both lines are
    significant net emission and Hβ is not dominated by P-Cygni absorption.

Sign convention (both methods, as in the figures under review):
    EW = ∫ (1 - F/Fc) dλ, so emission gives EW < 0.
"""
import warnings

import numpy as np

C_KMS = 299792.458
HALPHA = 6562.80   # Å, air
HBETA = 4861.33    # Å, air

_trapz = getattr(np, "trapezoid", None) or np.trapz


def v2w(v_kms, lam0):
    """Doppler-shifted wavelength for a velocity in km/s (non-relativistic)."""
    return lam0 * (1.0 + v_kms / C_KMS)


def integrate(w, y, lo, hi):
    """Trapezoidal ∫ y dλ between lo and hi with linear interpolation at the edges."""
    if hi <= lo:
        return np.nan
    m = (w > lo) & (w < hi)
    ww = np.concatenate([[lo], w[m], [hi]])
    yy = np.concatenate([[np.interp(lo, w, y)], y[m], [np.interp(hi, w, y)]])
    return float(_trapz(yy, ww))


def covers(w, lo, hi, max_gap=50.0):
    """True if the spectrum covers [lo, hi] without gaps larger than max_gap Å."""
    if w[0] > lo or w[-1] < hi:
        return False
    m = (w >= lo) & (w <= hi)
    if m.sum() < 3:
        return False
    return np.max(np.diff(w[m])) <= max_gap


# ---------------------------------------------------------------------------
# Replica of the specutils procedure
# ---------------------------------------------------------------------------

def replica_continuum(w, f, cont_range=(4000.0, 7000.0), exclude_hw=100.0, degree=3,
                      median_window=3):
    """Generic continuum as in specutils.fit_generic_continuum, restricted to cont_range.

    Returns a callable continuum model, or None if the fit is impossible.
    """
    import astropy.units as u
    from astropy.modeling import models
    from specutils import SpectralRegion
    from specutils.fitting import fit_generic_continuum
    try:                                   # specutils >= 2.0
        from specutils import Spectrum
    except ImportError:                    # specutils 1.x (used in 2024)
        from specutils import Spectrum1D as Spectrum

    m = (w >= cont_range[0]) & (w <= cont_range[1])
    if m.sum() < 4 * (degree + 1):
        return None
    spec = Spectrum(flux=f[m] * u.Unit("erg cm-2 s-1 AA-1"), spectral_axis=w[m] * u.AA)
    regions = []
    for lam in (HBETA, HALPHA):
        lo, hi = lam - exclude_hw, lam + exclude_hw
        if hi > w[m][0] and lo < w[m][-1]:
            regions.append((lo * u.AA, hi * u.AA))
    excl = SpectralRegion(regions) if regions else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_generic_continuum(spec, median_window=median_window,
                                      model=models.Chebyshev1D(degree), exclude_regions=excl)

    def cont(x):
        return model(np.asarray(x) * u.AA).value
    return cont


def replica_measure(w, f, win_hw=100.0, exclude_hw=None, cont_range=(4000.0, 7000.0),
                    require_full_window=True):
    """Student-style measurement. Returns dict with EW_Ha, F_Ha, F_Hb, R (=F_Ha/F_Hb).

    win_hw      half-width (Å) of the integration windows around Hα and Hβ
    exclude_hw  half-width (Å) of the regions excluded from the continuum fit
                (defaults to win_hw)
    """
    exclude_hw = win_hw if exclude_hw is None else exclude_hw
    out = dict(EW_Ha=np.nan, F_Ha=np.nan, F_Hb=np.nan, R=np.nan)
    cont = replica_continuum(w, f, cont_range=cont_range, exclude_hw=exclude_hw)
    if cont is None:
        return out
    c = cont(w)
    for lam, key in ((HALPHA, "Ha"), (HBETA, "Hb")):
        lo, hi = lam - win_hw, lam + win_hw
        ok = covers(w, lo, hi) if require_full_window else (w[0] < lam < w[-1])
        if not ok:
            continue
        lo, hi = max(lo, w[0]), min(hi, w[-1])
        out[f"F_{key}"] = integrate(w, f - c, lo, hi)
        if key == "Ha":
            with np.errstate(divide="ignore", invalid="ignore"):
                out["EW_Ha"] = integrate(w, 1.0 - f / c, lo, hi)
    if np.isfinite(out["F_Ha"]) and np.isfinite(out["F_Hb"]) and out["F_Hb"] != 0:
        out["R"] = out["F_Ha"] / out["F_Hb"]
    return out


# ---------------------------------------------------------------------------
# Robust local-continuum measurement
# ---------------------------------------------------------------------------

NOMINAL = dict(v_int=10000.0, v_in=12000.0, v_out=16000.0)

# Window sets (km/s).  "broad": the full line profile (default).  "narrow": the
# narrow/intermediate-width component of interacting SNe (IIn, Ia-CSM, SLSN-II),
# measured against a local continuum that may include the broad wings.
WINDOWS = {
    "broad": dict(nominal=NOMINAL, v_int=(8000.0, 12000.0), gap=(1000.0, 3000.0), width=(3000.0, 5000.0)),
    "narrow": dict(nominal=dict(v_int=3000.0, v_in=4000.0, v_out=7000.0),
                   v_int=(2500.0, 3500.0), gap=(500.0, 1500.0), width=(2000.0, 4000.0)),
}


# Narrow non-Balmer emission lines that fall inside the Hα/Hβ integration windows or their
# sidebands: the recurring narrow peaks of the SuperLite interaction models (Si II 6347/6371 and
# He I 6678 beside Hα; He II 4686, He I/Fe II 4922/4924 and 5016/5018, Si II 5041/5056 and
# Fe II 5169 around Hβ).  He II 6560/4859 coincide with Hα/Hβ and cannot be separated.
NARROW_CONTAMINANTS = {
    "Si II 6347": 6347.11, "Si II 6371": 6371.37, "He I 6678": 6678.15,
    "He II 4686": 4685.71, "He I 4922 / Fe II 4924": 4922.9, "He I 5016 / Fe II 5018": 5017.1,
    "Si II 5041": 5041.03, "Si II 5056": 5055.98, "Fe II 5169": 5169.03,
}


def mask_narrow_lines(w, f, lines=None, hw_kms=1000.0, anchor_kms=600.0):
    """Replace ±hw_kms around each listed line by a straight line between the median flux just
    outside the mask (anchor_kms wide on each side).  Masks closer than two anchor widths are
    merged (e.g. the Si II 6347/6371 doublet), so an anchor never sits on another masked line.
    Removes narrow emission/absorption at those wavelengths and leaves broad profiles
    essentially unchanged."""
    lines = NARROW_CONTAMINANTS if lines is None else lines
    w = np.asarray(w, float)
    out = np.array(f, float, copy=True)
    iv = sorted((v2w(-hw_kms, lam), v2w(hw_kms, lam)) for lam in lines.values())
    merged = []
    for lo, hi in iv:
        if merged and lo <= merged[-1][1] * (1.0 + 2.0 * anchor_kms / C_KMS):
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    for lo, hi in merged:
        m = (w >= lo) & (w <= hi)
        if not m.any():
            continue
        left = (w >= lo * (1.0 - anchor_kms / C_KMS)) & (w < lo)
        right = (w > hi) & (w <= hi * (1.0 + anchor_kms / C_KMS))
        if not left.any() or not right.any():
            continue
        xl, yl = np.mean(w[left]), np.median(out[left])
        xr, yr = np.mean(w[right]), np.median(out[right])
        out[m] = yl + (yr - yl) * (w[m] - xl) / (xr - xl)
    return out


def _noise(w, f, ef):
    """Per-pixel 1σ noise: the supplied error spectrum if usable, else a robust
    estimate from the residuals around a running median."""
    if ef is not None and np.all(np.isfinite(ef)) and np.median(ef) > 0:
        return np.abs(ef)
    from scipy.ndimage import median_filter
    resid = f - median_filter(f, size=7, mode="nearest")
    mad = 1.4826 * median_filter(np.abs(resid), size=51, mode="nearest")
    return np.maximum(mad, 1e-6 * np.median(np.abs(f)))


def _line_once(w, f, lam0, v_int, v_in, v_out, cont_stat=np.median):
    b_lo, b_hi = v2w(-v_out, lam0), v2w(-v_in, lam0)
    r_lo, r_hi = v2w(v_in, lam0), v2w(v_out, lam0)
    mb = (w >= b_lo) & (w <= b_hi)
    mr = (w >= r_lo) & (w <= r_hi)
    if mb.sum() < 3 or mr.sum() < 3:
        return None
    xb, yb = np.mean(w[mb]), cont_stat(f[mb])
    xr, yr = np.mean(w[mr]), cont_stat(f[mr])
    slope = (yr - yb) / (xr - xb)
    c = yb + slope * (w - xb)
    lo, hi = v2w(-v_int, lam0), v2w(v_int, lam0)
    win = (w >= lo) & (w <= hi)
    cmin = np.min(c[win]) if win.any() else np.nan
    d = f - c
    pos = np.clip(d, 0, None)
    vel = (w / lam0 - 1.0) * C_KMS
    norm_em = integrate(w, pos, lo, hi)
    res = dict(
        v_cen=integrate(w, vel * pos, lo, hi) / norm_em if norm_em > 0 else np.nan,
        F_net=integrate(w, d, lo, hi),
        F_em=integrate(w, np.clip(d, 0, None), lo, hi),
        F_abs=integrate(w, np.clip(-d, 0, None), lo, hi),
        cmin=cmin,
        cmed=float(np.median(c[win])) if win.any() else np.nan,
    )
    if cmin > 0:
        res["EW_net"] = integrate(w, 1.0 - f / c, lo, hi)
        res["EW_em"] = -integrate(w, np.clip(d, 0, None) / c, lo, hi)
        res["EW_abs"] = integrate(w, np.clip(-d, 0, None) / c, lo, hi)
    else:
        res["EW_net"] = res["EW_em"] = res["EW_abs"] = np.nan
    return res


def _smooth_on_velocity_grid(w, f, lam0, vlo, vhi, dv=100.0, sigma_kms=600.0):
    from scipy.ndimage import gaussian_filter1d
    v = np.arange(vlo, vhi + dv, dv)
    lam = v2w(v, lam0)
    fs = gaussian_filter1d(np.interp(lam, w, f), sigma_kms / dv)
    return v, lam, fs


def _trough_and_blue_anchor(v, fs, v_trough=(-20000.0, -2000.0), blue_reach=12000.0, min_depth=0.05):
    """Locate a P-Cygni absorption trough and the first local maximum blueward of it.

    v, fs: velocity grid (ascending) and smoothed flux.  The trough is the
    minimum of fs within v_trough; a minimum at either end of that range is not
    a trough (continuum slope or the rising emission wing).  Walking blueward
    from the trough, the anchor is the first local maximum: the running maximum
    at the point where fs has dropped >2% below it, searched at most blue_reach
    km/s away.  Returns (i_trough, i_anchor, depth, is_trough); i_anchor is None
    when no local maximum is found within reach, depth = 1 - fs[trough]/max.
    """
    in_tr = np.flatnonzero((v >= v_trough[0]) & (v <= v_trough[1]))
    it = int(in_tr[np.argmin(fs[in_tr])])
    if v[it] >= v_trough[1] - 1000.0 or v[it] <= v_trough[0] + 1000.0:
        return it, None, 0.0, False
    best, i, found = it, it, False
    while i > 0 and v[i - 1] >= v[it] - blue_reach:
        i -= 1
        if fs[i] > fs[best]:
            best = i
        elif fs[i] < 0.98 * fs[best]:
            found = True
            break
    depth = 1.0 - fs[it] / fs[best] if fs[best] > 0 else 0.0
    if depth < min_depth:
        return it, None, depth, False
    return it, (best if found else None), depth, True


def pseudo_continuum_decomposition(w, f, lam0, v_trough=(-20000.0, -2000.0), v_red=(11000.0, 16000.0),
                                   blue_reach=12000.0):
    """Standard SN pseudo-continuum: straight line from the first flux maximum
    blueward of the P-Cygni absorption trough to the median flux redward of the
    emission (v_red).  For a line without an absorption trough the blue anchor
    is the median flux in the nominal blue sideband (so this reduces to the
    sideband continuum).  Returns emission and absorption parts of the profile
    between the anchors (EW_em < 0, EW_abs > 0) and a/e = EW_abs/|EW_em|
    (Gutiérrez et al. 2014); None without coverage or when a trough exists but
    no local maximum bounds it within blue_reach.
    """
    vlo = v_trough[0] - blue_reach
    if not (covers(w, v2w(vlo, lam0), v2w(v_red[1], lam0))):
        return None
    v, lam, fs = _smooth_on_velocity_grid(w, f, lam0, vlo, v_trough[1])
    it, ib, depth, is_trough = _trough_and_blue_anchor(v, fs, v_trough, blue_reach)
    if is_trough and ib is None:
        return None
    if is_trough:
        lam_b, yb, vb = lam[ib], fs[ib], float(v[ib])
    else:
        mb = (w >= v2w(-NOMINAL["v_out"], lam0)) & (w <= v2w(-NOMINAL["v_in"], lam0))
        lam_b, yb = float(np.mean(w[mb])), float(np.median(f[mb]))
        vb = -0.5 * (NOMINAL["v_in"] + NOMINAL["v_out"])
    mr = (w >= v2w(v_red[0], lam0)) & (w <= v2w(v_red[1], lam0))
    lam_r, yr = float(np.mean(w[mr])), float(np.median(f[mr]))
    c = yb + (yr - yb) / (lam_r - lam_b) * (w - lam_b)
    lo, hi = lam_b, v2w(v_red[0], lam0)
    win = (w >= lo) & (w <= hi)
    if not win.any() or np.min(c[win]) <= 0:
        return None
    d = f - c
    em = -integrate(w, np.clip(d, 0, None) / c, lo, hi)
    ab = integrate(w, np.clip(-d, 0, None) / c, lo, hi)
    return dict(EW_em=em, EW_abs=ab, ae=ab / abs(em) if em < 0 else np.inf, v_blue_anchor=vb,
                v_trough=float(v[it]) if is_trough else np.nan, depth=float(depth), EW_net=em + ab,
                lam_b=float(lam_b), y_b=float(yb), lam_r=lam_r, y_r=yr)


def sideband_diagnostics(w, f, lam0, v_in=12000.0, v_out=16000.0):
    """Two warning signs that a fixed-sideband continuum is wrong for this spectrum:
    blue_sb_low  a real P-Cygni absorption trough (depth >5%) reaches into the
                 blue sideband (trough velocity beyond -(v_in - 1500) km/s), so
                 the continuum is anchored inside the absorption
    sb_peaky     a sideband contains an emission feature (max/median > 1.25),
                 e.g. [O I] 6300,6364 next to Hα in nebular stripped-envelope SNe
    """
    from scipy.ndimage import median_filter
    out = dict(blue_sb_low=False, sb_peaky=False)
    if not covers(w, v2w(-32000.0, lam0), v2w(v_out, lam0)):
        return out
    mb = (w >= v2w(-v_out, lam0)) & (w <= v2w(-v_in, lam0))
    mr = (w >= v2w(v_in, lam0)) & (w <= v2w(v_out, lam0))
    v, _, fs = _smooth_on_velocity_grid(w, f, lam0, -32000.0, -2000.0)
    it, ib, depth, is_trough = _trough_and_blue_anchor(v, fs)
    out["blue_sb_low"] = bool(is_trough and v[it] < -(v_in - 1500.0))
    for m in (mb, mr):
        med = np.median(f[m])
        if med > 0 and m.sum() >= 3:
            peak = np.max(median_filter(f[m], size=3, mode="nearest"))
            out["sb_peaky"] |= bool(peak > 1.25 * med)
    # a sideband sitting on top of an emission feature: its level exceeds both the
    # opposite sideband and the adjacent outer region by >40% (a sloped continuum
    # cannot do that)
    outer_b = (w >= v2w(-24000.0, lam0)) & (w <= v2w(-19000.0, lam0))
    outer_r = (w >= v2w(19000.0, lam0)) & (w <= v2w(24000.0, lam0))
    lev = {k: np.median(f[m]) if m.sum() >= 3 else np.nan
           for k, m in (("b", mb), ("r", mr), ("ob", outer_b), ("or", outer_r))}
    if np.isfinite(lev["b"]) and np.isfinite(lev["r"]):
        if np.isfinite(lev["ob"]) and lev["b"] > 1.4 * max(lev["r"], lev["ob"]):
            out["sb_peaky"] = True
        if np.isfinite(lev["or"]) and lev["r"] > 1.4 * max(lev["b"], lev["or"]):
            out["sb_peaky"] = True
    return out


def robust_measure(w, f, ef=None, n_mc=300, seed=0, sn_min=3.0, absfrac_max=0.3, windows="broad",
                   v_cen_max=3000.0):
    """Robust Hα/Hβ measurement with Monte-Carlo uncertainties and quality flags.

    Returns a flat dict.  Quantities are given as nominal value and 16/50/84
    percentiles of the MC distribution (keys ``<q>``, ``<q>_lo``, ``<q>_med``,
    ``<q>_hi``).  Flags:
        cov_Ha, cov_Hb     line + sidebands fully covered
        corrupt_Ha/_Hb     pixel values incompatible with a physical spectrum
                           (strongly negative flux) inside the analysis region
        cont_bad_Ha/_Hb    local continuum not positive under the line
        offcentre_Ha/_Hb   flux-weighted velocity of the emission excess more than
                           v_cen_max from the line centre (flux dominated by a blend)
        sig_Ha, sig_Hb     net line flux different from zero at > sn_min sigma
        em_Ha, em_Hb       net emission at > sn_min sigma
        pcyg_Ha, pcyg_Hb   absorption carries > absfrac_max of |line| flux
        blue_sb_low_*      P-Cygni trough reaches into the blue sideband
        sb_peaky_*         an emission feature sits in a sideband
        ew_valid           Hα measurable (coverage, no corrupt pixels, continuum > 0)
        ew_clean           ew_valid and a clean, significant, emission-dominated Hα
        ratio_valid        Balmer decrement meaningful: both lines clean, significant
                           net emission, absorption < absfrac_max, no blend
        flags              human-readable list of the reasons a line failed
    pc_* keys hold the pseudo-continuum decomposition (broad windows only).
    windows: "broad" (full profile, default) or "narrow" (see WINDOWS).
    """
    w = np.asarray(w, float)
    f = np.asarray(f, float)
    sig = _noise(w, f, ef)
    rng = np.random.default_rng(seed)
    win = WINDOWS[windows]
    nom = win["nominal"]
    out = {}
    lines = {"Ha": HALPHA, "Hb": HBETA}
    vmax = NOMINAL["v_out"] + 2500.0
    for key, lam0 in lines.items():
        lo, hi = v2w(-vmax, lam0), v2w(vmax, lam0)
        out[f"cov_{key}"] = covers(w, v2w(-NOMINAL["v_out"], lam0), v2w(NOMINAL["v_out"], lam0))
        reg = (w >= lo) & (w <= hi)
        corrupt = False
        if reg.sum() > 5:
            # Flux far below zero cannot be noise in any usable spectrum.  Per-pixel
            # errors are not used: corrupted pixels often carry corrupted errors too.
            scale = np.median(np.abs(f[reg]))
            corrupt = bool(np.any(f[reg] < -5.0 * scale))
        out[f"corrupt_{key}"] = corrupt

    # nominal measurement
    for key, lam0 in lines.items():
        res = _line_once(w, f, lam0, **nom) if out[f"cov_{key}"] else None
        for q in ("F_net", "F_em", "F_abs", "EW_net", "EW_em", "EW_abs", "cmin", "v_cen"):
            out[f"{q}_{key}"] = res[q] if res else np.nan
        out[f"offcentre_{key}"] = bool(res is not None and np.isfinite(res["v_cen"])
                                       and abs(res["v_cen"]) > v_cen_max)
        out[f"cont_bad_{key}"] = bool(res is None or not (res["cmin"] > 0))

    # Monte Carlo: flux noise + window/sideband jitter
    draws = {f"{q}_{k}": [] for q in ("F_net", "F_em", "F_abs", "EW_net") for k in lines}
    draws["R_net"] = []
    draws["R_em"] = []
    if out["cov_Ha"] or out["cov_Hb"]:
        for _ in range(n_mc):
            fp = f + rng.normal(0.0, 1.0, size=f.size) * sig
            v_int = rng.uniform(*win["v_int"])
            v_in = v_int + rng.uniform(*win["gap"])
            v_out = v_in + rng.uniform(*win["width"])
            r = {}
            for key, lam0 in lines.items():
                r[key] = _line_once(w, fp, lam0, v_int, v_in, v_out) if out[f"cov_{key}"] else None
                for q in ("F_net", "F_em", "F_abs", "EW_net"):
                    draws[f"{q}_{key}"].append(r[key][q] if r[key] else np.nan)
            if r["Ha"] and r["Hb"]:
                draws["R_net"].append(r["Ha"]["F_net"] / r["Hb"]["F_net"] if r["Hb"]["F_net"] != 0 else np.nan)
                draws["R_em"].append(r["Ha"]["F_em"] / r["Hb"]["F_em"] if r["Hb"]["F_em"] > 0 else np.nan)
    for q, vals in draws.items():
        a = np.asarray(vals, float)
        a = a[np.isfinite(a)]
        if a.size >= 0.8 * n_mc:
            out[f"{q}_lo"], out[f"{q}_med"], out[f"{q}_hi"] = np.percentile(a, [16, 50, 84])
            if q.startswith("F_net"):
                out[f"P_pos_{q[-2:]}"] = float(np.mean(a > 0))
        else:
            out[f"{q}_lo"] = out[f"{q}_med"] = out[f"{q}_hi"] = np.nan

    for key in lines:
        lo, hi = out.get(f"F_net_{key}_lo", np.nan), out.get(f"F_net_{key}_hi", np.nan)
        err = 0.5 * (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
        fnet = out[f"F_net_{key}"]
        out[f"sig_{key}"] = bool(np.isfinite(err) and err > 0 and abs(fnet) > sn_min * err)
        out[f"em_{key}"] = bool(out[f"sig_{key}"] and fnet > 0)
        tot = out[f"F_em_{key}"] + out[f"F_abs_{key}"]
        out[f"absfrac_{key}"] = out[f"F_abs_{key}"] / tot if tot and np.isfinite(tot) and tot > 0 else np.nan
        out[f"pcyg_{key}"] = bool(np.isfinite(out[f"absfrac_{key}"]) and out[f"absfrac_{key}"] > absfrac_max)

    for key, lam0 in lines.items():
        diag = sideband_diagnostics(w, f, lam0, v_in=nom["v_in"], v_out=nom["v_out"])
        out[f"blue_sb_low_{key}"] = diag["blue_sb_low"]
        out[f"sb_peaky_{key}"] = diag["sb_peaky"]
        pc = (pseudo_continuum_decomposition(w, f, lam0)
              if windows == "broad" and not out[f"corrupt_{key}"] else None)
        for q in ("EW_em", "EW_abs", "ae", "v_blue_anchor", "v_trough", "depth", "EW_net"):
            out[f"pc_{q}_{key}"] = pc[q] if pc else np.nan

    fa, fb = out["F_net_Ha"], out["F_net_Hb"]
    out["R_net"] = fa / fb if np.isfinite(fa) and np.isfinite(fb) and fb != 0 else np.nan
    fa, fb = out["F_em_Ha"], out["F_em_Hb"]
    out["R_em"] = fa / fb if np.isfinite(fa) and np.isfinite(fb) and fb > 0 else np.nan

    def measurable(k):
        return out[f"cov_{k}"] and not out[f"corrupt_{k}"] and not out[f"cont_bad_{k}"]

    def clean(k):
        return (measurable(k) and not out[f"blue_sb_low_{k}"] and not out[f"sb_peaky_{k}"]
                and not out[f"offcentre_{k}"])

    out["ew_valid"] = bool(measurable("Ha"))
    # EW is a clean emission-line measurement: significant net emission, little absorption,
    # and a sideband continuum that is not obviously compromised.
    out["ew_clean"] = bool(clean("Ha") and out["em_Ha"] and not out["pcyg_Ha"])
    # Balmer decrement only for two clean, significant, emission-dominated lines.
    out["ratio_valid"] = bool(clean("Ha") and clean("Hb") and out["em_Ha"] and out["em_Hb"]
                              and not out["pcyg_Ha"] and not out["pcyg_Hb"])
    reasons = []
    for k in lines:
        if not out[f"cov_{k}"]:
            reasons.append(f"{k}:no-coverage")
            continue
        if out[f"corrupt_{k}"]:
            reasons.append(f"{k}:corrupt-pixels")
        if out[f"cont_bad_{k}"]:
            reasons.append(f"{k}:continuum<=0")
        if out[f"blue_sb_low_{k}"]:
            reasons.append(f"{k}:blue-sideband-in-absorption")
        if out[f"sb_peaky_{k}"]:
            reasons.append(f"{k}:emission-in-sideband")
        if out[f"offcentre_{k}"]:
            reasons.append(f"{k}:emission-centroid>3000km/s(blend)")
        if out[f"pcyg_{k}"]:
            reasons.append(f"{k}:P-Cygni-absorption>30%")
        if not out[f"sig_{k}"]:
            reasons.append(f"{k}:net-flux<3sigma")
        elif not out[f"em_{k}"]:
            reasons.append(f"{k}:net-absorption")
    out["flags"] = ";".join(reasons)
    return out
