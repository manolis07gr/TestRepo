#!/usr/bin/env python
"""SuperLite spectra and SNLT line profiles against the observed SNe IIn, on one footing.

The SNLT profile files cover only about ±5000 km/s around each line and give the flux in units of
the code's own continuum, so everything here is restricted to that window and measured the same
way for the three data sets:

  EW5    -∫ (F/Fc - 1) dλ over |v| < 5000 km/s (negative = emission; |EW5| is plotted)
  R5     F5(Hα) / F5(Hβ), with F5 = ∫ (F - Fc) dλ over |v| < 5000 km/s
  absfr  share of the absorption in the line flux within the window, F_abs / (F_em + F_abs), on the
         profile smoothed to 300 km/s (observed SNe IIn: Hα essentially never above 0.3)
  depth  1 - min(F/Fc) of the smoothed profile within the window
  vpeak  velocity of the maximum of the smoothed profile

Continuum Fc: for the observed and the SuperLite spectra, the straight line through the median flux
in the 12,000-16,000 km/s sidebands (the continuum of the robust method); for SNLT, the code's own
emergent continuum (F_norm_corrected = F/Fc), which follows the SuperLite flux level away from the
line.  The narrow non-Balmer lines are masked in all three data sets first (bv.measure).

Observed SNe IIn: the spectra with a clean Hα emission line (robm_ew_clean) and, for R5, a valid
Balmer decrement (robm_ratio_valid), merged per object and night as in results_data.py.

Usage:
    python -I scripts/compare_line_codes.py --archive data/archive \
        --ameas results/archive_measurements_phased.csv --superlite <dir with A_Series/B_Series .dat> \
        --snlt <dir with SNLT_Balmer/<model>/*.profile.txt> --catalogue <xlsx> --out results/model_codecomp.csv
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from bv.measure import C_KMS, HALPHA, HBETA, covers, integrate, mask_narrow_lines, v2w  # noqa: E402
from bv.models import find_spectra, load_spectrum, read_catalogue, resolve_epochs  # noqa: E402
from scripts_common import rest_frame  # noqa: E402

VW = 5000.0                  # half-width of the common window, km/s
SB = (12000.0, 16000.0)      # sidebands of the robust method, km/s
SMOOTH = 300.0               # km/s, Gaussian sigma for the shape metrics
LINES = {"Ha": HALPHA, "Hb": HBETA}


def sideband_continuum(w, f, lam0):
    mb = (w >= v2w(-SB[1], lam0)) & (w <= v2w(-SB[0], lam0))
    mr = (w >= v2w(SB[0], lam0)) & (w <= v2w(SB[1], lam0))
    if mb.sum() < 3 or mr.sum() < 3:
        return None
    xb, yb, xr, yr = np.mean(w[mb]), np.median(f[mb]), np.mean(w[mr]), np.median(f[mr])
    return yb + (yr - yb) / (xr - xb) * (w - xb)


def window_metrics(lam, ratio, cont, lam0):
    """Metrics from F/Fc (ratio) and Fc (cont) sampled on lam."""
    v = (lam / lam0 - 1.0) * C_KMS
    m = np.abs(v) <= VW
    if m.sum() < 5 or v[m].min() > -0.9 * VW or v[m].max() < 0.9 * VW:
        return None
    lo, hi = v2w(-VW, lam0), v2w(VW, lam0)
    ew = integrate(lam, 1.0 - ratio, lo, hi)
    flux = integrate(lam, (ratio - 1.0) * cont, lo, hi)
    # shape metrics on a uniform velocity grid, smoothed
    vg = np.arange(-VW, VW + 1.0, 50.0)
    rg = gaussian_filter1d(np.interp(vg, v, ratio), SMOOTH / 50.0)
    em = np.trapezoid(np.clip(rg - 1.0, 0, None), vg)
    ab = np.trapezoid(np.clip(1.0 - rg, 0, None), vg)
    return dict(EW5=ew, F5=flux, absfr=ab / (em + ab) if em + ab > 0 else np.nan,
                depth=1.0 - rg.min(), vpeak=float(vg[np.argmax(rg)]))


def measure_spectrum(w, f):
    """Observed or SuperLite spectrum: sideband continuum, narrow lines masked."""
    fm = mask_narrow_lines(w, f)
    out = {}
    for k, lam0 in LINES.items():
        if not covers(w, v2w(-SB[1], lam0), v2w(SB[1], lam0)):
            continue
        c = sideband_continuum(w, fm, lam0)
        if c is None or np.any(c[(w > v2w(-VW, lam0)) & (w < v2w(VW, lam0))] <= 0):
            continue
        r = window_metrics(w, fm / c, c, lam0)
        if r:
            out.update({f"{q}_{k}": val for q, val in r.items()})
    return out


def read_snlt(path):
    head = open(path).readline()
    num = r"([-+]?(?:[0-9.]+(?:[eE][-+]?\d+)?|nan|inf))"
    meta = {k: float(re.search(k + "=" + num, head).group(1)) for k in ("epoch", "L_corr", "EW_corr")}
    a = np.loadtxt(path, comments="#")
    return a, meta


def observed(archive, ameas):
    am = pd.read_csv(ameas, low_memory=False)
    am = am[(am.figure == "SNIIn") & am.usable.astype(bool) & (am.phase >= -20) & (am.phase <= 150)
            & am.robm_ew_clean.astype(bool)]
    rows = []
    for name, g in am.groupby("name"):
        d = json.load(open(os.path.join(archive, f"{name}.json")))
        by_idx = {sp["idx"]: sp for sp in d["spectra"]}
        for _, r in g.iterrows():
            sp = by_idx.get(r.idx)
            if sp is None:
                continue
            zs = sp.get("z_spec")
            try:
                z = float(zs) if zs not in (None, "") else d["meta"].get("z")
            except (TypeError, ValueError):
                z = d["meta"].get("z")
            w, f = rest_frame(sp, z)
            res = measure_spectrum(w, f)
            res.update(name=name, mjd=r.mjd, phase=r.phase, ratio_ok=bool(r.robm_ratio_valid))
            rows.append(res)
    o = pd.DataFrame(rows)
    o["night"] = np.round(o.mjd).astype(int)
    # one value per object and night (median), as in results_data.py
    agg = {c: "median" for c in o.columns if c not in ("name", "night", "mjd", "ratio_ok")}
    n = o.groupby(["name", "night"]).agg(agg).reset_index()
    rr = o[o.ratio_ok & (o.F5_Ha > 0) & (o.F5_Hb > 0)].assign(R5=lambda x: x.F5_Ha / x.F5_Hb)
    n = n.merge(rr.groupby(["name", "night"]).R5.median().reset_index(), on=["name", "night"], how="left")
    return n


def superlite(root, cat):
    files = resolve_epochs([f for f in find_spectra(root) if f["model"]], cat)
    rows = []
    for f in files:
        w, fl, _ = load_spectrum(f["path"])
        res = measure_spectrum(w, fl)
        res.update(model=f["model"], epoch=f["epoch"], code="SuperLite")
        rows.append(res)
    return pd.DataFrame(rows)


def snlt(root):
    rows = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "prod_day*_H*.profile.txt"))):
        model = os.path.basename(os.path.dirname(p))
        line = "Ha" if "_Halpha." in p else "Hb"
        a, meta = read_snlt(p)
        dv, lam, fnc, fn, fc, fl = a.T
        lam0 = LINES[line]
        fm = mask_narrow_lines(lam, fnc * fc) / fc          # same masking, applied to the flux
        r = window_metrics(lam, fm, fc, lam0)
        key = (model, meta["epoch"])
        rows.setdefault(key, dict(model=model, epoch=meta["epoch"], code="SNLT"))
        rows[key][f"L_corr_{line}"] = meta["L_corr"]
        if r:
            rows[key].update({f"{q}_{line}": val for q, val in r.items()})
    return pd.DataFrame(list(rows.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--ameas", required=True)
    ap.add_argument("--superlite", required=True)
    ap.add_argument("--snlt", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cat = read_catalogue(a.catalogue)
    obs = observed(a.archive, a.ameas)
    mod = pd.concat([superlite(a.superlite, cat), snlt(a.snlt)], ignore_index=True)
    mod["R5"] = np.where((mod.F5_Ha > 0) & (mod.F5_Hb > 0), mod.F5_Ha / mod.F5_Hb, np.nan)
    mod = mod.merge(cat[["model", "series", "e51", "m_csm"]], on="model", how="left")
    mod["series"] = mod.series.fillna(mod.model.str[0])
    mod.to_csv(a.out, index=False)
    obs.to_csv(os.path.splitext(a.out)[0] + "_obs_IIn.csv", index=False)
    print(f"wrote {a.out}: {len(mod)} model spectra/profiles ({mod.groupby('code').size().to_dict()}); "
          f"observed SNe IIn: {len(obs)} object-nights of {obs.name.nunique()} objects")


if __name__ == "__main__":
    main()
