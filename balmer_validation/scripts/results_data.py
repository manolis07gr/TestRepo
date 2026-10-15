#!/usr/bin/env python
"""Per-class Hα EW and Balmer-decrement trends from the observed (public) spectra.

Only reliable measurements enter: clean emission-line EWs (`rob_ew_clean`) and valid
Balmer decrements (`rob_ratio_valid`).  Spectra of the same object taken on the same
night (duplicates across archives, several instruments) are merged into one
measurement (median), so heavily observed objects do not dominate.  Population
statistics treat the object, not the spectrum, as the independent unit: binned
trends use per-object medians, and confidence intervals come from resampling
objects (bootstrap).

Usage:
    python -I scripts/results_data.py --ameas results/archive_measurements_phased.csv \
        --out results/observed_results.json [--prefix robm_]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CLASSES = ["SNIIn", "SLSN-II", "SNIa-CSM", "SNII", "SNIIb"]
INTERACTING = {"SNIIn", "SLSN-II", "SNIa-CSM"}
SLSN_IIN = {"2006gy", "PTF10fel", "PTF10jwd", "PTF10qaf", "PTF10scc", "PTF10tpz"}
BINS = [-20, 0, 20, 40, 70, 100, 150]
CASE_B = 2.86
RNG = np.random.default_rng(42)
P = "rob_"          # measurement columns: rob_ (as measured) or robm_ (narrow non-Balmer lines masked)


def nightly(a):
    """One row per (object, night): medians of the reliable measurements."""
    a = a.copy()
    a["night"] = np.round(a.mjd).astype(int)
    rows = []
    for (name, night), g in a.groupby(["name", "night"]):
        e = g[g[P + "ew_clean"].astype(bool)]
        r = g[g[P + "ratio_valid"].astype(bool)]
        if e.empty and r.empty:
            continue
        row = dict(name=name, figure=g.figure.iloc[0], phase=float(g.phase.median()),
                   phase_ref=g.phase_ref.iloc[0], source=g.source.iloc[0],
                   reference=str(g.reference.dropna().iloc[0]) if g.reference.notna().any() else "",
                   n_spec=len(g))
        if not e.empty:
            row.update(ew=float(-e[P + "EW_net_Ha"].median()), ew_lo=float(-e[P + "EW_net_Ha_hi"].median()),
                       ew_hi=float(-e[P + "EW_net_Ha_lo"].median()))
        if not r.empty:
            row.update(r=float(r[P + "R_net"].median()), r_lo=float(r[P + "R_net_lo"].median()),
                       r_hi=float(r[P + "R_net_hi"].median()))
            if P + "R_net_mwcorr" in r and r[P + "R_net_mwcorr"].notna().any():
                row["r_mw"] = float(r[P + "R_net_mwcorr"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def boot_spearman(d, xcol, ycol, n=2000):
    """Spearman rho of the pooled points, 95% CI from resampling objects."""
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    if len(d) < 6 or d.name.nunique() < 3:
        return None
    rho = float(spearmanr(d[xcol], d[ycol]).statistic)
    names = d.name.unique()
    groups = {k: g for k, g in d.groupby("name")}
    bs = []
    for _ in range(n):
        pick = RNG.choice(names, size=len(names), replace=True)
        s = pd.concat([groups[k] for k in pick])
        if s[xcol].nunique() > 2 and s[ycol].nunique() > 2:
            bs.append(spearmanr(s[xcol], s[ycol]).statistic)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rho=rho, lo=float(lo), hi=float(hi), n_points=len(d), n_objects=int(d.name.nunique()))


def per_object_slopes(d, ycol, min_n=3, min_span=20.0):
    """OLS slope of log10(y) vs phase per object, in dex per 100 d."""
    out = []
    for name, g in d[np.isfinite(d[ycol])].groupby("name"):
        if len(g) < min_n or np.ptp(g.phase) < min_span:
            continue
        b = np.polyfit(g.phase, np.log10(g[ycol]), 1)[0] * 100.0
        out.append(dict(name=name, slope=float(b), n=len(g), span=float(np.ptp(g.phase)),
                        first=float(g.sort_values("phase")[ycol].iloc[0]),
                        last=float(g.sort_values("phase")[ycol].iloc[-1])))
    return out


def binned(d, ycol, min_obj=3):
    """Per phase bin: median and IQR over objects of each object's median log10(y)."""
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = d[(d.phase >= lo) & (d.phase < hi) & np.isfinite(d[ycol])]
        per = s.groupby("name")[ycol].median()
        if len(per) < min_obj:
            continue
        lv = np.log10(per.values)
        out.append(dict(lo=lo, hi=hi, mid=0.5 * (lo + hi), n_obj=int(len(per)), n_pts=int(len(s)),
                        med=float(10 ** np.median(lv)), q1=float(10 ** np.percentile(lv, 25)),
                        q3=float(10 ** np.percentile(lv, 75))))
    return out


def window_median(d, ycol, lo, hi):
    s = d[(d.phase >= lo) & (d.phase < hi) & np.isfinite(d[ycol])]
    per = s.groupby("name")[ycol].median()
    if per.empty:
        return None
    return dict(med=float(10 ** np.median(np.log10(per))), n_obj=int(len(per)),
                q1=float(10 ** np.percentile(np.log10(per), 25)), q3=float(10 ** np.percentile(np.log10(per), 75)))


def class_stats(d, cls):
    e = d[np.isfinite(d.get("ew", pd.Series(dtype=float)))] if "ew" in d else d.iloc[0:0]
    r = d[np.isfinite(d.get("r", pd.Series(dtype=float)))] if "r" in d else d.iloc[0:0]
    st = dict(cls=cls, interacting=cls in INTERACTING)
    st["ew"] = dict(n_nights=len(e), n_objects=int(e.name.nunique()), binned=binned(e, "ew"),
                    spearman_phase=boot_spearman(e.assign(lew=np.log10(e.ew)), "phase", "lew"),
                    slopes=per_object_slopes(e, "ew"),
                    early=window_median(e, "ew", -20, 20), late=window_median(e, "ew", 60, 150))
    sl = [s["slope"] for s in st["ew"]["slopes"]]
    st["ew"]["slope_median"] = float(np.median(sl)) if sl else None
    st["ew"]["slope_npos"] = int(np.sum(np.array(sl) > 0)) if sl else 0
    st["ew"]["slope_n"] = len(sl)
    st["r"] = dict(n_nights=len(r), n_objects=int(r.name.nunique()))
    if len(r):
        lv = np.log10(r.r)
        per_obj = r.groupby("name").r.median()
        st["r"].update(
            median=float(10 ** np.median(lv)), q1=float(10 ** np.percentile(lv, 25)), q3=float(10 ** np.percentile(lv, 75)),
            min=float(r.r.min()), max=float(r.r.max()),
            frac_above_caseB=float(np.mean(r.r > CASE_B)), frac_below_2=float(np.mean(r.r < 2.0)),
            obj_median=float(10 ** np.median(np.log10(per_obj))),
            spearman_phase=boot_spearman(r.assign(lr=np.log10(r.r)), "phase", "lr"),
            slopes=per_object_slopes(r, "r"),
            early=window_median(r, "r", -20, 20), late=window_median(r, "r", 40, 150),
            binned=binned(r, "r", min_obj=3))
        both = r[np.isfinite(r.get("ew", np.nan))] if "ew" in r else r.iloc[0:0]
        if len(both) >= 6:
            st["r"]["spearman_ew"] = boot_spearman(both.assign(lr=np.log10(both.r), lew=np.log10(both.ew)), "lew", "lr")
        if "r_mw" in r and r.r_mw.notna().any():
            st["r"]["median_mwcorr"] = float(10 ** np.median(np.log10(r.r_mw.dropna())))
    return st


def ia_csm_limits(am):
    """Lower limits on Hα/Hβ for SNe Ia-CSM, relative to the local continuum.

    Hβ is never a significant emission line in these spectra, so the 16th percentile
    of the Hα net flux is divided by the 84th percentile of the Hβ emission-only
    flux (the positive excess above the sideband continuum).  The underlying SN Ia
    spectrum makes the Hβ continuum model-dependent, so these are limits for this
    continuum definition only.
    """
    ia = am[(am.figure == "SNIa-CSM") & am[P + "ew_clean"].astype(bool) & am[P + "cov_Hb"].astype(bool)].copy()
    ia["r_ll"] = ia[P + "F_net_Ha_lo"] / ia[P + "F_em_Hb_hi"]
    ia["night"] = np.round(ia.mjd).astype(int)
    g = ia.groupby(["name", "night"]).agg(phase=("phase", "median"), r_ll=("r_ll", "median")).reset_index()
    g = g[np.isfinite(g.r_ll) & (g.r_ll > 0)]
    return dict(points=json.loads(g[["name", "phase", "r_ll"]].round(4).to_json(orient="records")),
                n_nights=len(g), n_objects=int(g.name.nunique()), median=float(g.r_ll.median()),
                frac_gt_caseB=float(np.mean(g.r_ll > CASE_B)), frac_gt_5=float(np.mean(g.r_ll > 5)),
                frac_gt_10=float(np.mean(g.r_ll > 10)))


def partial_iin(d):
    """IIn: Spearman of log EW vs log R, partialled on phase, with object bootstrap."""
    from scipy.stats import rankdata
    b = d[np.isfinite(d.get("r", np.nan)) & np.isfinite(d.get("ew", np.nan))]
    if len(b) < 8:
        return None

    def pr(df):
        rx, ry, rz = rankdata(df.ew), rankdata(df.r), rankdata(df.phase)
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return float(np.corrcoef(ex, ey)[0, 1])
    names = b.name.unique()
    groups = {k: g for k, g in b.groupby("name")}
    bs = [pr(pd.concat([groups[k] for k in RNG.choice(names, len(names))])) for _ in range(2000)]
    per = b.groupby("name").agg(ew=("ew", "median"), r=("r", "median"))
    between = spearmanr(per.ew, per.r)
    return dict(partial=pr(b), lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)),
                between_rho=float(between.statistic), between_p=float(between.pvalue), n_objects=len(per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ameas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="rob_", choices=("rob_", "robm_"),
                    help="rob_: as measured (default); robm_: narrow non-Balmer lines masked")
    a = ap.parse_args()
    global P
    P = a.prefix
    am = pd.read_csv(a.ameas)
    am = am[am.usable.astype(bool) & (am.phase >= -20) & (am.phase <= 150)]
    d = nightly(am)
    d["slsn_iin"] = d.name.isin(SLSN_IIN)
    out = dict(classes={}, bins=BINS, case_b=CASE_B)
    for cls in CLASSES:
        out["classes"][cls] = class_stats(d[d.figure == cls], cls)
    if (d.figure == "SLSN-II").any():
        s = d[(d.figure == "SLSN-II")]
        out["slsn_split"] = {k: class_stats(s[s.slsn_iin == flag], k)
                             for k, flag in (("SLSN-IIn", True), ("SLSN-II (no narrow lines)", False))}
    cols = ["name", "figure", "phase", "phase_ref", "source", "reference", "n_spec", "slsn_iin",
            "ew", "ew_lo", "ew_hi", "r", "r_lo", "r_hi"]
    pts = d[[c for c in cols if c in d]].round(4)
    out["points"] = json.loads(pts.to_json(orient="records"))
    out["ia_csm_lower_limits"] = ia_csm_limits(am)
    out["partial_iin"] = partial_iin(d[d.figure == "SNIIn"])
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    pts.to_csv(os.path.splitext(a.out)[0] + "_points.csv", index=False)
    print(f"wrote {a.out}: {len(pts)} object-nights")


if __name__ == "__main__":
    main()
