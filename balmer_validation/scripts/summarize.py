#!/usr/bin/env python
"""Collect the numbers quoted in REPORT.md into results/summary.json.

Usage: python -I scripts/summarize.py --results results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CLASSES = ["SNIIn", "SNII", "SNIIb", "SNIa-CSM", "SLSN-II"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    R = lambda f: pd.read_csv(os.path.join(a.results, f))  # noqa: E731
    out = {}
    m = R("templates_measurements.csv")
    m = m[(m.phase >= -20) & (m.phase <= 150)]
    out["templates"] = {}
    for c in CLASSES:
        s = m[m.figure == c]
        v = s[s.rob_ratio_valid.astype(bool)].rob_R_net
        out["templates"][c] = dict(
            n=len(s), n_sne=int(s.name.nunique()), ha_cov=int(s.rob_cov_Ha.sum()), ew_clean=int(s.rob_ew_clean.sum()),
            both_cov=int((s.rob_cov_Ha.astype(bool) & s.rob_cov_Hb.astype(bool)).sum()), r_valid=int(len(v)),
            r_min=float(v.min()) if len(v) else None, r_med=float(v.median()) if len(v) else None,
            r_max=float(v.max()) if len(v) else None)
    c = R("student_vs_thiswork.csv")
    d = R("student_digitized.csv")
    out["student"] = dict(digitized=len(d), matched=int(d.key.notna().sum()), per_figure={})
    for (fig, qty), s in c.groupby(["figure", "quantity"], sort=False):
        r = s[s.reliable.astype(bool)]
        out["student"]["per_figure"][fig] = dict(
            n=len(s), reliable=len(r), negative=int((s.d_value < 0).sum()), positive=int((s.d_value > 0).sum()),
            abs_gt_30=int((s.d_value.abs() > 30).sum()), abs_gt_100=int((s.d_value.abs() > 100).sum()),
            med_absdlog_reliable=float(np.nanmedian(np.abs(r.dlog))) if len(r) else None,
            frac_within_0p1dex=float(np.mean(np.abs(r.dlog) < 0.1)) if len(r) else None,
            signflips_reliable=int(r.sign_flip.sum()))
    for qty in ("EW", "R"):
        s = c[c.quantity == qty]
        out["student"][qty] = dict(n=len(s), negative=int((s.d_value < 0).sum()), positive=int((s.d_value > 0).sum()),
                                   reliable=int(s.reliable.sum()), abs_gt_30=int((s.d_value.abs() > 30).sum()),
                                   abs_gt_100=int((s.d_value.abs() > 100).sum()))
    am = R("archive_measurements_phased.csv")
    out["archive"] = dict(n_spectra=len(am), n_sne=int(am.name.nunique()), usable=int(am.usable.sum()),
                          sources=am.drop_duplicates("name").source.value_counts().to_dict(), per_class={})
    w = am[am.usable.astype(bool) & (am.phase >= -20) & (am.phase <= 150)]
    for cl in CLASSES:
        s = w[w.figure == cl]
        v = s[s.rob_ratio_valid.astype(bool)].rob_R_net
        out["archive"]["per_class"][cl] = dict(
            n=len(s), n_sne=int(s.name.nunique()), ew_valid=int(s.rob_ew_valid.sum()), ew_clean=int(s.rob_ew_clean.sum()),
            both_cov=int((s.rob_cov_Ha.astype(bool) & s.rob_cov_Hb.astype(bool)).sum()), r_valid=int(len(v)),
            r_med=float(v.median()) if len(v) else None, r_min=float(v.min()) if len(v) else None,
            r_max=float(v.max()) if len(v) else None)
    p = R("template_archive_pairs.csv")
    ew = p[p.rob_ew_clean.astype(bool) & p.a_rob_ew_clean.astype(bool)]
    rr = p[p.rob_ratio_valid.astype(bool) & p.a_rob_ratio_valid.astype(bool)]
    dl = np.log10(rr.rob_R_net / rr.a_rob_R_net)
    cal = p.hb_ha_calib_ratio.dropna()
    out["template_vs_archive"] = dict(
        matched=len(p), ew_pairs=len(ew), ew_med_absdlog=float(np.nanmedian(np.abs(np.log10(ew.rob_EW_net_Ha / ew.a_rob_EW_net_Ha)))),
        r_pairs=len(rr), r_med_dlog=float(np.nanmedian(dl)), r_std_dlog=float(np.nanstd(dl)),
        calib_n=len(cal), calib_frac_gt_0p1dex=float(np.mean(np.abs(np.log10(cal)) > 0.1)))
    with open(os.path.join(a.results, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
