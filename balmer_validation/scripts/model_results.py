#!/usr/bin/env python
"""SuperLite models against the observed classes: numbers for the report and the Models tab.

Inputs
  --mmeas      results/model_measurements.csv (measure_models.py; rob_ = as for the observed
               spectra, robm_ = narrow non-Balmer lines masked, rep_ = the paper's specutils recipe)
  --obs        results/observed_results.json        (results_data.py, prefix rob_)
  --obs-masked results/observed_results_masked.json (results_data.py, prefix robm_)
  --published  results/published_model_lines.csv    (paper Tables A1-A2)
Output
  --out        results/model_results.json, and <out>_points.csv (one row per model spectrum)

Phase: model epochs are days since the bolometric maximum; observed phases are days since the
optical maximum (or discovery).  A model epoch is compared with the observed objects of the same
phase bin (BINS of results_data.py): offset = log10(model / median of the objects' medians), and
whether the model lies inside the interquartile range / the full range of the objects.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BINS = [-20, 0, 20, 40, 70, 100, 150]
CLASSES = ["SNIIn", "SLSN-II", "SNIa-CSM", "SNII", "SNIIb"]
VARIANTS = {"strict": "rob_", "masked": "robm_"}

# first failing criterion, in order of severity
REASONS_LINE = [
    ("no-coverage", "no coverage"), ("corrupt-pixels", "corrupt pixels"), ("continuum<=0", "continuum ≤ 0"),
    ("net-absorption", "net absorption"), ("P-Cygni-absorption>30%", "absorption > 30 % of the line"),
    ("emission-centroid>3000km/s(blend)", "blend: emission centroid > 3000 km/s off"),
    ("emission-in-sideband", "emission line in a sideband"),
    ("blue-sideband-in-absorption", "absorption trough in the blue sideband"),
    ("net-flux<3sigma", "net flux < 3σ"),
]


def first_reason(flags, line):
    toks = [t.split(":", 1)[1] for t in str(flags or "").split(";") if t.startswith(line + ":")]
    for key, text in REASONS_LINE:
        if key in toks:
            return text
    return None


def model_points(m):
    rows = []
    for _, r in m.iterrows():
        if isinstance(r.get("load_error"), str):
            continue
        row = dict(m=r.model, s=r.series, E=float(r.e51), M=float(r.m_csm if np.isfinite(r.m_csm) else 0.0),
                   t=float(r.epoch), comp=r.csm_comp if isinstance(r.csm_comp, str) else None)
        for v, p in VARIANTS.items():
            sfx = "" if v == "strict" else "m"
            ew = -r[p + "EW_net_Ha"]
            row["e" + sfx] = float(ew) if np.isfinite(ew) else None
            row["el" + sfx] = float(-r[p + "EW_net_Ha_hi"]) if np.isfinite(r[p + "EW_net_Ha_hi"]) else None
            row["eh" + sfx] = float(-r[p + "EW_net_Ha_lo"]) if np.isfinite(r[p + "EW_net_Ha_lo"]) else None
            row["eok" + sfx] = bool(r[p + "ew_clean"])
            row["ewhy" + sfx] = None if row["eok" + sfx] else first_reason(r[p + "flags"], "Ha")
            rr = r[p + "R_net"]
            row["r" + sfx] = float(rr) if np.isfinite(rr) else None
            row["rl" + sfx] = float(r[p + "R_net_lo"]) if np.isfinite(r[p + "R_net_lo"]) else None
            row["rh" + sfx] = float(r[p + "R_net_hi"]) if np.isfinite(r[p + "R_net_hi"]) else None
            row["rok" + sfx] = bool(r[p + "ratio_valid"])
            if not row["rok" + sfx]:
                row["rwhy" + sfx] = (first_reason(r[p + "flags"], "Hb") and "Hβ: " + first_reason(r[p + "flags"], "Hb")) \
                    or (first_reason(r[p + "flags"], "Ha") and "Hα: " + first_reason(r[p + "flags"], "Ha"))
            else:
                row["rwhy" + sfx] = None
        # share of the window emission that the narrow non-Balmer lines carry (strict vs masked)
        fe, fem = r["rob_F_em_Ha"], r["robm_F_em_Ha"]
        row["metal_frac_Ha"] = float(1 - fem / fe) if np.isfinite(fe) and fe > 0 and np.isfinite(fem) else None
        fe, fem = r["rob_F_em_Hb"], r["robm_F_em_Hb"]
        row["metal_frac_Hb"] = float(1 - fem / fe) if np.isfinite(fe) and fe > 0 and np.isfinite(fem) else None
        rows.append(row)
    return pd.DataFrame(rows)


def class_reference(points_csv):
    """Per class, quantity and phase bin: objects' medians (for IQR / range)."""
    pts = pd.read_csv(points_csv)
    ref = {}
    for cls in CLASSES:
        s = pts[pts.figure == cls]
        for q in ("ew", "r"):
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                b = s[(s.phase >= lo) & (s.phase < hi) & np.isfinite(s[q]) & (s[q] > 0)]
                per = b.groupby("name")[q].median()
                if len(per) < 3:
                    continue
                lv = np.log10(per.values)
                ref[(cls, q, lo)] = dict(lo=lo, hi=hi, n=int(len(per)), med=float(10 ** np.median(lv)),
                                         q1=float(10 ** np.percentile(lv, 25)), q3=float(10 ** np.percentile(lv, 75)),
                                         vmin=float(per.min()), vmax=float(per.max()))
    return ref


def bin_of(t):
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        if lo <= t < hi:
            return lo
    return None


def compare(pts, ref, sfx):
    """Each valid model value against the observed objects of each class at the same phase."""
    out = []
    for _, p in pts.iterrows():
        lo = bin_of(p.t)
        if lo is None:
            continue
        for q, key in (("ew", "e"), ("r", "r")):
            v, ok = p[key + sfx], p[key + "ok" + sfx]
            if not ok or v is None or not np.isfinite(v) or v <= 0:
                continue
            for cls in CLASSES:
                b = ref.get((cls, q, lo))
                if not b:
                    continue
                out.append(dict(m=p.m, s=p.s, M=p.M, E=p.E, t=p.t, q=q, cls=cls, v=v, dex=float(np.log10(v / b["med"])),
                                in_iqr=bool(b["q1"] <= v <= b["q3"]), in_range=bool(b["vmin"] <= v <= b["vmax"])))
    return pd.DataFrame(out)


def summarize_compare(c):
    """Per class, quantity and CSM-mass group (H-rich A/B models)."""
    res = []
    if c.empty:
        return res
    c = c[c.s.isin(["A", "B"])]
    for (cls, q, M), g in c.groupby(["cls", "q", "M"]):
        res.append(dict(cls=cls, q=q, M=float(M), n=int(len(g)), n_models=int(g.m.nunique()),
                        med_dex=float(g.dex.median()), frac_iqr=float(g.in_iqr.mean()), frac_range=float(g.in_range.mean())))
    return res


def validity(pts):
    res = []
    for (s, csm), g in pts.groupby([pts.s, pts.M > 0]):
        d = dict(series=s, csm=bool(csm), n=int(len(g)), n_models=int(g.m.nunique()))
        for sfx, v in (("", "strict"), ("m", "masked")):
            d[f"ew_clean_{v}"] = int(g["eok" + sfx].sum())
            d[f"ratio_valid_{v}"] = int(g["rok" + sfx].sum())
            d[f"ew_reasons_{v}"] = g.loc[~g["eok" + sfx], "ewhy" + sfx].value_counts().to_dict()
            d[f"r_reasons_{v}"] = g.loc[~g["rok" + sfx], "rwhy" + sfx].value_counts().to_dict()
        res.append(d)
    return res


def published_check(m, pub):
    """Paper Tables A1-A2 (NLTE rows) against the new spectra, same model and epoch."""
    p = pub[pub.treatment == "NLTE"][["model", "epoch", "EW_Ha", "R", "inconsistent"]]
    j = p.merge(m, on=["model", "epoch"], how="inner")
    out = dict(n_pairs=int(len(j)), n_models=int(j.model.nunique()))
    if j.empty:
        return out
    best = None
    for hw in (50, 100, 150, 200, 300):
        col = f"rep_{hw}_EW_Ha"
        if col not in j:
            continue
        ok = (j.EW_Ha < 0) & (j[col] < 0)
        d = np.abs(np.log10(j.loc[ok, col] / j.loc[ok, "EW_Ha"]))
        rec = dict(hw=hw, n=int(ok.sum()), med_absdex_ew=float(d.median()), frac_within_0p05=float((d < 0.05).mean()))
        okr = (j.R > 0) & (j[f"rep_{hw}_R"] > 0) & (j.inconsistent.fillna("") == "")
        dr = np.abs(np.log10(j.loc[okr, f"rep_{hw}_R"] / j.loc[okr, "R"]))
        rec.update(n_r=int(okr.sum()), med_absdex_r=float(dr.median()) if len(dr) else None)
        out.setdefault("replica", []).append(rec)
        if best is None or rec["med_absdex_ew"] < best["med_absdex_ew"]:
            best = rec
    out["best_replica"] = best
    for v, pfx in VARIANTS.items():
        ok = (j.EW_Ha < 0) & (j[pfx + "EW_net_Ha"] < 0)
        d = np.log10(j.loc[ok, pfx + "EW_net_Ha"] / j.loc[ok, "EW_Ha"])
        okr = (j.R > 0) & (j[pfx + "R_net"] > 0)
        dr = np.log10(j.loc[okr, pfx + "R_net"] / j.loc[okr, "R"])
        out[v] = dict(n=int(ok.sum()), med_dex_ew=float(d.median()), med_dex_r=float(dr.median()) if len(dr) else None,
                      n_r=int(okr.sum()), n_r_valid=int(j[pfx + "ratio_valid"].sum()))
    out["pairs"] = json.loads(j[["model", "epoch", "EW_Ha", "R"] +
                                [c for c in ("rep_100_EW_Ha", "rep_100_R", "rob_EW_net_Ha", "rob_R_net",
                                             "robm_EW_net_Ha", "robm_R_net") if c in j]].round(4).to_json(orient="records"))
    return out


def evolution(pts, sfx):
    """Per H-rich model: |EW| near peak (t <= 1 d, or the first epoch), at 10-20 d, at 30-50 d, and the
    first epoch at which Hα is net absorption."""
    out = []
    for name, g in pts[pts.s.isin(["A", "B"])].groupby("m"):
        g = g.sort_values("t")

        def at(lo, hi):
            s = g[(g.t >= lo) & (g.t <= hi) & g["e" + sfx].notna()]
            return None if s.empty else dict(t=float(s.t.iloc[0]), e=float(s["e" + sfx].iloc[0]),
                                             ok=bool(s["eok" + sfx].iloc[0]))
        absn = g[g["e" + sfx] < 0]
        out.append(dict(m=name, s=g.s.iloc[0], E=float(g.E.iloc[0]), M=float(g.M.iloc[0]), first=at(-1, 5.5),
                        mid=at(9, 21), late=at(29, 55), very_late=at(70, 160),
                        first_abs=float(absn.t.iloc[0]) if len(absn) else None, epochs=[float(t) for t in g.t]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmeas", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--obs-masked", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = pd.read_csv(a.mmeas, low_memory=False)
    pts = model_points(m)
    obs = {"strict": json.load(open(a.obs)), "masked": json.load(open(a.obs_masked))}
    refs = {v: class_reference(os.path.splitext(f)[0] + "_points.csv")
            for v, f in (("strict", a.obs), ("masked", a.obs_masked))}
    out = dict(n_spectra=int(len(pts)), n_models=int(pts.m.nunique()),
               models=sorted(pts.m.unique(), key=lambda s: (s[0], int(s[1:]))),
               epochs=sorted(set(pts.t)), validity=validity(pts), bins=BINS,
               points=json.loads(pts.round(4).to_json(orient="records")))
    for v, sfx in (("strict", ""), ("masked", "m")):
        c = compare(pts, refs[v], sfx)
        out[f"compare_{v}"] = summarize_compare(c)
        out[f"evolution_{v}"] = evolution(pts, sfx)
        out[f"obs_{v}"] = {cls: dict(ew=obs[v]["classes"][cls]["ew"]["binned"],
                                     r=obs[v]["classes"][cls]["r"].get("binned", []),
                                     n_ew=obs[v]["classes"][cls]["ew"]["n_objects"],
                                     n_r=obs[v]["classes"][cls]["r"]["n_objects"])
                           for cls in CLASSES}
        if not c.empty:
            c.to_csv(os.path.splitext(a.out)[0] + f"_compare_{v}.csv", index=False)
    h = pts[pts.s.isin(["A", "B"]) & (pts.M > 0)]
    out["metal_frac"] = dict(Ha_median=float(h.metal_frac_Ha.median()), Hb_median=float(h.metal_frac_Hb.median()),
                             Ha_early=float(h[h.t <= 20].metal_frac_Ha.median()),
                             Hb_early=float(h[h.t <= 20].metal_frac_Hb.median()))
    cser = pts[pts.s == "C"]
    out["c_series"] = dict(n=int(len(cser)), ew_clean_strict=int(cser.eok.sum()), ew_clean_masked=int(cser.eokm.sum()),
                           ratio_valid_strict=int(cser.rok.sum()), ratio_valid_masked=int(cser.rokm.sum()),
                           ew_clean_values=[float(x) for x in cser.loc[cser.eok, "e"].round(1)])
    out["published"] = published_check(m, pd.read_csv(a.published))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None)
    pts.to_csv(os.path.splitext(a.out)[0] + "_points.csv", index=False)
    print(f"wrote {a.out}: {len(pts)} model spectra of {pts.m.nunique()} models")


if __name__ == "__main__":
    main()
