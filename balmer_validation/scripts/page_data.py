#!/usr/bin/env python
"""Compact data block for the Results tab of the report page.

Usage: python -I scripts/page_data.py --results results --out results/results_page_data.json \
           [--models results/model_results.json]
"""
import argparse
import json
import os

import numpy as np

CLASS_LABEL = {"SNIIn": "SNe IIn", "SLSN-II": "SLSN-II", "SNIa-CSM": "SNe Ia-CSM", "SNII": "SNe II", "SNIIb": "SNe IIb"}


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def bins(bb):
    return [dict(mid=b["mid"], lo=b["lo"], hi=b["hi"], n=b["n_obj"], med=r4(b["med"]), q1=r4(b["q1"]), q3=r4(b["q3"]))
            for b in bb]


def models_block(results, models_json):
    """Models tab: model points (both measurement variants), the observed reference of both variants,
    and the summary numbers of model_results.py."""
    md = json.load(open(models_json))
    keep = ["m", "s", "E", "M", "t", "e", "el", "eh", "eok", "ewhy", "r", "rl", "rh", "rok", "rwhy",
            "em", "elm", "ehm", "eokm", "ewhym", "rm", "rlm", "rhm", "rokm", "rwhym", "metal_frac_Ha", "metal_frac_Hb"]
    pts = []
    for p in md["points"]:
        q = {k: (r4(p[k]) if isinstance(p.get(k), float) else p.get(k)) for k in keep}
        q["mfa"], q["mfb"] = q.pop("metal_frac_Ha"), q.pop("metal_frac_Hb")
        pts.append(q)
    obs = {}
    for v in ("strict", "masked"):
        o = json.load(open(os.path.join(results, "observed_results.json" if v == "strict" else "observed_results_masked.json")))
        obs[v] = dict(classes={k: dict(ew=bins(c["ew"]["binned"]), r=bins(c["r"].get("binned", [])),
                                       n_ew=c["ew"]["n_objects"], n_r=c["r"]["n_objects"])
                               for k, c in o["classes"].items()},
                      points=[dict(n=x["name"], c=x["figure"], p=r4(x["phase"]), e=r4(x.get("ew")), r=r4(x.get("r")))
                              for x in o["points"]])
    pub = md["published"]
    return dict(pts=pts, obs=obs, validity=md["validity"], metal=md["metal_frac"], cser=md["c_series"],
                pub={k: pub.get(k) for k in ("n_pairs", "n_models", "best_replica", "replica", "strict", "masked")},
                cmp=dict(strict=md["compare_strict"], masked=md["compare_masked"]),
                n_spectra=md["n_spectra"], n_models=md["n_models"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None,
                    help="results/model_results.json: adds the Models tab data (not part of the public build)")
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.results, "observed_results.json")))
    pts = []
    for p in d["points"]:
        pts.append(dict(n=p["name"], c=p["figure"], p=r4(p["phase"]), ref=p["phase_ref"], s=p["source"],
                        k=p["n_spec"], iin=bool(p.get("slsn_iin")),
                        e=r4(p.get("ew")), el=r4(p.get("ew_lo")), eh=r4(p.get("ew_hi")),
                        r=r4(p.get("r")), rl=r4(p.get("r_lo")), rh=r4(p.get("r_hi"))))
    classes = {}
    for k, st in d["classes"].items():
        sl_e = {s["name"]: s["slope"] for s in st["ew"]["slopes"]}
        sl_r = {s["name"]: s["slope"] for s in st["r"].get("slopes", [])}
        classes[k] = dict(
            label=CLASS_LABEL[k], interacting=st["interacting"],
            ew=dict(n=st["ew"]["n_nights"], obj=st["ew"]["n_objects"],
                    bins=[dict(mid=b["mid"], lo=b["lo"], hi=b["hi"], n=b["n_obj"], med=r4(b["med"]),
                               q1=r4(b["q1"]), q3=r4(b["q3"])) for b in st["ew"]["binned"]],
                    slopes=sl_e),
            r=dict(n=st["r"]["n_nights"], obj=st["r"]["n_objects"],
                   bins=[dict(mid=b["mid"], lo=b["lo"], hi=b["hi"], n=b["n_obj"], med=r4(b["med"]),
                              q1=r4(b["q1"]), q3=r4(b["q3"])) for b in st["r"].get("binned", [])],
                   slopes=sl_r))
    lim = d["ia_csm_lower_limits"]
    out = dict(points=pts, classes=classes,
               limits=[dict(n=p["name"], p=r4(p["phase"]), r=r4(p["r_ll"])) for p in lim["points"]],
               case_b=d["case_b"])
    if a.models:
        out["models"] = models_block(a.results, a.models)
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("wrote", a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
