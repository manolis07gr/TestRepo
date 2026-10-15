#!/usr/bin/env python
"""Measure Hα EW and Hα/Hβ on the public (OSC / WISeREP) spectra.

Usage:
    python -I scripts/measure_archive.py --bank /path/to/duperfit/tempbank/sne \
        --archive data/archive --out results/archive_measurements.csv
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bv.measure import mask_narrow_lines, robust_measure  # noqa: E402
from bv.templates import FIGURE_CLASSES, list_templates, student_label  # noqa: E402
from scripts_common import rest_frame  # noqa: E402

# CCM89 (R_V = 3.1): A(Hβ) - A(Hα) = 1.07 E(B-V)
DA_HB_HA_PER_EBV = 1.07


def _one(job):
    name, cls, meta, sp = job
    zs = sp.get("z_spec")
    try:
        z = float(zs) if zs not in (None, "") else meta.get("z")
    except (TypeError, ValueError):
        z = meta.get("z")
    w, f = rest_frame(sp, z)
    row = dict(name=name, label=student_label(name), figure=cls, source=meta.get("source"), mjd=sp["mjd"], z=z,
               instrument=sp.get("instrument"), reference=sp.get("reference"), u_flux=sp.get("u_flux"),
               filename=sp.get("filename"), deredshifted=sp.get("deredshifted"), idx=sp.get("idx"),
               wmin=float(w[0]) if len(w) else np.nan, wmax=float(w[-1]) if len(w) else np.nan,
               ebv_mw=meta.get("ebv"))
    zz = 1.0 + (z or 0.0)
    row["phase_max"] = (sp["mjd"] - meta["mjd_max"]) / zz if meta.get("mjd_max") else np.nan
    row["phase_disc"] = (sp["mjd"] - meta["mjd_disc"]) / zz if meta.get("mjd_disc") else np.nan
    usable = len(w) >= 100 and w[-1] > 2000.0 and z is not None
    u = str(sp.get("u_flux") or "").lower()
    row["uncalibrated"] = any(k in u for k in ("uncalib", "count", "arbitrary", "normalized", "relative"))
    if not usable:
        row["usable"] = False
        return row
    row["usable"] = True
    rb = robust_measure(w, f, None, n_mc=200)
    for k, v in rb.items():
        row[f"rob_{k}"] = v
    # same measurement with the narrow non-Balmer lines masked (comparison with the models)
    rm = robust_measure(w, mask_narrow_lines(w, f), None, n_mc=200)
    for k, v in rm.items():
        if not k.startswith("pc_"):
            row[f"robm_{k}"] = v
    if meta.get("ebv") is not None and np.isfinite(row.get("rob_R_net", np.nan)):
        row["rob_R_net_mwcorr"] = row["rob_R_net"] * 10 ** (-0.4 * DA_HB_HA_PER_EBV * meta["ebv"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    cls_of = {}
    for cls, folders in FIGURE_CLASSES.items():
        for t in list_templates(a.bank, folders, load=False):
            cls_of[t.name] = cls
    jobs = []
    for path in sorted(glob.glob(os.path.join(a.archive, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        name = d["meta"]["name"]
        for sp in d["spectra"]:
            jobs.append((name, cls_of.get(name), d["meta"], sp))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_one, jobs, chunksize=8):
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, df.shape, "usable:", int(df.usable.sum()))


if __name__ == "__main__":
    main()
