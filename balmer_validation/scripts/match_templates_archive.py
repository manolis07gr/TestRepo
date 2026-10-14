#!/usr/bin/env python
"""Find the archive spectrum each template was made from (feature correlation),
derive each template's phase zero-point, and compare template vs. archive
measurements and relative flux calibration.

Usage:
    python -I scripts/match_templates_archive.py --bank /path/to/duperfit/tempbank/sne \
        --archive data/archive --out results/template_archive_matches.csv
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bv.match import compare  # noqa: E402
from bv.templates import FIGURE_CLASSES, list_templates  # noqa: E402
from scripts_common import rest_frame  # noqa: E402


def _match(job):
    t, meta, spectra = job
    best = None
    for sp in spectra:
        z = sp.get("z_spec")
        try:
            z = float(z) if z not in (None, "") else meta.get("z")
        except (TypeError, ValueError):
            z = meta.get("z")
        if z is None:
            continue
        w, f = rest_frame(sp, z)
        if len(w) < 100 or w[-1] < 2000:
            continue
        c = compare(t.wav, t.flux, w, f)
        if c and (best is None or c["corr"] > best["corr"]):
            best = dict(c, a_idx=sp.get("idx"), a_mjd=sp["mjd"], a_z=z, a_ref=sp.get("reference"),
                        a_inst=sp.get("instrument"), a_file=sp.get("filename"))
    row = dict(key=t.key, folder=t.folder, name=t.name, t_phase=t.phase)
    if best:
        row.update(best)
        zz = 1.0 + (best["a_z"] or 0.0)
        row["a_phase_max"] = (best["a_mjd"] - meta["mjd_max"]) / zz if meta.get("mjd_max") else np.nan
        row["a_phase_disc"] = (best["a_mjd"] - meta["mjd_disc"]) / zz if meta.get("mjd_disc") else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = []
    cache = {}
    for cls, folders in FIGURE_CLASSES.items():
        for t in list_templates(a.bank, folders, phase_range=(-30, 200)):
            if t.wav[-1] < 2000:
                continue
            if t.name not in cache:
                p = os.path.join(a.archive, f"{t.name}.json")
                cache[t.name] = json.load(open(p)) if os.path.exists(p) else None
            d = cache[t.name]
            if not d or not d["spectra"]:
                continue
            jobs.append((t, d["meta"], d["spectra"]))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_match, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    good = df[df["corr"] > 0.95] if "corr" in df else df.iloc[0:0]
    print(f"templates with archive spectra: {len(df)}; matched with corr>0.95: {len(good)}")


if __name__ == "__main__":
    main()
