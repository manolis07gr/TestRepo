#!/usr/bin/env python
"""Measure Hα EW and Hα/Hβ on every Duperfit template of the five figure classes.

Usage:
    python -I scripts/measure_templates.py --bank /path/to/duperfit/tempbank/sne \
        --out results/templates_measurements.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from bv import qa  # noqa: E402
from bv.measure import replica_measure, robust_measure  # noqa: E402
from bv.templates import FIGURE_CLASSES, list_templates, student_label  # noqa: E402

REPLICA_GRID = [(hw, cr) for cr in ("4000-7000", "full") for hw in (50, 75, 100, 125, 150, 200, 250, 300)]


def _one(args):
    t, = args
    row = dict(key=t.key, folder=t.folder, name=t.name, label=student_label(t.name), phase=t.phase,
               wmin=float(t.wav[0]), wmax=float(t.wav[-1]))
    row["unit_problem"] = qa.wavelength_unit_problem(t)
    if row["unit_problem"]:
        return row
    row["n_strong_neg"] = qa.strong_negative_pixels(t)
    row["max_over_median"] = qa.max_abs_over_median(t)
    for hw, cr in REPLICA_GRID:
        rng = (4000.0, 7000.0) if cr == "4000-7000" else (0.0, 1e6)
        r = replica_measure(t.wav, t.flux, win_hw=hw, cont_range=rng)
        for k, v in r.items():
            row[f"rep_{cr}_{hw}_{k}"] = v
    rb = robust_measure(t.wav, t.flux, t.eflux)
    for k, v in rb.items():
        row[f"rob_{k}"] = v
    rn = robust_measure(t.wav, t.flux, t.eflux, windows="narrow")
    for k, v in rn.items():
        if not k.startswith("pc_"):
            row[f"robn_{k}"] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="Duperfit tempbank/sne directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dups", default=None, help="optional CSV listing duplicate templates")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    rows = []
    all_templates = []
    for fig, folders in FIGURE_CLASSES.items():
        ts = list_templates(a.bank, folders)
        all_templates += ts
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for row in ex.map(_one, [(t,) for t in ts], chunksize=4):
                row["figure"] = fig
                rows.append(row)
        print(f"{fig}: {len(ts)} templates measured", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, df.shape)
    if a.dups:
        d = qa.find_duplicates(all_templates)
        pd.DataFrame([dict(key1=x.key, key2=y.key, phase1=x.phase, phase2=y.phase, corr=c, scale=s)
                      for x, y, c, s in d]).to_csv(a.dups, index=False)
        print("duplicates:", len(d))


if __name__ == "__main__":
    main()
