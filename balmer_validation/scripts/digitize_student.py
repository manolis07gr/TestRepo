#!/usr/bin/env python
"""Digitize the 10 figures under review and match every marker to a template.

Usage:
    python -I scripts/digitize_student.py --figdir figures/student \
        --bank /path/to/duperfit/tempbank/sne --out results/student_digitized.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from bv.digitize import digitize, match  # noqa: E402
from bv.templates import FIGURE_CLASSES, list_templates, student_label, student_style  # noqa: E402

# figure file -> (quantity, class, values of the major y ticks detected, top to bottom)
FIGURES = {
    "EW_SNIIn.jpg": ("EW", "SNIIn", [10, 1, -1, -10, -100, -1000, -10000]),
    "EW_SNII.jpg": ("EW", "SNII", [100, 10, 1, 0, -1, -10, -100, -1000, -1e4, -1e5, -1e6]),
    "EW_SNIIb.jpg": ("EW", "SNIIb", [10, 1, 0, -1, -10, -100, -1000]),
    "EW_SNIa-CSM.jpg": ("EW", "SNIa-CSM", [-100, -1000]),
    "EW_SLSN-II.jpg": ("EW", "SLSN-II", [10, 1, 0, -1, -10, -100, -1000]),
    "R_SNIIn.jpg": ("R", "SNIIn", [100, 10, 1, 0, -1, -10]),
    "R_SNII.jpg": ("R", "SNII", [1000, 100, 10, 1, 0, -1, -10, -100, -1000]),
    "R_SNIIb.jpg": ("R", "SNIIb", [10, 1, 0, -1, -10]),
    "R_SNIa-CSM.jpg": ("R", "SNIa-CSM", [100, 10, 1, 0, -1, -10, -100]),
    "R_SLSN-II.jpg": ("R", "SLSN-II", [100, 10, 1, 0, -1, -10]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1.6, help="phase tolerance (days) for matching")
    a = ap.parse_args()

    rows = []
    for fname, (qty, cls, ticks) in FIGURES.items():
        points, meta = digitize(os.path.join(a.figdir, fname), ticks)
        ts = list_templates(a.bank, FIGURE_CLASSES[cls], phase_range=(-21, 151), load=False)
        style = student_style([student_label(t.name) for t in list_templates(a.bank, FIGURE_CLASSES[cls], load=False)])
        entries = [dict(key=t.key, name=t.name, phase=t.phase, color=style[student_label(t.name)][0],
                        marker=style[student_label(t.name)][1]) for t in ts]
        pairs = match(points, entries, tol_days=a.tol)
        matched = {i for i, _, _ in pairs}
        for i, j, cost in pairs:
            p, e = points[i], entries[j]
            rows.append(dict(figure=fname, quantity=qty, cls=cls, key=e["key"], name=e["name"],
                             t_phase=e["phase"], d_phase=p["phase"], d_value=p["value"], d_symlog=p["t"],
                             color=p["color"], d_shape=p["shape"], t_marker=e["marker"],
                             shape_score=p["shape_score"], merged=p["merged"], cost=cost,
                             px=p["px"], py=p["py"]))
        for i, p in enumerate(points):
            if i not in matched:
                rows.append(dict(figure=fname, quantity=qty, cls=cls, key=None, name=None, t_phase=None,
                                 d_phase=p["phase"], d_value=p["value"], d_symlog=p["t"], color=p["color"],
                                 d_shape=p["shape"], t_marker=None, shape_score=p["shape_score"],
                                 merged=p["merged"], cost=None, px=p["px"], py=p["py"]))
        print(f"{fname:16s} points={len(points):4d} matched={len(pairs):4d} "
              f"tick-fit residual max={max(abs(r) for r in meta['tick_resid_px']):.2f}px")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
