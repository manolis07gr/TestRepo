#!/usr/bin/env python
"""Point-by-point comparison of the digitized student values with this work.

Usage:
    python -I scripts/compare_student.py --meas results/templates_measurements.csv \
        --dig results/student_digitized.csv --out results/student_vs_thiswork.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLASSES = ["SNIIn", "SNII", "SNIIb", "SNIa-CSM", "SLSN-II"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", required=True)
    ap.add_argument("--dig", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = pd.read_csv(a.meas).drop_duplicates("key").drop(columns=["figure", "name"])
    d = pd.read_csv(a.dig)
    d = d[d.key.notna()]
    j = d.merge(m, on="key", how="left").copy()
    j["this_value"] = np.where(j.quantity == "EW", j.rob_EW_net_Ha, j.rob_R_net)
    j["this_lo"] = np.where(j.quantity == "EW", j.rob_EW_net_Ha_lo, j.rob_R_net_lo)
    j["this_hi"] = np.where(j.quantity == "EW", j.rob_EW_net_Ha_hi, j.rob_R_net_hi)
    j["reliable"] = np.where(j.quantity == "EW", j.rob_ew_clean, j.rob_ratio_valid).astype(bool)
    same_sign = np.sign(j.d_value) == np.sign(j.this_value)
    with np.errstate(divide="ignore", invalid="ignore"):
        j["dlog"] = np.where(same_sign, np.log10(np.abs(j.d_value) / np.abs(j.this_value)), np.nan)
    j["sign_flip"] = ~same_sign & np.isfinite(j.this_value)
    cols = ["figure", "quantity", "cls", "key", "name", "t_phase", "d_value", "this_value", "this_lo", "this_hi",
            "dlog", "sign_flip", "reliable", "rob_flags", "merged", "cost"]
    j[cols].to_csv(a.out, index=False)

    print(f"{'figure':16s} {'N':>4s} {'reliable':>8s} {'|dlog|med(rel)':>14s} {'<0.1dex(rel)':>12s} "
          f"{'signflip(rel)':>13s} {'signflip(unrel)':>15s} {'student<0 / >0':>14s}")
    for (fig, qty), s in j.groupby(["figure", "quantity"], sort=False):
        r = s[s.reliable]
        u = s[~s.reliable]
        print(f"{fig:16s} {len(s):4d} {len(r):8d} {np.nanmedian(np.abs(r.dlog)):14.3f} "
              f"{np.mean(np.abs(r.dlog) < 0.1):12.2f} {int(r.sign_flip.sum()):13d} {int(u.sign_flip.sum()):15d} "
              f"{int((s.d_value < 0).sum()):6d}/{int((s.d_value > 0).sum()):<6d}")


if __name__ == "__main__":
    main()
