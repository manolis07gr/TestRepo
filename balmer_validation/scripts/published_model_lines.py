#!/usr/bin/env python
"""Published Hα/Hβ line properties of the SuperLite A- and B-series models.

Parses Tables A1 and A2 of Wagle, Chatzopoulos & Baer (2025, ApJ 978, 112; arXiv:2411.11517)
from the PDF text (``pdftotext -layout``): models A4-A15 and B4-B15, epochs 0.1-20 d after
the bolometric maximum, LTE and NLTE, measured with the specutils recipe (generic
continuum over 400-700 nm, ``equivalent_width``, ``line_flux``).

Internal consistency is checked: column 14 (F(Hα)/F(Hβ)) against columns 7/11, and column 13
(amplitude ratio) against 5/9; rows that disagree by more than 5 % are flagged (typesetting
errors in the tables).

Usage:
    pdftotext -layout 2411.11517.pdf paper.txt
    python -I scripts/published_model_lines.py --txt paper.txt --out results/published_model_lines.csv
"""
import argparse
import re

import numpy as np
import pandas as pd

ORDER = [f"A{i}" for i in range(4, 16)] + [f"B{i}" for i in range(4, 16)]
NUMTOK = r"-?\d+(?:\.\d+)?"
VALERR = re.compile(r"^(-?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)$")
COLS = ["amp_Ha_cn", "amp_Ha_cs", "F_Ha", "EW_Ha", "amp_Hb_cn", "amp_Hb_cs", "F_Hb", "EW_Hb",
        "amp_ratio", "R"]


def _val(tok):
    m = VALERR.match(tok)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(tok), np.nan
    except ValueError:
        return np.nan, np.nan


def _row(line):
    """Values of an LTE/NLTE row; '±' pairs (possibly spaced) are kept together."""
    s = re.sub(r"\s*±\s*", "±", line.strip())
    toks = s.split()
    treat = toks[0]
    vals = toks[1:]
    if len(vals) != 10:
        return None
    out = dict(treatment=treat)
    for c, t in zip(COLS, vals):
        v, e = _val(t)
        out[c] = v
        if c in ("F_Ha", "F_Hb"):
            out[c + "_err"] = e
    return out


def parse(lines):
    start = next(i for i, l in enumerate(lines) if "Table A1. Hydrogen Line Properties" in l)
    stop = next(i for i, l in enumerate(lines) if l.strip().startswith("REFERENCES") and i > start)
    pending = []
    for line in lines[start:stop]:
        s = line.strip()
        if re.match(r"^N?LTE\s", s):
            r = _row(s)
            if r:
                pending.append(r)
            continue
        m = re.match(r"^(?:[AB]\d{1,2}\s+)?(" + NUMTOK + r")\s+(" + NUMTOK + r")$", s)
        if m:
            pending.append(("epoch", (float(m.group(1)), float(m.group(2)))))
    # pending holds LTE row, epoch marker, NLTE row, ... in reading order
    triples, i = [], 0
    while i + 3 <= len(pending):
        a, b, c = pending[i:i + 3]
        if isinstance(a, dict) and isinstance(b, tuple) and isinstance(c, dict):
            triples.append((b[1], a, c))
            i += 3
        else:
            i += 1
    # a new model starts whenever the epoch does not increase
    models, cur, last = [], [], None
    for ep, a, c in triples:
        if last is not None and ep[0] <= last:
            models.append(cur)
            cur = []
        cur.append((ep, a, c))
        last = ep[0]
    models.append(cur)
    if len(models) != len(ORDER):
        raise ValueError(f"found {len(models)} model blocks, expected {len(ORDER)}")
    rows = []
    for name, blk in zip(ORDER, models):
        for (ep, lbol), a, c in blk:
            for r in (a, c):
                rows.append(dict(model=name, epoch=ep, lbol42=lbol, **r))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True, help="pdftotext -layout output of arXiv:2411.11517")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.txt, errors="replace") as fh:
        df = parse(fh.read().splitlines())
    with np.errstate(divide="ignore", invalid="ignore"):
        df["R_check"] = df.F_Ha / df.F_Hb
        df["amp_ratio_check"] = df.amp_Ha_cn / df.amp_Hb_cn
    bad_r = ~np.isclose(df.R, df.R_check, rtol=0.05)
    bad_a = ~np.isclose(df.amp_ratio, df.amp_ratio_check, rtol=0.05)
    df["inconsistent"] = np.where(bad_r & bad_a, "R;amp", np.where(bad_r, "R", np.where(bad_a, "amp", "")))
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows ({df.model.nunique()} models, epochs {sorted(df.epoch.unique())}); "
          f"R column inconsistent with F(Hα)/F(Hβ) in {int(bad_r.sum())} rows")


if __name__ == "__main__":
    main()
