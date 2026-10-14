#!/usr/bin/env python
"""Download public spectra of every SN in the five template classes.

OSC per-object JSON (GitHub data repositories) first; WISeREP public spectra
for objects the OSC has no spectra for.  One compact JSON per SN is written to
--outdir (metadata + spectra), which should be a dedicated data directory.

Usage:
    python -I scripts/fetch_archive.py --bank /path/to/duperfit/tempbank/sne --outdir data/archive
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import requests  # noqa: E402

from bv.archive import (WISEREP, osc_url, parse_ascii_spectrum, parse_osc, parse_wiserep_object,  # noqa: E402
                        wiserep_object_ids)
from bv.templates import FIGURE_CLASSES, list_templates  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) balmer-validation/1.0 (research use)"}


def get(url, timeout=180, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** (i + 1))
    return None


def fetch_osc(name):
    r = get(osc_url(name))
    if r is None or r.status_code != 200:
        return None, []
    return parse_osc(r.text, name)


def fetch_wiserep(name, delay=1.0):
    r = get(f"{WISEREP}/search?name={name}")
    if r is None or r.status_code != 200:
        return None, []
    ids = wiserep_object_ids(r.text)
    if len(ids) != 1:
        return dict(name=name, source="WISeREP", note=f"{len(ids)} objects found"), []
    time.sleep(delay)
    p = get(f"{WISEREP}/object/{ids[0]}")
    if p is None or p.status_code != 200:
        return None, []
    rows = parse_wiserep_object(p.text)
    zs = [row["z"] for row in rows if row["z"] is not None]
    meta = dict(name=name, source="WISeREP", wiserep_id=ids[0], z=zs[0] if zs else None,
                mjd_max=None, mjd_disc=None, ebv=None,
                claimedtype=rows[0]["obj_type"] if rows else None)
    spectra = []
    for k, row in enumerate(rows):
        if row["public"] != "Y" or not row["url"] or row["mjd"] is None or row["spectype"] not in ("Object", ""):
            continue
        time.sleep(delay)
        a = get(row["url"])
        if a is None or a.status_code != 200:
            continue
        parsed = parse_ascii_spectrum(a.text)
        if parsed is None:
            continue
        w, f = parsed
        spectra.append(dict(idx=k, mjd=row["mjd"], wav=w.tolist(), flux=f.tolist(), deredshifted=False,
                            z_spec=row["z"], u_wav="Angstrom", u_flux=row["flux_calib"] or None,
                            instrument=row["tel_inst"], telescope=None, reference=row["reference"],
                            filename=row["asciifile"], wl_medium=row["wl_medium"], ext_corr=row["ext_corr"]))
    return meta, spectra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these SN names")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    names = sorted({t.name for folders in FIGURE_CLASSES.values()
                    for t in list_templates(a.bank, folders, load=False)})
    if a.only:
        names = [n for n in names if n in a.only]
    log = []
    for name in names:
        out = os.path.join(a.outdir, f"{name}.json")
        if os.path.exists(out):
            with open(out) as fh:
                d = json.load(fh)
            log.append((name, d["meta"].get("source"), len(d["spectra"])))
            continue
        meta, spectra = fetch_osc(name)
        if not spectra:
            wmeta, wspec = fetch_wiserep(name)
            if wspec:
                if meta:   # keep OSC dates/E(B-V) when available
                    for k in ("mjd_max", "mjd_disc", "ebv", "maxdate", "discoverdate"):
                        if meta.get(k) is not None:
                            wmeta[k] = meta[k]
                meta, spectra = wmeta, wspec
        if meta is None:
            meta = dict(name=name, source=None)
        with open(out, "w") as fh:
            json.dump(dict(meta=meta, spectra=spectra), fh)
        log.append((name, meta.get("source"), len(spectra)))
        print(f"{name:20s} {str(meta.get('source')):8s} spectra={len(spectra):4d} z={meta.get('z')} "
              f"max={meta.get('maxdate')}", flush=True)
    missing = [n for n, s, k in log if k == 0]
    print(f"\n{len(log)} objects, {sum(k for _, _, k in log)} spectra; no spectra for: {missing}")


if __name__ == "__main__":
    main()
