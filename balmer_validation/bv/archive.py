"""Public SN spectra: Open Supernova Catalog (OSC) JSON files and WISeREP.

The OSC API (api.astrocats.space) is offline, but the catalog's per-object JSON
files are still served from its GitHub data repositories
(github.com/astrocatalogs/sne-<years>).  The OSC stopped ingesting spectra
around 2015-2016, so newer objects are taken from WISeREP object pages (public
spectra only, guest access).

Everything returned here is plain Python/NumPy data; downloaded files are
treated as untrusted input and only parsed as JSON / numeric text.
"""
import datetime as dt
import html
import json
import re
from urllib.parse import unquote

import numpy as np

OSC_RAW = "https://raw.githubusercontent.com/astrocatalogs/{repo}/master/{fname}.json"
WISEREP = "https://www.wiserep.org"
OSC_ALIASES = {"CSS121015": "CSS121015:004244+132827"}


def osc_year(name):
    m = re.match(r"^(\d{4})", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:-|^[A-Za-z]+)(\d{4})(?:-|$)", name)   # OGLE-2013-SN-016
    if m:
        return int(m.group(1))
    m = re.match(r"^[A-Za-z]+-?(\d{2})", name)               # LSQ12heq, PTF10gvf, ASASSN-14il, CSS121015
    if m:
        return 2000 + int(m.group(1))
    return None


def osc_repo(year):
    if year is None:
        return None
    for hi, repo in ((1990, "sne-pre-1990"), (2000, "sne-1990-1999"), (2005, "sne-2000-2004"),
                     (2010, "sne-2005-2009"), (2015, "sne-2010-2014"), (2020, "sne-2015-2019")):
        if year < hi:
            return repo
    return "sne-2020-2024"


def osc_url(name):
    fname = OSC_ALIASES.get(name, ("SN" + name) if re.match(r"^\d{4}", name) else name)
    return OSC_RAW.format(repo=osc_repo(osc_year(name)), fname=fname)


def _first_value(entry, key):
    vals = entry.get(key) or []
    for v in vals:
        if "value" in v:
            return v["value"]
    return None


def _date_to_mjd(s):
    """OSC dates look like 'YYYY/MM/DD' (day may be fractional or missing)."""
    if not s:
        return None
    parts = s.split("/")
    try:
        y = int(parts[0])
        mth = int(parts[1]) if len(parts) > 1 else 1
        day = float(parts[2]) if len(parts) > 2 else 1.0
    except ValueError:
        return None
    d0 = dt.datetime(y, mth, 1)
    return (d0 - dt.datetime(1858, 11, 17)).days + day - 1.0


def parse_osc(text, name):
    """Return (meta, spectra) from an OSC JSON document."""
    doc = json.loads(text)
    entry = next(iter(doc.values()))
    z = _first_value(entry, "redshift")
    meta = dict(name=name, source="OSC", osc_name=entry.get("name"),
                z=float(z) if z not in (None, "") else None,
                maxdate=_first_value(entry, "maxdate"), discoverdate=_first_value(entry, "discoverdate"),
                ebv=float(_first_value(entry, "ebv")) if _first_value(entry, "ebv") else None,
                claimedtype=_first_value(entry, "claimedtype"))
    meta["mjd_max"] = _date_to_mjd(meta["maxdate"])
    meta["mjd_disc"] = _date_to_mjd(meta["discoverdate"])
    srcs = {s.get("alias"): s for s in entry.get("sources", [])}
    spectra = []
    for k, sp in enumerate(entry.get("spectra", [])):
        data = sp.get("data")
        t = sp.get("time")
        if not data or t is None:
            continue
        try:
            arr = np.array([[float(x) for x in row[:2]] for row in data if len(row) >= 2])
        except (TypeError, ValueError):
            continue
        if arr.ndim != 2 or len(arr) < 50:
            continue
        refs = [srcs.get(a.strip(), {}) for a in str(sp.get("source", "")).split(",")]
        bib = ";".join(r.get("bibcode", r.get("name", "")) for r in refs if r)
        spectra.append(dict(
            idx=k, mjd=float(t if not isinstance(t, list) else t[0]), wav=arr[:, 0].tolist(), flux=arr[:, 1].tolist(),
            deredshifted=bool(sp.get("deredshifted", False)), z_spec=sp.get("redshift"),
            u_wav=sp.get("u_wavelengths"), u_flux=sp.get("u_fluxes"), instrument=sp.get("instrument"),
            telescope=sp.get("telescope"), reference=bib, filename=sp.get("filename")))
    return meta, spectra


# ---------------------------------------------------------------------------
# WISeREP (HTML pages, public spectra only)
# ---------------------------------------------------------------------------

def wiserep_object_ids(search_html):
    return sorted(set(int(x) for x in re.findall(r'href="/object/(\d+)"', search_html)))


def _cell(row, name):
    m = re.search(r'<td[^>]*class="[^"]*cell-' + name + r'[^"]*"[^>]*>(.*?)</td>', row, flags=re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", m.group(1)))).strip()


def parse_wiserep_object(page_html):
    """List of public spectra (dict) on a WISeREP object page."""
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, flags=re.S):
        if "cell-asciifile" not in row:
            continue
        asc = _cell(row, "asciifile")
        if not asc:
            continue
        url = None
        for href in re.findall(r'(https?://www\.wiserep\.org/system/files/uploaded/[^"\'\s<>]+)', row):
            if unquote(href).rsplit("/", 1)[-1] == asc.rsplit("/", 1)[-1]:
                url = href
                break
        obsdate = _cell(row, "obsdate")
        try:
            t = dt.datetime.strptime(obsdate[:19], "%Y-%m-%d %H:%M:%S")
            mjd = (t - dt.datetime(1858, 11, 17)).total_seconds() / 86400.0
        except ValueError:
            mjd = None
        z = _cell(row, "redshift")
        out.append(dict(asciifile=asc, url=url, mjd=mjd, obsdate=obsdate,
                        z=float(z) if re.match(r"^-?\d+(\.\d+)?$", z or "") else None,
                        public=_cell(row, "public"), tel_inst=_cell(row, "tel_inst"),
                        spectype=_cell(row, "spectype_name"), flux_calib=_cell(row, "flux_calib_name"),
                        ext_corr=_cell(row, "extinction_corr_name"), wl_medium=_cell(row, "wl_medium_name"),
                        reference=_cell(row, "publish"), obj_type=_cell(row, "obj_type_name"),
                        iauname=_cell(row, "iauname_w_prefix")))
    return out


def parse_ascii_spectrum(text):
    """Two or three numeric columns; comment/header lines ignored."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#%!;" or line[0].isalpha():
            continue
        parts = re.split(r"[\s,]+", line)
        try:
            vals = [float(p) for p in parts[:2]]
        except ValueError:
            continue
        if len(vals) == 2 and np.isfinite(vals).all():
            rows.append(vals)
    arr = np.array(rows)
    if arr.ndim != 2 or len(arr) < 50:
        return None
    return arr[:, 0], arr[:, 1]
