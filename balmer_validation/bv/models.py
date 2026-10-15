"""SuperLite model spectra (MESA -> STELLA -> SuperLite): catalogue, file discovery, loading.

Catalogue
    The spreadsheet MESA_STELLA_CSM_INTERACTION_MODELS.xlsx holds
      * a parameter sheet (header row starting with "Model ID"): progenitor, explosion and
        CSM parameters of the A (RSG, 99em-like), B (YSG, 12A-like), C (stripped, 13bvn-like)
        and X (test) models;
      * epoch sheets (header "Model", "day_Lbol_max", then epochs): the epochs that were
        run ("run") or finished ("done"), and the day of bolometric maximum since explosion.
    Sheets are found by their header row, not by name.

Spectra
    <root>/<model>/<file>.dat, one file per epoch, columns wavelength and flux (header
    lines, comment lines, commas and Fortran "D" exponents are tolerated).  The model is
    read from the directory name (or the file name), the epoch from the file name, the
    wavelength unit from the values (Å, nm, µm, cm, m, or frequency in Hz).
"""
import os
import re
from dataclasses import dataclass, field

import numpy as np

C_AA_S = 2.99792458e18          # speed of light in Å/s

# "A4", "b12", "C_7", "model_A4", "modelA4" (an upper-case series letter may follow lower case)
MODEL_RE = re.compile(r"(?:(?<![A-Za-z0-9])([ABCXabcx])|(?<=[a-z])([ABCX]))[_-]?(\d{1,2})(?![0-9])")
NUM = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
# epoch tokens, most specific first: "day20", "t_20", "ep5", "p0.1" / "20d", "20days"
EPOCH_PATTERNS = [
    re.compile(r"(?:^|[_\-\s.])(?:days?|d|t|time|ep|epoch|phase|p)[_\-]?(" + NUM + r")(?=$|[_\-\s.a-zA-Z])", re.I),
    re.compile(r"(?:^|[_\-\s.])(" + NUM + r")[_\-]?(?:days?|d)(?=$|[_\-\s.])", re.I),
]
SERIES = {"A": "RSG (SN 1999em-like)", "B": "YSG (SN 2012A-like)",
          "C": "stripped (iPTF13bvn-like)", "X": "test"}

PARAM_COLUMNS = {
    "model id": "model",
    "progenitor zams mass": "m_zams",
    "progenitor mass at explosion [msun]": "m_exp",
    "progenitor radius at explosion [rsun]": "r_exp",
    "explosion energy [10^51 erg]": "e51",
    "nickel-56 mass (mni) [msun]": "m_ni",
    "mcsm [msun]": "m_csm",
    "mdot,csm [msun/yr]": "mdot_csm",
    "v_csm [km/s]": "v_csm",
    "t_csm [yr]": "t_csm",
    "csm density index": "s_csm",
    "csm composition": "csm_comp",
    "comments": "comment",
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def read_catalogue(xlsx_path):
    """Model catalogue from the spreadsheet: one row per model.

    Columns: model, series, the PARAM_COLUMNS values, day_lbol_max (first epoch sheet),
    day_lbol_max_alt (second epoch sheet, if any), epochs_run / epochs_done (sorted lists).
    """
    import openpyxl
    import pandas as pd

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    params, epoch_sheets = None, []
    for ws in wb.worksheets:
        rows = [r for r in ws.iter_rows(values_only=True) if any(v is not None for v in r)]
        if not rows:
            continue
        head = [str(v).strip().lower() if v is not None else "" for v in rows[0]]
        if head and head[0] == "model id":
            params = (head, rows[1:])
        elif len(head) > 2 and head[0] == "model" and head[1] == "day_lbol_max":
            epoch_sheets.append((ws.title, rows))
    if params is None:
        raise ValueError(f"{xlsx_path}: no sheet with a 'Model ID' header row")
    head, body = params
    recs = []
    for r in body:
        if r[0] is None:
            continue
        rec = {}
        for j, h in enumerate(head):
            key = PARAM_COLUMNS.get(h)
            if key is None or j >= len(r):
                continue
            v = r[j]
            if key in ("model", "csm_comp", "comment"):
                rec[key] = None if v is None else str(v).strip()
            else:
                rec[key] = _num(v)
        rec["series"] = rec["model"][0].upper()
        recs.append(rec)
    cat = pd.DataFrame(recs)
    cat["csm_comp"] = cat["csm_comp"].where(cat["csm_comp"] != "N/A")
    cat["has_csm"] = cat["m_csm"].fillna(0) > 0
    for i, (title, rows) in enumerate(epoch_sheets[:2]):
        epochs = [_num(v) for v in rows[0][2:]]
        dmax, run = {}, {}
        for r in rows[1:]:
            if r[0] is None:
                continue
            dmax[str(r[0]).strip()] = _num(r[1])
            run[str(r[0]).strip()] = sorted(e for e, v in zip(epochs, r[2:]) if v is not None and np.isfinite(e))
        suffix = "" if i == 0 else "_alt"
        cat["day_lbol_max" + suffix] = cat.model.map(dmax)
        status = "run" if i == 0 else "done"
        cat[f"epochs_{status}"] = cat.model.map(lambda m: run.get(m, []))
    return cat


def parse_model(*names):
    """First model token (A4, b12, C_7, X1, ...) found in the given names, upper-case."""
    for name in names:
        m = MODEL_RE.search(os.path.basename(str(name)))
        if m:
            return _model_id(m)
    return None


def _model_id(m):
    return f"{(m.group(1) or m.group(2)).upper()}{int(m.group(3))}"


def parse_epoch(filename, model=None):
    """Epoch from a file name; returns (epoch, how) or (None, reason)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if model:
        stem = MODEL_RE.sub(lambda m: "_" if _model_id(m) == model else m.group(0), stem)
    stem = re.sub(r"(?i)n?lte", "_", stem)
    for pat in EPOCH_PATTERNS:
        hits = pat.findall(stem)
        if len(hits) == 1:
            return float(hits[0]), "token"
    nums = re.findall(NUM, stem)
    if len(nums) == 1:
        return float(nums[0]), "single-number"
    return None, f"ambiguous ({len(nums)} numbers in '{stem}')"


def parse_treatment(path, default="unknown"):
    s = path.replace("\\", "/")
    if re.search(r"(?i)nlte|non[-_]?lte", s):
        return "NLTE"
    if re.search(r"(?i)(?<![a-z])lte", s):
        return "LTE"
    return default


def find_spectra(root, exts=(".dat", ".txt", ".spec", ".csv", ".flx", ".out")):
    """All spectrum files under root as dicts (path, model, epoch, epoch_how, treatment)."""
    out = []
    root_treat = parse_treatment(os.path.basename(os.path.abspath(root)))
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(exts) or fn.startswith("."):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            model = parse_model(*reversed(rel.split(os.sep)[:-1]), fn)
            epoch, how = parse_epoch(fn, model)
            out.append(dict(path=path, rel=rel, model=model, epoch_file=epoch, epoch_how=how,
                            treatment=parse_treatment(rel, root_treat)))
    return out


def _to_float(tok):
    return float(re.sub(r"(?<=[0-9.])[dD](?=[+-]?\d)", "e", tok))


def read_columns(path):
    """Numeric rows of a text table (modal column count) and the non-numeric header lines."""
    header, rows = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s[0] in "#!%;":
                header.append(s)
                continue
            toks = [t for t in re.split(r"[,\s;]+", s) if t]
            try:
                rows.append([_to_float(t) for t in toks])
            except ValueError:
                header.append(s)
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    ncol = max(set(map(len, rows)), key=[len(r) for r in rows].count)
    arr = np.array([r for r in rows if len(r) == ncol], float)
    return arr, header


UNIT_TO_AA = (("AA", 1.0), ("nm", 10.0), ("um", 1e4), ("cm", 1e8), ("m", 1e10))


def infer_wavelength_unit(x):
    """Unit of the first column: 'Hz' for frequencies, else the unit that puts the Balmer
    region (4500-7000 Å) inside the data with a plausible maximum (< 1e6 Å)."""
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        raise ValueError("no positive abscissa values")
    if 1e12 < np.median(x) < 1e18:
        return "Hz"
    for name, k in UNIT_TO_AA:
        lam = x * k
        if lam.min() <= 4500.0 and lam.max() >= 7000.0 and lam.max() < 1e6:
            return name
    for name, k in UNIT_TO_AA:                  # partial coverage: median in the optical/IR
        if 1e3 <= np.median(x * k) <= 3e4:
            return name
    raise ValueError(f"cannot infer the wavelength unit (range {x.min():.3g}-{x.max():.3g})")


FNU_HINT = re.compile(r"(?i)(f_?nu\b|per\s*hz|/\s*hz|hz\^?-1|hz\*\*-1|hz-1)")


@dataclass
class ModelSpectrum:
    model: str
    epoch: float
    treatment: str
    path: str
    wav: np.ndarray            # Å, ascending
    flux: np.ndarray           # F_lambda, arbitrary normalisation
    unit_in: str
    fnu_in: bool
    ncols: int
    header: list = field(default_factory=list)


def load_spectrum(path, flux_col=1, fnu=None):
    """(wav [Å], F_lambda, info).  fnu=None: F_nu only for frequency abscissae or when the
    header says so; True/False forces the interpretation of the flux column."""
    arr, header = read_columns(path)
    if arr.shape[1] <= flux_col:
        raise ValueError(f"{path}: {arr.shape[1]} columns, flux column {flux_col} requested")
    x, y = arr[:, 0], arr[:, flux_col]
    unit = infer_wavelength_unit(x)
    if fnu is None:
        fnu = unit == "Hz" or any(FNU_HINT.search(h) for h in header)
    if unit == "Hz":
        lam = C_AA_S / x
    else:
        lam = x * dict(UNIT_TO_AA)[unit]
    if fnu:
        y = y * C_AA_S / lam ** 2
    ok = np.isfinite(lam) & np.isfinite(y) & (lam > 0)
    lam, y = lam[ok], y[ok]
    order = np.argsort(lam)
    lam, y = lam[order], y[order]
    keep = np.concatenate([[True], np.diff(lam) > 0])
    return lam[keep], y[keep], dict(unit_in=unit, fnu_in=bool(fnu), ncols=int(arr.shape[1]), header=header)


def resolve_epochs(files, cat=None, epoch_ref="auto", tol=0.6):
    """Days since bolometric maximum for every file.

    epoch_ref 'peak'      file epochs are days since Lbol max (as in the spreadsheet's epoch sheets)
              'explosion' file epochs are days since explosion; day_lbol_max is subtracted
              'auto'      per model: 'peak' if most file epochs are in the spreadsheet's epoch list,
                          'explosion' if most of (epoch - day_lbol_max) are, else 'peak' (flagged)
    Adds keys epoch (days since Lbol max), epoch_ref, epoch_listed (in the spreadsheet list).
    """
    info = {}
    if cat is not None:
        for _, r in cat.iterrows():
            listed = sorted(set(r.get("epochs_done", []) or []) | set(r.get("epochs_run", []) or []))
            dmax = r.get("day_lbol_max_alt")
            if dmax is None or not np.isfinite(dmax):
                dmax = r.get("day_lbol_max")
            info[r.model] = (np.array(listed, float), dmax)

    def in_list(e, listed):
        return listed.size > 0 and np.min(np.abs(listed - e)) <= tol

    by_model = {}
    for f in files:
        by_model.setdefault(f["model"], []).append(f)
    for model, fs in by_model.items():
        listed, dmax = info.get(model, (np.array([]), np.nan))
        eps = [f["epoch_file"] for f in fs if f["epoch_file"] is not None]
        ref = epoch_ref
        if ref == "auto":
            n_peak = sum(in_list(e, listed) for e in eps)
            n_expl = sum(in_list(e - dmax, listed) for e in eps) if np.isfinite(dmax) else -1
            ref = "explosion" if n_expl > n_peak else "peak"
        for f in fs:
            e = f["epoch_file"]
            if e is None:
                f.update(epoch=np.nan, epoch_ref=ref, epoch_listed=False)
                continue
            if ref == "explosion":
                e = e - dmax if np.isfinite(dmax) else np.nan
            f.update(epoch=e, epoch_ref=ref, epoch_listed=bool(in_list(e, listed)))
    return files
