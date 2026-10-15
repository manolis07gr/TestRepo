#!/usr/bin/env python
"""Measure Hα EW and Hα/Hβ on SuperLite model spectra with the same methods as the observations.

Expects one directory per model under --models (e.g. ~/Downloads/NLTE/A4/*.dat), one file
per epoch with wavelength and flux columns.  The model comes from the directory (or file)
name, the epoch from the file name, the wavelength unit from the values.  With
--catalogue (the MESA_STELLA_CSM_INTERACTION_MODELS.xlsx spreadsheet) the model
parameters are attached and the epochs are checked against the spreadsheet's epoch
lists ("days since Lbol max"; --epoch-ref explosion if the files count from explosion).

Needs numpy, scipy, pandas, matplotlib, openpyxl; specutils only for the optional
student-style ("replica") measurement, which is skipped if it is not installed.

Usage:
    python -I scripts/measure_models.py --models ~/Downloads/NLTE \
        --catalogue MESA_STELLA_CSM_INTERACTION_MODELS.xlsx --out results/model_measurements.csv \
        [--qa-dir figures/models/qa] [--flux-col 1] [--fnu] [--epoch-ref auto|peak|explosion]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bv.measure import HALPHA, C_KMS, _noise, mask_narrow_lines, robust_measure  # noqa: E402
from bv.models import find_spectra, load_spectrum, read_catalogue, resolve_epochs  # noqa: E402

REPLICA_HW = (50, 100, 150, 200, 300)        # Å; paper recipe: Cheb-3 continuum over 4000-7000 Å


def _replica_available():
    import importlib.util
    return importlib.util.find_spec("specutils") is not None


def _one(args):
    f, flux_col, fnu, n_mc, do_replica, qa_dir = args
    row = {k: f[k] for k in ("model", "rel", "treatment", "epoch_file", "epoch_how", "epoch", "epoch_ref",
                             "epoch_listed")}
    try:
        w, fl, info = load_spectrum(f["path"], flux_col=flux_col, fnu=fnu)
    except Exception as e:                       # report and carry on
        row["load_error"] = f"{type(e).__name__}: {e}"
        return row
    row.update(unit_in=info["unit_in"], fnu_in=info["fnu_in"], ncols=info["ncols"],
               header=" | ".join(info["header"])[:300], npix=int(w.size), wmin=float(w[0]), wmax=float(w[-1]))
    near = (w > HALPHA - 300) & (w < HALPHA + 300)
    if near.sum() > 2:
        dl = float(np.median(np.diff(w[near])))
        row.update(dlam_Ha=dl, dv_Ha=dl / HALPHA * C_KMS)
    from scipy.ndimage import median_filter
    band = (w >= 3500) & (w <= 9000)
    run_med = median_filter(fl, size=9, mode="nearest")
    row["n_spikes_opt"] = int(np.sum(band & (fl > 20.0 * np.abs(run_med))))   # single-bin MC spikes
    opt = (w >= 4000) & (w <= 7000)
    if opt.any():
        row["frac_nonpos_opt"] = float(np.mean(fl[opt] <= 0))
        sig = _noise(w[opt], fl[opt], None)
        row["mc_noise_rel"] = float(np.median(sig / np.abs(fl[opt]).clip(1e-300)))
    rb = robust_measure(w, fl, None, n_mc=n_mc)
    row.update({f"rob_{k}": v for k, v in rb.items()})
    rn = robust_measure(w, fl, None, n_mc=n_mc, windows="narrow")
    row.update({f"robn_{k}": v for k, v in rn.items() if not k.startswith("pc_")})
    # same measurement with the narrow non-Balmer lines masked (as for the observed spectra)
    rm = robust_measure(w, mask_narrow_lines(w, fl), None, n_mc=n_mc)
    row.update({f"robm_{k}": v for k, v in rm.items() if not k.startswith("pc_")})
    if do_replica:
        from bv.measure import replica_measure
        for hw in REPLICA_HW:
            r = replica_measure(w, fl, win_hw=hw)
            row.update({f"rep_{hw}_{k}": v for k, v in r.items()})
    if qa_dir:
        _qa_plot(w, fl, row, qa_dir)
    return row


def _qa_plot(w, fl, row, qa_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bv.measure import HBETA
    from bv.plotting import continuum_diagnostic
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    m = (w > 3500) & (w < 9000)
    axs[0].plot(w[m], fl[m], c="k", lw=0.7)
    axs[0].set_title(f"{row['model']}  {row['epoch']:+.1f} d  {row['treatment']}", fontsize=10)
    for lam in (HBETA, HALPHA):
        axs[0].axvline(lam, c="0.6", lw=0.5)
    ew, r = row.get("rob_EW_net_Ha"), row.get("rob_R_net")
    continuum_diagnostic(axs[1], w, fl, HBETA, "Hβ", show_replica=_replica_available(), label="model")
    continuum_diagnostic(axs[2], w, fl, HALPHA, f"Hα  EW={ew:.1f} Å  R={r:.2f}  "
                         f"{'R valid' if row.get('rob_ratio_valid') else 'R not valid'}",
                         show_replica=_replica_available(), label="model")
    axs[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    os.makedirs(qa_dir, exist_ok=True)
    fig.savefig(os.path.join(qa_dir, f"{row['model']}_{row['treatment']}_{row['epoch']:06.1f}.png"), dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="directory with one sub-directory per model")
    ap.add_argument("--catalogue", default=None, help="model spreadsheet (.xlsx)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flux-col", type=int, default=1, help="0-based column holding the flux (default 1)")
    ap.add_argument("--fnu", choices=("auto", "yes", "no"), default="auto",
                    help="flux per unit frequency? auto: only for frequency abscissae or a header saying so")
    ap.add_argument("--epoch-ref", choices=("auto", "peak", "explosion"), default="auto")
    ap.add_argument("--n-mc", type=int, default=200, help="Monte-Carlo draws (200, as for the archive spectra)")
    ap.add_argument("--no-replica", action="store_true", help="skip the student-style (specutils) measurement")
    ap.add_argument("--qa-dir", default=None, help="write one diagnostic plot per spectrum here")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    files = find_spectra(os.path.expanduser(a.models))
    if not files:
        sys.exit(f"no spectrum files found under {a.models}")
    cat = read_catalogue(a.catalogue) if a.catalogue else None
    unknown = sorted({f["rel"] for f in files if f["model"] is None})
    files = [f for f in files if f["model"] is not None]
    files = resolve_epochs(files, cat, a.epoch_ref)
    fnu = None if a.fnu == "auto" else a.fnu == "yes"
    do_rep = not a.no_replica and _replica_available()
    jobs = [(f, a.flux_col, fnu, a.n_mc, do_rep, a.qa_dir) for f in files if np.isfinite(f["epoch"])]
    print(f"{len(files)} files of {len({f['model'] for f in files})} models; "
          f"{len(jobs)} with an epoch; replica: {'yes' if do_rep else 'no (specutils missing or disabled)'}")
    if unknown:
        print(f"  {len(unknown)} files without a model name, skipped, e.g. {unknown[:3]}")
    no_ep = [f["rel"] for f in files if not np.isfinite(f["epoch"])]
    if no_ep:
        print(f"  {len(no_ep)} files without an epoch, skipped, e.g. {no_ep[:3]}")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(_one, jobs, chunksize=2):
            rows.append(row)
    df = pd.DataFrame(rows)
    if cat is not None:
        df = df.merge(cat.drop(columns=["epochs_run", "epochs_done"]), on="model", how="left")
    df = df.sort_values(["model", "treatment", "epoch"], key=lambda s: s.map(_model_key) if s.name == "model" else s)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    _report(df)
    print("wrote", a.out, df.shape)


def _model_key(m):
    return (m[0], int(m[1:])) if isinstance(m, str) and m[1:].isdigit() else (str(m), 0)


def _report(df):
    if "load_error" in df and df.load_error.notna().any():
        print(f"  {df.load_error.notna().sum()} files could not be read, e.g.:")
        print("   ", df[df.load_error.notna()][["rel", "load_error"]].head(3).to_string(index=False))
    ok = df[df.get("npix", pd.Series(dtype=float)).notna()] if "npix" in df else df.iloc[0:0]
    if ok.empty:
        return
    print("  input: wavelength unit", dict(ok.unit_in.value_counts()), "| F_nu converted:", int(ok.fnu_in.sum()),
          "| columns", dict(ok.ncols.value_counts()), "| header e.g.:", ok.header.dropna().head(1).tolist())
    print(f"  pixels near Hα: {ok.dv_Ha.median():.0f} km/s; epoch reference", dict(ok.epoch_ref.value_counts()),
          "| epochs not in the spreadsheet lists:", int((~ok.epoch_listed.astype(bool)).sum()))
    print(f"  Hα EW clean: {int(ok.rob_ew_clean.sum())}/{len(ok)}; valid Balmer decrements: "
          f"{int(ok.rob_ratio_valid.sum())}/{len(ok)}")


if __name__ == "__main__":
    main()
