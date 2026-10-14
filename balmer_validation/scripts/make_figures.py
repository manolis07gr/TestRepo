#!/usr/bin/env python
"""All figures of the validation: corrected versions of the 10 figures under review,
student-vs-this-work comparison, window sensitivity, continuum diagnostics, defects.

Usage:
    python -I scripts/make_figures.py --bank /path/to/duperfit/tempbank/sne \
        --meas results/templates_measurements.csv --cmp results/student_vs_thiswork.csv --outdir figures
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bv.digitize import symlog  # noqa: E402
from bv.measure import HALPHA, HBETA, replica_measure  # noqa: E402
from bv.plotting import balmer_figure, continuum_diagnostic, student_style_figure  # noqa: E402
from bv.templates import FIGURE_CLASSES, Template, list_templates, student_label  # noqa: E402

CLASSES = ["SNIIn", "SNII", "SNIIb", "SNIa-CSM", "SLSN-II"]


def corrected_figures(m, bank, outdir):
    os.makedirs(outdir, exist_ok=True)
    m = m[(m.phase >= -20) & (m.phase <= 150)]
    for cls in CLASSES:
        s = m[m.figure == cls].copy()
        labels_all = [student_label(t.name) for t in list_templates(bank, FIGURE_CLASSES[cls], load=False)]
        # EW: every measurable spectrum; grey = not a clean emission-line measurement
        e = s[s.rob_ew_valid.astype(bool)]
        sub = (f"this work: local continuum, ±10,000 km/s, 16-84% MC errors; "
               f"{int(e.rob_ew_clean.sum())}/{len(e)} clean")
        student_style_figure(e, "rob_EW_net_Ha", f"Duperfit {cls} templates", "EW",
                             os.path.join(outdir, f"EW_{cls}_corrected.png"),
                             err_cols=("rob_EW_net_Ha_lo", "rob_EW_net_Ha_hi"), flag_col="rob_ew_clean",
                             subtitle=sub, labels_all=labels_all)
        # Balmer decrement: valid measurements only, log axis
        r = s[s.rob_ratio_valid.astype(bool)]
        both = s[s.rob_cov_Ha.astype(bool) & s.rob_cov_Hb.astype(bool)]
        balmer_figure(r, f"Duperfit {cls} templates", labels_all, len(both),
                      os.path.join(outdir, f"R_{cls}_corrected.png"))


def comparison_figure(c, out):
    """Student (digitized) vs this work, per class, reliable vs unreliable."""
    fig, axs = plt.subplots(2, 5, figsize=(26, 11))
    ticks = [-1e4, -1e3, -1e2, -10, -1, 0, 1, 10, 1e2, 1e3]
    for r, qty in enumerate(("EW", "R")):
        for k, cls in enumerate(CLASSES):
            ax = axs[r, k]
            s = c[(c.quantity == qty) & (c.cls == cls)]
            for rel, kw in ((False, dict(c="0.75", s=14, label="ill-defined (flagged)")),
                            (True, dict(c="C0", s=18, label="well-defined"))):
                q = s[s.reliable == rel]
                ax.scatter(symlog(q.this_value.astype(float)), q.d_symlog if "d_symlog" in q else
                           symlog(q.d_value.astype(float)), **kw)
            lim = (-8.5, 8.5) if qty == "EW" else (-8.5, 8.5)
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xticks(symlog(ticks))
            ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8, rotation=45)
            ax.set_yticks(symlog(ticks))
            ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=8)
            rel = s[s.reliable]
            stat = (f"well-defined: {len(rel)}/{len(s)}\n"
                    f"median |Δlog| = {np.nanmedian(np.abs(rel.dlog)):.2f} dex" if len(rel) else
                    f"well-defined: 0/{len(s)}")
            ax.text(0.03, 0.97, stat, transform=ax.transAxes, va="top", fontsize=10)
            name = "EW(Hα) [Å]" if qty == "EW" else "Hα/Hβ"
            ax.set_title(f"{name} — {cls}", fontsize=12)
            ax.set_xlabel("this work (same template)")
            ax.set_ylabel("student (digitized from figure)")
            if r == 0 and k == 0:
                ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def window_sensitivity_figure(bank, out):
    """Student-style (global continuum) ratio vs integration window for a few spectra."""
    cases = [("SNII", "sn2004dj.p0056"), ("SNII", "sn2013am.p0028"), ("SNII", "sn2009N.p0033"),
             ("SNIa-CSM", "sn1999E.p0024"), ("SNIa-CSM", "sn2012ca.p0096"), ("SNIIn", "sn2010jl.p0100")]
    hws = np.arange(40, 321, 10)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
    for i, (c, n) in enumerate(cases):
        t = Template(path=os.path.join(bank, c, n + ".dat"), folder=c, name=n[2:].split(".")[0], phase=0).load()
        ew, rr = [], []
        for hw in hws:
            res = replica_measure(t.wav, t.flux, win_hw=float(hw))
            ew.append(res["EW_Ha"])
            rr.append(res["R"])
        lab = f"{c} {n[2:].replace('.p', ' +').replace('.dat', '')} d"
        axs[0].plot(hws, symlog(np.array(ew)), "-o", ms=3, label=lab, color=f"C{i}")
        axs[1].plot(hws, symlog(np.array(rr)), "-o", ms=3, label=lab, color=f"C{i}")
    for ax, ttl, ticks in ((axs[0], "Hα EW [Å]", [-1e3, -1e2, -10, -1, 0, 1, 10, 1e2]),
                           (axs[1], "Hα/Hβ", [-1e4, -1e3, -1e2, -10, -1, 0, 1, 10, 1e2, 1e3, 1e4])):
        ax.set_yticks(symlog(ticks))
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.axhline(0, c="k", lw=0.6)
        ax.set_xlabel("half-width of integration window [Å]  (continuum: global Chebyshev-3, 4000–7000 Å)")
        ax.set_ylabel(ttl + "  (symlog, linthresh 2)")
        ax.grid(alpha=0.3)
    axs[1].axhline(symlog(2.86), c="0.4", ls="--", lw=1)
    axs[0].set_title("Student-style EW vs. window choice", fontsize=13)
    axs[1].set_title("Student-style Hα/Hβ vs. window choice: sign flips where F(Hβ) ≈ 0", fontsize=13)
    axs[1].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def diagnostics_figure(bank, out):
    cases = [("SNIIn", "sn2010jl.p0100", "IIn: clean emission (valid)"),
             ("SNII", "sn2004dj.p0056", "SN II plateau: Hβ blended with Fe II"),
             ("SNIa-CSM", "sn1999E.p0024", "Ia-CSM: Hβ ≈ 0 on SN Ia spectrum"),
             ("SNIIb", "sn2008ax.m0015", "IIb pre-max: fast Hα absorption"),
             ("SNIIb", "sn2008ax.p0121", "IIb nebular: [O I] 6300 next to Hα"),
             ("SNII", "sn1999br.p0041", "corrupted template")]
    fig, axs = plt.subplots(len(cases), 2, figsize=(15, 3.3 * len(cases)))
    for i, (c, n, note) in enumerate(cases):
        t = Template(path=os.path.join(bank, c, n + ".dat"), folder=c, name=n[2:].split(".")[0], phase=0).load()
        nice = n[2:].replace(".p", " +").replace(".m", " −").lstrip("0")
        continuum_diagnostic(axs[i, 0], t.wav, t.flux, HALPHA, f"{nice} d — Hα — {note}")
        continuum_diagnostic(axs[i, 1], t.wav, t.flux, HBETA, f"{nice} d — Hβ")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axs[-1]:
        ax.set_xlabel("rest wavelength [Å]")
    fig.text(0.5, 0.995, "blue bands: continuum sidebands (this work); green: ±10,000 km/s integration window",
             ha="center", va="top", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=75)
    plt.close(fig)


def defects_figure(bank, dups, out):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    t = Template(path=os.path.join(bank, "SNII", "sn1999br.p0041.dat"), folder="SNII", name="1999br", phase=41).load()
    axs[0].plot(t.wav, t.flux, c="k", lw=0.8)
    axs[0].set_xlim(6000, 7200)
    axs[0].set_title("SN 1999br, '+41 d' template: flux reaches −240,000\n(median of spectrum ≈ 1) across Hα", fontsize=12)
    axs[0].set_xlabel("rest wavelength [Å]")
    for ax, (k1, k2) in zip(axs[1:], [("SNII/sn1987A.p0044.dat", "SNII/sn1987A.p0070.dat"),
                                       ("SNII/sn2005cs.p0060.dat", "SNII/sn2005cs.p0061.dat")]):
        for k, ls, lab in ((k1, "-", k1.split("/")[1]), (k2, "--", k2.split("/")[1])):
            c, n = k.split("/")
            tt = Template(path=os.path.join(bank, c, n), folder=c, name="", phase=0).load()
            norm = np.median(tt.flux[(tt.wav > 5000) & (tt.wav < 7000)])
            ax.plot(tt.wav, tt.flux / norm, ls=ls, lw=0.9, label=lab)
        row = dups[(dups.key1 == k1) & (dups.key2 == k2)]
        cc = float(row["corr"].iloc[0]) if len(row) else np.nan
        ax.set_title(f"duplicate spectra: correlation r = {cc:.5f}", fontsize=12)
        ax.set_xlim(3500, 9000)
        ax.legend()
        ax.set_xlabel("rest wavelength [Å]")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--meas", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--dups", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    m = pd.read_csv(a.meas)
    c = pd.read_csv(a.cmp)
    corrected_figures(m, a.bank, os.path.join(a.outdir, "corrected"))
    comparison_figure(c, os.path.join(a.outdir, "student_vs_thiswork.png"))
    window_sensitivity_figure(a.bank, os.path.join(a.outdir, "window_sensitivity.png"))
    diagnostics_figure(a.bank, os.path.join(a.outdir, "continuum_diagnostics.png"))
    defects_figure(a.bank, pd.read_csv(a.dups), os.path.join(a.outdir, "template_defects.png"))
    print("figures written to", a.outdir)


if __name__ == "__main__":
    main()
