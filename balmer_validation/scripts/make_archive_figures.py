#!/usr/bin/env python
"""Figures reproduced from public spectra (OSC / WISeREP) and the template-vs-archive checks.

Phases of the archive spectra are measured from the catalogue date of maximum
light when that date is plausible (between 30 d before and 120 d after the
discovery date), otherwise from the discovery date (marked in the CSV).

Usage:
    python -I scripts/make_archive_figures.py --bank /path/to/duperfit/tempbank/sne \
        --ameas results/archive_measurements.csv --matches results/template_archive_matches.csv \
        --tmeas results/templates_measurements.csv --outdir figures/archive
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

from bv.plotting import balmer_figure, student_style_figure  # noqa: E402
from bv.templates import FIGURE_CLASSES, list_templates, student_label  # noqa: E402

CLASSES = ["SNIIn", "SNII", "SNIIb", "SNIa-CSM", "SLSN-II"]


def best_phase(a):
    a = a.copy()
    disc_minus_max = a.phase_disc - a.phase_max      # = (mjd_max - mjd_disc)/(1+z)
    ok_max = np.isfinite(a.phase_max) & (disc_minus_max > -30) & (disc_minus_max < 120)
    a["phase"] = np.where(ok_max, a.phase_max, a.phase_disc)
    a["phase_ref"] = np.where(ok_max, "max", np.where(np.isfinite(a.phase_disc), "discovery", "none"))
    return a


def archive_figures(a, bank, outdir):
    os.makedirs(outdir, exist_ok=True)
    for cls in CLASSES:
        labels_all = [student_label(t.name) for t in list_templates(bank, FIGURE_CLASSES[cls], load=False)]
        s = a[(a.figure == cls) & a.usable.astype(bool) & (a.phase >= -20) & (a.phase <= 150)]
        e = s[s.rob_ew_valid.astype(bool)]
        n_obj = e.name.nunique()
        sub = (f"public spectra (OSC/WISeREP): {len(e)} spectra of {n_obj} SNe; "
               f"{int(e.rob_ew_clean.sum())} clean; phase from max light")
        student_style_figure(e, "rob_EW_net_Ha", f"{cls} (archive)", "EW",
                             os.path.join(outdir, f"EW_{cls}_archive.png"),
                             err_cols=("rob_EW_net_Ha_lo", "rob_EW_net_Ha_hi"), flag_col="rob_ew_clean",
                             subtitle=sub, labels_all=labels_all)
        both = s[s.rob_cov_Ha.astype(bool) & s.rob_cov_Hb.astype(bool)]
        # OSC 'Uncalibrated' = no absolute flux scale (e.g. CfA spectra are relatively calibrated);
        # kept, the flag is recorded in the CSV
        r = both[both.rob_ratio_valid.astype(bool)]
        balmer_figure(r, f"{cls} (archive)", labels_all, len(both),
                      os.path.join(outdir, f"R_{cls}_archive.png"), source="public spectra")


def template_archive_figure(mt, tm, am, out):
    """Phase zero-points, EW and Hα/Hβ: template vs. the archive spectrum it was made from."""
    g = mt[mt["corr"] > 0.95].copy()
    g = g.merge(tm[["key", "rob_EW_net_Ha", "rob_R_net", "rob_ew_clean", "rob_ratio_valid", "figure"]], on="key")
    am = am.rename(columns={c: "a_" + c for c in am.columns if c.startswith("rob_")})
    g = g.merge(am[["name", "idx", "a_rob_EW_net_Ha", "a_rob_R_net", "a_rob_ew_clean", "a_rob_ratio_valid"]],
                left_on=["name", "a_idx"], right_on=["name", "idx"], how="left")
    fig, axs = plt.subplots(1, 4, figsize=(26, 6.5))
    # (a) phase zero-points: offset w.r.t. discovery vs. w.r.t. catalogue maximum, per SN
    g["off_max"] = g.t_phase - g.a_phase_max
    g["off_disc"] = g.t_phase - g.a_phase_disc
    per = g.groupby(["figure", "name"]).agg(n=("key", "size"), off_max=("off_max", "median"),
                                             off_disc=("off_disc", "median")).reset_index()
    for k, cls in enumerate(CLASSES):
        q = per[per.figure == cls]
        axs[0].scatter(q.off_disc, q.off_max, s=30, label=cls, color=f"C{k}")
        for _, r in q.iterrows():
            if np.isfinite(r.off_max) and np.isfinite(r.off_disc) and min(abs(r.off_max), abs(r.off_disc)) > 5:
                axs[0].annotate(r["name"], (r.off_disc, r.off_max), fontsize=7, xytext=(3, 2),
                                textcoords="offset points", color=f"C{k}")
    axs[0].axhline(0, c="k", lw=0.8)
    axs[0].axvline(0, c="k", lw=0.8)
    axs[0].set_xscale("symlog", linthresh=10)
    axs[0].set_yscale("symlog", linthresh=10)
    axs[0].set_xlabel("template phase − days since discovery")
    axs[0].set_ylabel("template phase − days since catalogue maximum")
    axs[0].set_title("(a) template phase zero-points (median per SN)\n"
                     "on y=0: from maximum; on x=0: from discovery")
    axs[0].legend(fontsize=8, loc="lower right")
    # (b) EW
    q = g[np.isfinite(g.rob_EW_net_Ha) & np.isfinite(g.a_rob_EW_net_Ha)]
    cl = q.rob_ew_clean.astype(bool) & q.a_rob_ew_clean.astype(bool)
    axs[1].scatter(-q.a_rob_EW_net_Ha[~cl], -q.rob_EW_net_Ha[~cl], s=10, c="0.75", label="flagged in either")
    axs[1].scatter(-q.a_rob_EW_net_Ha[cl], -q.rob_EW_net_Ha[cl], s=14, c="C0", label="clean in both")
    lim = (1, 2e4)
    axs[1].plot(lim, lim, "k--", lw=0.8)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlim(lim)
    axs[1].set_ylim(lim)
    d = np.log10(q.rob_EW_net_Ha[cl] / q.a_rob_EW_net_Ha[cl])
    axs[1].set_title(f"(b) −EW(Hα): template vs. its archive spectrum\nclean pairs: {cl.sum()}, "
                     f"median |Δ| = {np.nanmedian(np.abs(d)):.3f} dex")
    axs[1].set_xlabel("archive spectrum  [Å]")
    axs[1].set_ylabel("Duperfit template  [Å]")
    axs[1].legend()
    # (c) ratio
    q = g[np.isfinite(g.rob_R_net) & np.isfinite(g.a_rob_R_net)]
    cl = q.rob_ratio_valid.astype(bool) & q.a_rob_ratio_valid.astype(bool)
    axs[2].scatter(q.a_rob_R_net[cl], q.rob_R_net[cl], s=16, c="C0", label="valid in both")
    lim = (0.3, 100)
    axs[2].plot(lim, lim, "k--", lw=0.8)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlim(lim)
    axs[2].set_ylim(lim)
    d = np.log10(q.rob_R_net[cl] / q.a_rob_R_net[cl])
    axs[2].set_title(f"(c) Hα/Hβ: template vs. its archive spectrum\nvalid pairs: {cl.sum()}, "
                     f"median Δ = {np.nanmedian(d):+.3f} dex, scatter {np.nanstd(d):.3f} dex")
    axs[2].set_xlabel("archive spectrum")
    axs[2].set_ylabel("Duperfit template")
    # (d) relative calibration Hβ/Hα (template / archive)
    v = g.hb_ha_calib_ratio[np.isfinite(g.hb_ha_calib_ratio)]
    axs[3].hist(np.log10(v), bins=np.linspace(-0.6, 0.6, 49), color="C1")
    axs[3].axvline(0, c="k", lw=0.8)
    axs[3].set_xlabel("log10 [ (F_tpl/F_arch) at Hβ / (F_tpl/F_arch) at Hα ]")
    axs[3].set_title(f"(d) template vs. archive relative flux calibration\n"
                     f"N={len(v)}, |Δ|>0.1 dex for {np.mean(np.abs(np.log10(v)) > 0.1):.0%}")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return g, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ameas", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--tmeas", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    am = best_phase(pd.read_csv(a.ameas))
    am.to_csv(a.ameas.replace(".csv", "_phased.csv"), index=False)
    archive_figures(am, a.bank, a.outdir)
    mt = pd.read_csv(a.matches)
    tm = pd.read_csv(a.tmeas).drop_duplicates("key")
    g, per = template_archive_figure(mt, tm, pd.read_csv(a.ameas), os.path.join(a.outdir, "template_vs_archive.png"))
    g.to_csv(os.path.join(os.path.dirname(a.matches), "template_archive_pairs.csv"), index=False)
    per.to_csv(os.path.join(os.path.dirname(a.matches), "template_phase_offsets.csv"), index=False)
    print("figures written to", a.outdir)


if __name__ == "__main__":
    main()
