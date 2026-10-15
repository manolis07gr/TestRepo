#!/usr/bin/env python
"""SuperLite model tracks against the observed Hα EW and Hα/Hβ of each SN class.

One figure per class and quantity (figures in <outdir>/EW_<class>.png, R_<class>.png): rows are
the model series (A: RSG, B: YSG), columns the three explosion energies, colour the CSM mass
(dashed grey: no CSM).  Behind the models, in grey, the observed objects of that class
(object-nights, and per phase bin the median and interquartile range over objects), from
results/observed_results_points.csv (results_data.py; reliable measurements only).

Model input, either
  --mmeas results/model_measurements.csv        (measure_models.py: robust method; only clean
                                                 EWs and valid Balmer decrements are drawn), or
  --published results/published_model_lines.csv --catalogue <xlsx>
                                                (values from the paper's Tables A1/A2, NLTE rows)

The C series (no hydrogen) is not drawn; it is listed in the measurement table.

Usage:
    python -I scripts/make_model_figures.py --mmeas results/model_measurements.csv \
        --obs results/observed_results_points.csv --outdir figures/models
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CLASSES = ["SNIIn", "SLSN-II", "SNIa-CSM", "SNII", "SNIIb"]
LABEL = {"SNIIn": "SNe IIn", "SLSN-II": "SLSNe II", "SNIa-CSM": "SNe Ia-CSM", "SNII": "SNe II",
         "SNIIb": "SNe IIb"}
SERIES_LABEL = {"A": "A series: RSG, 17.8 M$_\\odot$, 603 R$_\\odot$",
                "B": "B series: YSG, 11.6 M$_\\odot$, 525 R$_\\odot$",
                "X": "X series (test models)"}
BINS = [-20, 0, 20, 40, 70, 100, 150]
CASE_B = 2.86
# ordinal blue ramp (reference palette steps 300/400/550/700) for CSM mass; no CSM in grey
RAMP = ["#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d9d8d2"
OBS_DOT = "#b4b2ab"


def standardize_measured(m):
    """Columns used by the figures from the measure_models.py table."""
    d = m[m.get("load_error", pd.Series(index=m.index, dtype=object)).isna()].copy()
    d["ew"] = -d.rob_EW_net_Ha
    d["ew_lo"], d["ew_hi"] = -d.rob_EW_net_Ha_hi, -d.rob_EW_net_Ha_lo
    d["ew_ok"] = d.rob_ew_clean.astype(bool)
    d["r"], d["r_lo"], d["r_hi"] = d.rob_R_net, d.rob_R_net_lo, d.rob_R_net_hi
    d["r_ok"] = d.rob_ratio_valid.astype(bool)
    d["source"] = "this work (robust method)"
    return d


def standardize_published(p, cat):
    d = p[p.treatment == "NLTE"].merge(cat, on="model", how="left")
    d["ew"] = -d.EW_Ha
    d["ew_lo"] = d["ew_hi"] = np.nan
    d["ew_ok"] = d.EW_Ha < 0
    d["r_lo"] = d["r_hi"] = np.nan
    d["r"] = d.R
    d["r_ok"] = d.R > 0
    d["source"] = "published values (Wagle et al. 2025, Tables A1-A2, NLTE; specutils recipe)"
    return d


def observed_bins(obs, cls, col, min_obj=3):
    s = obs[(obs.figure == cls) & np.isfinite(obs[col]) & (obs[col] > 0)]
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        per = s[(s.phase >= lo) & (s.phase < hi)].groupby("name")[col].median()
        if len(per) < min_obj:
            continue
        lv = np.log10(per.values)
        out.append(dict(mid=0.5 * (lo + hi), lo=lo, hi=hi, n=len(per), med=10 ** np.median(lv),
                        q1=10 ** np.percentile(lv, 25), q3=10 ** np.percentile(lv, 75),
                        vmin=per.min(), vmax=per.max()))
    return s, pd.DataFrame(out)


def mcsm_style(values):
    pos = sorted(v for v in set(values) if v > 0)
    idx = np.linspace(0, len(RAMP) - 1, len(pos)).round().astype(int) if len(pos) <= len(RAMP) else None
    style = {0.0: dict(color=INK2, ls=(0, (4, 2)), mfc="white", label="no CSM")}
    for i, v in enumerate(pos):
        c = RAMP[idx[i]] if idx is not None else plt.cm.Blues(0.35 + 0.6 * i / max(len(pos) - 1, 1))
        style[v] = dict(color=c, ls="-", mfc=c, label=f"{v:g} M$_\\odot$")
    return style


def _log_labels(ax):
    """Plain-number labels on a log y axis; minor ticks labelled at 2 and 5 only."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    def minor(v, _):
        m = v / 10 ** np.floor(np.log10(v))
        return f"{v:g}" if round(m) in (2, 5) else ""
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(FuncFormatter(minor))


def _axes_style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9, which="both")


def class_figure(models, obs, cls, q, out, source_note):
    col = "ew" if q == "EW" else "r"
    ok = models[f"{col}_ok"] & np.isfinite(models[col]) & (models[col] > 0)
    s, b = observed_bins(obs, cls, col)
    series = [x for x in ("A", "B", "X") if (models.series == x).any()]
    if not series:
        return False
    style = mcsm_style(models.m_csm.fillna(0).values)
    fig, axes = plt.subplots(len(series), 3, figsize=(13.5, 3.5 * len(series) + 1.4), sharex=True, sharey=True,
                             squeeze=False)
    yvals = list(models.loc[ok, col]) + list(s[col]) + (list(b.q1) + list(b.q3) if len(b) else [])
    ylo, yhi = (np.nanmin(yvals), np.nanmax(yvals)) if yvals else (1, 100)
    if q == "R":
        ylo, yhi = min(ylo, 1.0), max(yhi, 10.0)
    for i, ser in enumerate(series):
        ms = models[models.series == ser]
        energies = sorted(ms.e51.dropna().unique())[:3]
        for j in range(3):
            ax = axes[i, j]
            _axes_style(ax)
            ax.scatter(s.phase, s[col], s=9, c=OBS_DOT, lw=0, alpha=0.6, zorder=1)
            if len(b):
                ax.fill_between(b.mid, b.q1, b.q3, color=BAND, alpha=0.7, lw=0, zorder=1.5)
                ax.plot(b.mid, b.med, c=INK2, lw=1.6, zorder=2)
            if q == "R":
                ax.axhline(CASE_B, c=MUTED, ls=":", lw=1, zorder=1.6)
            if j >= len(energies):
                ax.set_visible(False)
                continue
            e = energies[j]
            me = ms[np.isclose(ms.e51, e)]
            for mc, g in sorted(me.groupby(me.m_csm.fillna(0)), key=lambda t: t[0]):
                st = style.get(mc, style[0.0])
                g = g.sort_values("epoch")
                gv = g[ok.loc[g.index]]
                # line segments only between consecutive measured epochs
                seg = (gv.epoch.values, gv[col].values)
                ax.plot(*seg, color=st["color"], ls=st["ls"], lw=2, zorder=3)
                ax.plot(*seg, ls="none", marker="o", ms=5.5, mfc=st["mfc"], mec=st["color"], mew=1.4, zorder=4)
                ax.plot(*seg, ls="none", marker="o", ms=8, mfc="none", mec="white", mew=1.2, zorder=3.5)
            ax.set_title(f"{ser}: E = {e:g}×10$^{{51}}$ erg", fontsize=10, color=INK, loc="left")
            ax.set_yscale("log")
            _log_labels(ax)
            ax.set_xlim(-20, 152)
            ax.set_ylim(ylo / 1.4, yhi * 1.4)
            if i == len(series) - 1:
                ax.set_xlabel("days since maximum", fontsize=10, color=INK2)
            if j == 0:
                ax.set_ylabel(("Hα emission EW [Å]" if q == "EW" else "F(Hα)/F(Hβ)") + f"\n{ser} series",
                              fontsize=10, color=INK2)
    what = "Hα equivalent width" if q == "EW" else "Balmer decrement F(Hα)/F(Hβ)"
    fig.suptitle(f"{what}: SuperLite models vs observed {LABEL[cls]}", x=0.01, ha="left", fontsize=13,
                 color=INK, y=0.995)
    nobj = s.name.nunique()
    fig.text(0.01, 0.955, f"Models: {source_note}.  Observed {LABEL[cls]}: {nobj} objects, {len(s)} object-nights "
             f"(grey dots; days since optical maximum, or discovery where no maximum is catalogued); "
             f"line and band: median and IQR over objects per phase bin (≥3 objects)."
             + ("  Dotted: Case B (2.86)." if q == "R" else ""), fontsize=8.5, color=INK2, ha="left", va="top",
             wrap=True)
    present = set(models.m_csm.fillna(0).values)
    handles = [Line2D([], [], color=st["color"], ls=st["ls"], lw=2, marker="o", mfc=st["mfc"], mec=st["color"],
                      label=st["label"]) for k, st in sorted(style.items()) if k in present]
    handles += [Line2D([], [], ls="none", marker="o", ms=4, mfc=OBS_DOT, mec="none", label=f"{LABEL[cls]} object-nights"),
                Patch(fc=BAND, ec="none", label="observed IQR"), Line2D([], [], c=INK2, lw=1.6, label="observed median")]
    fig.legend(handles=handles, ncol=len(handles), loc="upper left", bbox_to_anchor=(0.01, 0.93), frameon=False,
               fontsize=8.5, title="CSM mass", title_fontsize=8.5, alignment="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmeas", help="model_measurements.csv from measure_models.py")
    ap.add_argument("--published", help="published_model_lines.csv (paper Tables A1/A2)")
    ap.add_argument("--catalogue", help="model spreadsheet (.xlsx); needed with --published")
    ap.add_argument("--obs", required=True, help="observed_results_points.csv from results_data.py")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    if a.mmeas:
        models = standardize_measured(pd.read_csv(a.mmeas))
        note = "this work (robust method; clean EWs and valid decrements only), days since bolometric maximum"
    elif a.published:
        from bv.models import read_catalogue
        models = standardize_published(pd.read_csv(a.published), read_catalogue(a.catalogue))
        note = ("published values (Wagle et al. 2025, Tables A1-A2, NLTE rows; specutils recipe, not re-measured), "
                "days since bolometric maximum")
    else:
        sys.exit("give --mmeas or --published")
    models = models[models.series.isin(["A", "B", "X"])]
    obs = pd.read_csv(a.obs)
    os.makedirs(a.outdir, exist_ok=True)
    for cls in CLASSES:
        for q in ("EW", "R"):
            if class_figure(models, obs, cls, q, os.path.join(a.outdir, f"{q}_{cls}.png"), note):
                print("wrote", os.path.join(a.outdir, f"{q}_{cls}.png"))


if __name__ == "__main__":
    main()
