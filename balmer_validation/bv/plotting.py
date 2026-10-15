"""Figures: the student's style (symlog scatter), corrected versions, and diagnostics."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .measure import NOMINAL, pseudo_continuum_decomposition, replica_continuum, v2w  # noqa: E402
from .templates import student_style  # noqa: E402

TITLES = {"EW": "Hα equivalent widths", "R": "Hα to Hβ ratios"}


def _style_axes(ax):
    ax.tick_params(which="both", direction="inout", top=True, right=True)
    ax.minorticks_on()


def student_style_figure(df, value_col, cls_label, quantity, out, err_cols=None, flag_col=None,
                         subtitle=None, xlim=(-20, 150), labels_all=None, phase_col="phase"):
    """Scatter in the student's style: symlog y (linthresh=2), colours/markers by SN.

    df must have columns 'label', phase_col and value_col.  If flag_col is given,
    rows with flag False are drawn in light grey (unreliable measurement).
    """
    labels_all = sorted(set(labels_all if labels_all is not None else df["label"]))
    style = student_style(labels_all)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    d = df[np.isfinite(df[value_col].astype(float))]
    for lab in labels_all:
        s = d[d["label"] == lab]
        if s.empty:
            continue
        c, m = style[lab]
        good = s[flag_col].astype(bool) if flag_col else np.ones(len(s), bool)
        for sel, kw in ((good, dict(mec=c, mfc="none", alpha=1.0, zorder=3)),
                        (~good, dict(mec="0.7", mfc="none", alpha=0.9, zorder=2))):
            ss = s[sel]
            if ss.empty:
                continue
            x, y = ss[phase_col].values, ss[value_col].values.astype(float)
            if err_cols:
                lo = ss[err_cols[0]].values.astype(float)
                hi = ss[err_cols[1]].values.astype(float)
                yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
                yerr = np.where(np.isfinite(yerr), yerr, 0)
                ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=kw["mec"], elinewidth=0.6, alpha=0.6,
                            zorder=kw["zorder"] - 1)
            ax.plot(x, y, m, ls="none", ms=6, mew=1.0, **kw)
    ax.set_yscale("symlog", linthresh=2)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Light-curve phase [days]", fontsize=18)
    ax.set_ylabel("EW[Hα] [Å]" if quantity == "EW" else "Hα/Hβ", fontsize=18)
    ax.tick_params(labelsize=16)
    _style_axes(ax)
    ttl = f"{TITLES[quantity]} - {cls_label}"
    ax.set_title(ttl if not subtitle else f"{ttl}\n{subtitle}", fontsize=18 if not subtitle else 15)
    handles = [Line2D([], [], ls="none", marker=style[l][1], mec=style[l][0], mfc="none", label=l)
               for l in labels_all]
    if flag_col:
        handles.append(Line2D([], [], ls="none", marker="o", mec="0.7", mfc="none", label="unreliable"))
    ax.legend(handles=handles, ncol=3, fontsize=9, loc="best", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def continuum_diagnostic(ax, w, f, lam0, title, show_replica=True, label="template"):
    """Plot a spectrum around one line with the three continuum definitions."""
    lo, hi = v2w(-26000, lam0), v2w(22000, lam0)
    m = (w >= lo) & (w <= hi)
    ax.plot(w[m], f[m], c="k", lw=0.8, label=label)
    # fixed sidebands (this work, method C1)
    b = (v2w(-NOMINAL["v_out"], lam0), v2w(-NOMINAL["v_in"], lam0))
    r = (v2w(NOMINAL["v_in"], lam0), v2w(NOMINAL["v_out"], lam0))
    for a, z in (b, r):
        ax.axvspan(a, z, color="C0", alpha=0.15)
    mb = (w >= b[0]) & (w <= b[1])
    mr = (w >= r[0]) & (w <= r[1])
    if mb.sum() > 2 and mr.sum() > 2:
        xb, yb, xr, yr = np.mean(w[mb]), np.median(f[mb]), np.mean(w[mr]), np.median(f[mr])
        xx = np.array([b[0], r[1]])
        ax.plot(xx, yb + (yr - yb) / (xr - xb) * (xx - xb), c="C0", lw=1.5, label="sidebands (C1)")
    ax.axvspan(v2w(-NOMINAL["v_int"], lam0), v2w(NOMINAL["v_int"], lam0), color="C2", alpha=0.07)
    # pseudo-continuum (method C2)
    pc = pseudo_continuum_decomposition(w, f, lam0)
    if pc is not None:
        ax.plot([pc["lam_b"], pc["lam_r"]], [pc["y_b"], pc["y_r"]], c="C1", lw=1.5, ls="--",
                label="pseudo-continuum (C2)")
    if show_replica:
        cont = replica_continuum(w, f, exclude_hw=100.0)
        if cont is not None:
            mm = m & (w >= 4000) & (w <= 7000)
            ax.plot(w[mm], cont(w[mm]), c="C3", lw=1.2, ls=":", label="global Cheb-3 (student-style)")
    ax.axvline(lam0, c="0.5", lw=0.5)
    ax.set_title(title, fontsize=10)
    fin = f[m][np.isfinite(f[m])]
    if fin.size:
        lo_, hi_ = np.percentile(fin, [1, 99.5])
        pad = 0.15 * (hi_ - lo_)
        ax.set_ylim(min(lo_ - pad, 0), hi_ + pad)
    ax.set_xlim(lo, hi)


CASE_B = 2.86


def balmer_figure(r, title_label, labels_all, n_cov, out, source="spectra"):
    """Valid Balmer decrements only, log axis, MC error bars, Case B reference."""
    style = student_style(labels_all)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    phase_col = "phase"
    for lab in sorted(set(r.label)):
        q = r[r.label == lab]
        c, mk = style[lab]
        y = q.rob_R_net.values.astype(float)
        yerr = np.vstack([np.clip(y - q.rob_R_net_lo.values.astype(float), 0, None),
                          np.clip(q.rob_R_net_hi.values.astype(float) - y, 0, None)])
        yerr = np.where(np.isfinite(yerr), yerr, 0)
        ax.errorbar(q[phase_col], y, yerr=yerr, fmt=mk, mfc="none", mec=c, ecolor=c, ms=7, mew=1.2,
                    elinewidth=0.7)
    ax.axhline(CASE_B, c="0.4", ls="--", lw=1)
    ax.text(148, CASE_B * 1.04, "Case B (2.86)", ha="right", va="bottom", color="0.4", fontsize=12)
    ax.set_yscale("log")
    lo, hi = 0.3, 100.0
    if len(r):
        lo = min(lo, 0.7 * np.nanmin(np.minimum(r.rob_R_net_lo.values.astype(float), r.rob_R_net.values.astype(float))))
        hi = max(hi, 1.3 * np.nanmax(np.maximum(r.rob_R_net_hi.values.astype(float), r.rob_R_net.values.astype(float))))
    ax.set_ylim(lo, hi)
    ax.set_xlim(-20, 150)
    ax.set_xlabel("Light-curve phase [days]", fontsize=18)
    ax.set_ylabel("F(Hα)/F(Hβ)  (not dereddened)", fontsize=18)
    ax.tick_params(labelsize=16, which="both", direction="inout", top=True, right=True)
    ax.set_title(f"Hα to Hβ ratios - {title_label}\n"
                 f"valid Balmer decrements only: {len(r)} of {n_cov} {source} covering both lines", fontsize=15)
    handles = [Line2D([], [], ls="none", marker=style[lab][1], mec=style[lab][0], mfc="none", label=lab)
               for lab in sorted(set(r.label))]
    if handles:
        ax.legend(handles=handles, ncol=3, fontsize=10, loc="upper left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
