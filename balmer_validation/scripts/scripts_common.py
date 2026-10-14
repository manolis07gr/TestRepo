"""Helpers shared by the scripts."""
import numpy as np


def rest_frame(sp, z):
    """Sorted, finite, strictly increasing rest-frame spectrum from an archive record."""
    w = np.asarray(sp["wav"], float)
    f = np.asarray(sp["flux"], float)
    good = np.isfinite(w) & np.isfinite(f)
    w, f = w[good], f[good]
    o = np.argsort(w)
    w, f = w[o], f[o]
    keep = np.concatenate([[True], np.diff(w) > 0]) if len(w) else np.array([], bool)
    w, f = w[keep], f[keep]
    if not sp.get("deredshifted") and z:
        w = w / (1.0 + z)
    return w, f
