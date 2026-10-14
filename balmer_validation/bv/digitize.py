"""Digitize the student's scatter plots and match each marker to a template.

The figures under review use matplotlib's default symmetric-log scale
(linthresh = 2, linscale = 1, base 10); tick positions recovered from the JPEGs
fit this transform to < 0.5 px.  Markers are found by hue segmentation, merged
blobs are split with k-means, the marker shape is classified against rendered
matplotlib markers, and points are matched to templates (same colour, phase
within a tolerance, shape when available) by optimal assignment.
"""
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

C_ADJ = 1.0 / (1.0 - 1.0 / 10.0)
LINTHRESH = 2.0


def symlog(y):
    y = np.asarray(y, float)
    a = np.abs(y)
    with np.errstate(divide="ignore"):
        out = np.sign(y) * LINTHRESH * (C_ADJ + np.log10(np.maximum(a, 1e-300) / LINTHRESH))
    return np.where(a <= LINTHRESH, y * C_ADJ, out)


def inv_symlog(t):
    t = np.asarray(t, float)
    out = np.sign(t) * LINTHRESH * 10 ** (np.abs(t) / LINTHRESH - C_ADJ)
    return np.where(np.abs(t) <= LINTHRESH * C_ADJ, t / C_ADJ, out)


def _box(gray):
    dark = gray < 100
    cols = [i for i, c in enumerate(dark.sum(0)) if c > 500]
    rows = [i for i, c in enumerate(dark.sum(1)) if c > 500]
    return cols[0], cols[-1], rows[0], rows[-1]


def _ticks(gray, box):
    """Major ticks ('inout', ~8 px each side) on the right spine."""
    left, right, top, bot = box

    def run(y, x, dx):
        n = 0
        while 0 <= x + dx * (n + 1) < gray.shape[1] and gray[y, x + dx * (n + 1)] < 150:
            n += 1
        return n
    sel = [y for y in range(top + 2, bot - 1) if 7 <= run(y, right, 1) <= 12 and 7 <= run(y, right, -1) <= 12]
    groups = []
    for y in sel:
        if groups and y - groups[-1][-1] <= 1:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [float(np.mean(g)) for g in groups]


def _legend(rgb):
    g = rgb.mean(2)
    light = ((np.abs(rgb[:, :, 0] - rgb[:, :, 1]) < 12) & (np.abs(rgb[:, :, 1] - rgb[:, :, 2]) < 12)
             & (g > 180) & (g < 238))
    rows = [i for i, v in enumerate(light.sum(1)) if v > 150 and 62 < i < 900]
    if len(rows) < 2:
        return None
    r0, r1 = min(rows), max(rows)
    cols = [i for i, v in enumerate(light[r0:r1 + 1].sum(0)) if v > 0.6 * (r1 - r0 + 1)]
    if len(cols) < 2:
        return None
    return min(cols), max(cols), r0, r1


def classify_colors(rgb):
    hsv = rgb_to_hsv(rgb / 255.0)
    h, s, v = hsv[:, :, 0] * 360, hsv[:, :, 1], hsv[:, :, 2]
    out = np.full(rgb.shape[:2], "", dtype="<U1")
    col = (s > 0.35) & (v > 0.12)
    out[col & ((h < 20) | (h > 340))] = "r"
    out[col & (h >= 40) & (h < 80)] = "y"
    out[col & (h >= 90) & (h < 170)] = "g"
    out[col & (h >= 200) & (h < 265)] = "b"
    out[col & (h >= 275) & (h < 335)] = "m"
    out[(v < 0.40) & (s < 0.30)] = "k"
    return out


_SHAPES = ["o", "v", "^", "<", ">", "d", "h", "*"]
_PATCH = 8  # half size of comparison patches


def _render_shapes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)
    ax.axis("off")
    pos = {}
    for i, m in enumerate(_SHAPES):
        x, y = 100 + 100 * i, 500
        ax.plot([x], [y], m, mfc="none", mec="k", ms=6, mew=1.0)
        pos[m] = (x, y)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].mean(2)
    plt.close(fig)
    out = {}
    for m, (x, y) in pos.items():
        p = img[y - _PATCH:y + _PATCH + 1, x - _PATCH:x + _PATCH + 1] < 160
        yy, xx = np.where(p)
        cy, cx = (yy.max() + yy.min()) / 2, (xx.max() + xx.min()) / 2   # bbox centre, as for detections
        p = ndimage.shift(p.astype(float), (_PATCH - cy, _PATCH - cx), order=1)
        out[m] = ndimage.gaussian_filter(p, 0.7)
    return out


_SHAPE_TEMPLATES = None


def classify_shape(mask, cx, cy):
    """Best-matching marker shape for the binary colour mask around (cx, cy)."""
    global _SHAPE_TEMPLATES
    if _SHAPE_TEMPLATES is None:
        _SHAPE_TEMPLATES = _render_shapes()
    ix, iy = int(round(cx)), int(round(cy))
    pad = _PATCH + 2
    big = mask[max(iy - pad, 0):iy + pad + 1, max(ix - pad, 0):ix + pad + 1].astype(float)
    if big.shape != (2 * pad + 1, 2 * pad + 1):
        return None, 0.0
    big = ndimage.gaussian_filter(big, 0.7)
    best, best_score = None, -1.0
    for m, tpl in _SHAPE_TEMPLATES.items():
        tz = (tpl - tpl.mean()) / (tpl.std() + 1e-9)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                p = big[2 + dy:2 + dy + 2 * _PATCH + 1, 2 + dx:2 + dx + 2 * _PATCH + 1]
                pz = (p - p.mean()) / (p.std() + 1e-9)
                score = float(np.mean(pz * tz))
                if score > best_score:
                    best, best_score = m, score
    return best, best_score


def digitize(path, tick_values, xlim=(-20.0, 150.0)):
    """Return (points, meta). tick_values: values of the detected major y ticks, top to bottom."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(float)
    gray = rgb.mean(2)
    box = _box(gray)
    left, right, top, bot = box
    ticks = _ticks(gray, box)
    if len(ticks) != len(tick_values):
        raise ValueError(f"{path}: found ticks at {ticks}, expected {len(tick_values)} values")
    T = symlog(tick_values)
    A = np.vstack([np.ones_like(T), -T]).T
    (p0, s), *_ = np.linalg.lstsq(A, np.asarray(ticks), rcond=None)
    resid = np.asarray(ticks) - (p0 - s * T)
    xs = (right - left) / (xlim[1] - xlim[0])
    lg = _legend(rgb.astype(int))
    cls = classify_colors(rgb)
    points = []
    for c in ["r", "b", "g", "y", "m", "k"]:
        m = cls == c
        m[:top + 3, :] = False
        m[bot - 2:, :] = False
        m[:, :left + 3] = False
        m[:, right - 2:] = False
        if c == "k":
            m[:top + 10, :] = False
            m[bot - 9:, :] = False
            m[:, :left + 10] = False
            m[:, right - 9:] = False
        if lg is not None:
            m[lg[2] - 2:lg[3] + 3, lg[0] - 2:lg[1] + 3] = False
        lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=1), structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            yy, xx = np.where((lab == i) & m)
            if len(yy) < 6:
                continue
            h, w = yy.max() - yy.min() + 1, xx.max() - xx.min() + 1
            if h < 5 and w < 5:
                continue
            k = 1
            if max(h, w) > 15 or len(yy) > 70:
                k = max(2, int(max(np.ceil(max(h, w) / 12.5), np.round(len(yy) / 38.0))))
            if k == 1:
                cx, cy = (xx.max() + xx.min()) / 2, (yy.max() + yy.min()) / 2
                shape, score = classify_shape(m, cx, cy)
                points.append(dict(color=c, px=float(cx), py=float(cy), merged=False, shape=shape,
                                   shape_score=score))
            else:
                X = np.vstack([xx, yy]).T.astype(float)
                best = None
                for trial in range(5):
                    cen, labk = kmeans2(X, k, minit="++", seed=1000 + trial)
                    sse = sum(((X[labk == j] - cen[j]) ** 2).sum() for j in range(k))
                    if best is None or sse < best[0]:
                        best = (sse, cen, labk)
                for j in range(k):
                    if (best[2] == j).sum() < 4:
                        continue
                    points.append(dict(color=c, px=float(best[1][j][0]), py=float(best[1][j][1]),
                                       merged=True, shape=None, shape_score=0.0))
    for p in points:
        p["phase"] = xlim[0] + (p["px"] - left) / xs
        p["value"] = float(inv_symlog((p0 - p["py"]) / s))
        p["t"] = float((p0 - p["py"]) / s)  # symlog coordinate
    meta = dict(path=path, box=box, ticks=ticks, p0=float(p0), s=float(s), xscale=float(xs),
                tick_resid_px=[float(r) for r in resid], legend=lg)
    return points, meta


def match(points, entries, tol_days=1.6, shape_penalty=0.6):
    """Optimal assignment of digitized points to template entries.

    entries: list of dicts with keys 'color', 'marker', 'phase' (+ anything else).
    Returns list of (point_index, entry_index, cost).  Cost = |Δphase|/tol plus
    a penalty when a confidently classified marker shape disagrees.
    """
    pairs = []
    for c in sorted({p["color"] for p in points} | {e["color"] for e in entries}):
        pi = [i for i, p in enumerate(points) if p["color"] == c]
        ei = [j for j, e in enumerate(entries) if e["color"] == c]
        if not pi or not ei:
            continue
        big = 1e6
        cost = np.full((len(pi), len(ei)), big)
        for a, i in enumerate(pi):
            p = points[i]
            for b, j in enumerate(ei):
                e = entries[j]
                d = abs(p["phase"] - e["phase"])
                if d > tol_days:
                    continue
                cc = d / tol_days
                if p["shape"] is not None and p["shape_score"] > 0.45 and p["shape"] != e["marker"]:
                    cc += shape_penalty
                cost[a, b] = cc
        r, k = linear_sum_assignment(cost)
        for a, b in zip(r, k):
            if cost[a, b] < big:
                pairs.append((pi[a], ei[b], float(cost[a, b])))
    return pairs
