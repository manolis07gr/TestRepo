"""Reading the Duperfit (Baer 2024) template bank.

Template files live in ``<bank>/<class>/sn<NAME>.<phase>.dat`` with columns
``wav flux eflux`` (rest frame, arbitrary flux units).  The phase token is
``pNNNN`` (+NNNN d), ``mNNNN`` (-NNNN d) or ``max`` (0 d), exactly as parsed by
Duperfit's own ``temp_handling.py``.
"""
import glob
import os
from dataclasses import dataclass, field

import numpy as np

# Classes plotted by the student.  His "SLSN-II" figures also contain every
# object of Duperfit's separate SLSN-IIn folder (2006gy and five PTF10 SNe).
FIGURE_CLASSES = {
    "SNIIn": ["SNIIn"],
    "SNII": ["SNII"],
    "SNIIb": ["SNIIb"],
    "SNIa-CSM": ["SNIa-CSM"],
    "SLSN-II": ["SLSN-II", "SLSN-IIn"],
}


def parse_phase(token):
    if token == "max":
        return 0.0
    sign = -1.0 if token[0] == "m" else 1.0
    return sign * float(token[1:])


@dataclass
class Template:
    path: str
    folder: str          # Duperfit folder (SNII, SLSN-IIn, ...)
    name: str            # SN name without the leading "sn"
    phase: float         # days, as encoded in the file name
    wav: np.ndarray = field(repr=False, default=None)
    flux: np.ndarray = field(repr=False, default=None)
    eflux: np.ndarray = field(repr=False, default=None)

    @property
    def key(self):
        return f"{self.folder}/{os.path.basename(self.path)}"

    def load(self):
        if self.wav is None:
            data = np.loadtxt(self.path)
            order = np.argsort(data[:, 0])
            data = data[order]
            good = np.all(np.isfinite(data[:, :2]), axis=1)
            data = data[good]
            self.wav, self.flux = data[:, 0], data[:, 1]
            self.eflux = data[:, 2] if data.shape[1] > 2 else None
        return self


def list_templates(bank, folders, phase_range=None, load=True):
    """Return templates of the given Duperfit folders, sorted by (name, phase)."""
    out = []
    for folder in folders:
        for path in glob.glob(os.path.join(bank, folder, "*.dat")):
            base = os.path.basename(path)
            parts = base.split(".")
            name = parts[0][2:] if parts[0].startswith("sn") else parts[0]
            phase = parse_phase(parts[-2])
            if phase_range and not (phase_range[0] <= phase <= phase_range[1]):
                continue
            t = Template(path=path, folder=folder, name=name, phase=phase)
            out.append(t.load() if load else t)
    out.sort(key=lambda t: (t.name, t.phase, t.path))
    return out


def student_label(name):
    """Legend label used in the student's figures.

    Both OGLE-2013-SN-016 and OGLE-2013-SN-019 appear under a single legend
    entry "OGLE-2013"; every other name is kept as is.
    """
    return "OGLE-2013" if name.startswith("OGLE-2013") else name


# Marker/colour cycle reproduced from the student's legends: labels are sorted
# (Python string order) and assigned 8 marker shapes per colour.
STUDENT_MARKERS = ["o", "v", "^", "<", ">", "d", "h", "*"]
STUDENT_COLORS = ["r", "b", "g", "y", "m", "k"]


def student_style(labels):
    labels = sorted(set(labels))
    return {lab: (STUDENT_COLORS[i // 8], STUDENT_MARKERS[i % 8]) for i, lab in enumerate(labels)}
