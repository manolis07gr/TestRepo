# Validation of Hα EW and Hα/Hβ measurements on Duperfit SN templates

Independent re-measurement of the Hα equivalent widths and Hα/Hβ flux ratios
shown in the ten figures under review ("Hα equivalent widths / Hα to Hβ ratios -
Duperfit SNIIn, SNII, SNIIb, SNIa-CSM, SLSN-II templates"), using

1. the same Duperfit template bank (Baer 2024, `github.com/baerrito/duperfit`, v0.1.x), and
2. the original public spectra of the same supernovae from the Open Supernova
   Catalog (per-object JSON in `github.com/astrocatalogs/sne-*`) and WISeREP
   (public spectra, guest access).

The findings are in **[REPORT.md](REPORT.md)**.

## Layout

```
bv/                      analysis package
  templates.py           Duperfit bank reader (phase parsing as in Duperfit), student's legend style
  measure.py             replica of the student-style specutils method + robust method (MC errors, flags)
  qa.py                  template defects: corrupt pixels, duplicates, wavelength units
  digitize.py            digitizer for the student's symlog scatter plots + template matching
  archive.py             OSC JSON / WISeREP parsing
  match.py               template <-> archive spectrum identification (feature correlation)
  plotting.py            student-style and corrected figures, continuum diagnostics
scripts/                 command-line steps (see below)
results/                 CSV outputs (one row per template / archive spectrum / digitized point)
figures/student/         the ten figures under review (as received)
figures/corrected/       the same ten figures re-measured on the same templates
figures/archive/         the ten figures reproduced from public spectra + template-vs-archive checks
figures/*.png            validation and diagnostic figures used in the report
data/                    downloaded public spectra (git-ignored; re-created by fetch_archive.py)
```

## Reproducing everything

Requirements: Python ≥ 3.10, `pip install -r requirements.txt`.
Get the template bank: `git clone --depth 1 https://github.com/baerrito/duperfit`.
Run from this directory (`-I` keeps downloaded data from shadowing modules):

```bash
BANK=/path/to/duperfit/tempbank/sne
python -I scripts/measure_templates.py --bank $BANK --out results/templates_measurements.csv \
       --dups results/template_duplicates.csv
python -I scripts/digitize_student.py  --figdir figures/student --bank $BANK --out results/student_digitized.csv
python -I scripts/compare_student.py   --meas results/templates_measurements.csv \
       --dig results/student_digitized.csv --out results/student_vs_thiswork.csv
python -I scripts/make_figures.py      --bank $BANK --meas results/templates_measurements.csv \
       --cmp results/student_vs_thiswork.csv --dups results/template_duplicates.csv --outdir figures
python -I scripts/fetch_archive.py     --bank $BANK --outdir data/archive          # ~20 min, ~250 MB
python -I scripts/measure_archive.py   --bank $BANK --archive data/archive --out results/archive_measurements.csv
python -I scripts/match_templates_archive.py --bank $BANK --archive data/archive \
       --out results/template_archive_matches.csv
python -I scripts/make_archive_figures.py --bank $BANK --ameas results/archive_measurements.csv \
       --matches results/template_archive_matches.csv --tmeas results/templates_measurements.csv \
       --outdir figures/archive
python -I scripts/summarize.py --results results       # numbers quoted in REPORT.md -> results/summary.json
```

## Measurement definitions (this work)

* Sign convention as in the figures under review: EW = ∫(1 − F/F_c) dλ, emission < 0, in Å.
* Continuum: straight line through the median flux in two sidebands at 12,000–16,000 km/s
  on either side of the line; integration over ±10,000 km/s; identical velocity windows for Hα and Hβ.
* Uncertainties: 16–84 % of 300 Monte-Carlo realisations combining flux noise and random
  window/sideband placement (v_int 8,000–12,000 km/s, sideband gap 1,000–3,000 km/s, width 3,000–5,000 km/s).
* Flags (column `rob_flags`): no coverage, corrupt pixels, non-positive continuum, P-Cygni trough reaching
  into the blue sideband, emission feature inside a sideband (e.g. [O I] 6300), absorption > 30 % of the
  line flux, |net flux| < 3σ, net absorption.
* `ew_clean`: significant net emission without any of the above.  `ratio_valid`: both lines clean,
  significant net emission, absorption < 30 %.  Ratios are **not** corrected for host extinction
  (`rob_R_net_mwcorr` in the archive table applies the catalogue Milky-Way E(B−V), CCM89, R_V = 3.1).
* `pc_*` columns: standard SN pseudo-continuum decomposition of the P-Cygni profile into emission and
  absorption parts (first flux maximum blueward of the absorption trough → red continuum), and
  a/e = EW_abs/|EW_em| (Gutiérrez et al. 2014).
