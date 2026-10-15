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
  models.py              SuperLite model spectra: spreadsheet catalogue, file discovery, unit-safe loading
scripts/                 command-line steps (see below)
results/                 CSV outputs (one row per template / archive spectrum / digitized point)
figures/student/         the ten figures under review (as received)
figures/corrected/       the same ten figures re-measured on the same templates
figures/archive/         the ten figures reproduced from public spectra + template-vs-archive checks
figures/*.png            validation and diagnostic figures used in the report
figures/models_published/  SuperLite models (published line values) vs the observed classes (preliminary)
report_page/             report page template and built page (Validation and Results tabs)
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
python -I scripts/results_data.py --ameas results/archive_measurements_phased.csv --out results/observed_results.json
python -I scripts/page_data.py --results results --out results/results_page_data.json
python -I scripts/build_page.py --template report_page/page.template.html \
       --data results/results_page_data.json --out report_page/balmer_audit.html
```

`results_data.py` computes the per-class trends of the observed sample (Section 5 of the report);
`report_page/balmer_audit.html` is the report page (Validation and Results tabs; it expects the figures
under `fig/` next to it, as published).

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

## SuperLite models (A/B/C series) vs the observed classes

Model spectra and everything derived from them stay out of this (public) repository: the paths below
are git-ignored.  The report page gets a third tab ("Models") only when it is built with `--models`.

```bash
# on the computer that holds the spectra, if they cannot be copied as they are:
python3 scripts/pack_model_spectra.py ~/Downloads/NLTE        # -> NLTE_optical.xlsx (3000-10000 Å, raw values)
python3 -I scripts/pack_model_spectra.py --unpack NLTE_optical.xlsx --outdir data/models/NLTE

python -I scripts/measure_models.py --models data/models/NLTE \
       --catalogue MESA_STELLA_CSM_INTERACTION_MODELS.xlsx \
       --out results/model_measurements.csv --qa-dir figures/models/qa
python -I scripts/results_data.py --ameas results/archive_measurements_phased.csv \
       --out results/observed_results_masked.json --prefix robm_
python -I scripts/model_results.py --mmeas results/model_measurements.csv \
       --obs results/observed_results.json --obs-masked results/observed_results_masked.json \
       --published results/published_model_lines.csv --out results/model_results.json
python -I scripts/make_model_figures.py --mmeas results/model_measurements.csv --variant strict \
       --obs results/observed_results_points.csv --outdir figures/models/strict
python -I scripts/make_model_figures.py --mmeas results/model_measurements.csv --variant masked \
       --obs results/observed_results_masked_points.csv --outdir figures/models/masked
python -I scripts/page_data.py --results results --out <private dir>/page_data_models.json \
       --models results/model_results.json
python -I scripts/build_page.py --template report_page/page.template.html \
       --data <private dir>/page_data_models.json --out <private dir>/balmer_audit.html
```

* Input: one directory per model (`A4/`, `B12/`, …, at any depth), one text file per epoch with wavelength
  and flux columns.  The model is read from the directory name, the epoch from the file name (`20d`, `t20`,
  `day_20`, or the only number in the name).  The wavelength unit is inferred from the values (Å, nm, µm,
  cm, m, or frequency in Hz).  Flux is taken as F_λ unless the abscissa is a frequency or a header line says
  "per Hz"; `--fnu yes|no` forces it, `--flux-col` picks another column.  Line ratios depend on this choice
  (F_ν integrated over λ would bias Hα/Hβ by (6563/4861)² ≈ 1.8); the IR continuum slope is a quick check
  (F_λ of a 5000–10,000 K photosphere falls as λ^−3.1…−3.6 at 1.5–3 µm).
* Epochs are checked against the spreadsheet's epoch lists (days since the bolometric maximum); when they
  only match after subtracting `day_Lbol_max`, the files count from explosion (`--epoch-ref` forces either).
* Measurements, identical to those of the observed spectra (200 Monte-Carlo draws; the per-pixel noise of
  the Monte-Carlo spectra is estimated from their own pixel-to-pixel scatter): `rob_*` as measured,
  `robm_*` with the narrow non-Balmer lines masked (`bv.measure.NARROW_CONTAMINANTS`, ±1000 km/s; the
  archive measurement carries the same `robm_*` columns), `robn_*` with the narrow windows, and `rep_*`
  the paper's specutils recipe (if specutils is installed) for comparison with its Tables A1–A2.
* `model_results.py`: per model spectrum both variants with the first failing quality criterion; each
  passing value placed against the observed objects of each class in the same phase bin; validity counts
  per model group; the share of the window emission carried by the masked lines; the check against the
  paper's tables.
* `scripts/published_model_lines.py` extracts Tables A1–A2 of Wagle, Chatzopoulos & Baer (2025) into
  `results/published_model_lines.csv` (rows whose printed ratio disagrees with the printed fluxes are
  flagged); `make_model_figures.py --published results/published_model_lines.csv --catalogue <xlsx>` draws
  them (`figures/models_published/`, NLTE rows; not re-measured).
