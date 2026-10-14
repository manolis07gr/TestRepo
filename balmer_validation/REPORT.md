# Validation of the Hα equivalent-width and Hα/Hβ figures (Duperfit templates)

*Independent re-measurement of the ten figures "Hα equivalent widths / Hα to Hβ ratios — Duperfit
SNIIn, SNII, SNIIb, SNIa-CSM, SLSN-II templates", on the same templates and on the original public
spectra (Open Supernova Catalog, WISeREP).  All numbers below are in `results/summary.json`.*

## Bottom line

* **Hα EWs of the interacting classes are reproduced.** For well-defined measurements the student's
  values agree with mine to a median of **0.08 dex (SNe IIn, 116 points)**, 0.10 dex (SLSN-II) and
  0.13 dex (SNe Ia-CSM), and the public spectra show the same IIn trend: median |EW| ≈ 100 Å near
  maximum, ≈ 270 Å at 60–150 d (Section 5).
* **The SN II and SN IIb EW figures are not reliable as plotted.** The 50 positive EWs (11 SN II,
  34 IIb) and the sign changes are produced by P-Cygni absorption and continuum placement, not by
  emission-line physics; in nebular SN IIb spectra the "Hα" window is dominated by [O I] 6300,6364.
  Only 14 of the 110 IIb points are well-defined emission-line EWs.
* **Most of the Hα/Hβ values are not Balmer decrements.** Only **77 of the 540** plotted ratios (14 %)
  come from spectra in which both lines are significant, emission-dominated and unblended:
  58/166 IIn, 10/159 SN II, 8/66 SLSN-II, 1/108 IIb, **0/41 Ia-CSM**.  The 170 negative ratios and the
  values beyond ±30 arise when the "Hβ flux" (an integral over a P-Cygni or Fe II-blended region) is
  close to zero; changing the integration window by 10 Å flips their sign.  Where the ratio *is*
  defined, the student's values agree with mine to 0.14 dex (IIn).
* **The template bank itself is faithful but has a few defects.** Each template agrees with the public
  spectrum it was built from to 0.01 dex in EW and 0.03 dex in Hα/Hβ (relative Hβ/Hα calibration within
  0.1 dex for 98 % of 275 pairs).  However: the SN 1999br "+41 d" template is corrupted (flux down to
  −240,000 at Hα), which is the source of the −10⁶ EW and +10⁴ ratio points; SN 1987A "+44 d" and
  "+70 d" are the same spectrum; two SN 1998S templates have wavelengths in µm; the SN 2006aa "+41 d"
  template has no public counterpart and an implausibly blue continuum (Hα/Hβ = 0.6 vs. 2.2–2.6 in the
  CSP spectra a few days away).
* **The phase axis mixes conventions.** SN II template phases are days since *discovery* (SN 1987A's
  "day 1" is ~78 d before maximum), most IIn/IIb/SLSN templates are days since *maximum*, several IIn
  (1998S, 2005cp, 2005ip, 2006jd, 2006aa) are again days since discovery, and SN 1995N's are counted from
  a spurious catalogue date (its "+51 d" spectrum is 411 d after discovery).
* **Presentation.** The "SLSN-II" figures also contain the six objects of Duperfit's separate
  SLSN-IIn folder (2006gy, PTF10fel/jwd/qaf/scc/tpz); OGLE-2013-SN-016 and -019 share one legend entry;
  the default symlog scale (linear between −2 and +2) visually separates near-zero values into a
  spurious second population; there are no units, error bars or extinction corrections.

## 1. What was done

| | |
|---|---|
| Templates | Duperfit (Baer 2024) bank, `tempbank/sne/{SNIIn,SNII,SNIIb,SNIa-CSM,SLSN-II,SLSN-IIn}`: 887 spectra, 660 within −20…150 d |
| Public spectra | 1,648 spectra of all 99 objects: OSC per-object JSON (76 SNe, from `github.com/astrocatalogs/sne-*`), WISeREP public spectra for the 23 newer objects the OSC holds no spectra for |
| Student values | all ten figures digitized: axis calibration to < 0.5 px (default matplotlib symlog, linthresh = 2); 1,100 markers, **1,069 matched to an individual template** by colour, phase and marker shape (shape agreement 90–100 %, mean phase residual 0.1 d) |
| Replica method | our reconstruction of the procedure in Wagle, Chatzopoulos & Baer (2025): specutils generic continuum (Chebyshev-3, 4000–7000 Å, lines excluded) → `equivalent_width`, `line_flux` |
| This work | local linear continuum from sidebands at 12,000–16,000 km/s, integration over ±10,000 km/s (same velocity windows for Hα and Hβ), 300 Monte-Carlo realisations of flux noise *and* window placement, explicit quality flags (README) |

The exact settings of the student's code could not be recovered from the figures: no variant of the
replica reproduces his SN II/IIb values, so his integration windows or continuum differ from the
paper's recipe.  His script would allow a line-by-line check.

## 2. Figure-by-figure verdict

"Well-defined" = the measurement passes all quality flags (for EW: significant net emission, < 30 %
absorption, uncompromised continuum; for Hα/Hβ: the same for both lines and no blend).  Agreement is
the median |log₁₀(student/this work)| over well-defined points.

| Figure | Student points | Well-defined | Agreement | Verdict |
|---|---:|---:|---:|---|
| EW — SNIIn | 172 | 116 | 0.08 dex | **validated**, except 1998S (+13, +26 d: lines too weak, sign undetermined) and broad-winged 1997eg (window truncates the wings, |EW| ~35 % low) |
| EW — SNII | 137 | 83 | 0.27 dex | **not reproduced**: student |EW| ≈ 1.9× smaller; 11 positive values are absorption-dominated P-Cygni profiles; −10⁶ point = corrupted template |
| EW — SNIIb | 110 | 14 | 0.49 dex | **not meaningful as plotted**: high-velocity absorption reaches the continuum regions; late points measure [O I] |
| EW — SNIa-CSM | 43 | 24 | 0.13 dex | **largely validated** |
| EW — SLSN-II | 67 | 40 | 0.10 dex | **largely validated**; class mixes SLSN-II and SLSN-IIn |
| Hα/Hβ — SNIIn | 166 | 58 | 0.14 dex | **partly validated**: the 58 agree; 108 are not decrements (15 negative; 1995N is [O III]-contaminated) |
| Hα/Hβ — SNII | 159 | 10 | — | **not valid**: Hβ is a P-Cygni/Fe II blend; 47 negative, 12 with |R| > 30 |
| Hα/Hβ — SNIIb | 108 | 1 | — | **not valid** (64 negative); the one formally valid value (1993J −18 d, 0.28) is unphysical |
| Hα/Hβ — SNIa-CSM | 41 | 0 | — | **not valid**: Hβ is never significant net emission over the line profile; values of ±30–330 come from F(Hβ) ≈ 0 |
| Hα/Hβ — SLSN-II | 66 | 8 | — | **mostly not valid** (22 negative) |

![Student vs this work](figures/student_vs_thiswork.png)
*Every matched point of the ten figures, student value (y) vs. this work on the same template (x).
Blue: well-defined measurements; grey: flagged.*

### Corrected versions of the ten figures (same templates)

| Student | This work (grey = flagged; error bars = 16–84 % MC) |
|---|---|
| ![](figures/student/EW_SNIIn.jpg) | ![](figures/corrected/EW_SNIIn_corrected.png) |
| ![](figures/student/EW_SNII.jpg) | ![](figures/corrected/EW_SNII_corrected.png) |
| ![](figures/student/EW_SNIIb.jpg) | ![](figures/corrected/EW_SNIIb_corrected.png) |
| ![](figures/student/EW_SNIa-CSM.jpg) | ![](figures/corrected/EW_SNIa-CSM_corrected.png) |
| ![](figures/student/EW_SLSN-II.jpg) | ![](figures/corrected/EW_SLSN-II_corrected.png) |
| ![](figures/student/R_SNIIn.jpg) | ![](figures/corrected/R_SNIIn_corrected.png) |
| ![](figures/student/R_SNII.jpg) | ![](figures/corrected/R_SNII_corrected.png) |
| ![](figures/student/R_SNIIb.jpg) | ![](figures/corrected/R_SNIIb_corrected.png) |
| ![](figures/student/R_SNIa-CSM.jpg) | ![](figures/corrected/R_SNIa-CSM_corrected.png) |
| ![](figures/student/R_SLSN-II.jpg) | ![](figures/corrected/R_SLSN-II_corrected.png) |

The corrected ratio panels show only valid Balmer decrements, on a log axis, with the Case B value
(2.86) for reference.

## 3. What goes wrong, with evidence

### 3.1 The ratio of two signed integrals diverges
F(Hα)/F(Hβ) is only a Balmer decrement when both integrals are emission fluxes.  In photospheric
SN II/IIb spectra and in SNe Ia-CSM, the "Hβ" window holds a P-Cygni profile blended with Fe II 4924,
5018 (or the underlying SN Ia spectrum); its continuum-subtracted integral scatters around zero, so the
ratio takes arbitrary sign and size.  With the student-style continuum, moving the window edge by 10 Å
changes the SN II and Ia-CSM ratios between +10⁴ and −10⁴, while a clean IIn line (2010jl) stays at 4–7:

![Window sensitivity](figures/window_sensitivity.png)

### 3.2 Continuum placement and P-Cygni absorption set the EW sign
![Continuum diagnostics](figures/continuum_diagnostics.png)
*Blue: our continuum sidebands; orange dashed: SN pseudo-continuum; red dotted: global Chebyshev-3
(student-style).  Rows: a clean IIn line; a SN II plateau spectrum whose Hβ region is an Fe II forest;
a SN Ia-CSM with Hβ ≈ 0; a pre-maximum IIb whose Hα absorption (−17,000 km/s) fills the blue
continuum region; a nebular IIb where [O I] 6300,6364 sits next to Hα; the corrupted 1999br template.*

For SNe II/IIb the physically meaningful quantities are the absorption and emission pseudo-EWs and their
ratio a/e (Gutiérrez et al. 2014); they are provided in the CSV (`pc_*` columns).

### 3.3 Blends and windows
* SN 1995N "+51 d": the Hβ window contains a strong [O III] 4959/5007 + He I complex; the student's
  ratio of 3.5 is contaminated (the narrow Hβ alone implies a far larger decrement).  Its EW of ≈ −10⁴ Å
  is genuine (bright narrow Hα on a faint continuum), but the spectrum was taken 411 d after discovery,
  not 51 d after maximum (Section 3.5).
* SN 1997eg: Hα wings extend to ±10,000 km/s; a fixed ±100–150 Å window loses ~35 % of the flux.

### 3.4 Template-bank defects
![Template defects](figures/template_defects.png)

| Template | Problem | Effect |
|---|---|---|
| SNII/sn1999br.p0041 | flux −240,000 … +53,000 over 6495–6653 Å (median 1); error column also ~10⁵, so Duperfit's weighted fits ignore it | EW −10⁶, Hα/Hβ +10⁴ in the figures |
| SNII/sn1987A.p0044 = p0070 | same spectrum (r = 0.99999) | one spectrum plotted at two phases |
| SNII/sn2005cs.p0060 ≈ p0061 | near-identical (r = 0.9998, rescaled) | probably duplicate |
| SNIIn/sn1998S.p0326, p0689 | near-IR spectra with wavelengths in µm (1.32 …) | outside the plots, but break Duperfit fits |
| SNIIn/sn2006aa.p0041 | no public counterpart (best r = 0.51); continuum bluer than Rayleigh–Jeans; Hα/Hβ = 0.6 vs. 2.2–2.6 in CSP spectra 3–7 d away | suspect calibration |
| SNIIn/sn1995N.* | the labels −276, −43, +51, +1791 d are exactly the epochs of the four public spectra counted from 1996-04-29, a spurious catalogue "maximum" ~1 yr after discovery | "+51 d" point is really 411 d after discovery |

### 3.5 Phase zero-points
Matching each template to its archive spectrum gives its phase zero-point (`results/template_phase_offsets.csv`):

| Convention | Objects |
|---|---|
| days since **discovery** (offset ≈ 0 d) | SN II: 1987A (+1.0), 1999br, 1999em, 1999gi, 2004et, 2005cs, 2006bp, 2012aw, 2013am; IIn: 1998S, 2005cp, 2005ip, 2006jd, 2006aa |
| days since **maximum** (within ±2 d) | IIn 1996L, 1997eg, 1999eb, 2000P, 2003G, 2005db, 2005gl, 2007pk; IIb 2004ff, 2006el, 2008ax, 2008bo; SLSN-IIn 2006gy, PTF10tpz; Ia-CSM 2005gj |
| from a spurious catalogue date | IIn 1995N (1996-04-29, ~1 yr after discovery) |
| other / unclear | SN II 2004dj (+31 d w.r.t. discovery, i.e. explosion), 2006ov (+89 d), 2009N (+7 d); IIb 1993J, 1998fa, 2000H (−9 to −21 d w.r.t. catalogue maximum, probably another peak or band); Ia-CSM 2011km, 2012ca (−7 d w.r.t. discovery), 1999E; IIn 2010jl, 2015da |

So "Light-curve phase" means different things for different points (SN 1987A's "day 1" is ~78 d before
its catalogue maximum; SN 2006aa peaked ~60 d after discovery; SN 1995N's "+51 d" is 411 d after
discovery).  Trends with phase, and comparisons between classes, need a
common reference.

## 4. Reproduction from public spectra

The same measurements on the original public spectra, with phases from the catalogue maximum when that
date is plausible (else from discovery):

| | |
|---|---|
| ![](figures/archive/EW_SNIIn_archive.png) | ![](figures/archive/R_SNIIn_archive.png) |
| ![](figures/archive/EW_SNII_archive.png) | ![](figures/archive/R_SNII_archive.png) |
| ![](figures/archive/EW_SNIIb_archive.png) | ![](figures/archive/R_SNIIb_archive.png) |
| ![](figures/archive/EW_SNIa-CSM_archive.png) | ![](figures/archive/R_SNIa-CSM_archive.png) |
| ![](figures/archive/EW_SLSN-II_archive.png) | ![](figures/archive/R_SLSN-II_archive.png) |

![Template vs archive](figures/archive/template_vs_archive.png)
*(a) phase zero-points; (b, c) EW and Hα/Hβ of each template against the public spectrum it was made
from (correlation of spectral features > 0.95; 314 pairs); (d) relative Hβ/Hα flux calibration.*

The trends these spectra show are in Section 5.

## 5. Trends in the observed sample

All public spectra measured with the method above; only clean Hα EWs and valid Balmer decrements enter.
Spectra of one object from the same night are merged (359 object-nights), and population statistics
count objects, not spectra: medians over objects per phase bin, 95 % intervals from resampling objects.
Phases are days from maximum light, or from discovery where the catalogue maximum is implausible.
Numbers: `results/observed_results.json`; per-night values: `results/observed_results_points.csv`;
interactive charts: the Results tab of the report page (`report_page/balmer_audit.html`).

**Across classes**

* Hα strengthens with time only in the interacting classes. From within ±20 d of maximum to 60–150 d the
  median |EW| grows ×2.7 in SNe IIn (100 → 273 Å), ×3.2 in SLSN-II (27 → 89 Å) and ×2.6 in SNe Ia-CSM
  (101 → 258 Å); normal SNe II stay flat (165 → 136 Å).
* SLSNe-II carry the weakest Hα for their continuum: 27 Å near maximum, a quarter of the IIn and Ia-CSM values.
* Balmer decrements separate the classes: SNe IIn median 4.4, two thirds of nights above Case B and
  steepening with time; SNe Ia-CSM show no Hβ emission (Hα/Hβ > 10 on 81 % of nights against the local
  continuum); SNe II give 2.2, and only within a week of maximum.

**SNe IIn** (37 objects with clean EWs on 161 nights; 15 with decrements on 52 nights)

* Hα grows relative to the continuum: median |EW| ≈ 80 Å before maximum, ≈ 100 Å at 0–20 d, ≈ 170 Å at
  40–70 d, ≈ 270 Å at 70–100 d, ≈ 610 Å at 100–150 d. 17 of 22 objects with ≥ 3 epochs over ≥ 20 d rise
  (median +0.39 dex per 100 d); pooled Spearman ρ = +0.51 (95 % interval 0.23–0.69); ρ = +0.44 without the
  six objects whose phases count from discovery.
* Early Hα strength differs widely: near maximum |EW| spans about 20–1000 Å. Some objects are Hα-dominated
  from the start (1997eg, 2015da, 2010jl), others start weak and climb as the continuum fades (1998S 34 →
  1340 Å, 2008fq 19 → 220 Å, 2000eo 35 → 130 Å).
* Decrements: median 4.4 (interquartile 2.6–5.9, range 1.1–16); 67 % of nights above 2.86. Milky Way
  reddening changes them by ≤ 8 % (median 2 %); host extinction is not corrected.
* Decrements steepen after maximum: ≈ 2.7 within ±20 d (12 objects), 4.4 after 40 d (7 objects), 6–8
  beyond 70 d; ρ(phase) = +0.49 (0.13–0.75). Five of six objects with ≥ 3 epochs steepen (2015da 4.7 → 13.5,
  1997eg 5.2 → 8.3, 2010jl 2.9 → 4.3, 2006aa 2.2 → 2.6, LSQ12hxg 1.1 → 1.3); 1999eb stays near 2.7.
* Flat decrements are rare: only LSQ12hxg stays below 2, the regime expected where Hα is optically thick
  in very dense gas (Drake & Ulrich 1980). Steep and steepening decrements point to collisional excitation
  in dense gas and/or growing internal extinction (dust formation was reported in SN 2010jl; Gall et al. 2014).
* Decrement vs. Hα strength is only suggestive: ρ = +0.59 over nights, but +0.43 (−0.27 to 0.86) at fixed
  phase and +0.46 between objects (p = 0.09, 15 objects).

**SLSN-II** (10 objects with clean EWs on 30 nights, including SLSN-IIn 2006gy; decrements for 1 object)

* Weakest Hα relative to the continuum: ≈ 27 Å within ±20 d (6 objects).
* Hα strengthens as the continuum fades: ≈ 90 Å at 60–150 d (4 objects); ρ(phase) = +0.52 (0.07–0.85);
  2008es 13 → 126 Å, 2006gy 29 → 62 Å, 2008am 5 → 7 Å.
* No class decrement: only SN 2008am (2.1, 2.3, 0.8 within ~20 d of maximum). The five PTF SLSNe-IIn have
  one public spectrum each, none with a clean Hα measurement.

**SNe Ia-CSM** (6 objects with clean EWs on 36 nights; no valid decrement)

* The tightest growth of Hα with time: ≈ 100 Å within ±20 d (5 objects), ≈ 260 Å at 60–150 d (4 objects);
  ρ(phase) = +0.80 (0.40–0.88); all four objects with ≥ 3 epochs rise (2011km 53 → 247 Å, 2012ca 192 →
  497 Å, 2017eby 102 → 162 Å, 1999E 234 → 288 Å).
* Hβ is not detected: in none of the 54 spectra covering both lines is Hβ significant net emission (21 are
  net absorption). Lower limits against the local continuum (16th percentile of the Hα flux over the 84th
  percentile of the Hβ emission excess) exceed 10 on 81 % of nights and Case B on all (median ≈ 20).
* The limits depend on the continuum under Hβ, which is the Fe-dominated SN Ia spectrum; values need it
  modelled and removed. Sharma et al. (2023) report decrements ≈ 7 with a different method.

**SNe II (comparison)** (11 objects with clean EWs on 114 nights; decrements for 3 objects)

* No growth of Hα relative to the continuum: ≈ 110–180 Å from −20 to +150 d; ρ(phase) = −0.05 (−0.24 to
  0.18); per-object slopes scatter around zero (median +0.09 dex per 100 d).
* Decrements only within a week of maximum (1999em, 2002gd, 2005cs): median 2.2 (1.7–2.8). These are
  photospheric lines, not a recombination value; later Hβ is P-Cygni absorption blended with Fe II.

**SNe IIb (comparison)** (6 objects with clean EWs on 18 nights)

* Hα EW is rarely well defined (fast absorption before maximum, [O I] after ~60 d); no trend can be
  measured. The single formal decrement (1993J, 0.26) is a continuum hump, not Hβ emission.

Caveats: small, heterogeneous samples with uneven phase coverage; host extinction not corrected;
full-line integrals over ±10,000 km/s (narrow and intermediate components not separated); phases from
discovery are offset from maximum by an unknown amount.

## 6. Recommendations

1. Drop or repair the defective templates (Section 3.4); de-duplicate before counting epochs.
2. Measure Balmer decrements only for emission-dominated lines; fit and remove blends (He I 4922/5016,
   [O III] 4959/5007, Fe II 4924/5018) or fit narrow/intermediate/broad components; require both fluxes
   to be significant; never report a negative "ratio"; state whether ratios are corrected for Milky-Way
   and host extinction (E(B−V) = 0.3 changes Hα/Hβ by 35 %).
3. For SNe II/IIb, replace the net EW by the absorption and emission pseudo-EWs (a/e); exclude nebular
   spectra where [O I] enters the Hα window.
4. Put all objects on one phase reference (explosion or maximum) using the offsets in
   `results/template_phase_offsets.csv`.
5. Plot positive quantities on log axes with uncertainties and units; keep SLSN-II and SLSN-IIn
   separate; fix the legend name parsing (OGLE).
6. Check every number's sensitivity to the window and continuum choice (the MC in `bv/measure.py` does
   this) before interpreting trends.

## 7. Limits of this validation

* The student's code was not available; his values were read off the JPEG figures (precision ≈ 1 % of a
  decade) and his exact windows could not be recovered.
* My quality flags are deliberately conservative; some flagged spectra could be rescued by component
  fitting.  Host-galaxy extinction is not corrected anywhere.
* Only 302 of the 660 templates within −20…150 d (46 %; SLSN-II 17 %, IIn 35 %) have a public
  counterpart with a feature correlation > 0.95, so the template-vs-archive statistics rest on that
  subset (some others are probably true matches degraded by smoothing or noise).  SN 2019gsp has no
  redshift on WISeREP and is not measured.  The archive figures use all public spectra, not only the
  template counterparts.

## References

Baer, M. J. 2024, Duperfit v0.1.0, Zenodo, doi:10.5281/zenodo.10992738 ·
Drake, S. A., & Ulrich, R. K. 1980, ApJS, 42, 351 ·
Gall, C., et al. 2014, Nature, 511, 326 ·
Gutiérrez, C. P., et al. 2014, ApJL, 786, L15 ·
Guillochon, J., et al. 2017, ApJ, 835, 64 (Open Supernova Catalog) ·
Sharma, Y., et al. 2023, ApJ, 948, 52 ·
Silverman, J. M., et al. 2013, ApJS, 207, 3 ·
Wagle, G. A., Chatzopoulos, E., & Baer, M. J. 2025, ApJ, 978, 112 ·
Yaron, O., & Gal-Yam, A. 2012, PASP, 124, 668 (WISeREP)
