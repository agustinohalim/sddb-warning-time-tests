# sddb-warning-time-tests

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23283430.svg)](https://doi.org/10.5281/zenodo.23283430)

Code for the paper

> Halim, A., Lukman, A., Andika, R., Macnamara, B.D., & Erwin. **Between-database horizon tests do not
> establish warning times in HRV-based prediction of sudden cardiac death.** Manuscript submitted to
> *Biomedical Signal Processing and Control*.

Cite this code as: Halim, A. (2026). *sddb-warning-time-tests* (version 1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.23283431

Studies report that heart rate variability (HRV) from the MIT-BIH Sudden Cardiac Death Holter
Database (sddb) predicts sudden cardiac death minutes to hours before ventricular fibrillation (VF),
mostly by contrasting sddb with the MIT-BIH Normal Sinus Rhythm Database (nsrdb). These scripts test
what that design can show: a horizon test, diseased controls, surrogate onsets, within-patient
contrasts with permutation and drift controls, a power simulation with resampled patients, a
replication of a published segment-level design, and a heart failure positive control.

Related: [scd-ecg-benchmark-audit](https://github.com/agustinohalim/scd-ecg-benchmark-audit), the
author's re-examination of his 2022 thesis on the same benchmark (correction note
[10.5281/zenodo.23034462](https://doi.org/10.5281/zenodo.23034462)).

## Running

Python 3.12. Only beat annotations and headers are used, never the raw signal.

```bash
pip install -r requirements.txt
cd scripts
python 00_get_data.py              # PhysioNet annotations into ../data (about 60 MB)
python 01_denyut.py                # beats -> keluaran/denyut.csv, keluaran/rekaman.csv
python 02_fitur.py                 # 5-minute HRV windows -> keluaran/fitur.csv (10,669 windows)
python 03_uji.py                   # then the scripts below, in order
```

Outputs go to `scripts/keluaran/` and `scripts/Gambar/`. The main analyses (01–09) run in under an
hour on a laptop without a GPU; the permutation tests and power simulations (10–13) take about eight
hours. Every random draw uses a fixed NumPy seed written in the script. Script docstrings and
comments are in Indonesian; variable names follow them (`jendela` window, `ciri` feature, `dekat`
near, `jauh` far, `daya` power, `pasien` patient).

## Where each result comes from

| Paper item | Script(s) | Output in `results/` |
|---|---|---|
| Table 1 (literature audit) | — (manual coding, one author, full text) | `audit/coding_sheet.json` (repository root) |
| Table 2, Figure 1A (horizon test) | `03_uji.py` | `uji.log`, `hasil_uji.csv` |
| Table 3 (fixed cohort; refitted bootstrap) | `04_uji_lanjut.py`, `09_revisi_panel.py refit` | `uji_lanjut.log`, `revisi_panel.log` |
| Table 4, Figure 1C (power simulation) | `11_daya_bootstrap.py daya`, `12_analisis_akhir.py dalam12 rf`, `13_peluruhan_lambat.py` | `daya_bootstrap.csv`, `akhir_dalam12.csv`, `akhir_rf.csv`, `lambat.csv`, `rasio.log` |
| Table 5 (surrogate onset) | `09_revisi_panel.py palsu` | `revisi_panel.log` |
| Table 6, Figure 1B (within-patient contrast, bands, rhythm subsets, paired tests) | `07_dalam_pasien.py`, `09_revisi_panel.py halus sinus pasangan` | `dalam_pasien.log`, `revisi_panel.log` |
| Table 6, permutation and drift control | `10_daya_ulang.py permutasi` (logistic), `11_daya_bootstrap.py permutasi` (random forest, 999), `12_analisis_akhir.py geser` | `daya_ulang_permutasi.log`, `daya_bootstrap_permutasi.log`, `akhir_geser_*.csv` |
| Table 7 (published segment-level design) | `05_bocor.py` | `hasil_bocor.csv` |
| Table 8 (heart failure positive control) | `chf_01_jendela_chf.py`, `chf_02_uji_chf.py` | `hasil_chf.csv`, `Hasil_Uji_CHF.md` |
| Table 9 (audited annotations) | `08_anotasi_teraudit.py` | `teraudit.log`, `hasil_teraudit.csv` |
| Rhythm labels, AF check | `09_revisi_panel.py irama`, `10_daya_ulang.py af` | `irama_jendela.csv`, `daya_ulang_af.log` |
| 128 Hz rounding, per-patient breakdown | `12_analisis_akhir.py hz128 perpasien` | `akhir.log` |
| Figure 1 | `06_gambar.py` | `figures/` |

`results/Hasil_Uji_1.md` is the full log of results in Indonesian, in the order the analyses were
run, including the versions of the power simulation that were superseded (and why).
`af_01_jendela.py` and `af_02_uji_af.py` are helper modules shared with a companion study on atrial
fibrillation; here they supply the feature function and cross-validation used by the heart failure
control.

## Data

PhysioNet, Open Data Commons Attribution License v1.0: sddb
([10.13026/C2W306](https://doi.org/10.13026/C2W306)), nsrdb, chfdb, chf2db, nsr2db and Fantasia.
`data/VF Data.csv` lists the VF onset time of each sddb record as given on the PhysioNet sddb page.
Please cite PhysioNet (Goldberger et al., 2000, *Circulation* 101:e215; Pollard et al., 2026,
*Nature Health* 1:792) and the databases when using these data.

## Licence

Code: MIT. See `LICENSE`.
