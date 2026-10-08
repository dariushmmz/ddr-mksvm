# Expanded dataset study — complete local inventory

Initially audited: 2026-09-20  
Final hash re-audit: 2026-09-27  
Scope: every file under `dataset/`

## Inventory decision

The directory contains nine classifier datasets and one acquisition-status
file. Eight classifier datasets already have frozen V7/Legacy evidence. The
only newly usable classifier dataset is the local **Breast Cancer recurrence**
cohort (`breast_cancer.csv`). It is scientifically usable as a distinct
categorical benchmark after an explicit schema-preserving repair/encoding, but
it is not the paper's different 683-row numeric Breast Cancer dataset and must
never be presented as a reproduction of that result.

Dermatology is also included in the expanded experiment because it has frozen
Legacy/V7 confirmation but no prior V8-C evaluation. Existing evidence for the
other seven covered datasets is reused rather than recomputed.

## File-level inventory

| File | Samples x features | Task/classes | Class distribution | Missing values | Stored / semantic feature types | Scientific-use decision |
|---|---:|---|---|---:|---|---|
| `blood_transfusion.csv` | 748 x 4 | binary | 0:570, 1:178 | 0 | 4 numeric | Accepted; already screened with Legacy/V7 and confirmed with V7/V8-C |
| `breast_cancer_diagnostic.csv` | 569 x 30 | binary | 0:212, 1:357 | 0 | 30 numeric | Accepted; existing 96-seed Legacy/V7 and V7/V8-C evidence |
| `breast_cancer.csv` | 286 x 9 | binary | 0:201, 1:85 | 9 categorical blanks: `DATA5` 8, `DATA8` 1 | 9 semantic categorical/ordinal predictors; `DATA6` is integer-coded malignancy grade | **Accepted as a new, separately named recurrence benchmark**; schema repair plus train-only one-hot encoding required |
| `dermatology.csv` | 366 x 34 raw; 358 complete x 34 | 6-class | raw 112/61/72/49/52/20; complete 111/60/71/48/48/20 | 8 cells in `DATA34` (age), affecting 8 rows | 33 coded clinical/histopathology ordinal predictors plus numeric age | Accepted under the frozen complete-case protocol; existing Legacy/V7 confirmation, new V8-C coverage needed |
| `heart_disease.csv` | 303 x 13 raw; 297 complete x 13 | 5-class local labels | raw 164/55/36/35/13; complete 160/54/35/35/13 | 6 cells in `DATA12`/`DATA13`, affecting 6 rows | stored numeric; approximately 5 continuous and 8 coded categorical/ordinal clinical predictors | Accepted under the frozen complete-case/source-numeric protocol; paper comparability remains limited |
| `iris_multiclass.csv` | 150 x 4 | 3-class | 50/50/50 | 0 | 4 numeric | Accepted; existing frozen 96-seed evidence; no rerun |
| `mammographicmass_binary.csv` | 830 x 5 | binary | 0:427, 1:403 | 0 in the local complete-case file | age numeric; BI-RADS/density ordinal and shape/margin categorical codes, all stored numeric | Accepted as the fixed 830-row local benchmark; existing Gate8 and V8-C confirmation evidence |
| `parkinson.csv` | 195 x 22 | binary | 0:48, 1:147 | 0 | 22 numeric | Accepted; existing frozen 96-seed evidence; no rerun |
| `wine.csv` | 178 x 13 | 3-class | 1:59, 2:71, 3:48 | 0 | 13 numeric | Accepted; existing frozen 96-seed evidence; no rerun |
| `uci_fetch_report.csv` | 12 x 8 metadata rows | not a classifier dataset | not applicable | 24 empty metadata cells | acquisition names, statuses, counts, and errors | Rejected from modeling: provenance/status table, not observations and labels |

The unavailable Arrhythmia, Climate Model Crashes, and QSAR Biodegradation
entries appear only in `uci_fetch_report.csv`; no corresponding dataset file
exists locally, so they cannot be evaluated without expanding the user's stated
local-data scope.

## Breast Cancer recurrence semantic repair

The local UCI-ID-14 export has generic column names and spreadsheet-style
rendering of several interval tokens. Before splitting, the following
deterministic, label-independent repairs restore the categorical tokens:

| Column | Stored token | Restored token |
|---|---|---|
| `DATA3` (tumor size) | `9-May` | `5-9` |
| `DATA3` | `14-Oct` | `10-14` |
| `DATA4` (involved nodes) | `5-Mar` | `3-5` |
| `DATA4` | `8-Jun` | `6-8` |
| `DATA4` | `11-Sep` | `9-11` |
| `DATA4` | `14-Dec` | `12-14` |

Blank `node-caps` and `breast-quad` entries are represented as an explicit
`__MISSING__` category. No row is removed and no outcome-dependent imputation
is used. Every model receives the same outer split. One-hot vocabularies are
fit on the current training partition only with unknown validation/test
categories mapped to all-zero columns. Standardization is then fit on that
same training partition, matching the source-configured `standardization` plus
`hom_linear` profile for the Legacy-compatible baseline.

This treatment preserves the local dataset's categorical scientific meaning;
it does change the numerical representation required by the algorithms. For
that reason the baseline is named **Legacy source-profile**, not exact MATLAB
executable parity.

## Existing coverage versus new work

| Dataset | Legacy/V7 status | V8-C status | Expanded-study action |
|---|---|---|---|
| Parkinson | 96 confirmed | 96 confirmed | reuse |
| Blood Transfusion | Gate8 negative | 96 confirmed against V7 | reuse negative/confirmed evidence |
| Mammographic Mass | Gate8 negative | 96 confirmed against V7 | reuse negative/confirmed evidence |
| Breast Cancer Diagnostic | 96 confirmed | 96 confirmed | reuse |
| Wine | 96 confirmed | 96 confirmed | reuse |
| Heart Disease | 96 confirmed | 96 confirmed | reuse with protocol caveat |
| Iris | 96 confirmed | 96 confirmed | reuse |
| Dermatology | two independent 96-seed Legacy/V7 confirmations | fixed V8-C control stopped at expanded Gate24 | expanded study complete |
| Breast Cancer recurrence | new 96-seed Legacy/V7 confirmation | fixed V8-C control stopped at Gate24 | expanded study complete as distinct cohort |

Machine-readable inventory and hashes are stored under
`results/expanded_dataset_study/inventory/`.

The final re-audit found no added, removed, or altered dataset file. SHA-256
hashes were regenerated from all ten CSV files and the usable set remains nine
classifier datasets plus one rejected metadata report.
