# Paper data tables

Generated from frozen machine-readable artifacts by `build_paper_data_tables.py`. Values are not copied from rounded prose reports. Percentage-point deltas are candidate minus reference.

## A0. Architecture terminology

| Internal label | Scientific manuscript name |
| --- | --- |
| Legacy | Source-Profile q1 Kernel SVM |
| V7 | RKHS-RBF-OVO SVM |
| V8-C | CS-RKHS-RBF-OVO SVM |
| Robust V8-C | RKHS-Norm Robust Class-Sensitive SVM |
| R8 | Source-Aligned Class-Sensitive Robust q1 LP |
| R8.1 | Equilibrated Source-Aligned Robust q1 LP |
| R9 | Bounded-RBF Class-Sensitive Robust q1 SVM |

## A1. Final local dataset inventory

| Dataset | N | Features | Classes / distribution | Final role |
| --- | ---: | ---: | --- | --- |
| Blood Transfusion | 748 | 4 | 570/178 | RKHS Gate8 negative; class-sensitive confirmed |
| Breast Cancer Diagnostic | 569 | 30 | 212/357 | both deterministic comparisons confirmed |
| Breast Cancer recurrence | 286 | 9 | 201/85 | RKHS confirmed; class-sensitive stopped at Gate24 |
| Dermatology | 358 complete | 34 | 111/60/71/48/48/20 | RKHS confirmed; class-sensitive stopped at Gate24 |
| Heart Disease | 297 complete | 13 | 160/54/35/35/13 | RKHS confirmed; class-sensitive tradeoff |
| Iris | 150 | 4 | 50/50/50 | both deterministic comparisons confirmed |
| Mammographic Mass | 830 | 5 | 427/403 | RKHS Gate8 negative; class-sensitive null confirmation |
| Parkinson | 195 | 22 | 48/147 | primary RKHS confirmation; class-sensitive control |
| Wine | 178 | 13 | 59/71/48 | both deterministic comparisons confirmed |

The tenth local CSV, `uci_fetch_report.csv`, is acquisition metadata rather than a labeled observation table and is excluded from modeling.

## A2. Source-Profile q1 versus RKHS-RBF-OVO: untouched 96-seed confirmations

| Dataset | Source-profile error % | RKHS-RBF-OVO error % | Delta pp [95% CI] | W/T/L | Wilcoxon p | Effect | Runtime ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Parkinson | 14.243 | 7.079 | -7.164 [-8.161, -6.167] | 85/9/2 | 1.77e-15 | -1.456 | 0.622x |
| Breast Cancer recurrence | 30.642 | 26.808 | -3.834 [-4.714, -2.954] | 80/4/12 | 1.24e-10 | -0.882 | 0.588x |
| Iris | 4.331 | 3.125 | -1.206 [-1.800, -0.612] | 40/39/17 | 2.72e-04 | -0.412 | 0.370x |
| Heart Disease (5 class) | 43.792 | 42.639 | -1.153 [-1.985, -0.321] | 54/13/29 | 0.0018 | -0.281 | 0.159x |
| Wine | 2.755 | 2.037 | -0.718 [-1.204, -0.231] | 44/31/21 | 0.0061 | -0.299 | 0.300x |
| Dermatology | 4.271 | 3.808 | -0.463 [-0.903, -0.023] | 46/21/29 | 0.0350 | -0.213 | 0.111x |
| Breast Cancer Diagnostic | 2.855 | 2.601 | -0.255 [-0.475, -0.035] | 48/26/22 | 0.0066 | -0.235 | 0.206x |

## A3. Class-sensitive versus unweighted RKHS-RBF-OVO: untouched 96-seed confirmations

| Dataset | Delta error pp | Delta balanced accuracy pp | Delta macro-F1 pp | Error W/T/L | Runtime ratio | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Blood Transfusion | 0.156 | 5.820 | 5.571 | 42/8/46 | 0.616x | strong balance/minority gain; neutral error |
| Breast Cancer Diagnostic | 0.029 | 0.058 | -0.026 | 23/42/31 | 1.464x | no confirmed advantage |
| Heart Disease (5 class) | 1.875 | 2.553 | 2.879 | 28/10/58 | 4.986x | balance/F1 gain with error and runtime cost |
| Iris | -0.000 | -0.002 | -0.001 | 3/89/4 | 4.763x | no confirmed advantage |
| Mammographic Mass | -0.045 | 0.056 | 0.050 | 36/30/30 | 0.954x | null control |
| Parkinson | 0.170 | 1.002 | 0.169 | 25/48/23 | 2.575x | no confirmed advantage |
| Wine | 0.231 | -0.170 | -0.221 | 9/71/16 | 3.931x | no confirmed advantage |

## A4. Bounded-RBF robust q1 fresh Gate8 primary paired comparison

Weighted robust RBF q1 minus weighted deterministic RBF q1; screening evidence only.

| Dataset | Pairs | Delta error pp | Delta BA pp | Delta macro-F1 pp | Delta safety/rare recall pp | Error W/T/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blood Transfusion | 7 | 1.451 | -1.377 | -1.955 | -1.248 | 0/1/6 |
| Parkinson | 8 | 2.296 | -5.039 | -4.301 | -10.417 | 1/3/4 |
| Mammographic Mass | 8 | 0.781 | -0.915 | -0.868 | -5.569 | 1/1/6 |
| Iris | 8 | 0.329 | -0.321 | -0.332 | 0.000 | 0/7/1 |

## A5. Robust-failure mechanistic diagnostics

| Dataset | Mean L1 ref -> robust | Support ref -> robust | Slack sum ref -> robust | Flips | Corrected | Introduced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blood Transfusion | 15.29 -> 6.12 | 18.57 -> 8.71 | 255.23 -> 286.80 | 45 | 13 | 32 |
| Parkinson | 37.79 -> 30.82 | 46.62 -> 30.38 | 11.57 -> 37.18 | 25 | 8 | 17 |
| Mammographic Mass | 33.17 -> 17.25 | 58.50 -> 27.50 | 185.69 -> 256.68 | 105 | 46 | 59 |
| Iris | 7.91 -> 7.69 | 2.92 -> 2.75 | 12.31 -> 13.42 | 1 | 0 | 1 |

## A6. Provenance

- Deterministic paired statistics: `results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv`
- Deterministic model summaries: `results/expanded_dataset_study/analysis/cross_dataset/confirmed_model_summary.csv`
- Bounded-RBF Gate8 paired statistics (historical artifact path): `results/robust/bounded_rbf_redesign/gates/r9-gate8-15300-15307-v1/paired_statistics.csv`
- Robust-failure objective diagnostics (historical artifact path): `results/robust/bounded_rbf_redesign/r10_failure_analysis/paired_objective_deltas.csv`
- Robust-failure flip counts (historical artifact path): `results/robust/bounded_rbf_redesign/r10_failure_analysis/dataset_flip_summary.csv`
