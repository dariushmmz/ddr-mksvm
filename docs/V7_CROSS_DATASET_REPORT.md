# V7 cross-dataset validation report

Date completed: 2026-09-09  
Status: frozen V7 generalizes beyond Iris, but its evidence is strongest on
multiclass data and is not universal on binary data

## 1. Executive Summary

Frozen V7 (`RKHS-l2 RBF OVO`) was compared with corrected executable q=1
Legacy on seven usable non-Iris datasets. All comparisons used matched sorted
stratified 75/25 splits and training-only preprocessing. Five datasets
qualified for untouched 96-seed confirmation (8000--8095); Blood Transfusion
and Mammographic Mass were pruned after eight development seeds.

V7 had lower mean error on all five confirmations. The paired 95% interval was
strictly below zero for Parkinson, Breast Cancer Diagnostic, Wine, and Heart
Disease. Dermatology was directionally better but statistically
indistinguishable. Parkinson showed the largest gain: 7.0791% versus 14.2432%,
a -7.1641 percentage-point paired change. V7 was faster on six of the seven
screened datasets; Mammographic Mass was both less accurate and 1.61 times
slower.

The evidence rejects an Iris-only interpretation. It does not establish V7 as
a universal improvement: binary outcomes split two wins and two losses. All
four evaluated multiclass datasets, including frozen Iris, favor V7 in mean
error, with statistically supported gains on Iris, Wine, and the locally
multiclass Heart protocol. The most accurate verdict is **mainly a multiclass
improvement, with dataset-dependent binary benefits**.

## 2. Motivation for Cross-Dataset Validation

Iris alone could not distinguish a general RKHS-l2/OVO advantage from a
dataset-specific result. This phase therefore froze all architectural choices
and changed only legitimate dataset-dependent hyperparameters through the
same training-only selection rule.

## 3. Frozen V7 Architecture

Binary data use one conventional soft-margin RKHS-l2 RBF SVM. Multiclass data
use libsvm OVO with its frozen voting and tie behavior. `C` is selected from
`{0.1,1,10,100}` and bandwidth from `paper, med:{-2,-1,-0.5,0,0.5,1,2}` by
three-fold stratified inner CV (`random_state=20000+outer_seed`). Mean inner
error is minimized; bandwidth declaration order and then C order break ties.
Scaling is refitted inside every inner fold and on each outer training set.

No normalized voting, calibration, class weighting, ARD, solver tuning, or V8
change was introduced.

## 4. MATLAB / Executable Legacy Baseline

The reference is the corrected executable deterministic q=1 formulation:
dataset-configured kernel, OVA for multiclass, native-order
`logspace(-3,0,5)` nu loop, solver-returned xi, and the 10,000-point
MATLAB-order first-minimum threshold grid. HiGHS is primary; the already
validated CVXPY/CLARABEL fallback solves the identical LP when HiGHS reports a
numerical failure.

This executable baseline is distinct from literature values and historical
project outputs. On the four binary datasets with historical 96-run results,
the new matched Legacy means are consistent in scale: Parkinson 14.24%
(historical 14.12%), Breast Diagnostic 2.86% (2.59%), Blood gate-8 20.05%
(20.45%), and Mammographic gate-8 15.38% (15.54%).

## 5. Experimental Protocol

- Outer evaluation: sorted stratified 75/25 holdout, identical rows for both
  models.
- Development registry: seeds 7000--7023.
- Confirmation registry: untouched seeds 8000--8095.
- Fixed complete-case dataset definitions precede splitting for Heart and
  Dermatology; no test statistic or label determines removal.
- Primary metric: split test error. Balanced accuracy, macro F1, pooled class
  metrics, confusion matrices, runtime, and fit diagnostics are also saved.
- Paired delta is `error_V7-error_Legacy`; negative favors V7. Intervals are
  paired t intervals. Wilcoxon tests exclude exact zero differences.

## 6. Dataset Inventory

| Dataset | Effective shape | Type | Class counts | Fixed preprocessing | Legacy kernel | Paper deterministic ref. |
|---|---:|---|---|---|---|---:|
| Parkinson | 195 x 22 | binary | 48/147 | min-max | homogeneous linear | 13.19% |
| Blood Transfusion | 748 x 4 | binary | 570/178 | standardize | inhomogeneous cubic | 20.72% |
| Mammographic Mass | 830 x 5 | binary | 427/403 | standardize | inhomogeneous quadratic | 15.71% |
| Breast Cancer Diagnostic | 569 x 30 | binary | 212/357 | min-max | inhomogeneous quadratic | 3.02% |
| Wine | 178 x 13 | 3-class | 59/71/48 | standardize | inhomogeneous linear | 2.77% |
| Heart Disease | 297 x 13 | local 5-class | 160/54/35/35/13 | complete-case + standardize | inhomogeneous linear | 17.48%* |
| Dermatology | 358 x 34 | local 6-class | 111/60/71/48/48/20 | complete-case, none | inhomogeneous quadratic | 1.64%* |
| Iris | 150 x 4 | 3-class | 50/50/50 | none | executable source quadratic | 3.10%; robust 2.87% |

`*` Heart and Dermatology local multiclass semantics are not demonstrably the
same as the paper evaluation, so their paper values are context only.

Excluded: Breast Cancer is a 286-row categorical file whereas the paper uses
683 numeric rows. Arrhythmia, Climate Model Crashes, and QSAR Biodegradation
are absent; `uci_fetch_report.csv` records failed imports.

## 7. Modal Resources and Credit-Saving Gates

All real experiments used 4 CPU, 8 GiB RAM, and no GPU on Modal volume
`ddr-mksvm-v7-cross-dataset-results`.

- Smoke app `ap-mga3JHBVaOCnTfkh1Z1QL8`: all seven datasets loaded and both
  models produced finite, nontrivial predictions.
- Blood fallback smoke `ap-RRmi5vPKdzcxqFL1Do8LzV`: failing seed 7003 passed
  with validated CLARABEL fallback.
- Gate-8 app `ap-n94khHTSc25ZmuCceKzATx`: all seven datasets.
- Gate-24 app `ap-Hx1wnL442iIArkpBkLiZFm`: five qualifiers.
- Final app `ap-RaEKPAT2fGZW7x9Hls9lEZ`: five datasets, seeds 8000--8095.

Successful scientific runs consumed about 709 worker-wall seconds, or 0.79
allocated vCPU-hours. Including two failed pre-fallback launches and image
setup, total usage is approximately 1.2 vCPU-hours. No GPU credit was used.

## 8. Per-Dataset Results

| Dataset | N | Type | Paper ref. | Legacy error | V7 error | Delta | Legacy total s | V7 total s | Statistical result | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| Iris | 96 | multiclass | 2.87% robust | 4.3311% | **3.1250%** | -1.2061 pp | 49.75 | **18.42** | supported, p=0.000137 | V7 wins |
| Parkinson | 96 | binary | 13.19% | 14.2432% | **7.0791%** | -7.1641 pp | 23.80 | **14.81** | supported | V7 wins |
| Breast Cancer Diagnostic | 96 | binary | 3.02% | 2.8555% | **2.6005%** | -0.2550 pp | 204.93 | **42.12** | supported | modest V7 win |
| Wine | 96 | multiclass | 2.77% | 2.7546% | **2.0370%** | -0.7176 pp | 48.64 | **14.59** | supported | V7 wins |
| Heart Disease* | 96 | multiclass | 17.48% | 43.7917% | **42.6389%** | -1.1528 pp | 190.68 | **30.31** | supported, poor absolute accuracy | V7 wins under local protocol |
| Dermatology* | 96 | multiclass | 1.64% | 4.2593% | **3.9815%** | -0.2778 pp | 336.88 | **37.05** | indistinguishable | accuracy tie/runtime win |
| Blood Transfusion | 8 | binary | 20.72% | **20.0535%** | 21.4572% | +1.4037 pp | 79.53 | **21.10** | directional regression | pruned |
| Mammographic Mass | 8 | binary | 15.71% | **15.3846%** | 16.6466% | +1.2620 pp | **56.05** | 90.04 | near-consistent regression | pruned |

Iris Legacy in this table is the executable MATLAB-faithful polynomial result;
the manual q1 RBF switch was 3.3717%. The published 2.87% was not reproduced.

Complete distribution and metric summary (`error +/- split SD`, intervals and
medians in percentage points; runtimes in seconds):

| Dataset/model | Mean +/- SD | 95% CI | Median | Balanced acc. | Macro F1 | Mean / median runtime |
|---|---:|---:|---:|---:|---:|---:|
| Parkinson Legacy | 14.2432 +/- 3.9946 | [13.4338,15.0526] | 14.2857 | 72.6211 | 76.0705 | 0.2479 / 0.2467 |
| Parkinson V7 | **7.0791 +/- 3.3886** | [6.3925,7.7657] | 8.1633 | **88.5381** | **89.8876** | 0.1542 / 0.1540 |
| Breast Diagnostic Legacy | 2.8555 +/- 1.2372 | [2.6048,3.1061] | 2.7972 | 96.4346 | 96.8982 | 2.1347 / 2.1480 |
| Breast Diagnostic V7 | **2.6005 +/- 1.0914** | [2.3794,2.8217] | 2.7972 | **96.8675** | **97.1877** | 0.4388 / 0.4392 |
| Wine Legacy | 2.7546 +/- 2.2969 | [2.2892,3.2200] | 2.2222 | 97.4306 | 97.2577 | 0.5066 / 0.5023 |
| Wine V7 | **2.0370 +/- 1.9789** | [1.6361,2.4380] | 2.2222 | **98.0421** | **97.9353** | 0.1520 / 0.1519 |
| Heart Legacy | 43.7917 +/- 2.5762 | [43.2697,44.3137] | 44.0000 | 25.8035 | 20.1666 | 1.9863 / 2.0098 |
| Heart V7 | **42.6389 +/- 3.2544** | [41.9795,43.2983] | 42.6667 | **29.1294** | **27.8751** | 0.3157 / 0.3157 |
| Dermatology Legacy | 4.2593 +/- 1.8569 | [3.8830,4.6355] | 4.4444 | **95.6104** | **95.2240** | 3.5092 / 3.5341 |
| Dermatology V7 | **3.9815 +/- 1.8778** | [3.6010,4.3620] | 3.3333 | 95.3425 | 95.1182 | 0.3859 / 0.3854 |
| Blood Legacy (N=8) | **20.0535 +/- 1.9175** | [18.4504,21.6565] | 19.2513 | **64.9960** | **66.4559** | 9.9416 / 7.4530 |
| Blood V7 (N=8) | 21.4572 +/- 1.7016 | [20.0347,22.8798] | 21.1230 | 61.2308 | 62.0064 | 2.6376 / 2.5228 |
| Mammographic Legacy (N=8) | **15.3846 +/- 2.6952** | [13.1313,17.6379] | 15.1442 | **84.5818** | **84.5866** | 7.0063 / 5.5957 |
| Mammographic V7 (N=8) | 16.6466 +/- 2.6073 | [14.4669,18.8264] | 17.0673 | 83.2337 | 83.2479 | 11.2556 / 8.8240 |

## 9. Statistical Comparisons

| Dataset | Mean paired delta | 95% paired CI | W/T/L | Wilcoxon p | Standardized effect |
|---|---:|---:|---:|---:|---:|
| Parkinson | -7.1641 pp | [-8.1609,-6.1674] | 85/9/2 | 1.95e-15 | -1.456 |
| Breast Diagnostic | -0.2550 pp | [-0.4746,-0.0353] | 48/26/22 | 0.00656 | -0.235 |
| Wine | -0.7176 pp | [-1.2043,-0.2309] | 44/31/21 | 0.00638 | -0.299 |
| Heart Disease | -1.1528 pp | [-1.9849,-0.3207] | 54/13/29 | 0.00191 | -0.281 |
| Dermatology | -0.2778 pp | [-0.6738,+0.1183] | 46/17/33 | 0.239 | -0.142 |

Discrete holdout errors create many ties; p-values are therefore supporting
evidence, not the sole decision criterion.

## 10. Runtime Comparisons

Final V7/Legacy aggregate runtime ratios were 0.622 Parkinson, 0.206 Breast
Diagnostic, 0.300 Wine, 0.159 Heart, and 0.110 Dermatology. V7 saved 665.1
model-seconds across these five confirmations. At gate 8 it was also 3.77 times
faster on Blood. Mammographic Mass was the exception: nested V7 selection made
it 1.61 times slower than five Legacy LPs.

V7 performs 97 libsvm fits per split (96 inner fits plus the outer fit), while
Legacy performs five LPs per binary task. Consequently V7's advantage grows
with Legacy's OVA task count and LP cost: 3 OVO classifiers for Wine, 10 for
Heart, and 15 for Dermatology, although libsvm internally performs them within
one multiclass estimator. The q1 OVA baseline requires respectively 15, 25,
and 30 LP fits per split.

## 11. Binary vs Multiclass Analysis

Binary evidence is mixed: Parkinson and Breast Diagnostic improve, Blood and
Mammographic regress at gate 8. The Parkinson gain is not merely overall
accuracy: pooled minority-class-0 recall rises from 46.88% to 79.95%, while
majority-class-1 recall changes from 98.37% to 97.13%. Breast Diagnostic class-0
recall rises from 93.69% to 94.81%, with a small class-1 recall cost.

Multiclass evidence is more consistent. Iris, Wine, and Heart have supported
mean-error improvements; Dermatology is directionally lower and much faster.
This phase does not isolate whether RKHS-l2, RBF, or OVO is causal because all
three differ from Legacy together.

## 12. Where V7 Improves

- Parkinson: large accuracy, balance, macro-F1, and runtime improvement.
- Breast Diagnostic: small but supported accuracy gain and 4.9-fold speedup.
- Wine: lower error, higher balanced accuracy, and 3.3-fold speedup.
- Heart local multiclass protocol: modest supported error reduction and much
  better macro F1, though performance remains unacceptable.
- Dermatology: comparable accuracy at roughly one ninth the runtime.

## 13. Where V7 Fails

Blood's V7 error is 1.40 points higher at eight seeds and its balanced accuracy
falls from 65.00% to 61.23%. Its inhomogeneous cubic Legacy Gram matrices are
severely ill-conditioned; two of 40 gate-8 LPs required CLARABEL, but all were
solved. Minority class-1 recall falls from 36.41% to 28.01%. This explains
solver fragility, not V7's accuracy regression.

Mammographic Mass is the clearest failure: +1.26 error points, losses on six of
eight splits, lower balanced accuracy, and higher runtime. The evidence does
not identify a single cause beyond the frozen RBF/C-bandwidth selection being
less suitable than the dataset's quadratic Legacy boundary. Class-1 recall
falls from 83.42% to 79.08%, while class-0 recall rises slightly.

Heart exposes a recurring imbalance weakness. V7 improves macro F1 from
0.202 to 0.279, but class 5 recall is still zero and overall error exceeds 42%.
Legacy selected 357 trivial binary OVA tasks among 480 final selected tasks;
V7 avoids trivial fits but does not solve the rare-class problem.

Dermatology's overall error improves slightly while balanced accuracy and
macro F1 are slightly worse. V7 helps classes 1--3 and 5 but increases errors
for classes 4 and 6. This is a class-distribution tradeoff, not an unqualified
accuracy win.

## 14. Relationship to Published Paper Results

Paper values are literature references only. Comparable-looking Legacy means
on the four established binary datasets support the pipeline, but unknown
original masks and solver details prevent exact reproduction claims. Heart and
Dermatology are especially non-comparable because their local multiclass label
interpretation is not established as the paper's protocol. For Iris:

```text
Published robust reference:          2.87%
Published deterministic reference:   3.10%
Executable MATLAB-faithful Legacy:   4.3311%
Manual q1 RBF switch:                3.3717%
Frozen V7 RKHS-l2 RBF OVO:           3.1250%
```

## 15. Reproducibility and Solver Notes

Local repository HEAD at execution was
`dabc5ca49bf44b11a1f3632af3ed6b8e35182d6f`; the dirty tree is recorded.
Because `.git` is not copied into the Modal image, remote configs record commit
as `unavailable`. The final scientific code SHA-256 is
`e193debab9f376dc7e840b6740394cdf6cffeee68feb233eb4b84aa56e250cc8`.
Dataset hashes, complete seed lists, split indices/hashes, predictions/hashes,
selected alpha/C, all nu fits, confusion matrices, and runtimes are stored.

Final confirmation diagnostics contain 54,240 fits: 7,680 successful HiGHS
LPs and 46,560 successful libsvm fits, with zero warnings, failures, or
fallbacks. Gate 8 contains two Blood and one Mammographic CLARABEL fallbacks;
all report `optimal`. The frozen Iris config, per-run, and split hashes remain
unchanged, and the complete suite passes 80 tests.

Reproduction command:

```powershell
modal run modal_v7_cross_dataset.py --datasets parkinson,breast_cancer_diagnostic,wine,heart_disease,dermatology --n-seeds 96 --seed-start 8000 --n-jobs 4 --run-name v7-cross-final96-reproduction
```

## 16. Limitations

- Confirmation qualification used development seeds, so only the untouched
  final runs support final estimates.
- Blood and Mammographic have eight seeds only; their exact effect sizes remain
  uncertain, although neither met the spending gate.
- The evaluation changes regularizer, kernel, and multiclass decomposition
  together and cannot attribute causality.
- Heart and Dermatology protocol comparability with the paper is unresolved.
- Runtime is model-process time on one Modal environment, not a hardware-neutral
  complexity measure.

## 17. Overall Verdict

V7 is **mainly a multiclass improvement**, not an Iris-specific result. It also
provides a large binary improvement on Parkinson and a modest one on Breast
Diagnostic, so the benefit is broader than multiclass alone. It is not a
general win across every dataset: Blood and Mammographic show real negative
evidence. Across all seven non-Iris screens V7 wins five dataset means and
loses two; across six of seven it is faster.

## 18. Evidence-Based Recommendation for V8

Do not deepen the architecture. The recurring weakness is class-sensitive
behavior on imbalanced binary and rare multiclass classes, plus occasional
dataset/kernel mismatch. The next V8 research question should be a minimal
**class-sensitive RKHS-l2 RBF OVO objective**, with training-only selection of
class penalties and balanced-error selection, benchmarked first on Blood,
Mammographic, Parkinson minority recall, and Heart class 5. Preserve the V7
path as the zero-change member and test whether class sensitivity fixes the
failures without erasing the confirmed Parkinson/Wine/Iris gains. Adaptive
kernel changes should be secondary unless that controlled study shows the
Mammographic quadratic-vs-RBF mismatch persists independently of imbalance.
