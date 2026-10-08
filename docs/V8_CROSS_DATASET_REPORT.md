# V8 class-sensitive cross-dataset study

Status: deterministic confirmation complete, 2026-09-09.
**V8-C qualifies as the new deterministic architecture** under the
prospectively fixed class-sensitive target and mean-error guard criteria.
This is not universal accuracy superiority, formal noninferiority, or a
reproduction of the paper. Frozen V7 remains unchanged and available.

## Scientific question and controlled changes

Can pair-normalized weighted RKHS slack penalties and balanced inner selection
repair Blood/Heart minority weaknesses without sacrificing Parkinson, Wine,
Iris and Breast Diagnostic? Mammographic is a secondary target with almost
balanced classes, so improvement there is not implied by the hypothesis.

The complete derivation precedes implementation in V8_MATHEMATICAL_DESIGN.md.
V8's primal is `0.5||f||²+C sum(w_i xi_i)` with standard margin constraints;
its dual has boxes `0<=a_i<=Cw_i`, equality `y^T a=0`, and concave quadratic
objective. Pair inverse/sqrt weights have exactly unit sample mean. Learned
ratio has geometric-mean C fixed but not arithmetic-mean penalty fixed.
This is a standard weighted SVM extension, not a new convex optimization
theorem. The study isolates weighting-only, selection-only, their combination,
sqrt attenuation and a learned ratio; macro-F1 selection is diagnostic, not a
post-hoc replacement for the prospectively primary balanced-error rule.

Frozen V7 code and artifacts are protected by a 23-file SHA-256 manifest in
`results/v8/analysis/v7_freeze.json`. Exact frozen Iris/Wine replay and synthetic
binary/multiclass neutral-pair prediction tests pass. A synthetic weighted QP
passes box, equality, free-support KKT and primal/dual-gap tests. The selection
function cannot receive outer-test data, and fold transforms/counts are tested.

## Baselines must remain distinct

| Dataset | Published deterministic | Published robust best | Historical project Legacy | Frozen matched executable Legacy | Frozen V7 |
|---|---:|---:|---:|---:|---:|
| Blood | 20.72% | 20.55% | 20.45% (96) | 20.0535% (8) | 21.4572% (8) |
| Mammographic | 15.71% | 15.42% | 15.54% (96) | 15.3846% (8) | 16.6466% (8) |
| Parkinson | 13.19% | 12.37% | 14.12% (96) | 14.2432% (96) | 7.0791% (96) |
| Breast Diagnostic | 3.02% | 2.39% | 2.59% (96) | 2.8555% (96) | 2.6005% (96) |
| Wine | 2.77% | 2.51% | not used | 2.7546% (96) | 2.0370% (96) |
| Heart* | 17.48% | 16.36% | not used | 43.7917% (96) | 42.6389% (96) |
| Iris | 3.10% | 2.87% | 3.0702% (96, pre-parity) | 4.3311% (96, active polynomial) | 3.1250% (96) |

References: SVM_Paper_Analysis.md, Reproduction_Report_Robust_SVM.md,
V7_CROSS_DATASET_REPORT.md, IRIS_EXPERIMENT_REPORT.md. These columns have
different seeds/protocols and are NOT paired V8 comparisons. Iris manual q1
RBF switch was 3.3717%. The checked-in robust MATLAB program does not directly
reproduce 2.87%; it must not be presented as reproduced by this work.
*Heart uses the local five-class, 297-complete-case dataset; equivalence to the
paper's apparent binary experiment is unresolved. No forced label collapse.

## Protocol, gates, and leakage controls

Use the existing fixed preprocessing: min-max Parkinson/Breast; standardize
Blood/Mammographic/Wine/Heart; no transform on reconstructed authors' Iris.
All fitted statistics are train-only and refitted inside every inner fold.
Sorted stratified outer 75/25; three-fold inner CV seed `20000+outer_seed`.
The eight alpha rules, four C values, solver tolerance and native OVO vote/tie
semantics remain fixed. One globally selected weighting family applies across
datasets; no dataset-specific performance-driven switch is permitted.

Development: 9000--9023, nested smoke/8/24. Final 10000--10095 was reserved
until C passed the predeclared guard/target/runtime criteria in
V8_EXPERIMENT_PLAN.md, and is now consumed. Repeated splits overlap heavily, so paired t intervals
and Wilcoxon tests describe conditional split variation, not independent
patient-population replication. A resampled-variance sensitivity interval is
also reported. Parkinson row holdout does not establish new-subject validity.

## Reproducibility

Implementation: `ddr_mksvm/v8_class_sensitive.py`; orchestration:
`run_v8_class_sensitive.py`, `modal_v8_class_sensitive.py`; artifact audit:
`analyze_v8_class_sensitive.py`; Windows-safe mirror: `sync_v8_artifacts.py`.
CPU-only Modal volume `ddr-mksvm-v8-class-sensitive-results`, 4 CPU/4096 MiB,
four workers, single-threaded BLAS. Pinned NumPy 1.26.4, SciPy 1.17.1,
scikit-learn 1.9.0, pandas 3.0.5, joblib 1.6.0, CVXPY 1.6.7, Python 3.12.
CVXPY is an import dependency of frozen research code, not the V8 solver.

Every V8 checkpoint contains inner fold indices, transforms, alpha/C/ratio grids,
all three validation metrics and predictions, weights/counts/penalties,
solver status/iterations, selected parameters, outer votes/margins/predictions,
split hashes and class confusion metrics. Frozen V7 retains its existing
diagnostic schema: selected parameters, inner losses and solver diagnostics,
outer predictions/confusions, but no outer margins/votes or inner prediction
vectors. Aggregate files hold raw rows,
summary, paired comparisons, pooled classes, config/code/data hashes, app ID
and runtime. Source snapshots and exact compatible checkpoint reuse preserve
provenance. No completed run is overwritten. Runtime per variant includes its
standalone search plus refit; shared-bank experiment wall time is counted once.
V7's support-vector count is native multiclass unique support rows; explicit
V8 pair counts sum across pairs and may count one observation repeatedly.
These diagnostic counts are not an apples-to-apples complexity comparison.
V8 also caches fold transforms/bandwidths while frozen V7 recomputes them;
runtime differences cannot be attributed solely to mathematical weighting.

## Eight-seed ablation results (development only)

Entries below are error / balanced accuracy / macro F1, in percent. Exact
per-seed metrics, intervals and class confusions are in the gate8 artifacts.

| Blood variant | Error | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| V7 | 22.3262 | 58.9995 | 59.6252 |
| selection-only balanced | 21.8583 | 61.9039 | 63.3316 |
| selection-only F1 | 21.7246 | 61.6124 | 63.0131 |
| A: inverse/error | 27.2059 | 58.4199 | 57.6477 |
| B: inverse/balanced | 31.5508 | 69.1270 | 63.8522 |
| inverse/F1 | 29.9465 | 67.4841 | 64.1394 |
| C: sqrt/balanced | 21.8583 | 66.5275 | 67.5396 |
| D: learned ratio/balanced | 35.9626 | 68.3887 | 61.0594 |

The objective/selection interaction matters. Inverse weighting with error
selection does not improve Blood balance; balanced selection alone improves it
modestly. Both mechanisms together improve recall strongly, but inverse weights
also cause many false positives. Square-root attenuation preserves more of the
accuracy/precision while improving recall. This is evidence for a cost-sensitive
tradeoff, not a universal accuracy improvement.

Heart's rarest-class pooled recall is 0% V7, 54.17% B, 8.33% C, 62.50% D.
Corresponding precisions are 0%, 12.75%, 8.70%, 13.04%. High rare-class recall
alone would conceal poor specificity and low precision. Both B and C improve
macro F1 on the development gate, with a modest overall-error cost.

B and C satisfy the predeclared target and Parkinson/Wine guards and advance
to 24 seeds. D fails the Parkinson error guard (+1.7857 pp) and is computationally
unattractive (39.37x V7 Blood runtime). It is pruned without solver retuning.
Mammographic B/C error deltas are -0.8413/-0.3005 pp but their paired intervals
include zero. No kernel change has been introduced.

## Twenty-four-seed decision (development only)

| Dataset | V7 error | C error | C-V7 error (pp) | C-V7 balanced accuracy (pp) | C-V7 macro F1 (pp) |
|---|---:|---:|---:|---:|---:|
| Blood | 21.7692% | 22.1480% | +0.3788 | +5.7716 | +5.9978 |
| Mammographic | 16.5665% | 16.3261% | -0.2404 | +0.2799 | +0.2711 |
| Heart | 42.9444% | 44.5556% | +1.6111 | +3.3806 | +4.1194 |
| Parkinson | 5.9524% | 6.7177% | +0.7653 | +0.4317 | -0.5063 |
| Wine | 1.9444% | 2.1296% | +0.1852 | -0.0540 | -0.1583 |
| Iris | 3.9474% | 4.0570% | +0.1096 | -0.1068 | -0.1100 |
| Breast Diagnostic | 2.6224% | 2.9720% | +0.3497 | -0.2939 | -0.3746 |

C qualifies for a single untouched confirmation under the prospectively fixed
screening bounds. Its Blood balanced-accuracy paired interval is
[+3.3477,+8.1955] pp (24/0/0 favorable splits); Heart macro-F1 interval is
[+1.8317,+6.4071] pp (17/0/7). Inverse B is rejected: Parkinson error rises
1.9558 pp, beyond the 1 pp guard, while Blood error rises 9.8708 pp.

Passing a point-estimate tolerance is not proof of noninferiority. In particular
Breast error's paired interval [+0.0743,+0.6250] pp warns that C may erase V7's
small historical Legacy advantage. This must not be hidden behind Blood recall.
No claim that all V7 gains are preserved is warranted at this stage.

Frozen candidate: square-root pair weights, balanced inner selection, unchanged
RBF/C grids, transforms, solver, folds and OVO aggregation. The full manifest
`results/v8/analysis/deterministic_candidate_freeze.json` predates evaluation
of seeds 10000--10095. Blood balanced accuracy is the primary confirmation
endpoint; Heart macro F1 is secondary. All seven datasets are matched to V7.
No confirmation-driven hyperparameter or architecture changes are permitted.

## Untouched 96-seed confirmation: final comparison

Run `final96/v8-final96-20260909`, completed by Modal app
`ap-j0ihyhyO3oUWKLgFWuP54V`: 672 matched dataset/split checkpoints, 1,344 model
records, seven datasets, seeds 10000--10095, variants `v7,v8_c` only.
The resumed job reused exact checkpoints after client connection failures;
there was no second candidate, seed replacement, or confirmation retuning.
This analysis continuation used only the locally synced artifacts: no training,
refitting, resync, or additional Modal spending.

Errors and balanced accuracy (BA) below are percentages; deltas are percentage
points (V8-C minus V7). Macro F1 is the mean of split macro F1, not F1 computed
from the pooled confusion matrix. Recall changes in this table are pooled.
Runtime ratio is total standalone V8-C time / total standalone V7 time.
The statistical column gives two-sided Wilcoxon results; full paired CIs follow.

| Dataset | V7 error | V8-C error | Δ error | V7 balanced acc | V8-C balanced acc | Δ macro-F1 | Key class-recall change | Runtime ratio | Statistical result | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| Blood Transfusion | 21.5798 | 21.7357 | +0.1560 | 61.3473 | 67.1678 | +5.5708 | class 1: 28.73 → 45.98 | 0.616x | BA p=3.33e-17; error p=.405 | strong class-sensitive gain |
| Mammographic Mass | 16.6066 | 16.5615 | -0.0451 | 83.2839 | 83.3398 | +0.0502 | class 1: 79.49 → 79.92 | 0.954x | error p=.712 | target not repaired |
| Heart Disease | 42.2639 | 44.1389 | +1.8750 | 30.3108 | 32.8639 | +2.8788 | class 5: 0.69 → 6.25 | 4.986x | F1 p=5.87e-7; error worse p=.000165 | balance/error tradeoff |
| Parkinson | 7.3129 | 7.4830 | +0.1701 | 88.6766 | 89.6784 | +0.1694 | class 0: 80.82 → 84.11 | 2.575x | error p=.938 | guard passes |
| Wine | 2.0139 | 2.2454 | +0.2315 | 98.0112 | 97.8414 | -0.2215 | class 3: 97.83 → 98.18; class 2 -0.58 pp | 3.931x | error p=.099 | point guard passes; uncertainty remains |
| Iris | 3.2895 | 3.2895 | 0.0000 | 96.7526 | 96.7504 | -0.0011 | class 2 +0.16; class 3 -0.16 pp | 4.763x | error p=1.000 | accuracy unchanged; slower |
| Breast Cancer Diagnostic | 2.4913 | 2.5204 | +0.0291 | 97.0270 | 97.0846 | -0.0258 | class 0: 95.17 → 95.56 | 1.464x | error p=.809 | guard passes |

Raw error counts V7 → C / predictions: Blood 3874 → 3902 / 17952;
Mammographic 3316 → 3307 / 19968; Heart 3043 → 3178 / 7200;
Parkinson 344 → 352 / 4704; Wine 87 → 97 / 4320; Iris 120 → 120 / 3648;
Breast Diagnostic 342 → 346 / 13728. Test appearances overlap across splits;
these are not that many independent people or distinct observations.

Historical Iris V7 remains **114/3648 = 3.1250%** on its original registry.
The new 120/3648 does not overwrite that result. Both new models exceed the
published 2.87%. The Legacy/manual-RBF columns above were not rerun in this
confirmation and must not be treated as matched to V8-C.

## Paired statistics and regression protection

The independently verified error deltas are:

| Dataset | Δ error (pp) | Paired 95% CI (pp) | C wins/ties/losses | Wilcoxon p | Paired effect d_z |
|---|---:|---:|---:|---:|---:|
| Blood | +0.1560 | [-0.3690, +0.6809] | 42/8/46 | .404843 | +0.0602 |
| Mammographic | -0.0451 | [-0.2595, +0.1693] | 36/30/30 | .712288 | -0.0426 |
| Heart | +1.8750 | [+1.0383, +2.7117] | 28/10/58 | .000165 | +0.4541 |
| Parkinson | +0.1701 | [-0.6453, +0.9854] | 25/48/23 | .938490 | +0.0423 |
| Wine | +0.2315 | [-0.0642, +0.5271] | 9/71/16 | .099007 | +0.1586 |
| Iris | 0.0000 | [-0.1730, +0.1730] | 3/89/4 | 1.000000 | 0.0000 |
| Breast Diagnostic | +0.0291 | [-0.1603, +0.2186] | 23/42/31 | .808623 | +0.0312 |

| Target metric | Δ (pp) | Paired 95% CI (pp) | C wins/ties/losses | Wilcoxon p | d_z |
|---|---:|---:|---:|---:|---:|
| Blood balanced accuracy (primary) | +5.8205 | [+4.9587, +6.6823] | 92/0/4 | 3.33e-17 | +1.3685 |
| Blood macro F1 | +5.5708 | [+4.4453, +6.6963] | 82/0/14 | 2.47e-15 | +1.0029 |
| Heart macro F1 (secondary) | +2.8788 | [+1.8911, +3.8666] | 69/0/27 | 5.87e-7 | +0.5905 |
| Heart balanced accuracy | +2.5532 | [+1.5521, +3.5542] | 63/0/33 | 1.37e-5 | +0.5168 |

The [complete independent table](../results/v8/analysis/confirmation_verification/verified_paired_metrics.csv)
contains all 28 dataset/metric comparisons (error, BA, macro F1, runtime),
including CIs, wins/ties/losses, Wilcoxon and standardized effects. The original
[21-row statistical artifact](../results/v8/final96/v8-final96-20260909/paired_comparisons.csv)
is unchanged. Recomputing from raw checkpoint floats reproduces every saved
Wilcoxon p-value; paired means, CIs, effects and runtime ratios agree within
1e-12. No favorable test was substituted for an unfavorable one.

Methods: d_i = metric_C - metric_V7; CI = mean(d) ± t(.975,95) s_d/sqrt(96);
d_z = mean(d)/s_d. Zero-variance effects are undefined (saved as null/blank),
except the nonzero-variance zero-mean Iris error effect is zero. Favorable is
negative for error/runtime and positive otherwise. Differences <=1e-14 count
as ties and are excluded from two-sided Wilcoxon. A supplemental rounded-to-12-
decimal Wilcoxon column makes discrete rank-tie sensitivity visible; it does
not replace the frozen statistics or change the qualification verdict.

The main endpoint was frozen before confirmation. Other p-values, including
per-class results, are secondary/exploratory and unadjusted for multiplicity;
they are not 21 independent superiority claims. Repeated holdouts are
conditional on the same small fixed datasets. For sensitivity, intervals
using s_d²(1/96+n_test/n_train) are also saved, with the actual integer split
ratio. The original table used the nominal 1/3 ratio; neither table is edited.
Blood BA remains positive under this conservative sensitivity [+0.8700,
+10.7709] pp; Heart F1 does not [-2.8325,+8.5901] pp. These sensitivity
intervals are not patient-population confidence guarantees either.

Qualification uses the predeclared **mean-error** guards, not post-hoc
noninferiority testing: Parkinson <=+1 pp; Wine/Iris/Breast <=+0.5 pp. All
four pass. Wine's upper paired bound +0.5271 pp exceeds +0.5 pp, so statistical
noninferiority at that margin is not established. The +0.3497 pp Breast
development warning shrinks to +0.0291 pp on confirmation, with CI spanning
zero. Preservation of V7's exact Legacy advantage is not established without
a new matched Legacy comparison; no such additional experiment is claimed.

## Class-wise benefits and costs

Pooled recall below uses summed confusion matrices. The corresponding paired
analysis averages each split equally; when class support varies (Blood/Iris),
the two estimands differ slightly. All 19 classes have split-level recall,
precision and F1 comparisons in
[paired_class_metrics.csv](../results/v8/analysis/confirmation_verification/paired_class_metrics.csv).
Complete pooled precision/recall/F1/support/confusions remain in
[class_metrics.csv](../results/v8/final96/v8-final96-20260909/class_metrics.csv).

| Dataset / class | Pooled recall V7 → C (%) | Mean paired recall Δ (pp) | Paired 95% CI (pp) | Recall W/T/L |
|---|---:|---:|---:|---:|
| Blood 1 (minority) | 28.7284 → 45.9794 | +17.2438 | [+15.5251,+18.9625] | 95/1/0 |
| Blood 0 | 93.9667 → 88.3648 | -5.6028 | [-6.1140,-5.0917] | 0/2/94 |
| Mammographic 1 | 79.4864 → 79.9196 | +0.4332 | [-0.1964,+1.0627] | 49/34/13 |
| Heart 1 | 93.0469 → 84.7396 | -8.3073 | [-9.5294,-7.0852] | 3/7/86 |
| Heart 2 | 17.1875 → 24.2560 | +7.0685 | [+5.0819,+9.0550] | 56/29/11 |
| Heart 3 | 19.5602 → 24.1898 | +4.6296 | [+1.7140,+7.5452] | 40/34/22 |
| Heart 4 | 21.0648 → 24.8843 | +3.8194 | [+1.4007,+6.2382] | 35/43/18 |
| Heart 5 (rarest) | 0.6944 → 6.2500 | +5.5556 | [+2.5087,+8.6024] | 15/80/1 |
| Parkinson 0 (minority) | 80.8160 → 84.1146 | +3.2986 | [+1.0872,+5.5101] | 31/56/9 |
| Parkinson 1 | 96.5372 → 95.2421 | -1.2950 | [-1.8948,-0.6953] | 6/61/29 |
| Breast 0 (malignant) | 95.1651 → 95.5582 | +0.3931 | [+0.0383,+0.7479] | 25/59/12 |
| Breast 1 (benign) | 98.8889 → 98.6111 | -0.2778 | [-0.4945,-0.0611] | 11/49/36 |

Blood C recovers 738 additional true minority positives, but incurs 766
additional majority errors. Minority precision falls 59.8345% → 55.2839%,
while minority F1 improves 38.8187% → 50.2042%. This explains how a large
balanced gain can coexist with 28 more total errors. Its recall gain survives
the resampled sensitivity interval [+7.3708,+27.1168] pp.

Mammographic class-1 recall's Wilcoxon p=.00801 differs from its mean-delta CI
crossing zero: discrete signed ranks and the mean test answer different
questions. The mean improvement is only +0.43 pp, overall error is essentially
unchanged, and this does not meet the declared >=0.5 pp error target. The
earlier Legacy/RBF mismatch is not repaired by class weighting; its original
Legacy comparison had only eight seeds and is not a matched 96-seed claim.

Heart C helps every non-majority class but costs 319 class-1 correct decisions;
net total errors increase by 135. Rarest-class precision is only 4.6512% →
6.7164%, with 18 true positives in 268 class-5 predictions (V7: 2/43).
Rarest-class F1 is 1.2085% → 6.4748%. There are only 13 distinct class-5
observations in the dataset, three per test split: the 288 pooled appearances
are not 288 different rare patients. This is a partial recovery, not a solved
rare-class problem, nor clinically useful performance. No newly collapsed
class is observed, but class sensitivity has a clear majority-recall cost.

Parkinson improves minority recall with a majority-recall/precision cost;
minority precision decreases 88.3302% → 85.1494%. Wine class recalls change
by -0.2778, -0.5787, +0.3472 pp for labels 1,2,3. Iris class 1 stays perfect;
labels 2 and 3 exchange two net errors. Breast malignant recall improves
slightly while benign recall declines. No threshold, weighting or metric was
retuned in response to these findings.

## Investigation of the 295 OVO vote ties

The original audit's counter sums available V8 traces. It is **295 V8-C outer
test ties**, not V7+C combined and not 295 zero decision values. An independent
scalar reconstruction from every saved pair margin verifies every vote and
prediction. Binary margins are positive for the second sorted class, exact
zero votes second, pairs are lexicographic, and aggregate ties choose the
smallest sorted class among the tied maxima. This is the frozen convention;
neither the implementation nor voting rule has changed.

| Dataset | V8-C vote ties / predictions |
|---|---:|
| Heart | 294 / 7200 (4.0833%) |
| Wine | 1 / 4320 (0.0231%) |
| Iris | 0 / 3648 |
| All four binary datasets | 0 |

Every tied example contains a directed three-class voting cycle. A simple
valid case is A beats B, B beats C, C beats A, giving 1/1/1 votes. No pair
margin in the entire final V8-C trace is zero or within 1e-9 of zero. Across
tied examples, the minimum absolute pair margin is 0.0005268351. Therefore
these are genuine disagreement cycles, not near-zero numeric sign accidents.
Of 295 tied predictions, 97 are correct; the true class belongs to the tied
maximum set in 191. This is descriptive error analysis, not evidence that an
alternative tie breaker would generalize. The counts reflect test appearances.

The frozen native V7 path did not save vote/margin traces on these splits.
Its tie count cannot be recovered from predictions alone; no expensive refit
was performed to manufacture it. Existing neutral-pair/native parity tests
and the saved V8 vote replay support the implementation. The explicit voting
code agrees with the first-maximum rule documented in the design and upstream
[libsvm source](https://github.com/cjlin1/libsvm/blob/master/svm.cpp).
The new regression tests replay all 295 cases. Full case-level evidence:
[vote_ties.csv](../results/v8/analysis/confirmation_verification/vote_ties.csv).
**No confirmed voting bug; no rule change is warranted by this audit.**

## Runtime, solver diagnostics and credit accounting

Runtimes include training-only selection and outer refit, not prediction alone.
Paired timing tests measure this execution order/environment, not an unbiased
hardware-independent performance law; V7 ran first, then C, on shared workers.

| Dataset | Mean V7 / C seconds | Median V7 / C | P95 V7 / C | Mean Δ seconds [paired 95% CI] |
|---|---:|---:|---:|---:|
| Blood | 2.331 / 1.436 | 1.907 / 1.422 | 4.292 / 1.650 | -0.896 [-1.119,-0.673] |
| Mammographic | 19.433 / 18.535 | 14.599 / 14.614 | 58.051 / 45.802 | -0.898 [-3.096,+1.299] |
| Heart | 0.351 / 1.748 | 0.349 / 1.739 | 0.367 / 1.797 | +1.397 [+1.393,+1.402] |
| Parkinson | 0.174 / 0.447 | 0.173 / 0.442 | 0.180 / 0.471 | +0.273 [+0.271,+0.275] |
| Wine | 0.173 / 0.680 | 0.173 / 0.677 | 0.176 / 0.699 | +0.507 [+0.505,+0.509] |
| Iris | 0.135 / 0.642 | 0.135 / 0.637 | 0.137 / 0.660 | +0.507 [+0.505,+0.509] |
| Breast Diagnostic | 0.482 / 0.706 | 0.481 / 0.703 | 0.497 / 0.727 | +0.224 [+0.222,+0.226] |

Runtime W/T/L: Blood 88/0/8, Mammographic 46/0/50, all other datasets 0/0/96.
Blood paired runtime p=1.59e-15, Mammographic p=.980; the other five p=1.78e-17.
Very large standardized timing effects on the small datasets primarily reflect
stable overhead, not large absolute costs. Maximum mean ratio is Heart 4.986x;
all pass the predefined practical screen. Explicit Python pair overhead makes
C substantially slower on multiclass data despite unchanged QP asymptotics.
There is no overall speedup claim.

Independent audit checks 65,184 successful V7 estimator diagnostic entries
(each multiclass estimator solves several pairs) and 186,240 successful V8
binary QPs including outer refits. Both algorithms together solve 372,480
binary QPs. There are no failed statuses in these final records. Original
audit's 184,320 count covers V8 inner-bank QPs only. All 690 local final-bundle
files, including derived outputs, have hashes in the promoted freeze.

Completed development jobs, counting reused fits once: 2,495.703 worker-wall
seconds. Completed final jobs: 4,547.834 worker-wall seconds (~1.263 worker
hours); combined ~1.957 worker hours. This is a compute-work proxy, not a
Modal invoice. Final `stage_wall_s=955.964` covers the successful resume only;
`new_compute_wall_s` on same-stage resume actually includes the saved completed
jobs from earlier apps. Do not add it again or label it resume-only spending.
Aborted in-flight work, setup, serialization and idle allocated time are not
fully measured, so actual billed allocation cannot be reconstructed exactly.
All experiments were CPU-only. This analysis continuation spent zero additional
Modal training credits.

## Final qualification, freeze and reproduction

All predefined requirements are met: primary Blood BA +5.8205 pp with positive
paired CI and minority recall gain; secondary Heart F1 +2.8788 pp; all four
mean-error guards pass; one sqrt pair-weight rule applies everywhere; runtime
is practical and no new class collapse occurs. Mammographic does not improve
meaningfully and Heart error worsens, but neither was a requirement for an
overall-error win on every target. We neither tighten nor relax these rules
after seeing the confirmation.

**V8-C qualifies as the new deterministic architecture.** More precisely, it
is the confirmed class-sensitive research reference, not a mandate to choose
it over V7 when unweighted total error or latency is the operational objective.
Frozen V7 remains a distinct, regression-protected baseline. The supported
method is a standard weighted RBF SVM with pair-normalized sqrt penalties and
balanced nested selection. The controlled evidence is useful, but novelty or
a high-tier publication is not established by renaming weighted SVM.

Promoted manifest:
[deterministic_v8_c_final_freeze.json](../results/v8/analysis/deterministic_v8_c_final_freeze.json),
SHA-256 `a325d998ea0eb19590b515586d1a00ba179d13772dbbb92621d0e8fe5642c2a4`.
It includes full configuration, all seeds, source snapshots/code hashes,
690 result hashes, verification hashes, metrics, qualification and limitations.
The original preconfirmation candidate freeze remains unchanged.

Key SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| V8 model | `fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448` |
| V7 model | `39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d` |
| Final config | `032b1ac773e4ea742d7fe79fc6de199c41fee902af02982acb3c8f8edfda8a40` |
| Per-run results | `121dedb9f95e54db3635794c042345e0a6855d709d5c8e57ca12938b82c5e736` |
| Split registry | `b86391c3feabb04d2ac703aa6f53d09fbb407e41513ab8d1f5840230d612bbfd` |
| Paired results | `06651bf1d8774dfa57e89ba46b0d75227f5edb9b1997396146a23308b7737302` |

Read-only reproduction of this analysis (no training):

```powershell
python run_v8_class_sensitive.py --freeze
python verify_v8_confirmation.py results/v8/final96/v8-final96-20260909
python analyze_v8_confirmation.py results/v8/final96/v8-final96-20260909
python freeze_v8_confirmation.py results/v8/final96/v8-final96-20260909
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
```

The independent analysis environment was Python 3.12.6, NumPy 1.26.4,
SciPy 1.15.2 and pandas 2.2.3; raw frozen training versions are in config above.
Use this analysis environment to reproduce byte-identical derived outputs;
other versions may vary slightly in distribution quantiles. Existing analysis
files refuse overwrite. An unrelated auto-loaded Hydra/OmegaConf/ANTLR plugin
fails locally before test collection; disabling external pytest plugin autoload
avoids it without modifying project dependencies or model code.
Final regression result: **100 tests pass**, with one unrelated pytz
deprecation warning. Five new tests cover independent vote replay, invalid
scores, paired-statistic edge cases, all saved final votes/artifact hashes,
and frozen V8 code identity. No robust tests were added or run.

The historical Modal command and pinned training environment are in
MODAL_EXPERIMENTS.md. **Do not rerun the completed final96.** Any future
independent confirmation must freeze new rules first and use new untouched
splits/data, not claim 10000--10095 is still untouched.

## Next phase decision: robust work blocked pending authoritative MATLAB source

Deterministic qualification opens, but does not automatically validate, a
robust counterpart. Further investigation is scientifically reasonable, but
the user explicitly paused the robust phase until supplying the authoritative
original MATLAB robust implementation. There is **no authorization to begin
robust execution now**, even at smoke scale. The note
[V8_ROBUST_MATHEMATICAL_DESIGN.md](V8_ROBUST_MATHEMATICAL_DESIGN.md) is expressly
provisional: it records candidate mathematical identities and questions, not
an approved implementation specification or claim of MATLAB/paper parity.

No robust implementation, robust experiments, kernel adaptation or alternative
voting search was started in this confirmation-analysis continuation. Robust
V8 has **no measured result and no qualification claim**. The next action is
to wait for that source, then audit its exact objective/constraints, p=1/2/
infinity, radii, kernel-to-feature transformation, multiclass and rho selection,
solver, threshold, preprocessing and split semantics against the paper.
Only after authoritative robust parity may the final Robust V8 design,
implementation and credit-gated experiments be considered. Mammographic kernel mismatch
is a separately documented unresolved hypothesis, not permission to combine
new mechanisms into the now-frozen C result.
