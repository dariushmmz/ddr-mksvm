# Expanded dataset study — final cross-dataset results

Status: complete. Smoke, Gate8, Gate24, and the authorized 96-seed
confirmations have finished. No architecture or gate was changed after seeing
test results, and the closed robust track was not reopened.

## Dataset disposition

The local directory contains nine usable classifier datasets and one metadata
file. All nine classifier datasets are retained in the paper evidence base;
`uci_fetch_report.csv` is not observational data and is excluded. Existing
immutable experiments were reused for seven datasets. New staged computation
covered Breast Cancer recurrence and the missing Dermatology V8-C control.

| Dataset | Legacy vs V7 evidence | V8-C evidence | Final paper role |
|---|---|---|---|
| Parkinson | 96-seed confirmation; V7 large error/runtime win | 96-seed control; modest minority-recall gain | primary V7 result |
| Blood Transfusion | Gate8 negative for V7 | 96-seed class-sensitive gain | primary V8-C result and V7 negative |
| Mammographic Mass | Gate8 negative for V7 | 96-seed null result | retained negative/control |
| Breast Cancer Diagnostic | 96-seed modest V7 error win | 96-seed control | confirmed V7 generalization |
| Wine | 96-seed V7 error/runtime win | 96-seed control | confirmed V7 generalization |
| Heart Disease | 96-seed V7 error/runtime win under the local five-class protocol | 96-seed balance/error tradeoff | V8-C tradeoff; absolute-accuracy caveat |
| Iris | 96-seed V7 error win over executable Legacy | 96-seed neutral/slower control | multiclass confirmation |
| Dermatology | **new 96-seed confirmation** | fixed control stopped at Gate24 | small error and large runtime win; rare-class caveat |
| Breast Cancer recurrence | **new 96-seed confirmation** | fixed control stopped at Gate24 | error/runtime win; substantial minority-recall caveat |

## Expanded-stage accounting

| Stage | Seeds | Jobs / model records | Failures | Decision |
|---|---|---:|---:|---|
| Smoke | 16000 | 2 / 6 | 0 | both datasets passed |
| Gate8 | 16100–16107 | 16 / 48 | 0 | both advanced through V7-vs-Legacy error only; V8-C did not advance |
| Gate24 | 16200–16223 | 48 / 144 | 0 | both V7 comparisons passed; V8-C remained a stopped fixed control |
| Confirmation | 16300–16395 | 192 / 384 | 0 | authorized `legacy,v7` comparisons completed |

The first Gate24 coordinator (`ap-voMqRfsS75vjF7C2TQXN8J`) failed before any
model fit because empty CSV cells became `NaN` during strict JSON
serialization. A null-safe metadata-only fix and regression test were applied.
No Gate24 seed was consumed by that app. The identical recovery run completed
under `ap-3AeTnx7MkbF8u4NFiEbWsY`.

## Gate24 results

Values are candidate minus reference. Confidence intervals and tests are
screening evidence at this stage.

| Dataset | Comparison | Delta error | Delta BA | Delta macro-F1 | Delta rare recall | Error W/T/L | Runtime ratio | Frozen decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Breast recurrence | V7 − Legacy | **−3.6458 pp** | −2.1534 pp | −2.8702 pp | −16.0714 pp | 19/0/5 | 0.549x | pass error endpoint |
| Breast recurrence | V8-C − V7 | +1.0995 pp | +5.2930 pp | +6.2894 pp | +20.6349 pp | 6/3/15 | 1.934x | fail; Gate8 already failed and error guard exceeded |
| Dermatology | V7 − Legacy | **−0.3241 pp** | −0.1675 pp | −0.0021 pp | −3.3333 pp | 12/4/8 | 0.097x | pass error endpoint |
| Dermatology | V8-C − V7 | +0.0463 pp | +0.0231 pp | −0.0157 pp | 0 | near-identical | 6.088x | fail; no qualifying balance gain |

For Breast recurrence, the Gate24 V7 error CI was
`[−5.6834, −1.6082]` pp (`p=.00163`, paired standardized effect `−0.756`).
For Dermatology it was `[−1.1711, +0.5229]` pp (`p=.323`), so promotion rested
on the prospective mean-error/W-T-L rule rather than significance. V8-C is
reported as a fixed control only and was not fitted on confirmation seeds.

## Untouched 96-seed confirmation

Primary paired comparison: V7 minus Legacy/source-profile. All 96 splits per
dataset are matched. Negative error/runtime deltas favor V7; positive
balanced-accuracy/F1/recall deltas favor V7.

| Dataset | Metric | Legacy | V7 | Paired delta | 95% paired CI | W/T/L | Wilcoxon p | Effect |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Breast recurrence | Error | 30.6424% | **26.8084%** | **−3.8339 pp** | [−4.7142, −2.9537] | 80/4/12 | 1.24e−10 | −0.882 |
|  | Balanced accuracy | 59.0978% | 59.3239% | +0.2261 pp | [−1.1829, +1.6351] | 53/0/43 | .559 | +0.033 |
|  | Macro-F1 | 58.9709% | 58.6360% | −0.3349 pp | [−2.1808, +1.5110] | 57/0/39 | .822 | −0.037 |
|  | Minority recall (class 1) | 34.4742% | **26.0417%** | **−8.4325 pp** | [−12.0413, −4.8238] | 24/12/60 | 5.04e−5 | −0.473 |
| Dermatology | Error | 4.2708% | **3.8079%** | **−0.4630 pp** | [−0.9032, −0.0227] | 46/21/29 | .03498 | −0.213 |
|  | Balanced accuracy | 95.5339% | 95.4643% | −0.0696 pp | [−0.6250, +0.4858] | 49/5/42 | .800 | −0.025 |
|  | Macro-F1 | 95.1375% | 95.2672% | +0.1297 pp | [−0.4007, +0.6601] | 51/2/43 | .544 | +0.050 |
|  | Rare recall (class 6) | 98.9583% | **96.0417%** | **−2.9167 pp** | [−4.4700, −1.3633] | 1/80/15 | .000465 | −0.380 |

The Breast recurrence error reduction comes primarily from class 0 recall
increasing by `+8.8848` pp; it is not a balance-sensitive improvement. V7 had
one majority-only Breast prediction out of 96 versus zero for Legacy, which is
not systematic but is disclosed. Dermatology class-recall changes were mixed:
classes 1, 3, and 5 improved, class 6 declined, and the remaining class CIs
included zero.

## Runtime and numerical reliability

| Dataset/model | Mean | Median | p95 | Maximum | Total |
|---|---:|---:|---:|---:|---:|
| Breast recurrence Legacy | .5831 s | .5824 s | .6241 s | .6349 s | 55.97 s |
| Breast recurrence V7 | .3428 s | .3400 s | .3632 s | .3892 s | 32.91 s |
| Dermatology Legacy | 3.9898 s | 4.0046 s | 4.2834 s | 4.4251 s | 383.02 s |
| Dermatology V7 | .4421 s | .4423 s | .4741 s | .4867 s | 42.44 s |

V7 was about `1.70x` faster on Breast recurrence and `9.02x` faster on
Dermatology. Confirmation logged 3,360 successful HiGHS calls for Legacy and
18,624 `libsvm-smo/ok` calls for V7. No failure or nonfinite objective, score,
prediction, or metric occurred. Stage wall time was 137.31 seconds on four
Modal CPUs; aggregate job wall time was 518.84 seconds.

## Scientific interpretation

Both new datasets strengthen the paper, but for different reasons. Breast
recurrence adds a statistically clear V7 error/runtime result and an equally
clear minority-recall limitation. Dermatology independently repeats the small
V7 error advantage with a large runtime reduction, while revealing a rare
class-6 recall cost. Neither result supports presenting V7 as universally
class-sensitive. V8-C remains supported by its earlier Blood and Heart
evidence; these new Gate24 controls did not satisfy their prospective
promotion rules.

The recommended paper dataset set is all nine usable classifier datasets,
organized by role rather than filtered by wins: confirmed V7 gains, confirmed
V8-C class-sensitive gains/tradeoffs, and prospective negative controls. This
is broader and more credible than selecting only favorable datasets.

The complete inferential tables, including all 56 paired metric rows and 42
paired class-recall rows, are in `STATISTICAL_ANALYSIS.md` and the consolidated
machine-readable analysis directory. Every usable dataset has 96-seed evidence
in at least one authorized comparison; stopped branches remain stopped.

## Evidence

- Gate24: `results/expanded_dataset_study/gate24/expanded-gate24-16200-16223-v1/`
- Confirmation: `results/expanded_dataset_study/confirmation/expanded-confirmation-16300-16395-v1/`
- Inventory and hashes: `results/expanded_dataset_study/inventory/`

Frozen hashes remained unchanged:

- V7: `39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d`
- V8-C: `fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448`
