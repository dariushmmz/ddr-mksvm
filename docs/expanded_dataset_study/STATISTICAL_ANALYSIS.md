# Expanded dataset study — final statistical analysis

## Scope and conventions

This report consolidates immutable confirmation artifacts; it does not pool
development gates with confirmation estimates. Candidate-minus-reference
deltas are reported throughout. Negative error/runtime deltas and positive
balanced-accuracy, macro-F1, and recall deltas favor the candidate. Confidence
intervals are paired t intervals over 96 matched splits. W/T/L uses the metric's
favorable direction. Wilcoxon values are retained from each source campaign's
frozen analysis artifact because tied discrete error counts can produce small
version-dependent differences when recomputed.

Seven datasets have 96-seed Legacy-versus-V7 confirmation. Seven have
96-seed V7-versus-V8-C confirmation. The union is all nine usable classifier
datasets. Blood and Mammographic were correctly pruned from Legacy-versus-V7
after Gate8 but later received independent V7/V8-C confirmation; Breast
recurrence and Dermatology stopped the V8-C branch at expanded Gate24.

## Legacy versus V7: 96-seed confirmation

| Dataset | Legacy error | V7 error | Delta (pp) | 95% paired CI (pp) | W/T/L | Wilcoxon p | Effect | Runtime ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parkinson | 14.2432 | **7.0791** | **−7.1641** | [−8.1609,−6.1674] | 85/9/2 | 1.95e−15 | −1.456 | .622 |
| Breast Cancer recurrence | 30.6424 | **26.8084** | **−3.8339** | [−4.7142,−2.9537] | 80/4/12 | 1.24e−10 | −.882 | .588 |
| Iris | 4.3311 | **3.1250** | **−1.2061** | [−1.8000,−.6123] | 40/39/17 | 1.37e−4 | −.412 | .370 |
| Heart Disease | 43.7917 | **42.6389** | **−1.1528** | [−1.9849,−.3207] | 54/13/29 | .00191 | −.281 | .159 |
| Wine | 2.7546 | **2.0370** | **−.7176** | [−1.2043,−.2309] | 44/31/21 | .00638 | −.299 | .300 |
| Dermatology (expanded) | 4.2708 | **3.8079** | **−.4630** | [−.9032,−.0227] | 46/21/29 | .03498 | −.213 | .111 |
| Breast Cancer Diagnostic | 2.8555 | **2.6005** | **−.2550** | [−.4746,−.0353] | 48/26/22 | .00656 | −.235 | .206 |

The error CIs exclude zero on all seven selected confirmation datasets. This
does not imply universal superiority: Blood and Mammographic failed the
prospective Legacy/V7 Gate8 and were never promoted to that confirmation.
Heart's absolute error remains poor under the local five-class protocol.

Balance-sensitive results qualify the error table:

- Parkinson also improved balanced accuracy by `15.92` pp and macro-F1 by
  `13.82` pp.
- Heart improved balanced accuracy by `3.33` pp and macro-F1 by `7.71` pp,
  although its rarest class remained poorly recovered by V7.
- Breast Diagnostic, Wine, and Iris had positive paired CIs for balanced
  accuracy and macro-F1.
- Breast recurrence's balanced-accuracy delta was only `+.23` pp with a CI
  crossing zero, while minority recall fell `8.43` pp.
- Expanded Dermatology's balanced accuracy was essentially unchanged and rare
  class-6 recall fell `2.92` pp. Its earlier independent 96-seed confirmation
  was statistically indistinguishable on error, so the small new error effect
  should be described as a replication-strengthened but modest result.

## V7 versus V8-C: 96-seed confirmation

| Dataset | Delta error (pp) | Delta BA (pp) | Delta macro-F1 (pp) | Error W/T/L | Runtime ratio | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Blood | +.1560 | **+5.8205** | **+5.5708** | 42/8/46 | .616 | strong class-sensitive gain; error unchanged |
| Heart | **+1.8750** | **+2.5532** | **+2.8788** | 28/10/58 | 4.986 | balance gain with significant error/runtime cost |
| Parkinson | +.1701 | +1.0018 | +.1694 | 25/48/23 | 2.575 | modest minority gain; primary CIs inconclusive |
| Mammographic | −.0451 | +.0560 | +.0502 | 36/30/30 | .954 | essentially null |
| Breast Diagnostic | +.0291 | +.0577 | −.0258 | 23/42/31 | 1.464 | essentially null |
| Wine | +.2315 | −.1698 | −.2215 | 9/71/16 | 3.931 | no supported benefit |
| Iris | .0000 | −.0022 | −.0011 | 3/89/4 | 4.763 | predictive tie, substantially slower |

Blood is the clearest V8-C result: balanced accuracy CI `[+4.9587,+6.6823]`
pp, W/T/L 92/0/4, p=`3.33e−17`, effect `+1.369`; macro-F1 CI
`[+4.4453,+6.6963]` pp, p=`2.47e−15`. Minority class-1 recall increased from
`28.73%` to `45.98%` (`+17.24` pp), while majority recall fell `5.60` pp.

Heart balanced accuracy CI was `[+1.5521,+3.5542]` pp (`p=1.37e−5`) and
macro-F1 CI `[+1.8911,+3.8666]` pp (`p=5.87e−7`). Rarest-class recall rose
from `.69%` to `6.25%`, but error also increased significantly by `1.875` pp
and runtime was nearly five times V7. This is a tradeoff, not a universal win.

Expanded Gate24 did not promote V8-C on Breast recurrence or Dermatology.
Breast showed balance gains but exceeded the frozen error guard; Dermatology
showed insufficient balance gain and a large runtime cost. These fixed-control
results are not mixed with untouched confirmation evidence.

## Runtime

V7 was faster than Legacy on every confirmed Legacy/V7 dataset: approximately
`1.61x` Parkinson, `1.70x` Breast recurrence, `2.70x` Iris, `3.33x` Wine,
`4.86x` Breast Diagnostic, `6.29x` Heart, and `9.02x` Dermatology.

V8-C was faster than V7 on Blood (`.616x`) and approximately neutral on
Mammographic (`.954x`), but slower on Breast Diagnostic (`1.46x`), Parkinson
(`2.57x`), Wine (`3.93x`), Iris (`4.76x`), and Heart (`4.99x`). Runtime is
therefore part of the model-selection conclusion, not ancillary reporting.

## Numerical reliability and multiplicity

The V7 final confirmation recorded 7,680 successful HiGHS LP calls and 46,560
successful libsvm fits with no warning, fallback, or failed final call. Three
development Gate8 LPs required the prospectively validated CLARABEL fallback
(two Blood, one Mammographic); these datasets' negative outcomes were retained.
The V8 final records contain no failed solver status. The expanded confirmation
added 3,360 optimal HiGHS and 18,624 successful libsvm calls with zero failure.
The Gate24 CSV serialization incident occurred before a fit and has a dedicated
regression test; it is not a scientific retry.

P-values are dataset/endpoint-specific descriptive evidence. No claim of a
single family-wise hypothesis test is made. The conclusions rely jointly on
prospective gates, effect direction and size, confidence intervals, class
metrics, runtime, and numerical completion.

## Machine-readable evidence

Consolidated tables are under
`results/expanded_dataset_study/analysis/cross_dataset/`:

- `confirmed_model_summary.csv`;
- `confirmed_paired_statistics.csv`;
- `confirmed_paired_class_recall.csv`;
- `confirmed_hyperparameter_frequency.csv`;
- `confirmed_solver_status.csv`;
- `stage_registry.csv`;
- `provenance.json`.

The original run-level predictions, confusion matrices, hyperparameters,
split hashes, runtimes, and diagnostics remain in their immutable source run
directories; the consolidation script records their SHA-256 hashes.

## Manuscript evidence map

Use these exact artifacts when building paper tables:

| Claim | Immutable experiment/result |
|---|---|
| Legacy/V7 confirmation: Parkinson, Breast Diagnostic, Wine, Heart | `results/v7_cross_dataset/v7-cross-final96-20260909/` |
| Original Dermatology Legacy/V7 confirmation replication | same V7 final96 run; preserved separately by the consolidator |
| Iris executable-Legacy/V7 confirmation | `results/iris_experiments/iris-v7-rkhs-l2-final96-20260908/` |
| Expanded Breast recurrence and Dermatology confirmation | `results/expanded_dataset_study/confirmation/expanded-confirmation-16300-16395-v1/` |
| Blood and Mammographic Legacy/V7 negative gate | `results/v7_cross_dataset/v7-cross-gate8-fallback-20260909/` |
| V7/V8-C seven-dataset confirmation | `results/v8/final96/v8-final96-20260909/` |
| Expanded V8-C fixed controls and stop decisions | `results/expanded_dataset_study/gate24/expanded-gate24-16200-16223-v1/` |
| Dataset definitions and hashes | `results/expanded_dataset_study/inventory/` |

Published-paper reference numbers must remain separate from these executable
results. Development Gate8/Gate24 values may explain promotion decisions but
must not be substituted for untouched confirmation estimates.
