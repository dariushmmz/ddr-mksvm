# Expanded dataset study — prospective experiment protocol

Frozen: 2026-09-20, before any expanded-study model fit

## Objective and exclusions

The study broadens dataset coverage without changing model architecture or
searching for favorable datasets. The robust track remains closed. No robust
model, kernel search, feature-pipeline search, or post-test hyperparameter
change is authorized.

New computation is restricted to:

1. `breast_cancer_recurrence`: all three frozen model families;
2. `dermatology`: all three models, adding the missing V8-C comparison.

All other usable local datasets reuse their existing immutable artifacts.

## Frozen models

| Label in this study | Frozen semantics |
|---|---|
| `legacy` | q1 coefficient-L1 deterministic source profile, five `nu` values in source order and source threshold/tie behavior |
| `v7` | Gaussian RBF, RKHS-L2, native OVO, existing eight alpha rules and four-C grid, mean validation-error selection |
| `v8_c` | the frozen V7 kernel/grid plus mean-one square-root inverse-frequency OVO pair weights and balanced-loss inner selection |

Frozen implementation files are not edited. The expanded adapter invokes their
existing optimization, weighting, voting, grids, and tie rules.

## Dataset-specific preprocessing

### Breast Cancer recurrence

- Repair the six unambiguous spreadsheet-rendered interval tokens listed in
  `DATASET_INVENTORY.md` before splitting.
- Treat all nine predictors as categorical/ordinal identities; do not impose
  artificial interval distances.
- Represent blanks as `__MISSING__`.
- Fit one-hot categories inside every inner-training fold, transform its
  validation fold with `handle_unknown=ignore`, then fit standardization on the
  encoded inner training data only.
- Refit the encoder and standardization on the full outer training split for
  final test prediction.
- Preserve the source registry's homogeneous-linear Legacy kernel profile.

### Dermatology

- Retain the frozen complete-case cohort of 358 rows.
- Retain the source `none` transform, inhomogeneous-quadratic Legacy profile,
  and unchanged V7/V8-C preprocessing.

No preprocessing choice may change after results are observed.

## Splits and seed registry

Every outer split is sorted, stratified 75/25 using the frozen split routine.
All three models use exactly the same split. Inner selection uses the frozen
three-fold stratified registry `random_state=20000+outer_seed`.

| Stage | Seeds | Initial status |
|---|---|---|
| smoke | 16000 | authorized first |
| Gate8 | 16100-16107 | blocked until smoke passes |
| Gate24 | 16200-16223 | blocked until the dataset passes Gate8 |
| confirmation | 16300-16395 | untouched; blocked until Gate24 passes |

These ranges are dedicated to this study and may not be described as untouched
after their respective stage is launched.

## Artifact and failure policy

Each dataset/seed job writes an immutable atomic checkpoint containing split
indices and hashes, predictions and hashes, confusion matrices, class metrics,
selected hyperparameters, preprocessing categories/scales and hashes, runtime,
solver diagnostics, and source/config/dataset fingerprints. Failures receive a
separate immutable record and count against completion; they are never skipped.

The Modal policy is 4 CPU, no GPU, no more than four parallel jobs, explicit
timeouts, persistent volume storage, compatible-config resume, and rejection
of an existing run name with a different fingerprint.

## Prospective stage gates

### Smoke

Pass per dataset only if all three models return finite objectives/metrics and
predictions, splits match, every reported solver status is accepted, hashes
verify, and no task is missing. Majority-only prediction is reported but is
not hidden or automatically repaired.

### Gate8 to Gate24

Numerical completion must be 100%. A dataset advances if at least one of these
predeclared comparisons passes:

- **V7 versus Legacy:** mean error delta no worse than `+0.5` percentage point,
  runtime ratio at most 10, and either error is nonpositive or balanced
  accuracy/macro-F1 improves by at least `0.5` point with wins at least losses
  on that improved metric.
- **V8-C versus V7:** mean error delta no worse than `+1.0` point, runtime ratio
  at most 10, and balanced accuracy, macro-F1, or rare-class recall improves by
  at least `0.5` point with wins at least losses on the improved metric.

This is a screening/spending rule, not confirmation evidence.

### Gate24 to 96-seed confirmation

Numerical completion again must be 100%. A comparison qualifies only if:

- **V7 versus Legacy:** mean error is lower with wins at least losses; or error
  is within `+0.25` point while both balanced accuracy and macro-F1 improve by
  at least `0.5` point.
- **V8-C versus V7:** mean error is within `+1.0` point, the chosen
  balance-sensitive endpoint improves by at least `1.0` point (or rare-class
  recall by at least `5.0` points), wins exceed losses on that endpoint, and no
  class loses more than `5.0` recall points on average.

Runtime ratio must remain at most 10 and no model may be systematically
majority-only. The endpoint is fixed by the first Gate8 condition satisfied;
it cannot be changed after Gate24 results.

If neither comparison qualifies, the dataset stops at Gate24. Confirmation is
never launched merely because a result is interesting or statistically
significant.

Confirmation is comparison-specific: only the promoted candidate and the
reference needed for that comparison are fitted. A fixed control that fails
its Gate8 or Gate24 promotion rule is not carried into the untouched
confirmation range. Thus, when only `v7_vs_legacy` qualifies, confirmation
contains exactly `legacy,v7`; `v8_c` remains reported from the completed gates
but does not consume confirmation seeds.

## Metrics and interpretation

Primary V7 endpoint: paired test error versus Legacy.  
Primary V8-C endpoint: paired balanced accuracy versus V7.  
Secondary: macro-F1, per-class and rare-class recall, error guard, runtime,
solver success, and majority-only frequency.

Gate8 estimates are screening diagnostics. Gate24 estimates support promotion
decisions. Only untouched 16300-16395 results, if authorized, are confirmation
evidence. Negative outcomes and incomplete gates remain in the final paper
inventory.

## Final full-study integration rule

This expanded protocol supplements rather than invalidates the earlier frozen
V7, Iris, and V8-C protocols. Completed immutable confirmations are reused;
they are not rerun on new seeds merely to place them under one directory.
Development gates remain separated from untouched confirmation estimates.

The final paper matrix is complete when each usable dataset has:

1. a prospectively stopped Legacy/V7 path, either at its failed gate or at
   96-seed confirmation; and
2. V8-C evidence where applicable, either a completed frozen confirmation or
   an explicit prospective stop as a fixed control.

Under this rule all nine usable datasets are complete. Blood and Mammographic
failed the Legacy/V7 Gate8 and therefore must not receive later Legacy/V7
stages. Breast recurrence and Dermatology failed V8-C promotion and therefore
must not receive V8-C confirmation. The other authorized branches already
have 96-seed artifacts. `stage_registry.csv` records the exact historical
seed ranges without pretending that all campaigns used a single registry.
