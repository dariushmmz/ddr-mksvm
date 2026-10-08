# Expanded dataset study — experiment log

## EDS-000 — inventory and protocol freeze

Date: 2026-09-20  
Compute: read-only local inventory; no model fitting

The complete `dataset/` directory was audited. Nine classifier datasets and
one metadata report were identified. Eight classifier datasets already have
frozen experimental evidence. The categorical 286-row Breast Cancer recurrence
cohort was accepted as a separately named new benchmark after deterministic
semantic token repair and leakage-safe one-hot encoding. It is not treated as
the paper's 683-row numeric Breast Cancer dataset. Dermatology was selected for
a V8-C coverage extension.

The model definitions, preprocessing, seed ranges, metrics, artifact policy,
and stage gates in `EXPERIMENT_PROTOCOL.md` were frozen before expanded-study
training. Robust-model research remains closed.

Decision: implement an isolated adapter and invariant tests, then run only the
one-seed Modal smoke on seed 16000.

## EDS-001 — isolated implementation and regression

Date: 2026-09-20  
Compute: local tests only; no local scientific fitting

Added an isolated categorical adapter, immutable checkpoint runner, Modal
wrapper, read-only analyzer, inventory generator, and six focused invariant
tests. The adapter calls the frozen q1, V7, and V8-C primitives without editing
their files. The complete regression suite passed 176/176 tests with one
unrelated pytz deprecation warning. Frozen hashes remained:

- V7: `39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d`;
- V8-C: `fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448`.

Decision: authorize the one-seed smoke only.

## EDS-002 — Modal smoke

Date: 2026-09-20  
Run: `smoke/expanded-smoke-16000-v1`  
Modal app: `ap-L9oe7vO0ZL54y5CeLNfjct` (completed)

Both datasets and all three models completed on seed 16000: two immutable
dataset checkpoints, six model records, no failures, finite metrics, accepted
solver statuses, matching splits, valid prediction hashes, and no majority-only
model. Breast Cancer recurrence used 43 outer one-hot columns. The smoke is a
numerical check only; its predictive values are not used for model selection.

Decision: smoke passed for both datasets. Gate8 seeds 16100-16107 are
authorized under the already frozen protocol.

## EDS-003 — fresh Gate8

Date: 2026-09-20  
Run: `gate8/expanded-gate8-16100-16107-v1`  
Modal app: `ap-2wMq3rAEcZBYvt67kVyhHN` (completed)

All 16 dataset-seed jobs and 48 model records completed with no failure,
nonfinite metric, or majority-only prediction.

| Dataset | Comparison | Delta error | Delta BA | Delta macro-F1 | Delta rare recall | Runtime ratio | Gate8 decision |
|---|---|---:|---:|---:|---:|---:|---|
| Breast Cancer recurrence | V7 - Legacy | -4.6875 pp | -0.3676 pp | +0.3215 pp | -12.5000 pp | .528x | pass through error endpoint |
| Breast Cancer recurrence | V8-C - V7 | +2.4306 pp | +3.7115 pp | +3.8917 pp | +18.4524 pp | 1.948x | fail: error guard exceeded |
| Dermatology | V7 - Legacy | -0.2778 pp | +0.1257 pp | +0.1329 pp | 0 | .099x | pass through error endpoint |
| Dermatology | V8-C - V7 | -0.2778 pp | +0.3125 pp | +0.3147 pp | 0 | 6.146x | fail: balance gain below .5 pp screen |

Gate8 is screening evidence only. The Breast recurrence V7 error signal comes
with lower rare-class recall and must not be described as a class-sensitive
win. V8-C's Breast balance/recall signal is not promoted because its error cost
violates the prospectively frozen `+1.0` point guard.

Decision: both datasets may enter Gate24 only on the frozen **V7 versus Legacy
test-error endpoint**. V8-C remains a fixed contextual control. Gate24 seeds
16200-16223 are authorized; confirmation remains blocked.

## EDS-004 — Gate24 pre-fit serialization failure

Date: 2026-09-20  
Modal app: `ap-voMqRfsS75vjF7C2TQXN8J` (failed and stopped)

The first Gate24 coordinator stopped before writing a configuration or calling
any model because empty cells in the uploaded Gate8 decision CSV were parsed as
`NaN`, which strict JSON correctly rejected. No split, seed, model, or solver
was evaluated. The fix only converts empty promotion-metadata cells to JSON
`null`; it does not change data, preprocessing, models, grids, endpoints, or
gates. A dedicated regression test was added, and the focused suite passed 7/7
while the full suite passed 177/177.

Decision: relaunch the identical Gate24 run name, seeds, datasets, endpoint,
and resource policy. This is infrastructure recovery, not a scientific retry.

## EDS-005 — recovered Gate24 completed

Date: 2026-09-27  
Run: `gate24/expanded-gate24-16200-16223-v1`  
Modal app: `ap-3AeTnx7MkbF8u4NFiEbWsY` (completed)

The null-safe promotion-table fix was covered by regression and the identical
Gate24 configuration was relaunched. All 48 dataset-seed jobs and 144 model
records completed with no failure. The run fingerprint is
`f1a5df81bbd62f23355bcf57df3854535086691688f1bc8c6f6fc4d35f9130f1`.

Breast recurrence V7 reduced mean error by 3.6458 pp (19/0/5) at .549x
Legacy runtime, but rare-class recall fell 16.0714 pp. Dermatology V7 reduced
mean error by .3241 pp (12/4/8) at .097x runtime, with a 3.3333 pp rare-class
recall loss. Both passed the prospectively frozen V7 error rule.

V8-C failed its frozen promotion rule on both datasets and remains a fixed
Gate24 control. Breast V8-C improved balance and rare recall but increased
error by 1.0995 pp, just beyond the +1 pp guard. Dermatology V8-C was nearly
identical and 6.09x slower than V7.

Decision: authorize confirmation seeds 16300–16395 for `legacy,v7` only.

## EDS-006 — untouched 96-seed confirmation

Date: 2026-09-27  
Run: `confirmation/expanded-confirmation-16300-16395-v1`  
Modal app: `ap-G3j7oOxZyrSuFhsnDYvp9O` (completed and stopped)

The comparison-specific launch guard verified the immutable Gate24 table
before fitting. All 192 dataset-seed checkpoints and 384 model records
completed with zero failures. The artifact audit verified split separation,
matched model splits, prediction hashes, true-label hashes, error counts, file
hashes, and fingerprint consistency.

Breast recurrence confirmed the V7 error effect: −3.8339 pp, 95% CI
[−4.7142,−2.9537], W/T/L 80/4/12, Wilcoxon p=1.24e−10. Minority recall fell
8.4325 pp. Dermatology confirmed a smaller error effect: −.4630 pp, CI
[−.9032,−.0227], W/T/L 46/21/29, p=.03498. Rare class-6 recall fell 2.9167
pp. Runtime ratios were .588 and .111, respectively.

Decision: expanded computation is complete. Preserve both accuracy/runtime
benefits and class-recall harms in the paper; launch no further model search.

## EDS-007 — final audit and paper integration

Date: 2026-09-27  
Compute: read-only artifact analysis and local tests; no model fitting

Added paired per-class recall statistics and solver-status aggregation from
the preserved checkpoints. The complete regression suite passed 180/180 tests
with one unrelated `pytz` deprecation warning. All JSON artifacts parsed,
Gate24 retained 48 checkpoints, confirmation retained 192 checkpoints, both
frozen implementation hashes matched, and the Modal app was stopped with zero
active tasks. Final documentation and machine-readable indices were updated.

## EDS-008 — complete cross-campaign consolidation

Date: 2026-09-27  
Compute: local read-only analysis; zero model fits and zero Modal jobs

Re-hashed all ten dataset CSV files and confirmed the same nine usable
classifier datasets plus one metadata rejection. Audited the V7 cross-dataset,
Iris V7, V8-C, and expanded-study manifests and seed registries. Every usable
dataset has 96-seed confirmation evidence in at least one authorized frozen
comparison. Seven datasets have Legacy/V7 confirmation; seven have V7/V8-C
confirmation; the union is all nine.

Added `consolidate_expanded_dataset_study.py`, which reads immutable run-level
artifacts and writes comparison-specific summaries without pooling different
seed registries. Source artifact hashes, model summaries, paired statistics,
per-class recall statistics, hyperparameter frequencies, and the complete
stage registry are saved under
`results/expanded_dataset_study/analysis/cross_dataset/`.

Decision: no scientifically required training stage remains. Rerunning a
pruned branch would violate its prospective stop rule. Proceed to manuscript
tables and figures.
