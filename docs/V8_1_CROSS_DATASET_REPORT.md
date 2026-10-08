# V8.1 deterministic geometry investigation

2026-09-15. Development study; no new confirmation claim.

**Verdict: no deterministic candidate improves enough; stop architecture search.**
The sole candidate failed its complete Mammographic gate8 endpoint and runtime
guard. No gate24 or final96 was launched. V7 and V8-C remain the baselines.

## Scope and independent audit

V7 and V8-C remain frozen scientific baselines. The independent investigation,
mathematical derivation and prospective plan were completed and hash-frozen
before candidate implementation or training. Their design manifest is
`results/v8_1/analysis/design_freeze.json`, SHA-256
`f861600788124a20dac420231f8cdb441bd486810e4ef865ecb5862bad782beb`.
The baseline inventory protects 760 historical files. No historical experiment
was rerun. All robust work is blocked; none was implemented or tested.

The audit does not assume that the prior kernel recommendation was correct:

- Mammographic's historical Legacy advantage confounds geometry with objective,
  threshold and selection. On one consumed training partition, centered label
  alignment favors RBF (.4194) over quadratic (.1140); conflicting duplicate
  features imply at least 35/622 empirical training errors. Neither statistic
  proves a population error floor or the best kernel.
- Heart's majority pair 1-2 is difficult, but equal-weight pair 3-4 is also
  poor. Class 5 often beats class 1 yet loses to other minority classes. Global
  hyperparameter selection and weights both change; saved V7 pair margins
  are unavailable, so causal pair attribution is not claimed.
- Only 12/339 new Heart majority losses are vote ties. A tied-maxima oracle
  could repair at most 93/7200 predictions, not an attainable calibration
  result. The frozen voting rule is correct and unchanged.
- Constant class-specific margin targets reparameterize C and the prediction
  threshold; they do not establish novel margin geometry. Native global class
  weights cannot generally reproduce pair-normalized V8-C penalties exactly.

Full derivations, source citations and ranking of five directions are in
`V8_1_ASTRA_INVESTIGATION.md`. One candidate was chosen; no second family,
mixture, calibration, weight change or architecture redesign was implemented.

## Candidate and controlled protocol

`v8_1_kernel_choice` retains the frozen weighted RKHS-L2 objective, OVO,
preprocessing, balanced three-fold inner selection, C grid and RBF bandwidth
grid. It appends four C configurations for
`K_Q(x,z)=(1+x'z)^2 / mean_train[(1+||x_i||²)^2]`.
Normalization and transforms are refitted on each inner training fold only.
All 32 RBF configurations precede quadratic in exact selection ties.
Quadratic uses native polynomial SVC at C divided by the normalizer; synthetic
tests verify equivalence to the normalized precomputed kernel. Disabling the
alternative dispatches to frozen V8-C; V7 uses its frozen evaluator directly.

The four saved rows are V7, V8-C, quadratic-only diagnostic and kernel-choice.
The diagnostic is not a backup promotion candidate. Banks and selected outer
predictions are shared, while standalone runtime charges both candidate banks.
There are 108 inner multiclass fits versus C's 96, with variable binary solver
cost; a small fit-count increase does not guarantee small runtime overhead.

Development uses matched stratified 75/25 splits, seeds 11000--11007 at gate8;
inner shuffle seed is 20000+outer_seed. Smoke uses 11000 and is reused, not
refitted. All parameters are chosen from outer-training inner CV only.
No outer-test metric chooses a kernel, C, threshold or dataset-specific rule.
Gate8 targets Mammographic error reduction >=0.5 pp with at least five strict
wins, plus explicit Blood/Heart/regression/runtime guards. All conditions
must pass. See the immutable `V8_1_EXPERIMENT_PLAN.md` for exact thresholds.

## Execution and evidence

The first app `ap-mxp5vYcZ5Pu79W5SyBdCCP` failed wrapper import before any
scientific fit. It was stopped. The one allowed infrastructure fix moved a
local-only provenance import into the local entrypoint; no model code changed.
Successful smoke app `ap-Uy2y7kGAnuMHYKR5wBvumZ` produced 20 records and passed
independent metric/hash/fold/solver/voting replay. Gate8 app
`ap-YyFhAPUf39GsX763SUIUPO` reused the five smoke checkpoints. Once all eight
Mammographic splits failed both primary criteria, the remaining slow Blood
seed 11003 was canceled to avoid spending on a decision it could not reverse.
All 39 completed checkpoints (156 rows) were mirrored, with Blood 7/8 and
every other dataset 8/8. This is an intentionally pruned stage, NOT a complete
40-job run. `analysis/pruning_manifest.json` records missing work and hashes;
no remote complete `manifest.json` or successful stage status was fabricated.
Completed-job analysis passed 13,716 inner weighted binary solver diagnostics,
metric/prediction/split hashes and independent OVO replay. No convergence bug
was found in completed fits. The canceled fit has no final diagnostics and
must not be counted as successfully converged or assigned zero runtime.

Evidence directories:

```text
results/v8_1/analysis/
results/v8_1/smoke/v8-1-smoke-20260915/
results/v8_1/gate8/v8-1-gate8-20260915/
```

Each completed stage preserves config, source hashes/snapshots, manifest,
split/prediction hashes, inner predictions/losses, fold transforms, normalizers,
solver diagnostics, outer margins/votes/predictions, class metrics, confusion
matrices and timing. Analysis independently replays votes and metrics; paired
tables compare both V7 and V8-C, including 95% t-CIs, Wilcoxon, standardized
effects and resampled-variance sensitivity. Repeated holdouts are correlated
and are not independent-population replications; exploratory p-values are
not universal-superiority evidence. Parkinson retains inherited row-level
splits and their subject-leakage limitation.

Published paper values, executable MATLAB Legacy, historical V7/V8 results,
and the new matched development results are separate evidence categories.
In particular, the paper's 2.87% Iris result is not reproduced by this study.
No V8.1 Iris or Breast Diagnostic evaluation was authorized: their gate24
extension was never reached. Confirmation seeds 12000--12095 remain untouched.

## Matched development results and frozen gate decision

Percentages below; deltas are percentage points relative to V8-C, runtime is
the ratio of mean standalone times. Blood* uses only seven completed matched
seeds (all except 11003), a runtime-censored subset: descriptive, not a valid
eight-seed qualification result. Other rows contain all eight planned seeds.

| Dataset | V7 error | V8-C error | Candidate error | Delta error | C BA | Candidate BA | Delta F1 | Key recall change vs C | Runtime/C | Error W/T/L | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| Blood* | 22.4599 | 21.0848 | 21.0848 | 0 | 67.0573 | 67.0573 | 0 | Minority unchanged, 44.27% | 24.761x | 0/7/0 | Incomplete; severe overhead |
| Mammographic | 16.8870 | 16.5865 | 16.2861 | -0.3005 | 83.3407 | 83.6848 | +0.3229 | Class 1 +1.8564; class 0 -1.1682 | 1.667x | 4/3/1 | Primary and runtime fail |
| Heart | 43.5000 | 44.1667 | 44.1667 | 0 | 31.7312 | 31.7312 | 0 | All classes unchanged | 1.120x | 0/8/0 | No improvement |
| Parkinson | 6.1224 | 6.6327 | 6.6327 | 0 | 92.7928 | 92.7928 | 0 | Both classes unchanged | 1.120x | 0/8/0 | Point guards pass |
| Wine | 1.9444 | 1.9444 | 1.9444 | 0 | 98.0324 | 98.0324 | 0 | All classes unchanged | 1.122x | 0/8/0 | Point guards pass |

Mammographic: candidate 271/1664 errors versus C 276/1664, only five saved
errors. Required reduction >=0.5 pp would need at least nine fewer errors;
required strict wins >=5/8, observed 4/8. Its runtime ratio also exceeds 1.5.
These complete primary results alone reject promotion, independent of missing
Blood data. The partial-data gate JSON's Blood `true` checks are descriptive
subset calculations, not passed eight-seed claims; `complete_expected_jobs`
is false. Its `next_stage` string names the protocol successor, NOT permission:
`passes=false` prevents authorization. No criteria were relaxed or retuned.

| Mammographic endpoint vs C | Mean delta (pp) | Paired 95% t-CI | Wilcoxon p, raw / rounded | Paired standardized effect |
|---|---:|---|---|---:|
| Error | -0.3005 | [-1.4360, +0.8351] | .5625 / .6875 | -.2212 |
| Balanced accuracy | +0.3441 | [-0.7740, +1.4622] | .4375 / .4375 | .2573 |
| Macro F1 | +0.3229 | [-0.8059, +1.4517] | .4375 / .4375 | .2391 |
| Class-1 recall | +1.8564 | [+0.3601, +3.3528] | .0625 / .0625 | 1.0372 |

The raw/rounded Wilcoxon difference reflects machine-precision rank ties;
both saved methods are reported, neither supports error superiority. The
resampled-variance sensitivity error CI is [-2.4775,+1.8765] pp and class-1
recall CI [-1.0122,+4.7250] pp. Do not overinterpret the ordinary recall t-CI
with eight correlated holdouts or replace the predefined primary error target.
For the other four datasets candidate predictions equal C on every completed
split: accuracy/BA/F1/recall deltas and empirical CIs are zero, Wilcoxon p=1,
and standardized effects are undefined (zero paired variance). This is sample
identity, not proof of out-of-sample equivalence of the expanded selector.

Versus matched V7, error deltas/95% CIs in pp and W/T/L are: Blood* -1.3751
[-4.8118,+2.0616], 3/1/3; Mammographic -0.6010 [-1.9891,+0.7872], 5/1/2;
Heart +0.6667 [-3.2404,+4.5738], 3/1/4; Parkinson +0.5102
[-0.6962,+1.7166], 0/7/1; Wine 0 [-0.9930,+0.9930], 1/6/1.
Full paired error/BA/F1/runtime and per-class recall/precision/F1 statistics
against BOTH references are in `analysis/paired_metrics.csv` and
`analysis/paired_class_metrics.csv`; pooled confusion counts are also saved.

## Ablation, class behavior and computational failure

Inner CV selects quadratic on 6/8 Mammographic splits and nowhere else among
completed jobs. Its selected quadratic C is 1 on five of those splits and
100 on one. The two remaining Mammographic splits select frozen RBF. Complete
selected alpha/C rules and all fold losses are retained in checkpoints and
`analysis/kernel_evidence.csv`; no retrospective parameter replacement occurs.
The smaller quadratic search often gives a plausible boundary but does not
consistently correct the RBF errors: error changes per Mammographic seed are
0,-1,-4,0,+5,-1,-4,0. One adverse split offsets five recovered errors elsewhere.
Across all eight splits the selector changes 35 predictions, correcting 20 C
errors but introducing 15. Quadratic-only changes 42, correcting 23 and
introducing 19. The saved `prediction_disagreements.csv` makes this explicit;
neither geometry consistently separates only the RBF mistakes.

Quadratic-only diagnostic mean errors: Blood* 21.0848%, Mammographic 16.3462%,
Heart 45.5000%, Parkinson 13.5204%, Wine 2.2222%. On Mammographic it has
272/1664 errors, just one more than the selector. On Parkinson it is much worse
than RBF. The diagnostic was never independently promotable; these results
do not authorize switching to quadratic-only, a dataset rule or a hybrid.
The causal claim is bounded: this normalization/C grid/selection protocol
does not repair the proposed geometry weakness enough. It does not rule out
all conceivable polynomial kernels or establish an irreducible population gap.

Blood* minority recall is 22.94% V7 versus 44.27% C/candidate, majority recall
94.77% versus 89.84%. BA gain over V7 is +8.1993 pp on this incomplete subset,
but the candidate adds no gain to C. Heart recalls for classes 1--5 are
V7 [93.13,15.18,12.50,20.83,0]% and C/candidate
[85.94,24.11,16.67,27.78,4.17]%. The candidate retains both C's minority gain
and majority sacrifice; it does not improve that tradeoff or the rarest class.
All these values use fresh development only, not historical confirmation.

Mean standalone seconds C -> candidate: Blood* 1.491 ->36.908;
Mammographic 23.074 ->38.453; Heart 1.739 ->1.948;
Parkinson .443 ->.496; Wine .689 ->.773. Completed quadratic inner fits sum
1.198 billion iterations on seven Blood splits (single maximum 135.46 million),
and 481.09 million on Mammographic (maximum 120.88 million). Completed statuses
are zero. This supports genuine solver-work overhead rather than only Python
orchestration; it does not identify a precise conditioning cause or prove a
bug. No solver/tolerance change was introduced to rescue the gate. Blood's
very slow missing split may make complete runtime worse, but its value is
unknown and no extrapolated ratio is asserted.

The successful smoke took 28.65 s stage wall time. Completed unique jobs total
805.27 worker-seconds, of which 756.14 were new completed gate8 work and 49.12
were reused smoke work. Do not add the reused work twice. Canceled in-flight
Blood computation, image setup and idle allocation are not included, so these
are NOT total billed credits or vCPU seconds. Each app used 4 CPU/4 GiB,
at most four workers, no GPU. All three apps are stopped with zero tasks.
Cancellation emitted a joblib cleanup warning about missing `pgrep`/`psutil`;
Modal subsequently confirmed the container stopped. No environment workaround
or further training was needed. Frozen baselines and candidate code unchanged.

## Reproducibility, preservation and next step

Analysis-only replay (does not launch training):

```powershell
python v8_1_provenance.py
python manage_v8_1.py analyze smoke/v8-1-smoke-20260915
python manage_v8_1.py analyze gate8/v8-1-gate8-20260915
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_v8_1_kernel_choice.py tests/test_v8_1_evidence.py tests/test_v8_class_sensitive.py tests/test_v8_confirmation_analysis.py tests/test_v7_cross_dataset.py
python freeze_v8_1_decision.py
```

If local copies are absent, `manage_v8_1.py sync` mirrors the complete smoke;
`sync-pruned` mirrors gate8's partial files from `ddr-mksvm-v8-1-results`.
Both refuse incompatible overwrites. Historical execution commands, NOT a
request to rerun, were `modal run --detach modal_v8_1.py --stage smoke
--run-name smoke/v8-1-smoke-20260915` and the same wrapper with `--stage gate8
--run-name gate8/v8-1-gate8-20260915 --reuse-root smoke/v8-1-smoke-20260915`.
The gate8 interruption is deliberate and must not be resumed to chase success.

The negative evidence freeze is `results/v8_1/analysis/negative_decision_freeze.json`:
hashes of candidate code, design/plan/report/log, analysis code/tests, configs,
raw results and all derived evidence. It is NOT `candidate_freeze.json` and
does not authorize confirmation. Seed registry 11000--11007 is consumed for
development; 11008--11023 and confirmation 12000--12095 were not evaluated.
The latter remains untouched, not a performance claim. Baseline integrity
still covers 760 original files, and focused deterministic regression tests
include exact reduction, kernel equivalence, leakage and saved-vote replay.
Final focused suite: 32 passed, one unrelated pytz deprecation warning;
robust tests were excluded. No changes were made to the frozen design files
or experimental model/runner code after successful smoke.

**No deterministic candidate improves enough; stop architecture search.**
Retain V7 as the accuracy/runtime reference and V8-C as the qualified
class-sensitive tradeoff. Recommend consolidating this negative geometry
result and, in a separately authorized task, investigating exact runtime
engineering with strict prediction/selection parity—not another architecture
grid. Robust work remains entirely blocked until authoritative MATLAB robust
source and a new explicit task are provided; no robust design or execution
was advanced here.

## Novelty and interpretation limits

1. The mechanism is standard: weighted SVM, PSD polynomial/RBF kernels,
   trace normalization and finite training-only kernel selection.
2. The defensible contribution here is a controlled causal comparison that
   separates geometry from the confounded Legacy objective/threshold change,
   with immutable baselines, exact reduction and prospective pruning.
3. Whether it resolves a recurring weakness must follow the gates, not the
   mere existence of a selected quadratic kernel or one favorable split.
4. RBF versus quadratic-only versus selector isolates geometry and selection;
   outer-performance oracle switching is neither evaluated nor deployable.
5. A dataset-specific effect with guard failures does not justify a general
   paper method. The fixed local datasets cannot establish population-wide
   generalization even with many repeated seeds.
6. A skeptical reviewer can verify the algebraic reparameterization, honest
   provenance, controlled ablation and explicit negative decision. These do
   not establish a new optimization theorem or a new cost-sensitive SVM.
7. Do not claim universal superiority, repaired Heart geometry, unavoidable
   Bayes error, robust parity, paper-result reproduction, or guaranteed Q1
   acceptance. Lower error alone would not establish mathematical novelty.
