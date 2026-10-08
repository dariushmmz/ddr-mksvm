# Architecture history and decisions

## Legacy

- **Change:** fixed dataset-configured kernel with the original deterministic
  q=1 LP; no DNN, MKL, or DRO.
- **Rationale:** faithful reference architecture and reduction target.
- **Experiment:** matched four-seed Modal pilot on Parkinson, Blood
  Transfusion, Mammographic Mass, and Iris.
- **Result:** mean test error `0.1344`, aggregate model runtime `6,182.9 s`;
  best mean error and runtime of the three architectures. It beat Residual V3
  on 10/16 pairs, tied 3, and lost 3.
- **Decision:** retain as the default architecture for the evaluated datasets.

## Current V2

- **Change:** spectral-normalized deep representation, learned two-kernel MKL,
  and Wasserstein-DRO q=2 SOCP.
- **Rationale:** jointly learn representation and kernel mixture while adding a
  robustness penalty.
- **Experiment:** same matched pilot, `epsilon=0.001`, one outer step, two inner
  steps, and final convex re-solve.
- **Result:** mean test error `0.1646`, aggregate runtime `8,463.4 s`; worse than
  Legacy on 10/16 pairs with 4 ties. It was also directionally less accurate
  than Residual V3.
- **Decision:** do not retain as the recommended model under the evaluated
  configuration.

## Residual V3.0

- **Change:** replace V2 MKL with a raw configured-kernel anchor plus a gated
  deep RBF correction. Gate initialization is `0.10`; penalty is `0.05 g^2`.
- **Rationale:** preserve a useful raw-kernel path while allowing a conservative
  learned correction.
- **Experiment:** smoke test followed by the same matched four-seed pilot.
- **Result:** smoke error `0.2245` in `12.8 s` versus V2 `0.1837` in `19.0 s`.
  In the controlled pilot V3 mean error was `0.1461` versus V2 `0.1646`, a
  paired change of `-0.0184` (11.2% relative); it won/tied/lost 9/3/4 matched
  runs. The paired 95% t interval `[-0.0400, 0.0032]` includes zero. Aggregate
  runtime was `15,451.0 s`, 1.83 times V2 and 2.50 times Legacy.
- **Decision:** the residual anchor is a useful research result and is preferred
  to V2 only when a learned/DRO architecture is mandatory and accuracy matters
  more than compute. It is not the default because Legacy dominates overall.

## Corrective V3 variant

No corrective variant was implemented. V3's directional accuracy improved
over V2 on three dataset means and tied one, so there was no clear accuracy
failure calling for a gate or normalization correction. Its confirmed weakness
was compute/conditioning on the larger datasets; a speculative parameter tweak
would not justify another pilot while Legacy already dominated both accuracy
and cost.

## 2026-09-08 — Iris q1 audit and exact-threshold candidate

- **Change:** added an isolated Iris-only paper-q1/HiGHS framework with exact
  breakpoint thresholding, nested bandwidth/nu selection, OVO, the paper
  bounded-input robust LP, lightweight raw multi-scale RBF, and complete
  checkpoint/diagnostic capture. Legacy/V2/V3 code paths were not changed.
- **Rationale:** test mathematically simple sources of Iris error before any
  new deep architecture and enforce matched, leakage-free staged gates.
- **Mathematics:** direct paper `nu` obeys
  `lambda_normalized=1/(nu*m)`; exact `b` enumerates all loss discontinuities;
  multi-scale weights are a PSD simplex fitted by centered-target NNLS; robust
  radii use the paper's RBF `p=infinity` formula. A late source audit corrected
  Legacy to retain the authors' descending `linspace` order and first-minimum
  tie rule.
- **Experiment:** CPU-only Modal 1/8/24/96 gates across three prospectively
  locked split registries. The pre-correction nu-CV candidate reached 96 but
  missed target at 3.1524% and was later superseded as a literal-paper
  comparison. Corrected A--D used seeds 2000--2007; exact-b alone advanced to
  seeds 2000--2023.
- **Result:** corrected exact-b mean error 3.8377% versus corrected Legacy
  3.9474% at 24 seeds; paired delta -0.1096 pp, 95% CI
  [-0.8032,+0.5839] pp, 3/18/3 wins/ties/losses, p=1.0. Exact-b was faster
  (6.06 versus 10.03 aggregate model-seconds). All 30,897 recorded LP solves
  succeeded across the program, with no selected trivial fits after the
  diagnostic fix.
- **Decision:** no Iris candidate is promoted and no new default is declared.
  The 2.87% paper target was not reproducibly beaten. Exact-b is retained as a
  mathematically correct, efficient research option, not an accuracy winner.
  Further work requires exact MATLAB/CVX split-level parity before another
  architecture or 96-seed spend.

## 2026-09-08 — authoritative MATLAB parity and RKHS-l2 candidate

- **Change:** elevated the supplied MATLAB directory to executable authority;
  corrected split row order and raw-xi threshold construction; added a
  split-level MATLAB/Python trace harness. After exhausting source-local q1
  variants, added one conventional RKHS-l2 RBF OVO candidate with training-only
  C/bandwidth selection.
- **Rationale:** distinguish reproduction bugs and numerical solver effects
  from intentional methodology, then test the smallest broader regularization
  change capable of addressing q1 coefficient geometry.
- **Mathematics:** executable Legacy is homogeneous quadratic q1 OVA. The new
  candidate minimizes `0.5||w||_H^2+C sum(xi)` in three RBF pairwise problems;
  all kernel/C choices are frozen by inner three-fold CV.
- **Experiment:** prospective Modal registries 3000, 4000, and 5000 passed
  1/8/24 gates. Only raw RKHS-l2 qualified for 96 seeds. Final app
  `ap-qeOYZYpnpUre4bNBeaMMwp` used 4 CPU/8 GiB/no GPU.
- **Result:** RKHS-l2 3.1250% versus executable polynomial 4.3311% over 96
  matched splits. Paired delta -1.2061 pp, CI [-1.8000,-0.6123], p=0.000137.
  Manual q1 RBF was 3.3717%. RKHS-l2 was fastest (18.42 aggregate seconds).
- **Decision:** retain RKHS-l2 as the simplest validated improvement over the
  executable baseline, but do not claim the paper target: 3.1250% is above
  2.87%. No robust or standardized variant advances. Exact MOSEK parity remains
  dependent on a licensed MATLAB/CVX/MOSEK run of the supplied harness.

## 2026-09-09 — frozen V7 cross-dataset validation

- **Change:** no architecture change. Generalized the frozen RKHS-l2 RBF model
  to binary (one classifier) and multiclass (unchanged libsvm OVO) datasets,
  with matched corrected q1 Legacy evaluation and complete diagnostics.
- **Rationale:** determine whether the Iris gain generalizes before starting
  V8.
- **Protocol:** train-only dataset-fixed preprocessing; nested three-fold V7
  alpha/C selection; matched sorted 75/25 splits; Modal 1/8/24 gates; untouched
  seeds 8000--8095 for five qualified confirmations.
- **Datasets:** seven non-Iris datasets screened. Breast Cancer was excluded as
  a materially different categorical file; three paper datasets were absent.
- **Results:** final V7 error versus Legacy was 7.0791% vs 14.2432% Parkinson,
  2.6005% vs 2.8555% Breast Diagnostic, 2.0370% vs 2.7546% Wine, 42.6389% vs
  43.7917% Heart, and 3.9815% vs 4.2593% Dermatology. Blood and Mammographic
  regressed at eight seeds and were pruned.
- **Statistical evidence:** paired 95% intervals exclude zero for Parkinson,
  Breast Diagnostic, Wine, and Heart. Dermatology is indistinguishable.
- **Runtime:** V7 was faster on six of seven screens; final ratios range from
  0.110 Dermatology to 0.622 Parkinson. Mammographic was 1.61 times slower.
- **Limitations:** binary results are mixed; Heart/Dermatology paper protocol
  comparability is unresolved; the bundled changes do not isolate causality.
- **Decision:** classify V7 as mainly a multiclass improvement with real but
  dataset-dependent binary gains. Preserve V7; future V8 should first test a
  minimal class-sensitive objective, not a deeper representation.

## 2026-09-09 — V8 class-sensitive controlled development

The new class-sensitive phase is separate from the rejected Iris-only targeted
phase. V7 implementation and result hashes remain unchanged. Mathematical
design preceded code: weighted RKHS-l2 pair SVM, normalized inverse/sqrt
weights, fixed RBF/C grids and OVO voting, with balanced-selection ablations.
Eight-seed ablations showed weighting/selection interaction and a large
accuracy-recall tradeoff. Learned asymmetric ratios failed the Parkinson
guard and were costly; inverse weighting later failed the 24-seed Parkinson
guard. Square-root weighting plus balanced selection passed the predeclared
point bounds and was frozen for one untouched 96-seed confirmation. Its Breast
Diagnostic regression is explicitly flagged; qualification for confirmation
does not establish preservation of every V7 gain. See V8_EXPERIMENT_LOG.md and
V8_CROSS_DATASET_REPORT.md for stage results and final disposition.

## 2026-09-09 — V8-C deterministic confirmation and freeze completed

- **Architecture:** frozen weighted RKHS-l2 RBF OVO; pair weights
  `m*n_c^(-1/2)/(sqrt(n_a)+sqrt(n_b))`, balanced-error inner CV, unchanged
  grids/preprocessing/solver/votes. No modification to V7 or its artifacts.
- **Confirmation:** seven datasets, 96 matched seeds 10000--10095, V7/C only,
  `final96/v8-final96-20260909`, completed app `ap-j0ihyhyO3oUWKLgFWuP54V`.
  Existing 1,344 synced records analyzed without rerunning training.
- **Evidence:** Blood BA +5.8205 pp, CI [+4.9587,+6.6823], minority recall
  28.73% →45.98%; Heart macro F1 +2.8788 pp, CI [+1.8911,+3.8666].
  Heart error worsens +1.8750 pp and rarest recall remains only 6.25%.
  Mammographic error -0.0451 pp: its weakness is not repaired.
- **Guards:** error deltas Parkinson +.1701, Wine +.2315, Iris 0, Breast
  +.0291 pp all satisfy predeclared mean bounds. Wine's CI upper +.5271 pp
  does not establish noninferiority at .5 pp. Runtime ratios .616--4.986x.
- **Audit:** all 295 saved C vote ties are expected cycles (294 Heart, 1 Wine),
  every prediction matches frozen voting; no zero-margin issue or rule change.
  All saved solver statuses pass; 100 regression tests pass.
- **Verdict:** **V8-C qualifies as the new deterministic architecture** for
  the stated class-sensitive objective, not as a universal error/runtime win.
  This is standard weighted SVM plus controlled selection, not established
  methodological novelty. Frozen V7 remains the distinct accuracy/runtime
  reference. Iris historical 3.1250% and published 2.87% remain distinct;
  the new confirmation error is 3.2895% for both V7 and C.
- **Freeze:** `results/v8/analysis/deterministic_v8_c_final_freeze.json`
  captures configuration/seeds/code and 690 result-file hashes. Model SHA-256
  `fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448`.
  Preconfirmation and all V7 artifacts remain unchanged.
- **Next:** deterministic phase complete. Robust phase explicitly blocked
  until the user supplies authoritative MATLAB robust source for mathematical
  and executable audit. Preliminary design is provisional only; no robust
  production code, experiments, Modal spending or kernel/voting search.

## 2026-09-15 — Astra V8.1 deterministic investigation rejected at gate8

Independent audit completed before code; five directions ranked, one chosen:
V8-C plus training-only RBF-or-trace-normalized-quadratic kernel selection.
No new weights, margin targets, voting rule, preprocessing or solver tuning.
Simple asymmetric class margins were shown to reparameterize C/threshold;
Heart cycles explain few new majority losses, not an implementation bug.

Prospectively frozen smoke/gate8 used fresh 11000--11007. Mammographic primary
completed eight seeds: 271/1664 candidate errors versus 276/1664 C, -.3005 pp,
4/8 wins; both required -.5 pp and 5/8 wins fail. Runtime 1.667x exceeds 1.5x.
One remaining slow Blood fit was canceled after this decisive failure; preserve
39/40 checkpoints and explicitly label Blood 7/8, not a completed gate.
On those seven splits selector predictions equal C with 24.761x runtime.
Heart/Parkinson/Wine also retain C predictions, with about 12% overhead.

Verdict: **no deterministic candidate improves enough; stop architecture search**.
No gate24/96, hybrid/second candidate, or robust work. Confirmation 12000--12095
untouched. All 760 frozen/inventoried baseline files remain unchanged. Negative
evidence is frozen separately under `results/v8_1/analysis/`; full derivation,
ablations, paired statistics, caveats and commands: V8_1_CROSS_DATASET_REPORT.md.
Retain V7 accuracy/runtime and V8-C class-sensitive baselines; any future
exact-parity engineering work is separate from methodological novelty.

## 2026-09-16 — robust source audit, parity baseline, and Robust V8-C smoke

- Audited every supplied robust MATLAB binary/multiclass file and separated
  executable behavior from the paper. Active code is q=1, p=infinity,
  polynomial (binary homogeneous quadratic; multiclass homogeneous linear),
  with OVA multiclass and no executable rho selector. Material threshold,
  aggregation, preprocessing/kernel and inhomogeneous-radius discrepancies are
  preserved in the audit.
- Added an isolated source-parity LP pipeline exposing deterministic and all
  three p cases, immutable Modal checkpoints, provenance and analysis. One-seed
  Parkinson/Iris smoke completed all 80 LPs optimally; rho `1e-4` changed no
  prediction. An exact-active Mammographic smoke added 20 optimal LPs and
  regressed from 15.38% to 22.12% with large raw-polynomial radii. These are
  one-seed correctness/warning signals only.
- Derived Robust V8-C with the RKHS support term `delta_i||w||_H`, weighted
  slacks and OVO conic subproblems. rho=0 dispatches frozen V8-C exactly.
  One-seed p=2 smoke matched V8-C predictions; 321/388 conic solves were
  optimal and 67 optimal_inaccurate with maximum violation `5.20e-9`.
- Decision: mathematics and infrastructure are viable, but no accuracy claim
  or full campaign is justified from one seed. Stop after smoke. V7 and V8-C
  remain frozen; no deterministic architecture was modified or searched.
- The 2026-09-17 handoff hardened long runs into four bounded, self-continuing
  Modal lane chains. A two-task infrastructure-only app verified remote
  successor spawning and finalization; it did not reopen the scientific gate.

## 2026-09-17 — Robust V8-C p=2/rho=.01 gate not qualified

The requested 24-seed Parkinson/Blood/Mammographic/Iris gate stopped after
repeatable CLARABEL failure on the first two Blood robust seeds. Parkinson's
24 matched splits showed only uncertain predictive gains, 21.14x runtime and
18/24 `optimal_inaccurate` selected outer fits. No solver rescue or 96-seed
confirmation was attempted. Verdict: do not freeze p=2/rho=.01; qualify a
solver policy on Blood under a new development run first.

## 2026-09-17 — Robust V8-C solver qualification remains blocked

The Blood failure was reproduced on exposed development seeds 15000 and
15001. The unchanged inner representation failed 120/192 CLARABEL calls in a
candidate-specific, fold-independent pattern. The evidence narrows the cause
to numerical breakdown in large, duplicate-induced near-singular RBF cone
systems; non-finite inputs and a DCP/formulation error were not found.

Exact aggregation of duplicate `(x, y)` margin groups is mathematically
equivalent and solved all 192 inner calls, with identical predictions in all
72 calls having an original solution. It did not solve either selected outer
fit. Relaxed CLARABEL and QDLDL retained the failure pattern; SCS returned all
representative calls but only 5/16 met the prospectively fixed feasibility
threshold. Exact objective scaling also failed the first selected outer fit.

Decision: **NOT YET QUALIFIED**. No fallback solver is approved, no Blood
robust seed completed, and R4 through R7 remain blocked. This is numerical
qualification only and does not revise the rejected Gate24 freeze decision or
the frozen deterministic architectures. Full evidence and policy are indexed
by `docs/robust/README.md`.

### Independent outer solver probe

CVXOPT 1.3.2 was prospectively tested on only the two exposed, selected Blood
outer problems using the exact duplicate-aggregated formulation and strict
high-accuracy settings. It returned no solution for either problem: seed
15000 reached 500 iterations and seed 15001 lost primal feasibility after a
near-converged iterate. CVXOPT is not qualified; the model, policy, and blocked
R4–R7 decision remain unchanged.

### 2026-09-18 — R3.2 exact-coordinate reformulation not instantiated

The proposed exact RKHS coordinate system was derived but failed its mandatory
factor-existence gate. Neither selected Blood outer Gram matrix admits an
unmodified full-rank Cholesky factor at float64 working precision. Constructing
one would require jitter, spectral clamping/truncation, or a pseudoinverse,
which were explicitly prohibited. No optimization or predictive experiment
was run. Decision: **R3.2 NOT QUALIFIED**; architecture and frozen baselines
remain unchanged.

### 2026-09-18 — R3.3 closes numerical rescue of Robust V8-C SOCP

Arbitrary-precision symmetric reconstruction establishes that both exposed
Blood Gaussian kernels are mathematically SPD at 2048 bits. Their estimated
conditions, `7.11e23` and `5.01e24`, explain why double precision loses rank
and PSD. Neither passes the prospectively frozen two-level stability gate, so
no solver-compatible conversion or outer fit was attempted. This is not an
architecture result and changes no frozen baseline. Further work must redesign
the robust formulation from the authoritative q=1 LP rather than continue
numerical rescue of the RKHS-L2 SOCP.

## 2026-09-19 — robust architecture investigation closed

The robust investigation ended after a controlled sequence of formulation,
numerical, and predictive gates. The stages must not be conflated:

| Architecture/stage | Mathematical change | Numerical result | Scientific decision |
|---|---|---|---|
| Original MATLAB/source robust | q1 coefficient-L1 robust LP; active source kernels; OVA | source audited and structurally reproduced | executable reference only |
| Python source parity | same source semantics with HiGHS | one-seed invariants/smoke passed | no performance promotion |
| Robust V8-C | V8-C RKHS-L2/OVO plus robust norm tightening | Blood SOCP outer fits failed; high-precision kernels SPD but condition above `1e23` | rejected; numerical rescue closed |
| R8 | source q1 LP plus mean-one sqrt inverse-frequency slack weights | Blood source-profile range up to `6.03e22`; 0/8 models | rejected |
| R8.1 | exact row/column LP coordinates only | Parkinson equivalent; Blood 5/8 | rejected; scaling rescue closed |
| R9 | bounded-RBF q1 LP with the R8 weights | exposed qualification passed; fresh Gate8 126/128 tasks | rejected at prospective Gate8 |
| R10 | checkpoint-only diagnosis | no training; mechanism reconstructed on 31 pairs | current robust track closed |

R9's primary Gate8 comparison was weighted robust versus weighted
deterministic RBF q1. Mean balanced-accuracy changes were negative on Blood,
Parkinson, Mammographic, and Iris; Parkinson and Mammographic also crossed the
prospective safety-recall loss guard. Mechanistic evidence attributes the
binary failures to global `delta_i||u||_1` shrinkage/slack substitution and,
for Parkinson/Mammographic, misalignment between class-sensitive training and
the source unweighted threshold rule.

Verdict: **no tested robust architecture is supported for promotion**. Frozen
V7 remains the accuracy/runtime reference and frozen V8-C remains the
class-sensitive deterministic reference. Robust results are publication-grade
negative evidence, not a successful performance claim. Seeds 15400–15423 and
14000–14095 remain untouched. Full synthesis:
`docs/robust/ROBUST_RESEARCH_FINAL_REPORT.md`.

## 2026-09-27 — frozen-model dataset expansion

No architecture was created or changed. The supported deterministic models
were evaluated on two missing local-data comparisons under a prospectively
frozen staged protocol. V7-versus-Legacy passed Gate24 and completed 96-seed
confirmation on Breast Cancer recurrence and Dermatology. V8-C failed its
predeclared promotion rule on both and remained a fixed Gate24 control.

The added evidence supports V7's error/runtime profile but also bounds its
claim: confirmed rare/minority recall declined on both datasets. All nine
usable local classifier datasets now have an explicit paper role, including
negative and tradeoff results. Frozen V7 and V8-C hashes are unchanged.
