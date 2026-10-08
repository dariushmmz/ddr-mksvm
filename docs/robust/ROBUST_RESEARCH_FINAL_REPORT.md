# Final report: robust SVM research track

Date: 2026-09-19  
Status: **scientifically closed**  
Final decision: **no tested robust candidate is promoted**

This report is the authoritative synthesis of the completed robust investigation. It does not replace the stage reports or their machine-readable evidence. It records the final scientific interpretation after the prospective R9 Gate8 failure and the checkpoint-only R10 diagnosis. No training, solver call, hyperparameter search, or new seed was used to produce this report.

## 1. Research objective

The robust phase asked whether uncertainty-aware training could improve reliability and class-sensitive performance beyond the supported deterministic models without sacrificing numerical validity or reproducibility. Frozen V7 remained the accuracy/runtime reference and frozen V8-C remained the class-sensitive deterministic reference. Neither was modified.

The work deliberately separated four questions:

1. What does the original MATLAB robust implementation actually solve?
2. Can that executable formulation be reproduced faithfully in Python?
3. Can robustness be incorporated into V8-C or a source-aligned class-sensitive model with defensible numerics?
4. Does a numerically qualified robust model improve fresh predictive results enough to justify promotion?

The answer to the first two questions is documented and reproducible. The tested answers to the third and fourth do not support a robust promotion.

## 2. Mathematical formulations and stage separation

### 2.1 Original MATLAB/source robust formulation

For a binary task, or one source one-versus-all (OVA) task, the executable q=1 model uses kernel coefficients `u`, intercept-like optimization variable `gamma`, slack `xi`, and the L1 epigraph `s`:

```text
minimize        sum_j s_j + nu sum_i xi_i

subject to      y_i[(K D u)_i - gamma] + xi_i
                - delta_i sum_j sqrt(K_jj) s_j >= 1,
                xi_i >= 0,
                -s_j <= u_j <= s_j.
```

Here `D=diag(y)`. Input uncertainty has class radius

```text
eta_c = rho max_r sample_std(X_c[:,r]),
```

and `delta_i` is the source/paper kernel-dependent feature-space bound. The `p` input norm maps to the dual norm in the derivation; executable branches for `p=1`, `p=2`, and `p=inf` are not interchangeable. The robust term tightens the constraints and is not a separate objective addend.

The active MATLAB profiles are not a single universal RBF implementation. The active binary script uses its raw homogeneous polynomial profile, the active multiclass script uses its raw linear profile, and the RBF alternatives are commented in the audited paths. Multiclass is source OVA over sorted classes. `nu` is selected in the original five-value order using first strict improvement in training error, then the source 10,000-point threshold grid and first-minimum behavior determine `b`. The executable details, including source defects and paper discrepancies, are preserved in the [source audit](../ROBUST_MATLAB_SOURCE_AUDIT.md) and [mathematical audit](../ROBUST_MATLAB_MATHEMATICAL_AUDIT.md).

### 2.2 Python source-parity reproduction

The isolated parity implementation preserves the q=1 coefficient-L1 objective, robust constraints, active profiles, OVA construction, `nu` loop, threshold order, and source quirks. It exposes deterministic and robust `p=1,2,inf` variants and substitutes HiGHS for the MATLAB CVX backends. This is structural and mathematical parity, not a claim of bitwise solver parity.

The one-seed smoke established valid construction and execution: 100/100 LPs were optimal and the maximum recorded violation was `5.93e-8`. It did not establish a performance advantage. Published paper values, MATLAB executable behavior, and Python reproduction results remain separate evidence categories.

### 2.3 Robust V8-C: RKHS-L2/SOCP

Frozen deterministic V8-C is Gaussian-RBF, OVO, RKHS-L2, with mean-one square-root inverse-frequency pair slack weighting and balanced inner selection. Its attempted robust counterpart solved

```text
minimize        0.5 t^2 + C sum_i w_i xi_i
subject to      y_i(<w,phi(x_i)> + b) - delta_i t >= 1 - xi_i,
                ||w||_H <= t,
                xi_i >= 0.
```

This is a convex SOCP and reduces to V8-C at `rho=0` through the implementation's exact deterministic dispatch. It is not the source q=1 LP. Its kernel-norm cone exposes Gram geometry that proved numerically unsuitable on Blood Transfusion.

### 2.4 R8: source-aligned class-sensitive q1 LP

R8 returned to the authoritative q=1 LP and changed only the slack price:

```text
minimize        ||u||_1 + nu sum_i w_i xi_i,
```

with positive mean-one square-root inverse-frequency sign-group weights (`tau=0.5`). The original robust constraints, source profiles, p/rho meanings, OVA construction, `nu` order, and threshold procedure remained fixed. Disabling weighting exactly recovers the original robust q1 LP; `rho=0` recovers the weighted deterministic same-profile q1 LP. It remains an LP and has no RKHS-L2 cone or Gram factorization.

### 2.5 R8.1: exact LP equilibration

R8.1 was not a new predictive model. It applied a bijective positive diagonal coordinate transformation to the same canonical LP:

```text
x = D z,
S A D z <= S b,
objective = (D^T c)^T z,
```

using one prospectively frozen eight-pass max-norm rule. All variables, bounds, objectives, and residuals were mapped back to original coordinates. No coefficient was dropped or truncated.

### 2.6 R9: bounded-RBF class-sensitive q1 LP

R9 was a new research candidate, not MATLAB parity and not a numerical rescue of R8. It retained the q=1 LP, `tau=0.5`, `p=2`, `rho=0.01`, source uncertainty interpretation, OVA, `nu` loop, and threshold semantics, but replaced the ill-scaled source-profile kernel with a Gaussian RBF using the prospectively frozen training-only alpha rule. Since

```text
0 < K(x,z) <= 1
```

and `K_jj=1`, the robust margin contribution becomes

```text
delta_i sum_j |u_j| = delta_i ||u||_1.
```

The optimization remains an LP. It requires neither an SOC nor a Gram factorization. Disabling weights recovers the unweighted RBF robust q1 model; setting `rho=0` recovers the corresponding weighted deterministic RBF q1 model.

### 2.7 R10: diagnosis, not another model

R10 performed no fitting. It deterministically reconstructed 31 available R9 Gate8 pairs from stored checkpoints and decomposed objectives, margins, slack, scores, thresholds, flips, and confusion changes. It introduced no candidate and accessed no new seed.

## 3. Numerical investigation

### 3.1 SOCP Gram and cone conditioning

The 24-seed Robust V8-C gate stopped after repeatable CLARABEL failures on Blood seeds 15000 and 15001. The original Blood representation failed 120/192 audited inner calls. Exact duplicate aggregation was mathematically equivalent on all 72 overlap cases, eliminated all inner exceptions, and preserved validation predictions, but both selected outer fits still failed. SCS failed the predefined feasibility policy, and independent high-accuracy CVXOPT returned no solution on either exposed outer problem. Parkinson solved but 18/24 selected outer fits were `optimal_inaccurate`, and robust runtime was about `21.14x` V8-C.

An exact coordinate proposal `K=R^T R`, `z=Rc` stopped before optimization because the stored float64 outer Gram matrices did not admit the required unmodified full-rank Cholesky factors. High-precision reconstruction then resolved the apparent contradiction: both mathematical Gaussian kernels are SPD at 2048-bit arithmetic, but their estimated conditions are `7.11e23` and `5.01e24`. Float64 entrywise errors around `1e-14` overwhelm genuine eigen-directions around `1e-22` to `1e-23`. The frozen two-consecutive-precision stability gate failed, so no high-precision factor was converted for solver use. This closed numerical rescue of the RKHS-L2 SOCP.

### 3.2 Source-profile polynomial LP conditioning

Removing the SOC did not cure Blood's source-profile arithmetic. In R8, all four model variants failed at the first `nu=.001` on both exposed Blood seeds, including the plain deterministic q1 LP. Canonical coefficient ranges were approximately `9.31e18` to `6.03e22`. Parkinson solved, which localized the failure to the Blood/source-profile representation rather than the class-sensitive extension alone.

R8.1 proved the frozen row/column transformation algebraically and passed all four Parkinson equivalence cases. It reduced Blood coefficient ranges by factors of about `3.77e3` to `1.18e4`, yet the scaled systems still spanned `1.06e15` to `5.13e18`. Only 5/8 mandatory Blood models solved. Returned solutions passed original-coordinate feasibility, but three deterministic models still failed with HiGHS status 4. Successful robust Blood fits selected `nu=.001` and collapsed to majority-only prediction. The source-profile branch was therefore closed.

### 3.3 Bounded-RBF stabilization

R9 bounded every kernel entry by one and reduced the maximum canonical coefficient to `1.0`, compared with R8 maxima of `8.73e5` to `4.55e8` on the matched exposed constructions. All 12 exposed-seed models and all 60 LP calls completed as optimal; Blood's maximum feasibility violation was `6.63e-8`. This was a real numerical improvement and justified one prospective fresh Gate8.

It did not make the research candidate successful. Gate8 produced 126 completed tasks and two explicit Blood failures out of 128 expected tasks. All returned models were finite and feasible, but the missing deterministic controls triggered the frozen completion and solver-success failures. More importantly, the 31 available primary pairs showed adverse predictive deltas on every dataset's mean balanced accuracy.

## 4. Prospective gates and branch decisions

| Stage | Prospective question | Evidence | Outcome | Decision |
|---|---|---|---|---|
| Source audit/parity | Was executable MATLAB behavior identified and structurally reproduced? | complete source audit, invariants, one-seed smoke | passed for structural reproduction | retain as reference, not a promoted predictor |
| Robust V8-C smoke | Does the SOCP implementation execute and reduce correctly? | Parkinson/Iris one-seed smoke | implementation signal only | allowed focused Gate24 |
| Robust V8-C Gate24 | Does `p=2,rho=.01` merit freezing? | Parkinson complete; Blood failed twice; later datasets stopped | failed/stopped | do not freeze; no confirmation |
| R3 solver qualification | Can the same SOCP be made reliable without changing the model? | CLARABEL, SCS, CVXOPT, duplicate aggregation | failed | R4-R7 blocked |
| R3.2 exact RKHS coordinates | Does the stored float64 `K` admit exact full-rank coordinates? | factor audit | failed | no optimization run |
| R3.3 high precision | Is rank loss merely arithmetic and convertible safely? | 2048-bit SPD certificates; frozen stability gate | SPD confirmed, conversion gate failed | SOCP rescue closed |
| R8 design/reductions | Does class-sensitive weighting integrate cleanly into source q1 LP? | mathematical proof and invariants | passed | exposed numerical qualification only |
| R8 exposed qualification | Does the unchanged source-profile LP solve on Blood? | Blood 15000/15001; Parkinson control | failed 0/8 Blood | no fresh R8 seed |
| R8.1 exact equilibration | Does exact coordinate scaling rescue all Blood tasks? | Parkinson equivalence; Blood 15000/15001 | failed 5/8 Blood | scaling rescue closed |
| R9 exposed qualification | Does bounded RBF make the q1 LP numerically usable? | Blood 15000/15001; Parkinson 15000 | passed 12/12 models | authorized one frozen Gate8 |
| R9 fresh Gate8 | Is R9 reliable and predictively useful on 15300-15307? | four datasets, four fixed variants | failed prospectively | Gate24 and confirmation permanently blocked for R9 |
| R10 analysis | Why did the robust term fail? | stored checkpoints only | diagnosis complete | close current robust track |

Stopped branches were not continued after their failure rules. Seeds 15400-15423 and 14000-14095 were not accessed.

## 5. R9 Gate8 predictive result

The primary comparison is frozen weighted robust RBF q1 minus weighted deterministic RBF q1. A positive error delta is worse; positive score/recall deltas are better. W/T/L is oriented in favor of the robust candidate. “Safety” is training-minority recall for binary datasets and minimum class recall for Iris.

| Dataset | Matched n | Delta error | Delta balanced accuracy | Delta macro F1 | Delta safety recall | W/T/L error; BA; F1; safety | Numerical completion | Median runtime ratio |
|---|---:|---:|---:|---:|---:|---|---|---:|
| Blood | 7 | +.0145 | -.0138 | -.0195 | -.0125 | 0/1/6; 1/1/5; 1/1/5; 1/5/1 | 30/32 tasks; primary 7/8 | 1.135 |
| Parkinson | 8 | +.0230 | -.0504 | -.0430 | -.1042 | 1/3/4; 2/2/4; 2/2/4; 2/2/4 | 32/32 | 1.051 |
| Mammographic | 8 | +.0078 | -.0092 | -.0087 | -.0557 | 1/1/6; 1/0/7; 1/0/7; 1/0/7 | 32/32 | .876 |
| Iris | 8 | +.0033 | -.0032 | -.0033 | .0000 | 0/7/1; 0/7/1; 0/7/1; 0/8/0 | 32/32 | 1.037 |

Across 31 matched pairs, the pooled deltas were error `+.0121`, balanced accuracy `-.0193`, macro F1 `-.0186`, and safety recall `-.0441`; balanced-accuracy W/T/L was `4/10/17`. Mammographic safety recall had a paired 95% CI wholly below zero (`-.0557 [-.1061,-.0053]`). Parkinson's mean safety loss was `-.1042`. With only seven or eight pairs, inferential results are screening diagnostics, but the prospective decision did not depend on a post-hoc significance rule.

All 128 tasks were accounted for: 126 completed and two explicit failures, both Blood seed 15305 deterministic controls at the first `nu=.001`. The weighted robust candidate itself returned 32/32 models and never collapsed to majority-only prediction. Nevertheless, the numerical gate failed because required controls were incomplete, and the predictive gate failed because every dataset's mean balanced-accuracy delta was negative, pooled accuracy/F1 were adverse, and the Parkinson and Mammographic safety guards were crossed. Gate24 and confirmation were not launched.

Returned-model compute totaled 2,056.30 model-seconds and 1,931.25 solver-seconds. Median model runtime was 2.56 s, p95 55.31 s, and maximum 83.01 s. Runtime practicality passed; it did not override the numerical and predictive failures.

## 6. R10 mechanistic failure analysis

### 6.1 Global coefficient shrinkage and slack substitution

For RBF, the robust tightening is `delta_i ||u||_1`. It makes a smaller global coefficient norm valuable in every constraint. Relative to weighted deterministic q1, the robust model reduced mean L1 magnitude and support count while increasing slack:

| Dataset | L1 reference -> robust | Supports reference -> robust | Sum slack reference -> robust |
|---|---:|---:|---:|
| Blood | 15.29 -> 6.12 | 18.57 -> 8.71 | 255.23 -> 286.80 |
| Parkinson | 37.79 -> 30.82 | 46.63 -> 30.38 | 11.57 -> 37.18 |
| Mammographic | 33.17 -> 17.25 | 58.50 -> 27.50 | 185.69 -> 256.68 |
| Iris | 7.91 -> 7.69 | 2.92 -> 2.75 | 12.31 -> 13.42 |

This is strong evidence that the frozen robust term behaved like excessive global coefficient regularization on the affected binary datasets. Iris, which had the smallest effective tightening, was nearly inert.

### 6.2 Threshold movement and class-specific harm

The source threshold minimizes ordinary training misclassification. It is not aligned with the class-weighted slack objective or balanced evaluation.

- **Parkinson:** the robust raw score function evaluated at the deterministic threshold preserved mean safety recall, but the robust threshold reduced it. Robustness corrected 5 minority errors and introduced 15, a net loss of 10/96 minority predictions, exactly `-10.42` percentage points of mean safety recall. The threshold moved toward the majority on six of eight seeds.
- **Mammographic:** robustness corrected 9 minority errors and introduced 54, a net loss of 45/808 (`-5.57` percentage points). Threshold-only counterfactual predictions closely reproduced the actual robust result, and the threshold moved toward the majority on six of eight seeds. The raw robust function alone traded increased minority recall for worse class-0 recall; the source threshold overcorrected toward class 0.
- **Blood:** this was not primarily a majority-threshold failure. The robust expansion shrank sharply and the score function worsened for both classes. The threshold was compensatory toward the minority on five of seven pairs. There were 13 corrected errors and 32 newly introduced errors. Blood-specific `nu` shifts contributed, but could not explain the cross-dataset pattern.
- **Iris:** training predictions were identical; seven of eight test prediction vectors were identical. One flip introduced one error. It functions as a near-null multiclass/OVA control.

### 6.3 Supported and unsupported explanations

The evidence strongly supports a two-stage mechanism: global uncertainty-norm coupling drives coefficient/support shrinkage and slack substitution; the unweighted post-fit threshold then converts that redistribution into minority harm on Parkinson and Mammographic. The source-derived radius may be conservative, but stored checkpoints cannot isolate radius conservatism from the structural global-L1 coupling. `nu` selection is not the general cause because Parkinson and Mammographic selected the same `nu=1` in every pair. Threshold movement is important but not universal: it compensates on Blood and is negligible on Iris.

## 7. Scientific conclusion

No tested robust candidate is supported for promotion.

- The source-parity implementation is valuable as an executable reference, not as evidence of superior prediction.
- Robust V8-C's RKHS-L2 SOCP is mathematically coherent but numerically impractical for the exposed Blood problems under the tested exact formulations.
- R8's class-sensitive q1 extension reduced correctly but inherited fatal source-profile coefficient scaling.
- Exact R8.1 equilibration was algebraically valid and helped, but did not meet its 8/8 Blood completion requirement.
- R9 bounded RBF materially improved numerical conditioning and passed exposed qualification, proving that numerical stabilization was possible.
- That stabilization did not yield predictive benefit. The frozen R9 robust term failed a fresh prospective Gate8 and harmed the balance/safety metrics it was intended to protect.

Frozen V7 and V8-C remain the supported deterministic baselines. Robustness is retained as controlled negative evidence and a reproducible limitations result; it is not presented as a successful performance contribution. The current robust research track is closed.

## 8. Future work

Future robust work is justified only as an independently derived mathematical hypothesis, not as another R9 rescue. Before implementation or seed allocation, such a formulation would need to:

- replace or materially alter the global uncertainty-norm coupling that buys robustness through broad coefficient shrinkage;
- protect minority margins explicitly inside the robust risk or constraints rather than only changing slack prices;
- define class-sensitive uncertainty/risk without turning minority observations into disproportionately tightened constraints;
- align threshold/intercept construction with the same class-sensitive objective used in training;
- preserve a stable convex representation and prospectively prove deterministic and disabled-robust reductions.

This report does not propose novelty, choose a new candidate, or authorize an experiment. Untouched ranges 15400-15423 and 14000-14095 remain reserved for a future independently derived method under a new dated protocol.

## Provenance map

- [MATLAB executable-source audit](../ROBUST_MATLAB_SOURCE_AUDIT.md)
- [MATLAB/paper/Python mathematical audit](../ROBUST_MATLAB_MATHEMATICAL_AUDIT.md)
- [Source-parity smoke report](../ROBUST_SMOKE_REPORT.md)
- [Robust V8-C design](../ROBUST_V8_C_MATHEMATICAL_DESIGN.md)
- [Stopped Robust V8-C Gate24](../ROBUST_V8_C_GATE24_REPORT.md)
- [SOCP solver qualification](02_solver_qualification/SOLVER_QUALIFICATION_RESULTS.md)
- [Exact RKHS-coordinate audit](02_solver_qualification/RKHS_COORDINATE_REFORMULATION.md)
- [High-precision kernel qualification](02_solver_qualification/HIGH_PRECISION_KERNEL_QUALIFICATION.md)
- [R8 source-aligned design](06_source_aligned_redesign/ROBUST_CLASS_SENSITIVE_LP_DESIGN.md)
- [R8 exposed qualification](06_source_aligned_redesign/SMOKE_REPORT.md)
- [R8.1 exact equilibration](06_source_aligned_redesign/LP_EQUILIBRATION_QUALIFICATION.md)
- [R9 bounded-RBF design](07_bounded_rbf_redesign/ROBUST_RBF_Q1_DESIGN.md)
- [R9 exposed qualification](07_bounded_rbf_redesign/SMOKE_REPORT.md)
- [R9 prospective Gate8 result](07_bounded_rbf_redesign/GATE8_REPORT.md)
- [R10 mechanistic diagnosis](08_r9_failure_analysis/R9_GATE8_FAILURE_ANALYSIS.md)
- [Robust result index](05_results/README.md)

Final integrity state: the complete project suite last passed 170/170 after R10. Frozen SHA-256 values remain V7 `39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d`, V8-C `fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448`, and R9 `e30a6b8f5402211e57ac251c4a2ec12c44a26d8915b8c31460842bb2729661f9`.
