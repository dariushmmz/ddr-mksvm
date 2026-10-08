# Audited Nonlinear Support Vector Machines for Imbalanced Classification: Cross-Dataset Evaluation and Mechanistic Evidence from Robust Optimization

**Anonymous manuscript draft**  
Author names, affiliations, corresponding-author details, and journal-specific declarations must be inserted before submission.

## Abstract

Nonlinear support vector machines (SVMs) are governed jointly by regularization geometry, kernel scale, multiclass decomposition, class imbalance, threshold construction, and numerical optimization. These dependencies complicate executable reproduction and can cause mathematically plausible robust counterparts to fail either numerically or predictively. We first audited an original coefficient-L1 nonlinear SVM and its robust MATLAB implementation, then prospectively evaluated two frozen deterministic architectures: an RKHS-L2 Gaussian-RBF SVM with one-versus-one prediction (RKHS-RBF-OVO SVM) and a class-sensitive extension with mean-one square-root inverse-frequency slack weights (CS-RKHS-RBF-OVO SVM). Ten local files were audited and nine were usable classification datasets. Across seven untouched 96-seed repeated-holdout confirmations, the RKHS-RBF-OVO SVM reduced mean error relative to the Source-Profile q1 Kernel SVM. The largest reductions occurred on Parkinson, from 14.24% to 7.08% (paired change -7.16 percentage points; descriptive 95% paired split-variation interval -8.16 to -6.17), and Breast Cancer recurrence, from 30.64% to 26.81% (-3.83 points; descriptive interval -4.71 to -2.95), although recurrence minority recall decreased by 8.43 points. The stated 95% coverage is nominal and descriptive: overlapping repeated holdouts are dependent, so these ranges are not formal independent-sample confidence intervals. The class-sensitive model increased Blood Transfusion balanced accuracy by 5.82 points and minority recall by 17.24 points without a confirmed error change. On Heart Disease it increased balanced accuracy and macro-F1 by 2.55 and 2.88 points, respectively, while increasing error by 1.88 points and runtime by a factor of 4.99. A later targeted Gate8 ablation found that square-root weighting with error-based selection retained Blood and Heart balance gains versus the unweighted model while reducing mean error relative to square-root/balanced selection; this retrospective eight-split result is development evidence, not confirmation. None of the robust extensions was promoted. An RKHS-norm robust SOCP was numerically impractical under extreme Gram/conic conditioning; a source-aligned polynomial robust LP remained ill-scaled; and a bounded-RBF robust q1 LP became solvable but failed a fresh four-dataset Gate8. Relative to its weighted deterministic control, bounded-RBF robust fits had lower balanced accuracy on every dataset and lower safety recall by 10.42 points on Parkinson and 5.57 points on Mammographic Mass. Frozen-artifact analysis associated these losses with coefficient shrinkage, support reduction, slack substitution, and threshold-objective mismatch. The evidence supports the RKHS-RBF-OVO SVM as an error-oriented model and its class-sensitive extension when balanced performance is the declared objective under the fixed local benchmark protocol; it does not support the tested robust formulations.

**Keywords:** nonlinear support vector machine; class imbalance; robust optimization; reproducibility; numerical conditioning; paired validation; negative results

## 1. Introduction

Support vector machines remain a useful model class for small and medium tabular datasets because they combine explicit regularization with flexible kernel decision boundaries [1-4]. Their apparent simplicity can nevertheless conceal substantial experimental degrees of freedom. A nonlinear SVM result depends not only on the nominal kernel but also on its parameterization, the norm used to regularize the learned function, the multiclass reduction, class-sensitive penalties, threshold construction, preprocessing, and the data used for model selection. In optimization-based implementations, solver tolerances and numerical conditioning can further determine whether the stated mathematical problem is solved at all.

Three challenges motivated this study. First, executable-source behavior can differ materially from the surrounding mathematical description. A commented kernel block, a changed threshold scan, or one-versus-all (OVA) rather than one-versus-one (OVO) prediction can make a nominal reproduction scientifically incomparable. Second, accuracy alone is inadequate on imbalanced data. A lower overall error may be obtained by sacrificing minority recall, while a class-sensitive objective can improve balanced accuracy yet increase conventional error. Third, robust optimization introduces a distinction that is sometimes overlooked: a formulation can be mathematically defensible, numerically solvable, and still predict worse on genuinely fresh data.

We address these challenges through a staged, auditable evaluation rather than a single retrospective benchmark. The starting point is the nonlinear deterministic and robust framework of Maggioni and Spinelli [5]. We treat its supplied MATLAB programs as an executable specification and the article as a separate mathematical authority. The audit identifies active kernels, norm choices, uncertainty radii, multiclass behavior, threshold selection, solver assumptions, and source-paper discrepancies. We then compare the Source-Profile q1 Kernel SVM with two frozen RKHS architectures. The RKHS-RBF-OVO SVM uses conventional RKHS-L2 regularization, a Gaussian RBF kernel, and OVO multiclass prediction. The CS-RKHS-RBF-OVO SVM preserves that structure while adding normalized square-root inverse-frequency slack weights and balanced inner selection.

The robust investigation is deliberately reported as negative evidence. An RKHS-Norm Robust Class-Sensitive SVM produced a second-order cone program (SOCP) whose Blood Transfusion instances exposed nearly singular Gaussian Gram geometry. Independent conic solvers, exact-coordinate reformulation attempts, and high-precision kernel reconstruction distinguished mathematical positive definiteness from solver-usable conditioning. A subsequent Source-Aligned Class-Sensitive Robust q1 LP avoided the cone but inherited extreme polynomial-kernel coefficient ranges. Exact equilibration did not solve all required models. Finally, a Bounded-RBF Class-Sensitive Robust q1 SVM reduced the dominant positive coefficient scale and completed every weighted robust fit in a fresh Gate8, yet degraded the prespecified balance-sensitive endpoints. Because numerical reliability and predictive utility were evaluated as separate questions, the candidate was rejected rather than rescued through post hoc tuning.

This study makes the following contributions:

1. A reproducible source audit separates the mathematical article, supplied MATLAB behavior, and Python reproduction, including active kernels, multiclass decomposition, uncertainty radii, threshold construction, and solver semantics.
2. Frozen paired experiments quantify the cross-dataset behavior of the RKHS-RBF-OVO SVM and its class-sensitive extension, including confirmed error reductions, minority-recall gains, and adverse class-specific tradeoffs.
3. A prospective smoke/Gate8/Gate24/confirmation protocol with matched splits, dedicated seed registries, immutable checkpoints, and explicit stopping rules preserves the distinction between development and confirmation evidence.
4. Numerical qualification across SOCP and LP robust formulations identifies failure modes involving Gram conditioning, finite-precision rank loss, polynomial coefficient range, and incomplete recovery under exact equilibration.
5. A fresh robust gate and frozen-artifact mechanism analysis show that numerical stabilization alone did not yield held-out predictive benefit under the fixed protocol; the tested penalty was associated with coefficient shrinkage, support reduction, slack substitution, and threshold-sensitive minority degradation.

The contribution is not a claim that RBF kernels, OVO prediction, class weighting, or uncertainty sets are individually novel. It is the controlled architecture comparison, reproducibility evidence, and joint predictive-numerical diagnosis.

## 2. Related Work

The classical maximum-margin classifier and kernel construction are rooted in the work of Vapnik and colleagues [1-3], with RKHS regularization and kernel methods developed systematically by Schölkopf and Smola [4]. Standard soft-margin SVMs control a tradeoff between the RKHS norm and hinge violations. Alternative formulations use coefficient norms or mathematical programming representations; Liu and Potra [6], for example, studied pattern separation through linear and semidefinite programming, while Schölkopf et al. [7] developed related support-vector formulations.

Multiclass SVMs are commonly built by OVO, OVA, or joint multiclass formulations. Weston and Watkins [8] proposed an early joint multiclass approach. In practice, OVO and OVA also differ in their voting, score scaling, and tie behavior, so the decomposition is part of the model rather than an implementation detail. Class imbalance adds a second axis. Cost-sensitive SVMs assign different penalties to observations or classes; such weighting can improve minority recall or balanced accuracy, but its effect depends on model selection and decision thresholds. The CS-RKHS-RBF-OVO SVM evaluated here uses a normalized square-root inverse-frequency weighting rule. We treat this as a controlled combination of established ideas, not as a new primitive.

Recent work from 2022--2026 confirms that imbalanced SVM design remains an active area spanning resampling, algorithm-level weighting, joint hyperparameter selection, and adaptive margin modification. Rezvani et al. [20] reviewed and empirically compared class-imbalanced SVM methods, while Guido et al. [21] studied multi-objective hyperparameter tuning for cost-sensitive SVMs. Fu et al. [22] proposed an asymmetric-loss cost-sensitive ν-SVM, and Purwadi et al. [26] studied an adaptive SVM constraint adjustment driven by the observed class ratio. These studies motivate explicit minority-aware evaluation, but they do not imply that a weighted SVM must improve aggregate error on every dataset.

Robust optimization studies decisions that remain feasible or effective under uncertainty sets [9,10]. For SVMs, uncertainty in observations can produce regularization or explicit worst-case margin corrections [11-15]. Xu et al. [11] established connections between robustness and regularization, while robust classification and input-uncertainty formulations have been developed by Trafalis and Gilbert [12], Bi and Zhang [13], Wang and Pardalos [14], and Bertsimas et al. [15]. Faccini et al. [16] studied robust and distributionally robust linear SVMs. The nonlinear robust model audited here maps input perturbations to feature-space uncertainty and bounds their margin effect through kernel-dependent radii [5].

The recent literature also separates model-specific robust classification from broader distributionally robust learning. Asimit et al. [23] developed SVM classifiers for feature uncertainty, whereas Yu et al. [24] addressed scalable min--max optimization for distributionally robust supervised learning and Zhu et al. [25] studied label-distributionally robust multiclass losses. These formulations protect against different uncertainty mechanisms and are not interchangeable. Accordingly, the present study evaluates only the uncertainty sets and robust counterparts stated in Sections 3 and 4; it does not claim coverage of distribution shift, label noise, or adversarial attacks in general.

Four meanings are kept distinct throughout this study. *Numerical stability* denotes reliable finite-precision solution with acceptable conditioning, residuals, and feasibility. *Optimization robustness* denotes the worst-case protection encoded by a stated uncertainty set and its robust counterpart. *Statistical generalization* denotes stability beyond the sampled data or under distribution shift, while *adversarial robustness* denotes resistance to deliberately chosen perturbations or attacks. The robust formulations evaluated here instantiate optimization robustness, the solver studies assess numerical stability, and repeated holdouts measure held-out predictive utility under the fixed local data protocol. They do not establish cross-population statistical generalization or adversarial robustness.

The present work is also related to reproducible empirical comparison. Demšar [17] emphasized paired, multi-dataset evaluation and appropriate nonparametric comparisons. Scikit-learn [18] and the UCI repository [19] provide widely used implementations and benchmark data, but software availability alone does not resolve protocol ambiguity. Our focus is narrower: exact executable-source semantics, frozen prospective gates, paired seed-level descriptive analysis, and numerical diagnostics that distinguish a model failure from a solver or representation failure.

## 3. Baseline and Source Audit

### 3.1 Notation and executable deterministic formulation

Let $\mathcal D=\{(x_i,y_i)\}_{i=1}^{m}$ denote a binary training set, where $x_i\in\mathbb R^d$ and $y_i\in\{-1,+1\}$. Let $k$ be a positive-semidefinite kernel, $K_{ij}=k(x_i,x_j)$ its training Gram matrix, and $D_y=\operatorname{diag}(y_1,\ldots,y_m)$. The audited deterministic source model represents the classifier by coefficients $u\in\mathbb R^m$ and minimizes their $\ell_1$ norm. Introducing $s_j\ge |u_j|$ and hinge slacks $\xi_i\ge0$ gives the linear program

$$
\begin{aligned}
\min_{u,\gamma,\xi,s}\quad
    & \mathbf 1^\top s+\nu\mathbf 1^\top\xi \\
\text{subject to}\quad
    & D_y(KD_yu-\gamma\mathbf 1)+\xi\succeq\mathbf 1,\\
    & -s\preceq u\preceq s,\qquad \xi\succeq0,\quad s\succeq0.
\end{aligned} \tag{1}
$$

Here $\nu>0$ penalizes margin violations and $\gamma$ is the optimization intercept. The corresponding unthresholded kernel expansion and final binary rule are

$$
h_u(x)=\sum_{j=1}^{m} y_j u_j k(x_j,x),
\qquad
\widehat y(x)=\operatorname{sign}\!\left[h_u(x)-b\right]. \tag{2}
$$

The source does not set $b=\gamma$ directly. For $\omega_A=\max_i(D_y\xi)_i$, $\omega_B=\max_i(-D_y\xi)_i$, and $T=10{,}000$, it scans

$$
b_t=\gamma+1-\omega_B+
\frac{t}{T-1}\left[(\gamma-1+\omega_A)-(\gamma+1-\omega_B)\right],
\quad t=0,\ldots,T-1, \tag{3}
$$

in the generated order. The selected $b$ is the first candidate attaining the minimum source-defined training violation count. The bounds use raw solver-returned $\xi$, not residual-reconstructed slack. Likewise, $\nu$ is selected from the five ascending values $\operatorname{logspace}(-3,0,5)$ by training misclassification, with the first minimum retained.

The active kernels are program specific. The audited robust binary program uses the homogeneous quadratic kernel $k(x,z)=(x^\top z)^2$; its RBF block is commented. The robust multiclass program uses the homogeneous linear kernel $k(x,z)=x^\top z$, whereas its neighboring deterministic program uses a different kernel. Binary rows are class-blocked after mapping labels to $\pm1$. Multiclass fitting is OVA over source-ordered labels: one binary model is fitted per class and prediction uses the largest class score. The unchanged robust program hard-codes uncertainty construction for three class blocks, and exact score ties have no safe scalar resolution in the MATLAB code.

### 3.2 Executable robust q1 formulation

The robust source augments every nominal margin with a precomputed feature-space uncertainty radius. Its implemented q1 problem is

$$
\begin{aligned}
\min_{u,\gamma,\xi,s}\quad
    & \mathbf 1^\top s+\nu\mathbf 1^\top\xi \\
\text{subject to}\quad
    & y_i\big[(KD_yu)_i-\gamma\big]+\xi_i
      -\delta_i\sum_{j=1}^{m}\sqrt{K_{jj}}\,s_j\ge1,
      \quad i=1,\ldots,m,\\
    & -s\preceq u\preceq s,\qquad \xi\succeq0,\quad s\succeq0.
\end{aligned} \tag{4}
$$

The uncertainty is not an optimization variable. For original class $c$, the input-space radius and the norm-embedding factor are

$$
\eta_c=\rho\max_{1\le r\le d}\operatorname{sd}(X_{c,r}),
\qquad
C(d,p)=
\begin{cases}
1, & 1\le p\le2,\\
d^{(p-2)/(2p)}, & 2<p<\infty,\\
\sqrt d, & p=\infty.
\end{cases} \tag{5}
$$

Here $\rho\ge0$ is shared across classes, whereas $\eta_c$ and hence $\delta_i$ are class specific. The factor $C(d,p)$ embeds the input p-ball into an Euclidean ball. The Hölder dual is $p^*=\infty,2,1$ for $p=1,2,\infty$, respectively, but no dual-norm decision variable appears in Eq. (4); $p$ only changes the propagated radius. The distributed executable fixes regularizer $q=1$ and input norm $p=\infty$; $p=1$ and $p=2$ require a manual source switch.

For the Gaussian RBF feature map with bandwidth $\alpha$, the commented source branch uses the bound

$$
\delta_c=
\sqrt{2-2\exp\!\left[-\frac{C(d,p)^2\eta_c^2}{2\alpha^2}\right]}.
\tag{6}
$$

The paper describes seven rho values for a principal table and a 60-point sensitivity study. The binary executable instead loops over 60 ascending values and the multiclass executable over seven, creating a new unseeded split inside each rho loop. Rho columns are therefore unmatched, and no validation-based rho selector is implemented. Binary CVX defaults to SDPT3; multiclass explicitly requests MOSEK with high precision. The Python reproduction matches the LP structure but does not claim solver-level numerical parity.

### 3.3 Source-paper discrepancies and reproduction boundary

Discrepancies were preserved as audit findings rather than silently corrected. They include commented versus active kernels, preprocessing in adjacent deterministic scripts but not robust drivers, OVA threshold inconsistencies, a multiclass output-array naming defect, unmatched rho sweeps, and an inhomogeneous-polynomial uncertainty accumulation defect that is inactive under the supplied $c=0$ robust kernels. Accordingly, *source-faithful* means matching a specified executable path. It does not mean that every paper table was reproduced, nor does it make literature point estimates paired evidence for the present experiments.

## 4. Evaluated Architectures

### 4.1 Scientific names and project-label mapping

Table 1 defines the manuscript terminology. Internal labels appear only to connect the paper to immutable code and result artifacts; all subsequent discussion uses the scientific names.

**Table 1. Mapping from internal project labels to scientific manuscript names.**

| Internal label | Scientific manuscript name | Defining structure | Final status |
|---|---|---|---|
| Legacy | Source-Profile q1 Kernel SVM | coefficient-L1 kernel LP; source threshold; OVA for multiclass | executable/reference baseline |
| V7 | RKHS-RBF-OVO SVM | standard nested-tuned Gaussian-RBF SVM control; RKHS-L2; OVO; error-based inner selection | better-supported error-oriented control under the evaluated protocol |
| V8-C | CS-RKHS-RBF-OVO SVM | RKHS-RBF-OVO plus normalized class-sensitive slack and balanced selection | supported conditional alternative |
| V8.1 | Training-Only Dual-Kernel Selection SVM | training-only RBF versus normalized quadratic choice | rejected at Gate8 |
| Robust V8-C | RKHS-Norm Robust Class-Sensitive SVM | class-sensitive RKHS-L2 robust SOCP | numerically rejected |
| R8 | Source-Aligned Class-Sensitive Robust q1 LP | weighted source q1 robust LP with source-profile kernels | numerically rejected |
| R8.1 | Equilibrated Source-Aligned Robust q1 LP | exact diagonal coordinate scaling of the preceding LP | numerically rejected |
| R9 | Bounded-RBF Class-Sensitive Robust q1 SVM | weighted robust q1 LP with Gaussian RBF kernel | rejected at fresh Gate8 |

CS denotes *class-sensitive*. The naming is descriptive rather than a novelty claim.

### 4.2 Source-Profile q1 Kernel SVM

The reference architecture is Eq. (1) with the frozen dataset-specific source profile. Numeric preprocessing follows the assigned source-compatible transform; the categorical recurrence cohort uses the documented train-only encoding described in Section 5.2. The kernel is profile specific rather than universally RBF. Regularization is $\lVert u\rVert_1$, slack is unweighted, binary decisions use Eqs. (2)-(3), and multiclass tasks use OVA score maximization. Hyperparameter selection scans the five $\nu$ values in source order and retains the first training-error minimum. Because some local datasets require an adapter not present in the supplied MATLAB program, *source-profile* is used instead of claiming literal executable parity.

### 4.3 RKHS-RBF-OVO SVM

This architecture is the study's standard nested-tuned Gaussian-RBF SVM control. It is identified internally as V7 and serves as the deterministic reference for the later class-sensitive and robust extensions.

All transformations for this architecture are fitted on the current training partition, including within inner folds. Its Gaussian kernel is

$$
k_\alpha(x,z)=\exp\!\left(-\frac{\lVert x-z\rVert_2^2}{2\alpha^2}\right),
\qquad \alpha>0, \tag{7}
$$

with RKHS $\mathcal H_\alpha$. For each binary problem, including each OVO class pair, the model solves

$$
\begin{aligned}
\min_{f\in\mathcal H_\alpha,b,\xi}\quad
    & \frac12\lVert f\rVert_{\mathcal H_\alpha}^{2}+C\sum_{i=1}^{m}\xi_i\\
\text{subject to}\quad
    & y_i\big[f(x_i)+b\big]\ge1-\xi_i,
      \qquad \xi_i\ge0.
\end{aligned} \tag{8}
$$

The regularizer is therefore the squared RKHS norm, not the coefficient $\ell_1$ norm in Eq. (1). No class weights are used. The frozen search is $C\in\{0.1,1,10,100\}$ and eight training-only bandwidth rules: the paper rule plus median-distance multipliers $2^r$ for $r\in\{-2,-1,-0.5,0,0.5,1,2\}$. Three-fold stratified inner cross-validation minimizes mean validation error; declaration order and then C order resolve exact ties. Binary prediction uses the native SVM intercept, and multiclass prediction uses deterministic native OVO voting. No probability calibration, normalized voting, or test-dependent selection is applied.

Relative to the Source-Profile q1 Kernel SVM, this architecture jointly changes regularizer geometry, kernel policy, multiclass decomposition, solver path, hyperparameter selection, and intercept construction. Its comparison with the baseline is consequently an architecture-level evaluation, not a single-factor ablation.

### 4.4 CS-RKHS-RBF-OVO SVM

The class-sensitive architecture keeps the preprocessing, Gaussian kernel, RKHS-L2 regularizer, OVO decomposition, search grid, and prediction rule of Eq. (8). For an OVO pair with class counts $n_a,n_b$, $m_{ab}=n_a+n_b$, and $\tau=0.5$, every observation from class $c\in\{a,b\}$ receives

$$
w_c=
\frac{m_{ab}n_c^{-\tau}}
     {n_a^{1-\tau}+n_b^{1-\tau}},
\qquad
n_aw_a+n_bw_b=m_{ab}. \tag{9}
$$

The second identity shows that the sample mean weight is exactly one. The binary-pair objective becomes

$$
\min_{f\in\mathcal H_\alpha,b,\xi}
\frac12\lVert f\rVert_{\mathcal H_\alpha}^{2}
+C\sum_{i=1}^{m_{ab}}w_{y_i}\xi_i,
\quad
y_i[f(x_i)+b]\ge1-\xi_i,\quad \xi_i\ge0. \tag{10}
$$

Hyperparameters are selected by mean balanced validation loss rather than mean error. At $\tau=0$, $w_i=1$ and Eq. (10) reduces exactly to Eq. (8) under error-based selection. Thus the class-sensitive architecture differs only through the normalized slack penalties and the aligned inner selection criterion. Balanced accuracy is its primary endpoint; conventional error is retained as a prespecified guard.

### 4.5 Robust formulations evaluated as negative evidence

The RKHS-Norm Robust Class-Sensitive SVM replaces the nominal margin in Eq. (10) by

$$
y_i[f(x_i)+b]-\delta_i\lVert f\rVert_{\mathcal H_\alpha}
\ge1-\xi_i. \tag{11}
$$

For fixed radii this is convex and reduces to the class-sensitive deterministic architecture when $\rho=0$, but its finite representation is an SOCP that exposes Gaussian Gram geometry inside the cone.

The Source-Aligned Class-Sensitive Robust q1 LP instead retains Eq. (4) and replaces $\nu\sum_i\xi_i$ by $\nu\sum_i w_i\xi_i$, using mean-one weights defined over each binary sign group. It remains an LP and reduces to Eq. (4) when weighting is disabled. The equilibrated variant applies only the bijective coordinate transformation $x=D_sz$ and positive row scaling $SAD_sz\le Sb$ to the canonical LP.

The Bounded-RBF Class-Sensitive Robust q1 SVM uses Eq. (7) with the prospectively fixed training-only bandwidth

$$
\alpha=\max_r\operatorname{sd}(X_{\mathrm{train},r}),
\qquad p=2,\quad \rho=0.01. \tag{12}
$$

Because $K_{jj}=1$, Eq. (4)'s robust term becomes $\delta_i\sum_j s_j$. The objective and every constraint remain affine: no SOC, quadratic form, Gram inverse, factorization, jitter, or spectral truncation is required. At $\rho=0$ it reduces exactly to the weighted deterministic RBF q1 LP; disabling the weights recovers the unweighted robust RBF q1 LP.

## 5. Experimental Methodology

### 5.1 Dataset audit and inclusion

Every file under the local dataset directory was inventoried and hashed. Nine files were usable classification datasets; a tenth was an acquisition-status table rather than observations. Table 2 summarizes the final cohorts and frozen preprocessing. Eight Dermatology rows with missing age and six incomplete Heart rows were removed under the pre-existing complete-case protocols. The 830-row Mammographic file was already complete. Breast Cancer recurrence is a distinct 286-row categorical cohort, not the 569-row diagnostic cohort. Six spreadsheet-rendered interval tokens were repaired deterministically, nine missing categorical cells were represented by an explicit category, and no recurrence observation was removed.

**Table 2. Dataset characteristics and frozen preprocessing. $N$ is the modeled sample count, $d$ the input dimension before categorical expansion, and $L$ the number of classes.**

| Dataset | $N$ | $d$ | $L$ | Class counts | Frozen preprocessing |
|---|---:|---:|---:|---|---|
| Blood Transfusion | 748 | 4 | 2 | 570/178 | standardization |
| Breast Cancer Diagnostic | 569 | 30 | 2 | 212/357 | min-max scaling |
| Breast Cancer recurrence | 286 | 9 | 2 | 201/85 | train-only one-hot encoding and standardization |
| Dermatology | 358 | 34 | 6 | 111/60/71/48/48/20 | complete-case; no numeric transform |
| Heart Disease | 297 | 13 | 5 | 160/54/35/35/13 | complete-case and standardization |
| Iris | 150 | 4 | 3 | 50/50/50 | no transform |
| Mammographic Mass | 830 | 5 | 2 | 427/403 | standardization |
| Parkinson | 195 | 22 | 2 | 48/147 | min-max scaling |
| Wine | 178 | 13 | 3 | 59/71/48 | standardization |

### 5.2 Splits, preprocessing, and model selection

Outer evaluation used stratified 75/25 holdouts with sorted indices. Models compared within an experiment received identical splits. Preprocessing parameters were fitted on training data only, including inside every inner fold. Numeric profiles used the frozen standardization or min-max rule assigned to each dataset. Breast Cancer recurrence used train-only one-hot vocabularies with unknown validation/test categories mapped to all-zero columns, followed by train-only standardization. No architecture or search space was changed after test results were observed.

Hyperparameters were selected only from outer-training data. The RKHS-RBF-OVO and CS-RKHS-RBF-OVO SVMs used three-fold stratified inner cross-validation with deterministic seed $20000+$ outer seed. The Source-Profile q1 Kernel SVM retained the source selection and threshold semantics in Eqs. (1)-(3). Predictions, confusion matrices, per-class recall, selected hyperparameters, runtimes, solver status, split hashes, dataset/preprocessing hashes, and source/configuration fingerprints were checkpointed per seed.

### 5.3 Prospective stages and seed-range separation

Experiments proceeded through one-seed smoke, eight-seed Gate8, 24-seed Gate24, and 96-seed confirmation only when the predeclared gate permitted promotion. The deterministic campaigns used separate development and confirmation ranges: the 7000/8000 series for the original source-profile comparison, the 9000/10000 series for the class-sensitive study, the 0/4000 series for Iris, and the 16000-16395 series for the expanded recurrence and Dermatology study. Gate estimates were used only for screening or promotion and were never relabeled as untouched confirmation. Here, *confirmation* means that a repeated-holdout seed range remained untouched until after the architecture and decision rules were frozen; it does not mean that the 96 splits are independent cohorts, datasets, or experiments.

After the original class-sensitive campaign and confirmation were complete, a later targeted ablation filled the previously missing square-root-weighting/error-selection cell on the already consumed Gate8 seeds 9000-9007 and the same five Gate8 datasets. It reused the immutable square-root inner-validation banks and exact stored splits, fitted only the missing outer model on Modal, and did not alter the original promotion rules or confirmation model. Because the splits and prior Gate8 results were already known, this experiment is retrospective development evidence rather than a fresh prospective gate or confirmation.

Robust seeds 15000-15023 were exposed during solver qualification. The Bounded-RBF Class-Sensitive Robust q1 SVM used fresh Gate8 seeds 15300-15307. Its intended Gate24 range 15400-15423 and confirmation range 14000-14095 remained untouched after the candidate failed. These unused ranges are not evidence in this article.

The promotion rules coupled predictive and numerical criteria: complete task accounting, accepted solver status, finite outputs, matched splits, feasibility, bounded runtime, and endpoint-specific performance guards were all required. Table 3 makes the resulting evidence levels explicit. Stopped branches were not resumed or replaced by different seeds.

**Table 3. Final evidence level by dataset and deterministic comparison. “Confirmation” denotes an untouched 96-seed repeated-holdout run, not independent cohorts or datasets; Gate8 and Gate24 are development evidence.**

| Dataset | Source-Profile q1 vs RKHS-RBF-OVO | RKHS-RBF-OVO vs CS-RKHS-RBF-OVO |
|---|---|---|
| Blood Transfusion | stopped at Gate8 (8 seeds) | confirmation (96 seeds) |
| Breast Cancer Diagnostic | confirmation (96 seeds) | confirmation (96 seeds) |
| Breast Cancer recurrence | confirmation (96 seeds) | stopped at Gate24 (24 seeds) |
| Dermatology | confirmation (96 seeds; independent replication also retained) | stopped at Gate24 (24 seeds) |
| Heart Disease | confirmation (96 seeds) | confirmation (96 seeds) |
| Iris | confirmation (96 seeds) | confirmation (96 seeds) |
| Mammographic Mass | stopped at Gate8 (8 seeds) | confirmation (96 seeds) |
| Parkinson | confirmation (96 seeds) | confirmation (96 seeds) |
| Wine | confirmation (96 seeds) | confirmation (96 seeds) |

### 5.4 Outcomes and statistical analysis

The primary endpoint for the RKHS-RBF-OVO SVM was paired test error relative to the Source-Profile q1 Kernel SVM. The primary endpoint for the class-sensitive extension was paired balanced accuracy relative to the unweighted RKHS architecture. For $L$ classes, the reported metrics are

$$
\operatorname{Err}=\frac1n\sum_{i=1}^{n}\mathbb I(\widehat y_i\ne y_i),
\qquad
\operatorname{BA}=\frac1L\sum_{\ell=1}^{L}\operatorname{Recall}_{\ell},
\qquad
\operatorname{MacroF1}=\frac1L\sum_{\ell=1}^{L}F1_{\ell}. \tag{13}
$$

Minority recall denotes the training-minority class for binary data. For multiclass safety summaries, the rare or minimum-recall class was tracked according to the frozen protocol. Runtime and solver outcomes were recorded for every seed.

For each paired metric, $\Delta_s=M_{\mathrm{candidate},s}-M_{\mathrm{reference},s}$ on seed $s$. Untouched confirmations report the following descriptive paired split-variation interval and standardized paired effect:

$$
\overline\Delta\ \pm\ t_{0.975,n-1}\frac{s_\Delta}{\sqrt n},
\qquad
d_{\mathrm{paired}}=\frac{\overline\Delta}{s_\Delta}, \tag{14}
$$

where $s_\Delta$ is the sample standard deviation of paired differences. Its 95% coverage is nominal and descriptive. Because repeated holdouts overlap and their paired differences are dependent, the formula summarizes conditional split variation under the fixed local dataset and protocol; it is not a formal independent-sample confidence interval and does not quantify population uncertainty across institutions, acquisition settings, or cohorts. We also report wins/ties/losses and standardized paired effects. A win means lower candidate error or higher candidate score. The main manuscript omits resampling-based $p$-values because overlapping repeated holdouts do not create independent experimental units. Secondary and classwise results remain exploratory rather than separate confirmatory superiority claims. No confirmatory claim is based on a family of $p$-values. These summaries characterize evidence strength rather than impose an automatic significance rule. Gate8 summaries are screening diagnostics; they are not treated as equivalent to 96-seed confirmation.

### 5.5 Computing environment and provenance

Model fitting used checkpointed Modal CPU jobs with no GPU and no more than four concurrent workers. Runs were resumable, used immutable identifiers and atomic per-seed checkpoints, and rejected incompatible configuration fingerprints. Each scientific record preserves the source, configuration, dataset, preprocessing, and split hashes together with predictions, selected hyperparameters, solver diagnostics, and runtime.

The targeted square-root/error Gate8 used Modal app `ap-5Qt6AOVEn68np3HqGt1BFj`. Its saved configuration records the five datasets, seeds 9000-9007, fixed alpha and C grids, exact prior-run fingerprint, source hashes, reused inner-bank provenance, and 40 newly fitted outer models. All downloaded files matched the SHA-256 values in the completed manifest.

The full SHA-256 values for the frozen RKHS-RBF-OVO and CS-RKHS-RBF-OVO implementations are provided in the accompanying evidence map and QA report. They were rechecked after manuscript generation and remained unchanged. Runtime ratios include the complete frozen model-selection workload in the recorded CPU environment; they are operational comparisons, not hardware-independent complexity constants.

## 6. Deterministic Results

### 6.1 RKHS-RBF-OVO SVM versus the source-profile baseline

Seven comparisons reached untouched 96-seed repeated-holdout confirmation. The RKHS-RBF-OVO SVM had lower mean error on all seven, and every descriptive 95% paired split-variation interval lay below zero (Table 4). This 95% coverage is nominal/descriptive for dependent repeated-holdout differences, not a formal independent-sample confidence interval. The largest effect occurred on Parkinson: mean test error decreased from 14.24% to 7.08%, a paired reduction of 7.16 percentage points. Breast Cancer recurrence decreased from 30.64% to 26.81%, a 3.83-point reduction. Improvements on Iris, Heart, Wine, Dermatology, and Breast Cancer Diagnostic were smaller but retained the same paired direction.

**Table 4. Confirmed error performance of the RKHS-RBF-OVO SVM against the Source-Profile q1 Kernel SVM. All rows contain 96 untouched paired repeated-holdout seeds. Lower error is better; $\Delta$ is candidate minus reference. The 95% intervals have nominal/descriptive coverage for dependent splits and are not formal independent-sample confidence intervals.**

| Dataset | $N_{\mathrm{seeds}}$ | Source-profile error | RKHS-RBF-OVO error | $\Delta$ error, pp [descriptive 95% paired split interval] |
|---|---:|---:|---:|---:|
| Parkinson | 96 | 14.243% | **7.079%** | **-7.164 [-8.161, -6.167]** |
| Breast Cancer recurrence | 96 | 30.642% | **26.808%** | **-3.834 [-4.714, -2.954]** |
| Iris | 96 | 4.331% | **3.125%** | **-1.206 [-1.800, -0.612]** |
| Heart Disease | 96 | 43.792% | **42.639%** | **-1.153 [-1.985, -0.321]** |
| Wine | 96 | 2.755% | **2.037%** | **-0.718 [-1.204, -0.231]** |
| Dermatology | 96 | 4.271% | **3.808%** | **-0.463 [-0.903, -0.023]** |
| Breast Cancer Diagnostic | 96 | 2.855% | **2.601%** | **-0.255 [-0.475, -0.035]** |

**Table 5. Paired descriptive diagnostics for the confirmed error endpoint in Table 4. W/T/L denotes candidate wins, ties, and losses; the effect is $d_{\mathrm{paired}}$ from Eq. (14).**

| Dataset | W/T/L | Paired effect |
|---|---:|---:|
| Parkinson | 85/9/2 | -1.456 |
| Breast Cancer recurrence | 80/4/12 | -0.882 |
| Iris | 40/39/17 | -0.412 |
| Wine | 44/31/21 | -0.299 |
| Heart Disease | 54/13/29 | -0.281 |
| Breast Cancer Diagnostic | 48/26/22 | -0.235 |
| Dermatology | 46/21/29 | -0.213 |

![Figure 1. Paired error changes for the RKHS-RBF-OVO SVM on seven untouched 96-seed repeated-holdout confirmations. Points are mean candidate-minus-reference changes and bars are descriptive 95% paired split-variation intervals. Coverage is nominal/descriptive because the repeated holdouts overlap; the bars are not formal independent-sample confidence intervals. Every interval lies below zero, although effect sizes vary substantially across datasets.](figures/fig1_rkhs_error_deltas.png){ width=88% }

The aggregate-error improvement was not uniformly class sensitive. On Breast Cancer recurrence, majority recall increased by 8.88 points while minority recall decreased from 34.47% to 26.04%, a paired loss of 8.43 points (descriptive 95% paired split-variation interval -12.04 to -4.82). Dermatology error improved slightly, but rare class-6 recall decreased from 98.96% to 96.04% (-2.92 points; descriptive interval -4.47 to -1.36). These 95% ranges are again nominal descriptions of dependent split variation, not formal confidence intervals from independent samples. Table 6 further shows that the Heart rare-class recall decreased to zero despite improvements in balanced accuracy and macro-F1. The architecture is therefore supported as an error-oriented model, not as a universal remedy for class imbalance.

**Table 6. Confirmed class-balanced metrics for the Source-Profile q1 and RKHS-RBF-OVO SVMs. Entries show reference to candidate, followed by the paired percentage-point change. All rows use 96 seeds; higher is better.**

| Dataset | Balanced accuracy | Macro-F1 | Rare/minority recall |
|---|---:|---:|---:|
| Parkinson | 72.621% to **88.538%** (+15.917) | 76.071% to **89.888%** (+13.817) | 46.875% to **79.948%** (+33.073) |
| Heart Disease | 25.804% to **29.129%** (+3.326) | 20.167% to **27.875%** (+7.708) | 5.208% to 0.000% (-5.208) |
| Iris | 95.651% to **96.868%** (+1.218) | 95.645% to **96.854%** (+1.209) | 94.025% to **95.768%** (+1.743) |
| Wine | 97.431% to **98.042%** (+0.612) | 97.258% to **97.935%** (+0.678) | 97.917% to **98.177%** (+0.260) |
| Breast Cancer Diagnostic | 96.435% to **96.867%** (+0.433) | 96.898% to **97.188%** (+0.290) | 93.691% to **94.811%** (+1.120) |
| Breast Cancer recurrence | 59.098% to **59.324%** (+0.226) | **58.971%** to 58.636% (-0.335) | **34.474%** to 26.042% (-8.433) |
| Dermatology | **95.534%** to 95.464% (-0.070) | 95.138% to **95.267%** (+0.130) | **98.958%** to 96.042% (-2.917) |

Two additional branches were stopped prospectively at Gate8 and are not mixed with Table 4. On eight development seeds, the RKHS-RBF-OVO error was 1.40 points higher on Blood Transfusion and 1.26 points higher on Mammographic Mass. These are retained as negative screening results and were not promoted to Gate24 or confirmation.

Runtime favored the RKHS architecture on every confirmed source-profile comparison (Table 7). The largest reductions occurred on Dermatology and Heart, while Parkinson showed the smallest proportional reduction. These ratios describe the complete frozen training and selection pipelines rather than asymptotic solver complexity.

**Table 7. Runtime on the seven untouched 96-seed confirmations. Means are seconds per outer seed; ratios below one favor the RKHS-RBF-OVO SVM.**

| Dataset | $N_{\mathrm{seeds}}$ | Source-profile mean, s | RKHS-RBF-OVO mean, s | Runtime ratio |
|---|---:|---:|---:|---:|
| Dermatology | 96 | 3.990 | **0.442** | **0.111x** |
| Heart Disease | 96 | 1.986 | **0.316** | **0.159x** |
| Breast Cancer Diagnostic | 96 | 2.135 | **0.439** | **0.206x** |
| Wine | 96 | 0.507 | **0.152** | **0.300x** |
| Iris | 96 | 0.518 | **0.192** | **0.370x** |
| Breast Cancer recurrence | 96 | 0.583 | **0.343** | **0.588x** |
| Parkinson | 96 | 0.248 | **0.154** | **0.622x** |

### 6.2 Effect of class-sensitive weighting

The original eight-seed development ablation separated error-based versus balanced selection from inverse-frequency weighting (Appendix Table S1). Balanced selection alone improved Blood balanced accuracy modestly, inverse weighting with error selection did not, and their combination produced a larger balance shift with a substantial error cost. A later targeted run filled the missing square-root/error cell on the same stored Gate8 splits (Appendix Table S2). It reused each square-root inner-validation bank and changed only the inner selection metric from balanced loss to error before fitting a new outer model.

Against V7 on Blood, the square-root/error arm reduced mean error by 0.80 percentage points (nominal/descriptive 95% paired split-variation interval -3.47 to +1.87) and increased balanced accuracy by 6.39 points (+1.80 to +10.99) and macro-F1 by 7.18 points (+0.98 to +13.38). Against the square-root/balanced arm on the same splits, error was 0.33 points lower (-0.67 to -0.002), while balanced accuracy was 1.14 points lower (-3.21 to +0.94). On Heart, square-root/error versus V7 changed error by -0.67 points (-2.97 to +1.64), balanced accuracy by +2.25 points (+0.35 to +4.16), and macro-F1 by +2.23 points (+0.10 to +4.36); compared with square-root/balanced selection, it had 2.33 points lower mean error but also 1.03 points lower balanced accuracy and 2.46 points lower macro-F1. Mammographic Mass, Parkinson, and Wine showed only small differences with nominal intervals spanning zero. All intervals in this targeted Gate8 analysis have nominal/descriptive 95% coverage for eight dependent repeated holdouts; they are not formal independent-sample confidence intervals.

The supported interpretation is limited: on these already examined development splits, the Blood and Heart balance gains were not produced by balanced selection alone because they remained under error selection, while balanced selection produced larger mean balance shifts at an error cost, especially on Heart. The targeted arm was retrospective, used only eight dependent splits, and did not rerun or replace the untouched 96-seed square-root/balanced confirmation; it therefore refines component attribution without authorizing a new promoted model.

The confirmed CS-RKHS-RBF-OVO SVM produced two substantively useful class-sensitive outcomes and several null or adverse controls. On Blood, balanced accuracy increased from 61.35% to 67.17%, a +5.82-point paired change (descriptive 95% paired split-variation interval +4.96 to +6.68; 92/0/4; paired effect 1.369). The interval has nominal/descriptive coverage for dependent repeated holdouts and is not a formal independent-sample confidence interval. Macro-F1 increased by 5.57 points. Minority recall increased from 28.73% to 45.97%, whereas majority recall decreased by 5.60 points. Mean error changed by only +0.16 points and its paired split-variation interval included zero. This is a redistribution of errors toward minority recovery, not an overall accuracy gain.

On the local five-class Heart task, class-sensitive weighting increased balanced accuracy by 2.55 points and macro-F1 by 2.88 points. The tradeoff was explicit: error increased by 1.88 points (descriptive 95% paired split-variation interval +1.04 to +2.71) and runtime increased by a factor of 4.99. Here too, 95% denotes nominal/descriptive coverage for dependent repeated-holdout differences rather than a formal independent-sample confidence interval. Absolute Heart error remained high, so the result supports class-sensitive redistribution under this local protocol, not clinical readiness.

**Table 8. CS-RKHS-RBF-OVO SVM versus its unweighted RKHS-RBF-OVO reference on untouched 96-seed confirmations. Each cell gives unweighted to class-sensitive performance and the paired change. Positive error is worse; positive balanced accuracy and macro-F1 are better.**

| Dataset | Error | Balanced accuracy | Macro-F1 | Runtime ratio |
|---|---:|---:|---:|---:|
| Blood Transfusion | 21.580% to 21.736% (+0.156) | 61.347% to **67.168%** (+5.820) | 62.511% to **68.082%** (+5.571) | 0.616x |
| Heart Disease | **42.264%** to 44.139% (+1.875) | 30.311% to **32.864%** (+2.553) | 29.101% to **31.980%** (+2.879) | 4.986x |
| Parkinson | 7.313% to 7.483% (+0.170) | 88.677% to 89.678% (+1.002) | 89.635% to 89.804% (+0.169) | 2.575x |
| Mammographic Mass | 16.607% to 16.562% (-0.045) | 83.284% to 83.340% (+0.056) | 83.296% to 83.346% (+0.050) | 0.954x |
| Breast Cancer Diagnostic | 2.491% to 2.520% (+0.029) | 97.027% to 97.085% (+0.058) | 97.310% to 97.284% (-0.026) | 1.464x |
| Wine | 2.014% to 2.245% (+0.231) | **98.011%** to 97.841% (-0.170) | **97.970%** to 97.748% (-0.221) | 3.931x |
| Iris | 3.289% to 3.289% (0.000) | 96.753% to 96.750% (-0.002) | 96.722% to 96.721% (-0.001) | 4.763x |

![Figure 2. Error and balanced-accuracy changes associated with class-sensitive weighting on seven 96-seed confirmations. The upper-left region represents simultaneous error and balance improvement; the upper-right represents a balance gain purchased with higher error. Blood shows the strongest balance gain with no confirmed error change, whereas Heart shows a clear tradeoff.](figures/fig2_class_sensitive_tradeoff.png){ width=83% }

Classwise recall clarifies those tradeoffs (Table 9). On Blood the minority gain was large and consistent, accompanied by a smaller majority loss. On Heart, recall increased for classes 2-5 but decreased by 8.31 points for the majority class. The rarest class improved from 0.69% to 6.25%, yet remained poorly recognized.

**Table 9. Class-recall changes on the two datasets with confirmed class-sensitive signal. All estimates use 96 paired confirmation seeds; $\Delta$ is class-sensitive minus unweighted recall. Bracketed 95% paired split intervals have nominal/descriptive coverage for dependent repeated holdouts, not formal independent-sample confidence coverage.**

| Dataset / class | Unweighted recall | Class-sensitive recall | $\Delta$, pp [descriptive 95% paired split interval] |
|---|---:|---:|---:|
| Blood / class 0 (majority) | **93.967%** | 88.364% | -5.603 [-6.114, -5.092] |
| Blood / class 1 (minority) | 28.728% | **45.972%** | +17.244 [+15.525, +18.963] |
| Heart / class 1 | **93.047%** | 84.740% | -8.307 [-9.529, -7.085] |
| Heart / class 2 | 17.188% | **24.256%** | +7.069 [+5.082, +9.055] |
| Heart / class 3 | 19.560% | **24.190%** | +4.630 [+1.714, +7.545] |
| Heart / class 4 | 21.065% | **24.884%** | +3.819 [+1.401, +6.238] |
| Heart / class 5 (rarest) | 0.694% | **6.250%** | +5.556 [+2.509, +8.602] |

The other confirmations constrain the scope of the class-sensitive claim. Mammographic Mass, Breast Cancer Diagnostic, Wine, and Iris showed no confirmed balance advantage. Parkinson's mean balanced-accuracy change had a paired split-variation interval that included zero. The recurrence and Dermatology class-sensitive controls did not satisfy their prospective Gate24 promotion rules and therefore did not consume confirmation seeds.

![Figure 3. Runtime ratios for the two supported deterministic comparisons. The left panel shows RKHS-RBF-OVO divided by Source-Profile q1 runtime; the right panel shows class-sensitive divided by unweighted RKHS runtime. The parity line is one and the horizontal scale is logarithmic.](figures/fig3_runtime_ratios.png){ width=90% }

## 7. Expanded Dataset Study

The expanded audit required every scientifically suitable local dataset to have a completed protocol path; datasets were not added selectively because they favored a model. Existing immutable evidence covered seven of the nine classifier datasets. New staged computation evaluated Breast Cancer recurrence and added the previously missing class-sensitive control on Dermatology.

Both datasets completed smoke and Gate8 without failure. Only the prespecified RKHS-RBF-OVO-versus-source-profile error endpoint advanced. Gate24 used seeds 16200-16223 and completed 48 dataset-seed jobs and 144 model records. Table 10 reports these development results separately from the later confirmation estimates.

**Table 10. Expanded-study Gate24 promotion evidence. These are 24-seed development results, not confirmation estimates. Negative error change favors the RKHS-RBF-OVO SVM. Bracketed 95% paired split intervals have nominal/descriptive coverage for dependent repeated holdouts and are not formal independent-sample confidence intervals.**

| Dataset | $N_{\mathrm{seeds}}$ | $\Delta$ error, pp [descriptive 95% paired split interval] | W/T/L | Prospective decision |
|---|---:|---:|---:|---|
| Breast Cancer recurrence | 24 | -3.646 [-5.683, -1.608] | 19/0/5 | promote source-profile comparison |
| Dermatology | 24 | -0.324 [-1.171, +0.523] | 12/4/8 | promote under frozen mean/W-T-L rule |

The recurrence class-sensitive control increased balance but exceeded its error guard; the Dermatology class-sensitive control had no qualifying balance gain. Neither was carried into confirmation.

The untouched expanded confirmation used seeds 16300-16395 and fitted only the Source-Profile q1 and RKHS-RBF-OVO models, as prospectively authorized. It completed 192 jobs and 384 model records without failure or nonfinite output. Recurrence confirmed the error reduction in Table 4 and ran at 0.588 times source-profile runtime, but its minority-recall loss was also confirmed. Dermatology confirmed a small error reduction and a 0.111 runtime ratio, accompanied by the class-6 recall loss. These datasets strengthen the study because they reproduce an error/runtime advantage while revealing where aggregate-error optimization can be unsafe for rare classes.

Dermatology also has an earlier independent 96-seed source-profile comparison. That earlier result was statistically indistinguishable, whereas the expanded replication produced a small interval below zero. Both records are preserved; the later result does not erase the earlier null. Across the complete study, seven source-profile comparisons reached confirmation and two stopped at Gate8; seven class-sensitive comparisons reached confirmation and two stopped at Gate24.

## 8. Robust Optimization Study

### 8.1 RKHS-norm robust SOCP

The RKHS-Norm Robust Class-Sensitive SVM was evaluated first. In a 24-seed development gate, Parkinson completed all matched seeds and showed small favorable mean predictive changes whose intervals included zero. Runtime increased by a factor of approximately 21.14, and 18 of 24 selected outer models were reported as `optimal_inaccurate`. Blood seeds 15000 and 15001 repeatedly failed under CLARABEL; the remaining dataset comparisons were therefore stopped and the candidate was not frozen.

Numerical qualification separated implementation, data, conditioning, and solver explanations. Exact aggregation of duplicate rows removed redundant geometry and repaired all overlapping accepted inner cases, but both exposed Blood outer problems still failed. SCS produced only 5 accepted solutions among 16 required calls under the prospective feasibility policy, and an independent high-accuracy CVXOPT path returned no solution for either outer problem. No fallback solver qualified.

The proposed exact coordinate representation required $K=R^\top R$ and $z=Rc$, giving $\lVert f\rVert_{\mathcal H}=\lVert z\rVert_2$. Stored float64 Blood matrices did not admit the required full-rank factorization without a model-changing repair. Recomputing the kernels at 2,048-bit precision showed that the mathematical matrices were positive definite: estimated minimum eigenvalues were $2.47\times10^{-22}$ and $2.38\times10^{-23}$, with condition estimates $7.11\times10^{23}$ and $5.01\times10^{24}$. High-precision Cholesky succeeded, but no unchanged solver-usable float64 representation preserved this full-rank geometry. Mathematical positive definiteness therefore did not imply usable finite-precision cone coordinates.

### 8.2 Source-aligned robust LP and exact equilibration

The Source-Aligned Class-Sensitive Robust q1 LP removed the SOCP while retaining Eq. (4), source-profile kernels, and mean-one class weights. Algebraic reductions and convexity tests passed, and Parkinson solved. Blood nevertheless failed even for the deterministic source-profile LP at the first $\nu=0.001$. Its inhomogeneous polynomial profile produced canonical coefficient dynamic ranges near $10^{19}$-$10^{20}$ for deterministic problems and $10^{21}$-$10^{22}$ for robust problems.

The Equilibrated Source-Aligned Robust q1 LP then applied a prospectively fixed positive diagonal variable transformation and row scaling, preserving the feasible set and objective in exact arithmetic. Parkinson passed the strict scaled/unscaled equivalence gate. Only 5 of 8 mandatory Blood variants returned accepted solutions, however, and successful robust fits predicted only the majority class. Exact equilibration was therefore valid but insufficient, and the branch was closed without further solver or tolerance search.

**Table 11. Numerical qualification of the rejected robust formulations. All Blood probes used already exposed development seeds 15000 and 15001; no fresh predictive seed was consumed.**

| Scientific formulation | Numerical evidence | Qualification result |
|---|---|---|
| RKHS-Norm Robust Class-Sensitive SVM | float64 Gram factor unavailable; high-precision matrices SPD but condition estimates $7.11\times10^{23}$ and $5.01\times10^{24}$; CLARABEL and CVXOPT outer failures | not qualified |
| Source-Aligned Class-Sensitive Robust q1 LP | Blood coefficient ranges about $10^{19}$-$10^{22}$; deterministic and robust source-profile failures | not qualified |
| Equilibrated Source-Aligned Robust q1 LP | Parkinson equivalence passed; only 5/8 Blood variants solved; successful robust fits majority-only | not qualified |
| Bounded-RBF Class-Sensitive Robust q1 SVM | weighted robust candidate completed 32/32 fresh fits; two deterministic Blood controls failed; all returned fits feasible within $10^{-7}$ | numerically improved, predictive Gate8 failed |

### 8.3 Bounded-RBF robust LP and fresh Gate8

The Bounded-RBF Class-Sensitive Robust q1 SVM retained $q=1$, $p=2$, $\rho=0.01$, $\tau=0.5$, the five-value $\nu$ order, OVA construction, and source threshold semantics while replacing the polynomial kernel by Eq. (7) under the fixed rule in Eq. (12). Exposed-seed qualification completed on Blood and Parkinson with feasibility within $10^{-7}$ and no SOC or Gram factorization. The architecture and numerical policy were then frozen before the only fresh Gate8 on seeds 15300-15307.

The gate accounted for all 128 expected model tasks. The weighted robust candidate completed 32/32 fits, produced no nonfinite result or majority-only prediction, and every returned model satisfied the feasibility threshold. Two Blood deterministic control tasks failed under HiGHS at seed 15305, leaving 31 matched primary pairs; the failures were recorded rather than omitted. The predictive screening signal was nevertheless adverse (Table 12): mean balanced accuracy decreased on all four datasets, and the Parkinson and Mammographic safety-recall losses violated the prospective guard. Gate24 and confirmation were not launched.

**Table 12. Fresh Gate8 comparison of the Bounded-RBF Class-Sensitive Robust q1 SVM with its weighted deterministic RBF q1 control. These are screening results, not confirmation evidence. Positive error is worse; positive score changes are better.**

| Dataset | Matched pairs | $\Delta$ error, pp | $\Delta$ BA, pp | $\Delta$ macro-F1, pp | $\Delta$ safety recall, pp | Error W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| Blood Transfusion | 7 | +1.451 | -1.377 | -1.955 | -1.248 | 0/1/6 |
| Parkinson | 8 | +2.296 | -5.039 | -4.301 | **-10.417** | 1/3/4 |
| Mammographic Mass | 8 | +0.781 | -0.915 | -0.868 | **-5.569** | 1/1/6 |
| Iris | 8 | +0.329 | -0.321 | -0.332 | 0.000 | 0/7/1 |

![Figure 4. Fresh Gate8 changes for the Bounded-RBF Class-Sensitive Robust q1 SVM relative to its weighted deterministic control. Error increased and balanced accuracy decreased on every dataset. Safety recall decreased most strongly on Parkinson and Mammographic Mass. Blood has seven matched pairs because two deterministic controls failed; the other datasets have eight.](figures/fig4_bounded_rbf_gate8_deltas.png){ width=90% }

The bounded kernel reduced the large positive coefficients that dominated the polynomial LP, but an upper bound of one did not guarantee a narrow nonzero coefficient range. On the failed Blood control split, the minimum positive RBF entry was $2.73\times10^{-47}$, producing a nonzero dynamic range of $3.66\times10^{46}$. This did not prevent completion of the weighted robust candidate, but it reinforces the distinction between bounded kernel values and favorable numerical conditioning.

## 9. Mechanistic Failure Analysis

This post hoc observational mechanism analysis used only stored Bounded-RBF robust Gate8 checkpoints; no new model was fitted and no fresh seed was consumed. It compared each weighted robust model with its weighted deterministic RBF q1 reference and decomposed the trained objective, margins, slacks, coefficients, thresholds, score distributions, prediction flips, and classwise confusion changes.

Across Blood, Parkinson, and Mammographic Mass, the robust term behaved as an additional global pressure on $\lVert u\rVert_1$. Mean coefficient L1 norm fell from 15.29 to 6.12 on Blood, 37.79 to 30.82 on Parkinson, and 33.17 to 17.25 on Mammographic. Mean support counts above $10^{-8}$ fell from 18.57 to 8.71, 46.62 to 30.38, and 58.50 to 27.50, respectively. The optimizer substituted slack for functional capacity: mean slack sums increased from 255.23 to 286.80 on Blood, 11.57 to 37.18 on Parkinson, and 185.69 to 256.68 on Mammographic. Iris changed little, consistent with its nearly identical predictions.

This pattern follows directly from the robust margin correction $\delta_i\sum_j|u_j|$. Increasing the coefficient norm makes every robust margin constraint more expensive, so a coefficient-L1 model can reduce the common uncertainty exposure by shrinking the entire expansion and paying class-weighted slack instead. The protection is class-specific through $\delta_i$, but the coupled norm is global; it does not explicitly guarantee a lower bound on minority margins or recall.

Prediction changes were directionally harmful. Pooling available test observations, robustness changed 45 Blood predictions, correcting 13 deterministic errors and introducing 32 new ones. Parkinson had 25 flips (eight corrected, 17 introduced), Mammographic 105 (46 corrected, 59 introduced), and Iris one (zero corrected, one introduced). Thus the Gate8 losses did not arise from a few numerical outliers; new errors consistently exceeded corrections.

Threshold reconstruction showed a larger adverse pattern on Parkinson and Mammographic. Counterfactual recombination of stored raw functions and thresholds showed that their robust score functions alone were less harmful than the final robust predictions, while movement toward the robust threshold was associated with a larger safety-recall loss (Appendix Table S2). These coordinate diagnostics reused frozen checkpoints and did not fit new models. This pattern is consistent with a mismatch between the class-weighted training objective and the source threshold routine, which minimizes unweighted training misclassification. On Blood the pattern differed: poorer classwise performance was already present in the raw robust score function, while threshold movement partially offset it. Selected $\nu$ shifts were present on some Blood seeds but did not explain the consistent cross-dataset pattern; Parkinson and Mammographic selected $\nu=1$ throughout.

![Figure 5. Mechanism analysis of the failed Bounded-RBF robust Gate8. Left: robust-to-deterministic ratios of coefficient L1 norm, support count, and slack sum. Right: deterministic errors corrected versus new errors introduced by robust prediction flips. Robust fits had lower coefficient magnitude and support and higher slack; new errors exceeded corrections on every dataset.](figures/fig5_robust_failure_mechanism.png){ width=92% }

The strongest evidence-supported interpretation is therefore conditional. The source-derived global uncertainty-norm correction was associated with coefficient shrinkage and slack substitution, together with lower support counts and compressed decision functions. On Parkinson and Mammographic, the observed threshold behavior was consistent with additional safety-class losses under a procedure not optimized for the weighted objective. On Blood, degradation was already present in the score function and threshold movement partially offset it. The data do not support a single claim that $\rho=0.01$ was merely “too conservative,” because the same rho had different classwise and threshold patterns and was never tuned on Gate8.

## 10. Discussion

### 10.1 Where the RKHS-RBF-OVO architecture helps

The RKHS-RBF-OVO SVM is the most consistently supported error-oriented architecture in this study. It achieved a large Parkinson improvement, a moderate recurrence improvement, and smaller paired-supported gains on five additional confirmations. It was also faster than the Source-Profile q1 Kernel SVM on every confirmed dataset. The plausible explanation is not a single component but the combined move from coefficient-L1 regularization, profile-specific kernels, OVA, and threshold scans to RKHS-L2 regularization, Gaussian RBF geometry, nested bandwidth/C selection, and OVO prediction.

That joint change is also a limitation because it prevents attribution to any one component. The Training-Only Dual-Kernel Selection SVM prospectively tested whether choosing between RBF and a normalized quadratic kernel added value. It failed Gate8: Mammographic error improved by only 0.30 points with four wins in eight and a 1.67 runtime ratio, below the frozen promotion requirements, while Blood predictions matched the class-sensitive RBF architecture at approximately 24.76 times its runtime. This negative result supports retaining the simpler frozen kernel policy rather than adding retrospective complexity.

### 10.2 When class-sensitive weighting is appropriate

The CS-RKHS-RBF-OVO SVM answers a different question from its unweighted reference. On Blood it recovered minority observations strongly with almost unchanged total error; on Heart it raised macro-level metrics and rare-class recall while increasing error and runtime. These outcomes are useful when balanced accuracy or minority recovery is an explicitly declared operational objective. They do not justify replacing the unweighted RKHS model everywhere. On relatively balanced or already high-performing datasets, weighting produced negligible gains and often additional computation.

The recurrence result illustrates why class-sensitive evaluation must accompany aggregate error. The RKHS-RBF-OVO SVM's lower error concealed an 8.43-point minority-recall loss. The expanded class-sensitive control recovered balance at Gate24 but exceeded its prospective error guard and was not promoted. This is more informative than selecting the favorable metric after the fact: the choice between the two deterministic RKHS architectures depends on an application-specific loss function declared before evaluation.

### 10.3 Numerical stability is not held-out predictive utility

The robust study establishes three distinct propositions. First, the original robust mathematics and its executable realization must be separated: active kernels, p/rho loops, threshold defects, and solvers differed from a simplified reading of the paper. Second, mathematically positive-definite kernels can be unusable in finite-precision conic representations. The high-precision Blood matrices were SPD, yet their condition numbers exceeded $10^{23}$. Third, removing a numerical failure does not validate the robust hypothesis. The bounded-RBF formulation converted the model to an LP, completed every weighted robust Gate8 fit, and met feasibility, yet still reduced every dataset's mean balanced accuracy.

The robust negative result is therefore stronger than a software failure. The SOCP study showed that the direct RKHS robust counterpart was numerically impractical under the tested geometry. The bounded-RBF LP then showed that, after avoiding the SOCP and bounding the kernel above, the frozen robust term still redistributed capacity unfavorably. Future work should therefore change the robust risk geometry rather than continue solver retries.

### 10.4 Binary and multiclass behavior

The strongest deterministic gain occurred on binary Parkinson, although the RKHS-RBF-OVO SVM also improved three- and six-class tasks. The clearest class-sensitive result was binary Blood; Heart showed a multiclass balance/error tradeoff, and Iris was essentially neutral. Source OVA and RKHS OVO models are not directly interchangeable. In OVA, pooled negative classes share a score and, in the bounded-RBF robust formulation, one negative weight; in OVO, every pair has its own class balance and vote. This structural distinction may partly explain why weighting transfers differently across tasks and why source-style thresholds require separate analysis.

All of these comparisons are internal benchmark evaluations under the fixed local protocol. They do not constitute external validation and do not establish cross-institution generalization.

## 11. Limitations

Several limitations constrain interpretation.

First, executable reproduction is not identical to reproducing every published table. The robust MATLAB loops used unmatched unseeded splits, and the supplied deterministic and robust scripts did not always share preprocessing or active kernels. Literature values are therefore contextual references, not paired comparators.

Second, the RKHS-RBF-OVO SVM changes multiple architectural factors simultaneously. Its confirmed improvement establishes the performance of the frozen package, not the causal superiority of any single component. The confirmed class-sensitive architecture likewise changes both slack weights and the inner selection criterion relative to its unweighted reference. The later square-root/error Gate8 arm isolates the selection criterion within the square-root family, but it is a retrospective eight-split ablation on already examined development seeds and cannot by itself establish a population-level causal component effect.

Third, the Heart data use a local five-class protocol with high absolute error and limited comparability to binary heart-disease studies. The Breast Cancer recurrence cohort requires a documented categorical encoding and must not be conflated with Breast Cancer Diagnostic. Dermatology and Heart use complete-case subsets.

Fourth, not every dataset reached every comparison. The Blood and Mammographic source-profile comparisons stopped at Gate8; the recurrence and Dermatology class-sensitive comparisons stopped at Gate24. This is by design under prospective spending rules, but their estimates have greater uncertainty than the 96-seed confirmations.

Fifth, paired seeds are repeated holdouts rather than independent datasets. The independent cross-dataset evidence units are the seven datasets, not $96\times7$ experiments. The descriptive paired split-variation intervals use nominal 95% coverage to quantify conditional variation under the fixed local protocol. Because the holdouts overlap and are dependent, they are not formal independent-sample confidence intervals and do not quantify population uncertainty across institutions or acquisition settings. Wins/ties/losses and standardized effects are descriptive complements, while secondary or classwise results are exploratory rather than separate confirmatory claims. Resampling-based $p$-values are omitted from the main manuscript, and no confirmatory claim is based on a family of $p$-values.

Sixth, runtime comparisons are tied to the recorded Modal CPU environments, software versions, solver policies, and search spaces. They should be read as operational costs of the frozen pipelines, not as hardware-neutral benchmarks.

Seventh, all robust candidates were rejected. The bounded-RBF robust gate had only seven matched Blood pairs because two deterministic controls failed, and eight pairs elsewhere; its descriptive diagnostics are screening evidence. The subsequent mechanism analysis is observational, explanatory, and post hoc with respect to the failure, although it used only frozen artifacts and no new fitting.

Finally, the targeted literature update covers representative 2022--2026 work on imbalanced SVMs, cost-sensitive learning, robust classification, and distributionally robust learning. A journal submission should still perform a target-journal-specific systematic search and reporting check without changing the empirical claims reported here.

## 12. Conclusion

An executable-source audit, frozen architecture design, and prospective paired evaluation produced two supported deterministic conclusions and one controlled robust negative result. The RKHS-RBF-OVO SVM substantially reduced error on Parkinson and Breast Cancer recurrence and achieved smaller confirmed reductions on five other datasets, often with lower runtime than the Source-Profile q1 Kernel SVM. Those gains were not uniformly class sensitive: recurrence and Dermatology exposed minority or rare-class recall costs. The CS-RKHS-RBF-OVO SVM addressed this limitation selectively. It substantially improved Blood balanced accuracy, macro-F1, and minority recall without a confirmed total-error change, and improved Heart balance metrics at clear error and runtime costs. It did not improve all datasets. A later targeted Gate8 ablation showed that Blood and Heart balance gains persisted when square-root weighting used error rather than balanced selection, with smaller mean balance shifts and lower mean error than the square-root/balanced arm; its retrospective eight-split scope does not change the confirmed model recommendation.

No tested robust candidate is supported for promotion. Direct RKHS-L2 robustness was numerically impractical on the exposed Blood geometry; exact source-LP reformulations remained ill-scaled; and the bounded-RBF LP, although numerically stable for every weighted robust Gate8 fit, worsened the frozen balance-sensitive endpoints. Post hoc mechanism diagnostics found patterns consistent with global coefficient shrinkage, support reduction, slack substitution, and—in Parkinson and Mammographic Mass—threshold-objective mismatch. The robust study is therefore retained as reproducible negative evidence rather than presented as a performance contribution.

The recommendation is conditional: the RKHS-RBF-OVO SVM is the better-supported error-oriented option under the evaluated datasets and fixed protocol when aggregate predictive error and runtime are primary; the CS-RKHS-RBF-OVO SVM is the supported alternative when minority recovery or balanced performance is declared as the primary objective and its error/runtime tradeoff is acceptable. Neither model is universally superior. The cross-dataset conclusion rests on seven dataset-level evidence units, not $96\times7$ independent experiments. These results do not constitute external validation and do not establish cross-institution generalization.

## 13. Future Work

Future robust research should begin from a new mathematical hypothesis rather than further tolerance, solver, kernel, p, or rho searches on the rejected candidates. A promising direction would replace the global uncertainty-norm coupling with class-aware robust risk or explicit minority-margin protection. The formulation should align its intercept or threshold rule with the same class-sensitive loss optimized during training, rather than combining a weighted objective with an unweighted threshold criterion. It should also preserve a numerically stable LP or otherwise demonstrate solver-usable conditioning before consuming fresh predictive seeds.

More broadly, future validation should add external datasets or institutionally distinct cohorts, pre-register application-specific error costs, and use nested resampling or external test sets where available. Component-wise ablations could separate the contributions of RKHS-L2 regularization, RBF bandwidth policy, OVO decomposition, and class weighting. These studies should retain the prospective gate discipline used here and preserve negative results.

## Appendix A. Existing Development and Mechanism Diagnostics

**Table S1. Eight-seed Blood Transfusion development ablation used to separate model-selection and class-weighting effects, including the later targeted square-root/error row. Entries are percentages. These are Gate8 development results, not confirmation estimates.**

| Development variant | Error | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Unweighted / error selection | 22.3262 | 58.9995 | 59.6252 |
| Unweighted / balanced selection | 21.8583 | 61.9039 | 63.3316 |
| Unweighted / macro-F1 selection | 21.7246 | 61.6124 | 63.0131 |
| Inverse weighting / error selection | 27.2059 | 58.4199 | 57.6477 |
| Inverse weighting / balanced selection | 31.5508 | 69.1270 | 63.8522 |
| Inverse weighting / macro-F1 selection | 29.9465 | 67.4841 | 64.1394 |
| Square-root weighting / error selection (later targeted Gate8) | 21.5241 | 65.3909 | 66.8032 |
| Square-root weighting / balanced selection | 21.8583 | 66.5275 | 67.5396 |
| Learned-ratio weighting / balanced selection | 35.9626 | 68.3887 | 61.0594 |

The first, second, fourth, and fifth rows form the original four-cell unweighted/weighted by error/balanced ablation for the inverse-frequency family. The later square-root/error row completes the corresponding selection comparison for the promoted weighting family on the same dependent Gate8 splits. It was added retrospectively and was not eligible to revise the original prospective promotion decision.

**Table S2. Targeted square-root/error-selection Gate8 on seeds 9000-9007. Values are candidate-minus-reference percentage-point changes; brackets are nominal/descriptive 95% paired split-variation intervals. Because the eight repeated holdouts overlap, these are not formal independent-sample confidence intervals.**

| Dataset | $\Delta$ error vs V7 | $\Delta$ balanced accuracy vs V7 | $\Delta$ macro-F1 vs V7 | $\Delta$ error vs square-root/balanced | $\Delta$ balanced accuracy vs square-root/balanced |
|---|---:|---:|---:|---:|---:|
| Blood Transfusion | -0.802 [-3.474, +1.870] | +6.391 [+1.797, +10.986] | +7.178 [+0.979, +13.377] | -0.334 [-0.667, -0.002] | -1.137 [-3.210, +0.937] |
| Heart Disease | -0.667 [-2.974, +1.641] | +2.255 [+0.349, +4.161] | +2.228 [+0.100, +4.357] | -2.333 [-5.357, +0.690] | -1.032 [-5.322, +3.259] |
| Mammographic Mass | -0.300 [-1.373, +0.772] | +0.327 [-0.740, +1.393] | +0.321 [-0.762, +1.404] | 0.000 [-0.215, +0.215] | +0.003 [-0.199, +0.206] |
| Parkinson | 0.000 [-0.912, +0.912] | +0.352 [-2.198, +2.902] | +0.180 [-1.594, +1.955] | 0.000 [-1.824, +1.824] | -1.408 [-4.176, +1.361] |
| Wine | -0.278 [-1.828, +1.273] | +0.463 [-1.341, +2.267] | +0.318 [-1.327, +1.963] | -0.278 [-0.935, +0.379] | +0.231 [-0.316, +0.779] |

The targeted arm used the original mean-one square-root inverse-frequency OVO pair weights, three-fold training-only validation folds, eight alpha rules, four C values, outer splits, metrics, and tie order. Forty new outer models were fitted on Modal. Their inner square-root banks were reused from the immutable original Gate8 checkpoints, so the changed selection rule was evaluated without repeating the already completed 3,840 inner candidate-fold fits.

**Table S3. Frozen-checkpoint counterfactual decomposition of score-function and threshold effects in the rejected Bounded-RBF robust Gate8. Each cell is safety recall / balanced accuracy. These are coordinate diagnostics, not newly trained models.**

| Dataset | Deterministic reference | Robust model | Robust function + reference threshold | Reference function + robust threshold |
|---|---:|---:|---:|---:|
| Blood Transfusion | .3386 / .6337 | .3261 / .6199 | .2462 / .5945 | .3994 / .6475 |
| Parkinson | .6146 / .8022 | .5104 / .7518 | .6146 / .7988 | .3333 / .6667 |
| Mammographic Mass | .8020 / .8122 | .7463 / .8031 | .8181 / .7788 | .7364 / .8039 |
| Iris | .8253 / .9322 | .8253 / .9290 | .8253 / .9354 | .8149 / .9255 |

The decomposition identifies threshold movement as a major loss amplifier on Parkinson and Mammographic Mass, a compensatory change on Blood Transfusion, and a negligible factor on Iris. It therefore does not support a universal threshold-only explanation.

## References

1. Cortes, C., and Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273-297.
2. Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer-Verlag.
3. Boser, B. E., Guyon, I. M., and Vapnik, V. N. (1992). A training algorithm for optimal margin classifiers. In *Proceedings of the Fifth Annual Workshop on Computational Learning Theory*, 144-152.
4. Schölkopf, B., and Smola, A. J. (2001). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.
5. Maggioni, F., and Spinelli, A. (2025). A novel robust optimization model for nonlinear Support Vector Machine. *European Journal of Operational Research*, 322, 237-253. https://doi.org/10.1016/j.ejor.2024.12.014
6. Liu, Y., and Potra, F. A. (2009). Pattern separation and prediction via linear and semidefinite programming. *Studies in Informatics and Control*, 18(1), 71-82.
7. Schölkopf, B., Smola, A. J., Williamson, R. C., and Bartlett, P. L. (2000). New support vector algorithms. *Neural Computation*, 12(5), 1207-1245.
8. Weston, J., and Watkins, C. (1998). Multi-class support vector machines. Technical Report CSD-TR-98-04, Royal Holloway, University of London.
9. Ben-Tal, A., El Ghaoui, L., and Nemirovski, A. (2009). *Robust Optimization*. Princeton University Press.
10. Bertsimas, D., Brown, D. B., and Caramanis, C. (2011). Theory and applications of robust optimization. *SIAM Review*, 53, 464-501.
11. Xu, H., Caramanis, C., and Mannor, S. (2009). Robustness and regularization of support vector machines. *Journal of Machine Learning Research*, 10, 1485-1510.
12. Trafalis, T. B., and Gilbert, R. C. (2006). Robust classification and regression using support vector machines. *European Journal of Operational Research*, 173, 893-909.
13. Bi, J., and Zhang, T. (2005). Support vector classification with input data uncertainty. In *Advances in Neural Information Processing Systems*, 161-168.
14. Wang, Z., and Pardalos, P. M. (2014). A survey of support vector machines with uncertainties. *Annals of Data Science*, 1, 293-309.
15. Bertsimas, D., Dunn, J., Pawlowski, C., and Zhuo, Y. D. (2019). Robust classification. *INFORMS Journal on Optimization*, 1, 2-34.
16. Faccini, D., Maggioni, F., and Potra, F. A. (2022). Robust and distributionally robust optimization models for linear support vector machine. *Computers & Operations Research*, 147, 105930.
17. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
18. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
19. Kelly, M., Longjohn, R., and Nottingham, K. (2023). The UCI Machine Learning Repository. https://archive.ics.uci.edu
20. Rezvani, S., Pourpanah, F., Lim, C. P., and Wu, Q. M. J. (2024). Methods for class-imbalanced learning with support vector machines: a review and an empirical evaluation. *Soft Computing*, 28(20), 11873-11894. https://doi.org/10.1007/s00500-024-09931-5
21. Guido, R., Groccia, M. C., and Conforti, D. (2022). A hyper-parameter tuning approach for cost-sensitive support vector machine classifiers. *Soft Computing*, 27(18), 12863-12881. https://doi.org/10.1007/s00500-022-06768-8
22. Fu, S., Yu, X., and Tian, Y. (2022). Cost sensitive ν-support vector machine with LINEX loss. *Information Processing & Management*, 59(2), 102809. https://doi.org/10.1016/j.ipm.2021.102809
23. Asimit, A. V., Kyriakou, I., Santoni, S., Scognamiglio, S., and Zhu, R. (2022). Robust classification via support vector machines. *Risks*, 10(8), 154. https://doi.org/10.3390/risks10080154
24. Yu, Y., Lin, T., Mazumdar, E. V., and Jordan, M. I. (2022). Fast distributionally robust learning with variance-reduced min-max optimization. In *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics*, PMLR 151, 1219-1250. https://proceedings.mlr.press/v151/yu22a.html
25. Zhu, D., Ying, Y., and Yang, T. (2023). Label distributionally robust losses for multi-class classification: consistency, robustness and adaptivity. In *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 43289-43325. https://proceedings.mlr.press/v202/zhu23o.html
26. Purwadi, J., Fithriasari, K., and Kuswanto, H. (2026). An adaptive robust support vector machine with sequential minimal optimization for imbalanced data classification. *IEEE Access*, 14, 17031-17038. https://doi.org/10.1109/ACCESS.2026.3655329

## Data and Code Availability

The project repository contains the frozen implementations, source snapshots, configuration and dataset hashes, split registries, seed-level predictions, model-selection outputs, solver diagnostics, failure records, paired statistical tables, and the evidence map accompanying this manuscript. The targeted square-root/error-selection Gate8 is preserved at `results/v8/gate8/v8-sqrt-error-gate8-20261005-v1`, including its completed Modal manifest, configuration, 40 checkpoints, paired comparisons, selected hyperparameters, and source snapshot. Public release details and a permanent archive identifier should be added before submission. No patient-identifiable information is present in the benchmark files used here.

## Declarations

- **Funding:** To be completed by the authors.
- **Conflicts of interest:** To be completed by the authors.
- **Author contributions:** To be completed by the authors.
- **Ethics approval:** The study uses pre-existing benchmark datasets; journal-specific wording and dataset licenses should be verified before submission.
- **Generative-AI disclosure:** To be completed in accordance with the target journal's policy.
