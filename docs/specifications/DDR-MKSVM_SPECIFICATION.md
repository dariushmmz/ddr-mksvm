# Deep Distributionally-Robust Multiple-Kernel SVM (DDR-MKSVM)
### A combined Deep-Kernel-Learning + Wasserstein-DRO extension of Maggioni & Spinelli (2025)
### Implementation specification and development instructions for an AI coding agent (e.g. Claude)

---

## 0. How to use this document

This document is written **as a specification handed to another AI model** (e.g. an instance of Claude operating an agentic coding tool) that will implement the system from scratch. It is organized so that:

- **Part I (Sections 1–5)** is the mathematical specification: definitions, model, theorems, and proofs. Treat this as the source of truth for what the code must compute. Every symbol used in Part II is defined here.
- **Part II (Sections 6–10)** is the engineering specification: repository layout, modules, training loop, tests, and milestones. Implement in the order given in Section 10.

If any theorem in Part I cannot be reconciled with an implementation choice in Part II, **Part I wins** — stop and flag the discrepancy rather than silently deviating from the math.

Base paper referenced throughout: Maggioni, F., & Spinelli, A. (2025). *Robust support vector machines with nonlinear separators*. European Journal of Operational Research, 322, 237–253. Its results (Table 3, Theorem 1, Corollary 1, Propositions 1–2) are the object being extended; every new result below is stated as an extension of, or reduction to, a specific numbered result in that paper.

---

# PART I — Mathematical Specification

## 1. Motivation and design principle

The base paper's robust model has three independent, separable design choices, each currently fixed by hand:

1. **The kernel** `k(·,·)` is chosen from a small fixed menu (linear, polynomial, Gaussian RBF) via grid search per dataset (Table 3).
2. **The uncertainty geometry** is a per-point bounded-ℓ_p-norm ball with a manually swept radius η⁽ⁱ⁾ (Section 5.1 of the paper), and the feature-space radius δ⁽ⁱ⁾ is available in closed form *only* for polynomial and Gaussian RBF kernels (Propositions 1–2).
3. **The feature map** φ(·) is whatever the fixed kernel implies — there is no learned representation.

The combined model below replaces (3) with a learned deep feature extractor, replaces (1) with a learned convex combination of kernels applied on top of that extractor ("deep multiple kernel learning"), and replaces (2) with a single global Wasserstein ambiguity set whose radius has one data-driven meaning across the whole training set, rather than m manually-swept per-point radii. The key mathematical fact that makes this combination *tractable* — not just heuristic — is that all three extensions can be expressed through **Lipschitz constants of the composite feature map**, which is a strict generalization of how the base paper already expresses its robust bound (its δ⁽ⁱ⁾·√K_jj term in Theorem 1 is itself a Lipschitz-type quantity). Section 4.8 makes this reduction precise.

## 2. Notation and preliminaries

Training data: `{(x⁽ⁱ⁾, y⁽ⁱ⁾)}_{i=1..m}`, `x⁽ⁱ⁾ ∈ ℝⁿ`, `y⁽ⁱ⁾ ∈ {−1,+1}`. `P̂ = (1/m) Σᵢ δ_{(x⁽ⁱ⁾,y⁽ⁱ⁾)}` is the empirical distribution. `‖·‖_p` denotes the ℓ_p norm on ℝⁿ, `p ∈ [1,∞]`, with dual exponent `q` s.t. `1/p + 1/q = 1`.

**Definition 1 (Feature map / RKHS).** A function `k: ℝⁿ × ℝⁿ → ℝ` is a *kernel* if it is symmetric and positive semidefinite (PSD): for all finite `{z_1,…,z_r} ⊂ ℝⁿ` and `c ∈ ℝʳ`, `Σ_{i,j} c_i c_j k(z_i,z_j) ≥ 0`. By the Moore–Aronszajn theorem, every such `k` induces a unique reproducing kernel Hilbert space `(ℍ, ⟨·,·⟩_ℍ)` and a feature map `φ: ℝⁿ → ℍ` with `k(z,z') = ⟨φ(z), φ(z')⟩_ℍ`.

**Definition 2 (Lipschitz constant).** `g: (X,‖·‖_X) → (Y,‖·‖_Y)` is *L-Lipschitz* if `‖g(z) − g(z')‖_Y ≤ L‖z − z'‖_X` for all `z,z' ∈ X`. `Lip(g)` denotes the smallest such `L` (the *exact* Lipschitz constant); any valid `L ≥ Lip(g)` is called a *Lipschitz bound*.

**Definition 3 (Deep kernel).** Given a feed-forward network `f_θ: ℝⁿ → ℝ^d` with parameters `θ`, and a *base kernel* `k_base: ℝ^d × ℝ^d → ℝ` with feature map `φ_base`, the induced *deep kernel* is
```
k_θ(x,x') := k_base(f_θ(x), f_θ(x'))
```
with composite feature map `φ_θ := φ_base ∘ f_θ : ℝⁿ → ℍ_base`.

**Definition 4 (Multiple kernel combination).** Given base kernels `k_1,…,k_L` (each possibly itself a deep kernel per Definition 3, with its own `θ_l`) and weights `η` in the simplex `Δ_L = {η ∈ ℝ^L_+ : Σ_l η_l = 1}`, the combined kernel is
```
K_{θ,η}(x,x') := Σ_{l=1}^{L} η_l · k_{θ_l,l}(x,x')
```

**Definition 5 (Wasserstein distance and ambiguity set).** For probability measures `Q, P` on `ℝⁿ × {−1,+1}` and transportation cost `c((x,y),(x',y')) = ‖x−x'‖_p` if `y=y'`, and `+∞` otherwise (i.e. mass cannot move between classes), the type-1 Wasserstein distance is
```
W(Q,P) := inf_{Π ∈ Π(Q,P)} E_Π[ c((x,y),(x',y')) ]
```
where `Π(Q,P)` is the set of couplings of `Q` and `P`. The **Wasserstein ambiguity set** of radius `ε ≥ 0` around `P̂` is `𝔅_ε(P̂) := { Q : W(Q,P̂) ≤ ε }`.

**Definition 6 (Hinge loss on a feature map).** For classifier weight `w ∈ ℍ` (Hilbert space of the combined kernel) and bias `γ`, the hinge loss at `(x,y)` is
```
ℓ_{w,γ,θ,η}(x,y) := max( 0, 1 − y·(⟨w, φ_{θ,η}(x)⟩_ℍ − γ) )
```

**Definition 7 (DRO risk).** The *distributionally robust risk* of a classifier at ambiguity radius `ε` is
```
R_ε(w,γ,θ,η) := sup_{Q ∈ 𝔅_ε(P̂)} E_Q[ ℓ_{w,γ,θ,η}(x,y) ]
```

## 3. Model definition

**Definition 8 (DDR-MKSVM).** The combined model solves
```
min_{w,γ,θ,η∈Δ_L}   R_ε(w,γ,θ,η) + ν·‖w‖²_ℍ                       (M)
```
i.e., the base paper's soft-margin objective (its model (1)/(5), specialized to `q=2` regularization) with the empirical average hinge loss replaced by its *worst case over the Wasserstein ball*, and with the kernel/feature map itself now a learnable object via `(θ, η)`.

This is the target object; Sections 4.1–4.8 show it reduces to a **finite-dimensional, alternately-convex** problem that generalizes the base paper's Theorem 1 and Corollary 1.

## 4. Lemmas and theorems

### 4.1 Lemma 1 (PSD closure of conic combinations)

*Statement.* If `k_1,…,k_L` are kernels (Definition 1) and `η ∈ Δ_L`, then `K_{θ,η} = Σ_l η_l k_l` is a kernel.

*Proof.* For any finite set `{z_i}` and `c ∈ ℝʳ`: `Σ_{i,j} c_i c_j K_{θ,η}(z_i,z_j) = Σ_l η_l Σ_{i,j} c_i c_j k_l(z_i,z_j) ≥ 0`, since each inner sum is `≥0` by `k_l` being PSD and `η_l ≥ 0`. Symmetry is inherited termwise. ∎

*Consequence.* `K_{θ,η}` induces a valid RKHS `ℍ` for every `θ` and every `η` in the simplex, so `(M)` is well-posed for all feasible `(θ,η)` — this is what licenses treating `η` as a continuously optimizable variable rather than a discrete kernel-selection choice (in contrast to the base paper's per-dataset grid search over a discrete kernel menu).

### 4.2 Lemma 2 (Network Lipschitz bound via spectral norms)

*Statement.* Let `f_θ(x) = W_T σ(W_{T-1} σ(⋯ σ(W_1 x)⋯))` be a `T`-layer feed-forward network with 1-Lipschitz elementwise activation `σ` (true for ReLU, tanh, sigmoid). Then `f_θ` is `L_f`-Lipschitz w.r.t. `(‖·‖_2, ‖·‖_2)` with
```
L_f ≤ Π_{t=1}^{T} ‖W_t‖_2      (‖·‖_2 here = spectral norm of the matrix)
```

*Proof.* By induction on the number of layers. Base case `T=0` (identity map): `L=1`, trivial. Inductive step: suppose `h(x) := σ(W_{T-1}σ(⋯))` is `L_{T-1}`-Lipschitz with `L_{T-1} ≤ Π_{t<T}‖W_t‖_2`. Then for the next affine layer, `‖W_T h(x) − W_T h(x')‖_2 ≤ ‖W_T‖_2 ‖h(x)−h(x')‖_2 ≤ ‖W_T‖_2 L_{T-1}‖x−x'‖_2` (operator norm definition, then the inductive hypothesis). Composing with the 1-Lipschitz activation `σ` does not increase this bound, since `‖σ(u)−σ(u')‖_2 ≤ ‖u−u'‖_2` elementwise-and-in-aggregate for the activations listed. Hence `L_T ≤ ‖W_T‖_2 · L_{T-1} ≤ Π_{t≤T}‖W_t‖_2`. ∎

*Remark.* This bound is generally loose (the true Lipschitz constant of a composed network can be far smaller than the product of per-layer spectral norms). Section 6.4 requires the implementation to *also* provide an empirical, tighter estimate (via power iteration on the input-output Jacobian) and to treat the analytic bound as a certified upper bound used only where a certified guarantee is required.

### 4.3 Lemma 3 (Lipschitz constant of the Gaussian RBF feature map — exact, global)

*Statement.* Let `k_base(u,u') = exp(−α‖u−u'‖²_2)`, `α>0`, with feature map `φ_base`. Then, globally on `ℝ^d`,
```
‖φ_base(u) − φ_base(u')‖_ℍ ≤ √(2α) · ‖u−u'‖_2      for all u,u' ∈ ℝ^d
```
i.e. `φ_base` is `√(2α)`-Lipschitz w.r.t. `(‖·‖_2, ‖·‖_ℍ)`, with equality attained in the limit `‖u−u'‖_2 → 0`.

*Proof.* By the reproducing property, `‖φ_base(u)−φ_base(u')‖²_ℍ = k(u,u) + k(u',u') − 2k(u,u') = 2 − 2exp(−α d²)`, where `d := ‖u−u'‖_2`. Define `g(d) := 2αd² − (2 − 2e^{−αd²})` for `d ≥ 0`. We show `g(d) ≥ 0` for all `d ≥ 0`, which gives `2αd² ≥ 2−2e^{−αd²} = ‖φ_base(u)−φ_base(u')‖²_ℍ`, i.e. the claim after taking square roots. We have `g(0)=0` and `g'(d) = 4αd − 4αd·e^{−αd²} = 4αd(1 − e^{−αd²})`. For `d ≥ 0`, `e^{−αd²} ≤ 1`, so `g'(d) ≥ 0`; hence `g` is nondecreasing on `[0,∞)`, and since `g(0)=0`, `g(d) ≥ 0` for all `d ≥ 0`. ∎

*Remark.* This is a sharpening/rederivation of Proposition 2 of the base paper (which states the analogous bound for the specific case relevant to that paper's ℓ_p-ball uncertainty sets); here it is proved directly and used as a *global exact Lipschitz constant*, which is the form needed for the Wasserstein-DRO duality theorem in Section 4.6.

### 4.4 Lemma 4 (Lipschitz constants of linear and polynomial base kernels)

*Statement.* (a) For the linear kernel `k_base(u,u')=u^⊤u'`, `φ_base(u)=u`, so `Lip(φ_base)=1` exactly (w.r.t. `‖·‖_2`). (b) For the inhomogeneous polynomial kernel `k_base(u,u')=(u^⊤u'+c)^d` restricted to a bounded domain `‖u‖_2 ≤ B`, `φ_base` is Lipschitz with constant bounded by `d(B²+c)^{(d-1)/2} · B'` for an explicit constant depending on `B,c,d` (derivable by the same telescoping/mean-value argument the base paper uses in its Appendix A proof of Proposition 1, applied to the difference `φ(u)−φ(u')` instead of the norm bound `‖φ(x)‖` alone).

*Proof sketch.* (a) is immediate. (b) follows the same finite-difference telescoping technique as the base paper's Appendix A: write `k(u,u)+k(u',u')-2k(u,u') = (u^⊤u+c)^d + (u'^⊤u'+c)^d - 2(u^⊤u'+c)^d`, bound this via the mean value theorem applied to `t ↦ t^d` on the bounded interval `[c, B²+c]`, and take square roots. Full derivation deferred to the codebase's `docs/lemma4_derivation.md` (Section 8) rather than repeated here, since it is a direct adaptation of the base paper's own Appendix A proof of Proposition 1 rather than a new argument.

*Remark.* Unlike Lemma 3, this Lipschitz constant is **not global** — it depends on a domain bound `B` on the inputs, exactly mirroring how the base paper's Proposition 1 bound requires bounded input data (implicit in its use of a fixed dataset). The implementation must therefore compute `B = max_i ‖x⁽ⁱ⁾‖_2` (or the appropriate transformed-feature bound) from the training data before using a polynomial base kernel in the DRO-Lipschitz pipeline (Section 6.4).

### 4.5 Lemma 5 (Composite Lipschitz bound for the deep multiple kernel)

*Statement.* Let `f_θ` be `L_f`-Lipschitz (Lemma 2) and let each base kernel's feature map `φ_{base,l}` be `L_{φ,l}`-Lipschitz (Lemmas 3–4). Then the composite feature map of the combined deep kernel `K_{θ,η}` (Definitions 3–4) satisfies, for `η ∈ Δ_L`,
```
‖φ_{θ,η}(x) − φ_{θ,η}(x')‖_ℍ  ≤  L_f · ( Σ_l η_l · L_{φ,l}² )^{1/2} · ‖x−x'‖_2   =:  L_{θ,η} · ‖x−x'‖_2
```

*Proof.* Write `ℍ = ⊕_l ℍ_l` (direct sum RKHS of the combined kernel, standard fact for conic kernel combinations, dual to Lemma 1). Then `φ_{θ,η}(x) = (√η_1 φ_{base,1}(f_θ(x)), …, √η_L φ_{base,L}(f_θ(x)))` componentwise (this representation reproduces `K_{θ,η}` by construction: `⟨φ_{θ,η}(x),φ_{θ,η}(x')⟩ = Σ_l η_l ⟨φ_{base,l}(f_θ(x)),φ_{base,l}(f_θ(x'))⟩ = Σ_l η_l k_{base,l}(f_θ(x),f_θ(x')) = K_{θ,η}(x,x')`, as required). Hence
```
‖φ_{θ,η}(x)−φ_{θ,η}(x')‖²_ℍ = Σ_l η_l ‖φ_{base,l}(f_θ(x))−φ_{base,l}(f_θ(x'))‖²_{ℍ_l}
                              ≤ Σ_l η_l L_{φ,l}² ‖f_θ(x)−f_θ(x')‖²_2     (Lemmas 3–4, applied to inputs f_θ(x), f_θ(x'))
                              ≤ (Σ_l η_l L_{φ,l}²) · L_f² · ‖x−x'‖²_2     (Lemma 2, applied once more)
```
Taking square roots gives the claim. ∎

*Interpretation.* `L_{θ,η}` is a single scalar summarizing how much a bounded input perturbation can move the combined, learned feature representation — this is the *direct generalization of the base paper's δ⁽ⁱ⁾(η⁽ⁱ⁾) quantity* (Section 5.2, Propositions 1–2) to a learned, multi-kernel feature map. Where the base paper computes `δ⁽ⁱ⁾` in closed form per kernel type and per point, `L_{θ,η}` is one number for the whole model, computed by composing Lemmas 2–4.

### 4.6 Theorem 1 (Wasserstein-DRO reformulation of the hinge risk)

*Statement.* Let `ℓ_{w,γ,θ,η}(·,y)` (Definition 6) be `L_ℓ`-Lipschitz in `x` w.r.t. `‖·‖_2` (this is established constructively below). Then, for the ambiguity set of Definition 5 with the same norm,
```
R_ε(w,γ,θ,η) = ε · L_ℓ  +  (1/m) Σ_{i=1}^m ℓ_{w,γ,θ,η}(x⁽ⁱ⁾, y⁽ⁱ⁾)                        (T1)
```
and moreover `L_ℓ ≤ ‖w‖_ℍ · L_{θ,η}` where `L_{θ,η}` is as in Lemma 5.

*Proof.*
**Step 1 (Lipschitz constant of the hinge loss composed with the feature map).** Fix `y ∈ {−1,+1}`. The map `u ↦ max(0, 1−y(⟨w,u⟩_ℍ−γ))` is `‖w‖_ℍ`-Lipschitz in `u ∈ ℍ`: `max(0,1-y(·))` is 1-Lipschitz as a scalar function of its argument, and `u ↦ ⟨w,u⟩_ℍ` is `‖w‖_ℍ`-Lipschitz by Cauchy–Schwarz (`|⟨w,u⟩−⟨w,u'⟩| = |⟨w,u−u'⟩| ≤ ‖w‖_ℍ‖u−u'‖_ℍ`). Composing with `φ_{θ,η}` (Lipschitz constant `L_{θ,η}` by Lemma 5) and using that Lipschitz constants multiply under composition gives `x ↦ ℓ_{w,γ,θ,η}(x,y)` is `L_ℓ := ‖w‖_ℍ · L_{θ,η}`-Lipschitz in `x`, for both values of `y`. This proves the stated bound on `L_ℓ`.

**Step 2 (Strong duality).** By the strong-duality theorem for Wasserstein DRO with a transportation cost that is a norm and an upper-semicontinuous loss (Mohajerin Esfahani & Kuhn, 2018, Thm. 4.2; equivalently Gao & Kleywegt, 2016; Blanchet & Murthy, 2019 — cited rather than re-derived here, since it is a general measure-theoretic optimal-transport duality result independent of the SVM setting), for `P̂` a finitely-supported empirical distribution:
```
R_ε(w,γ,θ,η) = inf_{λ≥0} { λε + (1/m) Σ_i sup_{x} [ ℓ_{w,γ,θ,η}(x,y⁽ⁱ⁾) − λ‖x−x⁽ⁱ⁾‖_2 ] }
```
(the label-transport cost is `+∞` for `y≠y'`, so the sup over `Q` only ever moves the `x`-marginal of each atom, independently per point — this is why the outer sum is per-`i`, not a joint transport plan).

**Step 3 (Elementary evaluation of the inner supremum).** Fix `i` and `λ ≥ 0`. Since `ℓ_{w,γ,θ,η}(·,y⁽ⁱ⁾)` is `L_ℓ`-Lipschitz (Step 1): for any `x`, `ℓ(x,y⁽ⁱ⁾) − λ‖x−x⁽ⁱ⁾‖_2 ≤ ℓ(x⁽ⁱ⁾,y⁽ⁱ⁾) + L_ℓ‖x−x⁽ⁱ⁾‖_2 − λ‖x−x⁽ⁱ⁾‖_2 = ℓ(x⁽ⁱ⁾,y⁽ⁱ⁾) + (L_ℓ−λ)‖x−x⁽ⁱ⁾‖_2`.
- If `λ ≥ L_ℓ`: the right-hand side is maximized at `x=x⁽ⁱ⁾` (any deviation only decreases it, since `(L_ℓ−λ)≤0`), giving `sup_x[·] = ℓ(x⁽ⁱ⁾,y⁽ⁱ⁾)`.
- If `λ < L_ℓ`: since `L_ℓ` is (an upper bound on, tight in the worst case over `w`) the exact local rate of increase of `ℓ` in some direction `v` at `x⁽ⁱ⁾` (by definition of Lipschitz constant as the *smallest* such bound being approached), taking `x = x⁽ⁱ⁾ + t·v` for `t→∞` along a direction where the hinge is still active (`1−y(⟨w,φ_{θ,η}(x)⟩−γ) > 0`, which holds for small enough `t` and can be extended using the unboundedness of the linear term inside the hinge) drives `ℓ(x,y⁽ⁱ⁾) − λ‖x−x⁽ⁱ⁾‖_2 → +∞`. Hence `sup_x[·] = +∞`.

**Step 4 (Optimizing over λ).** The outer `inf_{λ≥0}` therefore only considers `λ ≥ L_ℓ` (else the objective is `+∞`), and for `λ ≥ L_ℓ` the objective `λε + (1/m)Σ_i ℓ(x⁽ⁱ⁾,y⁽ⁱ⁾)` is nondecreasing in `λ` (since `ε≥0`), so the infimum is attained at `λ = L_ℓ`, giving exactly (T1). ∎

*Sanity check against the base paper.* Set `L_ℓ` at its bound `‖w‖_ℍ · L_{θ,η}`, expand `w` in the usual dual/kernel representation `w = Σ_j u_j y⁽ʲ⁾φ(x⁽ʲ⁾)` so that `‖w‖²_ℍ = Σ_{i,j} u_i u_j y⁽ⁱ⁾y⁽ʲ⁾K_{ij}`, and note that the base paper's Theorem 1 penalty term is `δ⁽ⁱ⁾ Σ_j √K_jj |u_j|` — a *per-point* worst-case penalty using the exact closed-form kernel-specific radius `δ⁽ⁱ⁾`. (T1)'s penalty `ε·‖w‖_ℍ·L_{θ,η}` is the *same type of quantity* (a Lipschitz/dual-norm penalty on the classifier weight), but global (one `ε`, one `L_{θ,η}` for the whole model) rather than per-point and per-kernel-closed-form. This confirms Theorem 1 above is a genuine generalization, in the sense of Section 4.8, rather than an unrelated construction.

### 4.7 Corollary 1 (Tractable training objective)

*Statement.* Combining (T1) with model `(M)`, DDR-MKSVM training is equivalent to
```
min_{w,γ,θ,η∈Δ_L}    (1/m) Σ_i max(0, 1 − y⁽ⁱ⁾(⟨w,φ_{θ,η}(x⁽ⁱ⁾)⟩_ℍ − γ))  +  ε·‖w‖_ℍ·L_{θ,η}(θ)  +  ν‖w‖²_ℍ     (T-obj)
```
which, **for fixed `(θ,η)`** (hence fixed `L_{θ,η}` and fixed kernel matrix `K_{θ,η}`), is a convex problem in `(w,γ)` — an SOCP, structurally identical in form to the base paper's Corollary 1 case `q=2`, with the extra linear-in-`‖w‖_ℍ` robustness term playing the same role as the base paper's `δ⁽ⁱ⁾Σ√K_jj|u_j|` term. For fixed `(w,γ)`, the objective is differentiable in `θ` (through `φ_{θ,η}` and, where Lemma 2's bound is used for `L_{θ,η}`, through the spectral norms `‖W_t‖_2`, which admit subgradients) and differentiable in `η` on the interior of the simplex.

*Proof.* Direct substitution of (T1) into `(M)` gives (T-obj). Convexity in `(w,γ)` for fixed `(θ,η)`: the empirical hinge term is a pointwise max of affine functions of `(w,γ)` (via the fixed, PSD kernel-induced inner product), hence convex; `‖w‖_ℍ` is a norm, hence convex, and `ε,L_{θ,η}≥0` so scaling by a nonnegative constant preserves convexity; `ν‖w‖²_ℍ` is convex. A sum of convex functions is convex. The SOCP form follows exactly the base paper's Appendix A derivation for its `q=2` case, with `δ⁽ⁱ⁾` replaced by the constant `ε·L_{θ,η}` (which does not depend on `i`, so it can be pulled out of the per-point structure — in fact this makes the resulting cone program *simpler* than the base paper's, which needs a per-point `δ⁽ⁱ⁾√K_jj` term inside the LP/SOCP constraints). ∎

*Consequence for the implementation.* The overall problem `(T-obj)` is **biconvex-by-block**: convex in `(w,γ)` for fixed `(θ,η)`, and (non-convex in general but) differentiable in `(θ,η)` for fixed `(w,γ)`. This licenses an **alternating minimization** scheme (Section 6.5): solve the convex SOCP for `(w,γ)` via a standard convex solver (cvxpy), then take gradient steps on `(θ,η)` via backpropagation, and repeat.

### 4.8 Proposition (The base paper's robust model is the `L=1`, single-kernel, no-network special case)

*Statement.* If `f_θ = Identity` (no network, `θ` empty), `L=1` (a single fixed base kernel, no MKL), and the ambiguity set is replaced by `m` independent per-point ℓ_p-balls of radius `η⁽ⁱ⁾` in the input space (Definition in Section 5.1 of the base paper) rather than a single joint Wasserstein ball, then the robust risk (T1)-analogue reduces exactly to the base paper's Theorem 1 objective, with `δ⁽ⁱ⁾ = δ⁽ⁱ⁾(η⁽ⁱ⁾)` given by the base paper's Propositions 1–2 in place of the global `ε·L_{θ,η}`.

*Proof.* With per-point balls instead of a joint distributional ball, the worst-case risk decomposes as a sum of `m` independent per-point worst cases, `Σ_i sup_{x∈𝔅_p^{η⁽ⁱ⁾}(x⁽ⁱ⁾)} ℓ(x,y⁽ⁱ⁾)` (no `1/m` averaging duality mass to distribute, since each point's ball is fixed rather than shared transport budget). Each per-point supremum, by the Cauchy–Schwarz argument used identically in the base paper's proof of its Theorem 1 (bounding `|⟨ζ⁽ⁱ⁾,φ(x⁽ʲ⁾)⟩| ≤ δ⁽ⁱ⁾√K_jj`), evaluates to exactly the base paper's `δ⁽ⁱ⁾Σ_j√K_jj|u_j|` term. Setting `L=1` and `f_θ=Identity` in Lemma 5 gives `L_{θ,η} = L_{φ,1}`, the base kernel's own Lipschitz constant (Lemmas 3–4), which is precisely the closed-form quantity the base paper computes as `δ⁽ⁱ⁾/η⁽ⁱ⁾` in Propositions 1–2. ∎

*Why this matters.* This proposition is the formal justification for calling DDR-MKSVM a **combination**, rather than a replacement, of the base paper's method: setting the new degrees of freedom to their trivial values recovers the original model exactly. It also gives a mandatory regression test (Section 6.7, Test T-1): DDR-MKSVM with network disabled, `L=1`, `η⁽ⁱ⁾` all equal, and `ε` set from a single shared `δ` must reproduce the base paper's numbers.

---

# PART II — Implementation instructions

## 6. Architecture

### 6.1 Repository layout

```
ddr-mksvm/
├── configs/
│   └── datasets.yaml            # reuse the DATASET_CONFIG registry described in the
│                                 # earlier reproduction report (transform, base kernel(s), p)
├── src/
│   ├── kernels/
│   │   ├── base_kernels.py      # linear, polynomial, gaussian_rbf: k(u,u'), and
│   │   │                        # analytic Lip bound per Lemmas 3-4
│   │   ├── deep_kernel.py       # f_theta network + composition with a base kernel (Def. 3)
│   │   └── mkl_combination.py   # eta-weighted sum of kernels (Def. 4), simplex projection
│   ├── lipschitz/
│   │   ├── analytic_bounds.py   # Lemma 2 (spectral-norm product), Lemma 5 (composite bound)
│   │   └── empirical_estimator.py  # power-iteration / finite-difference estimate of Lip(f_theta)
│   ├── dro/
│   │   └── wasserstein_risk.py  # implements (T-obj): builds L_ell from L_{theta,eta} and ||w||
│   ├── optim/
│   │   ├── convex_subproblem.py # cvxpy SOCP solve for (w, gamma) given fixed (theta, eta) — Cor. 1
│   │   └── alternating_trainer.py  # outer loop: convex solve <-> backprop on (theta, eta)
│   ├── data/
│   │   └── dataset_registry.py  # load + transform, matches original paper's per-dataset config
│   ├── eval/
│   │   └── holdout_protocol.py  # 96-run 75/25 stratified holdout, matches Table 3 protocol
│   └── legacy_reduction/
│       └── base_paper_mode.py   # forces network=Identity, L=1, per-point balls — Prop. in 4.8
├── tests/
│   ├── test_lemma1_psd.py
│   ├── test_lemma2_spectral_bound.py
│   ├── test_lemma3_rbf_lipschitz.py
│   ├── test_theorem1_duality.py
│   └── test_reduction_to_base_paper.py
├── experiments/
│   └── run_ablation.py          # DNN-off / MKL-off / DRO-off ablations, see Section 6.6
└── README.md
```

### 6.2 Environment and dependencies

- Python ≥3.10; `numpy`, `scipy`, `torch` (for `f_θ` and autodiff on `θ,η`), `cvxpy` with solver fallback chain `HIGHS → CLARABEL → SCS` (reuse the exact fallback logic already validated in the reproduction report, since MOSEK is not assumed available), `scikit-learn` (data preprocessing utilities only, not the SVM), `pytest`.
- Set `OMP_NUM_THREADS=1` inside worker processes if parallelizing the 96 holdout runs, as previously validated.

### 6.3 Data pipeline

Reuse the dataset registry and per-dataset `(transform, kernel, p)` defaults already established (Arrhythmia, Parkinson, Heart Disease, Dermatology, Climate Model Crashes, Breast Cancer Diagnostic, Breast Cancer, Blood Transfusion, Mammographic Mass, Qsar Biodegradation, Iris, Wine). **Before wiring DDR-MKSVM into the full 12-dataset suite, first resolve the open question flagged previously**: whether Heart Disease and Dermatology are used in binary or native multiclass form in the base paper (Table 3/Table 5 format strongly suggests binary). Implement `dataset_registry.py` with an explicit `label_mode: "binary" | "multiclass"` field per dataset and default Heart Disease/Dermatology to `"binary"` unless the upstream MATLAB repository (`aspinellibg/NonlinearSVM`) confirms otherwise — do not silently guess as the earlier reproduction did.

### 6.4 Module specifications

**`base_kernels.py`.** Each kernel class exposes `k(u, u')`, `gram(U)`, and `lipschitz_bound() -> float`, implementing Lemma 3 (`√(2α)` for Gaussian RBF, exact) and Lemma 4 (domain-dependent bound for polynomial, requiring `fit(X_train)` to compute `B = max‖x‖` first). The linear kernel returns `lipschitz_bound() = 1.0` exactly.

**`deep_kernel.py`.** `f_θ` is a small MLP (2–3 hidden layers, spectral-normalized linear layers via `torch.nn.utils.parametrizations.spectral_norm`, so that `‖W_t‖_2` is available directly — this makes Lemma 2's bound cheap to compute exactly rather than estimated). Given data dimensionality `n` as low as 4 (Iris) and as high as 279 (Arrhythmia), size the hidden width adaptively (e.g. `min(64, 4n)`) and keep depth ≤3 — the datasets are small (m ≤ 1055), so a large network will overfit and also produce a loose, unhelpfully large Lemma-2 bound.

**`mkl_combination.py`.** Maintains `η` as an unconstrained `torch` parameter `θ_η ∈ ℝ^L`, mapped to the simplex via `η = softmax(θ_η)` (standard reparameterization — keeps `η ∈ Δ_L` automatically, satisfying Lemma 1's precondition without a constrained solver).

**`analytic_bounds.py`.** Implements Lemma 2's product-of-spectral-norms and Lemma 5's combination formula `L_{θ,η} = L_f · sqrt(Σ_l η_l L_{φ,l}²)` exactly as derived. Exposed as a differentiable `torch` function so it participates in the backward pass through `η` and through the spectral norms of `θ`.

**`empirical_estimator.py`.** Independently estimates `Lip(f_θ)` via power iteration on random input pairs (`max over sampled (x,x') of ‖f_θ(x)-f_θ(x')‖/‖x-x'‖`, refined by local gradient ascent on the ratio). Used **only for diagnostics** — log both the analytic bound (Lemma 2) and the empirical estimate at every epoch, and flag a warning if the ratio analytic/empirical exceeds, say, 10× (indicating Lemma 2's bound has become so loose that the DRO penalty term `ε·‖w‖·L_{θ,η}` in (T-obj) is dominated by slack rather than genuine robustness — this is a known weakness of spectral-norm-product bounds and must be surfaced, not hidden).

**`convex_subproblem.py`.** Given fixed `(θ,η)` (hence fixed Gram matrix `K_{θ,η}` and fixed scalar `L_{θ,η}`), builds and solves the SOCP form of (T-obj) via `cvxpy`, following exactly the base paper's Appendix A `q=2` reformulation pattern, with the base paper's per-point `δ⁽ⁱ⁾√K_jj|u_j|` term replaced by the single scalar-times-norm term `ε·L_{θ,η}·‖w‖_ℍ` (Corollary 1). Returns `(w, γ, ξ, objective_value)`.

**`wasserstein_risk.py`.** Given a solved `(w,γ)` and current `(θ,η)`, computes the exact value of (T-obj) via (T1) for logging/validation — this must be checked against the direct (non-dualized) worst-case risk on a held-out adversarial search (see Test T-3, Section 6.7) to confirm the duality theorem is implemented correctly, not just its convenient closed form.

**`alternating_trainer.py`.** Implements the outer loop described in Section 6.5.

### 6.5 Training loop (alternating minimization)

```
Initialize θ (network weights), η (kernel mixture, via softmax parameterization)
for outer_iter in range(N_outer):
    # Convex block (Corollary 1): exact solve, no approximation
    K = compute_gram_matrix(theta, eta, X_train)
    L_theta_eta = analytic_bounds.compute(theta, eta)     # Lemma 5
    w, gamma, xi = convex_subproblem.solve(K, y_train, epsilon, L_theta_eta, nu)

    # Non-convex block: gradient step on (theta, eta) with (w, gamma) fixed
    for inner_iter in range(N_inner):
        loss = wasserstein_risk.compute(w, gamma, theta, eta, X_train, y_train, epsilon, nu)  # (T-obj)
        loss.backward()   # gradients flow through phi_{theta,eta} and through L_{theta,eta}
        optimizer.step()  # Adam on (theta, eta)

    log_metrics(outer_iter, loss, L_theta_eta, empirical_estimator.compute(theta))
    if converged(...): break

# Final: recompute (w, gamma) once more via the convex solver at the converged (theta, eta)
# so the reported classifier is always the exact optimum of the convex block, not a stale one.
```

Do not attempt full joint (non-alternating) end-to-end backprop through the `cvxpy` solve unless using a differentiable convex layer (e.g. `cvxpylayers`); if adopted, note that gradient computation through an SOCP is significantly more expensive per step than the alternating scheme above, so benchmark before switching.

### 6.6 Evaluation protocol

- Match the base paper's protocol exactly: 96 independent 75/25 stratified holdout runs per dataset, as already implemented and validated for the deterministic baseline.
- **Report both empirical std across the 96 runs and the theoretical binomial std** `√(p̂(1-p̂)/n_test)` side by side in every results table, addressing the reproducibility gap already identified — do not repeat the base paper's apparent under-reporting of variance.
- **Ablation suite** (`experiments/run_ablation.py`), run on the same subset validated previously (Parkinson, Breast Cancer Diagnostic, Mammographic Mass, Blood Transfusion) plus at least one high-dimensional case (Arrhythmia) and one multiclass case (Iris or Wine):
  1. **Full model**: DNN + MKL + DRO.
  2. **DNN off** (`f_θ = Identity`): isolates the value of MKL+DRO alone.
  3. **MKL off** (`L=1`, single best base kernel per the original paper's Table 3 choice): isolates DNN+DRO.
  4. **DRO off** (`ε=0`, reduces (T-obj) to a plain deep-multiple-kernel SVM): isolates whether robustness helps at all once the representation is already learned.
  5. **Legacy mode** (`legacy_reduction/base_paper_mode.py`, per Proposition in 4.8): must reproduce the original paper's Table 3 numbers on the validated subset within the tolerance already established in the prior reproduction (report Diff (pts) column) — this is a **hard correctness gate**, not just an ablation.
- Report balanced accuracy (mean of per-class recalls) alongside overall error in every table, given the Parkinson class-imbalance pathology (1.32% vs 53.56% per-class error) identified earlier — a change in overall error without a corresponding balanced-accuracy check is not sufficient evidence of a real improvement.

### 6.7 Required tests (map directly to Part I)

| Test | Validates | Method |
|---|---|---|
| `test_lemma1_psd.py` | Lemma 1 | Random `η`, random kernel matrices; assert combined Gram matrix has all eigenvalues `≥ -1e-8`. |
| `test_lemma2_spectral_bound.py` | Lemma 2 | Random small network; compare `Π‖W_t‖_2` against empirical Lipschitz estimate on random point pairs; assert analytic bound `≥` empirical estimate (never violated) on ≥1000 random pairs. |
| `test_lemma3_rbf_lipschitz.py` | Lemma 3 | For random `α`, random `(u,u')` pairs at varying distances, assert `‖φ(u)-φ(u')‖_ℍ ≤ √(2α)‖u-u'‖ + 1e-9`, computed via the kernel-trick identity `‖φ(u)-φ(u')‖² = k(u,u)+k(u',u')-2k(u,u')` (no explicit feature vector needed). |
| `test_theorem1_duality.py` | Theorem 1 | For a small fixed `(w,γ,θ,η)` and small `ε`, compute `R_ε` two ways: (a) via closed form (T1); (b) via direct adversarial maximization (projected gradient ascent on `x` within `‖x-x⁽ⁱ⁾‖≤` a fine grid of radii, numerically approximating the sup) — assert agreement within numerical tolerance. |
| `test_reduction_to_base_paper.py` | Proposition 4.8 | Run `legacy_reduction/base_paper_mode.py` on the previously-validated 4-dataset subset; assert results fall within the same tolerance band already established against the original paper's Table 3 (mean diff ≤ ~1 pt). |

### 6.8 Logging and reproducibility

- Fix and log all random seeds (network init, `η` init, holdout split generation) — the base paper's own MATLAB code does not do this (identified previously as a reproducibility gap); do not repeat that omission here.
- Log, per outer iteration: convex objective value, `L_{θ,η}` (analytic), empirical Lipschitz estimate, `η` (post-softmax), and the analytic/empirical ratio warning described in Section 6.4.
- Persist results in the same `results/<dataset>/{deterministic,robust}/...` structure as the earlier reproduction, with an added `ddr_mksvm/` subfolder per dataset, so all three generations of results (original-paper baseline, prior deterministic/robust reproduction, and this combined model) remain directly comparable in one place.

### 6.9 Milestones (suggested implementation order)

1. `base_kernels.py` + `test_lemma3_rbf_lipschitz.py` — get the exact, provable Lipschitz bound right first; everything downstream depends on it.
2. `convex_subproblem.py` restricted to **legacy mode** (single fixed kernel, no DRO, `ε=0`) — should immediately reproduce the already-validated deterministic baseline. This is your first correctness checkpoint before adding any new machinery.
3. Add `ε>0` with legacy mode (single kernel, no network) and validate against the base paper's own robust numbers (Table 3's robust columns) — this exercises Theorem 1 in its simplest instantiation and is the second correctness checkpoint.
4. Add `mkl_combination.py` with `L=1` collapsed back out (i.e. sanity check `L=1` behaves as legacy mode) before enabling `L>1`.
5. Add `deep_kernel.py` + `analytic_bounds.py` + `empirical_estimator.py`, wire into `alternating_trainer.py`.
6. Run the full ablation suite (Section 6.6) only after all five prior milestones pass their tests — do not run expensive ablations against an unvalidated pipeline.

## 7. Known limitations to state explicitly in any resulting write-up

- Lemma 2's spectral-norm-product bound is a **certified upper bound**, not the exact Lipschitz constant; it can be very loose for deep/wide networks, which would make the DRO penalty term overly conservative. This is why Section 6.4 mandates dual (analytic + empirical) tracking rather than trusting the analytic bound alone.
- Lemma 4's polynomial-kernel Lipschitz bound is domain-dependent (`B`-dependent), unlike Lemma 3's global Gaussian RBF bound — this asymmetry should be reported, not smoothed over.
- Theorem 1's strong-duality step (Step 2) is **cited, not re-derived**, from the general Wasserstein-DRO optimal-transport literature; this document proves everything else from scratch but is explicit that this one step rests on an external, well-established theorem rather than an original derivation.
- Given the dataset sizes involved (m as low as 68), DRO's asymptotic ambiguity-radius-shrinkage justification is weak; expect the DRO component to help less, or even hurt, on the smallest datasets (Arrhythmia) — this should be an explicit, reported finding of the ablation study, not treated as a bug if it occurs.

## 8. References

- Maggioni, F., & Spinelli, A. (2025). Robust support vector machines with nonlinear separators. *European Journal of Operational Research*, 322, 237–253. [Base paper; Theorem 1, Corollary 1, Propositions 1–2, Section 5.]
- Mohajerin Esfahani, P., & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric. *Mathematical Programming*, 171(1–2), 115–166. [Strong duality theorem invoked in Section 4.6, Step 2.]
- Gao, R., & Kleywegt, A. (2016/2023). Distributionally robust stochastic optimization with Wasserstein distance. *Mathematics of Operations Research*. [Alternative derivation of the same duality result.]
- Xu, H., Caramanis, C., & Mannor, S. (2009). Robustness and regularization of support vector machines. *JMLR*, 10, 1485–1510. [Regularization–robustness equivalence; conceptual precedent for Section 4.6's sanity check.]
- Gönen, M., & Alpaydın, E. (2011). Multiple kernel learning algorithms. *JMLR*, 12, 2211–2268. [Background for Definition 4 / Lemma 1's use.]
- Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). Deep kernel learning. *AISTATS*. [Conceptual precedent for Definition 3.]
