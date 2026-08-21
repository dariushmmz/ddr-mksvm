"""
Section 6.5's alternating trainer: exact convex solve of (u, gamma) for
fixed (theta, eta) [Corollary 1], alternated with gradient steps on
(theta, eta) for fixed (u, gamma) [differentiable through Theorem 1's
closed-form (T-obj)].

Ablation flags (Section 6.6):
  dnn_on : bool  -- if False, f_theta = Identity (Proposition 4.8)
  mkl_on : bool  -- if False (or len(base_kernel_specs)==1), eta is fixed
                    to a one-hot vector, no kernel-mixture optimization
  dro_on : bool  -- if False, epsilon is forced to 0 (plain deep/MK SVM,
                    ablation #4 in Section 6.6)
"""

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from ddr_mksvm.kernels.deep_kernel import DeepFeatureExtractor, IdentityFeatureExtractor
from ddr_mksvm.kernels.mkl_combination import KernelMixture
from ddr_mksvm.lipschitz.analytic_bounds import network_lipschitz_bound, composite_lipschitz_bound
from ddr_mksvm.optim.convex_subproblem import train_with_nu_search

from ddr_mksvm.lipschitz.empirical_estimator import check_representation_collapse

def _kernel_lipschitz_bounds(base_kernel_specs):
    bounds = []
    for spec in base_kernel_specs:
        if spec["kind"] == "linear":
            bounds.append(1.0)
        elif spec["kind"] == "rbf":
            a = 1.0 / (2 * spec["alpha"] ** 2)
            bounds.append(float(np.sqrt(2 * a)))
        elif spec["kind"] == "poly":
            bounds.append(float(spec.get("_lip_bound", 1.0)))
        else:
            raise ValueError(f"unknown kernel kind: {spec['kind']}")
    return bounds


def _torch_gram(spec, Z):
    """Differentiable Gram matrix for one base kernel, given deep-transformed
    features Z of shape (n_features, m) (torch tensor)."""
    if spec["kind"] == "linear":
        return Z.T @ Z
    if spec["kind"] == "poly":
        return (Z.T @ Z + spec["c"]) ** spec["degree"]
    if spec["kind"] == "rbf":
        diff = Z.unsqueeze(2) - Z.unsqueeze(1)
        sqdist = (diff ** 2).sum(0)
        return torch.exp(-sqdist / (2 * spec["alpha"] ** 2))
    raise ValueError(f"unknown kernel kind: {spec['kind']}")


class AlternatingTrainer:
    """
    X_np passed to fit() must be (n, m) -- features as COLUMNS, matching
    the `dati` convention used throughout the uploaded codebase.
    """

    def __init__(self, base_kernel_specs, in_dim, dnn_on=True, mkl_on=True,
                 dro_on=True, epsilon=0.001, nu_grid=None, out_dim=None,
                 n_outer=6, n_inner=15, lr=1e-2, seed=0):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for AlternatingTrainer.")

        self.base_kernel_specs = base_kernel_specs
        self.dnn_on = dnn_on
        self.mkl_on = mkl_on and len(base_kernel_specs) > 1
        self.dro_on = dro_on
        self.epsilon = float(epsilon) if dro_on else 0.0
        self.nu_grid = nu_grid if nu_grid is not None else np.logspace(-6, 0, 7)
        self.n_outer = n_outer
        self.n_inner = n_inner

        torch.manual_seed(seed)
        self.f_theta = DeepFeatureExtractor(in_dim, out_dim=out_dim) if dnn_on else IdentityFeatureExtractor()
        self.mixture = KernelMixture(len(base_kernel_specs)) if _TORCH_AVAILABLE else None

        params = list(self.f_theta.parameters()) if dnn_on else []
        if self.mkl_on:
            params += list(self.mixture.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr) if params else None

        self.history = []
        self.final_solution = None
        self.final_gram = None
        self.final_L_theta_eta = None

    def _eta_numpy(self):
        if self.mkl_on:
            return self.mixture.eta_numpy()
        return np.array([1.0] + [0.0] * (len(self.base_kernel_specs) - 1))

    def _current_L_theta_eta(self):
        L_f = network_lipschitz_bound(self.f_theta) if self.dnn_on else 1.0
        Lphi_list = _kernel_lipschitz_bounds(self.base_kernel_specs)
        eta = self._eta_numpy()
        return composite_lipschitz_bound(L_f, Lphi_list, eta)

    def _combined_gram_numpy(self, X_np):
        with torch.no_grad():
            Xt = torch.tensor(X_np, dtype=torch.float32)
            Z = self.f_theta(Xt.T).T if self.dnn_on else Xt  # keep (n_feat, m) layout
            Z_np = Z.detach().numpy()
        grams = []
        for spec in self.base_kernel_specs:
            if spec["kind"] == "linear":
                grams.append(Z_np.T @ Z_np)
            elif spec["kind"] == "poly":
                grams.append((Z_np.T @ Z_np + spec["c"]) ** spec["degree"])
            elif spec["kind"] == "rbf":
                diff = Z_np[:, :, None] - Z_np[:, None, :]
                sqdist = np.sum(diff ** 2, axis=0)
                grams.append(np.exp(-sqdist / (2 * spec["alpha"] ** 2)))
            else:
                raise ValueError(spec["kind"])
        eta = self._eta_numpy()
        K = sum(w * G for w, G in zip(eta, grams))
        return K

    def _run_collapse_check(self, X_np):
        """
        Called right before fit()/fit_one_vs_all() return. Only meaningful
        when a real network is in play (dnn_on=True) -- Identity has no
        way to "collapse" beyond the data itself being degenerate, which
        is a separate (data-quality) issue, not a training-failure one.
        Stores the diagnostic dict on self.collapse_diagnostics regardless
        of outcome, and prints a warning only if collapse was detected.
        """
        self.collapse_diagnostics = None
        if not self.dnn_on:
            return
        diag = check_representation_collapse(self.f_theta, X_np.T)  # X_np is (n,m) -> (m,n) batch-first
        svm_diag = None if self.final_solution is None else self.final_solution.get("diagnostics")
        if svm_diag is not None:
            diag.update({k: svm_diag[k] for k in (
                "weight_collapse", "score_collapse", "single_class_prediction",
                "slack_dominated", "baseline_equivalent", "degenerate")})
            diag["representation_collapse"] = diag.pop("collapsed")
            diag["collapsed"] = bool(diag["representation_collapse"] or diag["degenerate"])
        self.collapse_diagnostics = diag
        if diag["collapsed"]:
            print(
                "  [AlternatingTrainer] WARNING: f_theta appears to have collapsed to a "
                f"near-constant output (relative_std={diag['relative_std']:.2e}, "
                f"abs_std={diag['abs_std']:.2e}). The learned feature map is barely "
                "distinguishing between different training points, so the SVM built on top "
                "of it is degenerate. Inspect the SVM weight/score/slack flags before tuning."
            )

    def fit(self, X_np, y):
        y = np.asarray(y, dtype=float)
        X_np = np.asarray(X_np, dtype=float)
        if X_np.ndim != 2 or X_np.shape[1] != y.size:
            raise ValueError(f"X must have shape (features, samples) matching y; got {X_np.shape}, {y.shape}")
        if not np.isfinite(X_np).all() or not np.isfinite(y).all():
            raise ValueError("X and y must be finite")
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError(f"binary labels must be -1/+1; got {np.unique(y)}")
        best = None
        previous_error = None

        for outer in range(self.n_outer):
            # ---------------- convex block (Corollary 1) ----------------
            K = self._combined_gram_numpy(X_np)
            L_theta_eta = self._current_L_theta_eta() if self.dro_on else 0.0
            theta_before = torch.cat([p.detach().reshape(-1) for p in self.f_theta.parameters()]) if self.dnn_on else torch.zeros(1)
            eta_before = self._eta_numpy().copy()

            best = train_with_nu_search(K, y, self.nu_grid, epsilon=self.epsilon, L_theta_eta=L_theta_eta)
            if best is None:
                raise RuntimeError(f"convex subproblem failed to solve at outer iter {outer} for every nu")

            # ---------------- non-convex block: grad steps on (theta, eta) ----------------
            if self.optimizer is not None:
                u_t = torch.tensor(best["u"], dtype=torch.float32)
                gamma_t = torch.tensor(best["gamma"], dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                X_t = torch.tensor(X_np, dtype=torch.float32)

                for _ in range(self.n_inner):
                    self.optimizer.zero_grad()
                    Z = self.f_theta(X_t.T).T if self.dnn_on else X_t
                    grams = [_torch_gram(spec, Z) for spec in self.base_kernel_specs]
                    eta = self.mixture.eta() if self.mkl_on else torch.tensor(self._eta_numpy(), dtype=torch.float32)
                    K_t = sum(w * G for w, G in zip(eta, grams))
                    M_t = torch.outer(y_t, y_t) * K_t
                    scores = M_t @ u_t - y_t * gamma_t
                    hinge = torch.clamp(1 - scores, min=0).mean()

                    if self.dro_on:
                        L_f = network_lipschitz_bound(self.f_theta) if self.dnn_on else 1.0
                        Lphi = torch.tensor(_kernel_lipschitz_bounds(self.base_kernel_specs), dtype=torch.float32)
                        L_theta_eta_t = L_f * torch.sqrt(torch.clamp((eta * Lphi ** 2).sum(), min=1e-12))
                        w_norm = torch.sqrt(torch.clamp(u_t @ (M_t @ u_t), min=1e-12))
                        loss = hinge + self.epsilon * L_theta_eta_t * w_norm + best["selected_nu"] * w_norm ** 2
                    else:
                        w_norm_sq = torch.clamp(u_t @ (M_t @ u_t), min=0.0)
                        loss = hinge + best["selected_nu"] * w_norm_sq

                    loss.backward()
                    self.optimizer.step()

            K_after = self._combined_gram_numpy(X_np)
            theta_after = torch.cat([p.detach().reshape(-1) for p in self.f_theta.parameters()]) if self.dnn_on else torch.zeros(1)
            eta_after = self._eta_numpy()
            d = best["diagnostics"]
            self.history.append(dict(outer=outer, training_error=best["training_error"],
                delta_training_error=(None if previous_error is None else best["training_error"]-previous_error),
                L_theta_eta=L_theta_eta, eta=eta_after.tolist(), selected_nu=best["selected_nu"],
                theta_delta_norm=float(torch.linalg.vector_norm(theta_after-theta_before)),
                eta_delta_norm=float(np.linalg.norm(eta_after-eta_before)),
                feature_matrix_delta_norm=float(np.linalg.norm(K_after-K)),
                svm_u_norm=d["w_norm_H"], sum_xi=d["sum_xi"], score_std=d["score_std"]))
            previous_error = best["training_error"]
            print(f"  [OUTER {outer}] error={best['training_error']:.6f} delta={self.history[-1]['delta_training_error']} "
                  f"L={L_theta_eta:.6g} theta_delta={self.history[-1]['theta_delta_norm']:.6g} "
                  f"eta_delta={self.history[-1]['eta_delta_norm']:.6g} K_delta={self.history[-1]['feature_matrix_delta_norm']:.6g}")

        # final exact convex re-solve at the converged (theta, eta), so the
        # reported classifier is always the exact optimum of the convex
        # block for the FINAL feature map, not a stale one from an earlier
        # outer iteration (spec Section 6.5)
        final_K = self._combined_gram_numpy(X_np)
        final_L = self._current_L_theta_eta() if self.dro_on else 0.0
        final_best = train_with_nu_search(final_K, y, self.nu_grid, epsilon=self.epsilon, L_theta_eta=final_L)
        if final_best is None:
            final_best = best  # fall back to the last successful outer-iter solution

        self.final_solution = final_best
        self.final_gram = final_K
        self.final_L_theta_eta = final_L

        self._run_collapse_check(X_np)
        return self


    def fit_one_vs_all(self, X_np, y_label, L):
        """
        Multiclass extension (one-vs-all, L binary subproblems), used by
        unit_of_work_ddr_multiclass.py. The deep feature extractor f_theta
        and kernel mixture eta are SHARED across all L subproblems -- a
        single learned representation optimized jointly against the mean
        one-vs-all hinge loss (and, if dro_on, the mean DRO penalty)
        across classes -- rather than training L independent networks.

        y_label : (m,) integer labels in {1,...,L}
        Sets self.final_solutions : list[dict], one solve_svm_dro-style
        dict per class (index l-1 for class l), and self.final_gram.
        """
        import numpy as np
        y_label = np.asarray(y_label, dtype=int)
        y_hats = [np.where(y_label == l, 1.0, -1.0) for l in range(1, L + 1)]

        solutions = None
        for outer in range(self.n_outer):
            # ---------------- convex block: L independent solves, shared K ----------------
            K = self._combined_gram_numpy(X_np)
            L_theta_eta = self._current_L_theta_eta() if self.dro_on else 0.0

            solutions = []
            for y_hat in y_hats:
                sol = train_with_nu_search(K, y_hat, self.nu_grid, epsilon=self.epsilon, L_theta_eta=L_theta_eta)
                if sol is None:
                    raise RuntimeError(f"convex subproblem failed at outer iter {outer} for one one-vs-all class")
                solutions.append(sol)

            self.history.append(dict(
                outer=outer, L_theta_eta=L_theta_eta, eta=self._eta_numpy().tolist(),
                training_errors=[s["training_error"] for s in solutions],
            ))

            # ---------------- non-convex block: grad steps on shared (theta, eta) ----------------
            if self.optimizer is not None:
                X_t = torch.tensor(X_np, dtype=torch.float32)
                u_list = [torch.tensor(s["u"], dtype=torch.float32) for s in solutions]
                gamma_list = [torch.tensor(s["gamma"], dtype=torch.float32) for s in solutions]
                y_hat_t_list = [torch.tensor(yh, dtype=torch.float32) for yh in y_hats]

                for _ in range(self.n_inner):
                    self.optimizer.zero_grad()
                    Z = self.f_theta(X_t.T).T if self.dnn_on else X_t
                    grams = [_torch_gram(spec, Z) for spec in self.base_kernel_specs]
                    eta = self.mixture.eta() if self.mkl_on else torch.tensor(self._eta_numpy(), dtype=torch.float32)
                    K_t = sum(w * G for w, G in zip(eta, grams))

                    total_hinge = 0.0
                    for u_t, gamma_t, y_hat_t in zip(u_list, gamma_list, y_hat_t_list):
                        M_t = torch.outer(y_hat_t, y_hat_t) * K_t
                        scores = M_t @ u_t - y_hat_t * gamma_t
                        total_hinge = total_hinge + torch.clamp(1 - scores, min=0).mean()
                    total_hinge = total_hinge / L

                    total_regularization = 0.0
                    for u_t, y_hat_t, sol in zip(u_list, y_hat_t_list, solutions):
                        M_t = torch.outer(y_hat_t, y_hat_t) * K_t
                        total_regularization = total_regularization + sol["selected_nu"] * torch.clamp(
                            u_t @ (M_t @ u_t), min=0.0)
                    total_regularization = total_regularization / L

                    if self.dro_on:
                        L_f = network_lipschitz_bound(self.f_theta) if self.dnn_on else 1.0
                        Lphi = torch.tensor(_kernel_lipschitz_bounds(self.base_kernel_specs), dtype=torch.float32)
                        L_theta_eta_t = L_f * torch.sqrt(torch.clamp((eta * Lphi ** 2).sum(), min=1e-12))
                        total_wnorm = 0.0
                        for u_t, y_hat_t in zip(u_list, y_hat_t_list):
                            M_t = torch.outer(y_hat_t, y_hat_t) * K_t
                            total_wnorm = total_wnorm + torch.sqrt(torch.clamp(u_t @ (M_t @ u_t), min=1e-12))
                        total_wnorm = total_wnorm / L
                        loss = total_hinge + self.epsilon * L_theta_eta_t * total_wnorm + total_regularization
                    else:
                        loss = total_hinge + total_regularization

                    loss.backward()
                    self.optimizer.step()

        # final exact convex re-solve for every class at the converged (theta, eta)
        final_K = self._combined_gram_numpy(X_np)
        final_L = self._current_L_theta_eta() if self.dro_on else 0.0
        final_solutions = []
        for l_idx, y_hat in enumerate(y_hats):
            sol = train_with_nu_search(final_K, y_hat, self.nu_grid, epsilon=self.epsilon, L_theta_eta=final_L)
            final_solutions.append(sol if sol is not None else solutions[l_idx])

        self.final_solutions = final_solutions
        self.final_gram = final_K
        self.final_L_theta_eta = final_L
        self._run_collapse_check(X_np)
        return self
