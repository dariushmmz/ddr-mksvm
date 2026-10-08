"""Paper-faithful, leakage-safe Iris research primitives.

This module is intentionally independent of the newer DDR q=2/Wasserstein
stack.  It implements the Maggioni--Spinelli q=1 LP, its bounded-input robust
counterpart, exact threshold search, and OVA/OVO reductions.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linprog, nnls
from scipy.spatial.distance import pdist
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC


PAPER_NUMERIC_SHA256 = "1b3e788db71aac2852e77ce253c00e9cedd173d8b5815b47aa530252a4c99e70"
PAPER_NU_GRID = tuple(float(v) for v in np.logspace(-3, 0, 5))
SOURCE_CONTINUED_NU_GRID = tuple(float(v) for v in np.logspace(-3, 1.5, 7))
EXPANDED_NU_GRID = tuple(float(v) for v in np.logspace(-3, 1, 9))
WIDE_NU_GRID = tuple(float(v) for v in np.logspace(-3, 2, 11))
ALPHA_RULES = ("paper", "med:-2", "med:-1", "med:-0.5", "med:0", "med:0.5", "med:1", "med:2")
COARSE_RHO_GRID = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SVC_C_GRID = (0.1, 1.0, 10.0, 100.0)


def numeric_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def reconstruct_authors_iris(local_values: np.ndarray) -> np.ndarray:
    """Return the authors' numeric Iris data without altering historical CSVs."""
    values = np.asarray(local_values, dtype=float).copy()
    if values.shape != (150, 5):
        raise ValueError(f"expected Iris numeric array (150,5), got {values.shape}")
    if not np.array_equal(values[:, -1], np.repeat([1.0, 2.0, 3.0], 50)):
        raise ValueError("Iris rows/labels do not have the expected authors' ordering")
    values[34] = [4.9, 3.1, 1.5, 0.2, 1.0]
    values[37] = [4.9, 3.6, 1.4, 0.1, 1.0]
    digest = numeric_sha256(values)
    if digest != PAPER_NUMERIC_SHA256:
        raise ValueError(f"reconstructed authors' Iris hash mismatch: {digest}")
    return values


def locked_split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels), dtype=int)
    train, test = train_test_split(
        indices, test_size=0.25, stratify=np.asarray(labels), random_state=int(seed)
    )
    # MATLAB applies cvpartition's logical masks to the table.  Logical table
    # indexing retains source-row order; train_test_split returns shuffled
    # index arrays.  Sorting is therefore required for split-level parity even
    # though membership and aggregate metrics are permutation invariant in
    # exact arithmetic.
    return np.sort(np.asarray(train, dtype=int)), np.sort(np.asarray(test, dtype=int))


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def paper_alpha(X: np.ndarray) -> float:
    """MATLAB ``max(std(dati,0,2))``: sample standard deviation."""
    value = float(np.max(np.std(np.asarray(X, float), axis=0, ddof=1)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("paper RBF bandwidth is not positive and finite")
    return value


def median_nonzero_distance(X: np.ndarray) -> float:
    distances = pdist(np.asarray(X, float), metric="euclidean")
    distances = distances[distances > np.finfo(float).eps]
    if not distances.size:
        raise ValueError("training features are collapsed")
    return float(np.median(distances))


def alpha_from_rule(X: np.ndarray, rule: str) -> float:
    if rule == "paper":
        return paper_alpha(X)
    if not rule.startswith("med:"):
        raise ValueError(f"unknown alpha rule {rule!r}")
    return median_nonzero_distance(X) * (2.0 ** float(rule.split(":", 1)[1]))


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Training-only sample-z-score transform (MATLAB std convention)."""
    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0, ddof=1)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return (X_train - mean) / scale, (X_test - mean) / scale, {
        "preprocess_mean": mean.tolist(), "preprocess_scale": scale.tolist(),
        "preprocess_fitted_rows": int(len(X_train)),
    }


def rbf_gram(X: np.ndarray, alpha: float, Xq: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xq = X if Xq is None else np.asarray(Xq, dtype=float)
    sq = np.sum((X[:, None, :] - Xq[None, :, :]) ** 2, axis=2)
    return np.exp(-sq / (2.0 * float(alpha) ** 2))


def polynomial_gram(X: np.ndarray, degree: int = 2, offset: float = 0.0,
                    Xq: np.ndarray | None = None) -> np.ndarray:
    """Literal MATLAB ``(x'*z+c)^d`` Gram/cross-kernel matrix."""
    X = np.asarray(X, dtype=float)
    Xq = X if Xq is None else np.asarray(Xq, dtype=float)
    if int(degree) != degree or degree < 1:
        raise ValueError("polynomial degree must be a positive integer")
    return (X @ Xq.T + float(offset)) ** int(degree)


def multiscale_rbf_gram(X: np.ndarray, alpha: float,
                        scales: Iterable[float], weights: Iterable[float],
                        Xq: np.ndarray | None = None) -> np.ndarray:
    scales = np.asarray(tuple(scales), dtype=float)
    weights = np.asarray(tuple(weights), dtype=float)
    if scales.ndim != 1 or weights.shape != scales.shape or scales.size == 0:
        raise ValueError("kernel scales and weights must be matching non-empty vectors")
    if np.any(scales <= 0.0) or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("kernel scales must be positive and weights must be on the simplex")
    return sum(
        float(weight) * rbf_gram(X, float(alpha) * float(scale), Xq)
        for scale, weight in zip(scales, weights)
    )


def centered_target_nnls_weights(X: np.ndarray, labels: np.ndarray, alpha: float,
                                 scales: Iterable[float]) -> tuple[np.ndarray, dict]:
    """Fit nonnegative multi-kernel weights using training labels only.

    The objective is the convex Frobenius regression
    ``min_{beta>=0} ||T_c-sum_s beta_s K_{s,c}||_F``.  Normalizing the fitted
    beta onto the simplex preserves the direction and yields a PSD kernel.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    scales = tuple(float(value) for value in scales)
    n = len(labels)
    H = np.eye(n) - np.ones((n, n)) / n
    classes = sorted(int(value) for value in np.unique(labels))
    one_hot = np.column_stack([labels == value for value in classes]).astype(float)
    target = H @ (one_hot @ one_hot.T) @ H
    centered = [H @ rbf_gram(X, alpha * scale) @ H for scale in scales]
    design = np.column_stack([kernel.reshape(-1) for kernel in centered])
    beta, residual = nnls(design, target.reshape(-1))
    if beta.sum() <= np.finfo(float).eps:
        weights = np.full(len(scales), 1.0 / len(scales))
        fallback = True
    else:
        weights = beta / beta.sum()
        fallback = False
    fitted = design @ beta
    target_norm = max(float(np.linalg.norm(target)), np.finfo(float).eps)
    return weights, {
        "kernel_weight_method": "centered-target-nnls",
        "kernel_nnls_beta": beta.tolist(),
        "kernel_weights": weights.tolist(),
        "kernel_nnls_residual": float(residual),
        "kernel_nnls_relative_residual": float(np.linalg.norm(target.reshape(-1) - fitted) / target_norm),
        "kernel_weight_fallback": fallback,
    }


def gram_diagnostics(K: np.ndarray) -> dict:
    values = np.linalg.eigvalsh((np.asarray(K) + np.asarray(K).T) / 2.0)
    largest = max(float(values[-1]), np.finfo(float).eps)
    positive = values[values > largest * 1e-12]
    condition = float(largest / positive[0]) if positive.size else float("inf")
    effective_rank = float(np.exp(-np.sum((values.clip(min=0) / values.clip(min=0).sum()) *
                                          np.log(np.maximum(values.clip(min=0) /
                                                            values.clip(min=0).sum(), 1e-300)))))
    return {
        "gram_min_eigenvalue": float(values[0]),
        "gram_max_eigenvalue": float(values[-1]),
        "gram_condition_number": condition,
        "gram_effective_rank": effective_rank,
    }


def _classification_error(scores: np.ndarray, y: np.ndarray, b: float) -> float:
    pred = np.where(np.asarray(scores) - float(b) > 0.0, 1.0, -1.0)
    return float(np.mean(pred != np.asarray(y)))


def fixed_grid_threshold(scores: np.ndarray, y: np.ndarray, gamma: float, xi: np.ndarray,
                         robust_shift: np.ndarray | None = None,
                         points: int = 10_000) -> tuple[float, float, dict]:
    dxi = np.asarray(y) * np.asarray(xi)
    raw_left = float(gamma + 1.0 - np.max(-dxi))
    raw_right = float(gamma - 1.0 + np.max(dxi))
    left, right = sorted((raw_left, raw_right))
    # Preserve the authors' literal MATLAB order. Their strict-improvement
    # loop keeps the first minimum, so reversing a descending strip would
    # silently change b even though the candidate set is unchanged.
    grid = np.linspace(raw_left, raw_right, int(points))
    shift = np.zeros_like(y, dtype=float) if robust_shift is None else np.asarray(robust_shift, float)
    signed = np.asarray(y)[None, :] * (np.asarray(scores)[None, :] - grid[:, None]) - shift[None, :]
    # MATLAB counts D*(-score+b)>0.  This is signed<0 exactly; equality is not
    # counted as a training mistake for either sign (which differs from any
    # total binary prediction convention at a breakpoint).
    errors = np.mean(signed < 0.0, axis=1)
    best = int(np.argmin(errors))
    return float(grid[best]), float(errors[best]), {
        "threshold_candidates": int(points),
        "strip_left": left,
        "strip_right": right,
        "strip_reversed": bool(raw_left > raw_right),
    }


def exact_threshold(scores: np.ndarray, y: np.ndarray, gamma: float, xi: np.ndarray,
                    robust_shift: np.ndarray | None = None) -> tuple[float, float, dict]:
    """Globally minimize binary training error over the Liu--Potra strip."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    shift = np.zeros_like(y) if robust_shift is None else np.asarray(robust_shift, float)
    dxi = y * np.asarray(xi, dtype=float)
    raw_left = float(gamma + 1.0 - np.max(-dxi))
    raw_right = float(gamma - 1.0 + np.max(dxi))
    left, right = sorted((raw_left, raw_right))

    # y_i(scores_i-b)-shift_i=0 gives the robust discontinuity.
    breakpoints = scores - shift / y
    inside = np.unique(breakpoints[(breakpoints >= left) & (breakpoints <= right)])
    boundaries = np.unique(np.concatenate(([left], inside, [right])))
    mids = (boundaries[:-1] + boundaries[1:]) / 2.0 if boundaries.size > 1 else np.empty(0)
    candidates = np.unique(np.concatenate((boundaries, mids)))
    signed = y[None, :] * (scores[None, :] - candidates[:, None]) - shift[None, :]
    if robust_shift is None:
        # Match the implemented classifier exactly: score-b > 0 is positive;
        # equality is negative.  In particular, a negative example on the
        # threshold is correctly classified, while a positive example is not.
        predictions = np.where(scores[None, :] - candidates[:, None] > 0.0, 1.0, -1.0)
        errors = np.mean(predictions != y[None, :], axis=1)
    else:
        # A point is robustly correct only when its worst-case signed margin is
        # strictly positive; equality has no positive robustness certificate.
        errors = np.mean(signed <= 0.0, axis=1)
    order = np.lexsort((candidates, np.abs(candidates - float(gamma)), errors))
    best = int(order[0])
    return float(candidates[best]), float(errors[best]), {
        "threshold_candidates": int(candidates.size),
        "strip_left": left,
        "strip_right": right,
        "strip_reversed": bool(raw_left > raw_right),
    }


@dataclass
class BinaryModel:
    train_X: np.ndarray
    y: np.ndarray
    u: np.ndarray
    gamma: float
    b: float
    xi: np.ndarray
    alpha: float
    nu: float
    positive_class: int
    negative_class: int | None
    diagnostics: dict
    alpha_scales: tuple[float, ...] = (1.0,)
    kernel_weights: tuple[float, ...] = (1.0,)
    kernel_kind: str = "rbf"
    polynomial_degree: int = 2
    polynomial_offset: float = 0.0

    def decision(self, Xq: np.ndarray) -> np.ndarray:
        if self.kernel_kind == "rbf":
            kernel = multiscale_rbf_gram(
                self.train_X, self.alpha, self.alpha_scales, self.kernel_weights, Xq
            )
        elif self.kernel_kind == "poly":
            kernel = polynomial_gram(
                self.train_X, self.polynomial_degree, self.polynomial_offset, Xq
            )
        else:
            raise ValueError(f"unknown fitted kernel {self.kernel_kind!r}")
        return kernel.T @ (self.y * self.u) - self.b


def robust_delta(X: np.ndarray, original_labels: np.ndarray, rho: float,
                 alpha: float) -> np.ndarray:
    """Paper Proposition-2 radii for p=infinity, training statistics only."""
    X = np.asarray(X, float)
    labels = np.asarray(original_labels, int)
    output = np.zeros(len(X), dtype=float)
    C = np.sqrt(X.shape[1])
    for label in np.unique(labels):
        class_X = X[labels == label]
        scale = float(np.max(np.std(class_X, axis=0, ddof=1)))
        eta = float(rho) * scale
        delta = np.sqrt(2.0 - 2.0 * np.exp(-((C * eta) ** 2) / (2.0 * alpha ** 2)))
        output[labels == label] = delta
    return output


def build_q1_lp(K: np.ndarray, y: np.ndarray, nu: float,
                delta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Build the literal MATLAB q=1 LP in SciPy's ``A_ub z <= b_ub`` form.

    ``z = [u, gamma, xi, s]`` and ``M = D K D``.  With ``delta=0`` this is
    exactly ``D*(K*D*u-1*gamma)+xi >= 1``, ``xi>=0``, and ``-s<=u<=s``.
    The optional delta term is retained for the separately audited robust LP.
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    m = len(y)
    if K.shape != (m, m):
        raise ValueError("K and y dimensions do not agree")
    delta = np.zeros(m, dtype=float) if delta is None else np.asarray(delta, float)
    if delta.shape != (m,):
        raise ValueError("delta and y dimensions do not agree")
    M = np.outer(y, y) * K
    nvar = 3 * m + 1
    objective = np.zeros(nvar)
    objective[m + 1:2 * m + 1] = float(nu)
    objective[2 * m + 1:] = 1.0
    margin = np.zeros((m, nvar))
    margin[:, :m] = -M
    margin[:, m] = y
    margin[:, m + 1:2 * m + 1] = -np.eye(m)
    if np.any(delta):
        margin[:, 2 * m + 1:] = np.outer(
            delta, np.sqrt(np.clip(np.diag(K), 0.0, None))
        )
    upper = np.zeros((2 * m, nvar))
    upper[:m, :m] = np.eye(m)
    upper[:m, 2 * m + 1:] = -np.eye(m)
    upper[m:, :m] = -np.eye(m)
    upper[m:, 2 * m + 1:] = -np.eye(m)
    A_ub = np.vstack((margin, upper))
    b_ub = np.concatenate((-np.ones(m), np.zeros(2 * m)))
    bounds = [(None, None)] * (m + 1) + [(0.0, None)] * (2 * m)
    return objective, A_ub, b_ub, bounds


def _solve_lp(objective: np.ndarray, A_ub: np.ndarray, b_ub: np.ndarray,
              bounds: list, solver_backend: str) -> dict:
    """Solve one parity LP while preserving the solver-returned primal point."""
    if solver_backend.startswith("scipy-"):
        method = solver_backend.removeprefix("scipy-")
        result = linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
        if not result.success:
            raise RuntimeError(
                f"SciPy {method} LP failed: status={result.status}, {result.message}"
            )
        return {
            "x": np.asarray(result.x, float), "objective": float(result.fun),
            "status": str(result.message), "status_code": int(result.status),
            "iterations": int(result.nit), "solver": solver_backend,
        }
    if solver_backend.startswith("cvxpy-"):
        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - optional parity backend
            raise ImportError("CVXPY is required for this parity solver backend") from exc
        solver_key = solver_backend.removeprefix("cvxpy-").upper()
        solvers = {"HIGHS": cp.HIGHS, "CLARABEL": cp.CLARABEL, "CVXOPT": cp.CVXOPT}
        if solver_key not in solvers:
            raise ValueError(f"unsupported CVXPY parity solver {solver_key!r}")
        z = cp.Variable(len(objective))
        first_nonnegative = (len(objective) - 1) // 3 + 1
        problem = cp.Problem(
            cp.Minimize(objective @ z),
            [A_ub @ z <= b_ub, z[first_nonnegative:] >= 0.0],
        )
        kwargs = {}
        if solver_key == "CLARABEL":
            kwargs = dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
        elif solver_key == "CVXOPT":
            kwargs = dict(abstol=1e-9, reltol=1e-9, feastol=1e-9)
        problem.solve(solver=solvers[solver_key], **kwargs)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or z.value is None:
            raise RuntimeError(f"CVXPY {solver_key} LP failed: {problem.status}")
        iterations = problem.solver_stats.num_iters
        return {
            "x": np.asarray(z.value, float), "objective": float(problem.value),
            "status": str(problem.status),
            "status_code": 0 if problem.status == cp.OPTIMAL else 1,
            "iterations": int(iterations) if iterations is not None else -1,
            "solver": solver_backend,
        }
    raise ValueError(f"unsupported solver backend {solver_backend!r}")


def solve_binary_q1(X: np.ndarray, y: np.ndarray, alpha: float, nu: float,
                    threshold_mode: str = "exact", rho: float = 0.0,
                    original_labels: np.ndarray | None = None,
                    context: dict | None = None,
                    alpha_scales: Iterable[float] = (1.0,),
                    kernel_weights: Iterable[float] = (1.0,),
                    kernel_kind: str = "rbf", polynomial_degree: int = 2,
                    polynomial_offset: float = 0.0,
                    solver_backend: str = "scipy-highs") -> tuple[BinaryModel, dict]:
    start = time.perf_counter()
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = len(y)
    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("binary labels must contain -1 and +1")
    alpha_scales = tuple(float(value) for value in alpha_scales)
    kernel_weights = tuple(float(value) for value in kernel_weights)
    if kernel_kind == "rbf":
        if rho > 0.0 and alpha_scales != (1.0,):
            raise ValueError("robust multi-scale radius is not enabled in this experiment family")
        K = multiscale_rbf_gram(X, alpha, alpha_scales, kernel_weights)
    elif kernel_kind == "poly":
        if rho > 0.0:
            raise ValueError("robust polynomial radius is not enabled in this experiment family")
        K = polynomial_gram(X, polynomial_degree, polynomial_offset)
    else:
        raise ValueError(f"unknown kernel kind {kernel_kind!r}")
    M = np.outer(y, y) * K
    delta = np.zeros(m)
    if rho > 0:
        if original_labels is None:
            raise ValueError("robust solve needs original multiclass labels")
        delta = robust_delta(X, original_labels, rho, alpha)

    # Variables: [u(m), gamma, xi(m), s(m)].
    objective, A_ub, b_ub, bounds = build_q1_lp(K, y, nu, delta)
    result = _solve_lp(objective, A_ub, b_ub, bounds, solver_backend)
    u = np.asarray(result["x"][:m], float)
    gamma = float(result["x"][m])
    raw_xi = np.asarray(result["x"][m + 1:2 * m + 1], float)
    robust_term = delta * float(np.sqrt(np.clip(np.diag(K), 0, None)) @ np.abs(u))
    required_xi = np.maximum(0.0, 1.0 - (M @ u - y * gamma - robust_term))
    # The executable MATLAB source constructs omega and the b strip from the
    # optimizer-returned xi_l.  Replacing it with a residual-derived value is
    # useful as a feasibility repair, but is not reproduction-faithful and can
    # move both strip endpoints.  Keep the returned primal variable here and
    # report the residual-derived quantity only as a numerical diagnostic.
    xi = raw_xi
    raw_scores = K @ (y * u)
    if threshold_mode == "grid":
        b, robust_train_error, threshold_diag = fixed_grid_threshold(
            raw_scores, y, gamma, xi, robust_term if rho > 0 else None
        )
    elif threshold_mode == "exact":
        b, robust_train_error, threshold_diag = exact_threshold(
            raw_scores, y, gamma, xi, robust_term if rho > 0 else None
        )
    else:
        raise ValueError(f"unknown threshold mode {threshold_mode!r}")
    prediction = np.where(raw_scores - b > 0.0, 1.0, -1.0)
    gdiag = gram_diagnostics(K)
    w_norm = float(np.sqrt(max(u @ (M @ u), 0.0)))
    diag = {
        **(context or {}),
        "alpha": float(alpha), "nu": float(nu), "normalized_lambda": float(1.0 / (nu * m)),
        "alpha_scales": list(alpha_scales), "kernel_weights": list(kernel_weights),
        "kernel_kind": kernel_kind, "polynomial_degree": int(polynomial_degree),
        "polynomial_offset": float(polynomial_offset),
        "rho": float(rho), "p": "inf" if rho > 0 else "none", "sample_count": int(m),
        "solver": result["solver"], "solver_status": result["status"],
        "solver_status_code": result["status_code"], "solver_iterations": result["iterations"],
        "solver_objective": result["objective"], "solver_runtime_s": time.perf_counter() - start,
        "u_l1": float(np.abs(u).sum()), "u_l2": float(np.linalg.norm(u)),
        "w_norm_H": w_norm, "sum_xi": float(xi.sum()), "gamma": gamma, "b": b,
        "nonzero_coefficients": int(np.sum(np.abs(u) > 1e-8)),
        "training_error": float(np.mean(prediction != y)),
        "robust_training_error": robust_train_error,
        "trivial_u": bool(np.linalg.norm(u, 1) <= 1e-8),
        "single_class_prediction": bool(np.unique(prediction).size == 1),
        "raw_xi_repair_max": float(np.max(np.abs(required_xi - raw_xi))),
        "minimum_required_xi_sum": float(required_xi.sum()),
        "max_delta": float(delta.max()),
        **threshold_diag, **gdiag,
    }
    model = BinaryModel(
        X, y, u, gamma, b, xi, float(alpha), float(nu), 0, None, diag,
        alpha_scales, kernel_weights, kernel_kind, int(polynomial_degree),
        float(polynomial_offset),
    )
    return model, diag


def _binary_tasks(labels: np.ndarray, decomposition: str):
    classes = sorted(int(v) for v in np.unique(labels))
    if decomposition == "ova":
        for positive in classes:
            mask = np.ones(len(labels), dtype=bool)
            yield positive, None, mask, np.where(labels == positive, 1.0, -1.0)
    elif decomposition == "ovo":
        for positive, negative in itertools.combinations(classes, 2):
            mask = (labels == positive) | (labels == negative)
            yield positive, negative, mask, np.where(labels[mask] == positive, 1.0, -1.0)
    else:
        raise ValueError(f"unknown decomposition {decomposition!r}")


def fit_multiclass(X: np.ndarray, labels: np.ndarray, alpha: float,
                   decomposition: str, threshold_mode: str,
                   nu: float | None = None, nu_grid: Iterable[float] = PAPER_NU_GRID,
                   rho: float = 0.0, context: dict | None = None,
                   alpha_scales: Iterable[float] = (1.0,),
                   kernel_weights: Iterable[float] = (1.0,),
                   kernel_kind: str = "rbf", polynomial_degree: int = 2,
                   polynomial_offset: float = 0.0,
                   solver_backend: str = "scipy-highs",
                   nu_selection: str = "per_binary",
                   bias_selection: str = "per_binary") -> tuple[list[BinaryModel], list[dict]]:
    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    models, all_diagnostics, task_solutions = [], [], []
    for task_index, (positive, negative, mask, binary_y) in enumerate(_binary_tasks(labels, decomposition)):
        candidates = [float(nu)] if nu is not None else [float(v) for v in nu_grid]
        solved = []
        for candidate_nu in candidates:
            binary_context = {
                **(context or {}), "task_index": task_index,
                "positive_class": positive, "negative_class": negative,
            }
            model, diag = solve_binary_q1(
                X[mask], binary_y, alpha, candidate_nu, threshold_mode, rho,
                labels[mask], binary_context, alpha_scales, kernel_weights,
                kernel_kind, polynomial_degree, polynomial_offset, solver_backend,
            )
            model.positive_class, model.negative_class = positive, negative
            solved.append((diag["robust_training_error"], candidates.index(candidate_nu), model))
            diag["selected_for_task"] = False
            all_diagnostics.append(diag)
        task_solutions.append(solved)
        if nu_selection == "per_binary":
            # MATLAB uses strict improvement, so grid order breaks training-error ties.
            chosen = min(solved, key=lambda item: (item[0], item[1]))[2]
            chosen.diagnostics["selected_for_task"] = True
            models.append(chosen)
    if nu_selection == "joint_multiclass_train":
        if decomposition != "ova" or nu is not None:
            raise ValueError("joint multiclass nu selection requires OVA and a nu grid")
        best = None
        for combination in itertools.product(*(range(len(rows)) for rows in task_solutions)):
            candidate_models = [task_solutions[i][j][2] for i, j in enumerate(combination)]
            error = float(np.mean(predict_multiclass(candidate_models, X, "ova") != labels))
            key = (error, combination)
            if best is None or key < best[0]:
                best = (key, candidate_models)
        models = best[1]
        for model in models:
            model.diagnostics["selected_for_task"] = True
            model.diagnostics["joint_multiclass_training_error"] = best[0][0]
    elif nu_selection != "per_binary":
        raise ValueError(f"unknown nu selection mode {nu_selection!r}")
    if bias_selection == "joint_multiclass_train":
        models, _ = optimize_ova_biases(models, X, labels)
    elif bias_selection != "per_binary":
        raise ValueError(f"unknown bias selection mode {bias_selection!r}")
    return models, all_diagnostics


def predict_multiclass(models: list[BinaryModel], Xq: np.ndarray,
                       decomposition: str, normalize_ova: bool = False) -> np.ndarray:
    classes = sorted({m.positive_class for m in models} |
                     {m.negative_class for m in models if m.negative_class is not None})
    if decomposition == "ova":
        scores = np.column_stack([model.decision(Xq) for model in models])
        if normalize_ova:
            norms = np.asarray([
                max(float(model.diagnostics["w_norm_H"]), 1e-12) for model in models
            ])
            scores = scores / norms[None, :]
        return np.asarray(classes, int)[np.argmax(scores, axis=1)]
    votes = np.zeros((len(Xq), len(classes)), dtype=int)
    margin_sum = np.zeros_like(votes, dtype=float)
    class_to_col = {value: index for index, value in enumerate(classes)}
    for model in models:
        margin = model.decision(Xq)
        scale = max(float(model.diagnostics["w_norm_H"]), 1e-12)
        normalized = margin / scale
        pos_col, neg_col = class_to_col[model.positive_class], class_to_col[model.negative_class]
        positive_vote = margin > 0.0
        votes[positive_vote, pos_col] += 1
        votes[~positive_vote, neg_col] += 1
        margin_sum[:, pos_col] += normalized
        margin_sum[:, neg_col] -= normalized
    output = np.empty(len(Xq), dtype=int)
    for row in range(len(Xq)):
        tied = np.flatnonzero(votes[row] == votes[row].max())
        if tied.size > 1:
            best_margin = margin_sum[row, tied].max()
            tied = tied[np.isclose(margin_sum[row, tied], best_margin, rtol=0.0, atol=1e-12)]
        output[row] = classes[int(tied[0])]
    return output


def optimize_ova_biases(models: list[BinaryModel], X: np.ndarray,
                        labels: np.ndarray, max_cycles: int = 10) -> tuple[list[BinaryModel], dict]:
    """Coordinate-exact multiclass training search within the three b strips."""
    if len(models) != 3 or any(model.negative_class is not None for model in models):
        raise ValueError("joint bias optimization requires exactly three OVA models")
    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    base_scores = np.column_stack([model.decision(X) for model in models])
    original_b = np.asarray([model.b for model in models], float)
    bounds = np.asarray([
        [float(model.diagnostics["strip_left"]) - model.b,
         float(model.diagnostics["strip_right"]) - model.b]
        for model in models
    ])
    classes = np.asarray([model.positive_class for model in models], int)

    def error(offsets: np.ndarray) -> float:
        prediction = classes[np.argmax(base_scores - offsets[None, :], axis=1)]
        return float(np.mean(prediction != labels))

    outcomes = []
    for order in itertools.permutations(range(3)):
        offsets = np.zeros(3, dtype=float)
        cycles = 0
        for cycle in range(max_cycles):
            changed = False
            for column in order:
                other = np.delete(base_scores - offsets[None, :], column, axis=1)
                breakpoints = base_scores[:, column] - np.max(other, axis=1)
                left, right = bounds[column]
                inside = np.unique(breakpoints[(breakpoints >= left) & (breakpoints <= right)])
                boundaries = np.unique(np.r_[left, inside, right])
                mids = ((boundaries[:-1] + boundaries[1:]) / 2.0
                        if len(boundaries) > 1 else np.empty(0))
                candidates = np.unique(np.r_[boundaries, mids, offsets[column]])
                current = offsets[column]
                trials = []
                for candidate in candidates:
                    trial = offsets.copy()
                    trial[column] = candidate
                    trials.append((error(trial), abs(candidate - current), candidate))
                chosen = min(trials)
                if chosen[0] <= error(offsets) and chosen[2] != current:
                    offsets[column] = chosen[2]
                    changed = True
            cycles = cycle + 1
            if not changed:
                break
        outcomes.append((error(offsets), float(np.linalg.norm(offsets)), tuple(offsets), cycles))
    best = min(outcomes, key=lambda row: (row[0], row[1], row[2]))
    calibrated = copy.deepcopy(models)
    for column, model in enumerate(calibrated):
        model.b = float(original_b[column] + best[2][column])
        model.diagnostics["b_before_joint_calibration"] = float(original_b[column])
        model.diagnostics["b"] = model.b
        model.diagnostics["bias_adjustment"] = float(best[2][column])
        model.diagnostics["joint_bias_training_error"] = float(best[0])
        model.diagnostics["joint_bias_cycles"] = int(best[3])
    return calibrated, {
        "joint_bias_training_error": float(best[0]),
        "bias_adjustments": list(best[2]), "cycles": int(best[3]),
        "orders_evaluated": 6,
    }


def _cv_score(X: np.ndarray, labels: np.ndarray, decomposition: str,
              threshold_mode: str, alpha_rule: str, nu: float | None,
              nu_grid: Iterable[float], rho: float, seed: int, phase: str,
              candidate: str, alpha_scales: Iterable[float] = (1.0,),
              weight_mode: str = "single",
              normalize_ova: bool = False) -> tuple[float, list[dict]]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=10_000 + int(seed))
    errors, diagnostics = [], []
    for fold, (fit_idx, validation_idx) in enumerate(cv.split(X, labels)):
        alpha = alpha_from_rule(X[fit_idx], alpha_rule)
        scales = tuple(float(value) for value in alpha_scales)
        if weight_mode == "centered-target-nnls":
            weights, weight_diag = centered_target_nnls_weights(
                X[fit_idx], labels[fit_idx], alpha, scales
            )
        elif weight_mode == "single":
            weights, weight_diag = np.array([1.0]), {}
        else:
            raise ValueError(f"unknown kernel weight mode {weight_mode!r}")
        models, fold_diag = fit_multiclass(
            X[fit_idx], labels[fit_idx], alpha, decomposition, threshold_mode,
            nu=nu, nu_grid=nu_grid, rho=rho,
            context={"phase": phase, "fold": fold, "candidate": candidate},
            alpha_scales=scales, kernel_weights=weights,
        )
        prediction = predict_multiclass(
            models, X[validation_idx], decomposition, normalize_ova=normalize_ova
        )
        error = float(np.mean(prediction != labels[validation_idx]))
        errors.append(error)
        for item in fold_diag:
            item["validation_error"] = error
            item.update(weight_diag)
        diagnostics.extend(fold_diag)
    return float(np.mean(errors)), diagnostics


def select_alpha_rule(X: np.ndarray, labels: np.ndarray, decomposition: str,
                      threshold_mode: str, seed: int,
                      rules: Iterable[str] = ALPHA_RULES,
                      normalize_ova: bool = False) -> tuple[str, dict, list[dict]]:
    outcomes, diagnostics = [], []
    for order, rule in enumerate(rules):
        error, rows = _cv_score(
            X, labels, decomposition, threshold_mode, rule, None,
            PAPER_NU_GRID, 0.0, seed, "inner_alpha", rule,
            normalize_ova=normalize_ova,
        )
        outcomes.append((error, order, rule))
        diagnostics.extend(rows)
    best = min(outcomes)
    return best[2], {rule: error for error, _, rule in outcomes}, diagnostics


def select_feature_subset(X: np.ndarray, labels: np.ndarray, seed: int,
                          subsets: Iterable[tuple[int, ...]] | None = None
                          ) -> tuple[tuple[int, ...], dict, list[dict]]:
    """Select raw coordinates using only inner multiclass validation error."""
    if subsets is None:
        subsets = itertools.chain.from_iterable(
            itertools.combinations(range(X.shape[1]), size)
            for size in range(1, X.shape[1] + 1)
        )
    outcomes, diagnostics = [], []
    for order, subset in enumerate(tuple(tuple(int(v) for v in s) for s in subsets)):
        if not subset or min(subset) < 0 or max(subset) >= X.shape[1]:
            raise ValueError(f"invalid feature subset {subset}")
        candidate = ",".join(str(value + 1) for value in subset)
        error, rows = _cv_score(
            X[:, subset], labels, "ova", "grid", "paper", None,
            PAPER_NU_GRID, 0.0, seed, "inner_feature_subset", candidate,
        )
        outcomes.append((error, len(subset), subset, order))
        diagnostics.extend(rows)
    best = min(outcomes)
    scores = {",".join(str(v + 1) for v in subset): error
              for error, _, subset, _ in outcomes}
    return best[2], scores, diagnostics


def select_nu(X: np.ndarray, labels: np.ndarray, decomposition: str,
              threshold_mode: str, alpha_rule: str, seed: int,
              grid: Iterable[float] = EXPANDED_NU_GRID,
              alpha_scales: Iterable[float] = (1.0,),
              weight_mode: str = "single",
              normalize_ova: bool = False) -> tuple[float, dict, list[dict]]:
    outcomes, diagnostics = [], []
    for order, value in enumerate(grid):
        candidate = f"{float(value):.12g}"
        error, rows = _cv_score(X, labels, decomposition, threshold_mode, alpha_rule,
                                float(value), (), 0.0, seed, "inner_nu", candidate,
                                alpha_scales, weight_mode, normalize_ova)
        outcomes.append((error, order, float(value)))
        diagnostics.extend(rows)
    best = min(outcomes)
    return best[2], {f"{value:.12g}": error for error, _, value in outcomes}, diagnostics


def select_rho(X: np.ndarray, labels: np.ndarray, decomposition: str,
               threshold_mode: str, alpha_rule: str, nu: float | None, seed: int,
               grid: Iterable[float] = COARSE_RHO_GRID,
               normalize_ova: bool = False,
               tie_break: str = "smallest") -> tuple[float, dict, list[dict]]:
    outcomes, diagnostics = [], []
    for order, value in enumerate(grid):
        candidate = f"{float(value):.12g}"
        error, rows = _cv_score(
            X, labels, decomposition, threshold_mode, alpha_rule,
            nu, PAPER_NU_GRID if nu is None else (), float(value), seed, "inner_rho", candidate,
            normalize_ova=normalize_ova,
        )
        outcomes.append((error, order, float(value)))
        diagnostics.extend(rows)
    if tie_break == "smallest":
        best = min(outcomes, key=lambda item: (item[0], item[1]))
    elif tie_break == "largest":
        best = min(outcomes, key=lambda item: (item[0], -item[2]))
    else:
        raise ValueError(f"unknown rho tie break {tie_break!r}")
    return best[2], {f"{value:.12g}": error for error, _, value in outcomes}, diagnostics


MODEL_SPECS = {
    "legacy_grid_ova": dict(decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False),
    "exact_b_ova": dict(decomposition="ova", threshold="exact", alpha_cv=False, nu_cv=False, robust=False),
    "alpha_cv_ova": dict(decomposition="ova", threshold="grid", alpha_cv=True, nu_cv=False, robust=False),
    "nu_cv_ova": dict(decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=True, robust=False),
    "nu_cv_wide_ova": dict(decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=True,
                             robust=False, nu_grid=WIDE_NU_GRID),
    "nu_cv_ova_robust_pinf": dict(decomposition="ova", threshold="grid", alpha_cv=False,
                                    nu_cv=True, robust=True),
    "nu_cv_ova_robust_tiemax_pinf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=True, robust=True,
        rho_tie_break="largest",
    ),
    "source_rbf_robust_cv_pinf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=True,
    ),
    "source_rbf_robust_cv_tiemax_pinf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=True,
        rho_tie_break="largest",
    ),
    "joint_nu_multiclass_train_rbf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        nu_selection="joint_multiclass_train",
    ),
    "joint_b_multiclass_train_rbf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        bias_selection="joint_multiclass_train",
    ),
    "standardized_rbf_source_nu_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        preprocessing="zscore_train_only",
    ),
    "feature_subset_cv_rbf_source_nu_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        feature_subset_cv=True,
    ),
    "expanded_per_class_nu_rbf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        outer_nu_grid=EXPANDED_NU_GRID,
    ),
    "wide_per_class_nu_rbf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        outer_nu_grid=WIDE_NU_GRID,
    ),
    "source_continued_per_class_nu_rbf": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        outer_nu_grid=SOURCE_CONTINUED_NU_GRID,
    ),
    "multiscale_nnls_nu_cv_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=True, robust=False,
        alpha_scales=(0.5, 1.0, 2.0), weight_mode="centered-target-nnls",
    ),
    "multiscale_nnls_source_nu_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        alpha_scales=(0.5, 1.0, 2.0), weight_mode="centered-target-nnls",
    ),
    "nu_cv_ova_norm": dict(decomposition="ova", threshold="grid", alpha_cv=False,
                             nu_cv=True, robust=False, normalize_ova=True),
    "fixed_nu_sqrt10_ova": dict(decomposition="ova", threshold="grid", alpha_cv=False,
                                  nu_cv=False, robust=False, fixed_nu=float(np.sqrt(10.0))),
    "matlab_poly2_grid_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=2, polynomial_offset=0.0,
    ),
    "matlab_poly2_exact_ova": dict(
        decomposition="ova", threshold="exact", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=2, polynomial_offset=0.0,
    ),
    "matlab_poly1_grid_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=1, polynomial_offset=0.0,
    ),
    "matlab_poly2_c1_grid_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=2, polynomial_offset=1.0,
    ),
    "matlab_poly3_grid_ova": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=3, polynomial_offset=0.0,
    ),
    "matlab_source_legacy_highs": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=2, polynomial_offset=0.0,
        solver_backend="scipy-highs",
    ),
    "matlab_source_legacy_clarabel": dict(
        decomposition="ova", threshold="grid", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=2, polynomial_offset=0.0,
        solver_backend="cvxpy-clarabel",
    ),
    "matlab_source_exact_b_highs": dict(
        decomposition="ova", threshold="exact", alpha_cv=False, nu_cv=False, robust=False,
        kernel_kind="poly", polynomial_degree=2, polynomial_offset=0.0,
        solver_backend="scipy-highs",
    ),
    "legacy_grid_ovo": dict(decomposition="ovo", threshold="grid", alpha_cv=False, nu_cv=False, robust=False),
    "v4_ovo": dict(decomposition="ovo", threshold="exact", alpha_cv=True, nu_cv=True, robust=False),
    "v4_ovo_robust_pinf": dict(decomposition="ovo", threshold="exact", alpha_cv=True, nu_cv=True, robust=True),
    "rkhs_l2_rbf_svc_cv_ovo": dict(
        estimator="sklearn_svc", decomposition="ovo", threshold="libsvm",
        alpha_cv=True, nu_cv=False, robust=False,
    ),
    "rkhs_l2_standardized_rbf_svc_cv_ovo": dict(
        estimator="sklearn_svc", decomposition="ovo", threshold="libsvm",
        alpha_cv=True, nu_cv=False, robust=False, preprocessing="zscore_train_only",
    ),
}


def _evaluate_rkhs_svc(model_name: str, spec: dict, X_train: np.ndarray,
                       y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                       seed: int) -> tuple[dict, list[dict]]:
    """Training-only model selection for the conventional RKHS-l2 RBF SVM."""
    started = time.perf_counter()
    preprocessing = spec.get("preprocessing", "none")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=20_000 + int(seed))
    outcomes, diagnostics = [], []
    for alpha_order, rule in enumerate(ALPHA_RULES):
        for c_order, C in enumerate(SVC_C_GRID):
            fold_errors = []
            for fold, (fit_idx, validation_idx) in enumerate(cv.split(X_train, y_train)):
                fit_X, validation_X = X_train[fit_idx], X_train[validation_idx]
                if preprocessing == "zscore_train_only":
                    fit_X, validation_X, _ = standardize_train_test(fit_X, validation_X)
                elif preprocessing != "none":
                    raise ValueError(f"unknown SVC preprocessing mode {preprocessing!r}")
                alpha = alpha_from_rule(fit_X, rule)
                classifier = SVC(
                    C=float(C), kernel="rbf", gamma=1.0 / (2.0 * alpha ** 2),
                    decision_function_shape="ovo", shrinking=True, tol=1e-9,
                )
                classifier.fit(fit_X, y_train[fit_idx])
                prediction = classifier.predict(validation_X)
                error = float(np.mean(prediction != y_train[validation_idx]))
                fold_errors.append(error)
                diagnostics.append({
                    "phase": "inner_svc", "fold": fold,
                    "candidate": f"alpha={rule};C={C:g}", "alpha_rule": rule,
                    "alpha": alpha, "C": float(C), "validation_error": error,
                    "solver": "libsvm-smo", "solver_status": "ok",
                    "solver_status_code": int(classifier.fit_status_),
                    "solver_iterations": int(np.sum(classifier.n_iter_)),
                    "solver_objective": np.nan, "solver_runtime_s": np.nan,
                    "selected_for_task": False, "trivial_u": False,
                })
            outcomes.append((float(np.mean(fold_errors)), alpha_order, c_order, rule, float(C)))
    best = min(outcomes)
    selected_rule, selected_C = best[3], best[4]
    fit_X, test_X = X_train, X_test
    preprocessing_diagnostics = {}
    if preprocessing == "zscore_train_only":
        fit_X, test_X, preprocessing_diagnostics = standardize_train_test(X_train, X_test)
    alpha = alpha_from_rule(fit_X, selected_rule)
    classifier = SVC(
        C=selected_C, kernel="rbf", gamma=1.0 / (2.0 * alpha ** 2),
        decision_function_shape="ovo", shrinking=True, tol=1e-9,
    )
    classifier.fit(fit_X, y_train)
    train_prediction = classifier.predict(fit_X)
    test_prediction = classifier.predict(test_X)
    recalls = recall_score(y_test, test_prediction, labels=[1, 2, 3], average=None,
                           zero_division=0)
    diagnostics.append({
        "phase": "outer_fit", "fold": -1, "candidate": "selected",
        "alpha_rule": selected_rule, "alpha": alpha, "C": selected_C,
        "validation_error": best[0], "solver": "libsvm-smo", "solver_status": "ok",
        "solver_status_code": int(classifier.fit_status_),
        "solver_iterations": int(np.sum(classifier.n_iter_)),
        "solver_objective": np.nan, "solver_runtime_s": np.nan,
        "selected_for_task": True, "trivial_u": False,
        "support_vectors": int(classifier.support_.size),
    })
    record = {
        "architecture": model_name, "seed": int(seed), "split_seed": int(seed),
        "decomposition": "ovo", "preprocessing": preprocessing,
        "preprocessing_diagnostics": preprocessing_diagnostics,
        "selected_feature_indices_1based": [1, 2, 3, 4], "feature_cv_scores": {},
        "kernel_kind": "rbf", "polynomial_degree": None, "polynomial_offset": None,
        "solver_backend": "libsvm-smo", "nu_selection": "not_applicable",
        "bias_selection": "libsvm", "score_normalization": "none",
        "threshold_mode": "libsvm", "alpha_rule": selected_rule,
        "selected_alpha": float(alpha), "selected_C": float(selected_C),
        "alpha_scales": [1.0], "kernel_weights": [1.0],
        "kernel_weight_diagnostics": {}, "alpha_cv_scores": {
            f"{rule}|C={C:g}": error for error, _, _, rule, C in outcomes
        },
        "nu_cv_scores": {}, "rho_cv_scores": {}, "selected_nu": None,
        "selected_nu_by_task": [], "outer_nu_grid": [], "selected_rho": 0.0,
        "rho": 0.0, "p": "none", "training_error": float(np.mean(train_prediction != y_train)),
        "test_error": float(np.mean(test_prediction != y_test)),
        "accuracy": float(np.mean(test_prediction == y_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_prediction)),
        "macro_f1": float(f1_score(y_test, test_prediction, average="macro")),
        "recall_class_1": float(recalls[0]), "recall_class_2": float(recalls[1]),
        "recall_class_3": float(recalls[2]), "prediction_vector": test_prediction.tolist(),
        "prediction_hash": hashlib.sha256(
            np.asarray(test_prediction, dtype="<i8").tobytes()
        ).hexdigest(),
        "solver_status": "ok" if classifier.fit_status_ == 0 else "failed",
        "solver_iterations": int(np.sum(classifier.n_iter_)), "solver_solve_count": 97,
        "trivial_outer_classifiers": 0, "train_exact_score_ties": -1,
        "test_exact_score_ties": -1, "support_vectors": int(classifier.support_.size),
        "model_runtime_s": time.perf_counter() - started,
    }
    return record, diagnostics


def evaluate_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[dict, list[dict]]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model {model_name!r}")
    spec = MODEL_SPECS[model_name]
    if spec.get("estimator") == "sklearn_svc":
        return _evaluate_rkhs_svc(
            model_name, spec, np.asarray(X_train, float), np.asarray(y_train, int),
            np.asarray(X_test, float), np.asarray(y_test, int), seed,
        )
    diagnostics: list[dict] = []
    selected_feature_indices = tuple(range(np.asarray(X_train).shape[1]))
    feature_cv_scores = {}
    if spec.get("feature_subset_cv", False):
        selected_feature_indices, feature_cv_scores, rows = select_feature_subset(
            np.asarray(X_train, float), np.asarray(y_train, int), seed
        )
        diagnostics.extend(rows)
        X_train = np.asarray(X_train)[:, selected_feature_indices]
        X_test = np.asarray(X_test)[:, selected_feature_indices]
    preprocessing = spec.get("preprocessing", "none")
    preprocessing_diagnostics = {}
    if preprocessing == "zscore_train_only":
        X_train, X_test, preprocessing_diagnostics = standardize_train_test(X_train, X_test)
    elif preprocessing != "none":
        raise ValueError(f"unknown preprocessing mode {preprocessing!r}")
    normalize_ova = bool(spec.get("normalize_ova", False))
    alpha_rule = "paper"
    alpha_cv_scores = {}
    if spec["alpha_cv"]:
        alpha_rule, alpha_cv_scores, rows = select_alpha_rule(
            X_train, y_train, spec["decomposition"], spec["threshold"], seed,
            normalize_ova=normalize_ova,
        )
        diagnostics.extend(rows)
    selected_nu = spec.get("fixed_nu")
    nu_cv_scores = {}
    alpha_scales = tuple(spec.get("alpha_scales", (1.0,)))
    weight_mode = spec.get("weight_mode", "single")
    kernel_kind = spec.get("kernel_kind", "rbf")
    polynomial_degree = int(spec.get("polynomial_degree", 2))
    polynomial_offset = float(spec.get("polynomial_offset", 0.0))
    solver_backend = spec.get("solver_backend", "scipy-highs")
    nu_selection = spec.get("nu_selection", "per_binary")
    bias_selection = spec.get("bias_selection", "per_binary")
    if spec["nu_cv"]:
        selected_nu, nu_cv_scores, rows = select_nu(
            X_train, y_train, spec["decomposition"], spec["threshold"], alpha_rule, seed,
            grid=spec.get("nu_grid", EXPANDED_NU_GRID),
            alpha_scales=alpha_scales, weight_mode=weight_mode,
            normalize_ova=normalize_ova,
        )
        diagnostics.extend(rows)
    selected_rho = 0.0
    rho_cv_scores = {}
    if spec["robust"]:
        selected_rho, rho_cv_scores, rows = select_rho(
            X_train, y_train, spec["decomposition"], spec["threshold"],
            alpha_rule, None if selected_nu is None else float(selected_nu), seed,
            normalize_ova=normalize_ova,
            tie_break=spec.get("rho_tie_break", "smallest"),
        )
        diagnostics.extend(rows)
    alpha = alpha_from_rule(X_train, alpha_rule) if kernel_kind == "rbf" else 1.0
    if weight_mode == "centered-target-nnls":
        kernel_weights, kernel_weight_diagnostics = centered_target_nnls_weights(
            X_train, y_train, alpha, alpha_scales
        )
    else:
        kernel_weights, kernel_weight_diagnostics = np.array([1.0]), {}
    models, rows = fit_multiclass(
        X_train, y_train, alpha, spec["decomposition"], spec["threshold"],
        nu=selected_nu, nu_grid=spec.get("outer_nu_grid", PAPER_NU_GRID), rho=selected_rho,
        context={"phase": "outer_fit", "fold": -1, "candidate": "selected"},
        alpha_scales=alpha_scales, kernel_weights=kernel_weights,
        kernel_kind=kernel_kind, polynomial_degree=polynomial_degree,
        polynomial_offset=polynomial_offset, solver_backend=solver_backend,
        nu_selection=nu_selection, bias_selection=bias_selection,
    )
    diagnostics.extend(rows)
    train_prediction = predict_multiclass(
        models, X_train, spec["decomposition"], normalize_ova=normalize_ova
    )
    test_prediction = predict_multiclass(
        models, X_test, spec["decomposition"], normalize_ova=normalize_ova
    )
    if spec["decomposition"] == "ova":
        raw_train_scores = np.column_stack([model.decision(X_train) for model in models])
        raw_test_scores = np.column_stack([model.decision(X_test) for model in models])
        train_score_ties = int(np.sum(
            np.sum(raw_train_scores == raw_train_scores.max(axis=1, keepdims=True), axis=1) > 1
        ))
        test_score_ties = int(np.sum(
            np.sum(raw_test_scores == raw_test_scores.max(axis=1, keepdims=True), axis=1) > 1
        ))
    else:
        train_score_ties = -1
        test_score_ties = -1
    recalls = recall_score(y_test, test_prediction, labels=[1, 2, 3], average=None,
                           zero_division=0)
    record = {
        "architecture": model_name,
        "seed": int(seed),
        "split_seed": int(seed),
        "decomposition": spec["decomposition"],
        "preprocessing": preprocessing,
        "preprocessing_diagnostics": preprocessing_diagnostics,
        "selected_feature_indices_1based": [value + 1 for value in selected_feature_indices],
        "feature_cv_scores": feature_cv_scores,
        "kernel_kind": kernel_kind,
        "polynomial_degree": polynomial_degree if kernel_kind == "poly" else None,
        "polynomial_offset": polynomial_offset if kernel_kind == "poly" else None,
        "solver_backend": solver_backend,
        "nu_selection": nu_selection,
        "bias_selection": bias_selection,
        "score_normalization": "rkhs_norm" if normalize_ova else "none",
        "threshold_mode": spec["threshold"],
        "alpha_rule": alpha_rule,
        "selected_alpha": float(alpha),
        "alpha_scales": list(alpha_scales),
        "kernel_weights": kernel_weights.tolist(),
        "kernel_weight_diagnostics": kernel_weight_diagnostics,
        "selected_nu": None if selected_nu is None else float(selected_nu),
        "selected_nu_by_task": [float(model.nu) for model in models],
        "outer_nu_grid": list(spec.get("outer_nu_grid", PAPER_NU_GRID)),
        "selected_rho": float(selected_rho),
        "rho_tie_break": spec.get("rho_tie_break", "smallest"),
        "p": "inf" if spec["robust"] else "none",
        "alpha_cv_scores": alpha_cv_scores,
        "nu_cv_scores": nu_cv_scores,
        "rho_cv_scores": rho_cv_scores,
        "training_error": float(np.mean(train_prediction != y_train)),
        "test_error": float(np.mean(test_prediction != y_test)),
        "accuracy": float(np.mean(test_prediction == y_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_prediction)),
        "macro_f1": float(f1_score(y_test, test_prediction, average="macro", zero_division=0)),
        "recall_class_1": float(recalls[0]),
        "recall_class_2": float(recalls[1]),
        "recall_class_3": float(recalls[2]),
        "prediction_vector": test_prediction.tolist(),
        "prediction_hash": hashlib.sha256(np.asarray(test_prediction, dtype="<i8").tobytes()).hexdigest(),
        "solver_status": "ok" if all(row["solver_status_code"] == 0 for row in diagnostics) else "failed",
        "solver_iterations": int(sum(row["solver_iterations"] for row in diagnostics)),
        "solver_solve_count": int(len(diagnostics)),
        "train_exact_score_ties": train_score_ties,
        "test_exact_score_ties": test_score_ties,
        "trivial_outer_classifiers": int(
            sum(
                row["trivial_u"]
                for row in rows
                if row["phase"] == "outer_fit" and row.get("selected_for_task", False)
            )
        ),
    }
    return record, diagnostics
