"""R9 bounded-RBF class-sensitive robust q1 LP.

This is a new research candidate, not MATLAB executable parity.  It reuses
the audited source q1 LP/threshold semantics with one prospectively frozen RBF
kernel and does not contain scaling, SOC, or Gram-factorization machinery.
"""
from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np
from scipy.optimize import linprog

from ddr_mksvm.robust_matlab_parity import (
    NU_GRID,
    BinaryParityModel,
    KernelSpec,
    gram,
    matlab_binary_threshold,
    matlab_multiclass_threshold,
    uncertainty_delta,
)
from ddr_mksvm.robust_source_aligned import (
    FROZEN_P,
    FROZEN_RHO,
    FROZEN_TAU,
    build_weighted_q1_lp,
)

R9_P = FROZEN_P
R9_RHO = FROZEN_RHO
R9_TAU = FROZEN_TAU
SOLVER_METHOD = "highs"
SOLVER_OPTIONS = {"presolve": True, "time_limit": 600.0}
FEASIBILITY_TOLERANCE = 1.0e-7


def source_rbf_alpha(X: np.ndarray) -> float:
    """Commented-source bandwidth: maximum feature sample standard deviation."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
        raise ValueError("finite two-dimensional training data with >=2 rows required")
    alpha = float(np.max(np.std(X, axis=0, ddof=1)))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("source RBF alpha must be finite and positive")
    return alpha


def rbf_spec(X: np.ndarray) -> KernelSpec:
    return KernelSpec("rbf", alpha=source_rbf_alpha(X))


def rbf_uncertainty_delta(
    X: np.ndarray,
    original_labels: np.ndarray,
    rho: float,
    p: float = R9_P,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    kernel = KernelSpec("rbf", alpha=source_rbf_alpha(X) if alpha is None else float(alpha))
    return uncertainty_delta(
        X, original_labels, rho, p, kernel, matlab_semantics=True
    )


def build_r9_q1_lp(
    K: np.ndarray,
    y: np.ndarray,
    nu: float,
    delta: np.ndarray,
    *,
    tau: float = R9_TAU,
    class_sensitive: bool,
):
    """Build the audited q1 LP with optional R9 slack weights."""
    return build_weighted_q1_lp(
        K,
        y,
        nu,
        delta,
        tau=tau,
        class_sensitive=class_sensitive,
    )


def _positive_minimum(values: np.ndarray) -> float | None:
    values = np.abs(np.asarray(values, dtype=float)).ravel()
    values = values[values > 0]
    return None if values.size == 0 else float(np.min(values))


def _dynamic_range(values: np.ndarray) -> float | None:
    values = np.abs(np.asarray(values, dtype=float)).ravel()
    values = values[values > 0]
    return None if values.size == 0 else float(np.max(values) / np.min(values))


def _bound_violation(x: np.ndarray, bounds) -> float:
    violation = 0.0
    for value, (lower, upper) in zip(x, bounds):
        if lower is not None:
            violation = max(violation, float(lower - value))
        if upper is not None:
            violation = max(violation, float(value - upper))
    return max(0.0, violation)


def solve_r9_q1_lp(
    K: np.ndarray,
    y: np.ndarray,
    nu: float,
    delta: np.ndarray,
    *,
    tau: float = R9_TAU,
    class_sensitive: bool,
    solver: str = SOLVER_METHOD,
    solver_options: dict | None = None,
) -> tuple[dict, dict]:
    options = dict(SOLVER_OPTIONS if solver_options is None else solver_options)
    c, A, b, bounds, weights = build_r9_q1_lp(
        K,
        y,
        nu,
        delta,
        tau=tau,
        class_sensitive=class_sensitive,
    )
    started = time.perf_counter()
    result = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method=solver,
        options=options,
    )
    runtime = time.perf_counter() - started
    if not result.success or int(result.status) != 0:
        raise RuntimeError(f"HiGHS LP failed status={result.status}: {result.message}")

    x = np.asarray(result.x, dtype=float)
    m = len(y)
    if x.shape != (3 * m + 1,) or not np.isfinite(x).all():
        raise RuntimeError("R9 LP returned invalid primal values")
    u = x[:m]
    gamma = float(x[m])
    xi = x[m + 1 : 2 * m + 1]
    s = x[2 * m + 1 :]
    residual = b - A @ x
    inequality_violation = float(max(0.0, -np.min(residual)))
    bounds_violation = _bound_violation(x, bounds)
    maximum_violation = max(inequality_violation, bounds_violation)
    objective = float(c @ x)
    kernel_nonzero = np.abs(K[np.nonzero(K)])
    matrix_nonzero = np.abs(A[np.nonzero(A)])
    canonical_values = np.concatenate(
        (np.abs(A).ravel(), np.abs(b).ravel(), np.abs(c).ravel())
    )
    robust_scalar = float(np.sqrt(np.diag(K)) @ np.abs(u))
    values = {"u": u, "gamma": gamma, "xi": xi, "s": s}
    diagnostics = {
        "solver": f"scipy-linprog-{solver}",
        "solver_status": str(result.message),
        "solver_status_code": int(result.status),
        "solver_success": bool(result.success),
        "solver_options": options,
        "solver_iterations": int(result.nit),
        "crossover_iterations": int(getattr(result, "crossover_nit", 0)),
        "runtime_s": float(runtime),
        "objective_value": objective,
        "objective_solver_value": float(result.fun),
        "objective_abs_residual": float(abs(objective - float(result.fun))),
        "max_constraint_violation": float(maximum_violation),
        "max_inequality_violation": inequality_violation,
        "max_bound_violation": float(bounds_violation),
        "min_constraint_residual": float(np.min(residual)),
        "min_margin_residual": float(np.min(residual[:m])),
        "regularizer_epigraph": float(np.sum(s)),
        "weighted_slack_term": float(float(nu) * (weights @ xi)),
        "unweighted_sum_xi": float(np.sum(xi)),
        "u_l1": float(np.sum(np.abs(u))),
        "sum_s_minus_u_l1": float(np.sum(s) - np.sum(np.abs(u))),
        "robust_penalty_scalar": robust_scalar,
        "max_delta": float(np.max(delta)),
        "kernel_abs_max": float(np.max(np.abs(K))),
        "kernel_abs_min_positive": float(np.min(kernel_nonzero)),
        "kernel_zero_count": int(K.size - np.count_nonzero(K)),
        "kernel_dynamic_range": _dynamic_range(K),
        "constraint_abs_max": float(np.max(matrix_nonzero)),
        "constraint_abs_min_positive": float(np.min(matrix_nonzero)),
        "constraint_dynamic_range": _dynamic_range(A),
        "canonical_abs_max": float(np.max(canonical_values)),
        "canonical_abs_min_positive": _positive_minimum(canonical_values),
        "canonical_dynamic_range": _dynamic_range(canonical_values),
        "n_variables": int(c.size),
        "n_inequalities": int(A.shape[0]),
        "matrix_nnz": int(np.count_nonzero(A)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == -1)),
        "weight_positive": float(weights[y == 1][0]),
        "weight_negative": float(weights[y == -1][0]),
        "weight_mean": float(np.mean(weights)),
        "weight_sum": float(np.sum(weights)),
        "class_sensitive": bool(class_sensitive),
        "tau": float(tau) if class_sensitive else 0.0,
        "formulation": "r9_bounded_rbf_q1_lp",
        "uses_soc": False,
        "uses_gram_factorization": False,
        "uses_equilibration": False,
    }
    return values, diagnostics


def fit_binary(
    X: np.ndarray,
    y: np.ndarray,
    original_labels: np.ndarray,
    rho: float,
    p: float = R9_P,
    nu_grid: Iterable[float] = NU_GRID,
    *,
    multiclass_bug: bool = False,
    tau: float = R9_TAU,
    class_sensitive: bool,
) -> tuple[BinaryParityModel, list[dict]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    original_labels = np.asarray(original_labels)
    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("binary y must contain -1 and +1")
    kernel = rbf_spec(X)
    K = gram(X, None, kernel)
    delta = rbf_uncertainty_delta(
        X, original_labels, rho, p, alpha=kernel.alpha
    )
    threshold = matlab_multiclass_threshold if multiclass_bug else matlab_binary_threshold
    candidates = []
    diagnostics = []
    for order, nu in enumerate(nu_grid):
        values, diag = solve_r9_q1_lp(
            K,
            y,
            float(nu),
            delta,
            tau=tau,
            class_sensitive=class_sensitive,
        )
        intercept, training_error, threshold_diag = threshold(K, y, values, delta)
        diag.update(
            nu=float(nu),
            nu_order=int(order),
            alpha=float(kernel.alpha),
            gamma=float(values["gamma"]),
            b=float(intercept),
            training_error=float(training_error),
            rho=float(rho),
            p="inf" if math.isinf(p) else float(p),
            selected=False,
            **threshold_diag,
        )
        diagnostics.append(diag)
        candidates.append((training_error, order, values, intercept, diag))
    chosen = min(candidates, key=lambda item: (item[0], item[1]))
    chosen[4]["selected"] = True
    values, intercept = chosen[2], chosen[3]
    model = BinaryParityModel(
        X,
        y,
        values["u"],
        values["gamma"],
        values["xi"],
        float(intercept),
        delta,
        kernel,
        diagnostics=chosen[4],
    )
    return model, diagnostics


def fit_ova(
    X: np.ndarray,
    labels: np.ndarray,
    rho: float,
    p: float = R9_P,
    *,
    tau: float = R9_TAU,
    class_sensitive: bool,
) -> tuple[list[BinaryParityModel], list[dict]]:
    labels = np.asarray(labels)
    models = []
    diagnostics = []
    for task, label in enumerate(sorted(np.unique(labels).tolist())):
        binary = np.where(labels == label, 1.0, -1.0)
        model, rows = fit_binary(
            X,
            binary,
            labels,
            rho,
            p,
            multiclass_bug=True,
            tau=tau,
            class_sensitive=class_sensitive,
        )
        model.positive_class = int(label)
        for row in rows:
            row.update(task=int(task), positive_class=int(label))
        models.append(model)
        diagnostics.extend(rows)
    return models, diagnostics
