"""R8.1 exact-coordinate solver for the source-aligned q=1 models.

The mathematical LP is constructed by :mod:`robust_source_aligned` first and
is not modified.  A positive diagonal row/variable transformation is then
applied solely to the representation passed to HiGHS.
"""
from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np
from scipy.optimize import linprog

from ddr_mksvm.lp_equilibration import (
    bound_violation,
    equilibrate_lp,
    positive_dynamic_range,
    to_original_coordinates,
)
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
    FROZEN_TAU,
    SOLVER_METHOD,
    SOLVER_OPTIONS,
    build_weighted_q1_lp,
)


def solve_equilibrated_q1_lp(
    K: np.ndarray,
    y: np.ndarray,
    nu: float,
    delta: np.ndarray,
    *,
    tau: float = FROZEN_TAU,
    class_sensitive: bool,
    solver: str = SOLVER_METHOD,
    solver_options: dict | None = None,
) -> tuple[dict, dict]:
    """Solve the exact q1 LP in frozen positive-diagonal coordinates."""
    options = dict(SOLVER_OPTIONS if solver_options is None else solver_options)
    c, A, b, bounds, weights = build_weighted_q1_lp(
        K, y, nu, delta, tau=tau, class_sensitive=class_sensitive
    )
    transformed = equilibrate_lp(c, A, b, bounds)

    started = time.perf_counter()
    result = linprog(
        transformed.c,
        A_ub=transformed.A,
        b_ub=transformed.b,
        bounds=transformed.bounds,
        method=solver,
        options=options,
    )
    elapsed = time.perf_counter() - started
    if not result.success or int(result.status) != 0:
        raise RuntimeError(f"HiGHS LP failed status={result.status}: {result.message}")

    z = np.asarray(result.x, dtype=float)
    x = to_original_coordinates(z, transformed.variable_scale)
    m = len(y)
    if x.shape != (3 * m + 1,) or np.any(~np.isfinite(x)):
        raise RuntimeError("mapped LP primal is nonfinite or incorrectly shaped")
    u = x[:m]
    gamma = float(x[m])
    xi = x[m + 1 : 2 * m + 1]
    s = x[2 * m + 1 :]
    original_slack = b - A @ x
    scaled_slack = transformed.b - transformed.A @ z
    original_inequality_violation = float(max(0.0, -np.min(original_slack)))
    original_bound_violation = bound_violation(x, bounds)
    original_max_violation = max(
        original_inequality_violation, original_bound_violation
    )
    scaled_inequality_violation = float(max(0.0, -np.min(scaled_slack)))
    scaled_bound_violation = bound_violation(z, transformed.bounds)
    original_objective = float(c @ x)
    scaled_objective = float(transformed.c @ z)
    robust_scalar = float(np.sqrt(np.clip(np.diag(K), 0.0, None)) @ np.abs(u))
    ineqlin = getattr(result, "ineqlin", None)
    marginals = None if ineqlin is None else getattr(ineqlin, "marginals", None)

    values = {"u": u, "gamma": gamma, "xi": xi, "s": s}
    diagnostics = {
        "solver": f"scipy-linprog-{solver}",
        "solver_status": str(result.message),
        "solver_status_code": int(result.status),
        "solver_success": bool(result.success),
        "solver_options": dict(options),
        "primal_status": "optimal",
        "dual_status": "available" if marginals is not None else "unavailable",
        "solver_iterations": int(result.nit),
        "crossover_iterations": int(getattr(result, "crossover_nit", 0)),
        "runtime_s": float(elapsed),
        "objective_value": original_objective,
        "objective_scaled_solver": float(result.fun),
        "objective_scaled_recomputed": scaled_objective,
        "objective_coordinate_abs_residual": float(abs(original_objective - scaled_objective)),
        "objective_solver_abs_residual": float(abs(float(result.fun) - scaled_objective)),
        "regularizer_epigraph": float(np.sum(s)),
        "weighted_slack_term": float(float(nu) * (weights @ xi)),
        "unweighted_sum_xi": float(np.sum(xi)),
        "min_constraint_residual": float(np.min(original_slack)),
        "max_constraint_violation": float(original_max_violation),
        "max_original_inequality_violation": original_inequality_violation,
        "max_original_bound_violation": float(original_bound_violation),
        "max_scaled_inequality_violation": scaled_inequality_violation,
        "max_scaled_bound_violation": float(scaled_bound_violation),
        "min_margin_residual": float(np.min(original_slack[:m])),
        "u_l1": float(np.sum(np.abs(u))),
        "sum_s_minus_u_l1": float(np.sum(s) - np.sum(np.abs(u))),
        "robust_penalty_scalar": robust_scalar,
        "max_delta": float(np.max(delta)),
        "n_variables": int(c.size),
        "n_inequalities": int(A.shape[0]),
        "matrix_nnz": int(np.count_nonzero(A)),
        "unscaled_constraint_dynamic_range": positive_dynamic_range(A),
        "scaled_constraint_dynamic_range": positive_dynamic_range(transformed.A),
        "unscaled_canonical_dynamic_range": positive_dynamic_range(A, b, c),
        "scaled_canonical_dynamic_range": positive_dynamic_range(
            transformed.A, transformed.b, transformed.c
        ),
        "row_scale_min": float(np.min(transformed.row_scale)),
        "row_scale_max": float(np.max(transformed.row_scale)),
        "variable_scale_min": float(np.min(transformed.variable_scale)),
        "variable_scale_max": float(np.max(transformed.variable_scale)),
        "equilibration_passes": int(transformed.passes),
        "equilibration_update_min": float(transformed.update_min),
        "equilibration_update_max": float(transformed.update_max),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == -1)),
        "weight_positive": float(weights[y == 1][0]),
        "weight_negative": float(weights[y == -1][0]),
        "weight_mean": float(np.mean(weights)),
        "weight_sum": float(np.sum(weights)),
        "tau": float(tau) if class_sensitive else 0.0,
        "class_sensitive": bool(class_sensitive),
        "formulation": "source_q1_lp_exact_diagonal_coordinates",
        "uses_soc": False,
        "uses_gram_factorization": False,
        "uses_exact_lp_equilibration": True,
    }
    return values, diagnostics


def fit_binary(
    X: np.ndarray,
    y: np.ndarray,
    original_labels: np.ndarray,
    kernel: KernelSpec,
    rho: float,
    p: float,
    nu_grid: Iterable[float] = NU_GRID,
    multiclass_bug: bool = False,
    solver: str = SOLVER_METHOD,
    tau: float = FROZEN_TAU,
    class_sensitive: bool = True,
) -> tuple[BinaryParityModel, list[dict]]:
    """Fit the source model with unchanged selection/threshold semantics."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    original_labels = np.asarray(original_labels)
    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("binary y must contain -1 and +1")
    K = gram(X, None, kernel)
    delta = uncertainty_delta(X, original_labels, rho, p, kernel, matlab_semantics=True)
    threshold = matlab_multiclass_threshold if multiclass_bug else matlab_binary_threshold
    candidates = []
    diagnostics = []
    for order, nu in enumerate(nu_grid):
        values, diag = solve_equilibrated_q1_lp(
            K,
            y,
            float(nu),
            delta,
            tau=tau,
            class_sensitive=class_sensitive,
            solver=solver,
        )
        intercept, training_error, threshold_diag = threshold(K, y, values, delta)
        diag.update(
            nu=float(nu),
            nu_order=int(order),
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
    chosen = min(candidates, key=lambda row: (row[0], row[1]))
    chosen[4]["selected"] = True
    values, intercept = chosen[2], chosen[3]
    return (
        BinaryParityModel(
            X,
            y,
            values["u"],
            values["gamma"],
            values["xi"],
            float(intercept),
            delta,
            kernel,
            diagnostics=chosen[4],
        ),
        diagnostics,
    )
