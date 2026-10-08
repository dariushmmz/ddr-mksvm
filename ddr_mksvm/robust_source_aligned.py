"""R8 source-aligned class-sensitive robust q=1 LP.

This module is intentionally a thin extension of ``robust_matlab_parity``.
Only the slack objective coefficients change when class sensitivity is on.
The source Gram matrix, uncertainty radii, constraints, nu selection,
threshold search, OVA construction, and prediction semantics are reused.
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
    build_q1_lp,
    fit_binary as fit_source_binary,
    fit_ova as fit_source_ova,
    gram,
    matlab_binary_threshold,
    matlab_multiclass_threshold,
    uncertainty_delta,
)

FROZEN_TAU = 0.5
FROZEN_P = 2.0
FROZEN_RHO = 0.01
SOLVER_METHOD = "highs"
SOLVER_OPTIONS = {"presolve": True, "time_limit": 600.0}
FEASIBILITY_TOLERANCE = 1e-6


def sign_group_weights(y: np.ndarray, tau: float = FROZEN_TAU) -> np.ndarray:
    """Return mean-one weights for the binary sign groups.

    ``tau=0`` is constructed explicitly as ones so the disabled reduction is
    bitwise exact rather than dependent on floating-point exponentiation.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("y must be one-dimensional and contain -1 and +1")
    if not np.isfinite(tau) or tau < 0 or tau > 1:
        raise ValueError("tau must be finite and in [0,1]")
    if tau == 0:
        return np.ones(y.size, dtype=float)
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == -1.0))
    denominator = n_pos ** (1.0 - tau) + n_neg ** (1.0 - tau)
    group = {
        1.0: y.size * n_pos ** (-tau) / denominator,
        -1.0: y.size * n_neg ** (-tau) / denominator,
    }
    weights = np.where(y == 1.0, group[1.0], group[-1.0])
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("invalid class-sensitive weights")
    return weights


def build_weighted_q1_lp(
    K: np.ndarray,
    y: np.ndarray,
    nu: float,
    delta: np.ndarray,
    tau: float = FROZEN_TAU,
    class_sensitive: bool = True,
):
    """Build the source LP with only its slack costs optionally weighted."""
    c, A, b, bounds = build_q1_lp(K, y, nu, delta)
    m = len(y)
    weights = sign_group_weights(y, tau) if class_sensitive else np.ones(m)
    if class_sensitive:
        c = c.copy()
        c[m + 1 : 2 * m + 1] = float(nu) * weights
    return c, A, b, bounds, weights


def _positive_minimum(values: np.ndarray) -> float | None:
    positive = np.abs(np.asarray(values, dtype=float))
    positive = positive[positive > 0]
    return None if positive.size == 0 else float(np.min(positive))


def solve_weighted_q1_lp(
    K: np.ndarray,
    y: np.ndarray,
    nu: float,
    delta: np.ndarray,
    tau: float = FROZEN_TAU,
    solver: str = SOLVER_METHOD,
    solver_options: dict | None = None,
) -> tuple[dict, dict]:
    """Solve one weighted source LP and recompute primal feasibility."""
    options = dict(SOLVER_OPTIONS if solver_options is None else solver_options)
    c, A, b, bounds, weights = build_weighted_q1_lp(
        K, y, nu, delta, tau=tau, class_sensitive=True
    )
    started = time.perf_counter()
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method=solver, options=options)
    elapsed = time.perf_counter() - started
    if not result.success or int(result.status) != 0:
        raise RuntimeError(
            f"HiGHS LP failed status={result.status}: {result.message}"
        )

    m = len(y)
    z = np.asarray(result.x, dtype=float)
    if z.shape != (3 * m + 1,) or np.any(~np.isfinite(z)):
        raise RuntimeError("LP returned a nonfinite or incorrectly shaped primal")
    u = z[:m]
    gamma = float(z[m])
    xi = z[m + 1 : 2 * m + 1]
    s = z[2 * m + 1 :]
    residual = np.asarray(b - A @ z, dtype=float)
    bound_violation = max(
        0.0,
        float(-np.min(xi)),
        float(-np.min(s)),
    )
    inequality_violation = float(max(0.0, -np.min(residual)))
    max_violation = max(bound_violation, inequality_violation)
    recomputed_objective = float(np.sum(s) + float(nu) * (weights @ xi))
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
        "objective_value": float(result.fun),
        "objective_recomputed": recomputed_objective,
        "objective_abs_residual": float(abs(float(result.fun) - recomputed_objective)),
        "regularizer_epigraph": float(np.sum(s)),
        "weighted_slack_term": float(float(nu) * (weights @ xi)),
        "unweighted_sum_xi": float(np.sum(xi)),
        "min_constraint_residual": float(np.min(residual)),
        "max_constraint_violation": max_violation,
        "max_inequality_violation": inequality_violation,
        "max_bound_violation": bound_violation,
        "min_margin_residual": float(np.min(residual[:m])),
        "u_l1": float(np.sum(np.abs(u))),
        "sum_s_minus_u_l1": float(np.sum(s) - np.sum(np.abs(u))),
        "robust_penalty_scalar": robust_scalar,
        "max_delta": float(np.max(delta)),
        "n_variables": int(c.size),
        "n_inequalities": int(A.shape[0]),
        "matrix_rows": int(A.shape[0]),
        "matrix_cols": int(A.shape[1]),
        "matrix_nnz": int(np.count_nonzero(A)),
        "matrix_density": float(np.count_nonzero(A) / A.size),
        "objective_abs_max": float(np.max(np.abs(c))),
        "objective_abs_min_positive": _positive_minimum(c),
        "constraint_abs_max": float(np.max(np.abs(A))),
        "constraint_abs_min_positive": _positive_minimum(A),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == -1)),
        "weight_positive": float(weights[y == 1][0]),
        "weight_negative": float(weights[y == -1][0]),
        "weight_mean": float(np.mean(weights)),
        "weight_sum": float(np.sum(weights)),
        "tau": float(tau),
        "dual_marginal_abs_max": (
            None if marginals is None else float(np.max(np.abs(marginals)))
        ),
        "formulation": "source_q1_weighted_lp",
        "cone_dimensions": None,
        "uses_soc": False,
        "uses_gram_factorization": False,
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
    """Fit one task, dispatching exactly to source parity when disabled."""
    if not class_sensitive:
        return fit_source_binary(
            X,
            y,
            original_labels,
            kernel,
            rho,
            p,
            nu_grid=nu_grid,
            multiclass_bug=multiclass_bug,
            solver=solver,
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    original_labels = np.asarray(original_labels)
    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("binary y must contain -1 and +1")
    K = gram(X, None, kernel)
    delta = uncertainty_delta(X, original_labels, rho, p, kernel, matlab_semantics=True)
    threshold = matlab_multiclass_threshold if multiclass_bug else matlab_binary_threshold
    candidates: list[tuple] = []
    diagnostics: list[dict] = []
    for order, nu in enumerate(nu_grid):
        values, diag = solve_weighted_q1_lp(K, y, float(nu), delta, tau=tau, solver=solver)
        b, training_error, threshold_diag = threshold(K, y, values, delta)
        diag.update(
            nu=float(nu),
            nu_order=int(order),
            gamma=float(values["gamma"]),
            b=float(b),
            training_error=float(training_error),
            rho=float(rho),
            p="inf" if math.isinf(p) else float(p),
            selected=False,
            **threshold_diag,
        )
        diagnostics.append(diag)
        candidates.append((training_error, order, values, b, diag))
    chosen = min(candidates, key=lambda row: (row[0], row[1]))
    chosen[4]["selected"] = True
    values, b = chosen[2], chosen[3]
    model = BinaryParityModel(
        X,
        y,
        values["u"],
        values["gamma"],
        values["xi"],
        float(b),
        delta,
        kernel,
        diagnostics=chosen[4],
    )
    return model, diagnostics


def fit_ova(
    X: np.ndarray,
    labels: np.ndarray,
    kernel: KernelSpec,
    rho: float,
    p: float,
    solver: str = SOLVER_METHOD,
    tau: float = FROZEN_TAU,
    class_sensitive: bool = True,
) -> tuple[list[BinaryParityModel], list[dict]]:
    """Fit source-order OVA tasks with sign-group weights."""
    if not class_sensitive:
        return fit_source_ova(X, labels, kernel, rho, p, solver=solver)
    labels = np.asarray(labels)
    models: list[BinaryParityModel] = []
    diagnostics: list[dict] = []
    for task, label in enumerate(sorted(np.unique(labels).tolist())):
        y = np.where(labels == label, 1.0, -1.0)
        model, rows = fit_binary(
            X,
            y,
            labels,
            kernel,
            rho,
            p,
            multiclass_bug=True,
            solver=solver,
            tau=tau,
            class_sensitive=True,
        )
        model.positive_class = int(label)
        for row in rows:
            row.update(task=int(task), positive_class=int(label))
        models.append(model)
        diagnostics.extend(rows)
    return models, diagnostics
