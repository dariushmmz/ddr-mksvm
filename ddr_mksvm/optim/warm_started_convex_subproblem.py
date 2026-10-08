"""Opt-in warm-started CVXPY template for the unchanged DDR q=2 nu grid."""

from contextvars import ContextVar
import warnings

import cvxpy as cp
import numpy as np

from ddr_mksvm.optim.convex_subproblem import (
    _SOCP_SOLVER_CHAIN,
    _SOLVER_KWARGS,
    _line_search_b,
    _psd_sqrt,
    _validate,
    solution_diagnostics,
)


_WARM_START_RECORDS = ContextVar("warm_started_grid_records", default=())


def clear_warm_start_records():
    """Clear nu-grid reuse diagnostics in the current execution context."""
    _WARM_START_RECORDS.set(())


def get_warm_start_records():
    """Return copies of nu-grid reuse diagnostics in the current context."""
    return [dict(record) for record in _WARM_START_RECORDS.get()]


def _solve_template_with_fallback(problem, variables, warm_start, verbose=True):
    best = None
    for solver in _SOCP_SOLVER_CHAIN:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                problem.solve(
                    solver=solver,
                    warm_start=warm_start,
                    **_SOLVER_KWARGS.get(solver, {}),
                )
        except Exception:
            continue
        if any(variable.value is None for variable in variables.values()):
            continue
        stats = problem.solver_stats
        snapshot = dict(
            status=problem.status,
            solver_name=stats.solver_name,
            num_iters=stats.num_iters,
            solve_time=stats.solve_time,
            primal_objective=float(problem.value),
            **{
                name: (
                    np.asarray(variable.value)
                    if np.ndim(variable.value)
                    else float(variable.value)
                )
                for name, variable in variables.items()
            },
        )
        if problem.status == "optimal":
            return snapshot
        if best is None and problem.status == "optimal_inaccurate":
            best = snapshot
    if best is not None and verbose:
        warnings.warn("accepting optimal_inaccurate solver result")
    return best


def _finalize_solution(K, M, y, raw, nu, epsilon, L_theta_eta):
    if (
        not np.isfinite(raw["u"]).all()
        or not np.isfinite(raw["xi"]).all()
        or not np.isfinite(raw["gamma"])
    ):
        raise RuntimeError("non-finite solver variables")

    u_value = np.asarray(raw["u"], dtype=float)
    values, vectors = np.linalg.eigh(M)
    keep = values > max(1.0, float(values.max())) * 1e-10
    u_value = vectors[:, keep] @ (vectors[:, keep].T @ u_value)

    gamma_value = float(raw["gamma"])
    raw_xi = np.asarray(raw["xi"], dtype=float)
    xi_value = np.maximum(0.0, 1.0 - (M @ u_value - y * gamma_value))
    xi_repair = np.abs(xi_value - raw_xi)
    b_value = _line_search_b(M, u_value, y, gamma_value, xi_value)
    solution = dict(
        u=u_value,
        gamma=gamma_value,
        b=b_value,
        xi=xi_value,
        status=raw["status"],
        solver_name=raw["solver_name"],
        num_iters=raw["num_iters"],
        solve_time=raw["solve_time"],
        primal_objective=raw["primal_objective"],
        used_dro=epsilon > 0 and L_theta_eta > 0,
        formulation="ddr_q2",
        raw_xi_min=float(raw_xi.min()),
        xi_repair_max=float(xi_repair.max()),
    )
    solution["diagnostics"] = solution_diagnostics(
        K,
        y,
        solution,
        nu,
        epsilon,
        L_theta_eta,
        formulation="ddr_q2",
    )
    solution["training_error"] = solution["diagnostics"]["training_error"]
    return solution


def train_with_nu_search_warm_started(
    K,
    y,
    nu_grid,
    epsilon=0.0,
    L_theta_eta=0.0,
    formulation="ddr_q2",
    verbose=True,
):
    """Evaluate the unchanged nu grid using one reusable CVXPY problem."""
    if formulation != "ddr_q2":
        raise ValueError("warm-started grid reuse is restricted to the ddr_q2 path")
    grid = np.asarray(nu_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all() or np.min(grid) < 0:
        raise ValueError("nu_grid must be a non-empty finite non-negative vector")
    K, y = _validate(K, y, float(grid[0]), epsilon, L_theta_eta)
    sample_count = len(y)
    M = np.outer(y, y) * K
    H = _psd_sqrt(M)

    u = cp.Variable(sample_count)
    gamma = cp.Variable()
    xi = cp.Variable(sample_count)
    nu_parameter = cp.Parameter(nonneg=True)
    constraints = [M @ u - y * gamma + xi >= 1, xi >= 0]
    objective = cp.sum(xi) / sample_count + nu_parameter * cp.sum_squares(H @ u)
    if epsilon > 0 and L_theta_eta > 0:
        objective += epsilon * L_theta_eta * cp.norm(H @ u, 2)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    variables = dict(u=u, gamma=gamma, xi=xi)

    best = None
    candidates = []
    solver_names = []
    for index, nu in enumerate(grid):
        nu_parameter.value = float(nu)
        raw = _solve_template_with_fallback(
            problem,
            variables,
            warm_start=index > 0,
            verbose=verbose,
        )
        if raw is None:
            continue
        solution = _finalize_solution(
            K,
            M,
            y,
            raw,
            float(nu),
            epsilon,
            L_theta_eta,
        )
        diagnostics = solution["diagnostics"]
        candidate = dict(
            nu=float(nu),
            status=solution["status"],
            solver_name=solution["solver_name"],
            solution=solution,
            **diagnostics,
        )
        candidates.append(candidate)
        solver_names.append(solution["solver_name"])
        if verbose:
            print(
                f"    [SVM warm] nu={nu:.6g} status={solution['status']} "
                f"error={diagnostics['training_error']:.6f} "
                f"objective={diagnostics['objective']:.6g}"
            )
        key = (diagnostics["training_error"], diagnostics["objective"], float(nu))
        if best is None or key < best[0]:
            best = (key, solution, float(nu))

    record = dict(
        grid_size=int(grid.size),
        completed_candidates=len(candidates),
        cold_starts=(1 if candidates else 0),
        warm_starts=max(0, len(candidates) - 1),
        solver_names=tuple(solver_names),
    )
    _WARM_START_RECORDS.set(_WARM_START_RECORDS.get() + (record,))
    if best is None:
        return None
    result = best[1]
    result["selected_nu"] = best[2]
    result["_nu_candidates"] = candidates
    result["warm_start_grid"] = dict(record)
    return result
