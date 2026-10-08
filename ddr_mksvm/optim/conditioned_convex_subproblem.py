"""Opt-in condition-number control for learned-kernel SOCP solves.

This module deliberately wraps, rather than changes, the frozen convex
subproblem implementation.  It applies the smallest scale-aware diagonal
shift that bounds a positive-semidefinite Gram matrix's condition number,
then delegates to the existing solver and nu-search paths unchanged.
"""

from __future__ import annotations

from contextvars import ContextVar

import numpy as np

from ddr_mksvm.optim import convex_subproblem


DEFAULT_CONDITION_NUMBER_CAP = 1e8
_CONDITIONING_RECORDS = ContextVar("conditioned_socp_records", default=())


def clear_conditioning_records():
    """Clear diagnostics accumulated in the current execution context."""
    _CONDITIONING_RECORDS.set(())


def get_conditioning_records():
    """Return copies of all conditioned nu-search diagnostics in this context."""
    return [dict(record) for record in _CONDITIONING_RECORDS.get()]


def _record_conditioning(result, diagnostics):
    candidates = result.get("_nu_candidates", [])
    statuses = [candidate["solution"]["status"] for candidate in candidates]
    solve_times = [candidate["solution"]["solve_time"] for candidate in candidates]
    record = dict(diagnostics)
    record.update(
        selected_nu=float(result["selected_nu"]),
        selected_status=result["status"],
        selected_solver=result["solver_name"],
        selected_solve_time=result["solve_time"],
        candidate_count=len(candidates),
        optimal_count=statuses.count("optimal"),
        optimal_inaccurate_count=statuses.count("optimal_inaccurate"),
        reported_solver_time=sum(value for value in solve_times if value is not None),
    )
    _CONDITIONING_RECORDS.set(_CONDITIONING_RECORDS.get() + (record,))


def condition_gram_matrix(K, condition_number_cap=DEFAULT_CONDITION_NUMBER_CAP):
    """Return ``K + delta I`` with condition number at most the given cap.

    Small negative eigenvalues attributable to round-off are clipped as in
    the existing solver validation.  Materially indefinite and zero Gram
    matrices remain errors rather than being hidden by numerical jitter.
    """
    K = np.asarray(K, dtype=float)
    cap = float(condition_number_cap)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"expected a square Gram matrix; got K={K.shape}")
    if not np.isfinite(K).all():
        raise ValueError("K contains NaN or Inf")
    if not np.isfinite(cap) or cap <= 1.0:
        raise ValueError("condition_number_cap must be finite and greater than 1")

    scale = max(1.0, float(np.abs(K).max()))
    if np.abs(K - K.T).max() > 1e-7 * scale:
        raise ValueError("K is not symmetric")
    K_symmetric = (K + K.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(K_symmetric)
    original_min = float(eigenvalues[0])
    original_max = float(eigenvalues[-1])
    if original_min < -1e-5 * scale:
        raise ValueError(f"K is not PSD: min eigenvalue {original_min:.3e}")

    clipped = np.clip(eigenvalues, 0.0, None)
    lambda_min = float(clipped[0])
    lambda_max = float(clipped[-1])
    if lambda_max <= 0.0:
        raise ValueError("cannot condition a zero Gram matrix")

    if original_min < 0.0:
        K_psd = (eigenvectors * clipped) @ eigenvectors.T
        K_psd = (K_psd + K_psd.T) / 2.0
    else:
        K_psd = K_symmetric.copy()

    diagonal_shift = max(
        0.0,
        (lambda_max - cap * lambda_min) / (cap - 1.0),
    )
    if diagonal_shift > 0.0:
        K_conditioned = K_psd + diagonal_shift * np.eye(K.shape[0])
    else:
        K_conditioned = K_psd

    conditioned_min = lambda_min + diagonal_shift
    conditioned_max = lambda_max + diagonal_shift
    original_condition = np.inf if lambda_min == 0.0 else lambda_max / lambda_min
    conditioned_condition = conditioned_max / conditioned_min
    diagnostics = {
        "condition_number_cap": cap,
        "original_lambda_min": original_min,
        "original_lambda_max": original_max,
        "original_condition_number": float(original_condition),
        "diagonal_shift": float(diagonal_shift),
        "conditioned_lambda_min": float(conditioned_min),
        "conditioned_lambda_max": float(conditioned_max),
        "conditioned_condition_number": float(conditioned_condition),
    }
    return K_conditioned, diagnostics


def _attach_conditioning_diagnostics(solution, diagnostics):
    if solution is not None:
        solution["gram_conditioning"] = dict(diagnostics)
    return solution


def solve_svm_dro_conditioned(
    K,
    y,
    nu,
    epsilon=0.0,
    L_theta_eta=0.0,
    formulation="ddr_q2",
    condition_number_cap=DEFAULT_CONDITION_NUMBER_CAP,
):
    """Condition one learned Gram matrix, then use the frozen solver path."""
    if formulation != "ddr_q2":
        raise ValueError("Gram conditioning is restricted to the ddr_q2 SOCP path")
    conditioned_K, diagnostics = condition_gram_matrix(K, condition_number_cap)
    solution = convex_subproblem.solve_svm_dro(
        conditioned_K,
        y,
        nu,
        epsilon=epsilon,
        L_theta_eta=L_theta_eta,
        formulation=formulation,
    )
    return _attach_conditioning_diagnostics(solution, diagnostics)


def train_with_nu_search_conditioned(
    K,
    y,
    nu_grid,
    epsilon=0.0,
    L_theta_eta=0.0,
    formulation="ddr_q2",
    verbose=True,
    condition_number_cap=DEFAULT_CONDITION_NUMBER_CAP,
):
    """Condition once, then run the existing nu-grid search unchanged."""
    if formulation != "ddr_q2":
        raise ValueError("Gram conditioning is restricted to the ddr_q2 SOCP path")
    conditioned_K, diagnostics = condition_gram_matrix(K, condition_number_cap)
    result = convex_subproblem.train_with_nu_search(
        conditioned_K,
        y,
        nu_grid,
        epsilon=epsilon,
        L_theta_eta=L_theta_eta,
        formulation=formulation,
        verbose=verbose,
    )
    if result is None:
        return None
    _attach_conditioning_diagnostics(result, diagnostics)
    for candidate in result.get("_nu_candidates", []):
        _attach_conditioning_diagnostics(candidate.get("solution"), diagnostics)
    _record_conditioning(result, diagnostics)
    return result
