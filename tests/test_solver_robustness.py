"""
Validates the solver-selection fix in convex_subproblem.py:
  1. HIGHS is never in the SOCP chain (it can't solve second-order-cone
     problems -- including it would just waste an attempt on every
     DRO-enabled solve).
  2. A well-posed DRO-enabled (SOCP) problem reaches a clean "optimal"
     status using the new chain, on a small, well-conditioned synthetic
     example -- i.e. the fix actually produces "optimal", not
     "optimal_inaccurate", for an easy problem where there's no excuse
     for imprecision.

Requires cvxpy -- skipped automatically if unavailable.
"""

import pytest

cp = pytest.importorskip("cvxpy")
import numpy as np

from ddr_mksvm.optim.convex_subproblem import (
    solve_svm_dro,
    _LP_SOLVER_CHAIN,
    _SOCP_SOLVER_CHAIN,
)


def test_highs_excluded_from_socp_chain():
    assert cp.HIGHS not in _SOCP_SOLVER_CHAIN, (
        "HIGHS cannot solve SOCPs; including it in the DRO-enabled chain "
        "just wastes a doomed solver attempt on every call."
    )
    assert cp.HIGHS in _LP_SOLVER_CHAIN, "HIGHS should still be first for the plain LP (epsilon=0) case."


def test_dro_enabled_small_problem_reaches_optimal():
    rng = np.random.default_rng(21)
    m = 10
    n = 3
    X = rng.normal(size=(n, m))
    y = np.array([1.0] * 5 + [-1.0] * 5)
    K = X.T @ X + 5.0  # well-conditioned linear-ish kernel, easy problem

    sol = solve_svm_dro(K, y, nu=0.1, epsilon=0.05, L_theta_eta=1.0)

    assert sol is not None
    assert sol["used_dro"] is True
    assert sol["status"] == "optimal", (
        f"expected a clean 'optimal' status on an easy, well-conditioned SOCP, got {sol['status']!r} "
        "-- the fallback-past-optimal_inaccurate fix may not be working as intended."
    )
