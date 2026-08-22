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
import ddr_mksvm.optim.convex_subproblem as convex_subproblem
from ddr_mksvm.optim.alternating_trainer import _numpy_gram


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


def test_inaccurate_negative_slack_is_reconstructed_from_margin(monkeypatch):
    """A finite inaccurate solution must not kill every parallel run."""
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    K = np.eye(4)

    def inaccurate_result(*args, **kwargs):
        return {
            "status": "optimal_inaccurate",
            "solver_name": "synthetic",
            "num_iters": 100,
            "solve_time": 0.01,
            "primal_objective": -0.25,
            "u": np.zeros(4),
            "gamma": 0.0,
            "xi": np.full(4, -0.25),
        }

    monkeypatch.setattr(convex_subproblem, "_solve_with_fallback", inaccurate_result)
    sol = solve_svm_dro(K, y, nu=0.1, epsilon=0.01, L_theta_eta=1.0)

    np.testing.assert_allclose(sol["xi"], np.ones(4))
    assert sol["raw_xi_min"] == pytest.approx(-0.25)
    assert sol["xi_repair_max"] == pytest.approx(1.25)
    margins = np.outer(y, y) * K @ sol["u"] - y * sol["gamma"] + sol["xi"]
    assert np.min(margins) >= 1.0 - 1e-12


def test_learned_cubic_gram_is_accumulated_in_float64_and_remains_psd():
    # Mimics float32 features emitted by the DNN for the 561-point blood
    # transfusion training split.  The Gram builder must promote before its
    # reductions, not after float32 round-off has already occurred.
    rng = np.random.default_rng(45)
    Z = rng.normal(scale=0.25, size=(4, 561)).astype(np.float32)

    K = _numpy_gram({"kind": "poly", "degree": 3, "c": 1.0}, Z)

    assert K.dtype == np.float64
    scale = max(1.0, float(np.abs(K).max()))
    assert np.linalg.eigvalsh(K).min() >= -1e-10 * scale
