"""Frozen solver settings used by the solver-qualification study."""
from __future__ import annotations

from .feasibility import FeasibilityThresholds
from .policy import SolverAttempt, SolverPolicy


def qualification_attempts() -> tuple[SolverAttempt, ...]:
    """Return the prospectively declared SQ attempt order.

    The first attempt is byte-for-byte equivalent to the stopped gate's
    CVXPY call. The second changes only numerical termination tolerances. The
    third uses an alternative SOCP backend while leaving the CVXPY problem
    unchanged.
    """

    return (
        SolverAttempt(
            name="clarabel_strict_original",
            solver="CLARABEL",
            options={"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8},
        ),
        SolverAttempt(
            name="clarabel_relaxed_1e-7",
            solver="CLARABEL",
            options={"tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7, "tol_feas": 1e-7},
        ),
        SolverAttempt(
            name="scs_1e-6",
            solver="SCS",
            options={
                "eps_abs": 1e-6,
                "eps_rel": 1e-6,
                "max_iters": 100000,
                "normalize": True,
                "scale": 1.0,
                "acceleration_lookback": 10,
            },
        ),
        SolverAttempt(
            name="clarabel_qdldl_strict",
            solver="CLARABEL",
            options={
                "tol_gap_abs": 1e-8,
                "tol_gap_rel": 1e-8,
                "tol_feas": 1e-8,
                "direct_solve_method": "qdldl",
                "max_threads": 1,
            },
        ),
        SolverAttempt(
            name="cvxopt_high_accuracy_robust",
            solver="CVXOPT",
            options={
                "abstol": 1e-9,
                "reltol": 1e-9,
                "feastol": 1e-9,
                "max_iters": 500,
                "refinement": 3,
                "kktsolver": "robust",
            },
        ),
    )


def production_policy() -> SolverPolicy:
    """Fail-closed R3 policy; no alternative solver passed qualification."""

    return SolverPolicy(
        attempts=(qualification_attempts()[0],),
        thresholds=FeasibilityThresholds(),
    )
