"""Central numerical policy and diagnostics for Robust V8-C conic solves."""

from .feasibility import FeasibilityThresholds, assess_feasibility
from .policy import AttemptResult, SolverAttempt, SolverPolicy, run_solver_policy
from .registry import production_policy, qualification_attempts

__all__ = [
    "AttemptResult",
    "FeasibilityThresholds",
    "SolverAttempt",
    "SolverPolicy",
    "assess_feasibility",
    "qualification_attempts",
    "production_policy",
    "run_solver_policy",
]
