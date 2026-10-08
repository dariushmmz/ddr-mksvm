"""Deterministic, testable retry policy for conic optimizer calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .feasibility import FeasibilityThresholds, assess_feasibility


@dataclass(frozen=True)
class SolverAttempt:
    name: str
    solver: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AttemptResult:
    attempt: SolverAttempt
    status: str | None
    diagnostics: dict[str, Any]
    payload: Any = None
    exception_type: str | None = None
    exception_text: str | None = None


@dataclass(frozen=True)
class SolverPolicy:
    attempts: tuple[SolverAttempt, ...]
    thresholds: FeasibilityThresholds = FeasibilityThresholds()

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ValueError("a solver policy requires at least one attempt")
        names = [attempt.name for attempt in self.attempts]
        if len(names) != len(set(names)):
            raise ValueError("solver attempt names must be unique")


def run_solver_policy(
    policy: SolverPolicy,
    solve: Callable[[SolverAttempt, int], AttemptResult],
) -> tuple[AttemptResult | None, list[dict[str, Any]]]:
    """Run attempts in declaration order and stop at the first accepted one."""

    history: list[dict[str, Any]] = []
    for retry, attempt in enumerate(policy.attempts):
        result = solve(attempt, retry)
        assessment = assess_feasibility(result.status, result.diagnostics, policy.thresholds)
        history.append({
            "retry": retry,
            "attempt": attempt.name,
            "solver": attempt.solver,
            "status": result.status,
            "exception_type": result.exception_type,
            "exception_text": result.exception_text,
            **assessment,
        })
        if assessment["accepted"]:
            return result, history
    return None, history
