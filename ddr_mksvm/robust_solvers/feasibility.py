"""Prospective feasibility gates for Robust V8-C solver output."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class FeasibilityThresholds:
    """Scale-aware thresholds frozen before SQ solver comparisons."""

    max_constraint_violation: float = 1e-6
    primal_residual: float = 1e-5
    dual_residual: float = 1e-5
    relative_gap: float = 1e-5

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite_or_missing(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


def assess_feasibility(
    status: str | None,
    diagnostics: Mapping[str, Any],
    thresholds: FeasibilityThresholds = FeasibilityThresholds(),
) -> dict[str, Any]:
    """Classify a solver result without inventing unavailable diagnostics.

    Computed primal constraint violations are mandatory. Solver-reported
    residuals are checked when the backend exposes them and remain ``None``
    otherwise.
    """

    status = None if status is None else str(status).lower()
    reasons: list[str] = []
    if status not in {"optimal", "optimal_inaccurate"}:
        reasons.append(f"status={status}")

    checks = (
        ("max_constraint_violation", thresholds.max_constraint_violation),
        ("primal_residual", thresholds.primal_residual),
        ("dual_residual", thresholds.dual_residual),
        ("relative_gap", thresholds.relative_gap),
    )
    for field, limit in checks:
        value = diagnostics.get(field)
        if not _finite_or_missing(value):
            reasons.append(f"{field}=nonfinite")
        elif value is not None and abs(float(value)) > limit:
            reasons.append(f"{field}={float(value):.6g}>{limit:.6g}")

    accepted = not reasons
    classification = "reject"
    if accepted and status == "optimal":
        classification = "accept"
    elif accepted and status == "optimal_inaccurate":
        classification = "conditional"
    return {
        "accepted": accepted,
        "classification": classification,
        "reasons": reasons,
        "thresholds": thresholds.as_dict(),
    }
