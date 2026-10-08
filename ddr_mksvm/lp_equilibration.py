"""Exact positive-diagonal numerical coordinates for source q=1 LPs.

This module changes only the coordinates supplied to the LP solver.  Given
``x = D z`` and a positive row diagonal ``S``, it solves

    min (c * d)' z  subject to  diag(s) A diag(d) z <= diag(s) b.

Every reported solution is mapped back to the original ``x`` coordinates and
checked against the original LP.  No coefficient is dropped or repaired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

EQUILIBRATION_PASSES = 8
UPDATE_MIN = 1.0e-4
UPDATE_MAX = 1.0e4


@dataclass(frozen=True)
class EquilibratedLP:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    bounds: tuple[tuple[float | None, float | None], ...]
    row_scale: np.ndarray
    variable_scale: np.ndarray
    passes: int
    update_min: float
    update_max: float


def _validate_lp(c, A, b, bounds):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if c.ndim != 1 or A.ndim != 2 or b.ndim != 1:
        raise ValueError("expected c vector, A matrix, and b vector")
    if A.shape != (b.size, c.size) or len(bounds) != c.size:
        raise ValueError("incompatible LP dimensions")
    if not (np.isfinite(c).all() and np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("LP coefficients must be finite")
    normalized = []
    for lower, upper in bounds:
        if lower is not None and not np.isfinite(lower):
            raise ValueError("finite or None lower bounds are required")
        if upper is not None and not np.isfinite(upper):
            raise ValueError("finite or None upper bounds are required")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("invalid bound interval")
        normalized.append((lower, upper))
    return c, A, b, tuple(normalized)


def equilibrate_lp(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    *,
    passes: int = EQUILIBRATION_PASSES,
    update_min: float = UPDATE_MIN,
    update_max: float = UPDATE_MAX,
) -> EquilibratedLP:
    """Apply the prospectively frozen max-norm Ruiz-style transformation."""
    c0, A0, b0, bounds0 = _validate_lp(c, A, b, bounds)
    if passes < 0 or int(passes) != passes:
        raise ValueError("passes must be a nonnegative integer")
    if not (0 < update_min <= update_max < np.inf):
        raise ValueError("scaling update limits must be finite and positive")

    cs = c0.copy()
    As = A0.copy()
    bs = b0.copy()
    row_scale = np.ones(b0.size, dtype=float)
    variable_scale = np.ones(c0.size, dtype=float)

    for _ in range(int(passes)):
        row_norm = np.maximum(np.max(np.abs(As), axis=1), np.abs(bs))
        row_update = np.ones_like(row_norm)
        nonzero_rows = row_norm != 0.0
        row_update[nonzero_rows] = 1.0 / np.sqrt(row_norm[nonzero_rows])
        row_update = np.clip(row_update, update_min, update_max)
        As *= row_update[:, None]
        bs *= row_update
        row_scale *= row_update

        column_norm = np.maximum(np.max(np.abs(As), axis=0), np.abs(cs))
        column_update = np.ones_like(column_norm)
        nonzero_columns = column_norm != 0.0
        column_update[nonzero_columns] = 1.0 / np.sqrt(column_norm[nonzero_columns])
        column_update = np.clip(column_update, update_min, update_max)
        As *= column_update[None, :]
        cs *= column_update
        variable_scale *= column_update

    if not (
        np.isfinite(As).all()
        and np.isfinite(bs).all()
        and np.isfinite(cs).all()
        and np.isfinite(row_scale).all()
        and np.isfinite(variable_scale).all()
        and np.all(row_scale > 0)
        and np.all(variable_scale > 0)
    ):
        raise FloatingPointError("equilibration produced invalid positive scales")

    scaled_bounds = tuple(
        (
            None if lower is None else float(lower) / variable_scale[j],
            None if upper is None else float(upper) / variable_scale[j],
        )
        for j, (lower, upper) in enumerate(bounds0)
    )
    return EquilibratedLP(
        c=cs,
        A=As,
        b=bs,
        bounds=scaled_bounds,
        row_scale=row_scale,
        variable_scale=variable_scale,
        passes=int(passes),
        update_min=float(update_min),
        update_max=float(update_max),
    )


def to_scaled_coordinates(x: np.ndarray, variable_scale: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = np.asarray(variable_scale, dtype=float)
    if x.shape != scale.shape or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("invalid variable-coordinate mapping")
    return x / scale


def to_original_coordinates(z: np.ndarray, variable_scale: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    scale = np.asarray(variable_scale, dtype=float)
    if z.shape != scale.shape or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("invalid variable-coordinate mapping")
    return z * scale


def positive_dynamic_range(*arrays: np.ndarray) -> float | None:
    values = np.concatenate([np.abs(np.asarray(value, dtype=float)).ravel() for value in arrays])
    values = values[values > 0]
    return None if values.size == 0 else float(np.max(values) / np.min(values))


def bound_violation(x: np.ndarray, bounds) -> float:
    x = np.asarray(x, dtype=float)
    violation = 0.0
    for value, (lower, upper) in zip(x, bounds):
        if lower is not None:
            violation = max(violation, float(lower - value))
        if upper is not None:
            violation = max(violation, float(value - upper))
    return max(0.0, violation)
