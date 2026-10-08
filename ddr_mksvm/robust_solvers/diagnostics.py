"""CVXPY canonicalization and numerical-statistics helpers."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes()).hexdigest()


def finite_array_summary(value: np.ndarray, prefix: str) -> dict[str, Any]:
    value = np.asarray(value, float)
    finite = np.isfinite(value)
    result: dict[str, Any] = {
        f"{prefix}_shape": list(value.shape),
        f"{prefix}_size": int(value.size),
        f"{prefix}_finite": bool(np.all(finite)),
        f"{prefix}_hash": array_sha256(value),
    }
    if value.size and np.any(finite):
        selected = value[finite]
        result.update({
            f"{prefix}_min": float(np.min(selected)),
            f"{prefix}_max": float(np.max(selected)),
            f"{prefix}_mean": float(np.mean(selected)),
            f"{prefix}_std": float(np.std(selected)),
            f"{prefix}_abs_max": float(np.max(np.abs(selected))),
        })
    return result


def matrix_condition_summary(value: np.ndarray, prefix: str, tolerance: float = 1e-12) -> dict[str, Any]:
    value = np.asarray(value, float)
    singular = np.linalg.svd(value, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    positive = singular[singular > tolerance * max(1.0, largest)]
    smallest = float(positive[-1]) if positive.size else 0.0
    condition = float(largest / smallest) if smallest else math.inf
    return {
        f"{prefix}_singular_max": largest,
        f"{prefix}_singular_min_effective": smallest,
        f"{prefix}_effective_rank": int(positive.size),
        f"{prefix}_condition_effective": condition,
    }


def canonical_problem_summary(problem: Any, solver: str) -> dict[str, Any]:
    data, chain, inverse = problem.get_problem_data(solver)
    result: dict[str, Any] = {
        "is_dcp": bool(problem.is_dcp()),
        "is_dpp": bool(problem.is_dpp()),
        "solver": solver,
        "reduction_chain": [type(item).__name__ for item in chain.reductions],
        "canonical_keys": sorted(data),
    }
    for name in ("A", "G", "P"):
        matrix = data.get(name)
        if matrix is not None:
            result[f"canonical_{name}_shape"] = list(matrix.shape)
            result[f"canonical_{name}_nnz"] = int(matrix.nnz)
            if matrix.nnz:
                result[f"canonical_{name}_abs_max"] = float(np.max(np.abs(matrix.data)))
                nonzero = np.abs(matrix.data[np.nonzero(matrix.data)])
                result[f"canonical_{name}_abs_min_nonzero"] = float(np.min(nonzero)) if nonzero.size else 0.0
    for name in ("b", "c"):
        vector = data.get(name)
        if vector is not None:
            result.update(finite_array_summary(np.asarray(vector), f"canonical_{name}"))
    dims = data.get("dims")
    if dims is not None:
        result["cone_zero"] = int(getattr(dims, "zero", 0))
        result["cone_nonneg"] = int(getattr(dims, "nonneg", 0))
        result["cone_soc"] = [int(value) for value in getattr(dims, "soc", [])]
        result["cone_exp"] = int(getattr(dims, "exp", 0))
        result["cone_psd"] = [int(value) for value in getattr(dims, "psd", [])]
    result["canonical_n_variables"] = int(np.asarray(data["c"]).size)
    result["canonical_n_equalities"] = int(data["A"].shape[0]) if data.get("A") is not None else 0
    if data.get("G") is not None:
        result["canonical_n_inequalities"] = int(data["G"].shape[0])
    else:
        # CLARABEL/SCS expose a single stacked cone matrix as A.
        result["canonical_n_inequalities"] = int(data["A"].shape[0])
    result["canonical_n_constraints"] = (
        result["canonical_n_equalities"] + result["canonical_n_inequalities"]
    )
    return result


def solver_extra_stats(stats: Any) -> dict[str, Any]:
    """Extract only backend fields that really exist."""

    output = {"solver_iterations": None, "solver_solve_time": None, "solver_setup_time": None}
    if stats is None:
        return output
    output.update({
        "solver_iterations": getattr(stats, "num_iters", None),
        "solver_solve_time": getattr(stats, "solve_time", None),
        "solver_setup_time": getattr(stats, "setup_time", None),
    })
    extra = getattr(stats, "extra_stats", None)
    info = extra.get("info") if isinstance(extra, dict) and isinstance(extra.get("info"), dict) else None
    if info:
        mapping = {
            "res_pri": "primal_residual",
            "res_dual": "dual_residual",
            "gap": "duality_gap",
            "pobj": "primal_objective",
            "dobj": "dual_objective",
            "status": "backend_status",
            "iter": "backend_iterations",
            "scale": "backend_scale",
        }
        for source, target in mapping.items():
            if source in info:
                value = info[source]
                output[target] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
    return output
