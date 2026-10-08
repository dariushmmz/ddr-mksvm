"""Audit whether exact full-rank RKHS coordinates exist without changing K.

This script performs factorization diagnostics only. It never constructs or
solves a predictive optimization problem.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from ddr_mksvm.robust_matlab_parity import KernelSpec, gram
from ddr_mksvm.robust_solvers.diagnostics import array_sha256
from ddr_mksvm.robust_solvers.rkhs_coordinates import factor_full_rank_gram
from archive.research_scripts.qualify_robust_v8_c_solver import iter_inner_instances, iter_outer_instance
from archive.research_scripts.run_robust_matlab_parity import atomic_csv, atomic_json


ROOT = Path("results/robust/solver_qualification")
OUTPUT = ROOT / "r3_2_rkhs_coordinates"
REFERENCE = ROOT / "blood/sq-003-boundary-complete"
PARKINSON_REFERENCE = ROOT / "comparisons/sq-006-parkinson-conditioning"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), np.finfo(float).tiny))


def factorization_diagnostics(K: np.ndarray) -> dict[str, Any]:
    K = np.asarray(K, dtype=np.float64)
    symmetry_error = float(np.linalg.norm(K - K.T, ord="fro"))
    eigenvalues = np.linalg.eigvalsh(K)
    singular_values = np.linalg.svd(K, compute_uv=False)
    largest_singular = float(singular_values[0])
    smallest_singular = float(singular_values[-1])
    diagnostics: dict[str, Any] = {
        "gram_shape": list(K.shape),
        "gram_hash": array_sha256(K),
        "gram_symmetry_error_fro": symmetry_error,
        "gram_symmetry_error_relative": _relative(symmetry_error, np.linalg.norm(K, ord="fro")),
        "gram_eigenvalue_min": float(eigenvalues[0]),
        "gram_eigenvalue_max": float(eigenvalues[-1]),
        "gram_negative_eigenvalues": int(np.sum(eigenvalues < 0.0)),
        "gram_zero_eigenvalues": int(np.sum(eigenvalues == 0.0)),
        "gram_positive_eigenvalues": int(np.sum(eigenvalues > 0.0)),
        "gram_singular_value_min": smallest_singular,
        "gram_singular_value_max": largest_singular,
        "gram_condition_2": float(largest_singular / smallest_singular) if smallest_singular else float("inf"),
        "gram_rank_numpy": int(np.linalg.matrix_rank(K)),
        "cholesky_success": False,
        "cholesky_exception_type": None,
        "cholesky_exception_text": None,
        "R_condition_2": None,
        "gram_reconstruction_error_fro": None,
        "gram_reconstruction_error_relative": None,
        "transformation_residual_relative": None,
        "inverse_residual_relative": None,
    }
    try:
        R = factor_full_rank_gram(K)
        reconstructed = R.T @ R
        probe = np.linspace(1.0, 2.0, K.shape[0], dtype=np.float64)
        z = R @ probe
        recovered = np.linalg.solve(R, z)
        reconstruction_error = float(np.linalg.norm(K - reconstructed, ord="fro"))
        diagnostics.update({
            "cholesky_success": True,
            "R_hash": array_sha256(R),
            "R_condition_2": float(np.linalg.cond(R)),
            "gram_reconstruction_error_fro": reconstruction_error,
            "gram_reconstruction_error_relative": _relative(reconstruction_error, np.linalg.norm(K, ord="fro")),
            "transformation_residual_relative": _relative(
                np.linalg.norm(R.T @ z - K @ probe), np.linalg.norm(K @ probe)
            ),
            "inverse_residual_relative": _relative(np.linalg.norm(recovered - probe), np.linalg.norm(probe)),
        })
    except (np.linalg.LinAlgError, ValueError) as exc:
        diagnostics["cholesky_exception_type"] = type(exc).__name__
        diagnostics["cholesky_exception_text"] = str(exc)
    return diagnostics


def _reference_rows(path: Path) -> pd.DataFrame:
    frames = [pd.read_csv(item) for item in sorted(path.glob("**/seed-*/solver_calls.csv"))]
    if not frames:
        raise FileNotFoundError(f"no solver_calls.csv under {path}")
    return pd.concat(frames, ignore_index=True)


def _instance_map(dataset: str, seed: int) -> dict[int, dict[str, Any]]:
    return {int(item["ordinal"]): item for item in iter_inner_instances(dataset, seed)}


def _diagnose_instance(instance: dict[str, Any], role: str, reference_status: str | None) -> dict[str, Any]:
    started = time.perf_counter()
    unique_X = np.unique(np.asarray(instance["X"], float), axis=0)
    K = gram(unique_X, None, KernelSpec("rbf", alpha=float(instance["alpha"])))
    diagnostics = factorization_diagnostics(K)
    return {
        "role": role,
        "dataset": instance["dataset"],
        "seed": int(instance["seed"]),
        "ordinal": int(instance["ordinal"]),
        "phase": instance["phase"],
        "fold": int(instance["fold"]),
        "alpha_rule": instance["alpha_rule"],
        "alpha": float(instance["alpha"]),
        "C": float(instance["C"]),
        "sample_count": int(len(instance["X"])),
        "unique_feature_locations": int(len(unique_X)),
        "training_X_hash": array_sha256(np.asarray(instance["X"], float)),
        "training_y_hash": hashlib.sha256(np.asarray(instance["y"], dtype="<i8").tobytes()).hexdigest(),
        "outer_train_indices_hash": instance["outer_train_indices_hash"],
        "outer_test_indices_hash": instance["outer_test_indices_hash"],
        "reference_status": reference_status,
        "factorization_runtime_sec": float(time.perf_counter() - started),
        "solver_status": None,
        "objective": None,
        "max_constraint_violation": None,
        "prediction_hash": None,
        "optimization_runtime_sec": None,
        **diagnostics,
    }


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blood = _reference_rows(REFERENCE)
    blood = blood[blood["status"].isin(["optimal", "optimal_inaccurate"])].copy()
    for seed, group in blood.groupby("seed"):
        instances = _instance_map("blood_transfusion", int(seed))
        for _, reference in group.sort_values("ordinal").iterrows():
            rows.append(_diagnose_instance(
                instances[int(reference.ordinal)], "blood_inner_accepted_reference", str(reference.status)
            ))

    park = _reference_rows(PARKINSON_REFERENCE)
    for seed, group in park.groupby("seed"):
        instances = _instance_map("parkinson", int(seed))
        for _, reference in group.sort_values("ordinal").iterrows():
            rows.append(_diagnose_instance(
                instances[int(reference.ordinal)], "parkinson_known_solving_control", str(reference.status)
            ))

    for seed, rule, C in ((15000, "med:-0.5", 10.0), (15001, "paper", 100.0)):
        instance = next(iter_outer_instance("blood_transfusion", seed, rule, C))
        rows.append(_diagnose_instance(instance, "blood_selected_outer", None))

    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUTPUT / "factorization_diagnostics.csv", frame)

    grouped = (
        frame.groupby("role", dropna=False)
        .agg(
            matrices=("gram_hash", "size"),
            unique_matrices=("gram_hash", "nunique"),
            cholesky_successes=("cholesky_success", "sum"),
            minimum_eigenvalue=("gram_eigenvalue_min", "min"),
            maximum_condition_2=("gram_condition_2", "max"),
        )
        .reset_index()
    )
    atomic_csv(OUTPUT / "factorization_summary.csv", grouped)

    gates = pd.DataFrame([
        {
            "gate": "exact_full_rank_factorization",
            "status": "failed",
            "reason": "0/2 selected Blood outer matrices admit unmodified full-rank Cholesky",
        },
        {
            "gate": "inner_equivalence",
            "status": "not_run",
            "reason": "blocked by exact full-rank factorization gate",
        },
        {
            "gate": "blood_outer_optimization",
            "status": "not_run",
            "reason": "blocked by exact full-rank factorization gate",
        },
        {
            "gate": "optional_exact_row_scaling",
            "status": "not_run",
            "reason": "coordinate formulation could not be instantiated exactly",
        },
    ])
    atomic_csv(OUTPUT / "gate_status.csv", gates)

    outer = frame[frame.role == "blood_selected_outer"]
    summary = {
        "stage": "R3.2",
        "operation": "factorization_diagnostics_only",
        "optimization_calls": 0,
        "blood_inner_reference_rows": int((frame.role == "blood_inner_accepted_reference").sum()),
        "blood_inner_cholesky_successes": int(
            frame.loc[frame.role == "blood_inner_accepted_reference", "cholesky_success"].sum()
        ),
        "parkinson_control_rows": int((frame.role == "parkinson_known_solving_control").sum()),
        "parkinson_control_cholesky_successes": int(
            frame.loc[frame.role == "parkinson_known_solving_control", "cholesky_success"].sum()
        ),
        "blood_outer_rows": int(len(outer)),
        "blood_outer_cholesky_successes": int(outer.cholesky_success.sum()),
        "exact_full_rank_coordinate_factor_available": bool(len(outer) == 2 and outer.cholesky_success.all()),
        "equivalence_gate_run": False,
        "outer_optimization_run": False,
        "decision": "R3.2 NOT QUALIFIED",
        "stop_reason": (
            "Both stored Blood outer Gram matrices fail unmodified full-rank Cholesky; "
            "constructing R would require a prohibited modification or spectral truncation."
        ),
    }
    config = {
        "datasets": ["blood_transfusion", "parkinson"],
        "blood_seeds": [15000, 15001],
        "parkinson_seeds": [15000],
        "p": 2.0,
        "rho": 0.01,
        "factorization": "unmodified_numpy_float64_cholesky",
        "prohibited": ["jitter", "eigenvalue_clamping", "eigenvalue_truncation", "low_rank", "row_scaling"],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "source_hashes": {
            str(path): _file_hash(path)
            for path in (
                Path(__file__),
                Path("qualify_robust_v8_c_solver.py"),
                Path("ddr_mksvm/robust_v8_c.py"),
                Path("ddr_mksvm/robust_matlab_parity.py"),
                Path("ddr_mksvm/robust_solvers/rkhs_coordinates.py"),
                Path("ddr_mksvm/v7_cross_dataset.py"),
                Path("ddr_mksvm/v8_class_sensitive.py"),
            )
        },
    }
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["config_hash"] = hashlib.sha256(config_bytes).hexdigest()
    atomic_json(OUTPUT / "config.json", config)
    atomic_json(OUTPUT / "summary.json", summary)
    atomic_json(OUTPUT / "manifest.json", {
        "status": "complete",
        "immutable": True,
        "decision": summary["decision"],
        "config_hash": config["config_hash"],
        "artifacts": {
            name: _file_hash(OUTPUT / name)
            for name in (
                "config.json", "factorization_diagnostics.csv", "factorization_summary.csv",
                "gate_status.csv", "summary.json",
            )
        },
    })
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
