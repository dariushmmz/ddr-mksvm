"""Export and compare one exact Iris split against the authoritative MATLAB code.

The harness intentionally separates split generation from optimization.  MATLAB
loads the exact supplied indices, eliminating RNG and row-order differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from ddr_mksvm.iris_research import (
    PAPER_NU_GRID,
    fixed_grid_threshold,
    index_hash,
    locked_split_indices,
    numeric_sha256,
    paper_alpha,
    polynomial_gram,
    predict_multiclass,
    rbf_gram,
    solve_binary_q1,
)


AUTHORITATIVE_CSV = Path(
    r"F:\Projects\SVM_Improvement\.SVM-MATLAB-Version\NonlinearSVM-main"
    r"\2. Deterministic and multiclass\iris_multiclass.csv"
)
EXPECTED_NUMERIC_SHA256 = "1b3e788db71aac2852e77ce253c00e9cedd173d8b5815b47aa530252a4c99e70"


def _data() -> np.ndarray:
    if not AUTHORITATIVE_CSV.exists():
        raise FileNotFoundError(f"authoritative MATLAB CSV not found: {AUTHORITATIVE_CSV}")
    values = pd.read_csv(AUTHORITATIVE_CSV).to_numpy(float)
    digest = numeric_sha256(values)
    if digest != EXPECTED_NUMERIC_SHA256:
        raise ValueError(f"authoritative numeric data hash changed: {digest}")
    return values


def export_split(seed: int, kernel: str, output: Path) -> None:
    values = _data()
    train, test = locked_split_indices(values[:, -1], seed)
    X_train = values[train, :-1]
    alpha = paper_alpha(X_train)
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, {
        "DATAtrain": values[train].T,
        "DATAtest": values[test].T,
        "train_indices_1based": train.reshape(-1, 1) + 1,
        "test_indices_1based": test.reshape(-1, 1) + 1,
        "vectornu": np.asarray(PAPER_NU_GRID).reshape(1, -1),
        "kernel_kind": kernel,
        "alpha": np.array([[alpha]]),
        "polynomial_degree": np.array([[2]]),
        "polynomial_offset": np.array([[0.0]]),
        "grid_points": np.array([[10_000]]),
    }, do_compression=True)
    manifest = {
        "seed": seed, "kernel": kernel,
        "dataset_numeric_sha256": numeric_sha256(values),
        "train_indices_hash": index_hash(train), "test_indices_hash": index_hash(test),
        "train_indices_1based": (train + 1).tolist(),
        "test_indices_1based": (test + 1).tolist(),
        "train_shape_matlab": [5, len(train)], "test_shape_matlab": [5, len(test)],
        "alpha": alpha, "nu_grid": list(PAPER_NU_GRID),
        "mat_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def python_trace(input_path: Path, output: Path, solver: str) -> dict:
    payload = loadmat(input_path, squeeze_me=True)
    train = np.asarray(payload["DATAtrain"], float).T
    test = np.asarray(payload["DATAtest"], float).T
    X, labels = train[:, :-1], train[:, -1].astype(int)
    X_test, y_test = test[:, :-1], test[:, -1].astype(int)
    kernel = str(payload["kernel_kind"]).strip()
    alpha = float(payload["alpha"])
    nu_grid = np.atleast_1d(payload["vectornu"]).astype(float)
    K = polynomial_gram(X, 2, 0.0) if kernel == "poly" else rbf_gram(X, alpha)
    m, n_classes, n_nu = len(X), 3, len(nu_grid)
    u_all = np.zeros((m, n_classes, n_nu))
    gamma_all = np.zeros((n_classes, n_nu))
    xi_all = np.zeros((m, n_classes, n_nu))
    b_all = np.zeros((n_classes, n_nu))
    objective_all = np.zeros((n_classes, n_nu))
    train_error_all = np.zeros((n_classes, n_nu))
    left_all = np.zeros((n_classes, n_nu))
    right_all = np.zeros((n_classes, n_nu))
    selected_models = []
    selected_nu_index = np.zeros(n_classes, dtype=int)
    diagnostic_rows = []
    for class_index, label in enumerate((1, 2, 3)):
        y = np.where(labels == label, 1.0, -1.0)
        models = []
        for nu_index, nu in enumerate(nu_grid):
            model, diag = solve_binary_q1(
                X, y, alpha, float(nu), threshold_mode="grid",
                kernel_kind=kernel, polynomial_degree=2, polynomial_offset=0.0,
                solver_backend=solver,
            )
            models.append(model)
            u_all[:, class_index, nu_index] = model.u
            gamma_all[class_index, nu_index] = model.gamma
            xi_all[:, class_index, nu_index] = model.xi
            b_all[class_index, nu_index] = model.b
            objective_all[class_index, nu_index] = diag["solver_objective"]
            train_error_all[class_index, nu_index] = diag["robust_training_error"]
            # Preserve raw MATLAB endpoint order, not sorted diagnostic bounds.
            dxi = y * model.xi
            left_all[class_index, nu_index] = model.gamma + 1.0 - np.max(-dxi)
            right_all[class_index, nu_index] = model.gamma - 1.0 + np.max(dxi)
            diagnostic_rows.append({"class": label, "nu": float(nu), **diag})
        chosen = int(np.argmin(train_error_all[class_index]))
        selected_nu_index[class_index] = chosen
        selected_models.append(models[chosen])
        selected_models[-1].positive_class = label
    test_scores = np.column_stack([model.decision(X_test) for model in selected_models])
    prediction = np.argmax(test_scores, axis=1) + 1
    tie_count = int(np.sum(np.sum(test_scores == test_scores.max(axis=1, keepdims=True), axis=1) > 1))
    arrays = {
        "K": K, "u_by_nu": u_all, "gamma_by_nu": gamma_all,
        "xi_by_nu": xi_all, "b_by_nu": b_all,
        "objective_by_nu": objective_all, "training_error_by_nu": train_error_all,
        "raw_left_by_nu": left_all, "raw_right_by_nu": right_all,
        "selected_nu_index_1based": selected_nu_index + 1,
        "test_scores": test_scores, "prediction": prediction,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    summary = {
        "solver": solver, "kernel": kernel,
        "test_error": float(np.mean(prediction != y_test)), "score_tie_count": tie_count,
        "selected_nu_index_1based": (selected_nu_index + 1).tolist(),
        "selected_nu": [float(nu_grid[i]) for i in selected_nu_index],
        "diagnostics": diagnostic_rows,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def compare(input_path: Path, python_path: Path, matlab_path: Path, report_path: Path) -> None:
    py = dict(np.load(python_path))
    ml = loadmat(matlab_path, squeeze_me=True)
    fields = [
        "K", "u_by_nu", "gamma_by_nu", "xi_by_nu", "b_by_nu",
        "objective_by_nu", "training_error_by_nu", "raw_left_by_nu",
        "raw_right_by_nu", "test_scores", "prediction",
    ]
    comparisons = {}
    for field in fields:
        left, right = np.asarray(py[field]), np.asarray(ml[field])
        comparisons[field] = {
            "shape_python": list(left.shape), "shape_matlab": list(right.shape),
            "max_abs_difference": float(np.max(np.abs(left - right))),
            "allclose_rtol1e-7_atol1e-8": bool(np.allclose(left, right, rtol=1e-7, atol=1e-8)),
        }
    comparisons["prediction"]["exact_match"] = bool(
        np.array_equal(np.ravel(py["prediction"]), np.ravel(ml["prediction"]))
    )
    report = {
        "input": str(input_path), "python_trace": str(python_path),
        "matlab_trace": str(matlab_path), "comparisons": comparisons,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--seed", type=int, required=True)
    export.add_argument("--kernel", choices=("poly", "rbf"), default="poly")
    export.add_argument("--output", type=Path, required=True)
    trace = sub.add_parser("python-trace")
    trace.add_argument("--input", type=Path, required=True)
    trace.add_argument("--output", type=Path, required=True)
    trace.add_argument("--solver", default="scipy-highs",
                       choices=("scipy-highs", "scipy-highs-ds", "scipy-highs-ipm",
                                "cvxpy-highs", "cvxpy-clarabel", "cvxpy-cvxopt"))
    check = sub.add_parser("compare")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--python", type=Path, required=True)
    check.add_argument("--matlab", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_split(args.seed, args.kernel, args.output)
    elif args.command == "python-trace":
        summary = python_trace(args.input, args.output, args.solver)
        print(json.dumps({k: v for k, v in summary.items() if k != "diagnostics"}, indent=2))
    else:
        compare(args.input, args.python, args.matlab, args.report)


if __name__ == "__main__":
    main()
