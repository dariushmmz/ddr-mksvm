"""Checkpointed Robust V8-C solver-qualification diagnostics.

This driver does not alter the predictive formulation or select new model
hyperparameters. It enumerates the frozen inner-CV calls for a fixed
``p=2, rho=.01`` candidate and records failures instead of aborting on the
first exception.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback
from typing import Any, Iterator

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from ddr_mksvm import v7_cross_dataset as frozen_v7
from ddr_mksvm.iris_research import ALPHA_RULES, SVC_C_GRID, alpha_from_rule
from ddr_mksvm.robust_matlab_parity import KernelSpec, gram, uncertainty_delta
from ddr_mksvm.robust_solvers.diagnostics import (
    array_sha256,
    canonical_problem_summary,
    finite_array_summary,
    matrix_condition_summary,
    solver_extra_stats,
)
from ddr_mksvm.robust_solvers.feasibility import FeasibilityThresholds, assess_feasibility
from ddr_mksvm.robust_solvers.registry import qualification_attempts
from ddr_mksvm.robust_v8_c import _factor_psd, pair_weights
from archive.research_scripts.run_robust_matlab_parity import atomic_csv, atomic_json, load


FIXED_P = 2.0
FIXED_RHO = 0.01
THRESHOLDS = FeasibilityThresholds()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _feature_summary(X: np.ndarray) -> dict[str, Any]:
    X = np.asarray(X, float)
    std = np.std(X, axis=0)
    result = {
        **finite_array_summary(X, "feature"),
        "feature_min_per_column": np.min(X, axis=0).tolist(),
        "feature_max_per_column": np.max(X, axis=0).tolist(),
        "feature_mean_per_column": np.mean(X, axis=0).tolist(),
        "feature_std_per_column": std.tolist(),
        "near_zero_variance_features": int(np.sum(std <= 1e-12)),
        "row_norm_min": float(np.min(np.linalg.norm(X, axis=1))),
        "row_norm_max": float(np.max(np.linalg.norm(X, axis=1))),
        "row_norm_mean": float(np.mean(np.linalg.norm(X, axis=1))),
    }
    centered = X - np.mean(X, axis=0)
    result.update(matrix_condition_summary(centered, "centered_feature"))
    if X.shape[0] > 1:
        result.update(matrix_condition_summary(np.cov(X, rowvar=False), "covariance"))
    return result


def build_problem(X: np.ndarray, y: np.ndarray, C: float, alpha: float,
                  factor_mode: str = "full", representation_mode: str = "sample",
                  objective_scale_mode: str = "none") -> dict[str, Any]:
    """Build the same CVXPY problem used by ``robust_v8_c.solve_pair``."""

    X, y = np.asarray(X, float), np.asarray(y)
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite feature supplied before canonicalization")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("Blood qualification requires exactly two classes")
    signed = np.where(y == classes[0], -1.0, 1.0)
    kernel = KernelSpec("rbf", alpha=float(alpha))
    sample_K = gram(X, None, kernel)
    if not np.all(np.isfinite(sample_K)):
        raise ValueError("non-finite Gram matrix supplied before canonicalization")
    sample_delta = uncertainty_delta(X, y, FIXED_RHO, FIXED_P, kernel, matlab_semantics=False)
    weights_by_class = pair_weights(y, "sqrt")

    if representation_mode == "sample":
        basis_X = X
        constraint_y = y
        constraint_signed = signed
        score_matrix = sample_K
        coefficient_multiplier = signed
        delta = sample_delta
        weights = np.asarray([weights_by_class[value] for value in y], float)
        K = sample_K
        M = np.outer(signed, signed) * K
        multiplicities = np.ones(len(y), dtype=int)
    elif representation_mode == "deduplicate_exact":
        basis_X, location = np.unique(X, axis=0, return_inverse=True)
        group_keys = sorted(set(zip(location.tolist(), y.tolist())))
        group_location = np.asarray([key[0] for key in group_keys], int)
        constraint_y = np.asarray([key[1] for key in group_keys], dtype=y.dtype)
        constraint_signed = np.where(constraint_y == classes[0], -1.0, 1.0)
        multiplicities = np.asarray([
            np.sum((location == loc) & (y == label)) for loc, label in group_keys
        ], int)
        representative = np.asarray([
            np.flatnonzero((location == loc) & (y == label))[0] for loc, label in group_keys
        ], int)
        delta = sample_delta[representative]
        weights = multiplicities * np.asarray([weights_by_class[value] for value in constraint_y], float)
        K = gram(basis_X, None, kernel)
        M = K
        score_matrix = gram(basis_X, basis_X[group_location], kernel).T
        coefficient_multiplier = np.ones(len(basis_X), float)
    else:
        raise ValueError(f"unknown representation mode: {representation_mode}")

    H, min_eigenvalue = _factor_psd(M)
    factor_rows_full = int(H.shape[0])
    factor_zero_rows_full = int(np.sum(~np.any(H != 0.0, axis=1)))
    if factor_mode == "drop_exact_zero_rows":
        H = H[np.any(H != 0.0, axis=1)]
    elif factor_mode != "full":
        raise ValueError(f"unknown factor mode: {factor_mode}")
    if np.any(delta < 0) or not np.all(np.isfinite(delta)):
        raise ValueError("invalid robust radius before canonicalization")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("invalid class weight before canonicalization")
    basis_count = len(basis_X)
    constraint_count = len(constraint_y)
    a = cp.Variable(basis_count, name="a")
    intercept = cp.Variable(name="intercept")
    xi = cp.Variable(constraint_count, nonneg=True, name="xi")
    t = cp.Variable(nonneg=True, name="t")
    scores = score_matrix @ cp.multiply(coefficient_multiplier, a) + intercept
    constraints = [
        cp.multiply(constraint_signed, scores) - delta * t >= 1 - xi,
        cp.norm(H @ a, 2) <= t,
    ]
    if objective_scale_mode == "none":
        objective_scale = 1.0
    elif objective_scale_mode == "max_slack_coefficient":
        objective_scale = 1.0 / max(1.0, float(C) * float(np.max(weights)))
    else:
        raise ValueError(f"unknown objective scale mode: {objective_scale_mode}")
    unscaled_objective = 0.5 * cp.square(t) + float(C) * (weights @ xi)
    problem = cp.Problem(cp.Minimize(objective_scale * unscaled_objective), constraints)
    return {
        "problem": problem,
        "a": a,
        "intercept": intercept,
        "xi": xi,
        "t": t,
        "X": X,
        "y": y,
        "classes": classes,
        "signed": signed,
        "constraint_signed": constraint_signed,
        "constraint_y": constraint_y,
        "score_matrix": score_matrix,
        "basis_X": basis_X,
        "coefficient_multiplier": coefficient_multiplier,
        "multiplicities": multiplicities,
        "K": K,
        "M": M,
        "H": H,
        "delta": delta,
        "weights": weights,
        "weights_by_class": weights_by_class,
        "min_eigenvalue": min_eigenvalue,
        "C": float(C),
        "alpha": float(alpha),
        "factor_mode": factor_mode,
        "representation_mode": representation_mode,
        "objective_scale_mode": objective_scale_mode,
        "objective_scale": objective_scale,
        "factor_rows_full": factor_rows_full,
        "factor_zero_rows_full": factor_zero_rows_full,
    }


def instance_diagnostics(context: dict[str, Any], solver: str) -> dict[str, Any]:
    X, y, K, M = context["X"], context["y"], context["K"], context["M"]
    eigenvalues = np.linalg.eigvalsh((K + K.T) / 2)
    counts = {str(value): int(np.sum(y == value)) for value in context["classes"]}
    result = {
        **_feature_summary(X),
        **matrix_condition_summary(K, "gram"),
        **matrix_condition_summary(M, "signed_gram"),
        **canonical_problem_summary(context["problem"], solver),
        "original_n_variables": int(len(context["basis_X"]) + len(context["constraint_y"]) + 2),
        "original_margin_constraints": int(len(context["constraint_y"])),
        "original_soc_dimension": int(context["H"].shape[0] + 1),
        "factor_rows": int(context["H"].shape[0]),
        "factor_zero_rows_full": int(context["factor_zero_rows_full"]),
        "factor_rows_removed": int(context["factor_rows_full"] - context["H"].shape[0]),
        "class_counts": counts,
        "sample_count": int(len(y)),
        "unique_feature_locations": int(len(context["basis_X"])),
        "constraint_group_count": int(len(context["constraint_y"])),
        "maximum_group_multiplicity": int(np.max(context["multiplicities"])),
        "class_weight_ratio": float(np.max(context["weights"]) / np.min(context["weights"])),
        "weight_min": float(np.min(context["weights"])),
        "weight_max": float(np.max(context["weights"])),
        "objective_slack_coefficient_min": float(context["C"] * np.min(context["weights"])),
        "objective_slack_coefficient_max": float(context["C"] * np.max(context["weights"])),
        "delta_min": float(np.min(context["delta"])),
        "delta_max": float(np.max(context["delta"])),
        "delta_mean": float(np.mean(context["delta"])),
        "gram_eigenvalue_min": float(np.min(eigenvalues)),
        "gram_eigenvalue_max": float(np.max(eigenvalues)),
        "gram_numerical_rank_1e-12": int(np.sum(eigenvalues > 1e-12 * max(1.0, np.max(eigenvalues)))),
        "signed_gram_min_eigenvalue_before_clip": float(context["min_eigenvalue"]),
        "gram_hash": array_sha256(K),
        "delta_hash": array_sha256(context["delta"]),
    }
    return result


def _solution_diagnostics(context: dict[str, Any]) -> dict[str, Any]:
    a_value = context["a"].value
    xi_value = context["xi"].value
    intercept_value = context["intercept"].value
    t_value = context["t"].value
    if a_value is None or xi_value is None or intercept_value is None or t_value is None:
        return {"solution_available": False, "max_constraint_violation": None}
    coeff = np.asarray(a_value, float)
    slack = np.asarray(xi_value, float)
    intercept = float(intercept_value)
    norm_bound = float(t_value)
    raw = context["score_matrix"] @ (context["coefficient_multiplier"] * coeff) + intercept
    margins = context["constraint_signed"] * raw - context["delta"] * norm_bound - 1 + slack
    cone_residual = norm_bound - np.linalg.norm(context["H"] @ coeff)
    weighted_slack = float(context["C"] * (context["weights"] @ slack))
    regularization = 0.5 * norm_bound**2
    return {
        "solution_available": True,
        "intercept": intercept,
        "rkhs_norm_bound": norm_bound,
        "coefficient_norm_l2": float(np.linalg.norm(coeff)),
        "coefficient_hash": array_sha256(coeff),
        "coefficient_vector": coeff.tolist(),
        "slack_sum": float(np.sum(slack)),
        "slack_min": float(np.min(slack)),
        "slack_max": float(np.max(slack)),
        "min_margin_residual": float(np.min(margins)),
        "cone_residual": float(cone_residual),
        "max_constraint_violation": float(max(0.0, -np.min(margins), -cone_residual, -np.min(slack), -norm_bound)),
        "regularization_term": regularization,
        "weighted_slack_term": weighted_slack,
        "objective_reconstructed": regularization + weighted_slack,
    }


def solve_attempt(context: dict[str, Any], attempt: Any, verbose_replay: bool = True) -> tuple[dict[str, Any], str]:
    problem = context["problem"]
    started = time.perf_counter()
    exception_type = exception_text = full_traceback = None
    try:
        problem.solve(solver=attempt.solver, **dict(attempt.options))
    except Exception as exc:  # intentional boundary capture
        exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
        exception_text = str(exc)
        full_traceback = traceback.format_exc()
    wall = time.perf_counter() - started
    output = {
        "attempt": attempt.name,
        "solver": attempt.solver,
        "solver_options": dict(attempt.options),
        "status": problem.status,
        "runtime_sec": wall,
        "solver_objective_scaled": _safe_float(problem.value),
        "exception_type": exception_type,
        "exception_text": exception_text,
        "exception_traceback": full_traceback,
        **solver_extra_stats(problem.solver_stats),
        **_solution_diagnostics(context),
    }
    output["objective"] = output.get("objective_reconstructed")
    if output.get("duality_gap") is not None:
        denominator = max(1.0, abs(output.get("primal_objective", 0.0)), abs(output.get("dual_objective", 0.0)))
        output["relative_gap"] = abs(float(output["duality_gap"])) / denominator
    else:
        output["relative_gap"] = None
    output["feasibility"] = assess_feasibility(problem.status, output, THRESHOLDS)

    verbose_log = ""
    if exception_type and verbose_replay:
        replay = build_problem(
            context["X"], context["y"], context["C"], context["alpha"],
            context["factor_mode"], context["representation_mode"], context["objective_scale_mode"],
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                replay["problem"].solve(solver=attempt.solver, verbose=True, **dict(attempt.options))
            except Exception:
                traceback.print_exc(file=buffer)
        verbose_log = buffer.getvalue()
    return output, verbose_log


def iter_inner_instances(dataset: str, seed: int) -> Iterator[dict[str, Any]]:
    X, y, spec = load(dataset)
    train, test = frozen_v7.split_indices(y, seed)
    outer_X, outer_y = X[train], y[train]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=20000 + int(seed)).split(outer_X, outer_y))
    prepared = []
    for fold, (fit, validate) in enumerate(folds):
        A, B, prep = frozen_v7.fit_transform(outer_X[fit], outer_X[validate], spec.get("transform", "none"))
        prepared.append((fold, A, B, outer_y[fit], outer_y[validate], prep, fit, validate))
    ordinal = 0
    for alpha_index, rule in enumerate(ALPHA_RULES):
        for C_index, C in enumerate(SVC_C_GRID):
            for fold, A, B, ya, yb, prep, fit, validate in prepared:
                alpha = float(alpha_from_rule(A, rule))
                yield {
                    "ordinal": ordinal,
                    "dataset": dataset,
                    "seed": int(seed),
                    "phase": "inner",
                    "fold": int(fold),
                    "alpha_rule": rule,
                    "alpha_index": int(alpha_index),
                    "C": float(C),
                    "C_index": int(C_index),
                    "alpha": alpha,
                    "X": A,
                    "y": ya,
                    "validation_X": B,
                    "validation_y": yb,
                    "preprocessing": prep,
                    "outer_train_indices_hash": hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest(),
                    "outer_test_indices_hash": hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest(),
                }
                ordinal += 1


def iter_outer_instance(dataset: str, seed: int, alpha_rule: str, C: float) -> Iterator[dict[str, Any]]:
    X, y, spec = load(dataset)
    train, test = frozen_v7.split_indices(y, seed)
    A, B, prep = frozen_v7.fit_transform(X[train], X[test], spec.get("transform", "none"))
    yield {
        "ordinal": 0,
        "dataset": dataset,
        "seed": int(seed),
        "phase": "outer",
        "fold": -1,
        "alpha_rule": alpha_rule,
        "alpha_index": int(ALPHA_RULES.index(alpha_rule)),
        "C": float(C),
        "C_index": int(SVC_C_GRID.index(float(C))),
        "alpha": float(alpha_from_rule(A, alpha_rule)),
        "X": A,
        "y": y[train],
        "validation_X": B,
        "validation_y": y[test],
        "preprocessing": prep,
        "outer_train_indices_hash": hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest(),
        "outer_test_indices_hash": hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest(),
    }


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if key in {"coefficient_vector", "exception_traceback"}:
            continue
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    return flat


def aggregate(run_root: Path) -> dict[str, Any]:
    records = [json.loads(path.read_text()) for path in sorted((run_root / "calls").glob("*.json"))]
    rows = [_flatten(record) for record in records]
    frame = pd.DataFrame(rows)
    atomic_csv(run_root / "solver_calls.csv", frame)
    if frame.empty:
        feasibility = failures = conditioning = pd.DataFrame()
    else:
        feasibility_columns = [column for column in frame.columns if column in {
            "dataset", "seed", "ordinal", "phase", "fold", "alpha_rule", "C", "alpha",
            "attempt", "solver", "status", "runtime_sec", "max_constraint_violation",
            "min_margin_residual", "cone_residual", "primal_residual", "dual_residual",
            "duality_gap", "relative_gap", "feasibility", "objective",
        }]
        feasibility = frame[feasibility_columns].copy()
        failures = frame[(frame["exception_type"].notna()) | (~frame["status"].isin(["optimal", "optimal_inaccurate"]))].copy()
        condition_columns = [column for column in frame.columns if column.startswith((
            "feature_", "centered_", "covariance_", "gram_", "signed_gram_", "canonical_", "cone_",
            "objective_slack_", "class_", "delta_", "original_",
        ))]
        identity = [column for column in ("dataset", "seed", "ordinal", "fold", "alpha_rule", "C", "alpha", "attempt") if column in frame]
        conditioning = frame[identity + condition_columns].copy()
    atomic_csv(run_root / "feasibility.csv", feasibility)
    atomic_csv(run_root / "failure_matrix.csv", failures)
    atomic_csv(run_root / "conditioning.csv", conditioning)
    status_counts = pd.Series([record.get("status") for record in records], dtype="object").value_counts(dropna=False)
    summary = {
        "calls": len(records),
        "exceptions": int(sum(record.get("exception_type") is not None for record in records)),
        "statuses": {"null" if key is None else str(key): int(value) for key, value in status_counts.items()},
        "attempts": {str(key): int(value) for key, value in pd.Series([record.get("attempt") for record in records], dtype="object").value_counts().items()},
        "accepted": int(sum(record.get("feasibility", {}).get("accepted", False) for record in records)),
        "thresholds": THRESHOLDS.as_dict(),
    }
    atomic_json(run_root / "summary.json", summary)
    atomic_json(run_root / "failures.json", {
        "failures": [{key: record.get(key) for key in (
            "dataset", "seed", "ordinal", "phase", "fold", "alpha_rule", "C", "alpha", "attempt",
            "solver", "status", "exception_type", "exception_text", "exception_traceback",
        )} for record in records if record.get("exception_type") or record.get("status") not in {"optimal", "optimal_inaccurate"}]
    })
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.getenv("MODAL_IS_REMOTE") and not os.getenv("ROBUST_ALLOW_LOCAL_FIT"):
        raise RuntimeError("qualification fitting is Modal-only")
    all_attempts = {attempt.name: attempt for attempt in qualification_attempts()}
    names = args.attempts.split(",")
    if not names or any(name not in all_attempts for name in names):
        raise ValueError(f"attempts must be chosen from {sorted(all_attempts)}")
    attempts = [all_attempts[name] for name in names]
    requested_ordinals = None
    if args.ordinals:
        requested_ordinals = {int(value) for value in args.ordinals.split(",") if value}
    root = Path(args.output_root) / args.run_name / args.dataset / f"seed-{args.seed}"
    calls = root / "calls"
    logs = root / "solver_logs"
    calls.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "robust-v8-c-solver-qualification-1",
        "dataset": args.dataset,
        "seed": args.seed,
        "p": FIXED_P,
        "rho": FIXED_RHO,
        "attempts": [{"name": item.name, "solver": item.solver, "options": dict(item.options)} for item in attempts],
        "thresholds": THRESHOLDS.as_dict(),
        "installed_solvers": cp.installed_solvers(),
        "cvxpy_version": cp.__version__,
        "python": platform.python_version(),
        "git": os.getenv("ROBUST_LOCAL_GIT", "unknown"),
        "source_hashes": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "qualify_robust_v8_c_solver.py",
                "ddr_mksvm/robust_v8_c.py",
                "ddr_mksvm/v8_class_sensitive.py",
                "ddr_mksvm/v7_cross_dataset.py",
                "ddr_mksvm/robust_solvers/diagnostics.py",
                "ddr_mksvm/robust_solvers/feasibility.py",
                "ddr_mksvm/robust_solvers/registry.py",
            )
        },
        "requested_ordinals": sorted(requested_ordinals) if requested_ordinals is not None else None,
        "factor_mode": args.factor_mode,
        "representation_mode": args.representation_mode,
        "objective_scale_mode": args.objective_scale_mode,
        "phase": args.phase,
        "outer_alpha_rule": args.alpha_rule if args.phase == "outer" else None,
        "outer_C": args.C if args.phase == "outer" else None,
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("incompatible qualification run configuration")
    if not config_path.exists():
        atomic_json(config_path, config)
    provenance_path = root / "invocation_provenance.json"
    provenance = {"modal_app_ids": [], "invocations": []}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
    app_id = os.getenv("MODAL_APP_ID", "local")
    if app_id not in provenance["modal_app_ids"]:
        provenance["modal_app_ids"].append(app_id)
    provenance["invocations"].append({
        "app_id": app_id,
        "max_calls": args.max_calls,
        "attempts": names,
        "started_unix": time.time(),
    })
    atomic_json(provenance_path, provenance)

    processed = 0
    instances = (
        iter_inner_instances(args.dataset, args.seed)
        if args.phase == "inner"
        else iter_outer_instance(args.dataset, args.seed, args.alpha_rule, args.C)
    )
    for instance in instances:
        if args.max_calls is not None and instance["ordinal"] >= args.max_calls:
            break
        if requested_ordinals is not None and instance["ordinal"] not in requested_ordinals:
            continue
        for attempt in attempts:
            call_id = f"{instance['ordinal']:03d}-{attempt.name}"
            checkpoint = calls / f"{call_id}.json"
            if checkpoint.exists():
                continue
            context = build_problem(
                instance["X"], instance["y"], instance["C"], instance["alpha"],
                args.factor_mode, args.representation_mode, args.objective_scale_mode,
            )
            static = instance_diagnostics(context, attempt.solver)
            result, verbose_log = solve_attempt(context, attempt, verbose_replay=instance["ordinal"] == 0)
            record = {
                **{key: value for key, value in instance.items() if key not in {"X", "y", "validation_X", "validation_y"}},
                "p": FIXED_P,
                "rho": FIXED_RHO,
                **static,
                **result,
            }
            if result.get("solution_available"):
                validation_scores = gram(context["basis_X"], instance["validation_X"], KernelSpec("rbf", alpha=instance["alpha"])).T @ (
                    context["coefficient_multiplier"] * np.asarray(context["a"].value, float)
                ) + float(context["intercept"].value)
                prediction = np.where(validation_scores >= 0, context["classes"][1], context["classes"][0])
                record.update({
                    "validation_prediction_hash": hashlib.sha256(np.asarray(prediction, dtype="<i8").tobytes()).hexdigest(),
                    "validation_predictions": np.asarray(prediction, int).tolist(),
                    "validation_balanced_loss": float(1 - balanced_accuracy_score(instance["validation_y"], prediction)),
                    "validation_margin_min": float(np.min(np.abs(validation_scores))),
                })
            atomic_json(checkpoint, record)
            if verbose_log:
                (logs / f"{call_id}.log").write_text(verbose_log, encoding="utf-8")
            processed += 1
            aggregate(root)
            print(json.dumps({
                "dataset": args.dataset,
                "seed": args.seed,
                "ordinal": instance["ordinal"],
                "attempt": attempt.name,
                "status": result.get("status"),
                "exception": result.get("exception_type"),
                "accepted": result["feasibility"]["accepted"],
            }), flush=True)
    summary = aggregate(root)
    atomic_json(root / "manifest.json", {"status": "complete", "processed_this_invocation": processed, **summary})
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--dataset", default="blood_transfusion")
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--attempts", default="clarabel_strict_original")
    value.add_argument("--max-calls", type=int)
    value.add_argument("--ordinals", default="")
    value.add_argument("--factor-mode", choices=("full", "drop_exact_zero_rows"), default="full")
    value.add_argument("--representation-mode", choices=("sample", "deduplicate_exact"), default="sample")
    value.add_argument("--objective-scale-mode", choices=("none", "max_slack_coefficient"), default="none")
    value.add_argument("--phase", choices=("inner", "outer"), default="inner")
    value.add_argument("--alpha-rule", choices=ALPHA_RULES, default="paper")
    value.add_argument("--C", type=float, choices=SVC_C_GRID, default=0.1)
    value.add_argument("--run-name", required=True)
    value.add_argument("--output-root", default="results/robust/solver_qualification")
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
