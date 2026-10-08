"""Checkpointed Modal-only R9 exposed-seed numerical qualification."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support

from ddr_mksvm.robust_bounded_rbf import (
    FEASIBILITY_TOLERANCE,
    R9_P,
    R9_RHO,
    R9_TAU,
    build_r9_q1_lp,
    fit_binary,
    rbf_spec,
    rbf_uncertainty_delta,
)
from ddr_mksvm.robust_matlab_parity import array_hash, gram
from archive.research_scripts.run_robust_matlab_parity import SOURCE_HASHES, atomic_csv, atomic_json, digest, prepare

MODELS = (
    "deterministic_rbf_q1",
    "robust_rbf_q1",
    "weighted_deterministic_rbf_q1",
    "weighted_robust_rbf_q1",
)
MODEL_CONFIG = {
    "deterministic_rbf_q1": {"rho": 0.0, "class_sensitive": False},
    "robust_rbf_q1": {"rho": R9_RHO, "class_sensitive": False},
    "weighted_deterministic_rbf_q1": {"rho": 0.0, "class_sensitive": True},
    "weighted_robust_rbf_q1": {"rho": R9_RHO, "class_sensitive": True},
}
TASKS = tuple(
    (dataset, seed, model)
    for dataset, seed in (
        ("blood_transfusion", 15000),
        ("blood_transfusion", 15001),
        ("parkinson", 15000),
    )
    for model in MODELS
)
R8_REFERENCE = {
    (15000, False): {"dynamic_range": 9.312843120225047e18, "abs_max": 873064.7845930157},
    (15000, True): {"dynamic_range": 2.905659864223687e21, "abs_max": 272401163.71655595},
    (15001, False): {"dynamic_range": 2.5733146767049654e20, "abs_max": 1943139.7651334845},
    (15001, True): {"dynamic_range": 6.0288429350985144e22, "abs_max": 455244924.02672154},
}
FROZEN_HASHES = {
    "ddr_mksvm/v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
    "ddr_mksvm/v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
}
CODE_FILES = (
    "ddr_mksvm/robust_bounded_rbf.py",
    "run_robust_bounded_rbf.py",
    "modal_robust_bounded_rbf.py",
    "analyze_robust_bounded_rbf.py",
)


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def prepared_task(dataset: str, seed: int, model: str):
    Xtr, ytr, Xte, yte, old_kernel, split, inventory = prepare(
        dataset, seed, "paper_configured"
    )
    classes = sorted(np.unique(ytr).tolist())
    if len(classes) != 2:
        raise RuntimeError("R9 exposed qualification currently requires binary data")
    negative, positive = classes[0], classes[-1]
    order = np.concatenate(
        (np.flatnonzero(ytr == positive), np.flatnonzero(ytr == negative))
    )
    Xfit = Xtr[order]
    labels = ytr[order]
    yfit = np.where(labels == positive, 1.0, -1.0)
    kernel = rbf_spec(Xfit)
    cfg = MODEL_CONFIG[model]
    delta = rbf_uncertainty_delta(
        Xfit, labels, cfg["rho"], R9_P, alpha=kernel.alpha
    )
    K = gram(Xfit, None, kernel)
    c, A, b, bounds, weights = build_r9_q1_lp(
        K,
        yfit,
        1e-3,
        delta,
        tau=R9_TAU,
        class_sensitive=cfg["class_sensitive"],
    )
    canonical = np.concatenate((np.abs(A).ravel(), np.abs(b), np.abs(c)))
    positive_values = canonical[canonical > 0]
    kernel_positive = np.abs(K[K != 0])
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "Xte": Xte,
        "yte": yte,
        "Xfit": Xfit,
        "labels": labels,
        "yfit": yfit,
        "negative": negative,
        "positive": positive,
        "order": order,
        "kernel": kernel,
        "delta": delta,
        "split": split,
        "inventory": inventory,
        "replaced_kernel": old_kernel.__dict__,
        "first_nu_canonical_abs_max": float(np.max(positive_values)),
        "first_nu_canonical_abs_min_positive": float(np.min(positive_values)),
        "first_nu_canonical_dynamic_range": float(np.max(positive_values) / np.min(positive_values)),
        "kernel_abs_max": float(np.max(np.abs(K))),
        "kernel_abs_min_positive": float(np.min(kernel_positive)),
        "kernel_zero_count": int(K.size - np.count_nonzero(K)),
        "kernel_dynamic_range": float(np.max(kernel_positive) / np.min(kernel_positive)),
        "max_delta": float(np.max(delta)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "n_variables": int(c.size),
        "n_inequalities": int(A.shape[0]),
        "matrix_nnz": int(np.count_nonzero(A)),
        "all_inputs_finite": bool(
            np.isfinite(K).all()
            and np.isfinite(delta).all()
            and np.isfinite(A).all()
            and np.isfinite(b).all()
            and np.isfinite(c).all()
        ),
    }


def evaluate_one(dataset, seed, model, fingerprint, checkpoint):
    if checkpoint.exists():
        stored = json.loads(checkpoint.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != fingerprint:
            raise RuntimeError("checkpoint fingerprint mismatch")
        return stored
    task = prepared_task(dataset, seed, model)
    cfg = MODEL_CONFIG[model]
    started = time.perf_counter()
    fitted, calls = fit_binary(
        task["Xfit"],
        task["yfit"],
        task["labels"],
        cfg["rho"],
        R9_P,
        tau=R9_TAU,
        class_sensitive=cfg["class_sensitive"],
    )
    runtime = time.perf_counter() - started
    for row in calls:
        row.update(
            dataset=dataset,
            seed=int(seed),
            model=model,
            phase="outer_candidate",
        )
    selected = [row for row in calls if row["selected"]]
    if len(selected) != 1:
        raise RuntimeError("exactly one selected nu is required")
    selected = selected[0]
    scores = fitted.decision(task["Xte"])
    if not np.isfinite(scores).all():
        raise RuntimeError("nonfinite R9 scores")
    predictions = np.where(scores > 0, task["positive"], task["negative"])
    labels = sorted(np.unique(task["yte"]).tolist())
    precision, recall, class_f1, support = precision_recall_fscore_support(
        task["yte"], predictions, labels=labels, zero_division=0
    )
    train_counts = {label: int(np.sum(task["ytr"] == label)) for label in np.unique(task["ytr"])}
    minority = min(train_counts, key=lambda label: (train_counts[label], label))
    max_violation = float(max(row["max_constraint_violation"] for row in calls))
    accepted = bool(
        all(row["solver_success"] and row["solver_status_code"] == 0 for row in calls)
        and max_violation <= FEASIBILITY_TOLERANCE
        and np.isfinite(predictions).all()
        and runtime <= 600.0
    )
    majority_only = bool(len(np.unique(predictions)) == 1)
    reference = R8_REFERENCE.get((int(seed), cfg["rho"] > 0)) if dataset == "blood_transfusion" else None
    record = {
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "q": 1,
        "p": R9_P,
        "rho": cfg["rho"],
        "tau": R9_TAU if cfg["class_sensitive"] else 0.0,
        "class_sensitive": cfg["class_sensitive"],
        "profile": "r9_bounded_rbf_on_frozen_transform",
        "preprocessing": task["split"]["preprocessing"],
        "preprocessing_scope": task["split"]["preprocessing_scope"],
        "kernel": "rbf",
        "alpha": float(fitted.kernel.alpha),
        "alpha_rule": "max_training_feature_sample_std_ddof1",
        "selected_nu": float(selected["nu"]),
        "threshold": float(fitted.b),
        "gamma": float(fitted.gamma),
        "solver": selected["solver"],
        "solver_status": selected["solver_status"],
        "solver_status_code": int(selected["solver_status_code"]),
        "objective_value": float(selected["objective_value"]),
        "selected_constraint_max_violation": float(selected["max_constraint_violation"]),
        "all_calls_max_constraint_violation": max_violation,
        "all_solver_calls_accepted": accepted,
        "solver_calls": len(calls),
        "solver_iterations": int(sum(row["solver_iterations"] for row in calls)),
        "selected_solver_iterations": int(selected["solver_iterations"]),
        "runtime_s": float(runtime),
        "tie_count": int(np.sum(scores == 0)),
        "majority_only_prediction": majority_only,
        "test_error": float(np.mean(predictions != task["yte"])),
        "balanced_accuracy": float(balanced_accuracy_score(task["yte"], predictions)),
        "macro_f1": float(f1_score(task["yte"], predictions, average="macro", zero_division=0)),
        "minority_class": int(minority),
        "minority_recall": float(recall[labels.index(minority)]),
        "per_class_recall": json.dumps(dict(zip(map(str, labels), map(float, recall))), sort_keys=True),
        "class_precision": json.dumps(dict(zip(map(str, labels), map(float, precision))), sort_keys=True),
        "class_f1": json.dumps(dict(zip(map(str, labels), map(float, class_f1))), sort_keys=True),
        "class_support": json.dumps(dict(zip(map(str, labels), map(int, support))), sort_keys=True),
        "confusion_matrix": json.dumps(confusion_matrix(task["yte"], predictions, labels=labels).tolist()),
        "prediction_hash": array_hash(predictions, "<i8"),
        "score_hash": array_hash(scores),
        "true_label_hash": array_hash(task["yte"], "<i8"),
        "split_hash": hashlib.sha256((task["split"]["train_hash"] + task["split"]["test_hash"]).encode()).hexdigest(),
        "kernel_abs_max": task["kernel_abs_max"],
        "kernel_abs_min_positive": task["kernel_abs_min_positive"],
        "kernel_zero_count": task["kernel_zero_count"],
        "kernel_dynamic_range": task["kernel_dynamic_range"],
        "first_nu_canonical_abs_max": task["first_nu_canonical_abs_max"],
        "first_nu_canonical_abs_min_positive": task["first_nu_canonical_abs_min_positive"],
        "first_nu_canonical_dynamic_range": task["first_nu_canonical_dynamic_range"],
        "r8_reference_abs_max": None if reference is None else reference["abs_max"],
        "r8_reference_dynamic_range": None if reference is None else reference["dynamic_range"],
        "r8_to_r9_abs_max_improvement": None if reference is None else reference["abs_max"] / task["first_nu_canonical_abs_max"],
        "max_delta": task["max_delta"],
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    payload = {
        "fingerprint": fingerprint,
        "record": record,
        "split": task["split"],
        "dataset_inventory": task["inventory"],
        "replaced_kernel": task["replaced_kernel"],
        "solver_diagnostics": calls,
        "fitted_parameters": {
            "training_order_within_split": task["order"].tolist(),
            "positive_class": int(task["positive"]),
            "negative_class": int(task["negative"]),
            "u": fitted.u.tolist(),
            "xi": fitted.xi.tolist(),
            "gamma": float(fitted.gamma),
            "b": float(fitted.b),
            "delta": fitted.delta.tolist(),
        },
        "predictions": np.asarray(predictions, dtype=int).tolist(),
        "true_labels": np.asarray(task["yte"], dtype=int).tolist(),
        "scores": np.asarray(scores, dtype=float).tolist(),
    }
    atomic_json(checkpoint, jsonable(payload))
    print(f"complete {dataset} {seed} {model} accepted={accepted} runtime={runtime:.3f}s", flush=True)
    return payload


def preserve_failure(path, dataset, seed, model, fingerprint, exception):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        task = prepared_task(dataset, seed, model)
        diagnostic = {key: value for key, value in task.items() if key in {
            "first_nu_canonical_abs_max", "first_nu_canonical_abs_min_positive",
            "first_nu_canonical_dynamic_range", "kernel_abs_max",
            "kernel_abs_min_positive", "kernel_zero_count", "kernel_dynamic_range",
            "max_delta", "n_variables", "n_inequalities", "matrix_nnz",
            "all_inputs_finite", "weight_min", "weight_max",
        }}
        diagnostic["alpha"] = task["kernel"].alpha
    except Exception as diagnostic_exception:
        diagnostic = {"diagnostic_error": f"{type(diagnostic_exception).__name__}: {diagnostic_exception}"}
    payload = {
        "fingerprint": fingerprint,
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "exception_type": type(exception).__name__,
        "exception_text": str(exception),
        "traceback": traceback.format_exc(),
        "problem_diagnostics": diagnostic,
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    atomic_json(path, jsonable(payload))
    return payload


def aggregate(run_dir: Path) -> dict:
    jobs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "checkpoints").glob("*.json"))]
    failures = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "failures").glob("*.json"))]
    records = pd.DataFrame([job["record"] for job in jobs])
    calls = pd.DataFrame([row for job in jobs for row in job["solver_diagnostics"]])
    class_rows = []
    for _, row in records.iterrows():
        for label, recall in json.loads(row.per_class_recall).items():
            class_rows.append({"dataset": row.dataset, "seed": row.seed, "model": row.model, "class": label, "recall": recall})
    class_metrics = pd.DataFrame(class_rows)
    feasibility = records[["dataset", "seed", "model", "solver_status_code", "all_calls_max_constraint_violation", "all_solver_calls_accepted"]].copy() if not records.empty else pd.DataFrame()
    split_registry = pd.DataFrame([job["split"] for job in jobs]).drop_duplicates(subset=["dataset", "seed"]) if jobs else pd.DataFrame()
    coefficient_ranges = records[["dataset", "seed", "model", "alpha", "kernel_abs_max", "kernel_abs_min_positive", "kernel_zero_count", "kernel_dynamic_range", "first_nu_canonical_abs_max", "first_nu_canonical_abs_min_positive", "first_nu_canonical_dynamic_range", "r8_reference_abs_max", "r8_reference_dynamic_range", "r8_to_r9_abs_max_improvement"]].copy() if not records.empty else pd.DataFrame()
    summary = records.groupby("model", sort=False).agg(completed=("model", "size"), error=("test_error", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean"), minority_recall=("minority_recall", "mean"), runtime_s=("runtime_s", "sum"), max_violation=("all_calls_max_constraint_violation", "max")).reset_index() if not records.empty else pd.DataFrame()
    for name, frame in (
        ("per_run.csv", records),
        ("summary.csv", summary),
        ("solver_calls.csv", calls),
        ("solver_diagnostics.csv", calls),
        ("class_metrics.csv", class_metrics),
        ("feasibility.csv", feasibility),
        ("coefficient_ranges.csv", coefficient_ranges),
        ("split_registry.csv", split_registry),
        ("failures.csv", pd.DataFrame(failures)),
    ):
        atomic_csv(run_dir / name, frame)
    blood = records[records.dataset == "blood_transfusion"] if not records.empty else records
    decision = bool(
        len(records) == len(TASKS)
        and not failures
        and len(blood) == 8
        and blood.all_solver_calls_accepted.all()
        and (blood.all_calls_max_constraint_violation <= FEASIBILITY_TOLERANCE).all()
        and (blood.runtime_s <= 600.0).all()
        and (blood.r8_to_r9_abs_max_improvement >= 1e3).all()
    )
    result = {
        "verdict": (
            "R9 IMPLEMENTATION QUALIFIED — fresh Gate8 may proceed"
            if decision
            else "R9 IMPLEMENTATION NOT QUALIFIED — stop Blood robust development"
        ),
        "expected_models": len(TASKS),
        "completed_models": int(len(records)),
        "failed_models": int(len(failures)),
        "blood_expected_models": 8,
        "blood_accepted_models": int(blood.all_solver_calls_accepted.sum()) if not blood.empty else 0,
        "maximum_blood_violation": None if blood.empty else float(blood.all_calls_max_constraint_violation.max()),
        "minimum_abs_max_improvement": None if blood.empty else float(blood.r8_to_r9_abs_max_improvement.min()),
        "blood_majority_only_models": int(blood.majority_only_prediction.sum()) if not blood.empty else 0,
        "fresh_seeds_consumed": [],
        "gate8_launched": False,
        "gate24_launched": False,
        "confirmation_launched": False,
    }
    atomic_json(run_dir / "qualification.json", result)
    atomic_json(run_dir / "failure_report.json", failures)
    return result


def experiment_state(args):
    if not os.getenv("ROBUST_ALLOW_LOCAL_FIT") and not os.getenv("MODAL_IS_REMOTE"):
        raise RuntimeError("real fitting is Modal-only")
    run_name = Path(args.run_name)
    if run_name.is_absolute() or ".." in run_name.parts:
        raise ValueError("run name must be a safe relative path")
    for path, expected in FROZEN_HASHES.items():
        if digest(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {path: digest(path) for path in CODE_FILES}
    config = {
        "version": "r9-bounded-rbf-exposed-qualification-1",
        "tasks": [list(task) for task in TASKS],
        "models": list(MODELS),
        "q": 1,
        "p": R9_P,
        "rho": R9_RHO,
        "tau": R9_TAU,
        "kernel": "rbf",
        "alpha_rule": "max_training_feature_sample_std_ddof1",
        "nu_grid": [float(value) for value in np.logspace(-3, 0, 5)],
        "solver": "scipy.optimize.linprog(method=highs)",
        "solver_options": {"presolve": True, "time_limit": 600.0},
        "feasibility_tolerance": FEASIBILITY_TOLERANCE,
        "source_hashes": SOURCE_HASHES,
        "frozen_hashes": FROZEN_HASHES,
        "code_hashes": code_hashes,
        "r8_reference": {f"{seed}:{robust}": value for (seed, robust), value in R8_REFERENCE.items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "versions": {package: importlib.metadata.version(package) for package in ("numpy", "scipy", "pandas", "scikit-learn")}},
        "resources": "4 CPU, 8192 MiB, no GPU, <=4 workers",
        "local_git": os.getenv("ROBUST_LOCAL_GIT", "unknown"),
        "fresh_seed_ranges_blocked": ["15300-15307", "15400-15423", "14000-14095"],
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("run name has incompatible immutable configuration")
    if not config_path.exists():
        atomic_json(config_path, config)
    for path, sha in code_hashes.items():
        target = run_dir / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != sha:
            raise RuntimeError("immutable source snapshot mismatch")
        if not target.exists():
            shutil.copyfile(path, target)
    checkpoints = run_dir / "checkpoints"
    failures = run_dir / "failures"
    checkpoints.mkdir(exist_ok=True)
    failures.mkdir(exist_ok=True)
    manifest = run_dir / "manifest.json"
    return run_dir, manifest, checkpoints, failures, fingerprint


def run(args):
    run_dir, manifest, checkpoints, failures, fingerprint = experiment_state(args)
    if args.mode == "prepare":
        atomic_json(manifest, {"status": "running", "fingerprint": fingerprint, "expected_jobs": len(TASKS), "app_id": os.getenv("MODAL_APP_ID", "local")})
        print(json.dumps({"status": "prepared", "tasks": len(TASKS)}))
        return
    if not manifest.exists() or json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
        raise RuntimeError("matching prepare manifest required")
    if args.mode == "finalize":
        completed = len(list(checkpoints.glob("*.json"))) + len(list(failures.glob("*.json")))
        if completed != len(TASKS):
            raise RuntimeError(f"cannot finalize {completed}/{len(TASKS)} tasks")
        result = aggregate(run_dir)
        atomic_json(manifest, {"status": "complete", "fingerprint": fingerprint, "app_id": os.getenv("MODAL_APP_ID", "local"), **result})
        print(json.dumps(result, sort_keys=True))
        return
    start = 0 if args.task_start is None else args.task_start
    stop = len(TASKS) if args.task_stop is None else min(args.task_stop, len(TASKS))
    for dataset, seed, model in TASKS[start:stop]:
        stem = f"{dataset}-{seed}-{model}"
        checkpoint = checkpoints / f"{stem}.json"
        failure = failures / f"{stem}.json"
        if failure.exists():
            continue
        try:
            evaluate_one(dataset, seed, model, fingerprint, checkpoint)
        except Exception as exception:
            payload = preserve_failure(failure, dataset, seed, model, fingerprint, exception)
            print(f"failed {stem}: {payload['exception_text']}", flush=True)
    print(json.dumps({"status": "worker_complete", "task_start": start, "task_stop": stop}))


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--run-name", required=True)
    result.add_argument("--output-root", default="results/robust")
    result.add_argument("--mode", choices=("prepare", "worker", "finalize"), required=True)
    result.add_argument("--task-start", type=int)
    result.add_argument("--task-stop", type=int)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
