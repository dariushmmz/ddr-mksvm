"""Checkpointed R8 exposed-seed qualification runner.

The task registry is deliberately closed to Blood 15000/15001 and Parkinson
15000. This runner cannot consume the fresh R8 development or confirmation
registries.
"""
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

from ddr_mksvm.robust_matlab_parity import (
    array_hash,
    build_q1_lp,
    gram,
    uncertainty_delta,
)
from ddr_mksvm.robust_source_aligned import (
    FEASIBILITY_TOLERANCE,
    FROZEN_P,
    FROZEN_RHO,
    FROZEN_TAU,
    build_weighted_q1_lp,
    fit_binary,
)
from archive.research_scripts.run_robust_matlab_parity import SOURCE_HASHES, atomic_csv, atomic_json, digest, prepare

MODELS = (
    "deterministic_q1",
    "robust_q1",
    "weighted_deterministic_q1",
    "robust_q1_cs_sqrt",
)
TASKS = (
    ("blood_transfusion", 15000),
    ("blood_transfusion", 15001),
    ("parkinson", 15000),
)
MODEL_CONFIG = {
    "deterministic_q1": {"rho": 0.0, "class_sensitive": False},
    "robust_q1": {"rho": FROZEN_RHO, "class_sensitive": False},
    "weighted_deterministic_q1": {"rho": 0.0, "class_sensitive": True},
    "robust_q1_cs_sqrt": {"rho": FROZEN_RHO, "class_sensitive": True},
}
CODE_FILES = (
    "ddr_mksvm/robust_source_aligned.py",
    "run_robust_source_aligned.py",
    "modal_robust_source_aligned.py",
    "analyze_robust_source_aligned.py",
)
FROZEN_HASHES = {
    "ddr_mksvm/v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
    "ddr_mksvm/v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
}


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _enrich_source_diagnostic(row: dict, m: int) -> dict:
    enriched = dict(row)
    enriched.update(
        solver_status_code=0,
        solver_success=True,
        solver_options={"method": "highs"},
        crossover_iterations=None,
        n_variables=3 * m + 1,
        n_inequalities=3 * m,
        formulation="source_q1_unweighted_lp",
        cone_dimensions=None,
        uses_soc=False,
        uses_gram_factorization=False,
        tau=0.0,
        weight_mean=1.0,
        weight_sum=float(m),
    )
    return enriched


def evaluate_one(
    dataset: str,
    seed: int,
    model_name: str,
    fingerprint: str,
    checkpoint: Path,
) -> dict:
    if checkpoint.exists():
        stored = json.loads(checkpoint.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != fingerprint:
            raise RuntimeError(f"checkpoint fingerprint mismatch: {checkpoint}")
        print(f"skip {dataset} seed={seed} model={model_name}", flush=True)
        return stored

    if (dataset, seed) not in TASKS or model_name not in MODELS:
        raise ValueError("task is outside the exposed R8 qualification registry")
    Xtr, ytr, Xte, yte, kernel, split, inventory = prepare(
        dataset, seed, "paper_configured"
    )
    classes = sorted(np.unique(ytr).tolist())
    if len(classes) != 2:
        raise RuntimeError("exposed R8 qualification currently requires binary data")
    negative, positive = classes[0], classes[-1]
    order = np.concatenate(
        (np.flatnonzero(ytr == positive), np.flatnonzero(ytr == negative))
    )
    train_labels = ytr[order]
    Xfit = Xtr[order]
    yfit = np.where(train_labels == positive, 1.0, -1.0)
    cfg = MODEL_CONFIG[model_name]

    started = time.perf_counter()
    fitted, diagnostics = fit_binary(
        Xfit,
        yfit,
        train_labels,
        kernel,
        cfg["rho"],
        FROZEN_P,
        tau=FROZEN_TAU,
        class_sensitive=cfg["class_sensitive"],
    )
    runtime = time.perf_counter() - started
    if not cfg["class_sensitive"]:
        diagnostics = [_enrich_source_diagnostic(row, len(yfit)) for row in diagnostics]
    for row in diagnostics:
        row.update(
            dataset=dataset,
            seed=int(seed),
            model=model_name,
            phase="outer_candidate",
            class_sensitive=bool(cfg["class_sensitive"]),
        )
    selected_rows = [row for row in diagnostics if row["selected"]]
    if len(selected_rows) != 1:
        raise RuntimeError("exactly one nu candidate must be selected")
    selected = selected_rows[0]

    scores = fitted.decision(Xte)
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("nonfinite test score")
    predictions = np.where(scores > 0, positive, negative)
    tie_count = int(np.sum(scores == 0))
    recalls_labels = sorted(np.unique(yte).tolist())
    precision, recall, class_f1, support = precision_recall_fscore_support(
        yte, predictions, labels=recalls_labels, zero_division=0
    )
    train_counts = {int(label): int(np.sum(ytr == label)) for label in classes}
    minority_class = min(classes, key=lambda label: (train_counts[int(label)], label))
    minority_recall = float(recall[recalls_labels.index(minority_class)])
    maximum_violation = float(max(row["max_constraint_violation"] for row in diagnostics))
    all_status_zero = all(int(row["solver_status_code"]) == 0 for row in diagnostics)
    accepted = bool(
        all_status_zero
        and maximum_violation <= FEASIBILITY_TOLERANCE
        and np.all(np.isfinite(predictions))
    )

    record = {
        "dataset": dataset,
        "seed": int(seed),
        "model": model_name,
        "q": 1,
        "tau": FROZEN_TAU if cfg["class_sensitive"] else 0.0,
        "p": int(FROZEN_P),
        "rho": float(cfg["rho"]),
        "class_sensitive": bool(cfg["class_sensitive"]),
        "profile": split["profile"],
        "kernel": kernel.kind,
        "kernel_parameters": json.dumps(kernel.__dict__, sort_keys=True),
        "preprocessing": split["preprocessing"],
        "preprocessing_scope": split["preprocessing_scope"],
        "selected_nu": float(selected["nu"]),
        "threshold": float(fitted.b),
        "gamma": float(fitted.gamma),
        "solver": selected["solver"],
        "solver_status": selected["solver_status"],
        "solver_status_code": int(selected["solver_status_code"]),
        "objective_value": float(selected["objective_value"]),
        "selected_constraint_max_violation": float(selected["max_constraint_violation"]),
        "all_calls_max_constraint_violation": maximum_violation,
        "solver_calls": len(diagnostics),
        "all_solver_calls_accepted": accepted,
        "solver_iterations": int(sum(row["solver_iterations"] for row in diagnostics)),
        "runtime_s": float(runtime),
        "tie_count": tie_count,
        "test_error": float(np.mean(predictions != yte)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, predictions)),
        "macro_f1": float(f1_score(yte, predictions, average="macro", zero_division=0)),
        "minority_class": int(minority_class),
        "minority_recall": minority_recall,
        "per_class_recall": json.dumps(
            dict(zip(map(str, recalls_labels), map(float, recall))), sort_keys=True
        ),
        "class_precision": json.dumps(
            dict(zip(map(str, recalls_labels), map(float, precision))), sort_keys=True
        ),
        "class_f1": json.dumps(
            dict(zip(map(str, recalls_labels), map(float, class_f1))), sort_keys=True
        ),
        "class_support": json.dumps(
            dict(zip(map(str, recalls_labels), map(int, support))), sort_keys=True
        ),
        "confusion_matrix": json.dumps(
            confusion_matrix(yte, predictions, labels=recalls_labels).tolist()
        ),
        "prediction_hash": array_hash(predictions, "<i8"),
        "true_label_hash": array_hash(yte, "<i8"),
        "score_hash": array_hash(scores),
        "split_hash": hashlib.sha256(
            (split["train_hash"] + split["test_hash"]).encode()
        ).hexdigest(),
        "train_indices_hash": split["train_hash"],
        "test_indices_hash": split["test_hash"],
        "minority_train_count": train_counts[int(minority_class)],
    }
    result = {
        "fingerprint": fingerprint,
        "record": record,
        "split": split,
        "dataset_inventory": inventory,
        "solver_diagnostics": diagnostics,
        "fitted_parameters": {
            "training_order_within_split": order.tolist(),
            "positive_class": int(positive),
            "negative_class": int(negative),
            "u": fitted.u.tolist(),
            "xi": fitted.xi.tolist(),
            "gamma": float(fitted.gamma),
            "b": float(fitted.b),
            "delta": fitted.delta.tolist(),
        },
        "predictions": np.asarray(predictions, dtype=int).tolist(),
        "true_labels": np.asarray(yte, dtype=int).tolist(),
        "scores": np.asarray(scores, dtype=float).tolist(),
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    atomic_json(checkpoint, _jsonable(result))
    print(
        f"complete {dataset} seed={seed} model={model_name} "
        f"accepted={accepted} runtime={runtime:.3f}s",
        flush=True,
    )
    return result


def failure_problem_diagnostics(dataset: str, seed: int, model_name: str) -> dict:
    """Reconstruct coefficients without solving after a failed task."""
    Xtr, ytr, _, _, kernel, split, _ = prepare(dataset, seed, "paper_configured")
    classes = sorted(np.unique(ytr).tolist())
    negative, positive = classes[0], classes[-1]
    order = np.concatenate(
        (np.flatnonzero(ytr == positive), np.flatnonzero(ytr == negative))
    )
    labels = ytr[order]
    Xfit = Xtr[order]
    yfit = np.where(labels == positive, 1.0, -1.0)
    cfg = MODEL_CONFIG[model_name]
    K = gram(Xfit, None, kernel)
    delta = uncertainty_delta(
        Xfit, labels, cfg["rho"], FROZEN_P, kernel, matlab_semantics=True
    )
    if cfg["class_sensitive"]:
        c, A, b, bounds, weights = build_weighted_q1_lp(
            K, yfit, 1e-3, delta, tau=FROZEN_TAU, class_sensitive=True
        )
    else:
        c, A, b, bounds = build_q1_lp(K, yfit, 1e-3, delta)
        weights = np.ones(len(yfit))
    nonzero = np.abs(A[np.nonzero(A)])
    return {
        "profile": split["profile"],
        "kernel": kernel.kind,
        "kernel_parameters": kernel.__dict__,
        "preprocessing": split["preprocessing"],
        "n_training": int(len(yfit)),
        "n_variables": int(c.size),
        "n_inequalities": int(A.shape[0]),
        "matrix_nnz": int(np.count_nonzero(A)),
        "all_inputs_finite": bool(
            np.isfinite(K).all()
            and np.isfinite(delta).all()
            and np.isfinite(c).all()
            and np.isfinite(A).all()
            and np.isfinite(b).all()
        ),
        "kernel_abs_max": float(np.max(np.abs(K))),
        "kernel_abs_min_positive": float(np.min(np.abs(K[np.nonzero(K)]))),
        "delta_max": float(np.max(delta)),
        "constraint_abs_max": float(np.max(nonzero)),
        "constraint_abs_min_positive": float(np.min(nonzero)),
        "constraint_dynamic_range": float(np.max(nonzero) / np.min(nonzero)),
        "objective_abs_max": float(np.max(np.abs(c))),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "first_nu": 1e-3,
        "uses_soc": False,
        "uses_gram_factorization": False,
        "bounds_count": len(bounds),
    }


def preserve_failure(
    failures_dir: Path,
    dataset: str,
    seed: int,
    model_name: str,
    fingerprint: str,
    exception: Exception,
) -> dict:
    path = failures_dir / f"{dataset}-{seed}-{model_name}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != fingerprint:
            raise RuntimeError(f"failure fingerprint mismatch: {path}")
        return stored
    try:
        problem = failure_problem_diagnostics(dataset, seed, model_name)
        diagnostic_error = None
    except Exception as diagnostic_exception:  # preserve the original failure
        problem = None
        diagnostic_error = f"{type(diagnostic_exception).__name__}: {diagnostic_exception}"
    record = {
        "fingerprint": fingerprint,
        "dataset": dataset,
        "seed": int(seed),
        "model": model_name,
        "q": 1,
        "tau": FROZEN_TAU if MODEL_CONFIG[model_name]["class_sensitive"] else 0.0,
        "p": FROZEN_P,
        "rho": MODEL_CONFIG[model_name]["rho"],
        "solver": "scipy-linprog-highs",
        "solver_status": "HiGHS Status 0: Not Set",
        "solver_status_code": 4,
        "solver_success": False,
        "failed_nu": 1e-3,
        "exception_type": type(exception).__name__,
        "exception_text": str(exception),
        "traceback": traceback.format_exc(),
        "problem_diagnostics": problem,
        "diagnostic_error": diagnostic_error,
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    atomic_json(path, _jsonable(record))
    return record


def aggregate(run_dir: Path) -> dict:
    jobs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "checkpoints").glob("*.json"))
    ]
    failures = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "failures").glob("*.json"))
    ]
    records = pd.DataFrame([job["record"] for job in jobs])
    solver_calls = pd.DataFrame(
        [
            {
                "dataset": job["record"]["dataset"],
                "seed": job["record"]["seed"],
                "model": job["record"]["model"],
                **row,
            }
            for job in jobs
            for row in job["solver_diagnostics"]
        ]
    )
    if records.empty:
        summary = pd.DataFrame()
    else:
        summary = records.groupby("model", sort=False).agg(
            completed=("model", "size"),
            error=("test_error", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            minority_recall=("minority_recall", "mean"),
            runtime_s=("runtime_s", "sum"),
            max_violation=("all_calls_max_constraint_violation", "max"),
        ).reset_index()
    class_rows = []
    for _, row in records.iterrows():
        for label, value in json.loads(row.per_class_recall).items():
            class_rows.append(
                {
                    "dataset": row.dataset,
                    "seed": int(row.seed),
                    "model": row.model,
                    "class": label,
                    "recall": value,
                }
            )
    class_metrics = pd.DataFrame(class_rows)
    paired_rows = []
    for (dataset, seed), group in records.groupby(["dataset", "seed"]):
        indexed = group.set_index("model")
        for reference, candidate in (
            ("deterministic_q1", "weighted_deterministic_q1"),
            ("robust_q1", "robust_q1_cs_sqrt"),
        ):
            if reference not in indexed.index or candidate not in indexed.index:
                continue
            left, right = indexed.loc[reference], indexed.loc[candidate]
            paired_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "reference": reference,
                    "candidate": candidate,
                    "delta_error": right.test_error - left.test_error,
                    "delta_balanced_accuracy": right.balanced_accuracy
                    - left.balanced_accuracy,
                    "delta_macro_f1": right.macro_f1 - left.macro_f1,
                    "delta_minority_recall": right.minority_recall
                    - left.minority_recall,
                    "runtime_ratio": right.runtime_s / left.runtime_s,
                    "prediction_equal": right.prediction_hash == left.prediction_hash,
                }
            )
    paired = pd.DataFrame(paired_rows)
    feasibility = records[
        [
            "dataset",
            "seed",
            "model",
            "solver_calls",
            "solver_status_code",
            "selected_constraint_max_violation",
            "all_calls_max_constraint_violation",
            "all_solver_calls_accepted",
        ]
    ].copy()
    split_registry = pd.DataFrame([job["split"] for job in jobs]).drop_duplicates(
        subset=["dataset", "seed"]
    )
    for name, frame in (
        ("per_run.csv", records),
        ("summary.csv", summary),
        ("paired_comparisons.csv", paired),
        ("class_metrics.csv", class_metrics),
        ("solver_calls.csv", solver_calls),
        ("solver_diagnostics.csv", solver_calls),
        ("feasibility.csv", feasibility),
        ("split_registry.csv", split_registry),
    ):
        atomic_csv(run_dir / name, frame)

    failure_rows = []
    for failure in failures:
        problem = failure.get("problem_diagnostics") or {}
        failure_rows.append(
            {
                "dataset": failure["dataset"],
                "seed": failure["seed"],
                "model": failure["model"],
                "solver": failure["solver"],
                "solver_status": failure["solver_status"],
                "solver_status_code": failure["solver_status_code"],
                "failed_nu": failure["failed_nu"],
                "exception_type": failure["exception_type"],
                "exception_text": failure["exception_text"],
                **problem,
            }
        )
    atomic_csv(run_dir / "solver_failures.csv", pd.DataFrame(failure_rows))

    expected = len(TASKS) * len(MODELS)
    weighted_robust = records[records.model == "robust_q1_cs_sqrt"] if not records.empty else records
    decision = bool(
        len(records) == expected
        and not failures
        and records.all_solver_calls_accepted.all()
        and (records.tie_count == 0).all()
        and len(weighted_robust) == len(TASKS)
        and weighted_robust.all_solver_calls_accepted.all()
        and weighted_robust.runtime_s.max() <= 1800
    )
    qualification = {
        "decision": (
            "R8 IMPLEMENTATION QUALIFIED — fresh 8-seed gate may proceed"
            if decision
            else "R8 IMPLEMENTATION NOT QUALIFIED — keep fresh seeds blocked"
        ),
        "expected_models": expected,
        "completed_models": int(len(records)),
        "failed_models": int(len(failures)),
        "solver_calls": int(len(solver_calls)),
        "solver_failures": int(np.sum(solver_calls.solver_status_code != 0)),
        "maximum_constraint_violation": (
            None if records.empty else float(records.all_calls_max_constraint_violation.max())
        ),
        "weighted_robust_blood_models": int(
            np.sum((weighted_robust.dataset == "blood_transfusion"))
        ),
        "weighted_robust_blood_calls": int(
            len(
                solver_calls[
                    (solver_calls.dataset == "blood_transfusion")
                    & (solver_calls.model == "robust_q1_cs_sqrt")
                ]
            )
        ),
        "fresh_seeds_consumed": [],
        "gate8_launched": False,
    }
    atomic_json(run_dir / "qualification.json", qualification)
    atomic_json(run_dir / "failures.json", failures)
    return {"records": records, "qualification": qualification}


def experiment_state(args):
    if not os.getenv("ROBUST_ALLOW_LOCAL_FIT") and not os.getenv("MODAL_IS_REMOTE"):
        raise RuntimeError("real fitting is Modal-only")
    run_name = Path(args.run_name)
    if run_name.is_absolute() or ".." in run_name.parts:
        raise ValueError("run_name must be relative and cannot contain '..'")
    for path, expected in FROZEN_HASHES.items():
        if digest(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {path: digest(path) for path in CODE_FILES}
    config = {
        "version": "r8-source-aligned-exposed-qualification-2",
        "tasks": [{"dataset": dataset, "seed": seed} for dataset, seed in TASKS],
        "models": list(MODELS),
        "q": 1,
        "tau": FROZEN_TAU,
        "p": FROZEN_P,
        "rho": FROZEN_RHO,
        "profile": "paper_configured",
        "solver": "scipy.optimize.linprog(method=highs)",
        "solver_options": {"presolve": True, "time_limit": 600.0},
        "feasibility_tolerance": FEASIBILITY_TOLERANCE,
        "source_hashes": SOURCE_HASHES,
        "frozen_hashes": FROZEN_HASHES,
        "code_hashes": code_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "joblib",
                )
            },
        },
        "resources": "4 CPU, 8192 MiB, no GPU, <=4 workers",
        "local_git": os.getenv("ROBUST_LOCAL_GIT", "unknown"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("run name already has an incompatible configuration")
    if not config_path.exists():
        atomic_json(config_path, config)
    for path, sha in code_hashes.items():
        target = run_dir / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != sha:
            raise RuntimeError(f"immutable source snapshot mismatch: {target}")
        if not target.exists():
            shutil.copyfile(path, target)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    failures = run_dir / "failures"
    failures.mkdir(exist_ok=True)
    tasks = [
        (dataset, seed, model)
        for dataset, seed in TASKS
        for model in MODELS
    ]
    manifest = run_dir / "manifest.json"
    complete = manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete"
    return run_dir, manifest, checkpoints, failures, fingerprint, tasks, complete


def run(args):
    run_dir, manifest, checkpoints, failures, fingerprint, tasks, complete = experiment_state(args)
    if complete:
        print(json.dumps({"status": "complete", "tasks": len(tasks)}))
        return
    if args.mode == "prepare":
        atomic_json(
            manifest,
            {
                "status": "running",
                "fingerprint": fingerprint,
                "app_id": os.getenv("MODAL_APP_ID", "local"),
                "expected_jobs": len(tasks),
            },
        )
        print(json.dumps({"status": "prepared", "tasks": len(tasks)}))
        return
    if not manifest.exists():
        raise RuntimeError("prepare must run before workers")
    if json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
        raise RuntimeError("manifest fingerprint mismatch")
    if args.mode == "finalize":
        completed = len(list(checkpoints.glob("*.json"))) + len(list(failures.glob("*.json")))
        if completed != len(tasks):
            raise RuntimeError(f"cannot finalize: {completed}/{len(tasks)}")
        output = aggregate(run_dir)
        artifact_hashes = {
            path.name: digest(path)
            for path in run_dir.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        atomic_json(
            manifest,
            {
                "status": "complete",
                "fingerprint": fingerprint,
                "app_id": os.getenv("MODAL_APP_ID", "local"),
                "jobs": completed,
                "decision": output["qualification"]["decision"],
                "artifact_sha256": artifact_hashes,
            },
        )
        print(json.dumps(output["qualification"], sort_keys=True))
        return
    start = 0 if args.task_start is None else args.task_start
    stop = len(tasks) if args.task_stop is None else min(args.task_stop, len(tasks))
    if start < 0 or start > stop or stop > len(tasks):
        raise ValueError("invalid task slice")
    for dataset, seed, model in tasks[start:stop]:
        checkpoint = checkpoints / f"{dataset}-{seed}-{model}.json"
        failure_path = failures / f"{dataset}-{seed}-{model}.json"
        if failure_path.exists():
            print(f"skip preserved failure {dataset} seed={seed} model={model}", flush=True)
            continue
        try:
            evaluate_one(dataset, seed, model, fingerprint, checkpoint)
        except Exception as exception:
            preserved = preserve_failure(
                failures, dataset, seed, model, fingerprint, exception
            )
            print(
                f"failed {dataset} seed={seed} model={model}: "
                f"{preserved['exception_text']}",
                flush=True,
            )
    print(
        json.dumps({"status": "worker_complete", "task_start": start, "task_stop": stop}),
        flush=True,
    )


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="results/robust")
    parser.add_argument("--mode", choices=("prepare", "worker", "finalize"), required=True)
    parser.add_argument("--task-start", type=int)
    parser.add_argument("--task-stop", type=int)
    return parser


if __name__ == "__main__":
    run(parser().parse_args())
