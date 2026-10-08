"""Checkpointed R8.1 exact-LP-coordinate qualification runner.

Real fitting is Modal-only.  The closed registries contain only Parkinson
15000 (equivalence) and Blood 15000/15001 (already exposed qualification).
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

from ddr_mksvm.lp_equilibration import positive_dynamic_range
from ddr_mksvm.robust_matlab_parity import array_hash, gram, uncertainty_delta
from ddr_mksvm.robust_source_aligned import (
    FROZEN_P,
    FROZEN_RHO,
    FROZEN_TAU,
    build_weighted_q1_lp,
    fit_binary as fit_unscaled,
)
from ddr_mksvm.robust_source_aligned_equilibrated import fit_binary as fit_scaled
from archive.research_scripts.run_robust_matlab_parity import SOURCE_HASHES, atomic_csv, atomic_json, digest, prepare

MODELS = (
    "deterministic_q1",
    "robust_q1",
    "weighted_deterministic_q1",
    "robust_q1_cs_sqrt",
)
MODEL_CONFIG = {
    "deterministic_q1": {"rho": 0.0, "class_sensitive": False},
    "robust_q1": {"rho": FROZEN_RHO, "class_sensitive": False},
    "weighted_deterministic_q1": {"rho": 0.0, "class_sensitive": True},
    "robust_q1_cs_sqrt": {"rho": FROZEN_RHO, "class_sensitive": True},
}
PHASE_TASKS = {
    "parkinson": tuple(
        ("parkinson", 15000, model, representation)
        for model in MODELS
        for representation in ("unscaled", "scaled")
    ),
    "blood": tuple(
        ("blood_transfusion", seed, model, "scaled")
        for seed in (15000, 15001)
        for model in MODELS
    ),
}
FEASIBILITY_TOLERANCE = 1.0e-7
CODE_FILES = (
    "ddr_mksvm/lp_equilibration.py",
    "ddr_mksvm/robust_source_aligned_equilibrated.py",
    "run_robust_lp_equilibration.py",
    "modal_robust_lp_equilibration.py",
)
FROZEN_HASHES = {
    "ddr_mksvm/v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
    "ddr_mksvm/v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
}


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def enrich_unscaled(row: dict, m: int, class_sensitive: bool) -> dict:
    result = dict(row)
    result.setdefault("solver_status_code", 0)
    result.setdefault("solver_success", True)
    result.setdefault("solver_options", {"method": "highs"})
    result.setdefault("crossover_iterations", None)
    result.setdefault("n_variables", 3 * m + 1)
    result.setdefault("n_inequalities", 3 * m)
    result.setdefault("uses_exact_lp_equilibration", False)
    result.setdefault("class_sensitive", class_sensitive)
    result.setdefault("max_original_inequality_violation", result["max_constraint_violation"])
    result.setdefault("max_original_bound_violation", 0.0)
    return result


def problem_coefficients(dataset: str, seed: int, model: str):
    Xtr, ytr, Xte, yte, kernel, split, inventory = prepare(
        dataset, seed, "paper_configured"
    )
    classes = sorted(np.unique(ytr).tolist())
    if len(classes) != 2:
        raise RuntimeError("R8.1 qualification registry is binary only")
    negative, positive = classes[0], classes[-1]
    order = np.concatenate(
        (np.flatnonzero(ytr == positive), np.flatnonzero(ytr == negative))
    )
    labels = ytr[order]
    Xfit = Xtr[order]
    yfit = np.where(labels == positive, 1.0, -1.0)
    cfg = MODEL_CONFIG[model]
    K = gram(Xfit, None, kernel)
    delta = uncertainty_delta(
        Xfit, labels, cfg["rho"], FROZEN_P, kernel, matlab_semantics=True
    )
    c, A, b, bounds, weights = build_weighted_q1_lp(
        K,
        yfit,
        1.0e-3,
        delta,
        tau=FROZEN_TAU,
        class_sensitive=cfg["class_sensitive"],
    )
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "Xte": Xte,
        "yte": yte,
        "kernel": kernel,
        "split": split,
        "inventory": inventory,
        "negative": negative,
        "positive": positive,
        "order": order,
        "labels": labels,
        "Xfit": Xfit,
        "yfit": yfit,
        "delta": delta,
        "first_nu_unscaled_constraint_dynamic_range": positive_dynamic_range(A),
        "first_nu_unscaled_canonical_dynamic_range": positive_dynamic_range(A, b, c),
        "first_nu_variables": int(c.size),
        "first_nu_constraints": int(A.shape[0]),
        "first_nu_nnz": int(np.count_nonzero(A)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
    }


def evaluate_one(dataset, seed, model, representation, fingerprint, checkpoint):
    if checkpoint.exists():
        stored = json.loads(checkpoint.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != fingerprint:
            raise RuntimeError(f"checkpoint fingerprint mismatch: {checkpoint}")
        return stored
    inputs = problem_coefficients(dataset, seed, model)
    cfg = MODEL_CONFIG[model]
    fitter = fit_scaled if representation == "scaled" else fit_unscaled
    started = time.perf_counter()
    fitted, calls = fitter(
        inputs["Xfit"],
        inputs["yfit"],
        inputs["labels"],
        inputs["kernel"],
        cfg["rho"],
        FROZEN_P,
        tau=FROZEN_TAU,
        class_sensitive=cfg["class_sensitive"],
    )
    runtime = time.perf_counter() - started
    if representation == "unscaled":
        calls = [enrich_unscaled(row, len(inputs["yfit"]), cfg["class_sensitive"]) for row in calls]
    for row in calls:
        row.update(
            dataset=dataset,
            seed=int(seed),
            model=model,
            representation=representation,
            phase="outer_candidate",
        )
    selected = [row for row in calls if row["selected"]]
    if len(selected) != 1:
        raise RuntimeError("exactly one selected nu is required")
    selected = selected[0]
    scores = fitted.decision(inputs["Xte"])
    if not np.isfinite(scores).all():
        raise RuntimeError("nonfinite prediction score")
    predictions = np.where(scores > 0, inputs["positive"], inputs["negative"])
    labels = sorted(np.unique(inputs["yte"]).tolist())
    precision, recall, class_f1, support = precision_recall_fscore_support(
        inputs["yte"], predictions, labels=labels, zero_division=0
    )
    counts = {label: int(np.sum(inputs["ytr"] == label)) for label in np.unique(inputs["ytr"])}
    minority = min(counts, key=lambda label: (counts[label], label))
    max_violation = float(max(row["max_constraint_violation"] for row in calls))
    accepted = bool(
        all(bool(row["solver_success"]) and int(row["solver_status_code"]) == 0 for row in calls)
        and max_violation <= FEASIBILITY_TOLERANCE
    )
    record = {
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "representation": representation,
        "q": 1,
        "tau": FROZEN_TAU if cfg["class_sensitive"] else 0.0,
        "p": FROZEN_P,
        "rho": cfg["rho"],
        "class_sensitive": cfg["class_sensitive"],
        "profile": inputs["split"]["profile"],
        "kernel": inputs["kernel"].kind,
        "kernel_parameters": json.dumps(inputs["kernel"].__dict__, sort_keys=True),
        "preprocessing": inputs["split"]["preprocessing"],
        "selected_nu": float(selected["nu"]),
        "threshold": float(fitted.b),
        "gamma": float(fitted.gamma),
        "threshold_left": float(selected["threshold_left"]),
        "threshold_right": float(selected["threshold_right"]),
        "threshold_descending": bool(selected["threshold_descending"]),
        "training_error": float(selected["training_error"]),
        "solver": selected["solver"],
        "solver_status": selected["solver_status"],
        "solver_status_code": int(selected["solver_status_code"]),
        "objective_value": float(selected["objective_value"]),
        "selected_constraint_max_violation": float(selected["max_constraint_violation"]),
        "all_calls_max_constraint_violation": max_violation,
        "all_solver_calls_accepted": accepted,
        "solver_calls": len(calls),
        "solver_iterations": int(sum(row["solver_iterations"] for row in calls)),
        "runtime_s": float(runtime),
        "tie_count": int(np.sum(scores == 0)),
        "test_error": float(np.mean(predictions != inputs["yte"])),
        "balanced_accuracy": float(balanced_accuracy_score(inputs["yte"], predictions)),
        "macro_f1": float(f1_score(inputs["yte"], predictions, average="macro", zero_division=0)),
        "minority_class": int(minority),
        "minority_recall": float(recall[labels.index(minority)]),
        "per_class_recall": json.dumps(dict(zip(map(str, labels), map(float, recall))), sort_keys=True),
        "class_precision": json.dumps(dict(zip(map(str, labels), map(float, precision))), sort_keys=True),
        "class_f1": json.dumps(dict(zip(map(str, labels), map(float, class_f1))), sort_keys=True),
        "class_support": json.dumps(dict(zip(map(str, labels), map(int, support))), sort_keys=True),
        "confusion_matrix": json.dumps(confusion_matrix(inputs["yte"], predictions, labels=labels).tolist()),
        "prediction_hash": array_hash(predictions, "<i8"),
        "score_hash": array_hash(scores),
        "true_label_hash": array_hash(inputs["yte"], "<i8"),
        "split_hash": hashlib.sha256((inputs["split"]["train_hash"] + inputs["split"]["test_hash"]).encode()).hexdigest(),
        "first_nu_unscaled_constraint_dynamic_range": inputs["first_nu_unscaled_constraint_dynamic_range"],
        "first_nu_unscaled_canonical_dynamic_range": inputs["first_nu_unscaled_canonical_dynamic_range"],
        "selected_scaled_constraint_dynamic_range": selected.get("scaled_constraint_dynamic_range"),
        "selected_scaled_canonical_dynamic_range": selected.get("scaled_canonical_dynamic_range"),
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    payload = {
        "fingerprint": fingerprint,
        "record": record,
        "split": inputs["split"],
        "dataset_inventory": inputs["inventory"],
        "solver_diagnostics": calls,
        "fitted_parameters": {
            "u": fitted.u.tolist(),
            "xi": fitted.xi.tolist(),
            "gamma": float(fitted.gamma),
            "b": float(fitted.b),
            "delta": fitted.delta.tolist(),
        },
        "predictions": np.asarray(predictions, dtype=int).tolist(),
        "true_labels": np.asarray(inputs["yte"], dtype=int).tolist(),
        "scores": np.asarray(scores, dtype=float).tolist(),
    }
    atomic_json(checkpoint, jsonable(payload))
    print(f"complete {dataset} {seed} {model} {representation} accepted={accepted}", flush=True)
    return payload


def preserve_failure(path, dataset, seed, model, representation, fingerprint, exception):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        inputs = problem_coefficients(dataset, seed, model)
        problem = {key: value for key, value in inputs.items() if key.startswith("first_nu_") or key.startswith("weight_")}
    except Exception as diagnostic_error:
        problem = {"diagnostic_error": f"{type(diagnostic_error).__name__}: {diagnostic_error}"}
    payload = {
        "fingerprint": fingerprint,
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "representation": representation,
        "exception_type": type(exception).__name__,
        "exception_text": str(exception),
        "traceback": traceback.format_exc(),
        "problem": problem,
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    atomic_json(path, jsonable(payload))
    return payload


def aggregate(run_dir, expected):
    jobs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "checkpoints").glob("*.json"))]
    failures = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "failures").glob("*.json"))]
    records = pd.DataFrame([job["record"] for job in jobs])
    calls = pd.DataFrame([row for job in jobs for row in job["solver_diagnostics"]])
    atomic_csv(run_dir / "per_run.csv", records)
    atomic_csv(run_dir / "solver_calls.csv", calls)
    atomic_csv(run_dir / "failures.csv", pd.DataFrame(failures))
    if records.empty:
        summary = pd.DataFrame()
    else:
        summary = records.groupby(["model", "representation"], sort=False).agg(
            completed=("model", "size"),
            error=("test_error", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            runtime_s=("runtime_s", "sum"),
            max_violation=("all_calls_max_constraint_violation", "max"),
        ).reset_index()
    atomic_csv(run_dir / "summary.csv", summary)
    if set(records.get("representation", [])) == {"unscaled", "scaled"}:
        pairs = []
        for model in MODELS:
            group = records[records.model == model].set_index("representation")
            if not {"unscaled", "scaled"}.issubset(group.index):
                continue
            u, s = group.loc["unscaled"], group.loc["scaled"]
            objective_relative = abs(s.objective_value - u.objective_value) / max(1.0, abs(u.objective_value))
            threshold_tolerance = 1e-7 * max(1.0, abs(u.threshold))
            pairs.append({
                "model": model,
                "selected_nu_equal": bool(s.selected_nu == u.selected_nu),
                "prediction_equal": bool(s.prediction_hash == u.prediction_hash),
                "tie_count_equal": bool(s.tie_count == u.tie_count),
                "threshold_abs_difference": float(abs(s.threshold - u.threshold)),
                "threshold_tolerance": float(threshold_tolerance),
                "threshold_equivalent": bool(abs(s.threshold - u.threshold) <= threshold_tolerance),
                "objective_relative_difference": float(objective_relative),
                "objective_equivalent": bool(objective_relative <= 1e-7),
                "scaled_max_violation": float(s.all_calls_max_constraint_violation),
                "scaled_feasible": bool(s.all_calls_max_constraint_violation <= FEASIBILITY_TOLERANCE),
                "status_no_regression": bool(s.all_solver_calls_accepted),
                "score_max_abs_difference": 0.0,
            })
        paired = pd.DataFrame(pairs)
    else:
        paired = pd.DataFrame()
    # Compute score differences without depending on row/checkpoint ordering.
    if not paired.empty:
        lookup = {(job["record"]["model"], job["record"]["representation"]): job for job in jobs}
        for index, row in paired.iterrows():
            unscaled_scores = np.asarray(lookup[(row.model, "unscaled")]["scores"], dtype=float)
            scaled_scores = np.asarray(lookup[(row.model, "scaled")]["scores"], dtype=float)
            difference = np.abs(scaled_scores - unscaled_scores)
            paired.loc[index, "score_max_abs_difference"] = float(np.max(difference))
            paired.loc[index, "score_max_relative_difference"] = float(np.max(difference) / max(1.0, float(np.max(np.abs(unscaled_scores)))))
    atomic_csv(run_dir / "equivalence.csv", paired)
    phase_pass = bool(
        len(records) == expected
        and not failures
        and records.all_solver_calls_accepted.all()
        and (
            paired.empty
            or (
                paired.selected_nu_equal.all()
                and paired.prediction_equal.all()
                and paired.tie_count_equal.all()
                and paired.threshold_equivalent.all()
                and paired.objective_equivalent.all()
                and paired.scaled_feasible.all()
                and paired.status_no_regression.all()
            )
        )
    )
    result = {
        "phase_pass": phase_pass,
        "expected": expected,
        "completed": int(len(records)),
        "failures": int(len(failures)),
        "maximum_violation": None if records.empty else float(records.all_calls_max_constraint_violation.max()),
        "fresh_seeds_consumed": [],
    }
    atomic_json(run_dir / "qualification.json", result)
    return result


def state(args):
    if not os.getenv("ROBUST_ALLOW_LOCAL_FIT") and not os.getenv("MODAL_IS_REMOTE"):
        raise RuntimeError("real fitting is Modal-only")
    if args.phase not in PHASE_TASKS:
        raise ValueError("unknown qualification phase")
    run_name = Path(args.run_name)
    if run_name.is_absolute() or ".." in run_name.parts:
        raise ValueError("run_name must be relative")
    for path, expected in FROZEN_HASHES.items():
        if digest(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {path: digest(path) for path in CODE_FILES}
    config = {
        "version": "r8.1-exact-lp-equilibration-1",
        "phase": args.phase,
        "tasks": [list(task) for task in PHASE_TASKS[args.phase]],
        "equilibration": {"passes": 8, "norm": "max", "update_min": 1e-4, "update_max": 1e4, "row_augments_b": True, "column_augments_c": True},
        "model": {"q": 1, "tau": FROZEN_TAU, "p": FROZEN_P, "rho": FROZEN_RHO},
        "solver": "scipy.optimize.linprog(method=highs)",
        "solver_options": {"presolve": True, "time_limit": 600.0},
        "feasibility_tolerance": FEASIBILITY_TOLERANCE,
        "source_hashes": SOURCE_HASHES,
        "frozen_hashes": FROZEN_HASHES,
        "code_hashes": code_hashes,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "versions": {package: importlib.metadata.version(package) for package in ("numpy", "scipy", "pandas", "scikit-learn")}},
        "local_git": os.getenv("ROBUST_LOCAL_GIT", "unknown"),
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("incompatible immutable configuration")
    if not config_path.exists():
        atomic_json(config_path, config)
    for path, sha in code_hashes.items():
        target = run_dir / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        elif digest(target) != sha:
            raise RuntimeError(f"source snapshot mismatch: {target}")
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "failures").mkdir(exist_ok=True)
    manifest = run_dir / "manifest.json"
    return run_dir, manifest, fingerprint, PHASE_TASKS[args.phase]


def run(args):
    run_dir, manifest, fingerprint, tasks = state(args)
    if args.mode == "prepare":
        atomic_json(manifest, {"status": "running", "fingerprint": fingerprint, "expected": len(tasks), "app_id": os.getenv("MODAL_APP_ID", "local")})
        print(json.dumps({"status": "prepared", "tasks": len(tasks)}))
        return
    if not manifest.exists() or json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
        raise RuntimeError("prepare must establish the matching manifest")
    if args.mode == "finalize":
        completed = len(list((run_dir / "checkpoints").glob("*.json"))) + len(list((run_dir / "failures").glob("*.json")))
        if completed != len(tasks):
            raise RuntimeError(f"cannot finalize incomplete run {completed}/{len(tasks)}")
        result = aggregate(run_dir, len(tasks))
        atomic_json(manifest, {"status": "complete", "fingerprint": fingerprint, "app_id": os.getenv("MODAL_APP_ID", "local"), **result})
        print(json.dumps(result, sort_keys=True))
        return
    start = 0 if args.task_start is None else args.task_start
    stop = len(tasks) if args.task_stop is None else min(args.task_stop, len(tasks))
    for dataset, seed, model, representation in tasks[start:stop]:
        stem = f"{dataset}-{seed}-{model}-{representation}"
        checkpoint = run_dir / "checkpoints" / f"{stem}.json"
        failure = run_dir / "failures" / f"{stem}.json"
        if failure.exists():
            continue
        try:
            evaluate_one(dataset, seed, model, representation, fingerprint, checkpoint)
        except Exception as exception:
            payload = preserve_failure(failure, dataset, seed, model, representation, fingerprint, exception)
            print(f"failed {stem}: {payload['exception_text']}", flush=True)
    print(json.dumps({"status": "worker_complete", "task_start": start, "task_stop": stop}))


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--run-name", required=True)
    result.add_argument("--phase", choices=tuple(PHASE_TASKS), required=True)
    result.add_argument("--output-root", default="results/robust")
    result.add_argument("--mode", choices=("prepare", "worker", "finalize"), required=True)
    result.add_argument("--task-start", type=int)
    result.add_argument("--task-stop", type=int)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
