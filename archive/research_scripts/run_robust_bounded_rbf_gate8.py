"""Frozen R9 Gate8 runner for seeds 15300--15307 only."""
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
from scipy.stats import rankdata, t, wilcoxon
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support

from ddr_mksvm.robust_bounded_rbf import (
    FEASIBILITY_TOLERANCE,
    R9_P,
    R9_RHO,
    R9_TAU,
    fit_binary,
    fit_ova,
)
from ddr_mksvm.robust_matlab_parity import array_hash, predict_ova
from archive.research_scripts.run_robust_matlab_parity import SOURCE_HASHES, atomic_csv, atomic_json, digest, load, prepare

DATASETS = (
    "blood_transfusion",
    "parkinson",
    "mammographicmass_binary",
    "iris",
)
SEEDS = tuple(range(15300, 15308))
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
    for dataset in DATASETS
    for seed in SEEDS
    for model in MODELS
)
PRIMARY_REFERENCE = "weighted_deterministic_rbf_q1"
PRIMARY_CANDIDATE = "weighted_robust_rbf_q1"
FROZEN_HASHES = {
    "ddr_mksvm/v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
    "ddr_mksvm/v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
}
CODE_FILES = (
    "ddr_mksvm/robust_bounded_rbf.py",
    "run_robust_bounded_rbf_gate8.py",
    "modal_robust_bounded_rbf_gate8.py",
    "analyze_robust_bounded_rbf_gate8.py",
)
GATE_GUARDS = {
    "expected_tasks": 128,
    "max_original_coordinate_violation": 1e-7,
    "candidate_majority_only_max": 0,
    "candidate_model_runtime_max_s": 600.0,
    "dataset_median_runtime_ratio_max": 2.0,
    "pooled_mean_delta_error_max": 0.0,
    "pooled_mean_delta_balanced_accuracy_min": 0.0,
    "pooled_mean_delta_macro_f1_min": 0.0,
    "balanced_accuracy_wins_minus_losses_min": 0,
    "datasets_nonnegative_mean_delta_balanced_accuracy_min": 2,
    "dataset_mean_delta_safety_recall_min": -0.05,
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


def _task_data(dataset: str, seed: int):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("task is outside the frozen R9 Gate8 registry")
    Xtr, ytr, Xte, yte, replaced_kernel, split, inventory = prepare(
        dataset, seed, "paper_configured"
    )
    return Xtr, ytr, Xte, yte, replaced_kernel, split, inventory


def evaluate_one(dataset, seed, model, fingerprint, checkpoint):
    if checkpoint.exists():
        stored = json.loads(checkpoint.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != fingerprint:
            raise RuntimeError("checkpoint fingerprint mismatch")
        return stored
    if (dataset, seed, model) not in TASKS:
        raise ValueError("task is outside frozen Gate8")
    Xtr, ytr, Xte, yte, replaced_kernel, split, inventory = _task_data(dataset, seed)
    cfg = MODEL_CONFIG[model]
    classes = sorted(np.unique(ytr).tolist())
    started = time.perf_counter()
    if len(classes) == 2:
        negative, positive = classes[0], classes[-1]
        order = np.concatenate(
            (np.flatnonzero(ytr == positive), np.flatnonzero(ytr == negative))
        )
        Xfit = Xtr[order]
        original_labels = ytr[order]
        binary = np.where(original_labels == positive, 1.0, -1.0)
        fitted, diagnostics = fit_binary(
            Xfit,
            binary,
            original_labels,
            cfg["rho"],
            R9_P,
            tau=R9_TAU,
            class_sensitive=cfg["class_sensitive"],
        )
        scores = fitted.decision(Xte)
        predictions = np.where(scores > 0, positive, negative)
        tie_count = int(np.sum(scores == 0))
        selected_rows = [row for row in diagnostics if row["selected"]]
        fitted_parameters = [{
            "positive_class": int(positive),
            "negative_class": int(negative),
            "training_order_within_split": order.tolist(),
            "u": fitted.u.tolist(),
            "xi": fitted.xi.tolist(),
            "gamma": float(fitted.gamma),
            "b": float(fitted.b),
            "delta": fitted.delta.tolist(),
        }]
        alpha = float(fitted.kernel.alpha)
    else:
        fitted, diagnostics = fit_ova(
            Xtr,
            ytr,
            cfg["rho"],
            R9_P,
            tau=R9_TAU,
            class_sensitive=cfg["class_sensitive"],
        )
        predictions, scores, tie_count = predict_ova(
            fitted, Xte, tie_policy="matlab_error"
        )
        selected_rows = [row for row in diagnostics if row["selected"]]
        fitted_parameters = [{
            "positive_class": int(item.positive_class),
            "u": item.u.tolist(),
            "xi": item.xi.tolist(),
            "gamma": float(item.gamma),
            "b": float(item.b),
            "delta": item.delta.tolist(),
        } for item in fitted]
        alpha_values = {float(item.kernel.alpha) for item in fitted}
        if len(alpha_values) != 1:
            raise RuntimeError("OVA tasks must share the frozen R9 alpha")
        alpha = alpha_values.pop()
    runtime = time.perf_counter() - started
    if not np.isfinite(scores).all() or not np.isfinite(predictions).all():
        raise RuntimeError("nonfinite scores or predictions")
    if tie_count:
        raise RuntimeError(f"source-undefined prediction ties: {tie_count}")
    expected_calls = 5 if len(classes) == 2 else 5 * len(classes)
    if len(diagnostics) != expected_calls or len(selected_rows) != (1 if len(classes) == 2 else len(classes)):
        raise RuntimeError("unexpected solver-call or selected-task count")
    for row in diagnostics:
        row.update(dataset=dataset, seed=int(seed), model=model, phase="outer_candidate")
    labels = sorted(np.unique(yte).tolist())
    precision, recall, class_f1, support = precision_recall_fscore_support(
        yte, predictions, labels=labels, zero_division=0
    )
    train_counts = {label: int(np.sum(ytr == label)) for label in classes}
    minority = min(classes, key=lambda label: (train_counts[label], label))
    minority_recall = float(recall[labels.index(minority)])
    minimum_class_recall = float(np.min(recall))
    safety_recall = minority_recall if len(classes) == 2 else minimum_class_recall
    max_violation = float(max(row["max_constraint_violation"] for row in diagnostics))
    accepted = bool(
        all(row["solver_success"] and int(row["solver_status_code"]) == 0 for row in diagnostics)
        and all(np.isfinite(row["objective_value"]) for row in diagnostics)
        and max_violation <= FEASIBILITY_TOLERANCE
        and runtime <= GATE_GUARDS["candidate_model_runtime_max_s"]
    )
    selected_nu = [float(row["nu"]) for row in selected_rows]
    thresholds = [float(row["b"]) for row in selected_rows]
    record = {
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "q": 1,
        "p": R9_P,
        "rho": cfg["rho"],
        "tau": R9_TAU if cfg["class_sensitive"] else 0.0,
        "class_sensitive": cfg["class_sensitive"],
        "decomposition": "binary" if len(classes) == 2 else "ova",
        "profile": "r9_bounded_rbf_on_frozen_transform",
        "preprocessing": split["preprocessing"],
        "preprocessing_scope": split["preprocessing_scope"],
        "kernel": "rbf",
        "alpha": alpha,
        "alpha_rule": "max_training_feature_sample_std_ddof1",
        "selected_nu": json.dumps(selected_nu) if len(selected_nu) > 1 else selected_nu[0],
        "threshold": json.dumps(thresholds) if len(thresholds) > 1 else thresholds[0],
        "gamma": json.dumps([float(row["gamma"]) for row in selected_rows]) if len(selected_rows) > 1 else float(selected_rows[0]["gamma"]),
        "objective_value": float(sum(row["objective_value"] for row in selected_rows)),
        "solver": "scipy-linprog-highs",
        "solver_status": "+".join(sorted({row["solver_status"] for row in selected_rows})),
        "solver_status_code": 0,
        "solver_calls": len(diagnostics),
        "solver_iterations": int(sum(row["solver_iterations"] for row in diagnostics)),
        "selected_solver_iterations": int(sum(row["solver_iterations"] for row in selected_rows)),
        "all_calls_max_constraint_violation": max_violation,
        "all_solver_calls_accepted": accepted,
        "runtime_s": float(runtime),
        "tie_count": tie_count,
        "majority_only_prediction": bool(len(np.unique(predictions)) == 1),
        "test_error": float(np.mean(predictions != yte)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, predictions)),
        "macro_f1": float(f1_score(yte, predictions, average="macro", zero_division=0)),
        "minority_or_rare_class": int(minority),
        "minority_or_rare_recall": minority_recall,
        "minimum_class_recall": minimum_class_recall,
        "safety_recall": safety_recall,
        "per_class_recall": json.dumps(dict(zip(map(str, labels), map(float, recall))), sort_keys=True),
        "class_precision": json.dumps(dict(zip(map(str, labels), map(float, precision))), sort_keys=True),
        "class_f1": json.dumps(dict(zip(map(str, labels), map(float, class_f1))), sort_keys=True),
        "class_support": json.dumps(dict(zip(map(str, labels), map(int, support))), sort_keys=True),
        "confusion_matrix": json.dumps(confusion_matrix(yte, predictions, labels=labels).tolist()),
        "prediction_hash": array_hash(predictions, "<i8"),
        "score_hash": array_hash(scores),
        "true_label_hash": array_hash(yte, "<i8"),
        "split_hash": hashlib.sha256((split["train_hash"] + split["test_hash"]).encode()).hexdigest(),
        "kernel_abs_max": float(max(row["kernel_abs_max"] for row in diagnostics)),
        "kernel_abs_min_positive": float(min(row["kernel_abs_min_positive"] for row in diagnostics)),
        "kernel_zero_count": int(max(row["kernel_zero_count"] for row in diagnostics)),
        "kernel_dynamic_range": float(max(row["kernel_dynamic_range"] for row in diagnostics)),
        "canonical_abs_max": float(max(row["canonical_abs_max"] for row in diagnostics)),
        "canonical_abs_min_positive": float(min(row["canonical_abs_min_positive"] for row in diagnostics)),
        "canonical_dynamic_range": float(max(row["canonical_dynamic_range"] for row in diagnostics)),
        "max_delta": float(max(row["max_delta"] for row in diagnostics)),
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    payload = {
        "fingerprint": fingerprint,
        "record": record,
        "split": split,
        "dataset_inventory": inventory,
        "replaced_kernel": replaced_kernel.__dict__,
        "solver_diagnostics": diagnostics,
        "fitted_parameters": fitted_parameters,
        "predictions": np.asarray(predictions, dtype=int).tolist(),
        "true_labels": np.asarray(yte, dtype=int).tolist(),
        "scores": np.asarray(scores, dtype=float).tolist(),
    }
    atomic_json(checkpoint, jsonable(payload))
    print(f"complete {dataset} {seed} {model} accepted={accepted} runtime={runtime:.3f}s", flush=True)
    return payload


def preserve_failure(path, dataset, seed, model, fingerprint, exception):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "fingerprint": fingerprint,
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "exception_type": type(exception).__name__,
        "exception_text": str(exception),
        "traceback": traceback.format_exc(),
        "app_id": os.getenv("MODAL_APP_ID", "local"),
    }
    atomic_json(path, payload)
    return payload


def _wtl(values: np.ndarray, higher_is_better: bool, tolerance: float = 1e-12):
    signed = values if higher_is_better else -values
    return (
        int(np.sum(signed > tolerance)),
        int(np.sum(np.abs(signed) <= tolerance)),
        int(np.sum(signed < -tolerance)),
    )


def _paired_statistics(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) > 1:
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        critical = float(t.ppf(0.975, len(values) - 1))
        lower, upper = mean - critical * sem, mean + critical * sem
    else:
        lower = upper = mean
    nonzero = values[values != 0]
    if nonzero.size == 0:
        statistic, pvalue, effect = 0.0, 1.0, 0.0
    else:
        test = wilcoxon(nonzero, alternative="two-sided", method="auto")
        ranks = rankdata(np.abs(nonzero))
        positive = float(np.sum(ranks[nonzero > 0]))
        negative = float(np.sum(ranks[nonzero < 0]))
        statistic, pvalue = float(test.statistic), float(test.pvalue)
        effect = float((positive - negative) / (positive + negative))
    return mean, float(lower), float(upper), statistic, pvalue, effect


def primary_pairs(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, seed), group in records.groupby(["dataset", "seed"], sort=False):
        indexed = group.set_index("model")
        if PRIMARY_REFERENCE not in indexed.index or PRIMARY_CANDIDATE not in indexed.index:
            continue
        reference = indexed.loc[PRIMARY_REFERENCE]
        candidate = indexed.loc[PRIMARY_CANDIDATE]
        rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "reference": PRIMARY_REFERENCE,
            "candidate": PRIMARY_CANDIDATE,
            "reference_error": reference.test_error,
            "candidate_error": candidate.test_error,
            "delta_error": candidate.test_error - reference.test_error,
            "delta_balanced_accuracy": candidate.balanced_accuracy - reference.balanced_accuracy,
            "delta_macro_f1": candidate.macro_f1 - reference.macro_f1,
            "delta_safety_recall": candidate.safety_recall - reference.safety_recall,
            "reference_safety_recall": reference.safety_recall,
            "candidate_safety_recall": candidate.safety_recall,
            "runtime_ratio": candidate.runtime_s / reference.runtime_s,
            "prediction_equal": candidate.prediction_hash == reference.prediction_hash,
            "candidate_majority_only": bool(candidate.majority_only_prediction),
        })
    return pd.DataFrame(rows)


def statistics_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in pairs.groupby("dataset", sort=False):
        for metric, higher in (
            ("delta_error", False),
            ("delta_balanced_accuracy", True),
            ("delta_macro_f1", True),
            ("delta_safety_recall", True),
        ):
            values = group[metric].to_numpy(float)
            mean, lower, upper, statistic, pvalue, effect = _paired_statistics(values)
            wins, ties, losses = _wtl(values, higher)
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "n": len(values),
                "mean_delta": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": pvalue,
                "rank_biserial_effect": effect,
                "interpretation": "screening_only",
            })
    return pd.DataFrame(rows)


def gate_decision(records: pd.DataFrame, failures: list, pairs: pd.DataFrame):
    candidate = records[records.model == PRIMARY_CANDIDATE]
    dataset_means = pairs.groupby("dataset").agg(
        delta_error=("delta_error", "mean"),
        delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
        delta_macro_f1=("delta_macro_f1", "mean"),
        delta_safety_recall=("delta_safety_recall", "mean"),
        median_runtime_ratio=("runtime_ratio", "median"),
    ).reset_index()
    ba_wins, ba_ties, ba_losses = _wtl(
        pairs.delta_balanced_accuracy.to_numpy(float), True
    ) if len(pairs) else (0, 0, 0)
    guards = {
        "all_tasks_accounted": len(records) + len(failures) == len(TASKS),
        "all_tasks_completed": len(records) == len(TASKS) and not failures,
        "all_solver_calls_accepted": bool(len(records) == len(TASKS) and records.all_solver_calls_accepted.all()),
        "all_finite": bool(len(records) == len(TASKS) and np.isfinite(records[["objective_value", "test_error", "balanced_accuracy", "macro_f1", "runtime_s"]].to_numpy(float)).all()),
        "feasibility": bool(len(records) == len(TASKS) and records.all_calls_max_constraint_violation.max() <= GATE_GUARDS["max_original_coordinate_violation"]),
        "no_ties": bool(len(records) == len(TASKS) and (records.tie_count == 0).all()),
        "candidate_no_majority_only": bool(len(candidate) == len(DATASETS) * len(SEEDS) and candidate.majority_only_prediction.sum() <= GATE_GUARDS["candidate_majority_only_max"]),
        "candidate_runtime": bool(len(candidate) == len(DATASETS) * len(SEEDS) and candidate.runtime_s.max() <= GATE_GUARDS["candidate_model_runtime_max_s"]),
        "runtime_ratio": bool(len(dataset_means) == len(DATASETS) and dataset_means.median_runtime_ratio.max() <= GATE_GUARDS["dataset_median_runtime_ratio_max"]),
        "pooled_error": bool(len(pairs) == len(DATASETS) * len(SEEDS) and pairs.delta_error.mean() <= GATE_GUARDS["pooled_mean_delta_error_max"]),
        "pooled_balanced_accuracy": bool(len(pairs) == len(DATASETS) * len(SEEDS) and pairs.delta_balanced_accuracy.mean() >= GATE_GUARDS["pooled_mean_delta_balanced_accuracy_min"]),
        "pooled_macro_f1": bool(len(pairs) == len(DATASETS) * len(SEEDS) and pairs.delta_macro_f1.mean() >= GATE_GUARDS["pooled_mean_delta_macro_f1_min"]),
        "balanced_accuracy_wtl": ba_wins - ba_losses >= GATE_GUARDS["balanced_accuracy_wins_minus_losses_min"],
        "dataset_consistency": bool(len(dataset_means) == len(DATASETS) and np.sum(dataset_means.delta_balanced_accuracy >= 0) >= GATE_GUARDS["datasets_nonnegative_mean_delta_balanced_accuracy_min"]),
        "safety_recall": bool(len(dataset_means) == len(DATASETS) and dataset_means.delta_safety_recall.min() >= GATE_GUARDS["dataset_mean_delta_safety_recall_min"]),
    }
    return guards, dataset_means, (ba_wins, ba_ties, ba_losses)


def aggregate(run_dir: Path):
    jobs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "checkpoints").glob("*.json"))]
    failures = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "failures").glob("*.json"))]
    records = pd.DataFrame([job["record"] for job in jobs])
    calls = pd.DataFrame([row for job in jobs for row in job["solver_diagnostics"]])
    pairs = primary_pairs(records) if not records.empty else pd.DataFrame()
    statistics = statistics_table(pairs) if not pairs.empty else pd.DataFrame()
    guards, dataset_means, ba_wtl = gate_decision(records, failures, pairs)
    class_rows = []
    for _, row in records.iterrows():
        for label, recall in json.loads(row.per_class_recall).items():
            class_rows.append({"dataset": row.dataset, "seed": row.seed, "model": row.model, "class": label, "recall": recall})
    summary = records.groupby(["dataset", "model"], sort=False).agg(
        seeds=("seed", "nunique"),
        error=("test_error", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        safety_recall=("safety_recall", "mean"),
        runtime_mean_s=("runtime_s", "mean"),
        runtime_median_s=("runtime_s", "median"),
        runtime_max_s=("runtime_s", "max"),
        max_violation=("all_calls_max_constraint_violation", "max"),
        majority_only=("majority_only_prediction", "sum"),
    ).reset_index() if not records.empty else pd.DataFrame()
    split_registry = pd.DataFrame([job["split"] for job in jobs]).drop_duplicates(subset=["dataset", "seed"]) if jobs else pd.DataFrame()
    feasibility = records[["dataset", "seed", "model", "solver_calls", "solver_status_code", "all_calls_max_constraint_violation", "all_solver_calls_accepted"]].copy() if not records.empty else pd.DataFrame()
    coefficient_ranges = records[["dataset", "seed", "model", "alpha", "kernel_abs_max", "kernel_abs_min_positive", "kernel_zero_count", "kernel_dynamic_range", "canonical_abs_max", "canonical_abs_min_positive", "canonical_dynamic_range"]].copy() if not records.empty else pd.DataFrame()
    for name, frame in (
        ("per_run.csv", records),
        ("summary.csv", summary),
        ("solver_calls.csv", calls),
        ("solver_diagnostics.csv", calls),
        ("class_metrics.csv", pd.DataFrame(class_rows)),
        ("feasibility.csv", feasibility),
        ("coefficient_ranges.csv", coefficient_ranges),
        ("split_registry.csv", split_registry),
        ("paired_primary.csv", pairs),
        ("paired_comparisons.csv", pairs),
        ("paired_statistics.csv", statistics),
        ("dataset_paired_summary.csv", dataset_means),
        ("failures.csv", pd.DataFrame(failures)),
    ):
        atomic_csv(run_dir / name, frame)
    passed = bool(all(guards.values()))
    result = {
        "verdict": "R9 GATE8 PASSED — authorize Gate24" if passed else "R9 GATE8 FAILED — Gate24 remains blocked",
        "expected_tasks": len(TASKS),
        "completed_tasks": int(len(records)),
        "failed_tasks": int(len(failures)),
        "solver_calls": int(len(calls)),
        "solver_status_failures": int(np.sum(calls.solver_status_code != 0)) if not calls.empty else 0,
        "maximum_violation": None if records.empty else float(records.all_calls_max_constraint_violation.max()),
        "candidate_majority_only": int(records[records.model == PRIMARY_CANDIDATE].majority_only_prediction.sum()) if not records.empty else 0,
        "pooled_mean_deltas": {} if pairs.empty else {
            "error": float(pairs.delta_error.mean()),
            "balanced_accuracy": float(pairs.delta_balanced_accuracy.mean()),
            "macro_f1": float(pairs.delta_macro_f1.mean()),
            "safety_recall": float(pairs.delta_safety_recall.mean()),
        },
        "balanced_accuracy_wtl": {"wins": ba_wtl[0], "ties": ba_wtl[1], "losses": ba_wtl[2]},
        "guards": guards,
        "fresh_seeds_consumed": list(SEEDS),
        "gate24_launched": False,
        "confirmation_launched": False,
    }
    atomic_json(run_dir / "gate_decision.json", result)
    atomic_json(run_dir / "failure_report.json", failures)
    return result


def experiment_state(args):
    if not os.getenv("ROBUST_ALLOW_LOCAL_FIT") and not os.getenv("MODAL_IS_REMOTE"):
        raise RuntimeError("real fitting is Modal-only")
    run_name = Path(args.run_name)
    if run_name.is_absolute() or ".." in run_name.parts:
        raise ValueError("unsafe run name")
    for path, expected in FROZEN_HASHES.items():
        if digest(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {path: digest(path) for path in CODE_FILES}
    # Inventory construction must not evaluate a fresh Gate8 split.  `load`
    # hashes the immutable source dataset only; split 15300 is first touched by
    # an actual worker after the complete protocol/configuration is frozen.
    inventories = {dataset: load(dataset)[2] for dataset in DATASETS}
    config = {
        "version": "r9-bounded-rbf-gate8-1",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "primary_comparison": {"reference": PRIMARY_REFERENCE, "candidate": PRIMARY_CANDIDATE},
        "q": 1,
        "kernel": "rbf",
        "alpha_rule": "max_training_feature_sample_std_ddof1",
        "tau": R9_TAU,
        "p": R9_P,
        "rho": R9_RHO,
        "nu_grid": [float(value) for value in np.logspace(-3, 0, 5)],
        "profile": "paper_configured_transforms_replaced_by_frozen_r9_rbf",
        "solver": "scipy.optimize.linprog(method=highs)",
        "solver_options": {"presolve": True, "time_limit": 600.0},
        "gate_guards": GATE_GUARDS,
        "source_hashes": SOURCE_HASHES,
        "frozen_hashes": FROZEN_HASHES,
        "code_hashes": code_hashes,
        "dataset_inventory": inventories,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "versions": {package: importlib.metadata.version(package) for package in ("numpy", "scipy", "pandas", "scikit-learn")}},
        "resources": "4 CPU, 8192 MiB, no GPU, <=4 workers",
        "local_git": os.getenv("ROBUST_LOCAL_GIT", "unknown"),
        "blocked_seed_ranges": ["15400-15423", "14000-14095"],
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("incompatible immutable Gate8 configuration")
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
    return run_dir, run_dir / "manifest.json", checkpoints, failures, fingerprint


def run(args):
    run_dir, manifest, checkpoints, failures, fingerprint = experiment_state(args)
    if args.mode == "prepare":
        atomic_json(manifest, {"status": "running", "fingerprint": fingerprint, "expected_jobs": len(TASKS), "app_id": os.getenv("MODAL_APP_ID", "local")})
        print(json.dumps({"status": "prepared", "tasks": len(TASKS)}))
        return
    if not manifest.exists() or json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
        raise RuntimeError("matching prepare manifest required")
    if args.mode == "finalize":
        accounted = len(list(checkpoints.glob("*.json"))) + len(list(failures.glob("*.json")))
        if accounted != len(TASKS):
            raise RuntimeError(f"cannot finalize {accounted}/{len(TASKS)}")
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
