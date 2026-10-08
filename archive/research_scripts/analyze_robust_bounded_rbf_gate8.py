"""Validate and print a completed immutable R9 Gate8 result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(run_dir: Path):
    decision = json.loads((run_dir / "gate_decision.json").read_text(encoding="utf-8"))
    records = pd.read_csv(run_dir / "per_run.csv")
    pairs = pd.read_csv(run_dir / "paired_primary.csv")
    calls = pd.read_csv(run_dir / "solver_calls.csv")
    failures = pd.read_csv(run_dir / "failures.csv")
    paired_models = {"weighted_deterministic_rbf_q1", "weighted_robust_rbf_q1"}
    expected_pairs = int(
        records[records.model.isin(paired_models)]
        .groupby(["dataset", "seed"])
        .model.nunique()
        .eq(2)
        .sum()
    )
    if (
        len(records) != decision["completed_tasks"]
        or len(failures) != decision["failed_tasks"]
        or len(records) + len(failures) != decision["expected_tasks"]
        or len(pairs) != expected_pairs
    ):
        raise RuntimeError("Gate8 aggregate count mismatch")
    result = {
        **decision,
        "paired_rows": int(len(pairs)),
        "paired_rows_expected_from_completed_models": expected_pairs,
        "solver_calls_recounted": int(len(calls)),
        "solver_status_failures_recounted": int((calls.solver_status_code != 0).sum()),
        "failed_solver_calls_recorded_separately": int(len(failures)),
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    accounting = []
    for dataset in (
        "blood_transfusion", "parkinson", "mammographicmass_binary", "iris"
    ):
        for seed in decision["fresh_seeds_consumed"]:
            for model in (
                "deterministic_rbf_q1", "robust_rbf_q1",
                "weighted_deterministic_rbf_q1", "weighted_robust_rbf_q1",
            ):
                complete = records[(records.dataset == dataset) & (records.seed == seed) & (records.model == model)]
                failed = failures[(failures.dataset == dataset) & (failures.seed == seed) & (failures.model == model)]
                accounting.append({
                    "dataset": dataset, "seed": seed, "model": model,
                    "task_status": "complete" if len(complete) else "failed",
                    "failure_type": None if failed.empty else failed.iloc[0].exception_type,
                    "failure_text": None if failed.empty else failed.iloc[0].exception_text,
                })
    pd.DataFrame(accounting).to_csv(analysis_dir / "task_accounting.csv", index=False)

    pair_fields = [
        "dataset", "seed", "test_error", "balanced_accuracy", "macro_f1",
        "per_class_recall", "minority_or_rare_class", "minority_or_rare_recall",
        "safety_recall", "selected_nu", "threshold", "prediction_hash",
        "confusion_matrix", "majority_only_prediction", "runtime_s",
        "solver_iterations", "all_calls_max_constraint_violation",
    ]
    reference = records[records.model == "weighted_deterministic_rbf_q1"][pair_fields].copy()
    candidate = records[records.model == "weighted_robust_rbf_q1"][pair_fields].copy()
    reference.columns = [column if column in ("dataset", "seed") else f"reference_{column}" for column in reference.columns]
    candidate.columns = [column if column in ("dataset", "seed") else f"candidate_{column}" for column in candidate.columns]
    detailed_pairs = reference.merge(candidate, on=["dataset", "seed"], how="inner", validate="one_to_one")
    for metric in ("test_error", "balanced_accuracy", "macro_f1", "safety_recall"):
        detailed_pairs[f"delta_{metric}"] = detailed_pairs[f"candidate_{metric}"] - detailed_pairs[f"reference_{metric}"]
    detailed_pairs.to_csv(analysis_dir / "paired_seed_details.csv", index=False)

    runtime = records.groupby(["dataset", "model"], sort=False).agg(
        completed_models=("seed", "size"),
        runtime_mean_s=("runtime_s", "mean"),
        runtime_median_s=("runtime_s", "median"),
        runtime_p95_s=("runtime_s", lambda values: values.quantile(.95)),
        runtime_max_s=("runtime_s", "max"),
        iterations_mean=("solver_iterations", "mean"),
        iterations_median=("solver_iterations", "median"),
        iterations_max=("solver_iterations", "max"),
        max_violation=("all_calls_max_constraint_violation", "max"),
        majority_only=("majority_only_prediction", "sum"),
    ).reset_index()
    runtime.to_csv(analysis_dir / "runtime_numerical_summary.csv", index=False)

    recalls = pd.read_csv(run_dir / "class_metrics.csv")
    recalls.groupby(["dataset", "model", "class"], sort=False).recall.agg(
        ["count", "mean", "std", "min", "max"]
    ).reset_index().to_csv(analysis_dir / "class_recall_summary.csv", index=False)
    records.groupby(["dataset", "model", "selected_nu"], sort=False, dropna=False).size().rename(
        "count"
    ).reset_index().to_csv(analysis_dir / "nu_selection_counts.csv", index=False)
    pd.DataFrame([
        {"guard": key, "passed": bool(value)} for key, value in decision["guards"].items()
    ]).to_csv(analysis_dir / "gate_guards.csv", index=False)

    failed_calls = failures[["dataset", "seed", "model", "exception_type", "exception_text"]].copy()
    failed_calls.insert(3, "phase", "inner_model_selection")
    failed_calls["nu"] = 0.001
    failed_calls["solver"] = "scipy-linprog-highs"
    failed_calls["solver_status_code"] = 4
    failed_calls["solver_status"] = "HiGHS Status 0: Not Set"
    failed_calls["iterations"] = np.nan
    failed_calls["objective"] = np.nan
    failed_calls["max_constraint_violation"] = np.nan
    failed_calls.to_csv(analysis_dir / "failed_solver_calls.csv", index=False)
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(json.dumps(analyze(Path(parser.parse_args().run_dir)), indent=2))
